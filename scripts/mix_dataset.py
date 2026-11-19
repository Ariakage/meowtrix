"""Mix multiple chat-JSONL sources into one SFT dataset.

Features:
    * --source path[:weight] — merge sources proportionally (weights are
      normalized; sources smaller than their share are capped at their size).
    * --personas configs/personas.json — randomly inject one of the 100
      persona system prompts into a fraction of examples (--persona-prob),
      so the model decouples the persona contract from any single wording.
      Examples already carrying a system turn are left untouched.
    * dedupe by joined-content hash, deterministic shuffle (--seed).

Usage:
    uv run python scripts/mix_dataset.py \
        --source data/processed/qq_meow_scenes.final.jsonl:0.15 \
        --source data/processed/nekoqa_10k.jsonl:0.45 \
        --source data/processed/s1k_reasoning.jsonl:0.40 \
        --personas configs/personas.json --persona-prob 0.8 \
        --output data/processed/stage2_mix.jsonl
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def parse_source(spec: str) -> tuple[Path, float]:
    if ":" in spec and spec.rsplit(":", 1)[-1].replace(".", "", 1).isdigit():
        path, w = spec.rsplit(":", 1)
        return Path(path), float(w)
    return Path(spec), 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", action="append", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--personas", type=Path, default=None)
    ap.add_argument("--persona-prob", type=float, default=0.0)
    ap.add_argument(
        "--persona-prob-think",
        type=float,
        default=None,
        help="separate injection prob for examples whose assistant contains "
        "<think>; defaults to persona-prob",
    )
    ap.add_argument(
        "--persona-categories",
        default=None,
        help="comma-separated category filter (sweet,coding,agent,reasoning,variety)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    specs = [parse_source(s) for s in args.source]
    total_w = sum(w for _, w in specs)

    personas = []
    if args.personas and args.persona_prob > 0:
        personas = json.loads(args.personas.read_text())
        if args.persona_categories:
            allow = set(args.persona_categories.split(","))
            personas = [p for p in personas if p["category"] in allow]
        if not personas:
            raise SystemExit("[mix] no personas matched the category filter")

    pools: list[tuple[Path, float, list[dict]]] = []
    for path, weight in specs:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        pools.append((path, weight / total_w, rows))

    target_n = max(len(r) for _, _, r in pools)
    out_rows = []
    seen = set()
    for path, frac, rows in pools:
        quota = int(round(frac * target_n))
        rng.shuffle(rows)
        taken = 0
        for row in rows:
            if taken >= quota:
                break
            digest = hashlib.sha1(
                "\n".join(m["content"] for m in row["messages"]).encode()
            ).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            msgs = [dict(m) for m in row["messages"]]
            has_think = any(
                m["role"] == "assistant" and "<think>" in m.get("content", "")
                for m in msgs
            )
            prob = args.persona_prob
            if has_think and args.persona_prob_think is not None:
                prob = args.persona_prob_think
            if (
                personas
                and rng.random() < prob
                and (not msgs or msgs[0]["role"] != "system")
            ):
                msgs.insert(
                    0,
                    {"role": "system", "content": rng.choice(personas)["prompt"]},
                )
            out_rows.append({"messages": msgs})
            taken += 1
        print(f"[mix] {path}: weight={frac:.2f} quota={quota} took={taken}")

    rng.shuffle(out_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    sys_count = sum(1 for r in out_rows if r["messages"][0]["role"] == "system")
    cats = Counter()
    if personas:
        pmap = {p["prompt"]: p["category"] for p in personas}
        for r in out_rows:
            if r["messages"][0]["role"] == "system":
                cats[pmap.get(r["messages"][0]["content"], "?")] += 1
    print(f"[mix] wrote {len(out_rows)} -> {args.output}")
    print(f"[mix] system prompts injected: {sys_count} {dict(cats)}")


if __name__ == "__main__":
    main()
