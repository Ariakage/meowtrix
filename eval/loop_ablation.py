"""Test-time compute ablation: accuracy & latency vs. ``num_loops``.

Runs each prompt through generation at every requested loop count, records
wall-clock latency / tokens-per-second and a simple correctness check
(``contains`` substring or ``regex`` on the generated answer).

Prompt file: JSONL rows like
    {"prompt": "What is 17*23?", "contains": "391"}
    {"prompt": "...", "regex": "..."}          # alternative
    {"prompt": "..."}                          # latency only

Usage::

    uv run python eval/loop_ablation.py \\
        --model exported/meowtrix-0.8b --prompts-file eval/tasks.jsonl \\
        --loops 1,2,3 --output eval/results/loops.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from meowtrix import load_meowtrix_model


def check(text: str, row: dict) -> bool | None:
    if "contains" in row:
        return row["contains"] in text
    if "regex" in row:
        return re.search(row["regex"], text) is not None
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts-file", required=True)
    p.add_argument("--loops", default="1,2,3")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--output", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    loop_counts = [int(x) for x in args.loops.split(",")]
    rows = [
        json.loads(line)
        for line in Path(args.prompts_file).read_text().splitlines()
        if line.strip()
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_meowtrix_model(args.model, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    results = []
    for k in loop_counts:
        model.set_num_loops(k)
        correct = total_checked = 0
        elapsed = n_tokens = 0.0
        for row in rows:
            messages = [{"role": "user", "content": row["prompt"]}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            inputs = tokenizer(text, return_tensors="pt").to(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            dt = time.perf_counter() - t0
            gen = out[0][inputs["input_ids"].shape[1] :]
            n_new = gen.shape[0]
            elapsed += dt
            n_tokens += n_new
            answer = tokenizer.decode(gen, skip_special_tokens=True)
            ok = check(answer, row)
            if ok is not None:
                total_checked += 1
                correct += int(ok)
        rec = {
            "num_loops": k,
            "n_prompts": len(rows),
            "accuracy": (correct / total_checked) if total_checked else None,
            "checked": total_checked,
            "total_seconds": round(elapsed, 3),
            "new_tokens": int(n_tokens),
            "tokens_per_second": round(n_tokens / elapsed, 2) if elapsed else None,
            "seconds_per_prompt": round(elapsed / len(rows), 3),
        }
        results.append(rec)
        print(json.dumps(rec))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
