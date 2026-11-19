"""Per-epoch evaluation of LoRA checkpoints under outputs/stage2-persona.

For each ``checkpoint-*`` directory (one per epoch with the current
``save_strategy: epoch``), this:

    * wraps the base Qwen3.5-0.8B backbone in MeowtrixForCausalLM,
    * loads the LoRA adapter from the checkpoint,
    * runs a quick GSM8K slice at each --loops value,
    * samples two fixed persona probes so vibe can be eyeballed per epoch,
    * appends to eval/results/epoch_metrics.json,
    * renders eval/results/epochs_compare.png (accuracy vs epoch, per
      num_loops) and eval/results/epoch{N}_samples.md.

Already-evaluated checkpoints are skipped, so it's safe to rerun after
each epoch boundary.

Usage:
    uv run python scripts/eval_checkpoints.py \
        --ckpt-dir outputs/stage2-persona --gsm8k-limit 8 --loops 1 3
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.benchmark import extract_pred, load_problems, pick_device  # noqa: E402
from eval.plot_benchmark import BLUE, INK, PINK  # noqa: E402

RESULTS = Path("eval/results")
PROBES = [
    "宝宝睡不着，陪我说说话喵",
    "老婆，帮我看下这段代码为什么报错：```\nfor i in range(10)\n    print(i)\n```",
]


def eval_checkpoint(ckpt: Path, base: str, args) -> dict:
    from peft import PeftModel

    from meowtrix import load_meowtrix_model

    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = load_meowtrix_model(
        base, torch_dtype=getattr(torch, args.dtype), num_loops=args.loops[0]
    )
    model = PeftModel.from_pretrained(model, str(ckpt))
    model.to(args.device)
    model.eval()

    problems = load_problems("gsm8k", args.gsm8k_limit)
    entry = {"checkpoint": str(ckpt), "runs": [], "samples": {}}

    for loops in args.loops:
        model.get_base_model().set_num_loops(loops)
        n_correct, total_tok, total_s = 0, 0, 0.0
        for prob in problems:
            messages = [
                {
                    "role": "user",
                    "content": f"{prob['question']}\nPlease reason step by step, "
                    "and put your final answer within \\boxed{}.",
                }
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            dt = time.perf_counter() - t0
            gen = out[0][inputs.input_ids.shape[1] :]
            text = tokenizer.decode(gen, skip_special_tokens=True)
            pred = extract_pred(text)
            n_correct += int(pred is not None and abs(pred - prob["answer"]) < 1e-6)
            total_tok += len(gen)
            total_s += dt
            print(
                f"  [{ckpt.name} loops={loops}] {prob['id']} "
                f"gold={prob['answer']} pred={pred} {dt:.0f}s",
                flush=True,
            )
        entry["runs"].append(
            {
                "num_loops": loops,
                "n": len(problems),
                "correct": n_correct,
                "accuracy": round(n_correct / len(problems), 3),
                "tokens_per_s": round(total_tok / max(total_s, 1e-6), 2),
            }
        )
        print(f"  [{ckpt.name} loops={loops}] acc={n_correct}/{len(problems)}")

    personas = json.loads(Path("configs/personas.json").read_text())
    for pi, probe in enumerate(PROBES):
        messages = [
            {"role": "system", "content": personas[pi]["prompt"]},
            {"role": "user", "content": probe},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
            )
        gen = out[0][inputs.input_ids.shape[1] :]
        entry["samples"][f"probe_{pi}"] = tokenizer.decode(gen, skip_special_tokens=True)

    del model
    if args.device == "mps":
        torch.mps.empty_cache()
    return entry


def render_chart(metrics: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(metrics) + 1))
    loops_set = sorted({r["num_loops"] for m in metrics for r in m["runs"]})
    colors = [BLUE, PINK, "#9B8AFB", "#6FCF97"]
    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor="#FFFFFF")
    for li, loops in enumerate(loops_set):
        ys = [
            next(
                (r["accuracy"] for r in m["runs"] if r["num_loops"] == loops),
                None,
            )
            for m in metrics
        ]
        ax.plot(
            epochs,
            [100 * y if y is not None else None for y in ys],
            marker="o",
            color=colors[li % len(colors)],
            lw=1.8,
            label=f"num_loops={loops}",
        )
    ax.set_title("GSM8K accuracy per epoch", color=INK)
    ax.set_xlabel("epoch", color=INK)
    ax.set_ylabel("accuracy (%)", color=INK)
    ax.set_xticks(epochs)
    ax.set_ylim(bottom=0)
    ax.grid(color="#E8E9F3", lw=0.8)
    ax.legend(frameon=False, labelcolor=INK)
    ax.tick_params(colors=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="#FFFFFF")
    print(f"[eval] chart -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", type=Path, default=Path("outputs/stage2-persona"))
    ap.add_argument("--base", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--loops", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--gsm8k-limit", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    args.device = pick_device(args.device)
    if args.device == "mps":
        from meowtrix.kernels import enable_metal_kernels

        enable_metal_kernels()

    ckpts = sorted(
        (d for d in args.ckpt_dir.glob("checkpoint-*") if d.is_dir()),
        key=lambda d: int(d.name.split("-")[1]),
    )
    if not ckpts:
        raise SystemExit(f"[eval] no checkpoints under {args.ckpt_dir} yet")

    RESULTS.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS / "epoch_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else []
    done = {m["checkpoint"] for m in metrics}

    for ckpt in ckpts:
        if str(ckpt) in done:
            continue
        print(f"[eval] {ckpt.name} ...", flush=True)
        entry = eval_checkpoint(ckpt, args.base, args)
        metrics.append(entry)
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        epoch = len(metrics)
        with (RESULTS / f"epoch{epoch}_samples.md").open("w") as f:
            f.write(f"# {ckpt.name} samples\n\n")
            for k, v in entry["samples"].items():
                f.write(f"## {k}\n\n{v}\n\n")
        render_chart(metrics, RESULTS / "epochs_compare.png")

    print(f"[eval] {len(metrics)} checkpoint(s) evaluated -> {metrics_path}")


if __name__ == "__main__":
    main()
