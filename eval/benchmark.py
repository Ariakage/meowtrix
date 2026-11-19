"""Benchmark Meowtrix on math reasoning datasets across num_loops settings.

Evaluates recurrent-depth test-time compute scaling: for each num_loops value,
measures accuracy, latency, and generation throughput. Results are written as
JSON under eval/results/ for plotting.

Datasets:
    aime  — HuggingFaceH4/aime_2024 (integer answers 0-999)
    gsm8k — openai/gsm8k test split (answers after "####")

Usage:
    uv run python eval/benchmark.py \
        --model Qwen/Qwen3.5-0.8B \
        --num-loops 1 2 3 4 \
        --aime-limit 15 --gsm8k-limit 20
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS_DIR = Path(__file__).resolve().parent / "results"

MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extract_boxed(text: str) -> str | None:
    """Return the last \\boxed{...} content, or None."""
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    return matches[-1].strip() if matches else None


def extract_number(text: str) -> str | None:
    """Fallback: last integer/decimal in the text."""
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not matches:
        return None
    return matches[-1].replace(",", "").rstrip(".")


def normalize_number(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.replace(",", "").strip())
    except ValueError:
        return None


def extract_pred(text: str) -> float | None:
    return normalize_number(extract_boxed(text) or extract_number(text))


def load_problems(dataset: str, limit: int) -> list[dict]:
    if dataset == "aime":
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        problems = [
            {"id": f"aime_{r['id']}", "question": r["problem"], "answer": float(r["answer"])}
            for r in ds
        ]
    elif dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        problems = [
            {
                "id": f"gsm8k_{i}",
                "question": r["question"],
                "answer": normalize_number(r["answer"].split("####")[-1]),
            }
            for i, r in enumerate(ds)
        ]
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    return problems[:limit]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir to merge onto the base")
    ap.add_argument(
        "--wrapper",
        choices=["auto", "meowtrix", "stock"],
        default="auto",
        help=(
            "meowtrix: wrap a base HF checkpoint in MeowtrixForCausalLM; "
            "stock: plain AutoModelForCausalLM; auto: meowtrix unless the "
            "path is already a meowtrix export or --wrapper stock given."
        ),
    )
    ap.add_argument("--num-loops", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--aime-limit", type=int, default=15)
    ap.add_argument("--gsm8k-limit", type=int, default=20)
    ap.add_argument("--aime-max-new-tokens", type=int, default=768)
    ap.add_argument("--gsm8k-max-new-tokens", type=int, default=384)
    ap.add_argument("--aime-thinking", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gsm8k-thinking", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype)
    if device == "mps":
        from meowtrix.kernels import enable_metal_kernels

        enable_metal_kernels()
    elif device == "cuda":
        from meowtrix.compat import patch_cudnn_conv1d

        if patch_cudnn_conv1d():
            print("[compat] cudnn conv1d retry patch enabled")
    print(f"[benchmark] device={device} dtype={dtype} model={args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    wrapper = args.wrapper
    if wrapper == "auto":
        cfg_path = Path(args.model) / "config.json"
        wrapper = "stock"
        if cfg_path.is_file():
            wrapper = (
                "stock"
                if json.loads(cfg_path.read_text()).get("model_type") == "meowtrix"
                else "meowtrix"
            )
        else:
            wrapper = "meowtrix"
    if wrapper == "meowtrix":
        from meowtrix import MeowtrixForCausalLM

        model = MeowtrixForCausalLM.from_backbone_pretrained(args.model, torch_dtype=dtype)
        model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, trust_remote_code=True
        )
        model.to(device)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    is_meowtrix = hasattr(model, "loop_gates")
    print(
        f"[benchmark] meowtrix={is_meowtrix} "
        f"config_loops={getattr(model.config, 'num_loops', None)}"
    )

    suites = [
        (
            "aime",
            load_problems("aime", args.aime_limit),
            args.aime_max_new_tokens,
            args.aime_thinking,
        ),
        (
            "gsm8k",
            load_problems("gsm8k", args.gsm8k_limit),
            args.gsm8k_max_new_tokens,
            args.gsm8k_thinking,
        ),
    ]

    results = {
        "model": args.model,
        "device": device,
        "dtype": args.dtype,
        "meowtrix": is_meowtrix,
        "runs": [],
    }

    for name, problems, max_new, thinking in suites:
        for num_loops in args.num_loops:
            records = []
            n_correct = 0
            t_suite = time.perf_counter()
            for i, prob in enumerate(problems):
                if is_meowtrix:
                    model.set_num_loops(num_loops)
                elif num_loops != args.num_loops[0]:
                    break
                messages = [
                    {
                        "role": "user",
                        "content": f"{prob['question']}\n{MATH_INSTRUCTION}",
                    }
                ]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=thinking,
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new,
                        do_sample=False,
                    )
                elapsed = time.perf_counter() - t0
                gen = out[0][inputs.input_ids.shape[1] :]
                text = tokenizer.decode(gen, skip_special_tokens=True)
                pred = extract_pred(text)
                correct = pred is not None and abs(pred - prob["answer"]) < 1e-6
                n_correct += correct
                records.append(
                    {
                        "id": prob["id"],
                        "gold": prob["answer"],
                        "pred": pred,
                        "correct": correct,
                        "gen_tokens": len(gen),
                        "latency_s": round(elapsed, 3),
                        "tokens_per_s": round(len(gen) / max(elapsed, 1e-6), 2),
                        "truncated": len(gen) >= max_new,
                        "output": text,
                    }
                )
                print(
                    f"[{name} loops={num_loops}] {i + 1}/{len(problems)} "
                    f"{prob['id']} gold={prob['answer']} pred={pred} "
                    f"{'OK' if correct else 'X'} {elapsed:.1f}s {len(gen)}tok",
                    flush=True,
                )
            acc = n_correct / len(problems)
            total_gen = sum(r["gen_tokens"] for r in records)
            total_time = sum(r["latency_s"] for r in records)
            summary = {
                "dataset": name,
                "num_loops": num_loops,
                "thinking": thinking,
                "max_new_tokens": max_new,
                "n": len(problems),
                "correct": n_correct,
                "accuracy": round(acc, 4),
                "wall_s": round(time.perf_counter() - t_suite, 1),
                "mean_latency_s": round(total_time / len(records), 3),
                "mean_gen_tokens": round(total_gen / len(records), 1),
                "tokens_per_s": round(total_gen / max(total_time, 1e-6), 2),
                "records": records,
            }
            results["runs"].append(summary)
            print(f"== {name} loops={num_loops}: {n_correct}/{len(problems)} acc={acc:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(args.output)
        if args.output
        else (RESULTS_DIR / f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json")
    )
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[benchmark] wrote {out_path}")


if __name__ == "__main__":
    main()
