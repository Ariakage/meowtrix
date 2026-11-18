"""Hidden-state evolution across recurrent loops + loop_gates readout.

For each prompt, runs a forward pass with ``output_loop_hiddens=True`` and
reports the cosine similarity between consecutive loop outputs at the last
token position — a probe of how much latent refinement each extra loop adds —
plus the current values of ``loop_gates``.

Usage::

    uv run python eval/hidden_state_evolution.py \\
        --model outputs/stage2-persona --num-loops 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from meowtrix import load_meowtrix_model

DEFAULT_PROMPTS = [
    "Explain why the sky is blue.",
    "Write a haiku about rain.",
    "What is 17 times 23?",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts-file", default=None, help="JSONL with 'prompt' field")
    p.add_argument("--num-loops", type=int, default=3)
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
    if args.prompts_file:
        prompts = [
            json.loads(line)["prompt"]
            for line in Path(args.prompts_file).read_text().splitlines()
            if line.strip()
        ]
    else:
        prompts = DEFAULT_PROMPTS

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_meowtrix_model(
        args.model, torch_dtype=torch.bfloat16, num_loops=args.num_loops
    ).to(device)
    model.eval()

    per_prompt = []
    with torch.no_grad():
        for prompt in prompts:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            batch = tokenizer(text, return_tensors="pt").to(device)
            out = model(**batch, output_loop_hiddens=True)
            hiddens = out.loop_hidden_states or ()
            last = [h[0, -1].float() for h in hiddens]
            sims = [
                F.cosine_similarity(last[i - 1], last[i], dim=0).item() for i in range(1, len(last))
            ]
            per_prompt.append({"prompt": prompt[:60], "loop_cosine_sims": sims})
            print(f"{prompt[:50]:<52} sims={[f'{s:.4f}' for s in sims]}")

    gates = model.loop_gates.gates.detach().float().cpu().tolist()
    print(f"loop_gates = {[f'{g:.4f}' for g in gates]}")

    if args.output:
        payload = {"per_prompt": per_prompt, "loop_gates": gates}
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
