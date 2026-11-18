"""Thinking-aware abliteration for Meowtrix-0.8B.

Estimates a refusal direction in residual-stream space while the prompt is
rendered *with* the ``<think>`` block opened (``enable_thinking=True``), so
the direction captures refusal carried through the reasoning trace — not just
refusals in the final answer. The direction is then projected out of every
output-space projection weight (``o_proj`` / ``out_proj`` / ``down_proj``).

Pipeline (all tensors in bf16 on the selected device):

  1. For each prompt set (harmful vs. harmless), render the chat template
     with the thinking block open and record the residual-stream hidden
     state at the target layers (default 13..18, post-recurrent region).
  2. refusal direction d = normalize( mean(harmful) - mean(harmless) )
     averaged over the target layers.
  3. For every Linear named ``*.o_proj`` / ``*.out_proj`` / ``*.down_proj``
     whose output dim matches the residual width: W ← W − d·(dᵀW).

CLI::

    uv run python -m meowtrix.abliterate \\
        --model Qwen/Qwen3.5-0.8B \\
        --prompts configs/abliterate_prompts.yaml \\
        --output outputs/abliterated
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_LAYERS = (13, 18)
TARGET_SUFFIXES = ("o_proj", "out_proj", "down_proj")


# ---------------------------------------------------------------------------
# model introspection
# ---------------------------------------------------------------------------
def find_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the decoder ``layers`` ModuleList (works through wrappers)."""
    stack_getter = getattr(model, "_stack", None)
    if callable(stack_getter):
        stack = stack_getter()
        if hasattr(stack, "layers"):
            return stack.layers
    for module in model.modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            first = module[0]
            if hasattr(first, "self_attn") or hasattr(first, "mlp"):
                return module
    raise AttributeError("Could not locate the decoder layer list.")


def residual_width(model: nn.Module) -> int:
    return model.config.hidden_size


# ---------------------------------------------------------------------------
# activation capture
# ---------------------------------------------------------------------------
def render_prompt(tokenizer, prompt: str, enable_thinking: bool) -> str:
    """Render a single-turn prompt, opening the <think> block when requested."""
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except (TypeError, ValueError):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text + "<think>\n" if enable_thinking else text


@torch.no_grad()
def collect_layer_means(
    model: nn.Module,
    tokenizer,
    prompts: list[str],
    *,
    layers: range,
    layer_modules: nn.ModuleList,
    enable_thinking: bool,
    position: str = "last",
    device: str = "cpu",
) -> torch.Tensor:
    """Mean residual-stream activation per target layer, averaged over prompts."""
    accum = {i: [] for i in layers}
    captured: dict[int, torch.Tensor] = {}

    def capturing_hook(idx):
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            captured[idx] = h.detach()

        return hook

    hooks = []
    for i in layers:
        hooks.append(layer_modules[i].register_forward_hook(capturing_hook(i)))
    try:
        for prompt in prompts:
            text = render_prompt(tokenizer, prompt, enable_thinking)
            batch = tokenizer(text, return_tensors="pt").to(device)
            captured.clear()
            model(**batch)
            mask = batch["attention_mask"]
            for i in layers:
                h = captured[i].float()
                if position == "last":
                    last = mask.sum(dim=1) - 1
                    vec = h[torch.arange(h.shape[0]), last].mean(dim=0)
                else:  # mean over unmasked positions
                    vec = (h * mask[..., None]).sum(dim=(0, 1)) / mask.sum()
                accum[i].append(vec.cpu())
    finally:
        for h in hooks:
            h.remove()
    return torch.stack([torch.stack(accum[i]).mean(dim=0) for i in layers])  # [n_layers, hidden]


# ---------------------------------------------------------------------------
# abliteration
# ---------------------------------------------------------------------------
def refusal_direction(harmful_means: torch.Tensor, harmless_means: torch.Tensor) -> torch.Tensor:
    """Normalized mean difference, averaged across the probed layers."""
    d = (harmful_means - harmless_means).mean(dim=0)
    return d / d.norm()


@torch.no_grad()
def abliterate_(model: nn.Module, direction: torch.Tensor) -> list[str]:
    """Project ``direction`` out of every target projection weight, in-place."""
    ablated: list[str] = []
    d = direction.to(dtype=torch.float32)
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.rsplit(".", 1)[-1] not in TARGET_SUFFIXES:
            continue
        w = module.weight
        if w.shape[0] != d.shape[0]:
            continue
        dd = d.to(device=w.device)
        w.data = (w.float() - dd.unsqueeze(1) * (dd.unsqueeze(0) @ w.float())).to(w.dtype)
        ablated.append(name)
    return ablated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_prompts(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("harmful_prompts", "harmless_prompts"):
        if not cfg.get(key):
            raise ValueError(f"{path}: missing or empty '{key}'")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="HF repo id or local ckpt")
    parser.add_argument("--prompts", required=True, help="contrast prompts YAML")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--device", default=None, help="cuda / mps / cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--layers", default=None, help="e.g. 13-18 (inclusive)")
    parser.add_argument(
        "--position",
        default=None,
        choices=["last", "mean"],
        help="token position to pool (default: from YAML, else 'last')",
    )
    parser.add_argument("--max-prompts", type=int, default=None)
    args = parser.parse_args()

    cfg = _load_prompts(args.prompts)
    harmful = cfg["harmful_prompts"][: args.max_prompts]
    harmless = cfg["harmless_prompts"][: args.max_prompts]
    lo, hi = cfg.get("layers", list(DEFAULT_LAYERS))
    if args.layers:
        lo, hi = (int(x) for x in args.layers.split("-"))
    layers = range(lo, hi + 1)
    position = args.position or cfg.get("position", "last")
    enable_thinking = bool(cfg.get("enable_thinking", True))

    device = args.device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    dtype = getattr(torch, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()

    layer_modules = find_decoder_layers(model)
    if max(layers) >= len(layer_modules):
        raise ValueError(f"layers {lo}-{hi} out of range: model has {len(layer_modules)} layers")

    print(f"[abliterate] collecting harmful means  (n={len(harmful)})")
    harmful_means = collect_layer_means(
        model,
        tokenizer,
        harmful,
        layers=layers,
        layer_modules=layer_modules,
        enable_thinking=enable_thinking,
        position=position,
        device=device,
    )
    print(f"[abliterate] collecting harmless means (n={len(harmless)})")
    harmless_means = collect_layer_means(
        model,
        tokenizer,
        harmless,
        layers=layers,
        layer_modules=layer_modules,
        enable_thinking=enable_thinking,
        position=position,
        device=device,
    )

    direction = refusal_direction(harmful_means, harmless_means)
    print(f"[abliterate] ||d||={direction.norm():.4f}  dim={direction.shape[0]}")

    ablated = abliterate_(model, direction)
    print(f"[abliterate] ablated {len(ablated)} projection weights")
    for name in ablated:
        print(f"  - {name}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    # Preserve remote code files when the source was an exported Meowtrix dir.
    src = Path(args.model)
    if src.is_dir():
        for py in src.glob("*.py"):
            shutil.copy2(py, out / py.name)
    torch.save({"direction": direction, "layers": [lo, hi]}, out / "refusal_direction.pt")
    print(f"[abliterate] saved → {out}")


if __name__ == "__main__":
    main()
