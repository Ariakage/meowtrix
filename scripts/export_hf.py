"""Export a HF-ready Meowtrix checkpoint: weights + config + remote code.

The export writes ``config.json`` with ``auto_map`` and copies
``configuration_meowtrix.py`` / ``modeling_meowtrix.py`` next to the weights,
so the repo loads directly via ``trust_remote_code=True``::

    uv run python scripts/export_hf.py \\
        --model outputs/stage2-persona --out exported/meowtrix-0.8b

    # merge a LoRA adapter instead of pointing at a merged ckpt:
    uv run python scripts/export_hf.py \\
        --model outputs/abliterated --adapter outputs/stage2-persona \\
        --out exported/meowtrix-0.8b
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoTokenizer

from meowtrix import MeowtrixConfig, load_meowtrix_model

REMOTE_CODE_FILES = ("configuration_meowtrix.py", "modeling_meowtrix.py")
PKG_DIR = Path(__file__).resolve().parent.parent / "meowtrix"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        required=True,
        help="merged ckpt, or base ckpt when --adapter is given",
    )
    p.add_argument("--adapter", default=None, help="LoRA adapter dir to merge first")
    p.add_argument("--out", required=True, help="export directory")
    p.add_argument("--num-loops", type=int, default=None)
    p.add_argument("--push", default=None, help="optional HF repo id to upload")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = load_meowtrix_model(args.model, torch_dtype=torch.bfloat16)

    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
        print(f"[export] merged adapter {args.adapter}")

    if args.num_loops:
        model.set_num_loops(args.num_loops)

    cfg = model.config
    assert isinstance(cfg, MeowtrixConfig)
    cfg.backbone_config = model.backbone.config.to_dict()
    cfg.auto_map = {
        "AutoConfig": "configuration_meowtrix.MeowtrixConfig",
        "AutoModelForCausalLM": "modeling_meowtrix.MeowtrixForCausalLM",
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    for fname in REMOTE_CODE_FILES:
        shutil.copy2(PKG_DIR / fname, out / fname)

    tok_src = Path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        tok_src if (tok_src / "tokenizer_config.json").exists() else cfg.base_model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(out)

    meta = {
        "model_name": "Meowtrix-0.8B",
        "base": cfg.base_model_name_or_path,
        "num_loops": cfg.num_loops,
        "recurrent_span": [cfg.recurrent_start, cfg.recurrent_end],
    }
    (out / "meowtrix_export.json").write_text(json.dumps(meta, indent=2))
    print(f"[export] wrote {out}")

    if args.push:
        from huggingface_hub import upload_folder

        upload_folder(repo_id=args.push, folder_path=out, repo_type="model")
        print(f"[export] uploaded → https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
