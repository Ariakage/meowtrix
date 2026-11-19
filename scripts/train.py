"""Unified training entry point — LoRA SFT for both stages.

Usage::

    uv run python scripts/train.py --config configs/train_annealing.yaml
    uv run python scripts/train.py --config configs/train_persona.yaml

Dataset format: JSONL with a ``messages`` column (chat-conversational), see
``data/sample_train.jsonl``. Full corpora are hosted on Hugging Face.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

from meowtrix import load_meowtrix_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="stage YAML (see configs/)")
    p.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        metavar="CKPT",
        help="resume training: latest checkpoint-* under output_dir, or the given dir",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    lora_cfg = cfg.get("lora", {})
    data_cfg = cfg["data"]
    tr_cfg = cfg["trainer"]

    dtype = getattr(torch, model_cfg.get("torch_dtype", "bfloat16"))
    model_path = model_cfg["model_path"]

    if model_cfg.get("metal_kernels", True):
        from meowtrix.kernels import enable_metal_kernels

        enable_metal_kernels()

    from meowtrix.compat import patch_trl_chunked_logprob

    if patch_trl_chunked_logprob():
        print("[compat] CC<8.0: TRL chunked_logprob using pure-fp32 addmm fallback")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_dir = Path(model_path)
    is_adapter = adapter_dir.is_dir() and (adapter_dir / "adapter_config.json").exists()

    if is_adapter:
        # Continue training from a previous LoRA adapter: resolve its base
        # model, wrap it, then load the adapter as trainable.
        import json

        from peft import PeftModel

        base = json.loads((adapter_dir / "adapter_config.json").read_text())[
            "base_model_name_or_path"
        ]
        model = load_meowtrix_model(
            base,
            torch_dtype=dtype,
            num_loops=model_cfg.get("num_loops"),
        )
        model = PeftModel.from_pretrained(model, model_path, is_trainable=True)
        peft_config = None
    else:
        from peft import LoraConfig

        model = load_meowtrix_model(
            model_path,
            torch_dtype=dtype,
            num_loops=model_cfg.get("num_loops"),
        )
        peft_config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            target_modules=lora_cfg.get("target_modules", "all-linear"),
            modules_to_save=lora_cfg.get("modules_to_save") or None,
            task_type="CAUSAL_LM",
        )

    dataset = load_dataset("json", data_files=data_cfg["train_file"], split="train")

    from trl import SFTConfig, SFTTrainer

    sft_config = SFTConfig(
        output_dir=tr_cfg["output_dir"],
        num_train_epochs=tr_cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=tr_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=tr_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=tr_cfg.get("learning_rate", 1e-4),
        lr_scheduler_type=tr_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=tr_cfg.get("warmup_steps", 20),
        logging_steps=tr_cfg.get("logging_steps", 10),
        save_strategy=tr_cfg.get("save_strategy", "epoch"),
        save_steps=tr_cfg.get("save_steps", 500),
        save_total_limit=tr_cfg.get("save_total_limit"),
        bf16=tr_cfg.get("bf16", True),
        fp16=tr_cfg.get("fp16", False),
        gradient_checkpointing=tr_cfg.get("gradient_checkpointing", True),
        seed=tr_cfg.get("seed", 42),
        max_length=data_cfg.get("max_length", 2048),
        packing=False,
        assistant_only_loss=tr_cfg.get("assistant_only_loss", False),
        report_to=tr_cfg.get("report_to", "none"),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    if tr_cfg.get("save_epoch_checkpoints"):
        from transformers import TrainerCallback

        class _EpochSaver(TrainerCallback):
            """Save adapter weights at each epoch boundary for per-epoch eval.

            Writes epoch-{N}/ dirs that do not count toward save_total_limit
            rotation (no optimizer state — eval-only artifacts)."""

            def on_epoch_end(self, args, state, control, **kw):
                ep = round(state.epoch)
                out = Path(args.output_dir) / f"epoch-{ep}"
                trainer.save_model(out)
                print(f"[epoch-saver] epoch {ep} → {out}")

        trainer.add_callback(_EpochSaver())

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(sft_config.output_dir)
    tokenizer.save_pretrained(sft_config.output_dir)
    print(f"[train] stage '{cfg.get('stage', '?')}' done → {sft_config.output_dir}")


if __name__ == "__main__":
    main()
