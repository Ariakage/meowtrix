"""Meowtrix-0.8B — recurrent-depth latent reasoning on Qwen3.5-0.8B."""

from .configuration_meowtrix import MeowtrixConfig
from .modeling_meowtrix import (
    LoopGates,
    MeowtrixCausalLMOutput,
    MeowtrixForCausalLM,
    load_meowtrix_model,
)

__version__ = "0.1.0"

__all__ = [
    "LoopGates",
    "MeowtrixCausalLMOutput",
    "MeowtrixConfig",
    "MeowtrixForCausalLM",
    "load_meowtrix_model",
]


def register_auto_classes() -> None:
    """Register meowtrix with transformers Auto classes (local installs)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    AutoConfig.register("meowtrix", MeowtrixConfig)
    AutoModelForCausalLM.register(MeowtrixConfig, MeowtrixForCausalLM)
