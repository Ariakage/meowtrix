"""MeowtrixConfig — configuration for the Meowtrix-0.8B recurrent-depth model.

This file is self-contained so it can be copied next to exported weights and
resolved through ``config.auto_map`` with ``trust_remote_code=True``.
"""

from transformers import PretrainedConfig


class MeowtrixConfigError(ValueError):
    pass


class MeowtrixConfig(PretrainedConfig):
    """Configuration for Meowtrix-0.8B.

    The backbone is the released ``Qwen/Qwen3.5-0.8B`` hybrid stack
    (24 layers; every 4-layer cycle is 3x Gated DeltaNet + 1x GQA).
    A contiguous mid-stack span of decoder layers is executed recurrently
    at inference time ("test-time latent compute").

    Attributes:
        base_model_name_or_path: HF repo id or local path of the backbone
            checkpoint the recurrent wrapper is built around.
        backbone_config: Serialized backbone config dict (model_type +
            fields) so the exported repo is self-contained. When ``None``,
            the backbone config is resolved from ``base_model_name_or_path``.
        recurrent_start: First decoder-layer index of the recurrent span
            (0-indexed, inclusive). Default 8.
        recurrent_end: End of the recurrent span (0-indexed, exclusive).
            Default 16 → layers 8..15, i.e. exactly two full 4-layer cycles.
        num_loops: How many times the recurrent span runs per forward pass.
            Loop 1 is a normal pass that also writes KV / DeltaNet caches;
            loops 2..K refine the hidden state in latent space with cache
            writes disabled. ``num_loops=1`` degenerates to the vanilla stack.
        max_loops: Upper bound on ``num_loops``; sizes the ``loop_gates``
            parameter vector.
        loop_gate_init: Initial value of each residual loop gate. Kept
            near zero (default 0.05) so recurrent refinement starts as a
            near-identity perturbation of the pretrained distribution.
        abliteration_layers: Inclusive 0-indexed layer range whose residual-
            stream activations are averaged when estimating the refusal
            direction (default: layers 13..18 of the post-recurrent region).
        enable_thinking: Whether prompts are rendered with the ``<think>``
            block opened (passed through to the chat template).
    """

    model_type = "meowtrix"

    # Resolved from config.json when loading with trust_remote_code=True.
    # Assigned to self in __init__ as well: to_dict() only serializes instance
    # attributes, and save_pretrained() must emit auto_map into config.json.
    AUTO_MAP = {
        "AutoConfig": "configuration_meowtrix.MeowtrixConfig",
        "AutoModelForCausalLM": "modeling_meowtrix.MeowtrixForCausalLM",
    }

    def __init__(
        self,
        base_model_name_or_path: str = "Qwen/Qwen3.5-0.8B",
        backbone_config: dict | None = None,
        recurrent_start: int = 8,
        recurrent_end: int = 16,
        num_loops: int = 3,
        max_loops: int = 8,
        loop_gate_init: float = 0.05,
        abliteration_layers: tuple[int, int] = (13, 18),
        enable_thinking: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.auto_map = dict(self.AUTO_MAP)
        self.base_model_name_or_path = base_model_name_or_path
        self.backbone_config = backbone_config
        self.recurrent_start = int(recurrent_start)
        self.recurrent_end = int(recurrent_end)
        self.num_loops = int(num_loops)
        self.max_loops = int(max_loops)
        self.loop_gate_init = float(loop_gate_init)
        self.abliteration_layers = tuple(int(i) for i in abliteration_layers)
        self.enable_thinking = bool(enable_thinking)

        if self.recurrent_start < 0:
            raise MeowtrixConfigError("recurrent_start must be >= 0")
        if self.recurrent_end <= self.recurrent_start:
            raise MeowtrixConfigError("recurrent_end must be > recurrent_start")
        if not 1 <= self.num_loops <= self.max_loops:
            raise MeowtrixConfigError("num_loops must be in [1, max_loops]")

    def __getattr__(self, key):
        """Fall back to ``backbone_config`` for backbone fields.

        Generation/cache code reads attributes such as ``num_hidden_layers``
        or ``layer_types`` off ``model.config``; transparently serving them
        from the embedded backbone config keeps the hybrid cache intact.
        """
        parent_getattr = getattr(super(), "__getattr__", None)
        if parent_getattr is not None:
            try:
                return parent_getattr(key)
            except AttributeError:
                pass
        backbone = self.__dict__.get("backbone_config")
        if isinstance(backbone, dict) and key in backbone:
            return backbone[key]
        raise AttributeError(key)
