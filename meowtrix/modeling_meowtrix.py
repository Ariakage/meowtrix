"""Meowtrix-0.8B — recurrent-depth latent reasoning wrapper for Qwen3.5-0.8B.

Status: engineering scaffold. The forward path is written against the current
``transformers`` decoder-layer calling convention and must be validated against
the released ``Qwen/Qwen3.5-0.8B`` checkpoint (hybrid Gated DeltaNet + GQA
stack). See AGENTS.md → "Implementation status".

Exported checkpoints carry ``auto_map`` in config.json, so this file and
``configuration_meowtrix.py`` are copied next to the weights by
``scripts/export_hf.py`` and load with ``trust_remote_code=True``. The
try/except import below supports both the installed ``meowtrix`` package and
that flat remote-code layout.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

try:  # transformers >= 4.56 mask builders used by hybrid (DeltaNet + GQA) stacks
    from transformers.masking_utils import (
        create_causal_mask,
        create_recurrent_attention_mask,
    )

    _HAS_MASKING_UTILS = True
except ImportError:  # pragma: no cover - older transformers
    _HAS_MASKING_UTILS = False

try:  # installed package layout
    from .configuration_meowtrix import MeowtrixConfig
except ImportError:  # flat HF remote-code layout
    from configuration_meowtrix import MeowtrixConfig

__all__ = [
    "LoopGates",
    "MeowtrixCausalLMOutput",
    "MeowtrixForCausalLM",
    "load_meowtrix_model",
]

_LM_HEAD_CANDIDATES = ("lm_head", "embed_out")
_STACK_CANDIDATES = ("model", "transformer", "decoder")


class LoopGates(nn.Module):
    """Learnable near-zero residual gates for recurrent refinement loops.

    ``gates[k]`` scales the delta injected back into the residual stream
    after refinement loop ``k + 1``. Initialized to ``loop_gate_init``
    (default 0.05) so a freshly wrapped model is almost identical to its
    backbone. The vector is sized to ``max_loops`` so ``num_loops`` can be
    swept at eval time without resizing parameters.

    Registered under the attribute name ``loop_gates`` on
    :class:`MeowtrixForCausalLM` so PEFT ``modules_to_save=["loop_gates"]``
    unfreezes it during LoRA training.
    """

    def __init__(self, max_loops: int, init: float = 0.05):
        super().__init__()
        self.gates = nn.Parameter(torch.full((max_loops,), float(init)))

    def gate(self, loop_index: int) -> torch.Tensor:
        """Gate value for refinement loop ``loop_index`` (1-indexed)."""
        return self.gates[loop_index - 1]

    def forward(self, base: torch.Tensor, refined: torch.Tensor, loop_index: int):
        """Mix a refined hidden state back into the residual stream."""
        return base + self.gate(loop_index) * (refined - base)


class _TrunkView(nn.Module):
    """Fused-head adapter: ``model.base_model(...)`` -> last_hidden_state.

    TRL's fused LM head calls ``getattr(model, "base_model")`` expecting a
    backbone whose forward returns ``.last_hidden_state``. This thin view
    forwards to the owner's ``_trunk_forward`` so refinement loops (and
    ``loop_gates`` gradients) stay inside the training path. ``_owner`` is
    bound via ``object.__setattr__`` to keep it out of the module registry —
    otherwise the owner would appear as a nested submodule of itself.
    """

    def __init__(self, owner: MeowtrixForCausalLM):
        super().__init__()
        object.__setattr__(self, "_owner", owner)

    def forward(self, *args, **kwargs):
        hidden, _ = self._owner._trunk_forward(*args, **kwargs)
        return BaseModelOutputWithPast(last_hidden_state=hidden)


@dataclass
class MeowtrixCausalLMOutput(CausalLMOutputWithPast):
    """Adds per-loop hidden states to the standard causal-LM output.

    ``loop_hidden_states``: tuple of tensors ``[batch, seq, hidden]``, one per
    recurrent pass over the recurrent span (post-gate), populated only when
    ``output_loop_hiddens=True``.
    """

    loop_hidden_states: tuple[torch.Tensor, ...] | None = None


@functools.cache
def _accepted_kwargs(fn) -> frozenset[str] | None:
    """Parameter names accepted by ``fn``; ``None`` if it takes **kwargs."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return None
    names = {p.name for p in params}
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return None
    return frozenset(names)


def _filter_kwargs(fn, kwargs: dict) -> dict:
    accepted = _accepted_kwargs(fn)
    if accepted is None:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in accepted}


def _looks_linear_attention(layer: nn.Module) -> bool:
    """Whether a decoder layer is a linear-attention (Gated DeltaNet) layer.

    Resolution order: the ``block_type`` attribute used by the released
    Qwen3.5 stack, then presence of a ``linear_attn`` submodule, then a
    name-based heuristic for other hybrid stacks.
    """
    block_type = getattr(layer, "block_type", None)
    if isinstance(block_type, str):
        return block_type == "linear_attention"
    if hasattr(layer, "linear_attn"):
        return True
    text = f"{type(layer).__name__} {getattr(layer, 'attention_type', '')}"
    text = text.lower()
    return any(k in text for k in ("delta", "linear_attn", "linear_attention", "mamba"))


class MeowtrixForCausalLM(PreTrainedModel, GenerationMixin):
    # NOTE: since transformers 4.50, PreTrainedModel no longer mixes in
    # GenerationMixin — it must be inherited explicitly for generate().
    """Causal LM = Qwen3.5-0.8B backbone + recurrent mid-stack refinement.

    Forward pass:
      embed → layers[0:recurrent_start]
      → recurrent span layers[recurrent_start:recurrent_end] executed
        ``num_loops`` times (loop 1 writes caches; loops 2..K read caches,
        refine the hidden state, and inject it through ``loop_gates``)
      → layers[recurrent_end:N] → final norm → lm_head.
    """

    config_class = MeowtrixConfig
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = True
    _supports_cache_class = True

    def __init__(self, config: MeowtrixConfig):
        super().__init__(config)
        backbone_cfg = self._resolve_backbone_config(config)
        if config.backbone_config is None:
            # Freeze the resolved backbone config into our config so the
            # export is self-contained and __getattr__ fallback works.
            config.backbone_config = backbone_cfg.to_dict()
        self.backbone = AutoModelForCausalLM.from_config(backbone_cfg, trust_remote_code=True)
        self.loop_gates = LoopGates(config.max_loops, config.loop_gate_init)
        self._trunk_view = _TrunkView(self)
        self.post_init()

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_backbone_config(config: MeowtrixConfig):
        if config.backbone_config:
            payload = dict(config.backbone_config)
            model_type = payload.pop("model_type", None)
            try:
                return AutoConfig.for_model(model_type, **payload)
            except Exception:
                pass
        return AutoConfig.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)

    @classmethod
    def from_backbone(
        cls, backbone: PreTrainedModel, config: MeowtrixConfig | None = None
    ) -> MeowtrixForCausalLM:
        """Wrap an already-instantiated causal-LM backbone."""
        if config is None:
            config = MeowtrixConfig()
        if config.backbone_config is None:
            config.backbone_config = backbone.config.to_dict()
        model = cls.__new__(cls)
        PreTrainedModel.__init__(model, config)
        model.backbone = backbone
        model.loop_gates = LoopGates(config.max_loops, config.loop_gate_init).to(
            device=backbone.device, dtype=backbone.dtype
        )
        model._trunk_view = _TrunkView(model)
        model.post_init()
        return model

    @classmethod
    def from_backbone_pretrained(
        cls,
        base_model_name_or_path: str,
        *,
        torch_dtype: torch.dtype | str | None = "auto",
        device_map=None,
        config: MeowtrixConfig | None = None,
        **kwargs,
    ) -> MeowtrixForCausalLM:
        """Load a base checkpoint (e.g. Qwen/Qwen3.5-0.8B) and wrap it."""
        backbone = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            **kwargs,
        )
        if config is None:
            config = MeowtrixConfig(
                base_model_name_or_path=base_model_name_or_path,
                backbone_config=backbone.config.to_dict(),
            )
        return cls.from_backbone(backbone, config)

    # ------------------------------------------------------------------
    # backbone accessors
    # ------------------------------------------------------------------
    def _stack(self) -> nn.Module:
        """Return the decoder stack exposing ``layers``/``embed_tokens``/``norm``."""
        for owner in (self.backbone,) + tuple(
            getattr(self.backbone, name, None) for name in _STACK_CANDIDATES
        ):
            if owner is None:
                continue
            if hasattr(owner, "layers") and hasattr(owner, "embed_tokens"):
                return owner
        raise AttributeError(
            "Could not locate the decoder stack (needs .layers and "
            ".embed_tokens) inside the backbone; adjust _stack() for the "
            "released Qwen3.5 layout."
        )

    def _lm_head(self) -> nn.Module:
        for name in _LM_HEAD_CANDIDATES:
            head = getattr(self.backbone, name, None)
            if head is not None:
                return head
        return self.backbone.get_output_embeddings()

    # standard HF plumbing -------------------------------------------------
    @property
    def base_model(self):
        """Trunk view for fused-head trainers (e.g. TRL's fused LM head).

        Returns ``.last_hidden_state`` via ``_trunk_forward`` so the fused
        per-token loss path trains through the recurrent refinement loops —
        ``loop_gates`` receives gradients. Falls back to the raw backbone
        before ``_trunk_view`` exists.
        """
        view = getattr(self, "_trunk_view", None)
        if view is not None:
            return view
        return getattr(self, "backbone", self)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.backbone.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.backbone.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.backbone.set_output_embeddings(new_embeddings)

    def get_decoder(self):
        return self._stack()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, **kwargs):
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
            **kwargs,
        )

    def gradient_checkpointing_disable(self):
        self.backbone.gradient_checkpointing_disable()

    def _reorder_cache(self, past_key_values, beam_idx):
        reorder = getattr(self.backbone, "_reorder_cache", None)
        if reorder is not None:
            return reorder(past_key_values, beam_idx)
        return super()._reorder_cache(past_key_values, beam_idx)

    # ------------------------------------------------------------------
    # forward path
    # ------------------------------------------------------------------
    def set_num_loops(self, num_loops: int) -> None:
        if not 1 <= int(num_loops) <= self.config.max_loops:
            raise ValueError(f"num_loops must be in [1, {self.config.max_loops}]")
        self.config.num_loops = int(num_loops)

    def _layer_mask(self, layer, mask_mapping: dict):
        block_type = getattr(layer, "block_type", None)
        if block_type is None:
            block_type = "linear_attention" if _looks_linear_attention(layer) else "full_attention"
        if block_type not in mask_mapping:
            block_type = "full_attention"
        return mask_mapping[block_type]

    def _run_layer(
        self,
        layer,
        hidden: torch.Tensor,
        *,
        past_key_values,
        use_cache: bool,
        position_ids,
        cache_position,
        mask_mapping,
        position_embeddings,
    ) -> torch.Tensor:
        kwargs = {
            "attention_mask": self._layer_mask(layer, mask_mapping),
            "position_ids": position_ids,
            "past_key_value": past_key_values,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,
        }
        out = layer(hidden, **_filter_kwargs(layer.forward, kwargs))
        return out[0] if isinstance(out, (tuple, list)) else out

    def _causal_mask(
        self,
        attention_mask: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        past_len: int,
        device,
        dtype,
    ) -> torch.Tensor:
        """Additive [B, 1, T, S] causal mask, S = past_len + seq_len."""
        total = past_len + seq_len
        mask = torch.full((seq_len, total), torch.finfo(dtype).min, device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1 + past_len)
        mask = mask[None, None, :, :].expand(batch_size, 1, seq_len, total).clone()
        if attention_mask is not None:
            pad = (1.0 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min
            mask = mask + pad
        return mask

    def _trunk_forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        num_loops: int | None = None,
        output_loop_hiddens: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the full recurrent trunk; returns (hidden, loop_hiddens)."""
        stack = self._stack()
        num_loops = int(num_loops or self.config.num_loops)
        num_loops = max(1, min(num_loops, self.config.max_loops))

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Pass either input_ids or inputs_embeds, not both.")
        if inputs_embeds is None:
            inputs_embeds = stack.embed_tokens(input_ids)
        bsz, seq_len, _ = inputs_embeds.shape
        device, dtype = inputs_embeds.device, inputs_embeds.dtype

        if use_cache is None:
            use_cache = getattr(self.config, "use_cache", True)
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                past_len = past_key_values.get_seq_length()
            else:  # legacy tuple-of-tuples cache
                past_len = past_key_values[0][0].shape[-2]
        else:
            past_len = 0
        if cache_position is None:
            cache_position = torch.arange(past_len, past_len + seq_len, device=device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Per-layer-type masks, mirroring the stock hybrid forward:
        # full_attention layers get a causal mask, linear_attention layers get
        # the 2D padding mask via create_recurrent_attention_mask (or None).
        if _HAS_MASKING_UTILS:
            mask_kwargs = {
                "config": self.backbone.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }
        else:
            causal_mask_4d = self._causal_mask(
                attention_mask, bsz, seq_len, past_len, device, dtype
            )
            padding_mask_2d = (
                attention_mask
                if attention_mask is not None
                else torch.ones(bsz, past_len + seq_len, device=device, dtype=dtype)
            )
            mask_mapping = {
                "full_attention": causal_mask_4d,
                "linear_attention": padding_mask_2d,
            }

        position_embeddings = None
        rotary = getattr(stack, "rotary_emb", None)
        if rotary is not None:
            position_embeddings = rotary(inputs_embeds, position_ids)

        layers = stack.layers
        n_layers = len(layers)
        start = min(self.config.recurrent_start, n_layers)
        end = min(self.config.recurrent_end, n_layers)

        loop_hiddens: list[torch.Tensor] = []

        def run_span(lo: int, hi: int, h, cache, span_use_cache: bool):
            for i in range(lo, hi):
                h = self._run_layer(
                    layers[i],
                    h,
                    past_key_values=cache,
                    use_cache=span_use_cache,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    mask_mapping=mask_mapping,
                    position_embeddings=position_embeddings,
                )
            return h

        hidden = run_span(0, start, inputs_embeds, past_key_values, use_cache)
        for loop_idx in range(1, num_loops + 1):
            if loop_idx == 1:
                # First pass runs like the stock stack and writes KV /
                # DeltaNet states into the cache for autoregressive decode.
                hidden = run_span(start, end, hidden, past_key_values, use_cache)
            else:
                # Refinement loops are fully detached latent passes: the
                # stock hybrid cache has no read-only mode (attention
                # update() always appends), so the cache is dropped entirely
                # and each loop re-encodes the refined hidden state.
                refined = run_span(start, end, hidden, None, False)
                hidden = self.loop_gates(hidden, refined, loop_idx)
            if output_loop_hiddens:
                loop_hiddens.append(hidden)
        hidden = run_span(end, n_layers, hidden, past_key_values, use_cache)
        hidden = stack.norm(hidden)
        return hidden, loop_hiddens

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        num_loops: int | None = None,
        output_loop_hiddens: bool = False,
        **kwargs,
    ) -> MeowtrixCausalLMOutput:
        hidden, loop_hiddens = self._trunk_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            num_loops=num_loops,
            output_loop_hiddens=output_loop_hiddens,
            **kwargs,
        )
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int) and logits_to_keep > 0
            else slice(None)
        )
        logits = self._lm_head()(hidden[:, slice_indices, :])
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
                ignore_index=-100,
            )

        return MeowtrixCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=None,
            loop_hidden_states=tuple(loop_hiddens) if output_loop_hiddens else None,
        )

    # ------------------------------------------------------------------
    # generation plumbing
    # ------------------------------------------------------------------
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_loops=None,
        **kwargs,
    ):
        delegate = getattr(self.backbone, "prepare_inputs_for_generation", None)
        if delegate is not None:
            model_inputs = delegate(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                position_ids=position_ids,
                use_cache=use_cache,
                **kwargs,
            )
        else:
            # Generic fallback mirroring the HF convention.
            if past_key_values is not None:
                past_len = past_key_values.get_seq_length()
                if input_ids.shape[1] > past_len:
                    input_ids = input_ids[:, past_len:]
                if inputs_embeds is not None and inputs_embeds.shape[1] > past_len:
                    inputs_embeds = inputs_embeds[:, past_len:]
            if attention_mask is not None and position_ids is None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                if past_key_values is not None:
                    position_ids = position_ids[:, -input_ids.shape[1] :]
            model_inputs = {
                "input_ids": input_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "cache_position": cache_position,
            }
            if inputs_embeds is not None and past_key_values is None:
                model_inputs["inputs_embeds"] = inputs_embeds
        model_inputs["num_loops"] = num_loops or self.config.num_loops
        return model_inputs


def load_meowtrix_model(
    path_or_name: str,
    *,
    torch_dtype: torch.dtype | str | None = "auto",
    device_map=None,
    num_loops: int | None = None,
    **kwargs,
) -> MeowtrixForCausalLM:
    """Load Meowtrix from an exported dir; fall back to wrapping a base ckpt.

    - If ``path_or_name`` is an exported Meowtrix checkpoint (config.json with
      ``model_type == "meowtrix"`` + remote code), it loads directly.
    - Otherwise it is treated as a vanilla causal-LM checkpoint and wrapped
      with a fresh ``MeowtrixConfig``.
    """
    try:
        cfg = AutoConfig.from_pretrained(path_or_name, trust_remote_code=True)
        is_meowtrix = getattr(cfg, "model_type", None) == "meowtrix"
    except Exception:
        is_meowtrix = False

    if is_meowtrix:
        model = MeowtrixForCausalLM.from_pretrained(
            path_or_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **kwargs,
        )
    else:
        model = MeowtrixForCausalLM.from_backbone_pretrained(
            path_or_name, torch_dtype=torch_dtype, device_map=device_map, **kwargs
        )
    if num_loops is not None:
        model.set_num_loops(num_loops)
    return model
