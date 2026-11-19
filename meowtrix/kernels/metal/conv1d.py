"""Fused causal conv1d kernels (Metal) for the Qwen3.5 GatedDeltaNet layers.

Replaces the pure-PyTorch MPS fallbacks used by transformers when the
`causal-conv1d` package is unavailable. The forward kernel fuses the depthwise
conv, the causal padding, and the SiLU activation into one dispatch and also
emits the pre-activation tensor so the backward kernel can recompute the SiLU
derivative without a second conv pass.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch
import torch.nn.functional as F

_EXT = None
_BUILD_ERROR: Exception | None = None


def _ext():
    global _EXT, _BUILD_ERROR
    if _EXT is None and _BUILD_ERROR is None:
        try:
            from torch.utils.cpp_extension import load

            src_dir = Path(__file__).resolve().parent
            _EXT = load(
                name="meowtrix_metal_conv1d",
                sources=[str(src_dir / "conv1d_bridge.mm")],
                extra_cflags=[
                    "-ObjC++",
                    "-std=c++20",
                    f'-DMETAL_SRC_PATH=\\"{src_dir / "conv1d.metal"}\\"',
                    f'-DGDR_SRC_PATH=\\"{src_dir / "gdr.metal"}\\"',
                ],
                extra_ldflags=["-framework", "Foundation", "-framework", "Metal"],
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - build env dependent
            _BUILD_ERROR = exc
    return _EXT


def available() -> bool:
    """True when the extension built successfully and MPS is present."""
    return _ext() is not None


def _silu_grad(pre: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(pre)
    return s * (1 + pre * (1 - s))


class _CausalConv1dFn(torch.autograd.Function):
    """Differentiable wrapper: fused fwd, fused dx, torch fallback for dw/db.

    Under LoRA training the conv weight is frozen so dw/db are never
    requested; they are only computed via cheap torch ops when needed.
    """

    @staticmethod
    def forward(ctx, hidden_states, weight, bias, activation):
        ext = _ext()
        x = hidden_states.contiguous()
        w = weight.contiguous()
        b = bias.contiguous() if bias is not None else None
        ctx.save_for_backward(x, w, b)
        ctx.activation = activation
        return ext.conv1d_fwd(x, w, b, activation == "silu")

    @staticmethod
    def backward(ctx, grad_out):
        ext = _ext()
        x, w, b = ctx.saved_tensors
        dy = grad_out.contiguous()
        dx = ext.conv1d_bwd_dx(dy, x, w, b) if ctx.needs_input_grad[0] else None

        dw = db = None
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            kernel = w.shape[-1]
            length = x.shape[-1]
            pre = F.conv1d(
                x.float(), w.float().unsqueeze(1),
                b.float() if b is not None else None,
                padding=kernel - 1, groups=x.shape[1],
            )[:, :, :length]
            gate = _silu_grad(pre) if ctx.activation == "silu" else 1.0
            g = dy.float() * gate
        if ctx.needs_input_grad[1]:
            xp = F.pad(x.float(), (kernel - 1, 0))
            dw = torch.stack(
                [(g * xp[:, :, k : k + length]).sum(dim=(0, 2)) for k in range(kernel)],
                dim=-1,
            ).to(w.dtype)
        if ctx.needs_input_grad[2]:
            db = g.sum(dim=(0, 2)).to(b.dtype)
        return dx, dw, db, None


def _supported(x: torch.Tensor, activation) -> bool:
    return (
        x.is_mps
        and activation in (None, "silu")
        and x.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and available()
    )


def install(module) -> bool:
    """Monkeypatch `causal_conv1d_fn`/`causal_conv1d_update` on a transformers
    modeling module. Returns True when both symbols were replaced."""
    ref_fn = getattr(module, "causal_conv1d_fn", None)
    ref_update = getattr(module, "causal_conv1d_update", None)
    if ref_fn is None or ref_update is None:
        return False

    @functools.wraps(ref_fn)
    def fused_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
        if _supported(hidden_states, activation):
            return _CausalConv1dFn.apply(hidden_states, weight, bias, activation)
        return ref_fn(hidden_states, weight, bias, activation, **kwargs)

    @functools.wraps(ref_update)
    def fused_update(hidden_states, conv_state, weight, bias=None, activation=None):
        if (
            _supported(hidden_states, activation)
            and conv_state.is_mps
            and conv_state.is_contiguous()
            and conv_state.dtype == hidden_states.dtype
            and conv_state.shape[-1] + hidden_states.shape[-1] <= 16
            and not hidden_states.requires_grad
        ):
            return _ext().conv1d_update(
                hidden_states.contiguous(),
                conv_state,
                weight.contiguous(),
                bias.contiguous() if bias is not None else None,
                activation == "silu",
            )
        return ref_update(hidden_states, conv_state, weight, bias, activation)

    module.causal_conv1d_fn = fused_fn
    module.causal_conv1d_update = fused_update
    return True
