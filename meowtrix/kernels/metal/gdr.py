"""Fused gated delta-rule kernels (Metal) for the GatedDeltaNet layers.

Two fused paths replace the reference PyTorch fallbacks on MPS:

- ``gdr_step`` — single-token decode: normalization, decay, rank-1 state
  update and output matvec in one dispatch, mutating the fp32 recurrent
  state in place (matches the transformers cache contract).
- ``gdr_seq`` — prefill/inference sequential scan: each output column of the
  [dk, dv] state is owned by one thread, so the whole sequence streams
  through with zero threadgroup barriers and the state stays L2-resident.

Both are inference-only. When autograd is active (training) the patched
wrappers transparently fall back to the reference implementations. The L>1
scan is roughly on par with the chunked reference but allocates virtually no
intermediates; it is gated behind ``MEOWTRIX_GDR_SEQ=1`` and off by default.
"""

from __future__ import annotations

import functools
import os

import torch

from meowtrix.kernels.metal.conv1d import _ext, available

_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_SEQ_ENABLED = os.environ.get("MEOWTRIX_GDR_SEQ", "0") == "1"


def _usable(q, k, v, g, beta, use_qk_l2norm_in_kernel, kwargs) -> bool:
    if not available() or not use_qk_l2norm_in_kernel:
        return False
    if kwargs.get("cu_seqlens") is not None:
        return False
    if not all(t.is_mps and t.dtype in _DTYPES for t in (q, k, v, beta)):
        return False
    if g.dtype != torch.float32 or not g.is_mps:
        return False
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v, g, beta)):
        return False
    return True


def _prepare_state(initial_state, q, v):
    """Contiguous fp32 [B, Hv, dk, dv] state the kernels may mutate in place."""
    B, dk = q.shape[0], q.shape[3]
    Hv, dv = v.shape[2], v.shape[3]
    if initial_state is None:
        return torch.zeros(B, Hv, dk, dv, dtype=torch.float32, device=q.device)
    if initial_state.dtype != torch.float32 or not initial_state.is_contiguous():
        return initial_state.to(torch.float32).contiguous()
    return initial_state


def _run_fused(query, key, value, g, beta, initial_state, output_final_state):
    ext = _ext()
    L = query.shape[1]
    state = _prepare_state(initial_state, query, value)
    if L == 1:
        out = ext.gdr_step(
            query[:, 0].contiguous(),
            key[:, 0].contiguous(),
            value[:, 0].contiguous(),
            g[:, 0].contiguous(),
            beta[:, 0].contiguous(),
            state,
        ).unsqueeze(1)
    else:
        qn, kn = ext.gdr_norm(query.contiguous(), key.contiguous())
        out = ext.gdr_seq(
            qn,
            kn,
            value.contiguous(),
            g.contiguous(),
            beta.contiguous(),
            state,
        )
    return out, (state if output_final_state else None)


def install(module) -> bool:
    """Patch the recurrent/chunked delta-rule fallbacks on a transformers
    modeling module. Returns True when at least one symbol was replaced."""
    patched = False

    ref_recurrent = getattr(module, "torch_recurrent_gated_delta_rule", None)
    if ref_recurrent is not None:

        @functools.wraps(ref_recurrent)
        def fused_recurrent(
            query, key, value, g, beta,
            initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=False, **kwargs,
        ):
            seq_ok = query.shape[1] == 1 or _SEQ_ENABLED
            if seq_ok and _usable(query, key, value, g, beta, use_qk_l2norm_in_kernel, kwargs):
                return _run_fused(
                    query, key, value, g, beta, initial_state, output_final_state
                )
            return ref_recurrent(
                query, key, value, g, beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                **kwargs,
            )

        module.torch_recurrent_gated_delta_rule = fused_recurrent
        patched = True

    ref_chunk = getattr(module, "torch_chunk_gated_delta_rule", None)
    if ref_chunk is not None:

        @functools.wraps(ref_chunk)
        def fused_chunk(
            query, key, value, g, beta,
            initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=False, chunk_size=64, **kwargs,
        ):
            seq_ok = query.shape[1] == 1 or _SEQ_ENABLED
            if seq_ok and _usable(query, key, value, g, beta, use_qk_l2norm_in_kernel, kwargs):
                return _run_fused(
                    query, key, value, g, beta, initial_state, output_final_state
                )
            return ref_chunk(
                query, key, value, g, beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                chunk_size=chunk_size,
                **kwargs,
            )

        module.torch_chunk_gated_delta_rule = fused_chunk
        patched = True

    return patched
