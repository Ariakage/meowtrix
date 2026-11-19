"""Hardware compatibility patches for older GPUs.

TRL's chunked-logprob loss uses ``torch.addmm(..., out_dtype=torch.float32)``
for fp32 gradient accumulation, which requires CUDA compute capability >= 8.0.
On Turing (T4, CC 7.5) that path raises at the first backward pass. TRL keeps
a pure-fp32 fallback behind the module flag ``_MM_OUT_DTYPE``; disabling it
routes to ``acc.addmm_(a.float(), b.float())`` — slower but correct (and the
fp32 accumulation the comment requires is preserved).
"""

from __future__ import annotations


def patch_trl_chunked_logprob() -> bool:
    """Disable the CC>=8.0-only addmm path in TRL on older GPUs. Returns True if patched."""
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() >= (8, 0):
        return False
    try:
        from trl.kernels import chunked_logprob
    except ImportError:
        return False
    chunked_logprob._MM_OUT_DTYPE = False
    return True


def patch_cudnn_conv1d() -> bool:
    """Wrap the transformers depthwise ``causal_conv1d_fn`` fallback so a cuDNN
    engine-search failure (``GET was unable to find an engine``, seen on T4 when
    refinement loops re-run the recurrent span) retries once with cuDNN
    disabled, which dispatches to the native CUDA conv. Returns True if any
    module was patched."""
    import functools
    import importlib

    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False

    patched = False
    for mod_name in (
        "transformers.models.qwen3_5.modeling_qwen3_5",
        "transformers.models.qwen3_next.modeling_qwen3_next",
    ):
        try:
            mod = importlib.import_module(mod_name)
            orig = getattr(mod, "causal_conv1d_fn", None)
        except ImportError:
            continue
        if orig is None or getattr(orig, "_meowtrix_cudnn_retry", False):
            continue

        @functools.wraps(orig)
        def wrapped(*args, _orig=orig, **kwargs):
            try:
                return _orig(*args, **kwargs)
            except RuntimeError as e:
                if "engine" not in str(e):
                    raise
                with torch.backends.cudnn.flags(enabled=False):
                    return _orig(*args, **kwargs)

        wrapped._meowtrix_cudnn_retry = True
        mod.causal_conv1d_fn = wrapped
        patched = True
    return patched
