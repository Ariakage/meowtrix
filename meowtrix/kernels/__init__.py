"""Optional fused kernels. Metal kernels are compiled lazily on first use."""

from __future__ import annotations

import importlib
import sys
import warnings

_TARGET_MODULES = (
    "transformers.models.qwen3_5.modeling_qwen3_5",
    "transformers.models.qwen3_next.modeling_qwen3_next",
)


def enable_metal_kernels() -> bool:
    """Patch the transformers conv1d fallbacks with fused Metal kernels.

    No-op on non-Darwin hosts or when MPS/the extension is unavailable.
    Returns True when at least one modeling module was patched.
    """
    if sys.platform != "darwin" or importlib.util.find_spec("torch") is None:
        return False
    import torch

    if not torch.backends.mps.is_available():
        return False

    try:
        from meowtrix.kernels.metal import conv1d
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"meowtrix metal kernels unavailable: {exc}", stacklevel=2)
        return False

    if not conv1d.available():
        warnings.warn(
            f"meowtrix metal conv1d extension failed to build: {conv1d._BUILD_ERROR}",
            stacklevel=2,
        )
        return False

    patched = False
    for name in _TARGET_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        patched = conv1d.install(module) or patched
    return patched
