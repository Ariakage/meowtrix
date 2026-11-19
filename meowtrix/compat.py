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
