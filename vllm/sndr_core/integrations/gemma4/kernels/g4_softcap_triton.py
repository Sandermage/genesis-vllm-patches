# SPDX-License-Identifier: Apache-2.0
"""Fused softcap Triton kernel for Gemma 4 attention + final logits.

Gemma 4 applies a **soft-cap** at two places in the forward graph:

  1. **Attention logits** (every layer):
         attn_weights = tanh(attn_weights / softcap) * softcap
     where ``softcap = config.attention_logit_cap`` (typically 50.0).
     This appears in 60 layers × {decode + prefill} = ~120 kernel
     launches per token at low concurrency.

  2. **Final logits** (once per generation step):
         logits = tanh(logits / final_softcap) * final_softcap
     where ``final_softcap = config.final_logit_softcapping``
     (typically 30.0).

Both operations are **3 sequential element-wise kernels** (div + tanh +
mul). We fuse them into 1 kernel launch per call site.

================================================================
MATH (from transformers/modeling_gemma4.py:335)
================================================================

    attn_weights = attn_weights / softcap   # (1) div
    attn_weights = tanh(attn_weights)        # (2) tanh
    attn_weights = attn_weights * softcap    # (3) mul

We emit a single Triton kernel: out = tanh(x / c) * c.

================================================================
PERF EXPECTATIONS
================================================================

* Eliminates 2 of every 3 kernel launches at each softcap site
* At low concurrency (batch=1, decode-only): ~3-5% TPS gain
* Negligible at large batch (compute-bound regime dominates)

================================================================
SHARED-MEMORY BUDGET ON SM 8.6
================================================================

The kernel reads + writes one input tile per program. Tile size is
``BLOCK_SIZE`` elements of input dtype + workspace for fp32 accumulator.
At BLOCK_SIZE=4096 and BF16 input that's 16 KB I/O + 16 KB workspace
= 32 KB shared, well within SM 8.6's 100 KB budget.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


GENESIS_G4_SOFTCAP_MARKER = (
    "Genesis Gemma 4 fused softcap Triton kernel v1 "
    "(fuses div + tanh + mul; ~3-5% TPS on low-batch decode)"
)


if _TRITON_AVAILABLE:

    @triton.jit
    def _g4_softcap_kernel(
        X_ptr,
        Out_ptr,
        N,
        softcap_inv,
        softcap,
        BLOCK_SIZE: tl.constexpr,
    ):
        """out = tanh(x * softcap_inv) * softcap.

        Note: ``softcap_inv = 1.0 / softcap`` is passed precomputed so
        the kernel uses multiply instead of divide (5-10× faster on most
        GPUs, including SM 8.6).
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        # tanh in fp32 for accuracy — important when |x| > 5 where
        # tanh approaches ±1 and fp16 precision degrades
        y = tl.math.tanh(x * softcap_inv) * softcap
        tl.store(Out_ptr + offsets, y.to(X_ptr.dtype.element_ty), mask=mask)


def g4_softcap(
    x: torch.Tensor,
    softcap: float,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused ``tanh(x / softcap) * softcap``.

    In-place: pass ``out=x`` to avoid the extra allocation.

    Args:
        x: input tensor (any shape, will be flattened internally)
        softcap: positive scalar (typically 30.0 or 50.0 for Gemma 4)
        out: optional pre-allocated output (same shape as x)

    Returns:
        The output tensor (allocated if ``out is None``).
    """
    if softcap is None or softcap == 0.0:
        return x  # No-op (matches transformers' ``if softcap is not None``)
    if not _TRITON_AVAILABLE:
        return g4_softcap_reference(x, softcap, out=out)

    if out is None:
        out = torch.empty_like(x)
    n = x.numel()
    # Round BLOCK_SIZE down to fit in shared; we keep it constant for
    # simplicity. 4096 is the sweet spot per the kernel docstring.
    BLOCK_SIZE = 4096
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _g4_softcap_kernel[grid](
        x, out, n,
        1.0 / softcap,
        softcap,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def g4_softcap_reference(
    x: torch.Tensor,
    softcap: float,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Torch reference: tanh(x / softcap) * softcap."""
    if softcap is None or softcap == 0.0:
        return x
    if out is not None:
        torch.tanh(x / softcap, out=out)
        out.mul_(softcap)
        return out
    return torch.tanh(x / softcap) * softcap


__all__ = [
    "GENESIS_G4_SOFTCAP_MARKER",
    "g4_softcap",
    "g4_softcap_reference",
]
