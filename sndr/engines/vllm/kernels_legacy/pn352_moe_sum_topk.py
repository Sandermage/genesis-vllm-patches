# SPDX-License-Identifier: Apache-2.0
"""PN352 — Triton moe_sum for top-k values the compiled CUDA op lacks.

Genesis counterpart of OPEN vllm PR #44557 (xyang16, "Support more topk
values in moe_sum kernel"). The upstream PR adds top-k cases to the
compiled CUDA switch in ``csrc/moe/moe_align_sum_kernels.cu`` — which we
CANNOT vendor onto a prebuilt nightly wheel (no recompilation). This
module provides the same win as a Triton kernel launched from the
Python call site instead.

What this replaces
------------------
``vllm/_custom_ops.moe_sum`` dispatches into ``_moe_C.moe_sum`` whose
switch covers top-k ∈ {2, 3, 4} only (verified in csrc at our pin
g303916e93). Anything else — including **top-k=8 used by
Qwen3.6-35B-A3B (num_experts_per_tok=8, 40 layers)** — falls back to
``at::sum_out(output, input, 1)``: a generic TensorIterator reduction
that costs several kernel launches + iterator setup per call. The PR
author measures ~-700 us per decode step on a 40-layer top-k=8 MoE
(-1-3 % decode TPOT).

Kernel design
-------------
``out[m, :] = sum_k in[m, k, :]`` with K unrolled at compile time
(``tl.static_range``), fp32 accumulation (matches ATen acc_type for
half/bf16 — same or better numerics than the fallback), one program
per (token, hidden-block).

Genesis-side guards (in the ``moe_sum_topk`` wrapper)
-----------------------------------------------------
- non-CUDA / non-contiguous input -> upstream fallback
- any Triton failure -> single-strike disable + upstream fallback
- iron-rule #12: logs ONE line on first successful hot-path execution
"""
from __future__ import annotations

import logging

import torch

log = logging.getLogger("genesis.kernels.pn352")

_DISABLED = False
_FIRST_CALL_LOGGED = False

try:
    import triton
    import triton.language as tl

    _TRITON_OK = True
except Exception:  # noqa: BLE001 — no triton on this host (CI / docs build)
    _TRITON_OK = False


if _TRITON_OK:

    @triton.jit
    def _moe_sum_topk_kernel(
        in_ptr,
        out_ptr,
        hidden_size,
        TOPK: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs = pid_h * BLOCK + tl.arange(0, BLOCK)
        mask = offs < hidden_size
        base = pid_m * TOPK * hidden_size
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for k in tl.static_range(TOPK):
            acc += tl.load(
                in_ptr + base + k * hidden_size + offs, mask=mask, other=0.0
            ).to(tl.float32)
        tl.store(
            out_ptr + pid_m * hidden_size + offs,
            acc.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


def moe_sum_topk(input: torch.Tensor, output: torch.Tensor) -> bool:
    """Triton ``output = input.sum(dim=1)`` for [num_tokens, topk, hidden].

    Returns True when the Triton path ran; False means the caller must
    use the upstream ``ops.moe_sum`` fallback. Never raises.
    """
    global _DISABLED, _FIRST_CALL_LOGGED
    if _DISABLED or not _TRITON_OK:
        return False
    if not (input.is_cuda and output.is_cuda):
        return False
    if not (input.is_contiguous() and output.is_contiguous()):
        return False
    if input.dim() != 3 or output.dim() < 2:
        return False
    # CRITICAL (2026-06-10 OOB post-mortem): `input` is the PREALLOCATED
    # intermediate_cache3 scratch — its row count is the CACHE capacity
    # (e.g. 8192), NOT the live token count. The compiled upstream op
    # derives num_tokens from OUTPUT (`output.numel() / hidden_size`);
    # sizing the grid from input.shape[0] writes past the end of
    # `output` -> CUDA illegal memory access on the first real request
    # (dummy-run shapes masked it: there num_tokens == cache rows).
    topk = input.shape[1]
    hidden = input.shape[-1]
    num_tokens = output.numel() // hidden
    if num_tokens == 0:
        return True  # nothing to do; output untouched is fine for 0 rows
    if input.shape[0] < num_tokens or output.numel() != num_tokens * hidden:
        return False
    try:
        BLOCK = 1024 if hidden >= 1024 else max(
            16, 1 << (hidden - 1).bit_length()
        )
        grid = (num_tokens, triton.cdiv(hidden, BLOCK))
        _moe_sum_topk_kernel[grid](
            input, output, hidden, TOPK=topk, BLOCK=BLOCK,
        )
        if not _FIRST_CALL_LOGGED:
            log.info(
                "[PN352] triton moe_sum_topk first call OK: "
                "tokens=%d topk=%d hidden=%d BLOCK=%d",
                num_tokens, topk, hidden, BLOCK,
            )
            _FIRST_CALL_LOGGED = True
        return True
    except Exception as e:  # noqa: BLE001
        _DISABLED = True
        log.warning(
            "[PN352] triton moe_sum_topk failed (%s) — single-strike "
            "disable, reverting to upstream ops.moe_sum for the rest of "
            "this process",
            e,
        )
        return False
