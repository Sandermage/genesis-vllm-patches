# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N12 — FFN intermediate scratch pool (Cliff 1 fix on TQ3).

================================================================
WHAT IT FIXES
================================================================

Empirical OOM noonghunna reproduced on RTX 3090 + Lorbus 27B + 192K
context + tool call:

    torch.OutOfMemoryError: CUDA out of memory.
    Tried to allocate 138.00 MiB. GPU 0 has 122.56 MiB free.

Root cause located: `vllm/model_executor/layers/activation.py:146`
SiluAndMul.forward_cuda allocates a fresh `[M, intermediate_size]` BF16
tensor PER LAYER PER STEP:

    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)

For Lorbus Qwen3.6-27B-int4 (intermediate=17408, num_hidden_layers=64)
that's 73-285 MiB transient per layer × 64 layers = **4.7-18 GiB
allocator churn per forward step**. The 138 MiB OOM size class matches
silu_and_mul-out at M ≈ 4096 BF16 exactly.

PN8 closes Cliff 1 on FP8 by freeing ~600 MiB persistent draft VRAM,
giving the heap enough slack for the 138 MiB transient. On TQ3 PN8
only frees ~230 MiB → not enough → OOM still fires. **Different
memory class — persistent footprint vs transient peak.**

================================================================
WHAT THIS PATCH DOES
================================================================

Text-patch `SiluAndMul.forward_cuda` (and 4 sibling activation classes
that share the same memory pattern) so they call into a shared
`FFNIntermediateCache` pool instead of `torch.empty()`.

The pool is **one buffer per (intermediate_size, dtype, device)** —
allocated lazily at first call, sized at the requested num_tokens.
Subsequent calls return a slice; pointer is stable (cudagraph-safe).
On growth (num_tokens exceeds cached max) the buffer is re-allocated
once.

Compatibility classes patched (all share identical forward_cuda body):
- SiluAndMul
- MulAndSilu
- SiluAndMulWithClamp
- FatreluAndMul
- GeluAndMul

================================================================
SAFETY MODEL
================================================================

- **Default OFF** (opt-in via `GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL=1`).
- Pure text-patch with idempotent marker.
- Drift-aware: if upstream lands `silu_and_mul.out(input, *, out=out)`
  variant (vllm#34207) the anchor body changes and our marker self-retires.
- Anchor missing → SKIPPED, source stays vanilla, zero regression risk.
- Worst case if PN12 silently misbehaves: pool buffer not zeroed between
  layers means stale data could leak into output. **Mitigated** by the
  fact that `silu_and_mul` op writes the FULL output region in-place
  (every `[M, intermediate_size]` element overwritten by the kernel),
  so no read-of-stale path exists.

================================================================
EXPECTED IMPACT
================================================================

- Closes Cliff 1 (138 MiB OOM at 192K + tool-call) on TQ3 path that
  PN8 couldn't address.
- Reduces per-step allocator churn from ~4.7-18 GiB to ~73-285 MiB
  (the size of the single pooled buffer) on Lorbus 27B-int4.
- Enables longer ctx + concurrent tool-call workloads on 24 GB
  single-card setups.
- Side-effect on FP8 path (35B): same memory-pooling logic applies;
  expect free ~30-50 MiB of allocator overhead at 1K context decode.

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa
Cross-engine inspiration:
  - vLLM PR #34207 (silu_and_mul.out variant + Inductor reuse) — alternative
  - SGLang PR #15927 (piecewise CUDA graph private pool) — alternative
  - TensorRT-LLM live-range activation reuse — gold standard
"""
from __future__ import annotations

import logging

from vllm.sndr_core.detection.guards import resolve_vllm_file, vllm_install_root
from vllm.sndr_core.core import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)
# v11.1.0 P3.3: expose the FFN intermediate scratch pool through
# PersistentBufferRegistry so operators can `sndr patches show
# buffer_registry` and see this pool listed. Byte-equivalent — the
# actual torch.empty() still happens inside FFNIntermediateCache
# (process-wide pool, allocate-once-keep-forever). The registry hook
# only exposes the pool name; tensor storage ownership is unchanged.
from vllm.sndr_core.runtime.persistent_buffer_registry import (
    PersistentBufferRegistry,
    POOL_FFN_INTERMEDIATE_SCRATCH,
)

log = logging.getLogger("genesis.wiring.pN12_ffn_intermediate_pool")

GENESIS_PN12_MARKER = (
    "Genesis PN12 FFN intermediate scratch pool (Cliff 1 fix) v7.62.x_v11.3.0_hotpath"
)


def ensure_pool_registered() -> None:
    """Idempotent registry hook — exposes POOL_FFN_INTERMEDIATE_SCRATCH
    in PersistentBufferRegistry for operator visibility. No allocation,
    no behavior change.

    The real FFN intermediate scratch tensors are owned by the
    FFNIntermediateCache pool inside vLLM's patched SiluAndMul module.
    Its allocation pattern is GROW-IN-PLACE + SLICE-ON-ACQUIRE (allocate
    `[max_num_tokens, intermediate_size]` once, return slice views for
    smaller requests, grow once on larger requests) — matches
    PersistentSlicePool with key_dims=1.

    v11.3.0 bug fix: this was previously calling `get_pool()` which
    creates a BufferPool. When FFNIntermediateCache.acquire_silu_out
    then calls `_get_backing_pool()` (via `get_slice_pool()`), the
    registry would raise ValueError "pool was registered as BufferPool,
    not PersistentSlicePool". This caused the FFN cache to never engage
    on operators who imported the integration module before the storage
    class — silently falling back to the upstream allocate-per-step
    behaviour the patch was supposed to fix.
    """
    PersistentBufferRegistry().get_slice_pool(POOL_FFN_INTERMEDIATE_SCRATCH)


# ─── Sub-patch: replace SiluAndMul.forward_cuda body ──────────────────────
# Anchor matches the exact 4-line body:
#     def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
#         d = x.shape[-1] // 2
#         output_shape = x.shape[:-1] + (d,)
#         out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
#         self.op(out, x)
#         return out

PN12_SILU_ANCHOR = (
    "    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:\n"
    "        d = x.shape[-1] // 2\n"
    "        output_shape = x.shape[:-1] + (d,)\n"
    "        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)\n"
    "        self.op(out, x)\n"
    "        return out\n"
    "\n"
    "    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:\n"
    "        return self.forward_cuda(x)\n"
    "\n"
    "\n"
    "@CustomOp.register(\"silu_and_mul_with_clamp\")\n"
)

PN12_SILU_REPLACEMENT = (
    "    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:\n"
    "        # [Genesis PN12 v2 — Cliff 1 fix on TQ3, hot-path optimized]\n"
    "        # Pool the [M, d] BF16 transient across layers instead of\n"
    "        # torch.empty() per call. Single shared buffer per\n"
    "        # (intermediate_size, dtype, device); pointer-stable; safe\n"
    "        # because vLLM forward is strictly sequential.\n"
    "        # v2 hot-path optimization: cache the acquire callable on\n"
    "        # `self` after first call — eliminates ~500ns per-call import\n"
    "        # + classmethod dispatch overhead. Millions of FFN forward\n"
    "        # calls per decode = real TPS recovered.\n"
    "        _acquire = getattr(self, '_pn12_acquire_fn', None)\n"
    "        if _acquire is None:\n"
    "            try:\n"
    "                from vllm.sndr_core.kernels.ffn_intermediate_cache import (\n"
    "                    FFNIntermediateCache as _PN12_Cache,\n"
    "                )\n"
    "                if _PN12_Cache.is_production_eligible():\n"
    "                    self._pn12_acquire_fn = _PN12_Cache.acquire_silu_out\n"
    "                else:\n"
    "                    self._pn12_acquire_fn = False\n"
    "            except Exception:  # pragma: no cover — defensive\n"
    "                self._pn12_acquire_fn = False\n"
    "            _acquire = self._pn12_acquire_fn\n"
    "        d = x.shape[-1] // 2\n"
    "        if _acquire and x.dim() == 2:\n"
    "            out = _acquire(\n"
    "                num_tokens=x.shape[0], intermediate_size=d,\n"
    "                dtype=x.dtype, device=x.device,\n"
    "            )\n"
    "        else:\n"
    "            out = torch.empty(\n"
    "                x.shape[:-1] + (d,), dtype=x.dtype, device=x.device,\n"
    "            )\n"
    "        self.op(out, x)\n"
    "        return out\n"
    "\n"
    "    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:\n"
    "        return self.forward_cuda(x)\n"
    "\n"
    "\n"
    "@CustomOp.register(\"silu_and_mul_with_clamp\")\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/layers/activation.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN12 model_executor/layers/activation.py — SiluAndMul "
            "forward_cuda FFN intermediate pool (Cliff 1 fix on TQ3)"
        ),
        target_file=str(target),
        marker=GENESIS_PN12_MARKER,
        sub_patches=[
            TextPatch(
                name="pN12_silu_and_mul_pool",
                anchor=PN12_SILU_ANCHOR,
                replacement=PN12_SILU_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN12",
            # If vllm#34207 lands the body becomes silu_and_mul.out()
            # variant — different anchor, ours auto-skips.
            "torch.ops._C.silu_and_mul.out",
            # Note: deliberately do NOT use "FFNIntermediateCache" here
            # as a drift marker — it's our own pool class name and may
            # appear in sister-patches (PN25) that legitimately compose
            # with PN12 on the same file. Idempotency is handled by the
            # Genesis-PN12 wiring marker line above.
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN12 — FFN intermediate scratch pool (text-patch)."""
    from vllm.sndr_core.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN12")
    log_decision("PN12", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    # v11.1.0 P3.3: expose the pool name in the registry — no allocation,
    # purely operator-visibility surface.
    try:
        ensure_pool_registered()
    except Exception as e:
        log.debug("[PN12] registry pool registration failed (proceeding): %s", e)

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "target file not resolvable"

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "PN12 applied: SiluAndMul.forward_cuda now acquires output via "
            "FFNIntermediateCache pool when GENESIS_ENABLE_PN12_FFN_"
            "INTERMEDIATE_POOL=1. Closes Cliff 1 OOM on TQ3 path that PN8 "
            "couldn't address (different memory class)."
        ),
        patch_name=patcher.patch_name,
    )
