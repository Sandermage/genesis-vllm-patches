# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N25 — SiluAndMul.forward_native via opaque custom op.

Sister-patch to PN12. PN12 covers the eager-mode dispatch
(`forward_cuda`); PN25 covers the compile-mode dispatch
(`forward_native`) that Inductor inlines and lowers to
`empty_strided_cuda(...)`, completely bypassing PN12's pool.

================================================================
WHAT IT FIXES
================================================================

club-3090 issue #16 (noonghunna 2026-04-30, VolandBerlioz Reddit
trace, ampersandru confirmation): PN12-equivalent sidecar applies
cleanly at boot but is bypassed at runtime when vLLM compiles the
forward graph. The OOM site is

    inductor_cache/...py:1208
      buf9 = empty_strided_cuda((s18, 17408), (17408, 1), torch.float16)

That's the FFN intermediate buffer for chunked prefill at
`s18 = max_num_batched_tokens` × `intermediate_size = 17408`
(Qwen3.6-27B). 137.6 MiB transient on a 24 GB 3090 with ~131 MiB free.

Genesis stack inherits the same flaw — our PN12 patches only
`forward_cuda`. We don't see it in PROD only because our 27B Lorbus
config doesn't go through Inductor for this kernel. A future config
or model could expose the leak.

================================================================
WHAT THIS PATCH DOES
================================================================

1. Registers `genesis::silu_and_mul_pooled` as `torch.library.custom_op`
   in `vllm/_genesis/kernels/silu_and_mul_customop.py`. Inductor
   treats custom ops as opaque — emits a call, does NOT inline.

2. Text-patches `SiluAndMul.forward_native` body to dispatch through
   the opaque op when registration succeeded; falls back to vanilla
   `F.silu(x[..., :d]) * x[..., d:]` when the op is unavailable
   (torch<2.4, CPU-only build, etc.).

================================================================
COMPOSITION WITH PN12
================================================================

PN12: patches `forward_cuda` (eager dispatch path).
PN25: patches `forward_native` (compile dispatch path via opaque op).

Both can be enabled simultaneously. They patch different methods
and never collide. Pool is shared — `FFNIntermediateCache` is a
singleton keyed on `(intermediate_size, dtype, device)`.

If only PN12 enabled: eager workloads pool, compile workloads leak.
If only PN25 enabled: compile workloads pool, eager workloads leak.
If both enabled: full coverage. Recommended for inductor-heavy configs.

================================================================
SAFETY MODEL
================================================================

- env: `GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE=1`
- default OFF; opt-in.
- Idempotent (marker check).
- Drift-aware: if upstream lands `silu_and_mul.out(input, *, out=...)`
  variant or any other `forward_native` body change, anchor doesn't
  match → SKIPPED, source stays vanilla, zero regression.
- Op registration failure (torch<2.4, dynamo rejection of fake impl)
  → text-patch DOES still apply but the runtime branch falls back to
  the vanilla `F.silu(x[..., :d]) * x[..., d:]` math. Zero observable
  behavior change vs upstream.

================================================================
EXPECTED IMPACT
================================================================

- Closes Cliff 1 mech B on inductor-compiled FFN forward
  (the bug noonghunna's club-3090 hit on real OpenCode).
- Reduces per-step allocator churn under torch.compile from
  ~4.7-18 GiB to single pooled buffer (~73-285 MiB).
- Cost: 1 opaque op per FFN forward = Inductor cannot fuse SiluAndMul
  with surrounding ops. Estimated overhead 2-3% TPS on configs that
  previously fused (mostly synthetic; real workloads with KV cache
  are bandwidth-bound and won't notice).

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa
Cross-engine inspiration:
  - club-3090#16 (noonghunna 2026-04-30, work-in-progress on same flaw)
  - PR vllm#34207 (silu_and_mul.out variant — alternative upstream path)
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pn25_silu_inductor_safe_pool")

GENESIS_PN25_MARKER = (
    "Genesis PN25 SiluAndMul.forward_native opaque-op pool v7.65"
)


# ─── Text-patch: replace SiluAndMul.forward_native body ──────────────
PN25_FORWARD_NATIVE_ANCHOR = (
    "    @staticmethod\n"
    "    def forward_native(x: torch.Tensor) -> torch.Tensor:\n"
    "        \"\"\"PyTorch-native implementation equivalent to forward().\"\"\"\n"
    "        d = x.shape[-1] // 2\n"
    "        return F.silu(x[..., :d]) * x[..., d:]\n"
)

PN25_FORWARD_NATIVE_REPLACEMENT = (
    "    @staticmethod\n"
    "    def forward_native(x: torch.Tensor) -> torch.Tensor:\n"
    "        \"\"\"PyTorch-native — Genesis PN25 routes through opaque\n"
    "        custom op so torch.compile/Inductor cannot inline the FFN\n"
    "        intermediate alloc; pool from FFNIntermediateCache instead.\n"
    "        Falls back to vanilla math when registration unavailable.\n"
    "        \"\"\"\n"
    "        try:\n"
    "            from vllm._genesis.kernels.silu_and_mul_customop import (\n"
    "                get_op_callable as _genesis_pn25_get_op,\n"
    "            )\n"
    "            _genesis_pn25_op = _genesis_pn25_get_op()\n"
    "            if _genesis_pn25_op is not None:\n"
    "                return _genesis_pn25_op(x)\n"
    "        except Exception:  # pragma: no cover — defensive fallback\n"
    "            pass\n"
    "        d = x.shape[-1] // 2\n"
    "        return F.silu(x[..., :d]) * x[..., d:]\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/layers/activation.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN25 model_executor/layers/activation.py — SiluAndMul "
            "forward_native opaque-op pool (Inductor-safe Cliff 1 mech B)"
        ),
        target_file=str(target),
        marker=GENESIS_PN25_MARKER,
        sub_patches=[
            TextPatch(
                name="pN25_silu_and_mul_forward_native_opaque",
                anchor=PN25_FORWARD_NATIVE_ANCHOR,
                replacement=PN25_FORWARD_NATIVE_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN25",
            "_genesis_pn25_op",
            # If upstream lands silu_and_mul.out variant or rewrites
            # forward_native to use the C op directly, anchor will miss
            # naturally; these markers are extra belt-and-suspenders.
            "torch.ops._C.silu_and_mul.out",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN25 — SiluAndMul.forward_native via opaque custom op.

    Two-stage:
    1. Pre-register the custom op (lazy — does not import torch heavy
       paths until first PN25 enable). Failure to register is a soft
       degradation: text-patch still applies, runtime falls back to
       vanilla `F.silu * mul` math. We still patch because the dispatch
       indirection through the (failed) op call is a no-op cost wise.
    2. Apply the text-patch to `forward_native`.
    """
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN25")
    log_decision("PN25", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    # Pre-register op (best-effort; failure is soft).
    try:
        from vllm._genesis.kernels.silu_and_mul_customop import (
            _register_op_once,
        )
        _register_op_once()
    except Exception as e:
        log.info(
            "[PN25] pre-registration failed (%s) — text-patch will "
            "still apply with runtime fallback to vanilla math", e,
        )

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "target file not resolvable"

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "PN25 applied: SiluAndMul.forward_native now dispatches "
            "through genesis::silu_and_mul_pooled opaque custom op so "
            "torch.compile/Inductor cannot inline the FFN intermediate "
            "alloc. Closes Cliff 1 mech B on inductor-compiled FFN "
            "forward (club-3090#16 reproducer class). Sister to PN12 — "
            "PN12 covers eager forward_cuda, PN25 covers compile "
            "forward_native; both share the same FFNIntermediateCache "
            "pool and can be enabled simultaneously without conflict."
        ),
        patch_name=patcher.patch_name,
    )
