# SPDX-License-Identifier: Apache-2.0
"""backend_router — derive per-Attention backend from a CompressionPlan.

Library code. Pure functions. Consumed by other patches (e.g., a future
G4_71b-replacement that reads its routing decision from a plan instead
of hard-coded prefix/head-size rules).

The router is intentionally simple — its only job is to answer:

  "For this attention layer's (role, layer_idx, layer_type), what
   backend should be used and what kv_cache_dtype should it report?"

…given a CompressionPlan. The runtime patches still do the actual
monkey-patching; the router just makes the routing rule explicit and
testable.

Authored 2026-05-20 (C3 after compression_planner).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
from typing import NamedTuple

from .compression_planner import CompressionPlan, LayerDecision

log = logging.getLogger("genesis.spec_decode.backend_router")


class BackendDecision(NamedTuple):
    backend: str          # AttentionBackendEnum name e.g. 'TRITON_ATTN'
    kv_cache_dtype: str   # 'auto' / 'turboquant_4bit_nc' / 'fp8_*'
    reason: str


def _find_decision(plan: CompressionPlan, role: str,
                   layer_idx: int) -> LayerDecision | None:
    if role == "target":
        for d in plan.target_decisions:
            if d.layer_idx == layer_idx:
                return d
    elif role == "drafter":
        for d in plan.drafter_decisions:
            if d.layer_idx == layer_idx:
                return d
    return None


def choose_backend(
    plan: CompressionPlan,
    *,
    role: str,
    layer_idx: int,
    default_backend: str = "TURBOQUANT",
    default_kv_dtype: str = "turboquant_4bit_nc",
) -> BackendDecision:
    """Return (backend, kv_cache_dtype, reason) for one Attention.

    If the layer isn't covered by the plan, return defaults. Caller
    can decide whether to enforce or warn.
    """
    decision = _find_decision(plan, role, layer_idx)
    if decision is None:
        return BackendDecision(
            backend=default_backend,
            kv_cache_dtype=default_kv_dtype,
            reason=(
                f"role={role!r} layer={layer_idx} not in plan; "
                f"falling back to defaults ({default_backend} / "
                f"{default_kv_dtype})"
            ),
        )
    return BackendDecision(
        backend=decision.backend,
        kv_cache_dtype=decision.kv_dtype,
        reason=(
            f"plan[{plan.profile}].{role}[{layer_idx}]: {decision.reason}"
        ),
    )


def validate_drafter_target_contract(plan: CompressionPlan) -> list[str]:
    """Return a list of contract violations between drafter and its
    kv-sharing target layers. Empty list = plan is internally
    consistent.

    The check enforces: if drafter reads via physical kv_sharing
    AND the target layer is decided as 'auto' (native), then the
    drafter MUST also use a backend that reads native bytes
    (TRITON_ATTN or FLASH_ATTN), not TURBOQUANT.
    """
    if not plan.physical_kv_sharing:
        return []
    violations: list[str] = []
    for d in plan.drafter_decisions:
        if d.backend == "TURBOQUANT":
            # Drafter using TQ kernel — its kv_sharing source layer
            # must ALSO be TQ. If any kv-share target is native,
            # that's the β contract bug.
            target_natives = plan.target_native_layers
            if target_natives:
                violations.append(
                    f"drafter[{d.layer_idx}] backend=TURBOQUANT but "
                    f"target shared-source layers {target_natives} are "
                    f"native bf16 — KERNEL_STORAGE_DTYPE_MISMATCH"
                )
        elif d.backend in ("TRITON_ATTN", "FLASH_ATTN"):
            # Drafter native reader — target shared-source must also
            # be native (else drafter would read TQ bytes as bf16).
            target_tqs = plan.target_tq_layers
            target_natives = plan.target_native_layers
            if not target_natives:
                violations.append(
                    f"drafter[{d.layer_idx}] backend={d.backend} but no "
                    f"target layer is native — drafter would read "
                    f"TQ-packed bytes natively (KERNEL_STORAGE_DTYPE_"
                    f"MISMATCH inverse)"
                )
    return violations


def env_overrides_for_plan(plan: CompressionPlan) -> dict[str, str]:
    """Translate a plan to the env-variable overrides that the
    existing Genesis runtime patches consume. Lets a launcher author
    say "I want this plan" once and get the right set of env flags.

    Returns a dict suitable for shell ``-e KEY=VAL`` expansion.
    """
    out: dict[str, str] = {}
    # 1) Skip-list for native target layers
    native = plan.target_native_layers
    if native:
        out["GENESIS_G4_TQ_FORCE_SKIP_LAYERS"] = ",".join(
            str(i) for i in native)
        out["GENESIS_ENABLE_G4_69_SKIP_LAYERS_NATIVE_BACKEND"] = "1"
        out["GENESIS_ENABLE_G4_70_PN259B_MIXED_ALLOC"] = "1"
        out["GENESIS_ENABLE_G4_70_PN259B_FAIL_FAST"] = "1"
        out["GENESIS_ENABLE_G4_70_PN259C_ROUTE_B"] = "1"
    # 2) Drafter backend routing.
    db = plan.drafter_backend()
    if db == "TRITON_ATTN":
        # Sliding-head (256) drafter -> G4_71b; full-head (512) -> G4_75
        out["GENESIS_ENABLE_G4_71B_DRAFTER_SLIDING_TRITON"] = "1"
        out["GENESIS_ENABLE_G4_75_DRAFTER_HEAD512_TRITON"] = "1"
        # Disable legacy drafter-side fixes that fought kv_sharing
        out["GENESIS_ENABLE_G4_71_DRAFTER_NATIVE_BACKEND"] = "0"
        out["GENESIS_ENABLE_G4_72_DRAFTER_NATIVE_SPEC"] = "0"
        out["GENESIS_ENABLE_G4_73_DRAFTER_PROFILE_SKIP"] = "0"
        out["GENESIS_ENABLE_G4_74_DRAFTER_HND_LAYOUT"] = "0"
    # 3) kv_sharing on/off.
    if plan.physical_kv_sharing:
        # The disable patch must be OFF -> upstream _setup_gemma4_kv_sharing
        # runs.
        out["GENESIS_ENABLE_G4_76_DISABLE_DRAFTER_KV_SHARING"] = "0"
    # 4) Bridge.
    if plan.bridge_enabled:
        out["GENESIS_ENABLE_G4_78_DRAFTER_TARGET_KV_BRIDGE"] = "1"
    else:
        out["GENESIS_ENABLE_G4_78_DRAFTER_TARGET_KV_BRIDGE"] = "0"
    # 5) Research opt-in envs — caller decides; we surface them in
    # the dict only when the plan would not be the safe default.
    if (plan.physical_kv_sharing
            and any(d.kv_dtype.startswith(("turboquant", "tq"))
                    for d in plan.target_decisions)):
        out["GENESIS_ALLOW_SPEC_DECODE_KV_ADAPTER"] = "1"
        # Whether to also need FUNCTIONAL_UNKNOWN depends on whether
        # the plan has a validating artifact — handled by safety_guard
        # at boot, not by this dict.
    return out


__all__ = [
    "BackendDecision",
    "choose_backend",
    "validate_drafter_target_contract",
    "env_overrides_for_plan",
]
