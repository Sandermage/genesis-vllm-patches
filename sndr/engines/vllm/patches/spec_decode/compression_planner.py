# SPDX-License-Identifier: Apache-2.0
"""compression_planner — per-layer KV compression + backend plan.

A CompressionPlan is the canonical description of "what dtype is
each KV layer stored as, and which backend reads it". Replaces the
ad-hoc combination of:
  - global ``--attention-backend TURBOQUANT``
  - ``--kv-cache-dtype`` flag
  - ``GENESIS_G4_TQ_FORCE_SKIP_LAYERS=58,59``
  - ``GENESIS_ENABLE_G4_71B_*`` / ``G4_75_*`` (drafter backend route)
  - implicit ``physical kv_sharing`` rule

…with one structured object whose invariant is:

  shared-source layers MUST be stored in a format the reader kernel
  can natively read.

Planner does NOT apply anything to runtime — it emits the plan. The
backend_router consumes the plan; the existing G4_71b / G4_75 /
skip-list patches are the runtime-side that the plan describes.

Authored 2026-05-20 (C2 after β′-A bench).
Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("genesis.spec_decode.compression_planner")


# ----------------------- Plan dataclasses -----------------------

@dataclass
class LayerDecision:
    layer_idx: int
    layer_type: str       # 'sliding_attention' / 'full_attention' / 'mtp_*'
    role: str             # 'target' / 'drafter'
    kv_dtype: str         # 'turboquant_4bit_nc' / 'auto' (bf16/fp16) / 'fp8_e4m3'
    backend: str          # 'TURBOQUANT' / 'TRITON_ATTN' / 'FLASH_ATTN'
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "layer_type": self.layer_type,
            "role": self.role,
            "kv_dtype": self.kv_dtype,
            "backend": self.backend,
            "reason": self.reason,
        }


@dataclass
class CompressionPlan:
    profile: str
    model_id: str
    target_decisions: list[LayerDecision]
    drafter_decisions: list[LayerDecision]
    mtp_k: int | None
    physical_kv_sharing: bool
    bridge_enabled: bool
    workload_policy: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    # --- Convenience accessors -------------------------------

    @property
    def target_native_layers(self) -> list[int]:
        return [d.layer_idx for d in self.target_decisions
                if not d.kv_dtype.startswith(("turboquant", "fp8", "tq"))]

    @property
    def target_tq_layers(self) -> list[int]:
        return [d.layer_idx for d in self.target_decisions
                if d.kv_dtype.startswith(("turboquant", "tq"))]

    def drafter_backend(self) -> str | None:
        """Single backend if all drafter layers share one; else None."""
        backends = {d.backend for d in self.drafter_decisions}
        if len(backends) == 1:
            return next(iter(backends))
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "model_id": self.model_id,
            "target_decisions": [d.as_dict() for d in self.target_decisions],
            "drafter_decisions": [d.as_dict() for d in self.drafter_decisions],
            "mtp_k": self.mtp_k,
            "physical_kv_sharing": self.physical_kv_sharing,
            "bridge_enabled": self.bridge_enabled,
            "workload_policy": dict(self.workload_policy),
            "notes": self.notes,
        }


# ----------------------- Gemma 4 planner -----------------------

GEMMA4_DEFAULT_LAYER_TYPES: tuple[str, ...] = (
    ("sliding_attention",) * 5 + ("full_attention",)
) * 10
# = 50 sliding + 10 full = 60 total. Last sliding=58, last full=59.


def _gemma4_kv_share_targets(
    layer_types: tuple[str, ...],
    num_kv_shared: int = 0,
) -> dict[str, int]:
    """Replicate vLLM's _setup_gemma4_kv_sharing pick: for each
    drafter attention type, the LAST target layer of that type
    (before the kv_shared cutoff).
    """
    non_shared = layer_types[:len(layer_types) - num_kv_shared]
    out: dict[str, int] = {}
    for idx in range(len(non_shared) - 1, -1, -1):
        t = non_shared[idx]
        if t not in out:
            out[t] = idx
    return out


def plan_gemma4_tq_mtp_structured_k4(
    *,
    model_id: str = "cyankiwi/gemma-4-31B-it-AWQ-4bit",
    layer_types: tuple[str, ...] = GEMMA4_DEFAULT_LAYER_TYPES,
    num_kv_shared: int = 0,
    drafter_layer_count: int = 4,
    drafter_layer_types: tuple[str, ...] = (
        "sliding_attention", "sliding_attention", "sliding_attention",
        "full_attention",
    ),
    mtp_k: int = 4,
) -> CompressionPlan:
    """Emit the empirically-validated structured-workload profile.

    Decisions:
      target 0..(N-3) all TQ;
      target[last sliding] + target[last full] -> native (kv_sharing
          source layers; reader kernel = Triton must see native bytes);
      drafter 0..3 -> Triton native (matches source format);
      physical kv_sharing ON;
      no bridge.

    Workload policy:
      Allowed:  structured_count, tool_json
      Denied:   code_gen (Δ +7.4% below +10% threshold),
                free_chat, summarization
    """
    share_targets = _gemma4_kv_share_targets(layer_types, num_kv_shared)
    native_set = set(share_targets.values())

    target_decisions: list[LayerDecision] = []
    for idx, lt in enumerate(layer_types):
        if idx in native_set:
            target_decisions.append(LayerDecision(
                layer_idx=idx,
                layer_type=lt,
                role="target",
                kv_dtype="auto",
                backend="TRITON_ATTN",
                reason=(
                    f"kv_sharing source for drafter layers of type {lt!r}; "
                    "must be native so drafter Triton kernel reads bytes "
                    "natively (β′-A: PN271b KERNEL_STORAGE_DTYPE_MISMATCH "
                    "rule)"
                ),
            ))
        else:
            target_decisions.append(LayerDecision(
                layer_idx=idx,
                layer_type=lt,
                role="target",
                kv_dtype="turboquant_4bit_nc",
                backend="TURBOQUANT",
                reason="non-shared target layer; eligible for TQ compression",
            ))

    drafter_decisions: list[LayerDecision] = []
    for idx in range(drafter_layer_count):
        dlt = (drafter_layer_types[idx] if idx < len(drafter_layer_types)
               else "full_attention")
        # All drafter layers go Triton native (matches the
        # corresponding native target).
        drafter_decisions.append(LayerDecision(
            layer_idx=idx,
            layer_type=dlt,
            role="drafter",
            kv_dtype="auto",
            backend="TRITON_ATTN",
            reason=(
                "drafter reads target's native bf16 cache via physical "
                "kv_sharing; must use Triton (NHD native) so byte "
                "interpretation matches"
            ),
        ))

    return CompressionPlan(
        profile="gemma4-31b-tq-mtp-structured-k4",
        model_id=model_id,
        target_decisions=target_decisions,
        drafter_decisions=drafter_decisions,
        mtp_k=mtp_k,
        physical_kv_sharing=True,
        bridge_enabled=False,
        workload_policy={
            "allowed_workloads": ["structured_count", "tool_json"],
            "denied_workloads": [
                "code_gen", "free_chat", "summarization",
            ],
            "decision_rule": (
                "per-class TPS Δ vs TQ-only baseline >= +10% to enter "
                "allowed_workloads"
            ),
        },
        notes=(
            "Validated by β′-A bench 2026-05-20. structured workloads "
            "+24%, free-form workloads -51%, global geomean -14.7% — "
            "NOT a global default. Use behind explicit profile selection."
        ),
    )


def plan_gemma4_tq_default() -> CompressionPlan:
    """The current production default: TQ everywhere, MTP off."""
    layer_types = GEMMA4_DEFAULT_LAYER_TYPES
    return CompressionPlan(
        profile="gemma4-31b-tq-default",
        model_id="cyankiwi/gemma-4-31B-it-AWQ-4bit",
        target_decisions=[
            LayerDecision(
                layer_idx=idx,
                layer_type=lt,
                role="target",
                kv_dtype="turboquant_4bit_nc",
                backend="TURBOQUANT",
                reason="production-default TQ everywhere; MTP off",
            )
            for idx, lt in enumerate(layer_types)
        ],
        drafter_decisions=[],
        mtp_k=None,
        physical_kv_sharing=False,
        bridge_enabled=False,
        workload_policy={
            "allowed_workloads": [],
            "denied_workloads": [],
            "decision_rule": "MTP disabled; no workload-conditional logic",
        },
        notes="Canonical production posture as of 2026-05-20.",
    )


# ----------------------- Profile registry -----------------------

#: profile name -> factory function
PROFILES: dict[str, Any] = {
    "gemma4-31b-tq-default": plan_gemma4_tq_default,
    "gemma4-31b-tq-mtp-structured-k4": plan_gemma4_tq_mtp_structured_k4,
}


def build_plan(profile: str, **kwargs: Any) -> CompressionPlan:
    """Look up a profile factory and build the plan."""
    if profile not in PROFILES:
        raise ValueError(
            f"unknown profile {profile!r}; known: {sorted(PROFILES)}"
        )
    return PROFILES[profile](**kwargs)


__all__ = [
    "LayerDecision",
    "CompressionPlan",
    "plan_gemma4_tq_default",
    "plan_gemma4_tq_mtp_structured_k4",
    "build_plan",
    "PROFILES",
]
