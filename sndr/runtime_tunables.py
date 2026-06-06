# SPDX-License-Identifier: Apache-2.0
"""Registry of runtime tunable env knobs (distinct from patch enable flags).

Patch enable flags live in `PATCH_REGISTRY[<id>].env_flag` and toggle
whether a patch applies. Tunable knobs (this module) are runtime
parameters consumed by already-applied patches at request time, or by
the engine itself. Two examples:

  - `GENESIS_P82_THRESHOLD_SINGLE`  — float in [0,1], read by P82 to
                                       decide single-token acceptance.
  - `GENESIS_OBSERVABILITY`         — boolean, enables Wave 7 timing
                                       instrumentation across all patches.

Keeping the two registries separate prevents the audit gate
[R-011, audit_rules.py] from false-positiving a tunable as an unknown
env flag, and gives operators one place to enumerate all knobs.

Schema
------
Each entry is keyed by an exact env-var name (or a prefix when the
patch defines a family of variables, e.g. `GENESIS_PN16_*`). The value
is a `TunableKnob` describing intent + accepted values.

API
---
- `TUNABLE_KNOBS`            : dict[str, TunableKnob] — the registry.
- `is_known_tunable(name)`   : bool — exact name OR prefix match.
- `tunable_prefixes()`       : list[str] — back-compat for audit_rules.

Audit closure 2026-05-12 (P1-D): replaces the inline tuple in
`model_configs/audit_rules.py::_check_env_keys_exist`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class TunableKnob:
    """One runtime tunable env knob.

    Fields:
      kind:    "scalar" — single env var; `name` is exact.
               "family" — variable-arity; `name` is a prefix ending `_`.
      type:    Logical type for documentation + UI hints.
      default: String form of the default (`None` = patch reads None semantics).
      values:  When enum-like, the accepted values (string form).
      owner_patch: Patch ID the knob belongs to (empty for engine-wide knobs).
      doc:     One-line operator-facing description.
    """
    name: str
    kind: Literal["scalar", "family"]
    type: Literal["bool", "int", "float", "string", "enum"]
    default: str | None = None
    values: tuple[str, ...] = field(default_factory=tuple)
    owner_patch: str = ""
    doc: str = ""


# Single source of truth. Order matches operational frequency (most-used first).
TUNABLE_KNOBS: dict[str, TunableKnob] = {
    "GENESIS_OBSERVABILITY": TunableKnob(
        name="GENESIS_OBSERVABILITY", kind="scalar", type="bool",
        default="0",
        doc="Wave 7 per-patch timing instrumentation. Enable to record "
            "elapsed_ms + rss_delta_kb per patch at boot.",
    ),
    "GENESIS_PROFILE_RUN_CAP_M": TunableKnob(
        name="GENESIS_PROFILE_RUN_CAP_M", kind="scalar", type="int",
        default="4096",
        owner_patch="P72",
        doc="profile_run M cap. Unblocks --max-num-batched-tokens > 4096 on "
            "MoE configs without hitting the chunked-prefill scheduler "
            "warning.",
    ),
    "GENESIS_PROFILE_RUN_CAP": TunableKnob(
        name="GENESIS_PROFILE_RUN_CAP", kind="scalar", type="int",
        owner_patch="P72",
        doc="Alternate name kept for back-compat with v7.x start scripts.",
    ),
    "GENESIS_TQ_MAX_MODEL_LEN": TunableKnob(
        name="GENESIS_TQ_MAX_MODEL_LEN", kind="scalar", type="int",
        default="320000",
        doc="TurboQuant max-model-len cap. Operator-facing override of the "
            "vLLM-side limit; informs preallocation budget.",
    ),
    "GENESIS_PREALLOC_TOKEN_BUDGET": TunableKnob(
        name="GENESIS_PREALLOC_TOKEN_BUDGET", kind="scalar", type="int",
        doc="Token budget hint for preallocator. Used by TQ workspace and "
            "GDN buffer-pool sizing during warmup.",
    ),
    "GENESIS_BUFFER_MODE": TunableKnob(
        name="GENESIS_BUFFER_MODE", kind="scalar", type="enum",
        values=("auto", "persistent", "fresh"),
        default="auto",
        doc="Global buffer-acquisition mode. `auto` = patches decide; "
            "`persistent` forces pool reuse; `fresh` disables pools (debug).",
    ),
    "GENESIS_FLA_FWD_H_MAX_T": TunableKnob(
        name="GENESIS_FLA_FWD_H_MAX_T", kind="scalar", type="int",
        doc="FLA forward-h max tokens-per-shard cap. Used by FLA TP "
            "overflow preflight during orchestrator boot.",
    ),

    # Family-prefix knobs (variable arity within the namespace).
    "GENESIS_P82_THRESHOLD_": TunableKnob(
        name="GENESIS_P82_THRESHOLD_", kind="family", type="float",
        owner_patch="P82",
        doc="P82 SGLang threshold_single knobs (THRESHOLD_SINGLE float in [0,1]).",
    ),
    "GENESIS_P67_": TunableKnob(
        name="GENESIS_P67_", kind="family", type="string",
        owner_patch="P67",
        doc="P67 TurboQuant multi-query kernel knobs (NUM_KV_SPLITS, "
            "USE_UPSTREAM, USE_SPARSE_V).",
    ),
    "GENESIS_P68_P69_": TunableKnob(
        name="GENESIS_P68_P69_", kind="family", type="string",
        owner_patch="P68",
        doc="P68/P69 long-context tool reminder + auto-force-tool knobs.",
    ),
    "GENESIS_FLA_GUARD_": TunableKnob(
        name="GENESIS_FLA_GUARD_", kind="family", type="string",
        doc="FLA TP overflow preflight knobs (orchestrator gate).",
    ),
    "GENESIS_PN16_": TunableKnob(
        name="GENESIS_PN16_", kind="family", type="string",
        owner_patch="PN16",
        doc="PN16 lazy-reasoner V5/V7/V8 knobs (TOOL_THINK_BUDGET, "
            "CLASSIFIER_MAX_TOKENS, MAX_THINKING_TOKENS, THRESHOLD_CHARS, "
            "V1_LEGACY).",
    ),
    "GENESIS_PN59_": TunableKnob(
        name="GENESIS_PN59_", kind="family", type="string",
        owner_patch="PN59",
        doc="PN59 streaming-GDN orchestrator knobs.",
    ),
    "GENESIS_PN65_": TunableKnob(
        name="GENESIS_PN65_", kind="family", type="string",
        owner_patch="PN65",
        doc="PN65 v3 access-log knobs (LOG_HEALTH, QUIET_PATHS, "
            "KEEP_UVICORN_ACCESS).",
    ),
    "GENESIS_PN72_": TunableKnob(
        name="GENESIS_PN72_", kind="family", type="string",
        owner_patch="PN72",
        doc="PN72 frequency ngram drafter knobs (MIN_OBSERVATIONS, "
            "FREQUENCY_WINDOW).",
    ),
    "GENESIS_PN95_": TunableKnob(
        name="GENESIS_PN95_", kind="family", type="string",
        owner_patch="PN95",
        doc="PN95 Path C tier-aware cache knobs (CONFIG_KEY, TICK_EVERY, "
            "DEMOTE_FREE_MIB_THRESHOLD, PROMOTE_*, etc.).",
    ),
}


def is_known_tunable(name: str) -> bool:
    """True if `name` is a known tunable knob (exact name or family prefix)."""
    if not name:
        return False
    if name in TUNABLE_KNOBS and TUNABLE_KNOBS[name].kind == "scalar":
        return True
    for key, knob in TUNABLE_KNOBS.items():
        if knob.kind == "family" and name.startswith(key):
            return True
        if knob.kind == "scalar" and name == key:
            return True
    # Special case: P82 has both an exact (deprecated alias?) and a family
    # prefix style. Allow `GENESIS_P82_THRESHOLD*` as a family.
    if name.startswith("GENESIS_P82_THRESHOLD"):
        return True
    return False


def tunable_prefixes() -> tuple[str, ...]:
    """Back-compat helper for `model_configs/audit_rules.py`.

    Returns the prefix list the audit gate previously hardcoded.
    Behavior identical to a `startswith(...)` check against this tuple.
    """
    return tuple(
        key for key, knob in TUNABLE_KNOBS.items()
        if knob.kind == "family"
    ) + tuple(
        key for key, knob in TUNABLE_KNOBS.items()
        if knob.kind == "scalar"
    )


__all__ = [
    "TunableKnob",
    "TUNABLE_KNOBS",
    "is_known_tunable",
    "tunable_prefixes",
]
