# SPDX-License-Identifier: Apache-2.0
"""Stage 5 contract test (Idea 1, approved by Sander 2026-05-07):

Every `env_flag` declared in PATCH_REGISTRY MUST be enumerated as a
constant on `sndr.env.Flags`. This guards against:

  - Typos in registry env_flag values (e.g. "GENESIS_ENABLE_PN56_QWEN3"
    vs "GENESIS_ENABLE_PN56_QWEN3CODER_XML_FALLBACK") — test fails at
    CI rather than silently defaulting to OFF in production.
  - New patches added to registry without a Flags constant — Flags
    must stay 1:1 with registry.
  - Stale Flags constants that no longer correspond to a registry entry
    (Stage 6 hardening, 2026-05-30: now enforced as hard-assert via
    test_no_orphan_Flags_constants; non-meta orphans fail CI).

Bidirectional 1:1 coverage between Flags and registry env_flag values
is now strict. Meta flags (apply-behavior escape hatches) are exempt
via is_meta_flag().
"""
from __future__ import annotations

import re

import pytest


def _strip_prefix(flag: str) -> str:
    """Strip canonical prefix family.

    Four semantic categories:
      - ENABLE/LEGACY/DISABLE × {GENESIS, SNDR} — standard patch-toggle
      - ALLOW × {GENESIS, SNDR} — operator-consent gate (PN274, R3 audit
        2026-05-21; documented in safety_guard.py / functional_artifact.py).
      - INFO × {GENESIS, SNDR} — info-marker (no toggle semantics; e.g.
        G4_T1 PR42006 overlay mount status — operator-visible flag that
        documents an external vendored-overlay condition rather than
        gating patch application). Phase 10.5 2026-06-01.
    """
    for p in (
        "GENESIS_ENABLE_", "SNDR_ENABLE_",
        "GENESIS_LEGACY_", "SNDR_LEGACY_",
        "GENESIS_DISABLE_", "SNDR_DISABLE_",
        "GENESIS_ALLOW_", "SNDR_ALLOW_",
        "GENESIS_INFO_", "SNDR_INFO_",
    ):
        if flag.startswith(p):
            return flag[len(p):]
    return flag


@pytest.fixture(scope="module")
def registry_env_flags() -> set[str]:
    """All env_flag values from PATCH_REGISTRY (with prefix stripped).

    Includes `env_flag_aliases` (2026-06-19): when two patches that share
    one engine file are consolidated into one registry entry, the absorbed
    patch's env flag is retained on the merged entry as a recognized alias
    so existing builtin YAMLs keep working AND the absorbed flag's `Flags`
    constant stays 1:1-covered (not an orphan). See PN298 (absorbed PN29).
    """
    from sndr.dispatcher.registry import PATCH_REGISTRY
    flags: set[str] = set()
    for meta in PATCH_REGISTRY.values():
        flag = meta.get("env_flag")
        if flag:
            flags.add(_strip_prefix(flag))
        for alias in meta.get("env_flag_aliases", ()) or ():
            flags.add(_strip_prefix(alias))
    return flags


@pytest.fixture(scope="module")
def declared_flags() -> set[str]:
    """All bare flag names declared on Flags class."""
    from sndr.env import known_flags
    return set(known_flags())


def test_every_registry_env_flag_is_in_Flags_class(registry_env_flags, declared_flags):
    """Every registry entry's `env_flag` must have a matching constant
    on `sndr.env.Flags`.

    If this fails, the patch was added to PATCH_REGISTRY without a
    matching Flags constant — `is_enabled(Flags.X)` would AttributeError
    at runtime. Add the constant to env.py.
    """
    missing = registry_env_flags - declared_flags
    assert missing == set(), (
        f"PATCH_REGISTRY has {len(missing)} env_flag(s) NOT declared on "
        f"sndr.env.Flags class:\n  "
        + "\n  ".join(sorted(missing))
        + "\n\nFix: add these as class constants on Flags in env.py."
    )


def test_no_orphan_Flags_constants(registry_env_flags, declared_flags):
    """Every non-meta Flags constant must map to a registry entry.

    Meta flags (NO_PATCH_CACHE, FORCE_REAPPLY, DISABLE_BOOT_PATCHES,
    TIER_OVERRIDE, NO_VERIFY, TELEMETRY etc.) are exempt — those are
    apply-behavior escape hatches, not patch-enable flags. They are
    identified via `is_meta_flag()`.

    Tightened from soft-warning to hard-assert on 2026-05-30 once the
    orphan count reached zero (Stage 6 target). Any new Flags constant
    added without a corresponding registry entry now fails CI rather
    than emitting a skip.
    """
    from sndr.env import is_meta_flag
    orphans = {
        f for f in (declared_flags - registry_env_flags)
        if not is_meta_flag(f)
    }
    assert orphans == set(), (
        f"{len(orphans)} Flags constant(s) without a registry entry:\n  "
        + "\n  ".join(sorted(orphans))
        + "\n\nFix: either add a PATCH_REGISTRY entry referencing the "
        "flag, mark it as a meta-flag in env.py, or remove the orphan."
    )


def test_registry_env_flags_use_canonical_prefix(registry_env_flags):
    """Every registry env_flag must start with one of the canonical
    prefixes:
      - ENABLE/LEGACY/DISABLE × {GENESIS, SNDR} — standard patch-toggle.
      - ALLOW × {GENESIS, SNDR} — operator-consent semantic (PN274, R3
        audit 2026-05-21).
      - INFO × {GENESIS, SNDR} — info-marker (G4_T1 PR42006 overlay
        mount status; Phase 10.5 2026-06-01). No toggle semantics —
        the operator-visible flag documents an external vendored-
        overlay condition rather than gating patch application.
    """
    from sndr.dispatcher.registry import PATCH_REGISTRY
    invalid: list[tuple[str, str]] = []
    canonical_prefixes = (
        "GENESIS_ENABLE_", "SNDR_ENABLE_",
        "GENESIS_LEGACY_", "SNDR_LEGACY_",
        "GENESIS_DISABLE_", "SNDR_DISABLE_",
        "GENESIS_ALLOW_", "SNDR_ALLOW_",
        "GENESIS_INFO_", "SNDR_INFO_",
    )
    for pid, meta in PATCH_REGISTRY.items():
        flag = meta.get("env_flag")
        if flag and not flag.startswith(canonical_prefixes):
            invalid.append((pid, flag))
    assert not invalid, (
        f"{len(invalid)} entries with non-canonical env_flag prefix:\n  "
        + "\n  ".join(f"{p}: {f}" for p, f in invalid)
    )


def test_registry_tier_field_present(registry_env_flags):
    """Stage 5 contract: every PATCH_REGISTRY entry must declare `tier`."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    missing = [pid for pid, meta in PATCH_REGISTRY.items() if "tier" not in meta]
    assert not missing, (
        f"{len(missing)} registry entries missing `tier` field: "
        + ", ".join(sorted(missing)[:10])
        + ("..." if len(missing) > 10 else "")
    )


def test_registry_family_field_present(registry_env_flags):
    """Stage 5 contract: every PATCH_REGISTRY entry must declare `family`."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    missing = [pid for pid, meta in PATCH_REGISTRY.items() if "family" not in meta]
    assert not missing, (
        f"{len(missing)} registry entries missing `family` field: "
        + ", ".join(sorted(missing)[:10])
        + ("..." if len(missing) > 10 else "")
    )


def test_registry_tier_values_valid(registry_env_flags):
    """`tier` must be one of: community, engine."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    bad = [
        (pid, meta.get("tier"))
        for pid, meta in PATCH_REGISTRY.items()
        if meta.get("tier") not in ("community", "engine")
    ]
    assert not bad, f"Invalid tier values: {bad[:5]}"


def test_registry_family_values_valid(registry_env_flags):
    """`family` must be one of the 19 approved engine subsystems."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    VALID_FAMILIES = {
        "tool_parsing", "reasoning", "serving",
        "attention.gdn", "attention.turboquant", "attention.flash",
        "spec_decode", "scheduler", "worker", "kv_cache",
        "moe", "quantization", "kernels", "compile_safety",
        "loader", "middleware", "memory", "observability",
        "lora", "multimodal",
        "model_specific",  # rare — for truly model-tied patches
        "streaming",       # PN200-PN203 — Wave 8 streaming runtime
        "offload",         # PN102/PN104/PN105 — CPU offload tier
        "gemma4",          # G4_01..G4_25 — Gemma 4 model subsystem (18+ patches)
        # 2026-06 vendor wave — families mirror on-disk patches/ subdirs:
        "detection",            # PN296/PN300/PN302 — patches/detection/
        "attention",            # PN351 — patches/attention/ (area-level, no subarea)
        "model_compat.gemma4",  # PN349 — patches/model_compat/gemma4/
        "quantization.marlin",  # PN347 — patches/quantization/marlin/
    }
    bad = [
        (pid, meta.get("family"))
        for pid, meta in PATCH_REGISTRY.items()
        if meta.get("family") not in VALID_FAMILIES
    ]
    assert not bad, (
        f"{len(bad)} entries with unknown family value:\n  "
        + "\n  ".join(f"{p}: {f}" for p, f in bad[:10])
    )
