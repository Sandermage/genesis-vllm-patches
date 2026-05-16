# SPDX-License-Identifier: Apache-2.0
"""Stage 5 contract test (Idea 1, approved by Sander 2026-05-07):

Every `env_flag` declared in PATCH_REGISTRY MUST be enumerated as a
constant on `vllm.sndr_core.env.Flags`. This guards against:

  - Typos in registry env_flag values (e.g. "GENESIS_ENABLE_PN56_QWEN3"
    vs "GENESIS_ENABLE_PN56_QWEN3CODER_XML_FALLBACK") — test fails at
    CI rather than silently defaulting to OFF in production.
  - New patches added to registry without a Flags constant — Flags
    must stay 1:1 with registry.
  - Stale Flags constants that no longer correspond to a registry entry
    (loose coupling — not enforced as ERROR, but logged as orphan).

Stage 6 will tighten this further by validating that every Flags
constant resolves to exactly one PATCH_REGISTRY entry's env_flag
(strict 1:1 bidirectional coverage).
"""
from __future__ import annotations

import re

import pytest


def _strip_prefix(flag: str) -> str:
    """Strip SNDR_ENABLE_/GENESIS_ENABLE_/SNDR_LEGACY_/GENESIS_LEGACY_ prefix."""
    for p in (
        "GENESIS_ENABLE_", "SNDR_ENABLE_",
        "GENESIS_LEGACY_", "SNDR_LEGACY_",
        "GENESIS_DISABLE_", "SNDR_DISABLE_",
    ):
        if flag.startswith(p):
            return flag[len(p):]
    return flag


@pytest.fixture(scope="module")
def registry_env_flags() -> set[str]:
    """All env_flag values from PATCH_REGISTRY (with prefix stripped)."""
    from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
    flags: set[str] = set()
    for meta in PATCH_REGISTRY.values():
        flag = meta.get("env_flag")
        if flag:
            flags.add(_strip_prefix(flag))
    return flags


@pytest.fixture(scope="module")
def declared_flags() -> set[str]:
    """All bare flag names declared on Flags class."""
    from vllm.sndr_core.env import known_flags
    return set(known_flags())


def test_every_registry_env_flag_is_in_Flags_class(registry_env_flags, declared_flags):
    """Every registry entry's `env_flag` must have a matching constant
    on `vllm.sndr_core.env.Flags`.

    If this fails, the patch was added to PATCH_REGISTRY without a
    matching Flags constant — `is_enabled(Flags.X)` would AttributeError
    at runtime. Add the constant to env.py.
    """
    missing = registry_env_flags - declared_flags
    assert missing == set(), (
        f"PATCH_REGISTRY has {len(missing)} env_flag(s) NOT declared on "
        f"vllm.sndr_core.env.Flags class:\n  "
        + "\n  ".join(sorted(missing))
        + "\n\nFix: add these as class constants on Flags in env.py."
    )


def test_no_orphan_Flags_constants(registry_env_flags, declared_flags):
    """Every Flags constant should map to a registry entry (loose check).

    Reports orphan constants as a list. Marked xfail because Flags has
    intentional meta flags (NO_PATCH_CACHE, FORCE_REAPPLY etc.) that
    don't correspond to registry entries — those are escape hatches,
    not patch-enable flags.

    To turn this into a hard check at Stage 6, the meta-flag set must
    be carved out via `is_meta_flag()`.
    """
    from vllm.sndr_core.env import is_meta_flag
    orphans = {
        f for f in (declared_flags - registry_env_flags)
        if not is_meta_flag(f)
    }
    if orphans:
        # Currently soft warning — Stage 5 has 18+ legacy/test flags that
        # aren't in registry. Stage 6 cleanup will reduce to zero.
        pytest.skip(
            f"{len(orphans)} Flags constant(s) without registry entry: "
            + ", ".join(sorted(orphans))
            + " — soft warning, will tighten to hard check at Stage 6."
        )


def test_registry_env_flags_use_canonical_prefix(registry_env_flags):
    """Every registry env_flag must start with one of the canonical
    prefixes (`GENESIS_ENABLE_`, `SNDR_ENABLE_`, `GENESIS_LEGACY_`,
    `SNDR_LEGACY_`, `GENESIS_DISABLE_`, `SNDR_DISABLE_`)."""
    from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
    invalid: list[tuple[str, str]] = []
    canonical_prefixes = (
        "GENESIS_ENABLE_", "SNDR_ENABLE_",
        "GENESIS_LEGACY_", "SNDR_LEGACY_",
        "GENESIS_DISABLE_", "SNDR_DISABLE_",
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
    from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
    missing = [pid for pid, meta in PATCH_REGISTRY.items() if "tier" not in meta]
    assert not missing, (
        f"{len(missing)} registry entries missing `tier` field: "
        + ", ".join(sorted(missing)[:10])
        + ("..." if len(missing) > 10 else "")
    )


def test_registry_family_field_present(registry_env_flags):
    """Stage 5 contract: every PATCH_REGISTRY entry must declare `family`."""
    from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
    missing = [pid for pid, meta in PATCH_REGISTRY.items() if "family" not in meta]
    assert not missing, (
        f"{len(missing)} registry entries missing `family` field: "
        + ", ".join(sorted(missing)[:10])
        + ("..." if len(missing) > 10 else "")
    )


def test_registry_tier_values_valid(registry_env_flags):
    """`tier` must be one of: community, engine."""
    from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
    bad = [
        (pid, meta.get("tier"))
        for pid, meta in PATCH_REGISTRY.items()
        if meta.get("tier") not in ("community", "engine")
    ]
    assert not bad, f"Invalid tier values: {bad[:5]}"


def test_registry_family_values_valid(registry_env_flags):
    """`family` must be one of the 19 approved engine subsystems."""
    from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
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
