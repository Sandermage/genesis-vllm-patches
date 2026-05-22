# SPDX-License-Identifier: Apache-2.0
"""SNDR Core apply — shadow comparison: PatchSpec-driven order vs legacy.

PR38 Day 5 (2026-05-08): before flipping `orchestrator.run()` to use
`PatchSpec.apply_module` directly (Day 6-8), surface differences
between the two apply orders so operators can audit them off-line.

Two sources of order:

  1. **Legacy** — `apply._state.PATCH_REGISTRY` (list of (name, fn)),
     populated by `@register_patch` decorators in `_per_patch_dispatch.py`.
     This is the order Genesis has been running for years.

  2. **Spec-driven** — `dispatcher.iter_patch_specs()` yields a PatchSpec
     per `dispatcher.PATCH_REGISTRY` entry. Order today is registry
     dict-iteration order.

`compare_apply_orders()` returns a structured diff:

  - `legacy_only`: registered fn names with no PatchSpec match
  - `spec_only`: PatchSpec patch_ids with no legacy fn match
  - `legacy_count` / `spec_count`: total per-source counts
  - `coverage_pct`: fraction of spec-driven patches that have an
                    `apply_module` (i.e. could actually run via the
                    new dispatch loop)

CLI:

    python -m vllm.sndr_core.apply.shadow

Author: Sandermage (Sander) Barzov Aleksandr.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("genesis.apply.shadow")


# ─── Known divergent patches (P1-1 audit closure 2026-05-08) ─────────────
#
# Patches intentionally listed only in `dispatcher.PATCH_REGISTRY` and
# not in the legacy `_per_patch_dispatch.py` registry. Two reasons:
#
#   (a) registry-only documentation entries with no `apply_module`
#       — legacy ledger rows for retired/preflight/research patches
#       that don't have a runtime apply path. Adding @register_patch
#       for them would create a dummy dispatcher with no behavior.
#
#   (b) spec-only patches with `apply_module` set — these are the
#       direction the migration is going (registry as single source
#       of truth). Adding them to the legacy parking-lot module would
#       defeat the migration. Listed here so CI gate doesn't false-
#       positive on the intentional gap.
#
# Any new spec_only patch NOT in this set will surface as
# `unexpected_spec_only` and fail `--strict` mode. To add an entry
# here, the patch must either (a) lack an apply_module on purpose, or
# (b) be reviewed and confirmed as a registry-driven-only addition.
KNOWN_SPEC_ONLY_PATCHES: frozenset[str] = frozenset({
    # Category (a): registry-only ledger / preflight rows
    "P102",            # Spec-decode metadata + disagreement tracker (no apply yet)
    "P51",             # TQ-active runtime guard (legacy lifecycle, no on-disk impl)
    "PN60",            # Quant arg vs config.json validator (preflight DX)
    "PN63",            # fp8_e5m2 advisory (gpu_profile recommendation only)
    "PN64",            # Marlin MoE per-SM tuning placeholder for SM 12.0
    # Category (b): spec-only patches with apply_module — registry-
    # driven loop is the canonical path; legacy parking lot is going
    # away (PR38 Day 6-8 migration in progress).
    "P69",             # Long-context tool-format reminder (paired with P68)
    "PN40-classifier", # PN40 sub-D workload classifier middleware
    # Category (c): UNIFIED_CONFIG 2026-05-09+ spec-driven additions
    # — registered through patches/* modules with apply_module set,
    # but no legacy @register_patch entry (canonical path is
    # registry-driven from inception).
    "PN16_V6",         # Streaming <think> truncator middleware (Sprint 4)
    "PN122",  # renamed from SPRINT26_CG_DISPATCH_TRACE 2026-05-14
    # Category (c) continued — Phase 3 (2026-05-21) bucket 3+4
    # spec-driven onboarding. These patches were relocated to their
    # technical-area canonical home (spec_decode/ or attention/
    # turboquant/) AND made spec-only at the same time. Their apply()
    # is invoked by the registry-driven dispatcher loop, not by the
    # legacy @register_patch table. Adding @register_patch for them
    # would re-introduce the parking-lot dependency that PR38 is
    # migrating away from.
    #
    # Bucket 3 (spec_decode drafter routing relocated from gemma4/):
    "G4_71",           # drafter native attn-backend forcing
    "G4_71B",          # drafter sliding-window Triton routing
    "G4_72",           # drafter native KV cache spec
    "G4_73",           # drafter profile-skip
    "G4_74",           # drafter HND layout
    "G4_75",           # drafter head_size=512 Triton route
    "G4_76",           # disable drafter KV sharing
    "G4_78",           # drafter target KV bridge (lifecycle=retired)
    #
    # Bucket 1 (spec_decode probes — diagnostic):
    "PN262",           # FlashAttn drafter trace
    "PN262B",          # KV alloc trace
    #
    # Bucket 4 (TurboQuant cherry-pick overlay loader stack
    # relocated from gemma4/ to attention/turboquant/):
    "G4_19B",          # TQ KV spec integration (case-sensitive
                       # legacy register entry is 'G4_19b ...')
    "G4_19C",          # K,V round-trip attention wrapper
    "G4_31",           # preserve TQ dtype
    "G4_32",           # TQ validation bypass
    "G4_60A",          # TQ sliding-window spec
    "G4_60B",          # TQ overlay loader (turboquant_attn)
    "G4_60C",          # TQ overlay loader (triton_turboquant_decode)
    "G4_60D",          # TQ overlay loader (triton_turboquant_store)
    "G4_60E",          # KV cache utils overlay
    "G4_60G",          # attention dispatch overlay
    "G4_60H",          # TQ config augment overlay
    "G4_60K",          # arg_utils skip-list plumbing
    "G4_61",           # TQ shared workspace
    "G4_62",           # TQ kernel warmup
    "G4_67",           # TQ spec-verify routing
    "G4_68",           # TQ spec CG-downgrade overlay
    "G4_69",           # skip-layers native backend
    #
    # Bucket-6 R3 closure (2026-05-21) — marker-only registry rows:
    # envs read inside the bind-mount overlay, no apply_module by
    # design. Registered to close R3 config-keys catalog gap.
    "G4_70",           # mixed-allocator routing
    "G4_70B",          # mixed-allocator FAIL_FAST
    "G4_70C",          # skip-list plumbing companion
    "PN256",           # raw-K/V continuation inside overlay
    "PN261",           # TQ decode cache layout assert
    "PN271",           # KV contract audit (registry-only)
    "PN274",           # spec-decode KV adapter coordinator (lifecycle=coordinator)
})


# ─── Order extraction ─────────────────────────────────────────────────────


def _legacy_apply_names() -> list[str]:
    """Return the ordered list of registered apply-function names from
    `_per_patch_dispatch.py` (via `@register_patch`)."""
    # Force-import the parking lot module so the @register_patch
    # decorators run and populate `_state.PATCH_REGISTRY`.
    from vllm.sndr_core.apply import _per_patch_dispatch  # noqa: F401
    from vllm.sndr_core.apply._state import (
        PATCH_REGISTRY as APPLY_REGISTRY,
    )
    return [name for name, _fn in APPLY_REGISTRY]


# Legacy `@register_patch` names look like `"P67 TurboQuant ..."`. Extract
# the leading patch_id token so we can match them against spec patch_ids.
# Examples:
#   "P67 TurboQuant multi-query kernel"          → "P67"
#   "PN14 TQ decode IOOB safe_page_idx clamp"    → "PN14"
#   "P68/P69 long-ctx tool reminder"             → "P68"  (primary)
#   "P5b KV page-size pad-smaller-to-max"        → "P5b"
#   "G4_01 gemma4 Ampere FP8_BLOCK refusal guard" → "G4_01"
#   "G4_19b gemma4 TQ KV spec integration"       → "G4_19B"  (suffix uppercased)
# G4_NN[a-z]? prefix added 2026-05-22 (Phase 3A.1) after Phase 3 buckets
# 3/4 onboarded the G4 patch series into the legacy register table.
_PATCH_ID_LEAD = re.compile(r"^(P[Nn]?\d+[a-zA-Z]?|G4_\d+[a-zA-Z]?)\b")

# UNIFIED_CONFIG 2026-05-10 — non-P/PN style legacy registrations.
# These are sprint/middleware names registered before patch_id taxonomy
# was extended. Map them explicitly to their canonical spec patch_id.
_LEGACY_NAME_TO_PATCH_ID: dict[str, str] = {
    "Sprint 2.6 v2 — CUDA graph dispatch trace wire-in": "PN122",  # renamed from SPRINT26_CG_DISPATCH_TRACE 2026-05-14
    # SNDR_WORKSPACE_001 starts with `SNDR_`, not `P` / `PN`, so the
    # leading-token regex above can't lift the patch id. Explicit map.
    "SNDR_WORKSPACE_001 workspace grow-after-lock graceful fix": "SNDR_WORKSPACE_001",
}


def _patch_id_from_legacy_name(name: str) -> Optional[str]:
    # First check explicit map (non-P/PN style names)
    if name in _LEGACY_NAME_TO_PATCH_ID:
        return _LEGACY_NAME_TO_PATCH_ID[name]
    # Then leading P/PN/G4_ regex
    m = _PATCH_ID_LEAD.match(name)
    if not m:
        return None
    raw = m.group(1)
    # Normalize casing.
    if raw.lower().startswith("pn"):
        # PN-series: uppercase prefix, suffix letter preserved as-is.
        # Registry uses inconsistent suffix case (PN26b/PN96b lowercase
        # vs PN262B uppercase). Both forms are matched by the legacy
        # title's casing, so preserve.
        return "PN" + raw[2:]
    if raw.startswith("G4_") or raw.startswith("g4_"):
        # G4-series: prefix uppercased + suffix letter uppercased to match
        # spec registry shape (G4_19B uppercase suffix — Phase 3A.1, 2026-05-22).
        head = "G4_" + raw[3:]
        return head[:-1] + head[-1].upper() if head[-1].isalpha() else head
    # P-series default — uppercase prefix, suffix letter as-is.
    return "P" + raw[1:]


# ─── Diff result ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ApplyOrderDiff:
    """Structured diff between legacy and spec-driven apply orders."""
    legacy_count: int
    spec_count: int
    legacy_only: list[str] = field(default_factory=list)  # patch_ids in legacy not spec
    spec_only: list[str] = field(default_factory=list)    # patch_ids in spec not legacy (raw)
    spec_only_known: list[str] = field(default_factory=list)  # raw spec_only ∩ KNOWN_SPEC_ONLY
    spec_only_unexpected: list[str] = field(default_factory=list)  # raw spec_only \ KNOWN_SPEC_ONLY
    legacy_unparseable: list[str] = field(default_factory=list)  # legacy names we couldn't match to a patch_id
    spec_with_apply_module: int = 0
    spec_without_apply_module: int = 0

    @property
    def coverage_pct(self) -> float:
        """Fraction of specs whose apply_module is non-None."""
        if self.spec_count == 0:
            return 0.0
        return self.spec_with_apply_module / self.spec_count

    @property
    def is_clean(self) -> bool:
        """No UNEXPECTED mismatches and every legacy entry maps to a spec.

        P1-1 (audit 2026-05-08): "clean" no longer requires every spec
        to have apply_module — that's a separate coverage metric. It
        also no longer fails on KNOWN_SPEC_ONLY entries (intentional
        registry-driven additions). What remains:

          - no legacy_only (legacy parking lot must not have entries
            missing from registry)
          - no UNEXPECTED spec_only (new spec rows must be either in
            legacy or explicitly added to KNOWN_SPEC_ONLY_PATCHES)
          - no legacy_unparseable (every @register_patch name must
            resolve to a patch_id)
        """
        return (
            not self.legacy_only
            and not self.spec_only_unexpected
            and not self.legacy_unparseable
        )


# ─── Comparison logic ─────────────────────────────────────────────────────


def compare_apply_orders() -> ApplyOrderDiff:
    """Compute the diff between legacy and spec-driven apply orders.

    Pure function — no side effects on either registry. Safe to call
    in shadow mode during a real boot or off-line via the CLI.
    """
    from vllm.sndr_core.dispatcher.spec import iter_patch_specs

    # Legacy: list of names → set of (pid_or_None, name)
    legacy_names = _legacy_apply_names()
    legacy_pids: set[str] = set()
    legacy_unparseable: list[str] = []
    for n in legacy_names:
        pid = _patch_id_from_legacy_name(n)
        if pid is None:
            legacy_unparseable.append(n)
        else:
            legacy_pids.add(pid)

    # Spec-driven: iterate canonical specs
    specs = list(iter_patch_specs())
    spec_pids = {s.patch_id for s in specs}
    spec_with_module = sum(1 for s in specs if s.apply_module is not None)

    legacy_only = sorted(legacy_pids - spec_pids)
    spec_only = sorted(spec_pids - legacy_pids)
    # P1-1: split spec_only into known-intentional vs unexpected.
    spec_only_known = sorted(set(spec_only) & KNOWN_SPEC_ONLY_PATCHES)
    spec_only_unexpected = sorted(set(spec_only) - KNOWN_SPEC_ONLY_PATCHES)

    return ApplyOrderDiff(
        legacy_count=len(legacy_names),
        spec_count=len(specs),
        legacy_only=legacy_only,
        spec_only=spec_only,
        spec_only_known=spec_only_known,
        spec_only_unexpected=spec_only_unexpected,
        legacy_unparseable=legacy_unparseable,
        spec_with_apply_module=spec_with_module,
        spec_without_apply_module=len(specs) - spec_with_module,
    )


# ─── Human-readable report ────────────────────────────────────────────────


def format_diff(diff: ApplyOrderDiff) -> str:
    """Multi-line human-readable summary of an `ApplyOrderDiff`."""
    lines = [
        "═══════════════════════════════════════════════════════════════",
        "  Genesis apply-loop shadow report  (PR38 Day 5)",
        "═══════════════════════════════════════════════════════════════",
        f"  Legacy apply registrations:  {diff.legacy_count:>4d} "
        "(_per_patch_dispatch.py @register_patch)",
        f"  Spec-driven entries:         {diff.spec_count:>4d} "
        "(dispatcher.PATCH_REGISTRY)",
        f"  Specs with apply_module:     {diff.spec_with_apply_module:>4d}"
        f"  ({diff.coverage_pct:.0%})",
        f"  Specs without apply_module:  {diff.spec_without_apply_module:>4d}",
    ]

    if diff.legacy_only:
        lines.append("")
        lines.append(f"  ⚠ legacy_only ({len(diff.legacy_only)}) — "
                     "registered in _per_patch_dispatch.py but no "
                     "matching dispatcher.PATCH_REGISTRY entry:")
        for pid in diff.legacy_only[:20]:
            lines.append(f"      - {pid}")
        if len(diff.legacy_only) > 20:
            lines.append(f"      ... and {len(diff.legacy_only) - 20} more")

    if diff.spec_only_known:
        lines.append("")
        lines.append(
            f"  ℹ spec_only_known ({len(diff.spec_only_known)}) — "
            "intentionally registry-driven only (P1-1 KNOWN_SPEC_ONLY):"
        )
        for pid in diff.spec_only_known:
            lines.append(f"      - {pid}")
    if diff.spec_only_unexpected:
        lines.append("")
        lines.append(
            f"  ⚠ spec_only_unexpected ({len(diff.spec_only_unexpected)}) — "
            "in dispatcher.PATCH_REGISTRY, no @register_patch, NOT in "
            "KNOWN_SPEC_ONLY_PATCHES allow-list:"
        )
        for pid in diff.spec_only_unexpected[:20]:
            lines.append(f"      - {pid}")
        if len(diff.spec_only_unexpected) > 20:
            lines.append(f"      ... and "
                         f"{len(diff.spec_only_unexpected) - 20} more")

    if diff.legacy_unparseable:
        lines.append("")
        lines.append(
            f"  ⚠ legacy_unparseable ({len(diff.legacy_unparseable)}) — "
            "registered apply names whose patch_id couldn't be parsed:"
        )
        for n in diff.legacy_unparseable[:5]:
            lines.append(f"      - {n!r}")

    lines.append("")
    if diff.is_clean:
        lines.append(
            "  ✓ CLEAN — no unexpected divergence "
            f"(known spec-only: {len(diff.spec_only_known)})"
        )
    else:
        lines.append("  ⚠ DIVERGENT — see lists above")
    lines.append("═══════════════════════════════════════════════════════════════")
    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    """`python -m vllm.sndr_core.apply.shadow` entry point."""
    parser = argparse.ArgumentParser(
        description="Shadow comparison: PatchSpec apply order vs legacy"
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="exit non-zero if any divergence found",
    )
    args = parser.parse_args(argv)

    diff = compare_apply_orders()
    print(format_diff(diff))
    if args.strict and not diff.is_clean:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
