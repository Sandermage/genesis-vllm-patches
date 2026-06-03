# SPDX-License-Identifier: Apache-2.0
"""CI-wide regression guard — no DRIFT entries in the legacy↔spec
divergence audit (v12.0.0 readiness gate).

Why this matters
----------------

The dispatcher migration plan (master plan §3 Phase 4, §13 follow-up #7)
targets v12.0.0 to flip the default apply path from legacy
`@register_patch(...)` iteration to spec-driven
`iter_patch_specs()` dispatch. For that flip to be safe, every patch
the legacy path applies must be either:

  (a) ALSO applied by spec-driven path (covered by an iter-yielded
      spec with apply_module set), OR
  (b) DELIBERATELY skipped by spec-driven path via the
      `GENESIS_LEGACY_*` env_flag policy (spec entry exists with
      apply_module=None and env_flag starts with GENESIS_LEGACY_).

Any legacy-only patch that doesn't fit (b) is `DRIFT` — a real
migration gap that would silently disable the patch on v12.0.0 flip.

v11.3.0 baseline (BUG #6 audit, this commit): 0 drift entries,
7 intentional legacy (P1, P17, P18b, P20, P23, P29, P32). The
audit script's `_diff_matrices` classifies them; this test pins
`drift_count == 0`.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Status: v11.3.0+ regression guard.
"""
from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _import_audit_module():
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    import scripts.audit_legacy_vs_spec_driven_apply_matrix as A
    return A


def test_legacy_only_drift_zero():
    """v12.0.0 readiness invariant: no legacy-registered patch lacks
    EITHER a matching spec apply_module OR a policy-tagged spec
    informational entry (GENESIS_LEGACY_* env)."""
    audit = _import_audit_module()
    from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
    legacy = audit._enumerate_legacy_path()
    spec = audit._enumerate_spec_driven_path(PATCH_REGISTRY)
    diff = audit._diff_matrices(legacy, spec, PATCH_REGISTRY)
    assert diff["legacy_only_drift_count"] == 0, (
        f"v12.0.0 readiness regression: {diff['legacy_only_drift_count']} "
        f"legacy-only entries lack a spec policy entry. On v12.0.0 flip "
        f"these patches would silently stop applying. "
        f"Drift IDs: {diff['legacy_only_drift_ids']}. "
        f"Fix by EITHER adding apply_module to the spec entry (real "
        f"migration) OR marking the spec entry with env_flag prefix "
        f"GENESIS_LEGACY_<ID> + apply_module=None + lifecycle=legacy "
        f"(intentional policy deferral)."
    )


def test_v12_0_safe_flag_true():
    """Pin v12_0_safe (drift-only flag) at True. order_divergent is
    tracked separately via v12_0_strict_order (False at v11.3.0
    baseline — see scripts/audit_legacy_vs_spec_driven_apply_matrix.py
    for the dict-insertion-order vs decorator-call-order analysis).
    """
    audit = _import_audit_module()
    from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
    legacy = audit._enumerate_legacy_path()
    spec = audit._enumerate_spec_driven_path(PATCH_REGISTRY)
    diff = audit._diff_matrices(legacy, spec, PATCH_REGISTRY)
    assert diff["v12_0_safe"] is True, (
        f"v12_0_safe became False — drift introduced. "
        f"legacy_only_drift_count={diff['legacy_only_drift_count']}"
    )


def test_intentional_legacy_baseline():
    """Document the v11.3.0 intentional legacy baseline. Any NEW entry
    here means a NEW patch was registered legacy-only — confirm it's
    intentional and update this baseline."""
    audit = _import_audit_module()
    from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
    legacy = audit._enumerate_legacy_path()
    spec = audit._enumerate_spec_driven_path(PATCH_REGISTRY)
    diff = audit._diff_matrices(legacy, spec, PATCH_REGISTRY)
    # Baseline expected: 7 intentional legacy entries
    # (P1, P17, P18b, P20, P23, P29, P32)
    intentional = set(diff["legacy_only_intentional_ids"])
    expected = {"P1", "P17", "P18b", "P20", "P23", "P29", "P32"}
    assert intentional == expected, (
        f"Intentional legacy set changed:\n"
        f"  added: {sorted(intentional - expected)}\n"
        f"  removed: {sorted(expected - intentional)}\n"
        f"If intentional: update `expected` set in this test.\n"
        f"If a removal: that patch was migrated (good — verify "
        f"apply_module set + lifecycle != 'legacy')."
    )


def test_no_apply_module_entry_must_have_legacy_env_or_known_lifecycle():
    """Policy invariant: a spec entry with apply_module=None and
    lifecycle='legacy' MUST use the GENESIS_LEGACY_* env_flag
    convention. This prevents an operator from setting
    `GENESIS_ENABLE_<X>=1` and expecting the patch to apply in
    spec-driven mode (it never will — the patch is legacy-path-only).

    Exemptions:
    - implementation_status='marker_only' / 'placeholder': advisory or
      preflight-doctor entries that don't apply via either path (e.g.
      PN60 quant arg validator, PN63 fp8_e5m2 advisory — they live in
      CLI preflight, not the apply loop).
    - implementation_status='scaffold': in-progress wiring.
    - lifecycle ∈ {retired, research, coordinator}: each has its own
      semantics for apply_module=None handling.
    """
    from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
    EXEMPT_STATUS = {"marker_only", "placeholder", "scaffold"}
    violations: list[tuple[str, str, str]] = []
    for pid, meta in PATCH_REGISTRY.items():
        if not isinstance(meta, dict):
            continue
        if meta.get("apply_module") is not None:
            continue
        if meta.get("lifecycle") != "legacy":
            continue
        if meta.get("implementation_status") in EXEMPT_STATUS:
            continue
        env_flag = str(meta.get("env_flag", ""))
        if not env_flag.startswith("GENESIS_LEGACY_"):
            violations.append((pid, env_flag, str(meta.get("implementation_status", "-"))))
    assert not violations, (
        f"{len(violations)} spec entries with lifecycle='legacy' + "
        f"apply_module=None use a `GENESIS_ENABLE_*` env_flag instead "
        f"of the policy `GENESIS_LEGACY_*` prefix. An operator setting "
        f"that env flag would NOT enable the patch in spec-driven mode "
        f"(it always skips) — they'd get a silent no-op.\n\n"
        f"Either:\n"
        f"  (a) rename env_flag to GENESIS_LEGACY_<ID> (intentional "
        f"deferral — patch only applies via legacy path), OR\n"
        f"  (b) add an apply_module + change lifecycle to "
        f"`experimental`/`stable` (real migration), OR\n"
        f"  (c) set implementation_status='marker_only' if it's a "
        f"doctor/preflight advisory not a real apply patch.\n\n"
        f"Offenders:\n" + "\n".join(
            f"  - {pid}: env_flag={ef!r} status={st}"
            for pid, ef, st in violations
        )
    )
