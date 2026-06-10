# SPDX-License-Identifier: Apache-2.0
"""Dispatcher / apply baseline snapshot tests — M.1.1.T1.A (2026-05-27).

Locks the observable surface of ``sndr.dispatcher`` and
``sndr.apply._state`` against unintentional drift during
the M.1 dispatcher tier-matrix refactor queue (M.1.1.T1.B helper
splits, M.1.1.T2 audit-rule extraction, M.1.1.T3 retired-stub
collapse).

Pure-additive: no runtime code changes; no registry or apply-module
edits. The fixtures in ``tests/unit/dispatcher/fixtures/`` are the
contract. When you intentionally add or modify a patch (new
``patch_id``, env flag rename, ``apply_module`` move), regenerate
via:

    SNDR_SNAPSHOT_REGEN=1 python3 -m pytest \
        tests/unit/dispatcher/test_baseline_snapshots.py -q

then review the JSON diff in git before committing.

Surfaces protected
──────────────────
  apply_registry.json          ``apply._state.PATCH_REGISTRY`` —
                               ordered ``(name, wrapped_name)`` pairs.
  spec_set.json                ``iter_patch_specs()`` — stable subset
                               of fields per spec.
  apply_module_coverage.json   ``validate_apply_module_coverage()`` —
                               total / mapped / unmapped lists.
  decision_no_env.json         ``should_apply()`` swept across every
                               registry entry with all canonical
                               env-flag prefixes unset; locks the
                               operator-visible reason wording.

See ``tests/unit/dispatcher/fixtures/README.md`` for the regen workflow
+ what each fixture protects.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest


_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_REGEN_ENV = "SNDR_SNAPSHOT_REGEN"


# ─── Helpers ──────────────────────────────────────────────────────────────


def _regen_enabled() -> bool:
    return os.environ.get(_REGEN_ENV, "").strip() in ("1", "true", "yes", "on")


def _read_fixture(name: str) -> Any:
    path = _FIXTURES_DIR / name
    if not path.is_file():
        pytest.fail(
            f"fixture {path.name} missing — run "
            f"`{_REGEN_ENV}=1 pytest {Path(__file__).name}` to seed it."
        )
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_fixture(name: str, data: Any) -> None:
    path = _FIXTURES_DIR / name
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _assert_or_regen(fixture_name: str, live: Any) -> None:
    """Compare ``live`` to the fixture; under ``SNDR_SNAPSHOT_REGEN=1``
    overwrite the fixture instead.

    The diff that pytest renders on assertion failure is the actionable
    output — a future refactor that drifts on any of these surfaces
    will see the exact JSON entries that changed.
    """
    if _regen_enabled():
        _write_fixture(fixture_name, live)
        return
    expected = _read_fixture(fixture_name)
    assert live == expected, (
        f"snapshot drift in {fixture_name} — either fix the refactor, "
        f"or if the drift is intentional regen via "
        f"`{_REGEN_ENV}=1 pytest {Path(__file__).name}`."
    )


# ─── Surface 1: apply._state.PATCH_REGISTRY ───────────────────────────────


def _capture_apply_registry() -> list[dict[str, str]]:
    """Capture ``(name, wrapped_name)`` per entry in insertion order.

    ``wrapped_name`` reads through ``__wrapped__`` when ``register_patch``
    instrumented the callable (observability wrap); otherwise falls back
    to the callable's own ``__name__``. Both are part of the boot-time
    contract because they appear in observability labels + log lines.
    """
    from sndr.apply._state import PATCH_REGISTRY

    out: list[dict[str, str]] = []
    for name, fn in PATCH_REGISTRY:
        wrapped = getattr(fn, "__wrapped__", None)
        wrapped_name = (
            wrapped.__name__ if wrapped is not None and hasattr(wrapped, "__name__")
            else getattr(fn, "__name__", "")
        )
        out.append({
            "name": name,
            "wrapped_name": wrapped_name,
        })
    return out


class TestApplyRegistrySnapshot:
    """``apply._state.PATCH_REGISTRY`` insertion order + names — protects
    Tier 3 retired-stub collapse from drifting boot-log labels."""

    FIXTURE = "apply_registry.json"

    def test_count_matches_snapshot(self):
        if _regen_enabled():
            pytest.skip("regen mode — count check skipped (fixture rewritten below)")
        expected = _read_fixture(self.FIXTURE)
        from sndr.apply._state import PATCH_REGISTRY
        assert len(PATCH_REGISTRY) == len(expected), (
            f"apply registry count drift: live={len(PATCH_REGISTRY)}, "
            f"snapshot={len(expected)}"
        )

    def test_full_snapshot_match(self):
        _assert_or_regen(self.FIXTURE, _capture_apply_registry())


# ─── Surface 2: dispatcher.iter_patch_specs() ─────────────────────────────


def _capture_spec_set() -> list[dict[str, Any]]:
    """Capture the stable subset of PatchSpec fields, in iter order.

    Fields chosen on the basis of "refactor-shouldn't-touch-this":

      patch_id          identity — every other check keyed by this
      family            apply_module path is derived from this
      lifecycle         operational-role enum; M.5/M.1.T2 audit gate
      tier              community / engine wheel separation
      default_on        operator contract for boot behaviour
      env_flag          canonical env name (prefix audit covers this)
      has_upstream_pr   bool — counter-regression / drift triage
      upstream_pr_relationship  backport / counter-regression / etc.
      has_apply_module  bool — coverage report consumes this
      apply_module      dotted path (or null for intentionally unmapped)

    Title is intentionally excluded — title text drift is editorial
    and not a refactor-safety concern.
    """
    from sndr.dispatcher import iter_patch_specs

    out: list[dict[str, Any]] = []
    for spec in iter_patch_specs():
        out.append({
            "patch_id": spec.patch_id,
            "family": spec.family,
            "lifecycle": spec.lifecycle,
            "tier": spec.tier,
            "default_on": bool(spec.default_on),
            "env_flag": spec.env_flag,
            "has_upstream_pr": spec.upstream_pr is not None,
            "upstream_pr_relationship": spec.upstream_pr_relationship,
            "has_apply_module": spec.apply_module is not None,
            "apply_module": spec.apply_module,
        })
    return out


class TestSpecSetSnapshot:
    """``iter_patch_specs()`` stable-field set — protects Tier 1.B / T2
    refactors from drifting per-spec metadata."""

    FIXTURE = "spec_set.json"

    def test_count_matches_snapshot(self):
        if _regen_enabled():
            pytest.skip("regen mode")
        expected = _read_fixture(self.FIXTURE)
        from sndr.dispatcher import iter_patch_specs
        assert sum(1 for _ in iter_patch_specs()) == len(expected)

    def test_full_snapshot_match(self):
        _assert_or_regen(self.FIXTURE, _capture_spec_set())


# ─── Surface 3: validate_apply_module_coverage() ──────────────────────────


def _capture_apply_module_coverage() -> dict[str, Any]:
    """Capture coverage summary in a comparable, sorted form."""
    from sndr.dispatcher.spec import validate_apply_module_coverage

    coverage = validate_apply_module_coverage()
    return {
        "total": coverage.total,
        "mapped": coverage.mapped,
        "unmapped_count": len(coverage.unmapped),
        "unmapped": sorted(coverage.unmapped),
        "intentionally_unmapped_count": len(coverage.intentionally_unmapped),
        "intentionally_unmapped": sorted(coverage.intentionally_unmapped),
    }


class TestApplyModuleCoverageSnapshot:
    """``validate_apply_module_coverage()`` — protects from accidental
    apply_module relocation."""

    FIXTURE = "apply_module_coverage.json"

    def test_full_snapshot_match(self):
        _assert_or_regen(self.FIXTURE, _capture_apply_module_coverage())


# ─── Surface 4: should_apply() sweep with no canonical env vars ───────────


# Mirror of ``dispatcher._constants._CANONICAL_ENV_PREFIXES``; kept
# duplicated here so a test that's supposed to GUARD the constants can't
# accidentally short-circuit via the same import the constants live in.
# If a new prefix is added to ``_CANONICAL_ENV_PREFIXES`` and not added
# here, the env-clear setup leaks state and the decision snapshot drifts
# — which is the desired signal.
_GUARD_ENV_PREFIXES = (
    "SNDR_ENABLE_", "GENESIS_ENABLE_",
    "SNDR_DISABLE_", "GENESIS_DISABLE_",
    "SNDR_LEGACY_", "GENESIS_LEGACY_",
    "SNDR_ALLOW_", "GENESIS_ALLOW_",
)


def _clear_canonical_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delete every env var beginning with a canonical SNDR/GENESIS
    prefix so ``should_apply`` takes the default (no-operator-override)
    branch for every registry entry."""
    for key in list(os.environ.keys()):
        if any(key.startswith(p) for p in _GUARD_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)


def _capture_decisions_no_env() -> list[dict[str, Any]]:
    """Sweep ``should_apply`` for every PATCH_REGISTRY entry under a
    clean canonical-env environment. Output is the per-patch
    ``(applied, reason)`` tuple in iter order.

    Reason strings are operator-visible in boot logs; any T1.B
    helper-split refactor must produce byte-identical text for the
    no-env baseline to stay green.
    """
    from sndr.dispatcher import should_apply
    from sndr.dispatcher.registry import PATCH_REGISTRY

    out: list[dict[str, Any]] = []
    for pid in PATCH_REGISTRY:
        applied, reason = should_apply(pid)
        out.append({
            "patch_id": pid,
            "applied": bool(applied),
            "reason": reason,
        })
    return out


class TestDecisionNoEnvSnapshot:
    """``should_apply()`` no-env decisions — protects Tier 1.B
    decomposition from drifting reason strings (operator-visible)."""

    FIXTURE = "decision_no_env.json"

    def test_full_snapshot_match(self, monkeypatch):
        _clear_canonical_env(monkeypatch)
        _assert_or_regen(self.FIXTURE, _capture_decisions_no_env())
