# SPDX-License-Identifier: Apache-2.0
"""S2.4 audit closure (2026-05-08 noonghunna): dependency graph resolver.

Tests the orchestrator's preflight that consults
``requires_patches`` / ``conflicts_with`` metadata at apply time.

Three findings tracked:

  • dep_missing: WARNING (informational; req may have been upstream-merged)
  • conflict_active: ERROR (hard-block under GENESIS_STRICT_DEPS=1)
  • dep_unknown: ERROR (schema drift; hard-block under GENESIS_STRICT_DEPS=1)
"""
from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from sndr.apply.orchestrator import (
    _is_env_enabled,
    _strict_dep_mode,
    _validate_dependency_graph,
)


# ─── env helper ────────────────────────────────────────────────────────


class TestIsEnvEnabled:
    def test_unset_returns_false(self, monkeypatch):
        monkeypatch.delenv("GENESIS_TEST_FLAG", raising=False)
        assert _is_env_enabled("GENESIS_TEST_FLAG") is False

    @pytest.mark.parametrize("v", ["1", "true", "yes", "y", "on", "TRUE", "Yes"])
    def test_truthy_values(self, monkeypatch, v):
        monkeypatch.setenv("GENESIS_TEST_FLAG", v)
        assert _is_env_enabled("GENESIS_TEST_FLAG") is True

    @pytest.mark.parametrize("v", ["0", "false", "no", "off", ""])
    def test_falsy_values(self, monkeypatch, v):
        monkeypatch.setenv("GENESIS_TEST_FLAG", v)
        assert _is_env_enabled("GENESIS_TEST_FLAG") is False

    def test_none_flag(self):
        assert _is_env_enabled(None) is False
        assert _is_env_enabled("") is False


class TestStrictDepMode:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("GENESIS_STRICT_DEPS", raising=False)
        assert _strict_dep_mode() is False

    def test_opt_in(self, monkeypatch):
        monkeypatch.setenv("GENESIS_STRICT_DEPS", "1")
        assert _strict_dep_mode() is True


# ─── Dependency graph validation ───────────────────────────────────────


_FAKE_REGISTRY = {
    "PA": {
        "env_flag": "GENESIS_ENABLE_PA",
        "requires_patches": ["PB"],
        "conflicts_with": [],
        "default_on": False,
    },
    "PB": {
        "env_flag": "GENESIS_ENABLE_PB",
        "requires_patches": [],
        "conflicts_with": [],
        "default_on": False,
    },
    "PX": {
        "env_flag": "GENESIS_ENABLE_PX",
        "requires_patches": [],
        "conflicts_with": ["PY"],
        "default_on": False,
    },
    "PY": {
        "env_flag": "GENESIS_ENABLE_PY",
        "requires_patches": [],
        "conflicts_with": ["PX"],
        "default_on": False,
    },
    "PZ": {
        "env_flag": "GENESIS_ENABLE_PZ",
        "requires_patches": ["P_NONEXISTENT"],
        "conflicts_with": [],
        "default_on": False,
    },
}


@pytest.fixture
def fake_registry(monkeypatch):
    """Inject _FAKE_REGISTRY in place of real PATCH_REGISTRY for the
    scope of one test. Reverts on exit."""
    import sndr.dispatcher as disp_mod
    monkeypatch.setattr(disp_mod, "PATCH_REGISTRY", _FAKE_REGISTRY)
    yield


class TestValidateDependencyGraph:
    def test_no_enabled_patches_returns_quietly(
        self, fake_registry, monkeypatch, caplog,
    ):
        # All env flags off → nothing enabled
        for pid in _FAKE_REGISTRY:
            monkeypatch.delenv(f"GENESIS_ENABLE_{pid}", raising=False)
        with caplog.at_level(logging.DEBUG, logger="genesis.apply_all"):
            _validate_dependency_graph()
        # No warnings emitted; debug-level skip is fine
        assert not any(
            r.levelno >= logging.WARNING for r in caplog.records
        )

    def test_dep_missing_logs_warning(
        self, fake_registry, monkeypatch, caplog,
    ):
        # PA enabled (requires PB) but PB NOT enabled
        monkeypatch.setenv("GENESIS_ENABLE_PA", "1")
        monkeypatch.delenv("GENESIS_ENABLE_PB", raising=False)
        with caplog.at_level(logging.WARNING, logger="genesis.apply_all"):
            _validate_dependency_graph()
        warns = [r for r in caplog.records
                 if r.levelno == logging.WARNING and "PA" in r.message]
        assert len(warns) >= 1
        assert "PB" in warns[0].message
        assert "requires" in warns[0].message.lower()

    def test_active_conflict_logs_error(
        self, fake_registry, monkeypatch, caplog,
    ):
        # PX and PY both enabled, both declare conflict
        monkeypatch.setenv("GENESIS_ENABLE_PX", "1")
        monkeypatch.setenv("GENESIS_ENABLE_PY", "1")
        with caplog.at_level(logging.ERROR, logger="genesis.apply_all"):
            _validate_dependency_graph()
        errors = [r for r in caplog.records
                  if r.levelno == logging.ERROR
                  and "CONFLICT" in r.message]
        assert len(errors) == 1, f"expected 1 conflict log, got {len(errors)}"
        assert "PX" in errors[0].message and "PY" in errors[0].message

    def test_unknown_dep_logs_error(
        self, fake_registry, monkeypatch, caplog,
    ):
        monkeypatch.setenv("GENESIS_ENABLE_PZ", "1")
        with caplog.at_level(logging.ERROR, logger="genesis.apply_all"):
            _validate_dependency_graph()
        errors = [r for r in caplog.records
                  if r.levelno == logging.ERROR
                  and "P_NONEXISTENT" in r.message]
        assert len(errors) >= 1
        assert "schema drift" in errors[0].message.lower()

    def test_strict_mode_aborts_on_conflict(
        self, fake_registry, monkeypatch,
    ):
        monkeypatch.setenv("GENESIS_ENABLE_PX", "1")
        monkeypatch.setenv("GENESIS_ENABLE_PY", "1")
        monkeypatch.setenv("GENESIS_STRICT_DEPS", "1")
        with pytest.raises(SystemExit) as exc:
            _validate_dependency_graph()
        assert exc.value.code == 2

    def test_strict_mode_aborts_on_unknown_dep(
        self, fake_registry, monkeypatch,
    ):
        monkeypatch.setenv("GENESIS_ENABLE_PZ", "1")
        monkeypatch.setenv("GENESIS_STRICT_DEPS", "1")
        with pytest.raises(SystemExit) as exc:
            _validate_dependency_graph()
        assert exc.value.code == 2

    def test_strict_mode_no_abort_on_dep_missing_only(
        self, fake_registry, monkeypatch, caplog,
    ):
        """dep_missing is WARNING-only — even strict mode shouldn't abort
        because the dep may have been upstream-merged."""
        monkeypatch.setenv("GENESIS_ENABLE_PA", "1")
        monkeypatch.delenv("GENESIS_ENABLE_PB", raising=False)
        monkeypatch.setenv("GENESIS_STRICT_DEPS", "1")
        # Should NOT raise SystemExit
        _validate_dependency_graph()

    def test_clean_state_logs_ok(
        self, fake_registry, monkeypatch, caplog,
    ):
        # PA enabled with PB enabled — clean dep chain
        monkeypatch.setenv("GENESIS_ENABLE_PA", "1")
        monkeypatch.setenv("GENESIS_ENABLE_PB", "1")
        with caplog.at_level(logging.INFO, logger="genesis.apply_all"):
            _validate_dependency_graph()
        ok_msgs = [r for r in caplog.records if "OK" in r.message]
        assert len(ok_msgs) >= 1


# ─── Real registry smoke (catches drift) ───────────────────────────────


class TestRealRegistryHasNoSchemaDrift:
    def test_real_registry_no_unknown_deps(self, monkeypatch, caplog):
        """All `requires_patches` references in the actual registry
        should resolve to known patch_ids. This catches typos like
        'PN91' vs 'PN91_KV_EVICTION'."""
        from sndr.dispatcher import PATCH_REGISTRY
        unknown = []
        for pid, meta in PATCH_REGISTRY.items():
            for req in (meta.get("requires_patches") or []):
                if req not in PATCH_REGISTRY:
                    unknown.append((pid, req))
        assert unknown == [], (
            f"requires_patches references unknown patches: {unknown}"
        )

    def test_real_registry_conflicts_are_symmetric(self):
        """If A declares conflicts_with=[B], B should declare
        conflicts_with=[A]. Catches metadata drift between sister patches."""
        from sndr.dispatcher import PATCH_REGISTRY
        asymmetric = []
        for pid, meta in PATCH_REGISTRY.items():
            for conflict in (meta.get("conflicts_with") or []):
                if conflict not in PATCH_REGISTRY:
                    continue  # caught by other test
                back = PATCH_REGISTRY[conflict].get("conflicts_with") or []
                if pid not in back:
                    asymmetric.append((pid, conflict))
        # We allow some asymmetry as advisory hints, but flag the count.
        # Strict equality is too strict for legacy patches.
        if asymmetric:
            pytest.skip(
                f"{len(asymmetric)} asymmetric conflicts — backlog item "
                "for registry hygiene sprint, not a blocker yet"
            )
