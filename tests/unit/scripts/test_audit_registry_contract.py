# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/audit_registry_contract.py`` — focused on
invariant 8 (``docstring_lifecycle``) — TOOLING-HARDENING.2 L.2
(2026-05-26).

Invariant 8 catches the failure mode that motivated L.2: a patch
module's docstring declares itself retired (e.g. ``TOMBSTONED — fla
recurrent kernel cannot serve single-seq prefill``) while the registry
entry still says ``lifecycle: experimental``. That drift made PN108
silently apply on unsuspecting checkpoints until the 2026-05-14 manual
sync caught it.

Coverage strategy: synthesize ``types.ModuleType`` stubs in
``sys.modules`` keyed at importable ``sndr.engines.vllm.patches.*``
paths, hand the checker a synthetic registry dict, and assert each
distinct case fires (or stays silent).

The live smoke test at the bottom asserts the real registry + tree
produces zero invariant-8 issues, pinning the current clean state.

Other invariants (1–7) have working live coverage via the script's own
self-test and ``make gates``; this file deliberately scopes to L.2's
docstring-lifecycle gap and does not duplicate broader coverage.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_registry_contract.py"


def _import_audit_module():
    """Import ``audit_registry_contract.py`` as a module so we can call
    its private ``_check_docstring_lifecycle_sync`` directly."""
    spec = importlib.util.spec_from_file_location(
        "audit_registry_contract", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audit_registry_contract"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def audit_mod():
    return _import_audit_module()


@pytest.fixture
def fake_module():
    """Factory that injects a synthetic module into ``sys.modules`` with
    a custom ``__doc__``. The path uses ``sndr.engines.vllm.patches.``
    so the checker's ``".patches." in am`` gate accepts it.

    Caller is responsible for picking a path that does not collide with
    a real module. The fixture cleans up after the test.
    """
    created: list[str] = []

    def _make(name: str, docstring: str | None) -> str:
        assert ".patches." in name, (
            "synthetic module path must contain '.patches.' to "
            "exercise the checker's gate"
        )
        mod = types.ModuleType(name)
        mod.__doc__ = docstring
        sys.modules[name] = mod
        created.append(name)
        return name

    yield _make

    for name in created:
        sys.modules.pop(name, None)


class TestDocstringLifecycleSync:
    """Invariant 8 — docstring TOMBSTONED/RETIRED ↔ registry lifecycle."""

    def test_no_violations_on_clean_registry(self, audit_mod, fake_module):
        """Docstring has no lifecycle marker → zero issues regardless of
        registry lifecycle."""
        am = fake_module(
            "sndr.engines.vllm.patches._test_fake.clean_module",
            "Clean experimental patch. No retirement markers anywhere.",
        )
        registry = {
            "FAKE_CLEAN": {
                "lifecycle": "experimental",
                "apply_module": am,
            },
        }
        issues = audit_mod._check_docstring_lifecycle_sync(registry)
        assert issues == [], (
            f"clean docstring + experimental registry should produce "
            f"zero issues, got {issues}"
        )

    def test_tombstoned_marker_with_experimental_lifecycle_flagged(
        self, audit_mod, fake_module,
    ):
        """The exact PN108 scenario: TOMBSTONED in docstring, registry
        lifecycle=experimental → drift detected."""
        am = fake_module(
            "sndr.engines.vllm.patches._test_fake.tombstoned_module",
            "TOMBSTONED — fla recurrent kernel cannot serve "
            "single-seq prefill. This module is a harmless no-op now.",
        )
        registry = {
            "FAKE_PN108": {
                "lifecycle": "experimental",
                "apply_module": am,
            },
        }
        issues = audit_mod._check_docstring_lifecycle_sync(registry)
        assert len(issues) == 1, (
            f"TOMBSTONED docstring + experimental lifecycle should "
            f"produce exactly one issue, got {issues}"
        )
        assert "FAKE_PN108" in issues[0]
        assert "experimental" in issues[0]

    def test_tombstoned_marker_with_retired_lifecycle_silent(
        self, audit_mod, fake_module,
    ):
        """Registry lifecycle=retired short-circuits the check — the
        invariant assumes retired patches may legitimately retain
        retirement banners."""
        am = fake_module(
            "sndr.engines.vllm.patches._test_fake.retired_module",
            "TOMBSTONED — retired 2026-05-14, replaced by PN59.",
        )
        registry = {
            "FAKE_RETIRED": {
                "lifecycle": "retired",
                "apply_module": am,
            },
        }
        issues = audit_mod._check_docstring_lifecycle_sync(registry)
        assert issues == [], (
            f"retired lifecycle should short-circuit the check, "
            f"got {issues}"
        )

    def test_status_retired_phrase_flagged_when_registry_disagrees(
        self, audit_mod, fake_module,
    ):
        """``status: retired`` in docstring + registry lifecycle=stable
        → drift detected."""
        am = fake_module(
            "sndr.engines.vllm.patches._test_fake.status_retired_module",
            "Some patch.\n\nStatus: retired (upstream landed in vllm#41234).",
        )
        registry = {
            "FAKE_STATUS": {
                "lifecycle": "stable",
                "apply_module": am,
            },
        }
        issues = audit_mod._check_docstring_lifecycle_sync(registry)
        assert len(issues) == 1, (
            f"status: retired + stable lifecycle should produce "
            f"exactly one issue, got {issues}"
        )

    def test_mentioning_other_retired_patches_not_flagged(
        self, audit_mod, fake_module,
    ):
        """False-positive avoidance: the patch's docstring mentions
        OTHER patches' retirement, not its own. The invariant must NOT
        fire on prose like ``replaces retired P7`` or ``self-retired
        when upstream lands``."""
        am = fake_module(
            "sndr.engines.vllm.patches._test_fake.references_retired",
            "Active patch. Replaces retired P7 and supersedes PN13 "
            "(which is itself retired). When upstream PR42637 lands "
            "this patch will be self-retired.",
        )
        registry = {
            "FAKE_REFS": {
                "lifecycle": "experimental",
                "apply_module": am,
            },
        }
        issues = audit_mod._check_docstring_lifecycle_sync(registry)
        assert issues == [], (
            f"docstring mentioning OTHER retired patches must not "
            f"trigger invariant 8, got {issues}"
        )

    def test_missing_apply_module_skipped(self, audit_mod):
        """Registry entry without ``apply_module`` is skipped (covered by
        invariant 4 ``apply_module``); invariant 8 must not crash."""
        registry = {
            "FAKE_NO_MODULE": {"lifecycle": "experimental"},
        }
        issues = audit_mod._check_docstring_lifecycle_sync(registry)
        assert issues == []

    def test_unimportable_apply_module_skipped(self, audit_mod):
        """When ``apply_module`` cannot import, invariant 8 defers to
        invariant 4 (apply-module coverage) and reports nothing — it
        must not falsely accuse a module it cannot read."""
        registry = {
            "FAKE_UNIMPORTABLE": {
                "lifecycle": "experimental",
                "apply_module": (
                    "sndr.engines.vllm.patches._does_not_exist"
                    ".missing_module"
                ),
            },
        }
        issues = audit_mod._check_docstring_lifecycle_sync(registry)
        assert issues == []

    def test_non_integrations_apply_module_skipped(
        self, audit_mod, fake_module,
    ):
        """``apply_module`` not under the engine patches tree (e.g.
        helper module, future relocation) is skipped — the audit scope
        is integration patches only."""
        # Cannot use fake_module fixture (it asserts .patches. is in
        # the path); build manually for this skip case.
        name = "sndr._test_fake.helper_module"
        mod = types.ModuleType(name)
        mod.__doc__ = "TOMBSTONED helper, not an integration patch."
        sys.modules[name] = mod
        try:
            registry = {
                "FAKE_HELPER": {
                    "lifecycle": "experimental",
                    "apply_module": name,
                },
            }
            issues = audit_mod._check_docstring_lifecycle_sync(registry)
            assert issues == [], (
                f"non-integration apply_module should be skipped, "
                f"got {issues}"
            )
        finally:
            sys.modules.pop(name, None)


class TestLiveCorpus:
    """Run invariant 8 against the real registry tree (smoke)."""

    def test_live_invariant_8_zero_issues(self, audit_mod):
        from sndr.dispatcher.registry import PATCH_REGISTRY
        issues = audit_mod._check_docstring_lifecycle_sync(PATCH_REGISTRY)
        assert issues == [], (
            "live registry must have zero docstring-lifecycle drift "
            f"({len(issues)} issue(s) found):\n  - "
            + "\n  - ".join(issues)
        )

    def test_live_script_exit_zero(self):
        """Whole ``audit_registry_contract.py`` (all 8 invariants) must
        be green on the tracked tree."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert result.returncode == 0, (
            f"audit_registry_contract should pass on tracked tree, "
            f"got rc={result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
