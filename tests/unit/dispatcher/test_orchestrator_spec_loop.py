# SPDX-License-Identifier: Apache-2.0
"""TDD for `_run_via_specs` (PR38 Day 6-8).

Spec-driven apply loop activated by `SNDR_APPLY_VIA_SPECS=1`. Iterates
`dispatcher.iter_patch_specs()` and calls `module.apply()` directly on
each spec's `apply_module`. Replaces the hand-written
`_per_patch_dispatch.apply_patch_X` parking lot.
"""
from __future__ import annotations

import pytest


def _orch():
    from vllm.sndr_core.apply import orchestrator
    return orchestrator


# ─── Spec-loop function exists + is wired ─────────────────────────────────


class TestSpecLoopWiring:
    def test_run_via_specs_function_exists(self):
        o = _orch()
        assert callable(o._run_via_specs), (
            "_run_via_specs must exist on orchestrator"
        )

    def test_run_via_specs_uses_iter_patch_specs(self):
        """The spec-loop must drive its iteration from
        `dispatcher.iter_patch_specs()` — not from `_state.PATCH_REGISTRY`
        (the legacy registration list)."""
        import inspect
        o = _orch()
        src = inspect.getsource(o._run_via_specs)
        assert "iter_patch_specs" in src


# ─── End-to-end dry-run via spec loop ─────────────────────────────────────


class TestSpecLoopDryRun:
    def test_spec_loop_runs_clean_in_dry_run(self, monkeypatch):
        """With `SNDR_APPLY_VIA_SPECS=1` and dry-run mode, the
        orchestrator must classify every registry entry into exactly
        one of {applied, skipped, failed} without raising — and must
        report zero failures (no wiring imports broken).

        Phase 5.3.A (2026-05-22): refreshed from the pre-2026-05-17
        auto-apply expectation. Under the strict-opt-in policy
        (decision.py:305-360), with a clean test env (no per-patch
        `GENESIS_ENABLE_<X>=1` flags set), every `default_on=True`
        patch routes to 'skipped' rather than 'applied'. The previous
        assertion `applied >= 5` therefore no longer holds — it
        encoded the auto-apply semantics now reserved for the
        `GENESIS_LEGACY_DEFAULT_ON=1` escape hatch.

        Structural contract (what matters regardless of policy era):
          - applied + skipped + failed == total (every entry classified)
          - failed == 0 (no broken wiring imports / structural bugs)
          - applied ⊆ entries that satisfied dispatcher decision

        The 'no failures' guarantee is the load-bearing one: a torch-
        less laptop run must not crash inside any apply_module. That
        was the P0-2 audit's original promise and survives the policy
        flip.
        """
        monkeypatch.setenv("SNDR_APPLY_VIA_SPECS", "1")
        # Ensure clean strict-opt-in env — no LEGACY_DEFAULT_ON
        # rescue, no per-patch ENABLE flags leaking from the parent
        # shell. The test verifies the modern policy's classification
        # invariants.
        monkeypatch.delenv("GENESIS_LEGACY_DEFAULT_ON", raising=False)
        monkeypatch.delenv("SNDR_LEGACY_DEFAULT_ON", raising=False)

        o = _orch()
        stats = o.run(verbose=False, apply=False)
        applied = [r for r in stats.results if r.status == "applied"]
        skipped = [r for r in stats.results if r.status == "skipped"]
        failed = [r for r in stats.results if r.status == "failed"]
        total_classified = len(applied) + len(skipped) + len(failed)

        # Every entry must land in exactly one bucket — no unclassified
        # / leaked statuses.
        assert total_classified == len(stats.results), (
            f"classification gap: applied={len(applied)} + "
            f"skipped={len(skipped)} + failed={len(failed)} != "
            f"total={len(stats.results)}"
        )
        # No broken wiring imports — the load-bearing P0-2 guarantee.
        assert len(failed) == 0, (
            f"spec-loop dry-run produced failures: "
            f"{[(r.name, r.reason) for r in failed[:5]]}"
        )
        # Under strict-opt-in clean env, the bulk of entries route to
        # 'skipped' (informational metadata-only entries + default_off
        # opt-ins + default_on awaiting explicit ENABLE).
        assert len(skipped) > 0, (
            "dry-run classified nothing as skipped; either the "
            "registry is empty or the orchestrator misclassified — "
            "investigate stats.results"
        )

    def test_spec_loop_default_off(self, monkeypatch):
        """Without env flag, the legacy loop runs (not the spec loop).
        Verify by checking that `_run_via_specs` is NOT called when
        `SNDR_APPLY_VIA_SPECS` is absent."""
        monkeypatch.delenv("SNDR_APPLY_VIA_SPECS", raising=False)
        o = _orch()
        called = {"spec": 0}
        original = o._run_via_specs

        def spy(stats):
            called["spec"] += 1
            return original(stats)

        monkeypatch.setattr(o, "_run_via_specs", spy)
        o.run(verbose=False, apply=False)
        assert called["spec"] == 0, (
            "spec loop should not run when SNDR_APPLY_VIA_SPECS is unset"
        )

    @pytest.mark.parametrize("env_value", ["1", "true", "yes", "on", "TRUE"])
    def test_spec_loop_engages_on_truthy_env(self, monkeypatch, env_value):
        """All canonical truthy values activate the spec loop."""
        monkeypatch.setenv("SNDR_APPLY_VIA_SPECS", env_value)
        o = _orch()
        called = {"spec": 0}
        original = o._run_via_specs

        def spy(stats):
            called["spec"] += 1
            return original(stats)

        monkeypatch.setattr(o, "_run_via_specs", spy)
        o.run(verbose=False, apply=False)
        assert called["spec"] == 1


# ─── Spec-loop result shape ───────────────────────────────────────────────


class TestSpecLoopResultShape:
    def test_each_result_is_patchresult(self, monkeypatch):
        from vllm.sndr_core.apply._state import PatchResult
        monkeypatch.setenv("SNDR_APPLY_VIA_SPECS", "1")
        o = _orch()
        stats = o.run(verbose=False, apply=False)
        for r in stats.results:
            assert isinstance(r, PatchResult)
            assert r.status in ("applied", "skipped", "failed")

    def test_result_name_includes_patch_id(self, monkeypatch):
        """Display name must start with the patch_id so logs are
        grep-able by ID — same contract as the legacy loop."""
        monkeypatch.setenv("SNDR_APPLY_VIA_SPECS", "1")
        o = _orch()
        stats = o.run(verbose=False, apply=False)
        # Find PN82's entry — recently added, has apply_module
        pn82_results = [r for r in stats.results if "PN82" in r.name]
        assert pn82_results, "PN82 must appear in spec-loop results"
