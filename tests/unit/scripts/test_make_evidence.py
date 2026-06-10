# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/make_evidence.py` — Phase 0 supplement aggregate.

Contract: the script enumerates every release gate, runs them, captures
exit codes + tail output, and returns a structured summary. Gating
gate failures block release (exit 1); informational gate failures don't.

We use --only mode to exercise the single-gate path against a known
gate (`audit-no-new-v1` — very fast, no I/O cost on a clean tree).
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "make_evidence.py"


def _import_script():
    name = "_make_evidence_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── Gate catalogue ───────────────────────────────────────────────────


class TestGateCatalogue:
    def test_gates_tuple_nonempty(self):
        mod = _import_script()
        assert len(mod.GATES) >= 10, (
            f"GATES tuple should cover all release audits; got {len(mod.GATES)}"
        )

    def test_every_gate_has_required_fields(self):
        mod = _import_script()
        for g in mod.GATES:
            assert g.name
            assert g.make_target
            assert g.description
            assert g.severity in ("gating", "informational"), (
                f"gate {g.name!r} has invalid severity {g.severity!r}"
            )

    def test_unique_gate_names(self):
        mod = _import_script()
        names = [g.name for g in mod.GATES]
        assert len(names) == len(set(names)), (
            f"duplicate gate names: {[n for n in names if names.count(n) > 1]}"
        )

    def test_release_only_filter(self):
        mod = _import_script()
        default_set = mod._gates_for_mode(include_release=False)
        release_set = mod._gates_for_mode(include_release=True)
        # Release set ≥ default set.
        assert len(release_set) >= len(default_set)
        # Release-only gates appear ONLY when include_release=True.
        release_only_names = {g.name for g in mod.GATES if g.release_only}
        default_names = {g.name for g in default_set}
        for n in release_only_names:
            assert n not in default_names


# ─── _run_gate single-gate execution ─────────────────────────────────


class TestRunGate:
    def test_runs_real_fast_gate(self):
        """audit-no-new-v1 is fast and stable on a clean tree."""
        mod = _import_script()
        gate = next(g for g in mod.GATES if g.name == "audit-no-new-v1")
        result = mod._run_gate(gate, timeout_s=60)
        assert result.gate is gate
        assert isinstance(result.exit_code, int)
        assert isinstance(result.duration_s, float)
        assert result.duration_s >= 0

    def test_passing_gate_does_not_block_release(self):
        mod = _import_script()
        gate = next(g for g in mod.GATES if g.name == "audit-no-new-v1")
        result = mod._run_gate(gate, timeout_s=60)
        assert result.passed
        assert result.blocks_release is False

    def test_unknown_target_returns_nonzero(self):
        mod = _import_script()
        # Synthetic gate pointing at a make target that doesn't exist.
        fake = mod.Gate(
            name="fake-gate", make_target="does-not-exist-target",
            description="synthetic", severity="gating",
        )
        result = mod._run_gate(fake, timeout_s=30)
        assert result.exit_code != 0
        assert result.blocks_release is True


# ─── Renderers ────────────────────────────────────────────────────────


def _fake_result(mod, *, gate_name="x", severity="gating",
                 exit_code=0, release_only=False):
    gate = mod.Gate(
        name=gate_name, make_target=f"audit-{gate_name}",
        description="test", severity=severity, release_only=release_only,
    )
    return mod.GateResult(
        gate=gate, exit_code=exit_code, duration_s=0.1,
        stdout_tail="ok" if exit_code == 0 else "FAIL",
        stderr_tail="",
    )


class TestRenderers:
    def test_render_text_all_green(self):
        mod = _import_script()
        results = [
            _fake_result(mod, gate_name="a", exit_code=0),
            _fake_result(mod, gate_name="b", exit_code=0),
        ]
        text = mod.render_text(results)
        assert "2/2 gate(s) green" in text
        assert "RELEASE BLOCKED" not in text

    def test_render_text_blocking_failure(self):
        mod = _import_script()
        results = [
            _fake_result(mod, gate_name="a", exit_code=0),
            _fake_result(mod, gate_name="b", exit_code=1, severity="gating"),
        ]
        text = mod.render_text(results)
        assert "RELEASE BLOCKED" in text

    def test_render_text_informational_failure_not_blocking(self):
        mod = _import_script()
        results = [
            _fake_result(mod, gate_name="a", exit_code=0),
            _fake_result(mod, gate_name="b", exit_code=1,
                         severity="informational"),
        ]
        text = mod.render_text(results)
        # Not blocking → no RELEASE BLOCKED banner.
        assert "RELEASE BLOCKED" not in text
        assert "informational warning" in text

    def test_render_json_shape(self):
        mod = _import_script()
        results = [
            _fake_result(mod, gate_name="a", exit_code=0),
            _fake_result(mod, gate_name="b", exit_code=1, severity="gating"),
        ]
        payload = json.loads(mod.render_json(results))
        # Documented keys.
        for k in ("timestamp", "host", "commit_sha", "total_gates",
                  "passed", "blocking_failures", "informational_failures",
                  "release_blocked", "gates"):
            assert k in payload
        assert payload["total_gates"] == 2
        assert payload["passed"] == 1
        assert payload["blocking_failures"] == 1
        assert payload["release_blocked"] is True
        # Each gate has the documented shape.
        for g in payload["gates"]:
            for k in ("name", "make_target", "severity", "release_only",
                      "exit_code", "duration_s", "passed", "blocks_release",
                      "stdout_tail", "stderr_tail"):
                assert k in g

    def test_render_markdown_ledger_entry(self):
        mod = _import_script()
        results = [
            _fake_result(mod, gate_name="a", exit_code=0),
            _fake_result(mod, gate_name="b", exit_code=0),
        ]
        text = mod.render_markdown_ledger_entry(results)
        # Header + checkmarks + status banner.
        assert "### Entry X" in text
        assert "**Release status:** OK" in text
        assert "audit-a" in text or "a " in text
        # No blocking output section when all green.
        assert "Blocking gate output" not in text

    def test_markdown_includes_blocking_tails(self):
        mod = _import_script()
        results = [
            _fake_result(mod, gate_name="a", exit_code=0),
            _fake_result(mod, gate_name="b", exit_code=1, severity="gating"),
        ]
        text = mod.render_markdown_ledger_entry(results)
        assert "**Release status:** BLOCKED" in text
        assert "Blocking gate output" in text


# ─── CLI: --only and --json ───────────────────────────────────────────


class TestCLIInvocation:
    def test_only_single_gate_runs(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--only", "audit-no-new-v1", "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["total_gates"] == 1
        assert payload["gates"][0]["name"] == "audit-no-new-v1"

    def test_only_unknown_returns_two(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--only", "does-not-exist"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 2
        assert "not in known gates" in result.stderr

    def test_emit_md_writes_file(self, tmp_path):
        out = tmp_path / "ledger_entry.md"
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--only", "audit-no-new-v1",
             "--emit-md", str(out), "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        assert out.is_file()
        text = out.read_text(encoding="utf-8")
        assert "### Entry X" in text
        assert "audit-no-new-v1" in text


# ─── End-to-end: all-gating-gates-currently-green ─────────────────────


class TestAllGatingGreenOnRepoRoot:
    """The committed tree must pass every GATING gate. Informational
    gate failures (docs-stale, public-docs, security pre-existing
    drift) are accepted.
    """

    def test_aggregate_run_blocking_count_zero(self):
        # `audit-release-check` consumes the per-host
        # ``evidence/patch_proof/<id>__*.json`` artefacts which are
        # gitignored (each carries a hostname + measured_at + commit
        # SHA, so they re-generate per checkout). CI regenerates them
        # via `sndr patches prove --all` before running make_evidence;
        # mirror that contract here so a fresh clone exercises the
        # same green-on-green state as CI.
        subprocess.run(
            [sys.executable, "-m", "sndr.cli.legacy",
             "patches", "prove", "--all"],
            cwd=REPO_ROOT, capture_output=True, text=True,
            timeout=120, check=False,
        )

        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
        )
        # rc=0 means no gating gate failed (informational may have).
        payload = json.loads(result.stdout)
        assert payload["blocking_failures"] == 0, (
            f"blocking gating failures on committed tree: "
            f"{[g['name'] for g in payload['gates'] if g['blocks_release']]}"
        )
        assert result.returncode == 0
