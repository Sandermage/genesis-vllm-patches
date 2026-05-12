# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_bench_methodology.py` — stale-bench detector
(Entry 26).

Contract:

  • Empty proof_dir + default flag → vacuously passes (operator-gated GPU).
  • Empty proof_dir + `--no-bench-allow-empty` → fails.
  • Artefact with no bench_delta → passes (static-only, methodology N/A).
  • Artefact with bench_delta + matching methodology_sha → passes.
  • Artefact with bench_delta + stale methodology_sha → fails.
  • Artefact with bench_delta + missing methodology_sha → fails.
  • Live committed repo + canonical methodology = SHA stable.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_bench_methodology.py"


def _import_script():
    name = "_audit_bench_methodology_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_artefact(out_dir: Path, name: str, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _write_methodology(p: Path, text: str = "schema_version: 1\n") -> Path:
    p.write_text(text, encoding="utf-8")
    return p


# ─── Canonical SHA helper ─────────────────────────────────────────────


class TestCanonicalSha:
    def test_methodology_file_exists(self):
        mod = _import_script()
        sha = mod._canonical_methodology_sha()
        assert isinstance(sha, str)
        assert len(sha) == 64   # sha256 hex digest length

    def test_missing_file_raises(self, tmp_path):
        mod = _import_script()
        with pytest.raises(FileNotFoundError):
            mod._canonical_methodology_sha(tmp_path / "nope.yaml")

    def test_sha_is_stable_for_same_bytes(self, tmp_path):
        mod = _import_script()
        f1 = _write_methodology(tmp_path / "m.yaml", "a: 1\n")
        f2 = _write_methodology(tmp_path / "m2.yaml", "a: 1\n")
        assert (mod._canonical_methodology_sha(f1)
                == mod._canonical_methodology_sha(f2))


# ─── Per-artefact check ───────────────────────────────────────────────


class TestArtefactCheck:
    def test_no_bench_delta_passes(self, tmp_path):
        mod = _import_script()
        m = _write_methodology(tmp_path / "m.yaml")
        sha = mod._canonical_methodology_sha(m)
        a = _write_artefact(tmp_path, "P58__v1.json", {
            "patch_id": "P58", "static_passed": True, "bench_delta": None,
        })
        r = mod._audit_one_artefact(a, sha)
        assert r.status == "no_bench_delta"
        assert r.passed is True

    def test_matching_sha_passes(self, tmp_path):
        mod = _import_script()
        m = _write_methodology(tmp_path / "m.yaml")
        sha = mod._canonical_methodology_sha(m)
        a = _write_artefact(tmp_path, "P58__v1.json", {
            "patch_id": "P58",
            "bench_delta": {"median_tps": 1.0, "methodology_sha": sha},
        })
        r = mod._audit_one_artefact(a, sha)
        assert r.status == "match"
        assert r.passed is True

    def test_stale_sha_fails(self, tmp_path):
        mod = _import_script()
        m = _write_methodology(tmp_path / "m.yaml")
        sha = mod._canonical_methodology_sha(m)
        a = _write_artefact(tmp_path, "P58__v1.json", {
            "patch_id": "P58",
            "bench_delta": {
                "median_tps": 1.0,
                "methodology_sha": "deadbeef" * 8,
            },
        })
        r = mod._audit_one_artefact(a, sha)
        assert r.status == "stale"
        assert r.passed is False

    def test_missing_sha_fails(self, tmp_path):
        mod = _import_script()
        m = _write_methodology(tmp_path / "m.yaml")
        sha = mod._canonical_methodology_sha(m)
        a = _write_artefact(tmp_path, "P58__v1.json", {
            "patch_id": "P58",
            "bench_delta": {"median_tps": 1.0},   # no methodology_sha
        })
        r = mod._audit_one_artefact(a, sha)
        assert r.status == "missing_sha"
        assert r.passed is False

    def test_malformed_artefact_marked_error(self, tmp_path):
        mod = _import_script()
        m = _write_methodology(tmp_path / "m.yaml")
        sha = mod._canonical_methodology_sha(m)
        bad = tmp_path / "P58__v1.json"
        bad.write_text("not valid {json", encoding="utf-8")
        r = mod._audit_one_artefact(bad, sha)
        assert r.status == "error"
        assert r.passed is False


# ─── Whole-directory audit ────────────────────────────────────────────


class TestAuditDirectory:
    def test_empty_returns_zero_artefacts(self, tmp_path):
        mod = _import_script()
        proof = tmp_path / "evidence" / "patch_proof"
        proof.mkdir(parents=True)
        m = _write_methodology(tmp_path / "m.yaml")
        sha, results = mod.audit_bench_methodology(
            proof_dir=proof, methodology_file=m,
        )
        assert results == []
        assert isinstance(sha, str) and len(sha) == 64

    def test_mixed_results(self, tmp_path):
        mod = _import_script()
        m = _write_methodology(tmp_path / "m.yaml")
        sha = mod._canonical_methodology_sha(m)
        proof = tmp_path / "proof"
        proof.mkdir()
        # 3 artefacts: 1 match, 1 stale, 1 no-bench.
        _write_artefact(proof, "P1__v1.json", {
            "patch_id": "P1",
            "bench_delta": {"median_tps": 1, "methodology_sha": sha},
        })
        _write_artefact(proof, "P2__v1.json", {
            "patch_id": "P2",
            "bench_delta": {"median_tps": 1, "methodology_sha": "stale" * 16},
        })
        _write_artefact(proof, "P3__v1.json", {
            "patch_id": "P3", "bench_delta": None,
        })
        sha2, results = mod.audit_bench_methodology(
            proof_dir=proof, methodology_file=m,
        )
        assert sha == sha2
        statuses = {r.patch_id: r.status for r in results}
        assert statuses == {
            "P1": "match", "P2": "stale", "P3": "no_bench_delta",
        }


# ─── CLI ──────────────────────────────────────────────────────────────


class TestScriptCLI:
    def test_cli_empty_dir_default_passes(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--proof-dir", str(empty), "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout
        payload = json.loads(result.stdout)
        assert payload["passed"] is True
        assert payload["total_artefacts"] == 0

    def test_cli_empty_dir_strict_fails(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--proof-dir", str(empty),
             "--no-bench-allow-empty", "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 1
        payload = json.loads(result.stdout)
        assert payload["passed"] is False

    def test_cli_stale_artefact_fails(self, tmp_path):
        # Use live methodology file so canonical SHA is real.
        proof = tmp_path / "p"
        proof.mkdir()
        _write_artefact(proof, "P58__v1.json", {
            "patch_id": "P58",
            "bench_delta": {
                "median_tps": 1.0, "methodology_sha": "stale" * 16,
            },
        })
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--proof-dir", str(proof), "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 1
        payload = json.loads(result.stdout)
        assert payload["by_status"].get("stale", 0) == 1

    def test_cli_live_repo_passes(self):
        """Live `evidence/patch_proof/` is empty (or contains only match);
        either way CLI must exit 0 in default (allow-empty) mode."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout[:2000]
