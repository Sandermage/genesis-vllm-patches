# SPDX-License-Identifier: Apache-2.0
"""Phase 6 acceptance — bench methodology contract.

Contract: `sndr bench-validate <result.json>` catches every shape that
violates the methodology contract — missing fields, methodology_id /
methodology_sha mismatch, warmup/measure_runs mismatch, CV out of
tolerance, tool-call score below baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def methodology_yaml(tmp_path: Path) -> Path:
    """Drop a minimal methodology YAML in tmp_path so tests don't depend
    on the committed `tools/bench_methodology.yaml` (which is itself
    under test elsewhere)."""
    yaml = """\
schema_version: 1
methodology_id: test-methodology
maintainer: testuser
created: '2026-05-12'
prompt_corpus:
  path: tools/bench_corpus/v1/
  sha: null
  scenarios: [long_gen_512t]
measurement:
  warmup_runs: 3
  measure_runs: 10
  cv_warn_pct: 5.0
  cv_fail_pct: 10.0
  per_request_timeout_s: 600
sequence_count:
  policy: from_alias
  fixed_value: null
gpu_clock:
  lock_mode: base
  capture_pre: true
  capture_post: true
  required_in_artefact: true
tolerances:
  median_tps_regression_warn_pct: 2.0
  median_tps_regression_fail_pct: 5.0
  p95_tps_regression_fail_pct: 7.0
  ttft_regression_warn_pct: 5.0
  ttft_regression_fail_pct: 15.0
  tool_call_min_score: 9
  tool_call_must_match_baseline: true
soak:
  enabled: false
  duration_min: 200
  rolling_window_min: 10
  rolling_cv_fail_pct: 3.0
required_artefact_fields:
  - schema_version
  - methodology_id
  - methodology_sha
  - warmup_runs
  - measure_runs
  - cv_pct
  - tool_call_score
"""
    p = tmp_path / "bench_methodology.yaml"
    p.write_text(yaml, encoding="utf-8")
    return p


def _good_artefact(methodology_path: Path) -> dict:
    """Build a fully-conforming artefact for the test methodology."""
    from sndr.cli.legacy.bench import methodology_sha
    return {
        "schema_version": 1,
        "methodology_id": "test-methodology",
        "methodology_sha": methodology_sha(methodology_path),
        "warmup_runs": 3,
        "measure_runs": 10,
        "cv_pct": 2.5,
        "tool_call_score": "10/10",
    }


def _run_validate(artefact_path: Path, methodology_path: Path,
                  json_mode: bool = False) -> tuple[int, str]:
    from sndr.cli.legacy import bench as bench_cli
    opts = argparse.Namespace(
        artefact=str(artefact_path),
        methodology=str(methodology_path),
        json=json_mode,
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = bench_cli.run_validate(opts)
    return rc, buf.getvalue()


def _write_artefact(tmp_path: Path, data: dict, name: str = "result.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ─── Load + sha helpers ───────────────────────────────────────────────


class TestLoadMethodology:
    def test_load_default_methodology(self):
        from sndr.cli.legacy.bench import load_methodology
        # The committed tools/bench_methodology.yaml must parse cleanly.
        data = load_methodology()
        assert data["schema_version"] == 1
        assert data["methodology_id"] == "wave9-baseline"

    def test_load_test_methodology(self, methodology_yaml):
        from sndr.cli.legacy.bench import load_methodology
        data = load_methodology(methodology_yaml)
        assert data["methodology_id"] == "test-methodology"

    def test_missing_methodology_raises(self, tmp_path):
        from sndr.cli.legacy.bench import load_methodology
        with pytest.raises(FileNotFoundError):
            load_methodology(tmp_path / "does-not-exist.yaml")


class TestMethodologySha:
    def test_sha_is_stable(self, methodology_yaml):
        from sndr.cli.legacy.bench import methodology_sha
        s1 = methodology_sha(methodology_yaml)
        s2 = methodology_sha(methodology_yaml)
        assert s1 == s2
        # SHA-256 produces 64 hex chars.
        assert len(s1) == 64

    def test_sha_changes_when_yaml_changes(self, methodology_yaml):
        from sndr.cli.legacy.bench import methodology_sha
        s1 = methodology_sha(methodology_yaml)
        methodology_yaml.write_text(
            methodology_yaml.read_text() + "\n# trailing comment\n",
            encoding="utf-8",
        )
        s2 = methodology_sha(methodology_yaml)
        assert s1 != s2


# ─── Validator rules ──────────────────────────────────────────────────


class TestValidatorPassesGoodArtefact:
    def test_good_artefact_passes(self, methodology_yaml, tmp_path):
        artefact = _good_artefact(methodology_yaml)
        path = _write_artefact(tmp_path, artefact)
        rc, out = _run_validate(path, methodology_yaml)
        assert rc == 0
        assert "passes methodology contract" in out

    def test_good_artefact_json_mode(self, methodology_yaml, tmp_path):
        artefact = _good_artefact(methodology_yaml)
        path = _write_artefact(tmp_path, artefact)
        rc, out = _run_validate(path, methodology_yaml, json_mode=True)
        assert rc == 0
        payload = json.loads(out)
        assert payload["passed"] is True
        assert payload["errors"] == 0


class TestRuleM1MissingFields:
    def test_each_missing_required_field_caught(
        self, methodology_yaml, tmp_path,
    ):
        """Drop one required field at a time, confirm M-1 catches it."""
        from sndr.cli.legacy.bench import load_methodology
        methodology = load_methodology(methodology_yaml)
        required = methodology["required_artefact_fields"]
        for field_name in required:
            artefact = _good_artefact(methodology_yaml)
            del artefact[field_name]
            path = _write_artefact(tmp_path, artefact, f"{field_name}.json")
            rc, _ = _run_validate(path, methodology_yaml)
            assert rc == 1, f"missing {field_name!r} did not fail validation"

    def test_empty_artefact_reports_all_missing(
        self, methodology_yaml, tmp_path,
    ):
        path = _write_artefact(tmp_path, {})
        rc, out = _run_validate(path, methodology_yaml, json_mode=True)
        assert rc == 1
        payload = json.loads(out)
        m1_issues = [i for i in payload["issues"] if i["rule"] == "M-1"]
        # Test methodology declares 7 required fields.
        assert len(m1_issues) == 7


class TestRuleM2MethodologyIdMismatch:
    def test_wrong_methodology_id(self, methodology_yaml, tmp_path):
        artefact = _good_artefact(methodology_yaml)
        artefact["methodology_id"] = "wrong-id"
        path = _write_artefact(tmp_path, artefact)
        rc, out = _run_validate(path, methodology_yaml, json_mode=True)
        assert rc == 1
        m2 = [i for i in json.loads(out)["issues"] if i["rule"] == "M-2"]
        assert len(m2) == 1
        assert "methodology_id mismatch" in m2[0]["message"]


class TestRuleM3MethodologyShaMismatch:
    def test_wrong_sha(self, methodology_yaml, tmp_path):
        artefact = _good_artefact(methodology_yaml)
        artefact["methodology_sha"] = "0" * 64
        path = _write_artefact(tmp_path, artefact)
        rc, out = _run_validate(path, methodology_yaml, json_mode=True)
        assert rc == 1
        m3 = [i for i in json.loads(out)["issues"] if i["rule"] == "M-3"]
        assert len(m3) == 1
        assert "methodology_sha mismatch" in m3[0]["message"]


class TestRuleM4ProtocolMismatch:
    def test_warmup_mismatch(self, methodology_yaml, tmp_path):
        artefact = _good_artefact(methodology_yaml)
        artefact["warmup_runs"] = 1     # contract requires 3
        path = _write_artefact(tmp_path, artefact)
        rc, out = _run_validate(path, methodology_yaml, json_mode=True)
        assert rc == 1
        m4 = [i for i in json.loads(out)["issues"] if i["rule"] == "M-4"]
        assert any("warmup_runs mismatch" in i["message"] for i in m4)

    def test_measure_mismatch(self, methodology_yaml, tmp_path):
        artefact = _good_artefact(methodology_yaml)
        artefact["measure_runs"] = 5    # contract requires 10
        path = _write_artefact(tmp_path, artefact)
        rc, out = _run_validate(path, methodology_yaml, json_mode=True)
        assert rc == 1
        m4 = [i for i in json.loads(out)["issues"] if i["rule"] == "M-4"]
        assert any("measure_runs mismatch" in i["message"] for i in m4)


class TestRuleM5CVTolerance:
    def test_cv_above_fail_threshold(self, methodology_yaml, tmp_path):
        artefact = _good_artefact(methodology_yaml)
        artefact["cv_pct"] = 12.0    # > cv_fail_pct=10.0
        path = _write_artefact(tmp_path, artefact)
        rc, out = _run_validate(path, methodology_yaml, json_mode=True)
        assert rc == 1
        m5 = [i for i in json.loads(out)["issues"] if i["rule"] == "M-5"]
        assert len(m5) == 1
        assert m5[0]["severity"] == "error"

    def test_cv_in_warn_band(self, methodology_yaml, tmp_path):
        artefact = _good_artefact(methodology_yaml)
        artefact["cv_pct"] = 6.0    # > warn (5.0) but < fail (10.0)
        path = _write_artefact(tmp_path, artefact)
        rc, out = _run_validate(path, methodology_yaml, json_mode=True)
        # Warnings don't fail the gate.
        assert rc == 0
        m5 = [i for i in json.loads(out)["issues"] if i["rule"] == "M-5"]
        assert len(m5) == 1
        assert m5[0]["severity"] == "warning"

    def test_cv_under_warn(self, methodology_yaml, tmp_path):
        artefact = _good_artefact(methodology_yaml)
        artefact["cv_pct"] = 2.5   # under warn
        path = _write_artefact(tmp_path, artefact)
        rc, out = _run_validate(path, methodology_yaml, json_mode=True)
        assert rc == 0
        m5 = [i for i in json.loads(out)["issues"] if i["rule"] == "M-5"]
        assert m5 == []


class TestRuleM6ToolCallScore:
    def test_score_below_min(self, methodology_yaml, tmp_path):
        artefact = _good_artefact(methodology_yaml)
        artefact["tool_call_score"] = "5/10"   # below min=9
        path = _write_artefact(tmp_path, artefact)
        rc, out = _run_validate(path, methodology_yaml, json_mode=True)
        assert rc == 1
        m6 = [i for i in json.loads(out)["issues"] if i["rule"] == "M-6"]
        assert len(m6) == 1

    def test_score_meets_min(self, methodology_yaml, tmp_path):
        artefact = _good_artefact(methodology_yaml)
        artefact["tool_call_score"] = "9/10"
        path = _write_artefact(tmp_path, artefact)
        rc, _ = _run_validate(path, methodology_yaml)
        assert rc == 0

    def test_score_unparseable_warns(self, methodology_yaml, tmp_path):
        artefact = _good_artefact(methodology_yaml)
        artefact["tool_call_score"] = "weird"   # not 'N/10' format
        path = _write_artefact(tmp_path, artefact)
        rc, out = _run_validate(path, methodology_yaml, json_mode=True)
        # Warnings don't fail.
        assert rc == 0
        m6 = [i for i in json.loads(out)["issues"] if i["rule"] == "M-6"]
        assert len(m6) == 1
        assert m6[0]["severity"] == "warning"


# ─── CLI registration ─────────────────────────────────────────────────


class TestCLIRegistration:
    def test_bench_validate_registered(self):
        import argparse
        from sndr.cli.legacy.bench import add_argparser
        p = argparse.ArgumentParser()
        sub = p.add_subparsers(dest="cmd")
        add_argparser(sub)
        ns = p.parse_args(["bench-validate", "/tmp/x.json"])
        assert ns.cmd == "bench-validate"
        assert ns.artefact == "/tmp/x.json"

    def test_bench_methodology_registered(self):
        import argparse
        from sndr.cli.legacy.bench import add_argparser
        p = argparse.ArgumentParser()
        sub = p.add_subparsers(dest="cmd")
        add_argparser(sub)
        ns = p.parse_args(["bench-methodology", "--json"])
        assert ns.cmd == "bench-methodology"
        assert ns.json is True

    def test_top_level_includes_bench(self):
        from sndr.cli import legacy as cli_mod
        assert hasattr(cli_mod, "_bench_argparser")
        assert callable(cli_mod._bench_argparser)


# ─── Error path: file errors ──────────────────────────────────────────


class TestErrorPaths:
    def test_missing_artefact_file(self, methodology_yaml, tmp_path):
        rc, out = _run_validate(
            tmp_path / "does-not-exist.json",
            methodology_yaml,
        )
        # Exit 2 = internal error (missing artefact, not validation fail).
        assert rc == 2
        assert "not found" in out

    def test_malformed_artefact_json(self, methodology_yaml, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json", encoding="utf-8")
        rc, out = _run_validate(path, methodology_yaml)
        assert rc == 2
        assert "JSONDecodeError" in out or "Expecting" in out


# ─── Methodology contract sanity (committed YAML) ─────────────────────


class TestCommittedMethodology:
    """The committed `tools/bench_methodology.yaml` must be loadable
    and structurally sane."""

    def test_committed_yaml_loads(self):
        from sndr.cli.legacy.bench import load_methodology
        data = load_methodology()
        assert data["schema_version"] == 1
        assert data["methodology_id"] == "wave9-baseline"

    def test_committed_yaml_has_required_fields_list(self):
        from sndr.cli.legacy.bench import load_methodology
        data = load_methodology()
        required = data["required_artefact_fields"]
        # Phase 6 contract: at least 12 mandatory fields.
        assert len(required) >= 12
        # Each field is a non-empty string.
        for f in required:
            assert isinstance(f, str)
            assert f
