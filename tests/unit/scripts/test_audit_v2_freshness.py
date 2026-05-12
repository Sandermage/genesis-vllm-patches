# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_v2_freshness.py` — V2 model last_validated
staleness gate (Entry 27).

Contract:

  • Parseable ISO date → ok/stale based on age.
  • Future date → status=future.
  • Missing field → status=missing.
  • Unparseable string → status=unparseable.
  • --today override + --max-age-days override work.
  • Committed repo passes with default 180-day window.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_v2_freshness.py"


def _import_script():
    name = "_audit_v2_freshness_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_yaml(p: Path, text: str) -> Path:
    p.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    return p


# ─── Date parsing ─────────────────────────────────────────────────────


class TestParseIsoDate:
    def test_string_iso(self):
        mod = _import_script()
        assert mod._parse_iso_date("2026-05-12") == dt.date(2026, 5, 12)

    def test_string_with_whitespace(self):
        mod = _import_script()
        assert mod._parse_iso_date("  2026-05-12 ") == dt.date(2026, 5, 12)

    def test_date_object(self):
        mod = _import_script()
        d = dt.date(2026, 5, 12)
        assert mod._parse_iso_date(d) == d

    def test_datetime_returns_date_part(self):
        mod = _import_script()
        dtm = dt.datetime(2026, 5, 12, 14, 30)
        assert mod._parse_iso_date(dtm) == dt.date(2026, 5, 12)

    def test_invalid_string_returns_none(self):
        mod = _import_script()
        assert mod._parse_iso_date("not-a-date") is None

    def test_non_string_non_date_returns_none(self):
        mod = _import_script()
        assert mod._parse_iso_date(42) is None
        assert mod._parse_iso_date(None) is None


# ─── Per-file check ───────────────────────────────────────────────────


def _model_yaml(date_value: str) -> str:
    return textwrap.dedent(f"""
        kind: model
        id: synth-model
        last_validated: {date_value}
    """).lstrip("\n")


class TestCheckOneModel:
    def test_fresh_within_threshold(self, tmp_path):
        mod = _import_script()
        today = dt.date(2026, 5, 13)
        y = _write_yaml(tmp_path / "m.yaml", _model_yaml("'2026-05-10'"))
        r = mod.check_one_model(y, today=today, max_age_days=180)
        assert r.passed is True
        assert r.status == "ok"
        assert r.age_days == 3

    def test_stale_beyond_threshold(self, tmp_path):
        mod = _import_script()
        today = dt.date(2026, 5, 13)
        y = _write_yaml(tmp_path / "m.yaml", _model_yaml("'2025-01-01'"))
        r = mod.check_one_model(y, today=today, max_age_days=180)
        assert r.passed is False
        assert r.status == "stale"
        assert r.age_days > 180

    def test_future_dated(self, tmp_path):
        mod = _import_script()
        today = dt.date(2026, 5, 13)
        y = _write_yaml(tmp_path / "m.yaml", _model_yaml("'2099-01-01'"))
        r = mod.check_one_model(y, today=today, max_age_days=180)
        assert r.passed is False
        assert r.status == "future"
        assert r.age_days < 0

    def test_missing_field(self, tmp_path):
        mod = _import_script()
        y = _write_yaml(tmp_path / "m.yaml",
                        "kind: model\nid: synth\n")
        r = mod.check_one_model(y, today=dt.date.today(), max_age_days=180)
        assert r.passed is False
        assert r.status == "missing"

    def test_unparseable_string(self, tmp_path):
        mod = _import_script()
        y = _write_yaml(tmp_path / "m.yaml", _model_yaml("'not-a-date'"))
        r = mod.check_one_model(y, today=dt.date.today(), max_age_days=180)
        assert r.passed is False
        assert r.status == "unparseable"

    def test_boundary_exactly_at_threshold(self, tmp_path):
        """Exactly N days old passes (strict > inequality)."""
        mod = _import_script()
        today = dt.date(2026, 5, 13)
        old = (today - dt.timedelta(days=180)).isoformat()
        y = _write_yaml(tmp_path / "m.yaml", _model_yaml(f"'{old}'"))
        r = mod.check_one_model(y, today=today, max_age_days=180)
        assert r.passed is True


# ─── Directory-level audit ────────────────────────────────────────────


class TestAuditDirectory:
    def test_empty_dir(self, tmp_path):
        mod = _import_script()
        empty = tmp_path / "empty"
        empty.mkdir()
        out = mod.audit_v2_freshness(
            model_dir=empty,
            today=dt.date(2026, 5, 13),
            max_age_days=180,
        )
        assert out == []

    def test_mixed_results(self, tmp_path):
        mod = _import_script()
        d = tmp_path / "models"
        d.mkdir()
        _write_yaml(d / "fresh.yaml", _model_yaml("'2026-05-10'"))
        _write_yaml(d / "stale.yaml", _model_yaml("'2024-01-01'"))
        _write_yaml(d / "future.yaml", _model_yaml("'2099-01-01'"))
        out = mod.audit_v2_freshness(
            model_dir=d,
            today=dt.date(2026, 5, 13),
            max_age_days=180,
        )
        statuses = {r.model_id: r.status for r in out}
        # All have synth-model id since the template is shared; check
        # by status counts instead.
        status_counter = sorted(r.status for r in out)
        assert status_counter == ["future", "ok", "stale"]


# ─── Live repo — regression anchor ────────────────────────────────────


class TestLiveRepo:
    def test_committed_models_fresh_default_window(self):
        """All committed V2 models must be fresh under the default
        180-day window. If this test fails, operator must re-validate."""
        mod = _import_script()
        results = mod.audit_v2_freshness()
        stale = [r for r in results if not r.passed]
        assert stale == [], (
            f"Stale V2 models (default 180d window):\n"
            + "\n".join(
                f"  {r.model_id}: {r.status} ({r.age_days}d)" for r in stale
            )
        )
        assert len(results) >= 6


# ─── CLI ──────────────────────────────────────────────────────────────


class TestScriptCLI:
    def test_cli_zero(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout[:2000]

    def test_cli_json_shape(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert "by_status" in payload
        assert payload["failed"] == 0

    def test_cli_today_override(self):
        """`--today 2030-01-01` makes every committed model stale."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--today", "2030-01-01", "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 1
        payload = json.loads(result.stdout)
        assert payload["by_status"]["stale"] >= 1

    def test_cli_tight_window_fails(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--max-age-days", "0", "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 1
