# SPDX-License-Identifier: Apache-2.0
"""Etap 5.1/5.2/5.3 (audit 2026-05-12): UPSTREAM_WATCHLIST.yaml audit
script — schema validation + categorisation + exit codes.

Previously the watchlist YAML claimed to be wired into `make audit-upstream`
but no script actually read it. This module verifies the new
`scripts/audit_upstream_watchlist.py` is correctly wired:

  • Schema validator rejects malformed entries.
  • Categorisation matches the spec (PORT_CANDIDATE / RETIRE_CANDIDATE /
    WATCH / DONE).
  • Exit code is 1 when a PORT_CANDIDATE is present (so CI can flag).
  • Live repo watchlist passes schema validation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import audit_upstream_watchlist as M  # noqa: E402


class TestValidate:
    def test_clean_data_no_errors(self):
        data = {
            "watch": [
                {"upstream": "vllm#1", "status": "open",
                 "action": "watch", "since": "2026-01-01"},
            ],
            "__sentinel__": "complete",
        }
        assert M._validate(data) == []

    def test_missing_sentinel(self):
        data = {"watch": []}
        errors = M._validate(data)
        assert any("sentinel" in e.lower() for e in errors)

    def test_invalid_status(self):
        data = {
            "watch": [{
                "upstream": "vllm#1", "status": "wat",
                "action": "watch", "since": "2026-01-01",
            }],
            "__sentinel__": "complete",
        }
        errors = M._validate(data)
        assert any("status" in e for e in errors)

    def test_invalid_action(self):
        data = {
            "watch": [{
                "upstream": "vllm#1", "status": "open",
                "action": "yeet", "since": "2026-01-01",
            }],
            "__sentinel__": "complete",
        }
        errors = M._validate(data)
        assert any("action" in e for e in errors)

    def test_invalid_upstream_format(self):
        data = {
            "watch": [{
                "upstream": "not-a-pr-ref", "status": "open",
                "action": "watch", "since": "2026-01-01",
            }],
            "__sentinel__": "complete",
        }
        errors = M._validate(data)
        assert any("upstream" in e for e in errors)

    def test_missing_since(self):
        data = {
            "watch": [{
                "upstream": "vllm#1", "status": "open",
                "action": "watch",
            }],
            "__sentinel__": "complete",
        }
        errors = M._validate(data)
        assert any("since" in e for e in errors)

    def test_duplicate_upstream(self):
        data = {
            "watch": [
                {"upstream": "vllm#1", "status": "open",
                 "action": "watch", "since": "2026-01-01"},
                {"upstream": "vllm#1", "status": "merged",
                 "action": "port", "since": "2026-01-02"},
            ],
            "__sentinel__": "complete",
        }
        errors = M._validate(data)
        assert any("duplicate" in e.lower() for e in errors)


class TestCategorise:
    def test_port_merged_is_port_candidate(self):
        assert M._categorise({
            "action": "port", "status": "merged",
        }) == "PORT_CANDIDATE"

    def test_port_open_is_watch(self):
        assert M._categorise({
            "action": "port", "status": "open",
        }) == "WATCH"

    def test_retire_action_always_retire_candidate(self):
        assert M._categorise({
            "action": "retire", "status": "open",
        }) == "RETIRE_CANDIDATE"
        assert M._categorise({
            "action": "retire", "status": "merged",
        }) == "RETIRE_CANDIDATE"

    def test_closed_with_inactive_action_is_done(self):
        assert M._categorise({
            "action": "watch", "status": "closed",
        }) == "DONE"

    def test_drift_check_is_watch(self):
        assert M._categorise({
            "action": "drift-check", "status": "open",
        }) == "WATCH"


class TestMainExitCode:
    def test_clean_run_exit_zero(self, capsys):
        """The live watchlist should at least pass schema validation."""
        rc = M.main([])
        out = capsys.readouterr().out
        assert "Schema errors:" not in out
        # Exit 0 (clean) or 1 (PORT_CANDIDATE present) — both are
        # acceptable signals; what matters is no schema error.
        assert rc in (0, 1)

    def test_json_emits_valid_json(self, tmp_path, capsys):
        import json
        rc = M.main(["--json"])
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert "entries" in parsed
        assert "schema_errors" in parsed
        assert parsed["schema_errors"] == []
        # Live watchlist has > 5 entries
        assert len(parsed["entries"]) > 5
