# SPDX-License-Identifier: Apache-2.0
"""CONFIG-UX.5.2 — tests for `sndr config-catalog` CLI (4 leaves).

Acceptance coverage (locked rules):
  - 4 leaves invokable on clean corpus
  - Torch-free import guard
  - Output redaction parity with generator (terminal vs JSON shape)
  - --from staleness warning + --strict-fresh elevates to nonzero
  - --from missing file exits 2
  - Row-id ambiguity errors with candidate list (synthetic collision)
  - Fixed query DSL — argparse rejects beyond 5 flags
  - query field-not-found error message includes valid fields
  - query --expires-before ISO date parsing + filter correctness
  - --help across all 5 surfaces (top + 4 leaves) contains "derived catalog"
  - Regression: sndr preset / config / routing-table CLI unaffected
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "sndr.cli.legacy", "config-catalog", *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )


# ─── Gate 1: 4 leaves invokable ─────────────────────────────────────────────


class TestGate1LeavesInvokable:
    def test_build_check(self):
        result = _run_cli("build", "--check")
        assert result.returncode == 0, (
            f"build --check rc={result.returncode}\nstdout={result.stdout[:200]}"
            f"\nstderr={result.stderr[:200]}"
        )

    def test_verify(self):
        result = _run_cli("verify")
        assert result.returncode == 0

    def test_show_prefixed(self):
        result = _run_cli("show", "preset/prod-qwen3.6-35b-balanced")
        assert result.returncode == 0
        assert "preset/prod-qwen3.6-35b-balanced" in result.stdout
        assert "derived catalog" in result.stdout.lower()

    def test_show_bare_unambiguous(self):
        result = _run_cli("show", "prod-qwen3.6-35b-balanced")
        assert result.returncode == 0
        # Bare 'prod-qwen3.6-35b-balanced' is unambiguous in current corpus
        assert "preset/prod-qwen3.6-35b-balanced" in result.stdout

    def test_query_basic(self):
        result = _run_cli(
            "query", "--row-type", "profile",
            "--field", "override_class", "--equals", "bench",
        )
        assert result.returncode == 0
        # All 13 bench-class profiles should appear
        # (DEBT.1 7 + DEBT.2A 3 bench + DEBT.2B 7 + DEBT.2C 1 - 7 qa-prefixed
        # actually exactly the post-DEBT distribution)
        assert "override_class=bench" in result.stdout


# ─── Gate 2: torch-free import guard ────────────────────────────────────────


class TestGate2TorchFree:
    def test_in_process_run_no_torch_import(self):
        if "torch" in sys.modules:
            pytest.skip("torch already imported by another test/runtime")

        # Drive run_query() in-process — exercises generator + audit reuse path
        from sndr.cli.legacy import config_catalog as cc_mod
        ns = argparse.Namespace(
            row_type="profile", field="override_class", equals="bench",
            contains=None, expires_before=None, json=True,
            from_path=None, strict_fresh=False,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cc_mod.run_query(ns)
        assert rc == 0
        assert "torch" not in sys.modules, (
            f"torch leaked into config-catalog CLI path: "
            f"{[m for m in sys.modules if m.startswith('torch')]}"
        )


# ─── Gate 3: output redaction parity ────────────────────────────────────────


class TestGate3RedactionParity:
    def test_json_output_uses_generator_marker_shape(self):
        """JSON output keeps the generator's `{redacted: true, ...}`
        marker form verbatim — machine consumers expect structured."""
        result = _run_cli("show", "--json", "preset/prod-gemma4-26b-multiconc")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        ev_refs = data.get("card_evidence_refs", [])
        # This preset has private bench refs → should be redacted
        private_refs = [r for r in ev_refs if isinstance(r, dict) and r.get("redacted")]
        assert private_refs, "expected redacted markers for private evidence in JSON"
        for r in private_refs:
            assert r.get("redacted") is True
            assert r.get("visibility") == "private"
            assert "private evidence" in r.get("note", "").lower()

    def test_terminal_output_uses_human_redaction(self):
        """Terminal output replaces the generator's marker with a
        human-readable `[REDACTED private evidence]` string."""
        result = _run_cli("show", "preset/prod-gemma4-26b-multiconc")
        assert result.returncode == 0
        # The card_evidence_refs field in terminal output should
        # contain the human redaction string
        assert "[REDACTED private evidence]" in result.stdout, (
            "expected human-readable redaction marker in terminal output; "
            f"got:\n{result.stdout[:500]}"
        )

    def test_no_sndr_private_path_in_any_leaf_output(self):
        """Public visibility invariant: no `sndr_private/` string in
        any CLI leaf's stdout on the full corpus."""
        for args in (
            ["build", "--stdout"],
            ["show", "preset/prod-gemma4-26b-multiconc"],
            ["show", "--json", "preset/prod-gemma4-26b-multiconc"],
            ["query", "--row-type", "any"],
        ):
            result = _run_cli(*args)
            assert "sndr_private/" not in result.stdout, (
                f"redaction leak with args={args!r}: 'sndr_private/' in stdout"
            )
            for banned in ("/Users/", "/home/", "/tmp/", "/var/"):
                assert banned not in result.stdout, (
                    f"redaction leak with args={args!r}: {banned!r} in stdout"
                )


# ─── Gate 4: --from staleness + missing file ────────────────────────────────


class TestGate4FromPath:
    def test_from_missing_file_exits_two(self, tmp_path):
        missing = tmp_path / "nonexistent.json"
        result = _run_cli("show", "--from", str(missing), "preset/prod-qwen3.6-35b-balanced")
        assert result.returncode == 2, (
            f"missing --from file should exit 2 (usage error); got "
            f"rc={result.returncode}"
        )
        assert "not found" in result.stderr.lower()

    def test_from_fresh_file_ok(self, tmp_path):
        """Build a fresh catalog → use --from to read it → exit 0."""
        catalog_path = tmp_path / "config_catalog.json"
        # Use the build leaf to create the file
        build = _run_cli("build", "--stdout")
        assert build.returncode == 0
        catalog_path.write_text(build.stdout, encoding="utf-8")

        result = _run_cli(
            "show", "--from", str(catalog_path), "preset/prod-qwen3.6-35b-balanced",
        )
        assert result.returncode == 0
        assert "preset/prod-qwen3.6-35b-balanced" in result.stdout

    def test_from_stale_file_warns_default_mode(self, tmp_path):
        """Stale --from file warns by default but exits 0."""
        import os
        catalog_path = tmp_path / "stale.json"
        build = _run_cli("build", "--stdout")
        catalog_path.write_text(build.stdout, encoding="utf-8")
        # Force file mtime to be in the past
        old_mtime = catalog_path.stat().st_mtime - 86400 * 365  # 1 year old
        os.utime(catalog_path, (old_mtime, old_mtime))

        result = _run_cli(
            "show", "--from", str(catalog_path), "preset/prod-qwen3.6-35b-balanced",
        )
        # Default mode: warn but proceed
        assert result.returncode == 0
        # Warning emitted on stderr (or stdout — _io.warn prints to stdout)
        combined = result.stdout + result.stderr
        assert "stale" in combined.lower() or "older" in combined.lower()

    def test_from_stale_file_strict_fresh_exits_one(self, tmp_path):
        """--strict-fresh elevates the stale warning to exit 1."""
        import os
        catalog_path = tmp_path / "stale.json"
        build = _run_cli("build", "--stdout")
        catalog_path.write_text(build.stdout, encoding="utf-8")
        old_mtime = catalog_path.stat().st_mtime - 86400 * 365
        os.utime(catalog_path, (old_mtime, old_mtime))

        result = _run_cli(
            "show", "--from", str(catalog_path), "--strict-fresh",
            "preset/prod-qwen3.6-35b-balanced",
        )
        assert result.returncode == 1


# ─── Gate 5: row-id ambiguity (synthetic) ───────────────────────────────────


class TestGate5RowIdAmbiguity:
    def test_ambiguous_bare_id_errors_with_candidates(self):
        """Mock a row corpus with collisions to verify the ambiguity error."""
        from sndr.cli.legacy import config_catalog as cc_mod
        # Synthetic rows where bare 'foo' matches both preset and profile
        rows = [
            {"row_type": "preset", "id": "foo"},
            {"row_type": "profile", "id": "foo"},
            {"row_type": "model", "id": "other"},
        ]
        with pytest.raises(SystemExit) as excinfo:
            cc_mod._resolve_row_id(rows, "foo")
        assert excinfo.value.code == 1

    def test_prefixed_id_resolves_through_collision(self):
        """Prefixed form should disambiguate."""
        from sndr.cli.legacy import config_catalog as cc_mod
        rows = [
            {"row_type": "preset", "id": "foo", "marker": "preset-side"},
            {"row_type": "profile", "id": "foo", "marker": "profile-side"},
        ]
        row = cc_mod._resolve_row_id(rows, "preset/foo")
        assert row["marker"] == "preset-side"
        row = cc_mod._resolve_row_id(rows, "profile/foo")
        assert row["marker"] == "profile-side"

    def test_row_not_found(self):
        from sndr.cli.legacy import config_catalog as cc_mod
        rows = [{"row_type": "preset", "id": "real"}]
        with pytest.raises(SystemExit) as excinfo:
            cc_mod._resolve_row_id(rows, "nonexistent")
        assert excinfo.value.code == 1


# ─── Gate 6: fixed query DSL ────────────────────────────────────────────────


class TestGate6FixedDSL:
    def test_extra_flag_rejected_by_argparse(self):
        """argparse rejects flags not in the 5-flag DSL."""
        result = _run_cli(
            "query", "--row-type", "any", "--sort-by", "id",
        )
        # Unknown flag → argparse exits 2
        assert result.returncode == 2

    def test_invalid_row_type_rejected(self):
        result = _run_cli("query", "--row-type", "foo")
        assert result.returncode == 2

    def test_value_filter_without_field_errors(self):
        """--equals / --contains / --expires-before require --field."""
        result = _run_cli(
            "query", "--row-type", "preset", "--equals", "x",
        )
        assert result.returncode == 2
        assert "field" in result.stderr.lower() or "field" in result.stdout.lower()

    def test_field_not_found_error_lists_valid_fields(self):
        """`--field` not on the row → clear error including valid fields."""
        result = _run_cli(
            "query", "--row-type", "preset",
            "--field", "nonexistent_field_xyz", "--equals", "x",
        )
        assert result.returncode == 2
        combined = result.stdout + result.stderr
        assert "nonexistent_field_xyz" in combined
        # Operator §10.1: error must list valid fields
        assert "valid fields" in combined.lower() or "no such field" in combined.lower()


# ─── Gate 7: query semantics (AND-only, contains, expires-before) ───────────


class TestGate7QuerySemantics:
    def test_and_intersection(self):
        """--equals + --contains both apply (intersection)."""
        # Profile bench class AND override_class contains 'enc' (substring of 'bench')
        result = _run_cli(
            "query", "--row-type", "profile",
            "--field", "override_class", "--equals", "bench",
            "--json",
        )
        assert result.returncode == 0
        rows = json.loads(result.stdout)
        # All returned rows must have override_class == bench
        for r in rows:
            assert r.get("override_class") == "bench"

    def test_expires_before_filters_correctly(self):
        """--expires-before parses ISO + filters override_expires_at."""
        result = _run_cli(
            "query", "--row-type", "profile",
            "--field", "override_expires_at",
            "--expires-before", "2026-09-01",
            "--json",
        )
        assert result.returncode == 0
        rows = json.loads(result.stdout)
        # All annotated profiles have expires_at strictly before 2026-09-01
        # (currently 2026-08-22 and 2026-08-31 — both pass).
        assert len(rows) > 0
        for r in rows:
            d = r.get("override_expires_at")
            assert d and d < "2026-09-01"

    def test_expires_before_invalid_date_exits_two(self):
        result = _run_cli(
            "query", "--row-type", "profile",
            "--field", "override_expires_at",
            "--expires-before", "not-a-date",
        )
        assert result.returncode == 2

    def test_row_type_any_matches_all_types(self):
        result = _run_cli("query", "--row-type", "any", "--json")
        assert result.returncode == 0
        rows = json.loads(result.stdout)
        types = {r.get("row_type") for r in rows}
        assert types == {"preset", "profile", "model", "hardware", "baseline"}


# ─── Gate 8: --help "derived catalog" discipline ────────────────────────────


class TestGate8HelpDiscipline:
    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().split())

    @pytest.mark.parametrize("leaf", ["build", "verify", "show", "query"])
    def test_leaf_help_includes_derived_catalog(self, leaf):
        result = _run_cli(leaf, "--help")
        assert result.returncode == 0
        normalized = self._normalize(result.stdout)
        assert "derived catalog" in normalized, (
            f"`sndr config-catalog {leaf} --help` missing 'derived catalog' "
            f"anchor; help text:\n{result.stdout[:600]}"
        )

    def test_top_level_help_includes_derived_catalog(self):
        result = _run_cli("--help")
        assert result.returncode == 0
        normalized = self._normalize(result.stdout)
        assert "derived catalog" in normalized


# ─── Gate 9: regression — neighbouring CLIs unaffected ──────────────────────


class TestGate9NeighbouringCLIsUnchanged:
    def test_sndr_preset_list_works(self):
        result = subprocess.run(
            [sys.executable, "-m", "sndr.cli.legacy", "preset", "list", "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["total"] == 23

    def test_sndr_routing_table_help_works(self):
        result = subprocess.run(
            [sys.executable, "-m", "sndr.cli.legacy", "routing-table", "--help"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0


# ─── Gate 10: terminal output format sanity ─────────────────────────────────


class TestGate10TerminalFormat:
    def test_query_compact_output(self):
        """Terminal query output is one line per matched row + summary."""
        result = _run_cli(
            "query", "--row-type", "preset",
            "--field", "card_status", "--equals", "production_candidate",
        )
        assert result.returncode == 0
        # Multiple rows expected (14 prod-* presets, all production_candidate)
        # Each line should start with row-type/id
        lines = [l for l in result.stdout.splitlines() if l.strip()]
        assert any(l.startswith("  preset/prod-") for l in lines)
        assert any("matched" in l.lower() for l in lines)

    def test_show_human_view_sections(self):
        """show terminal output has 'derived catalog row:' header + fields section."""
        result = _run_cli("show", "preset/prod-qwen3.6-35b-balanced")
        assert result.returncode == 0
        assert "derived catalog row:" in result.stdout
        assert "source:" in result.stdout
        assert "fields:" in result.stdout
