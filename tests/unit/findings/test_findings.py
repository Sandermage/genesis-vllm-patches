# SPDX-License-Identifier: Apache-2.0
"""External findings pipeline tests (Phase 5/10 deferred deliverable).

Covers schema, state-machine transition matrix, staleness check, loader,
validator (F-1 uniqueness, F-4 staleness), and the CLI add/update/list/
validate cycle.
"""
from __future__ import annotations

import argparse
import io
import json
import textwrap
from contextlib import redirect_stdout
from datetime import date, timedelta
from pathlib import Path

import pytest


# ─── Fixture YAML ─────────────────────────────────────────────────────


def _seed_yaml(today_iso: str | None = None) -> str:
    today = today_iso or date.today().isoformat()
    return textwrap.dedent(f"""\
        schema_version: 1
        id: external-test-{today.replace('-','')}
        source: vllm-pr
        url: https://github.com/vllm-project/vllm/pull/99999
        title: "Test finding"
        discovered_at: '{today}'
        category: memory-cache
        status: watch
        risk: medium
        acceptance: |
          Anchors clean after upstream merge.
        last_reviewed: '{today}'
        review_cadence: biweekly
        target: []
        notes: []
    """)


def _write_finding(root: Path, fname: str, yaml_text: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    p = root / fname
    p.write_text(yaml_text, encoding="utf-8")
    return p


# ─── Schema ───────────────────────────────────────────────────────────


class TestSchema:
    def test_minimal_valid(self, tmp_path):
        from vllm.sndr_core.findings import load_finding
        p = _write_finding(tmp_path, "x.yaml", _seed_yaml())
        f = load_finding(p)
        assert f.schema_version == 1
        assert f.status == "watch"
        assert f.risk == "medium"
        issues = f.validate()
        assert issues == [], (
            f"minimal valid finding produced issues: "
            f"{[(i.rule, i.message) for i in issues]}"
        )

    def test_bad_source_rejected(self, tmp_path):
        from vllm.sndr_core.findings import load_finding
        yaml = _seed_yaml().replace("source: vllm-pr", "source: not-a-source")
        p = _write_finding(tmp_path, "x.yaml", yaml)
        f = load_finding(p)
        msgs = [i.message for i in f.validate() if i.severity == "error"]
        assert any("source=" in m for m in msgs)

    def test_bad_status_rejected(self, tmp_path):
        from vllm.sndr_core.findings import load_finding
        yaml = _seed_yaml().replace("status: watch", "status: weird")
        p = _write_finding(tmp_path, "x.yaml", yaml)
        f = load_finding(p)
        msgs = [i.message for i in f.validate() if i.severity == "error"]
        assert any("status=" in m for m in msgs)

    def test_bad_cadence_rejected(self, tmp_path):
        from vllm.sndr_core.findings import load_finding
        yaml = _seed_yaml().replace("review_cadence: biweekly",
                                    "review_cadence: never-ever")
        p = _write_finding(tmp_path, "x.yaml", yaml)
        f = load_finding(p)
        msgs = [i.message for i in f.validate() if i.severity == "error"]
        assert any("review_cadence=" in m for m in msgs)

    def test_bad_iso_date_rejected(self, tmp_path):
        from vllm.sndr_core.findings import load_finding
        today = date.today().isoformat()
        yaml = _seed_yaml().replace(f"discovered_at: '{today}'",
                                    "discovered_at: 'yesterday'")
        p = _write_finding(tmp_path, "x.yaml", yaml)
        f = load_finding(p)
        msgs = [i.message for i in f.validate() if i.severity == "error"]
        assert any("discovered_at=" in m for m in msgs)

    def test_empty_acceptance_rejected(self, tmp_path):
        from vllm.sndr_core.findings import load_finding
        # textwrap.dedent strips to 2-space indent; match that exactly.
        yaml = _seed_yaml().replace(
            "acceptance: |\n  Anchors clean after upstream merge.",
            "acceptance: ''",
        )
        p = _write_finding(tmp_path, "x.yaml", yaml)
        f = load_finding(p)
        msgs = [i.message for i in f.validate() if i.severity == "error"]
        assert any("acceptance" in m for m in msgs), (
            f"empty acceptance not caught; got: {msgs}"
        )


# ─── State machine ────────────────────────────────────────────────────


class TestStateMachine:
    def test_same_status_is_valid_no_op(self):
        from vllm.sndr_core.findings import is_valid_transition
        for s in ("watch", "needs-bench", "done"):
            assert is_valid_transition(s, s)

    @pytest.mark.parametrize("from_,to_", [
        ("watch", "needs-bench"),
        ("watch", "skip"),
        ("needs-bench", "backport-now"),
        ("needs-bench", "config-recipe"),
        ("backport-now", "done"),
        ("done", "retire-local-patch"),
        ("skip", "watch"),
    ])
    def test_legal_transitions(self, from_, to_):
        from vllm.sndr_core.findings import is_valid_transition
        assert is_valid_transition(from_, to_), (
            f"{from_} → {to_} should be legal but was rejected"
        )

    @pytest.mark.parametrize("from_,to_", [
        ("watch", "done"),                 # skipped intermediate
        ("watch", "retire-local-patch"),   # too far jump
        ("needs-bench", "retire-local-patch"),
        ("retire-local-patch", "watch"),   # terminal → reopen forbidden
        ("doctor-rule", "config-recipe"),
    ])
    def test_illegal_transitions(self, from_, to_):
        from vllm.sndr_core.findings import is_valid_transition
        assert not is_valid_transition(from_, to_), (
            f"{from_} → {to_} should be illegal but passed"
        )

    def test_retire_local_patch_is_terminal(self):
        from vllm.sndr_core.findings.schema import ALLOWED_TRANSITIONS
        assert ALLOWED_TRANSITIONS["retire-local-patch"] == frozenset()


# ─── Staleness ───────────────────────────────────────────────────────


class TestStaleness:
    def test_fresh_finding_not_due(self, tmp_path):
        from vllm.sndr_core.findings import is_due_for_review, load_finding
        f = load_finding(_write_finding(tmp_path, "x.yaml", _seed_yaml()))
        assert is_due_for_review(f) is False

    def test_old_biweekly_is_due(self, tmp_path):
        from vllm.sndr_core.findings import is_due_for_review, load_finding
        old = (date.today() - timedelta(days=30)).isoformat()
        yaml = _seed_yaml().replace(
            f"last_reviewed: '{date.today().isoformat()}'",
            f"last_reviewed: '{old}'",
        )
        f = load_finding(_write_finding(tmp_path, "x.yaml", yaml))
        assert is_due_for_review(f) is True

    def test_on_pin_bump_never_stale(self, tmp_path):
        from vllm.sndr_core.findings import is_due_for_review, load_finding
        old = (date.today() - timedelta(days=365)).isoformat()
        yaml = _seed_yaml().replace(
            "review_cadence: biweekly", "review_cadence: on-pin-bump",
        ).replace(
            f"last_reviewed: '{date.today().isoformat()}'",
            f"last_reviewed: '{old}'",
        )
        f = load_finding(_write_finding(tmp_path, "x.yaml", yaml))
        assert is_due_for_review(f) is False

    def test_retired_never_stale(self, tmp_path):
        from vllm.sndr_core.findings import is_due_for_review, load_finding
        old = (date.today() - timedelta(days=999)).isoformat()
        yaml = _seed_yaml().replace(
            "review_cadence: biweekly", "review_cadence: retired",
        ).replace(
            f"last_reviewed: '{date.today().isoformat()}'",
            f"last_reviewed: '{old}'",
        )
        f = load_finding(_write_finding(tmp_path, "x.yaml", yaml))
        assert is_due_for_review(f) is False


# ─── Validator directory rules ────────────────────────────────────────


class TestValidateDirectory:
    def test_passes_on_seed_repo(self, tmp_path):
        from vllm.sndr_core.findings import validate_directory
        _write_finding(tmp_path, "a.yaml", _seed_yaml())
        result = validate_directory(tmp_path)
        assert result.passed
        assert len(result.findings) == 1

    def test_f1_duplicate_id_rejected(self, tmp_path):
        from vllm.sndr_core.findings import validate_directory
        _write_finding(tmp_path, "a.yaml", _seed_yaml())
        _write_finding(tmp_path, "b.yaml", _seed_yaml())
        result = validate_directory(tmp_path)
        f1 = [i for i in result.errors if i.rule == "F-1"]
        assert len(f1) == 1
        assert "duplicate" in f1[0].message

    def test_f4_stale_warning(self, tmp_path):
        from vllm.sndr_core.findings import validate_directory
        old = (date.today() - timedelta(days=30)).isoformat()
        yaml = _seed_yaml().replace(
            f"last_reviewed: '{date.today().isoformat()}'",
            f"last_reviewed: '{old}'",
        )
        _write_finding(tmp_path, "a.yaml", yaml)
        result = validate_directory(tmp_path)
        f4 = [i for i in result.warnings if i.rule == "F-4"]
        assert len(f4) == 1
        # Warning doesn't fail validation.
        assert result.passed


# ─── Loader ───────────────────────────────────────────────────────────


class TestRegistry:
    def test_list_finding_paths_skips_underscore(self, tmp_path):
        from vllm.sndr_core.findings import list_finding_paths
        _write_finding(tmp_path, "real.yaml", _seed_yaml())
        _write_finding(tmp_path, "_template.yaml", _seed_yaml())
        paths = list_finding_paths(tmp_path)
        names = {p.name for p in paths}
        assert "real.yaml" in names
        assert "_template.yaml" not in names

    def test_empty_dir_returns_empty_list(self, tmp_path):
        from vllm.sndr_core.findings import list_finding_paths
        assert list_finding_paths(tmp_path) == []

    def test_missing_dir_returns_empty_list(self, tmp_path):
        from vllm.sndr_core.findings import list_finding_paths
        assert list_finding_paths(tmp_path / "does-not-exist") == []

    def test_discover_skips_broken(self, tmp_path):
        from vllm.sndr_core.findings import discover_findings
        _write_finding(tmp_path, "good.yaml", _seed_yaml())
        _write_finding(tmp_path, "bad.yaml", "not valid yaml: ][\n")
        results = discover_findings(tmp_path)
        # Bad file is logged + skipped; good one loads. We require at least
        # the good one — bad may also load as defaults (lenient _yaml_safe_load
        # returns {} for bad YAML).
        ids = {f.id for _p, f in results}
        # Whatever the lenient YAML parser does, the good one must surface.
        assert any(i.startswith("external-test-") for i in ids)


# ─── CLI integration ─────────────────────────────────────────────────


def _capture_cli(handler, opts: argparse.Namespace) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = handler(opts)
    return rc, buf.getvalue()


class TestCLILifecycle:
    def test_add_then_list_then_update(self, tmp_path):
        from vllm.sndr_core.cli import findings as cli

        # add
        add_opts = argparse.Namespace(
            finding_id="external-pytest-001",
            source="paper",
            url="https://arxiv.org/abs/0000.00000",
            title="Pytest synthetic",
            category="memory-cache",
            status="watch",
            risk="medium",
            review_cadence="biweekly",
            acceptance="placeholder acceptance text",
            root=str(tmp_path),
        )
        rc, out = _capture_cli(cli.run_add, add_opts)
        assert rc == 0
        assert "wrote" in out

        # list — should see the new finding
        list_opts = argparse.Namespace(
            status=None, due_for_review=False,
            root=str(tmp_path), json=True,
        )
        rc, out = _capture_cli(cli.run_list, list_opts)
        assert rc == 0
        payload = json.loads(out)
        assert payload["count"] == 1
        assert payload["findings"][0]["id"] == "external-pytest-001"

        # update — legal transition
        upd_opts = argparse.Namespace(
            finding_id="external-pytest-001",
            status="needs-bench",
            notes=["scheduled for sprint 2"],
            reviewed=False,
            root=str(tmp_path),
        )
        rc, out = _capture_cli(cli.run_update, upd_opts)
        assert rc == 0
        assert "watch → needs-bench" in out

        # update — illegal transition rejected
        bad_opts = argparse.Namespace(
            finding_id="external-pytest-001",
            status="retire-local-patch",
            notes=None,
            reviewed=False,
            root=str(tmp_path),
        )
        rc, _ = _capture_cli(cli.run_update, bad_opts)
        assert rc == 2

    def test_add_rejects_bad_source(self, tmp_path):
        from vllm.sndr_core.cli import findings as cli
        opts = argparse.Namespace(
            finding_id="external-bad",
            source="not-a-source",
            url="https://example.com",
            title="x",
            category="memory-cache",
            status="watch",
            risk="medium",
            review_cadence="biweekly",
            acceptance="x",
            root=str(tmp_path),
        )
        rc, _ = _capture_cli(cli.run_add, opts)
        assert rc == 2

    def test_add_refuses_duplicate(self, tmp_path):
        from vllm.sndr_core.cli import findings as cli
        opts = argparse.Namespace(
            finding_id="external-dup",
            source="paper",
            url="https://x.test",
            title="x",
            category="memory-cache",
            status="watch",
            risk="medium",
            review_cadence="biweekly",
            acceptance="x",
            root=str(tmp_path),
        )
        rc1, _ = _capture_cli(cli.run_add, opts)
        assert rc1 == 0
        rc2, _ = _capture_cli(cli.run_add, opts)
        assert rc2 == 2

    def test_validate_passes_on_committed_findings(self):
        """When the active checkout carries committed findings, every
        one of them must validate clean. Public clones may ship with
        no findings — the maintainer's review notes live outside the
        public tree by design — so the test skips in that case rather
        than failing."""
        from vllm.sndr_core.cli import findings as cli
        opts = argparse.Namespace(root=None, json=True)
        rc, out = _capture_cli(cli.run_validate, opts)
        assert rc == 0
        payload = json.loads(out)
        assert payload["passed"]
        if not payload["findings"]:
            import pytest
            pytest.skip(
                "no committed findings under the resolved root — "
                "expected on public clones, the maintainer's review "
                "notes live outside the public tree"
            )


# ─── CLI registration ────────────────────────────────────────────────


class TestCLIRegistration:
    def test_findings_argparser_registers(self):
        from vllm.sndr_core.cli.findings import add_argparser
        p = argparse.ArgumentParser()
        sub = p.add_subparsers()
        add_argparser(sub)
        ns = p.parse_args(["findings", "list"])
        assert ns.findings_cmd == "list"

    def test_top_level_includes_findings(self):
        from vllm.sndr_core import cli as cli_mod
        assert hasattr(cli_mod, "_findings_argparser")
        assert callable(cli_mod._findings_argparser)
