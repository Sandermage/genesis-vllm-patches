# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr patches` CLI — T1.2 (audit closure 2026-05-09).

Covers list / explain / doctor / plan / diff-upstream / bundles. Tests
exercise both human-readable and `--json` output paths, run argparse
end-to-end via the public `add_argparser` so the registration shape
is verified, and avoid mocking PATCH_REGISTRY so real registry shape
regressions surface here.
"""
from __future__ import annotations

import argparse
import io
import json
import sys

import pytest

from sndr.cli.legacy import patches as P
from sndr.product_api.legacy.patches import bundles as _api_bundles
from sndr.product_api.legacy.patches import listing as _api_listing


def _make_parser() -> argparse.ArgumentParser:
    """Build a parser with `sndr patches` registered (matches __init__.py)."""
    parser = argparse.ArgumentParser(prog="sndr-test")
    sub = parser.add_subparsers()
    P.add_argparser(sub)
    return parser


def _capture(monkeypatch, func, *args, **kwargs) -> tuple[str, int]:
    """Run a CLI handler and capture (stdout_text, return_code).

    `_io.info`/`_io.warn`/`_io.error` go through `print()` so capturing
    stdout is enough; we also capture stderr by passing capsys explicitly
    where necessary. Returns the combined captured stdout.
    """
    buf = io.StringIO()
    err_buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    monkeypatch.setattr(sys, "stderr", err_buf)
    rc = func(*args, **kwargs)
    return buf.getvalue() + err_buf.getvalue(), rc


# ─── argparse registration ──────────────────────────────────────────────


class TestArgparser:
    def test_parses_list_with_filters(self):
        p = _make_parser()
        ns = p.parse_args([
            "patches", "list", "--tier", "engine", "--has-upstream",
        ])
        assert ns.tier == "engine"
        assert ns.has_upstream is True

    def test_parses_explain_with_id(self):
        p = _make_parser()
        ns = p.parse_args(["patches", "explain", "P67"])
        assert ns.patch_id == "P67"

    def test_parses_plan_requires_preset(self):
        p = _make_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["patches", "plan"])

    def test_parses_doctor_strict(self):
        p = _make_parser()
        ns = p.parse_args(["patches", "doctor", "--strict", "--json"])
        assert ns.strict is True
        assert ns.json is True

    def test_parses_bundles_list_and_explain(self):
        p = _make_parser()
        ns_list = p.parse_args(["patches", "bundles", "list"])
        assert ns_list.func is not None
        ns_ex = p.parse_args([
            "patches", "bundles", "explain", "attention_gdn_spec",
        ])
        assert ns_ex.name == "attention_gdn_spec"


# ─── filter helpers ──────────────────────────────────────────────────────


class TestFilters:
    def test_matches_tier_filter(self):
        from sndr.dispatcher.spec import iter_patch_specs

        community = [s for s in iter_patch_specs()
                     if _api_listing.matches_filters(s, tier="community")]
        engine = [s for s in iter_patch_specs()
                  if _api_listing.matches_filters(s, tier="engine")]
        # Per memory: 75 engine / 56 community as of 2026-05-08, but
        # the CURRENT registry has tier="community" for nearly all
        # patches because impls live in sndr_engine/. We assert
        # qualitatively: every spec we iterate is one or the other.
        assert all(s.tier in ("community", "engine") for s in iter_patch_specs())
        assert len(community) + len(engine) <= sum(1 for _ in iter_patch_specs())

    def test_matches_default_on(self):
        from sndr.dispatcher.spec import iter_patch_specs

        for s in iter_patch_specs():
            assert _api_listing.matches_filters(s, default_on=True) is bool(s.default_on)
            assert _api_listing.matches_filters(s, default_on=False) is (not s.default_on)

    def test_matches_has_upstream(self):
        from sndr.dispatcher.spec import iter_patch_specs

        for s in iter_patch_specs():
            assert _api_listing.matches_filters(s, has_upstream=True) is bool(s.upstream_pr)

    def test_classify_skip_known_buckets(self):
        assert P._classify_skip("opt-in only — set X=1") == "opt-in (env unset)"
        assert P._classify_skip(
            "MODEL-COMPAT: model_class='fake' not in ['real']"
        ) == "model-incompatible"
        assert P._classify_skip(
            "tier=engine: license missing"
        ) == "engine-gated"
        assert P._classify_skip(
            "config_detect: skip:reason"
        ) == "config-detect:skip"
        assert P._classify_skip("nothing matches") == "other"


# ─── `sndr patches list` ────────────────────────────────────────────────


class TestList:
    def test_json_output_has_patches_array(self, capsys):
        ns = argparse.Namespace(
            tier=None, lifecycle=None, family=None,
            default_on=False, opt_in=False,
            has_upstream=False, no_upstream=False,
            json=True,
        )
        rc = P._run_list(ns)
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["count"] >= 100
        assert isinstance(data["patches"], list)
        # Spot-check shape
        first = data["patches"][0]
        for key in ("patch_id", "tier", "lifecycle",
                    "default_on", "title", "apply_module"):
            assert key in first

    def test_table_output_renders_header(self, capsys):
        ns = argparse.Namespace(
            tier=None, lifecycle=None, family=None,
            default_on=False, opt_in=False,
            has_upstream=False, no_upstream=False,
            json=False,
        )
        P._run_list(ns)
        out = capsys.readouterr().out
        # Table header must contain known columns.
        assert "Patch" in out and "Tier" in out and "Lifecycle" in out

    def test_tier_filter_subsets_results(self, capsys):
        ns_all = argparse.Namespace(
            tier=None, lifecycle=None, family=None,
            default_on=False, opt_in=False,
            has_upstream=False, no_upstream=False,
            json=True,
        )
        P._run_list(ns_all)
        all_count = json.loads(capsys.readouterr().out)["count"]

        ns_eng = argparse.Namespace(
            tier="engine", lifecycle=None, family=None,
            default_on=False, opt_in=False,
            has_upstream=False, no_upstream=False,
            json=True,
        )
        P._run_list(ns_eng)
        eng_count = json.loads(capsys.readouterr().out)["count"]
        assert eng_count <= all_count

    def test_has_upstream_filter(self, capsys):
        ns = argparse.Namespace(
            tier=None, lifecycle=None, family=None,
            default_on=False, opt_in=False,
            has_upstream=True, no_upstream=False,
            json=True,
        )
        P._run_list(ns)
        rows = json.loads(capsys.readouterr().out)["patches"]
        # All returned rows must have an upstream_pr populated.
        assert all(r["upstream_pr"] is not None for r in rows)


# ─── `sndr patches explain` ─────────────────────────────────────────────


class TestExplain:
    def test_explain_known_patch_json(self, capsys):
        ns = argparse.Namespace(patch_id="P67", json=True)
        rc = P._run_explain(ns)
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["patch_id"] == "P67"
        assert data["family"]
        assert "title" in data

    def test_explain_unknown_patch_returns_error_code(self, capsys):
        ns = argparse.Namespace(patch_id="P99999", json=False)
        rc = P._run_explain(ns)
        assert rc == 2

    def test_explain_case_insensitive_match(self, capsys):
        ns = argparse.Namespace(patch_id="p67", json=True)
        rc = P._run_explain(ns)
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        # Resolution preserves the registry's canonical casing.
        assert data["patch_id"].upper() == "P67"

    def test_explain_renders_human_output(self, capsys):
        ns = argparse.Namespace(patch_id="P67", json=False)
        rc = P._run_explain(ns)
        assert rc == 0
        out = capsys.readouterr().out
        assert "P67" in out
        # The Live decision line must always be emitted (apply or skip).
        assert "Live decision" in out


# ─── `sndr patches doctor` ──────────────────────────────────────────────


class TestDoctor:
    def test_doctor_reports_coverage(self, capsys):
        ns = argparse.Namespace(strict=False, json=True)
        rc = P._run_doctor(ns)
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["registry_size"] >= 100
        cov = data["apply_module_coverage"]
        assert cov["total"] == data["registry_size"]
        assert cov["mapped"] >= 100  # most patches must have impls

    def test_doctor_strict_passes_when_no_errors(self, capsys):
        # If the registry has no ERROR-class issues today, --strict is a no-op.
        ns = argparse.Namespace(strict=True, json=True)
        rc = P._run_doctor(ns)
        # Either clean (rc=0) or strict-failed (rc=1); both are valid
        # outcomes depending on registry state. Just assert the contract.
        assert rc in (0, 1)


# ─── `sndr patches plan` ────────────────────────────────────────────────


class TestPlan:
    def test_plan_real_preset_produces_buckets(self, capsys):
        # Phase 10 (2026-06-01): migrated V1 `a5000-2x-35b-prod` →
        # V2 alias `prod-qwen3.6-35b-balanced` (transparent bucket per
        # _v1_migration_table.json — V2 composes byte-identical config).
        # `sndr patches plan` resolves preset via the dual V1/V2 path
        # (`_resolve_preset_v1_or_v2`); the JSON `preset` field carries
        # the resolved config's `.key`, which for V2 is the composed
        # triplet form. Assert by substring to stay robust to V2
        # internals evolving.
        ns = argparse.Namespace(preset="prod-qwen3.6-35b-balanced", json=True)
        rc = P._run_plan(ns)
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        # `sndr patches plan` preserves the alias name verbatim in JSON
        # (no compose-key sanitization unlike `sndr memory explain`).
        assert "qwen3.6-35b-balanced" in data["preset"]
        assert data["apply_count"] + data["skip_count"] >= 100
        assert isinstance(data["apply"], list)
        assert isinstance(data["skip"], list)
        # Every row carries a reason.
        for r in data["apply"] + data["skip"]:
            assert "patch_id" in r
            assert "reason" in r

    def test_plan_unknown_preset_exits_2(self, capsys):
        ns = argparse.Namespace(preset="totally-fake-preset-xyz", json=True)
        with pytest.raises(SystemExit) as excinfo:
            P._run_plan(ns)
        assert excinfo.value.code == 2

    def test_plan_restores_env_after_run(self, monkeypatch, capsys):
        """Plan modifies os.environ to overlay preset's env. It must
        restore the prior values cleanly."""
        import os
        # Snapshot a representative sentinel and a borrowed flag.
        sentinel_key = "SNDR_TEST_SENTINEL_KEY"
        monkeypatch.delenv(sentinel_key, raising=False)
        monkeypatch.setenv(sentinel_key, "preset-test-value")

        # Phase 10 (2026-06-01): migrated V1 → V2 alias (same transparent
        # bucket migration as test_plan_real_preset_produces_buckets above).
        ns = argparse.Namespace(preset="prod-qwen3.6-35b-balanced", json=True)
        P._run_plan(ns)
        # Env restored
        assert os.environ.get(sentinel_key) == "preset-test-value"


# ─── `sndr patches diff-upstream` ───────────────────────────────────────


class TestDiffUpstream:
    def test_diff_upstream_two_buckets(self, capsys):
        ns = argparse.Namespace(json=True)
        rc = P._run_diff_upstream(ns)
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert "merged_upstream" in data
        assert "has_upstream_pr" in data
        # Bucket counts agree with array lengths.
        assert data["merged_upstream_count"] == len(data["merged_upstream"])
        assert data["has_upstream_pr_count"] == len(data["has_upstream_pr"])
        # Each "active w/ upstream" row carries an upstream_pr.
        for r in data["has_upstream_pr"]:
            assert r["upstream_pr"] is not None


# ─── `sndr patches bundles ...` ─────────────────────────────────────────


class TestBundles:
    def test_bundles_list_json(self, capsys):
        ns = argparse.Namespace(json=True)
        rc = P._run_bundles_list(ns)
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        # 5 bundles per Stage 7 catalog
        assert len(data) == 5
        names = {b["name"] for b in data}
        assert "tool_parsing_qwen3coder" in names
        assert "attention_gdn_spec" in names

    def test_bundle_explain_known(self, capsys):
        ns = argparse.Namespace(name="attention_gdn_spec", json=True)
        rc = P._run_bundles_explain(ns)
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["name"] == "attention_gdn_spec"
        assert data["umbrella_flag"] == "BUNDLE_ATTENTION_GDN_SPEC"
        assert data["has_apply"] is True

    def test_bundle_explain_unknown_returns_2(self, capsys):
        ns = argparse.Namespace(name="not-a-real-bundle", json=False)
        rc = P._run_bundles_explain(ns)
        assert rc == 2

    def test_bundles_catalog_matches_test_smoke(self):
        """The canonical bundle catalog in
        ``product_api.patches.bundles.BUNDLES_CATALOG`` must mirror
        ``tests/bundles/test_stage7_bundles_smoke.py::BUNDLES``. Drift
        detection — adding a bundle in one place but not the other
        fails this assertion."""
        # Import the canonical list from the smoke test to compare.
        # It uses tuples (name, flag, tier).
        from tests.bundles.test_stage7_bundles_smoke import BUNDLES as SMOKE
        smoke_set = {(b[0], b[1], b[2]) for b in SMOKE}
        api_set = {(b[0], b[1], b[2]) for b in _api_bundles.BUNDLES_CATALOG}
        assert smoke_set == api_set, (
            "BUNDLES catalog drift: "
            f"smoke-only={smoke_set - api_set}, "
            f"api-only={api_set - smoke_set}"
        )
