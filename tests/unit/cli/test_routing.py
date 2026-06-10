# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr routing-table` — Phase 7.G4.WORKLOAD-GATE-POLICY.IMPLEMENT.

Contract pins (must NOT change without a coordinated consumer-side
schema upgrade):

  • schema_version == 1
  • length_detection.short_threshold_tokens == 256
  • fallback.default_K == 1
  • every routing_rule.preset_key references an existing preset
  • at most one default_for_family per model_family
  • all current Gemma 4 26B-A4B + 31B presets appear in `presets[]`
  • 31B multi-conc gap is EXPLICIT in coverage_gaps (not silent)
  • 26B-A4B rules cover single_stream + multi_conc short-structured
  • workload_class evaluation_order is the policy-locked tie-break
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from sndr.cli.legacy import routing


REPO_ROOT = Path(__file__).resolve().parents[3]


# ─── Library-level invariants ───────────────────────────────────────────


@pytest.fixture(scope="module")
def table():
    return routing.compute_routing_table()


class TestSchemaInvariants:
    def test_schema_version_is_1(self, table):
        assert table["schema_version"] == 1

    def test_short_threshold_is_256(self, table):
        assert table["length_detection"]["short_threshold_tokens"] == 256

    def test_caller_hint_wins(self, table):
        assert table["length_detection"]["caller_hint_wins"] is True

    def test_default_K_is_1(self, table):
        assert table["fallback"]["default_K"] == 1

    def test_required_top_level_keys(self, table):
        for required in (
            "schema_version", "generated_by", "generated_at",
            "presets", "routing_rules", "workload_class_detection",
            "concurrency_mode_detection", "length_detection",
            "fallback", "coverage_gaps",
        ):
            assert required in table, f"missing {required}"


class TestStructuralValidation:
    def test_emits_valid_table_per_schema(self, table):
        errors = routing.validate_against_schema(table)
        assert errors == [], (
            "validate_against_schema reported errors:\n" + "\n".join(errors)
        )


# ─── Preset coverage ────────────────────────────────────────────────────


class TestPresetCoverage:
    def test_26b_a4b_presets_all_present(self, table):
        keys = {p["preset_key"] for p in table["presets"]}
        for required_key in (
            "prod-gemma4-26b-default",
            "prod-gemma4-26b-mtp-k4",
            "prod-gemma4-26b-multiconc",
            "prod-gemma4-26b-multiconc-k1",
        ):
            assert required_key in keys, f"missing preset {required_key}"

    def test_31b_presets_all_present(self, table):
        keys = {p["preset_key"] for p in table["presets"]}
        for required_key in (
            "prod-gemma4-31b-tq-default",
            "prod-gemma4-31b-tq-mtp-structured-k4",
        ):
            assert required_key in keys, f"missing preset {required_key}"

    def test_every_preset_has_required_fields(self, table):
        required_fields = {
            "preset_key", "model", "served_model_name", "model_family",
            "spec_decode_K", "max_num_seqs", "role", "intended_workloads",
            "default_for_family",
        }
        for p in table["presets"]:
            missing = required_fields - set(p.keys())
            assert not missing, f"preset {p.get('preset_key')!r} missing {missing}"

    def test_at_most_one_default_per_family(self, table):
        counts: dict[str, int] = {}
        for p in table["presets"]:
            if p["default_for_family"]:
                counts[p["model_family"]] = counts.get(p["model_family"], 0) + 1
        for fam, count in counts.items():
            assert count <= 1, f"family {fam!r} has {count} default flags"


# ─── Routing rules ──────────────────────────────────────────────────────


class TestRoutingRules:
    def test_no_rule_references_missing_preset(self, table):
        keys = {p["preset_key"] for p in table["presets"]}
        bad = [r for r in table["routing_rules"]
               if r["preset_key"] not in keys]
        assert not bad, (
            f"rules reference missing presets: "
            f"{[(r['model_family'], r['preset_key']) for r in bad]}"
        )

    def test_every_rule_has_evidence(self, table):
        for r in table["routing_rules"]:
            assert r["evidence"], (
                f"rule {r['model_family']!r}→{r['preset_key']!r} has "
                f"empty evidence; the contract requires a bench citation"
            )

    def test_26b_a4b_single_stream_short_structured_rule_exists(self, table):
        match = [
            r for r in table["routing_rules"]
            if r["model_family"] == "gemma4_moe_26b_a4b"
            and r["preset_key"] == "prod-gemma4-26b-mtp-k4"
            and "single_stream" in r["when"].get("concurrency_mode", [])
            and "short" in r["when"].get("expected_output_length", [])
        ]
        assert match, "missing 26B-A4B single-stream short-structured rule"

    def test_26b_a4b_multiconc_short_structured_rule_exists(self, table):
        match = [
            r for r in table["routing_rules"]
            if r["model_family"] == "gemma4_moe_26b_a4b"
            and r["preset_key"] == "prod-gemma4-26b-multiconc"
            and "multi_conc" in r["when"].get("concurrency_mode", [])
            and "short" in r["when"].get("expected_output_length", [])
        ]
        assert match, "missing 26B-A4B multi-conc short-structured rule"

    def test_31b_single_stream_structured_rule_exists(self, table):
        match = [
            r for r in table["routing_rules"]
            if r["model_family"] == "gemma4_dense_31b"
            and r["preset_key"] == "prod-gemma4-31b-tq-mtp-structured-k4"
        ]
        assert match, "missing 31B single-stream structured rule"

    def test_multiconc_rules_precede_single_stream_for_same_workload(self, table):
        """First-match-wins requires the multi-conc 26B-A4B rule to
        appear BEFORE the single-stream sibling — otherwise a
        multi_conc request would erroneously match single_stream."""
        rules = table["routing_rules"]
        sf_idx = next(
            (i for i, r in enumerate(rules)
             if r["model_family"] == "gemma4_moe_26b_a4b"
             and r["preset_key"] == "prod-gemma4-26b-mtp-k4"),
            None,
        )
        mc_idx = next(
            (i for i, r in enumerate(rules)
             if r["model_family"] == "gemma4_moe_26b_a4b"
             and r["preset_key"] == "prod-gemma4-26b-multiconc"),
            None,
        )
        assert mc_idx is not None and sf_idx is not None
        assert mc_idx < sf_idx, (
            f"multi-conc rule (idx {mc_idx}) must precede "
            f"single-stream rule (idx {sf_idx}) for first-match-wins"
        )

    def test_no_free_chat_or_code_gen_rules_emitted(self, table):
        """Policy: K=1 default for free-chat / code / summarization.
        These workloads must NOT appear in routing_rules — falling
        through to default_for_family is the correct behavior."""
        for r in table["routing_rules"]:
            wc = r["when"].get("workload_class", [])
            for forbidden in ("free_chat", "code_gen", "summarization"):
                assert forbidden not in wc, (
                    f"rule emits {forbidden!r} — policy says K=1 default "
                    f"for these workloads; remove the rule"
                )


# ─── Default-for-family + fallback ──────────────────────────────────────


class TestDefaults:
    def test_26b_a4b_default_is_K1_no_mtp(self, table):
        matches = [
            p for p in table["presets"]
            if p["model_family"] == "gemma4_moe_26b_a4b"
            and p["default_for_family"]
        ]
        assert len(matches) == 1, f"expected 1 default for 26B-A4B, got {len(matches)}"
        assert matches[0]["spec_decode_K"] == 1, (
            f"26B-A4B default must be K=1, got K={matches[0]['spec_decode_K']}"
        )

    def test_31b_default_is_K1_no_mtp(self, table):
        matches = [
            p for p in table["presets"]
            if p["model_family"] == "gemma4_dense_31b"
            and p["default_for_family"]
        ]
        assert len(matches) == 1
        assert matches[0]["spec_decode_K"] == 1

    def test_only_families_with_rules_have_defaults(self, table):
        families_with_rules = {r["model_family"] for r in table["routing_rules"]}
        for p in table["presets"]:
            if p["default_for_family"]:
                assert p["model_family"] in families_with_rules, (
                    f"preset {p['preset_key']} (family "
                    f"{p['model_family']!r}) has default_for_family=True "
                    f"but no rule references that family — misleading"
                )


# ─── Coverage gaps (operator visibility) ────────────────────────────────


class TestCoverageGaps:
    def test_31b_multiconc_gap_is_explicit(self, table):
        """31B multi-conc has no measured rule. Operator-locked
        decision (IMPLEMENT.R §9.1): ship with the gap but make it
        explicit in the emitted JSON so consumers can surface it."""
        gaps = table["coverage_gaps"]
        multiconc_gap = [
            g for g in gaps
            if g["model_family"] == "gemma4_dense_31b"
            and "multi_conc" in g["missing_cell"]
        ]
        assert multiconc_gap, (
            "31B multi-conc gap must be explicit in coverage_gaps "
            "(operator decision per IMPLEMENT.R §9.1); got: "
            f"{[g['missing_cell'] for g in gaps]}"
        )

    def test_every_gap_cites_next_phase(self, table):
        for g in table["coverage_gaps"]:
            assert g.get("next_phase"), (
                f"gap {g!r} has no next_phase pointer — operators "
                f"can't act on it without one"
            )

    def test_every_gap_names_a_fallback_preset(self, table):
        for g in table["coverage_gaps"]:
            assert g.get("fallback_preset"), (
                f"gap {g!r} missing fallback_preset"
            )


# ─── Workload-class detection contract ──────────────────────────────────


class TestWorkloadClassDetection:
    def test_evaluation_order_locked(self, table):
        # Tool-call must be highest priority so it wins when both
        # `tools[]` and `response_format` are set (the natural OpenAI
        # tool-use shape).
        order = table["workload_class_detection"]["evaluation_order"]
        assert order[0] == "tool_call", (
            f"tool_call must be first in evaluation_order; got {order}"
        )
        assert "free_chat" in order
        # free_chat is the last-resort default — must be at the end.
        assert order[-1] == "free_chat"

    def test_short_threshold_in_classifier_matches_top_level(self, table):
        """The structured_json.short trigger must reference the same
        threshold as length_detection.short_threshold_tokens (no
        drift between the two specs)."""
        threshold = table["length_detection"]["short_threshold_tokens"]
        short_rule = table["workload_class_detection"]["classes"]["structured_json"]["subtype"]["short"]
        assert str(threshold) in short_rule, (
            f"structured_json.short trigger {short_rule!r} doesn't "
            f"reference top-level short_threshold_tokens={threshold}"
        )


# ─── CLI surface ────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sndr")
    sub = parser.add_subparsers()
    routing.add_argparser(sub)
    return parser


class TestCLIArgparse:
    def test_json_flag(self):
        ns = _build_parser().parse_args(["routing-table", "--json"])
        assert ns.json is True

    def test_out_flag(self):
        ns = _build_parser().parse_args(["routing-table", "--out", "/tmp/x.json"])
        assert ns.out == "/tmp/x.json"

    def test_validate_flag(self):
        ns = _build_parser().parse_args(["routing-table", "--validate"])
        assert ns.validate is True


class TestCLIExitCodes:
    def test_validate_exits_zero_on_clean_emit(self):
        result = subprocess.run(
            [sys.executable, "-m", "sndr.cli.legacy",
             "routing-table", "--validate"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"validate exit {result.returncode}; stderr:\n{result.stderr}"
        )

    def test_json_emit_is_parseable(self):
        result = subprocess.run(
            [sys.executable, "-m", "sndr.cli.legacy",
             "routing-table", "--json"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert payload["schema_version"] == 1


class TestEmitSize:
    """Sanity: keep the emitted JSON under 50 KiB so the aggregator
    can buffer it cheaply at startup (acceptance criterion A14 from
    IMPLEMENT.R §8)."""
    def test_emitted_json_under_50_kib(self, table):
        payload = json.dumps(table, indent=2)
        assert len(payload) < 50 * 1024, (
            f"emitted JSON is {len(payload)} bytes (>= 50 KiB); "
            f"either trim the content or raise the bound"
        )
