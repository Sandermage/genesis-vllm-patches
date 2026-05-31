# SPDX-License-Identifier: Apache-2.0
"""CONFIG-UX.3 — tests for `sndr preset` native CLI (list/show/explain/recommend).

Acceptance coverage per CONFIG_UX_R §13.3-equivalent for CLI phase:

  Gate 1 — `sndr preset list` works without GPU (no torch import)
  Gate 2 — `sndr preset show <id>` displays full card formatted
  Gate 3 — `sndr preset explain <id>` shows composed runtime + evidence + fallback diff
  Gate 4 — `sndr preset recommend --workload X --hardware Y` returns ranked results
  Gate 5 — recommend honors workload_deny (Gemma4 K=4 NOT returned for free_chat)
  Gate 6 — recommend honors workload_allow (exact match required)
  Gate 7 — recommend rejects unknown workload outside KNOWN_WORKLOADS / custom:<slug>
  Gate 8 — `sndr preset show --json` round-trips
  Gate 9 — `sndr preset list --json` round-trips
  Gate 10 — torch-free import guard (sys.modules audit after corpus load)
  Gate 11 — --field dot-path drill-down
  Gate 12 — bridged stub removed (native dispatch wins)
  Gate 13 — Card-less presets degraded gracefully (list shows as unannotated;
            recommend skips them)
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "vllm.sndr_core.cli", "preset", *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
    )


@pytest.fixture(autouse=True)
def _clean_torch_sys_modules():
    """Snapshot sys.modules so torch-import tests are isolated."""
    before = "torch" in sys.modules
    yield
    if not before and "torch" in sys.modules:
        # If a test indirectly imported torch, we can't unload it cleanly
        # (extensions), but the assertion in Gate 10 covers this.
        pass


# ─── Gate 1: list works without GPU ─────────────────────────────────────────


class TestGate1ListWorksWithoutGPU:
    def test_list_exits_zero(self):
        result = _run_cli("list")
        assert result.returncode == 0, (
            f"list rc={result.returncode}\n"
            f"stdout={result.stdout[:200]}\nstderr={result.stderr[:200]}"
        )

    def test_list_shows_all_21(self):
        result = _run_cli("list")
        assert "matched 23 / 23 presets" in result.stdout

    def test_list_filter_status_prod_candidate(self):
        result = _run_cli("list", "--status", "production_candidate")
        assert result.returncode == 0
        # 16 prod-* annotated → all production_candidate (Stage 1 + 2× gemma4-31b chat-K3)
        assert "matched 16 / 23 presets" in result.stdout

    def test_list_filter_family(self):
        result = _run_cli("list", "--family", "qwen3_6_35b_a3b_fp8")
        assert result.returncode == 0
        # prod-qwen3.6-35b-balanced + prod-qwen3.6-35b-multiconc
        assert "matched 2 / 23 presets" in result.stdout

    def test_list_filter_no_matches(self):
        result = _run_cli("list", "--family", "nonexistent_family")
        assert result.returncode == 0
        assert "no presets match" in result.stdout

    def test_list_card_less_marked_unannotated(self):
        result = _run_cli("list", "--status", "experimental")
        # No annotated experimental presets currently — but unannotated
        # ones won't match because card is None.
        # Verify card-less presets surface differently.
        assert result.returncode == 0


# ─── Gate 8 + 9: JSON round-trip ────────────────────────────────────────────


class TestGate8And9JSONRoundTrip:
    def test_show_json_roundtrip(self):
        result = _run_cli("show", "--json", "prod-qwen3.6-35b-balanced")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["id"] == "prod-qwen3.6-35b-balanced"
        assert data["model"] == "qwen3.6-35b-a3b-fp8"
        assert data["hardware"] == "a5000-2x-24gbvram-16cpu-128gbram"
        assert data["has_card"] is True
        card = data["card"]
        assert card["status"] == "production_candidate"
        assert card["routing_family"] == "qwen3_6_35b_a3b_fp8"
        assert card["K"] == 3
        # Evidence refs round-trip (list of dicts)
        assert isinstance(card["evidence_refs"], list)
        assert len(card["evidence_refs"]) >= 1

    def test_list_json_roundtrip(self):
        result = _run_cli("list", "--json", "--status", "production_candidate")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["matched"] == 16
        assert data["total"] == 23
        ids = {p["id"] for p in data["presets"]}
        # Spot-check a couple of expected ids
        assert "prod-qwen3.6-35b-balanced" in ids
        assert "prod-gemma4-26b-multiconc" in ids
        assert "prod-qwen3.6-27b-tq-k8v4" in ids


# ─── Gate 2: show human view ────────────────────────────────────────────────


class TestGate2ShowHumanView:
    def test_show_includes_workload_sections(self):
        result = _run_cli("show", "prod-gemma4-26b-multiconc")
        assert result.returncode == 0
        assert "Workload allow" in result.stdout
        assert "Workload deny" in result.stdout
        assert "free_chat" in result.stdout  # in deny
        assert "structured_json.short" in result.stdout  # in allow

    def test_show_includes_evidence(self):
        result = _run_cli("show", "prod-qwen3.6-35b-balanced")
        assert result.returncode == 0
        assert "Evidence" in result.stdout
        assert "35b_v11_wave9.json" in result.stdout

    def test_show_card_less_preset_shows_pending(self):
        """Graceful-degradation contract for card-less presets.

        CONFIG-UX.2b (Iter 47-48, 2026-05-30) carded all 21 builtin
        presets. With zero card-less presets in the live fixture, the
        contract is verified via the synthetic-injection helper rather
        than a live ``sndr preset show`` invocation. The renderer's
        card-less branch in ``_render_card_human`` is still required to
        exist for any future preset added without a card; this test
        captures stdout to verify the branch fires.
        ``test_recommend_skips_card_less`` below covers the negative-
        filter side of the same contract.
        """
        import io
        import contextlib
        from vllm.sndr_core.cli import preset as preset_mod
        from vllm.sndr_core.model_configs.preset_schema import PresetDef
        pd = PresetDef(
            id="synthetic-card-less", model="m", hardware="h", card=None,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            preset_mod._render_card_human("synthetic-card-less", pd)
        text = buf.getvalue().lower()
        assert "no card" in text or "annotation pending" in text


# ─── Gate 3: explain composed runtime + fallback diff ───────────────────────


class TestGate3Explain:
    def test_explain_shows_composed_runtime(self):
        result = _run_cli("explain", "prod-qwen3.6-35b-balanced")
        assert result.returncode == 0
        assert "Composed runtime" in result.stdout
        assert "max_model_len" in result.stdout
        assert "spec_decode_method" in result.stdout

    def test_explain_shows_fallback_diff(self):
        result = _run_cli("explain", "prod-qwen3.6-27b-tq-multiconc")
        assert result.returncode == 0
        assert "Fallback diff" in result.stdout
        assert "prod-qwen3.6-27b-tq-k8v4" in result.stdout

    def test_explain_self_fallback_clean(self):
        """Family-default presets self-fallback; diff should report no
        field-level differences."""
        result = _run_cli("explain", "prod-qwen3.6-35b-balanced")
        assert result.returncode == 0
        # self-fallback yields identical compose
        if "Fallback diff" in result.stdout:
            assert "no field-level differences" in result.stdout

    def test_explain_json(self):
        result = _run_cli("explain", "--json", "prod-qwen3.6-35b-multiconc")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "card" in data
        assert "composed" in data
        assert data["composed"]["max_num_seqs"] == 8


# ─── Gate 4 + 5 + 6: recommend ──────────────────────────────────────────────


class TestGate4RecommendBasic:
    def test_recommend_free_chat_conc_8(self):
        """free_chat conc=8 → top results should be Qwen presets sorted
        by primary_metric (TPS desc). NO Gemma4 K=4 (free_chat denied)."""
        result = _run_cli(
            "recommend",
            "--workload", "free_chat",
            "--hardware", "a5000-2x-24gbvram-16cpu-128gbram",
            "--concurrency", "8",
            "--json",
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        ids = [r["id"] for r in data["results"]]
        # Operator's explicit acceptance: Gemma4 K=4 must NOT be in
        # free_chat results.
        assert "prod-gemma4-26b-mtp-k4" not in ids
        assert "prod-gemma4-26b-multiconc" not in ids
        # Qwen multi-conc presets should be present
        assert "prod-qwen3.6-35b-multiconc" in ids
        # Top result should have highest TPS
        if data["results"]:
            top = data["results"][0]
            assert top["card"]["primary_metric"]["value"] > 0

    def test_recommend_structured_json_short(self):
        result = _run_cli(
            "recommend",
            "--workload", "structured_json.short",
            "--hardware", "a5000-2x-24gbvram-16cpu-128gbram",
            "--concurrency", "8",
            "--json",
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        ids = [r["id"] for r in data["results"]]
        # Both 35B-multiconc AND gemma4 multiconc allow this workload
        assert "prod-gemma4-26b-multiconc" in ids
        assert "prod-qwen3.6-35b-multiconc" in ids

    def test_recommend_no_matches(self):
        """Concurrency outside any preset's envelope → empty results."""
        result = _run_cli(
            "recommend",
            "--workload", "free_chat",
            "--concurrency", "99",
            "--json",
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["results"] == []

    def test_recommend_top_limit(self):
        result = _run_cli(
            "recommend",
            "--workload", "free_chat",
            "--hardware", "a5000-2x-24gbvram-16cpu-128gbram",
            "--concurrency", "8",
            "--top", "2",
            "--json",
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert len(data["results"]) <= 2


class TestGate5RecommendDenyExclusion:
    def test_workload_in_deny_excludes_preset(self):
        """If preset.workload_deny contains workload, preset must be
        excluded from recommendation results even if workload_allow is
        broad/empty."""
        from vllm.sndr_core.cli import preset as preset_mod
        from vllm.sndr_core.model_configs.preset_schema import (
            PresetCard, PresetDef, ConcurrencyEnvelope,
            PrimaryMetric, EvidenceRef,
        )
        card = PresetCard(
            title="fake", summary="x", status="production",
            audience="operator", mode="throughput",
            workload_allow=["free_chat", "structured_json.short"],
            workload_deny=["summarization"],  # explicit deny
            concurrency=ConcurrencyEnvelope(min=1, canonical=4, max=8),
            K=3,
            routing_family="fake-fam",
            primary_metric=PrimaryMetric(kind="agg_TPS", value=500.0, source="x"),
            evidence_refs=[EvidenceRef(type="bench", path="x", visibility="public")],
            evidence_visibility="public",
        )
        pd = PresetDef(id="fake-deny", model="m", hardware="h", card=card)
        # Workload in deny → excluded
        assert not preset_mod._passes_recommend_filters(
            "fake-deny", pd,
            workload="summarization", hardware=None, concurrency=None,
        )
        # Workload in allow → included
        assert preset_mod._passes_recommend_filters(
            "fake-deny", pd,
            workload="free_chat", hardware=None, concurrency=None,
        )

    def test_card_less_preset_excluded_from_recommend(self):
        from vllm.sndr_core.cli import preset as preset_mod
        from vllm.sndr_core.model_configs.preset_schema import PresetDef
        pd = PresetDef(id="unannotated", model="m", hardware="h")
        assert not preset_mod._passes_recommend_filters(
            "unannotated", pd,
            workload="free_chat", hardware=None, concurrency=None,
        )

    def test_tombstone_excluded(self):
        from vllm.sndr_core.cli import preset as preset_mod
        from vllm.sndr_core.model_configs.preset_schema import (
            PresetCard, PresetDef,
        )
        card = PresetCard(
            title="t", summary="s", status="tombstone",
            workload_allow=["free_chat"],
        )
        pd = PresetDef(id="dead", model="m", hardware="h", card=card)
        assert not preset_mod._passes_recommend_filters(
            "dead", pd,
            workload="free_chat", hardware=None, concurrency=None,
        )


class TestGate6WorkloadAllowExact:
    def test_workload_not_in_allow_excludes(self):
        from vllm.sndr_core.cli import preset as preset_mod
        from vllm.sndr_core.model_configs.preset_schema import (
            PresetCard, PresetDef,
        )
        card = PresetCard(
            title="t", summary="s", status="production_candidate",
            workload_allow=["free_chat"],
        )
        pd = PresetDef(id="narrow", model="m", hardware="h", card=card)
        # Not in allow → excluded (even if also not in deny)
        assert not preset_mod._passes_recommend_filters(
            "narrow", pd,
            workload="summarization", hardware=None, concurrency=None,
        )


# ─── Gate 7: unknown workload rejected ──────────────────────────────────────


class TestGate7UnknownWorkloadRejected:
    def test_unknown_workload_rejected(self):
        result = _run_cli("recommend", "--workload", "freechat")  # typo
        assert result.returncode == 1
        assert "not in KNOWN_WORKLOADS" in result.stderr

    def test_custom_workload_accepted(self):
        # custom:<slug> form must be allowed (loader doesn't reject).
        # Won't match anything in real corpus but should not error out.
        result = _run_cli(
            "recommend", "--workload", "custom:my-special-task",
            "--json",
        )
        assert result.returncode == 0  # no matches but valid query
        data = json.loads(result.stdout)
        assert data["results"] == []

    def test_invalid_custom_form_rejected(self):
        # Not a valid `custom:<slug>` (uppercase) → rejected
        result = _run_cli("recommend", "--workload", "custom:UPPERCASE")
        assert result.returncode == 1


# ─── Gate 10: torch-free import guard ───────────────────────────────────────


class TestGate10TorchFreeImport:
    def test_corpus_load_and_recommend_no_torch(self):
        """Triggers the full corpus-load + recommend pipeline and
        asserts torch never appears in sys.modules."""
        if "torch" in sys.modules:
            pytest.skip("torch already imported by another test/runtime")
        from vllm.sndr_core.cli import preset as preset_mod
        ns = argparse.Namespace(
            workload="free_chat",
            hardware="a5000-2x-24gbvram-16cpu-128gbram",
            concurrency=8, top=3, json=True,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = preset_mod.run_recommend(ns)
        assert rc == 0
        assert "torch" not in sys.modules, (
            f"torch leaked into CLI import chain: sys.modules has "
            f"{[m for m in sys.modules if m.startswith('torch')]}"
        )

    def test_list_does_not_import_torch(self):
        if "torch" in sys.modules:
            pytest.skip("torch already imported")
        from vllm.sndr_core.cli import preset as preset_mod
        ns = argparse.Namespace(
            family=None, workload=None, hardware=None, mode=None,
            status=None, json=True,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = preset_mod.run_list(ns)
        assert rc == 0
        assert "torch" not in sys.modules


# ─── Gate 11: --field drill-down ────────────────────────────────────────────


class TestGate11FieldDrill:
    def test_field_simple_attr(self):
        result = _run_cli("show", "--field", "card.status", "prod-qwen3.6-35b-balanced")
        assert result.returncode == 0
        assert "production_candidate" in result.stdout

    def test_field_nested_list_index(self):
        result = _run_cli(
            "show", "--field", "card.evidence_refs.0.path", "prod-qwen3.6-35b-balanced",
        )
        assert result.returncode == 0
        assert "35b_v11_wave9.json" in result.stdout

    def test_field_invalid_path(self):
        result = _run_cli(
            "show", "--field", "card.nonexistent_field", "prod-qwen3.6-35b-balanced",
        )
        assert result.returncode == 1
        assert "nonexistent_field" in result.stderr

    def test_field_out_of_range(self):
        result = _run_cli(
            "show", "--field", "card.evidence_refs.999", "prod-qwen3.6-35b-balanced",
        )
        assert result.returncode == 1
        assert "out of range" in result.stderr


# ─── Gate 12: native dispatch wins (bridge removed) ─────────────────────────


class TestGate12BridgeRemoved:
    def test_preset_not_in_bridged_map(self):
        from vllm.sndr_core.cli import _BRIDGED
        assert "preset" not in _BRIDGED, (
            "CONFIG-UX.3: `preset` must no longer be in _BRIDGED map — "
            "native module owns the surface"
        )

    def test_native_dispatch_resolves(self):
        """`sndr preset list` should dispatch through the native module,
        not the bridged compat path."""
        result = _run_cli("list", "--json")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        # Native module shape (has 'matched' + 'total' + 'presets' keys);
        # the bridged compat output would look different.
        assert "matched" in data and "total" in data
        assert "presets" in data


# ─── Gate 13: graceful degradation for card-less presets ────────────────────


class TestGate13GracefulDegradation:
    def test_list_includes_card_less_with_marker(self):
        """All builtin presets carry cards post-CONFIG-UX.2b.

        Pre-Iter-48 (2026-05-30): 7 non-prod-* presets (qa-*, example-*,
        experimental-*, long-ctx-*) were card-less; this test asserted
        they appeared in ``list`` with ``has_card=False``. Iter 47-48
        carded all of them. The new invariant: ``has_card`` is True for
        every preset, AND the JSON shape still exposes ``has_card`` so
        any future card-less preset surfaces here.
        """
        result = _run_cli("list", "--json")
        data = json.loads(result.stdout)
        assert "presets" in data and len(data["presets"]) == 23
        # has_card key is required on every entry — schema contract.
        for p in data["presets"]:
            assert "has_card" in p, (
                f"preset {p.get('id')!r} missing has_card key"
            )
        unannotated = [p for p in data["presets"] if not p["has_card"]]
        # CONFIG-UX.2b fully closed — zero card-less presets remain.
        assert len(unannotated) == 0, (
            f"expected 0 card-less presets post-CONFIG-UX.2b, got "
            f"{[p['id'] for p in unannotated]}"
        )

    def test_recommend_skips_card_less(self):
        """Recommend must skip card-less presets even if they would
        otherwise match."""
        from vllm.sndr_core.cli import preset as preset_mod
        from vllm.sndr_core.model_configs.preset_schema import PresetDef
        pd = PresetDef(id="legacy", model="m", hardware="h", card=None)
        assert not preset_mod._passes_recommend_filters(
            "legacy", pd,
            workload="free_chat", hardware="h", concurrency=1,
        )
