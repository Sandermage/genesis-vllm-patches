# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_config_keys.py` — §10.3 #4 / §6.7 canonical
env-key registry audit.

Walks committed V1/V2 YAML configs and verifies every Genesis/SNDR
env key declared anywhere lives in the canonical union (PATCH_REGISTRY
+ V2 model.patches + V1 genesis_env).
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_config_keys.py"


def _import():
    name = "_audit_config_keys_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestGatherYamls:
    def test_returns_v2_layered_corpus(self):
        # Phase 10 Step 4 (2026-06-01): V1 monolithic preset tier 100%
        # retired. _gather_yamls now returns ONLY V2 layered triplet
        # files (model/, hardware/, profile/) — no top-level V1
        # monoliths remain. The test previously named test_returns_v1_
        # and_v2 asserted V1 presence; renamed to reflect post-sunset
        # state.
        mod = _import()
        yamls = mod._gather_yamls()
        assert yamls, "expected to find committed YAMLs"
        paths = [p.relative_to(REPO_ROOT).as_posix() for p in yamls]
        # At least one V2 layered file under builtin/{model,hardware,profile}/
        assert any(
            "model_configs/builtin/model/" in p
            or "model_configs/builtin/hardware/" in p
            or "model_configs/builtin/profile/" in p
            for p in paths
        )
        # V1 monolithic at the root of builtin/ no longer exists.
        v1_monoliths = [
            p for p in paths
            if "model_configs/builtin/" in p
            and p.rsplit("/", 1)[0].endswith("builtin")
        ]
        assert v1_monoliths == [], (
            f"Phase 10 V1 sunset complete — no V1 monoliths expected; "
            f"got {v1_monoliths}"
        )

    def test_skips_presets(self):
        mod = _import()
        yamls = mod._gather_yamls()
        for fp in yamls:
            assert "presets" not in fp.parts, (
                f"audit-config-keys must skip preset alias files: {fp}"
            )


class TestAudit:
    def test_live_corpus_clean(self):
        """Every committed YAML's Genesis/SNDR env keys must be in the
        canonical registry — this is the gating contract."""
        mod = _import()
        report = mod.audit()
        assert report["total_unknown"] == 0, (
            f"unknown keys in committed corpus:\n"
            + "\n".join(
                f"  {r['yaml']}: {r['unknown_keys']}"
                for r in report["per_yaml"] if r["count"] > 0
            )
        )

    def test_report_shape(self):
        mod = _import()
        report = mod.audit()
        assert "canonical_count" in report
        assert "yaml_count" in report
        assert "total_unknown" in report
        assert "per_yaml" in report
        assert report["yaml_count"] > 0
        assert report["canonical_count"] > 100, (
            "canonical registry should have hundreds of keys"
        )


class TestLoaderKeyRules:
    """PN379 mirror (vllm#45196): static pre-deploy validation of
    loader-related engine keys in committed YAMLs — the same rules the
    runtime patch enforces at LoadConfig/DefaultModelLoader
    construction, applied at audit time so the multithread-load
    experiment config fails HERE before it fails on the rig."""

    def test_valid_experiment_block_clean(self):
        mod = _import()
        text = (
            "engine:\n"
            "  model_loader_extra_config:\n"
            "    enable_multithread_load: true\n"
            "    num_threads: 8\n"
        )
        assert mod.audit_loader_keys(text) == []

    def test_typo_strategy_flagged(self):
        mod = _import()
        text = "engine:\n  safetensors_load_strategy: eagre\n"
        findings = mod.audit_loader_keys(text)
        assert len(findings) == 1
        assert "safetensors_load_strategy" in findings[0]
        assert "eagre" in findings[0]

    @pytest.mark.parametrize(
        "ok", ["lazy", "eager", "prefetch", "torchao", "null", "~"]
    )
    def test_valid_strategies_clean(self, ok):
        mod = _import()
        assert mod.audit_loader_keys(
            f"engine:\n  safetensors_load_strategy: {ok}\n"
        ) == []

    def test_multithread_with_non_lazy_strategy_flagged(self):
        mod = _import()
        text = (
            "engine:\n"
            "  safetensors_load_strategy: eager\n"
            "  model_loader_extra_config:\n"
            "    enable_multithread_load: true\n"
        )
        findings = mod.audit_loader_keys(text)
        assert any("does not support" in f for f in findings)

    def test_multithread_with_lazy_strategy_clean(self):
        mod = _import()
        text = (
            "engine:\n"
            "  safetensors_load_strategy: lazy\n"
            "  model_loader_extra_config:\n"
            "    enable_multithread_load: true\n"
        )
        assert mod.audit_loader_keys(text) == []

    @pytest.mark.parametrize("bad", ["0", "-4", "'8'", '"8"', "2.5", "true"])
    def test_bad_num_threads_flagged(self, bad):
        mod = _import()
        findings = mod.audit_loader_keys(f"engine:\n  num_threads: {bad}\n")
        assert len(findings) == 1
        assert "num_threads" in findings[0]

    @pytest.mark.parametrize("bad", ["yes", "1", "'true'"])
    def test_non_bool_multithread_flagged(self, bad):
        mod = _import()
        findings = mod.audit_loader_keys(
            f"engine:\n  enable_multithread_load: {bad}\n"
        )
        assert len(findings) == 1
        assert "enable_multithread_load" in findings[0]

    def test_scalar_extra_config_flagged(self):
        mod = _import()
        findings = mod.audit_loader_keys(
            "engine:\n  model_loader_extra_config: not-a-dict\n"
        )
        assert len(findings) == 1
        assert "model_loader_extra_config" in findings[0]

    def test_comments_ignored(self):
        mod = _import()
        text = "engine: {}\n# num_threads: 0 would be rejected by PN379\n"
        assert mod.audit_loader_keys(text) == []

    def test_live_corpus_has_no_loader_violations(self):
        """Gating contract: committed YAMLs must satisfy the PN379
        loader rules (today none carry loader keys at all)."""
        mod = _import()
        report = mod.audit()
        assert report["total_loader_violations"] == 0, (
            "loader-key violations in committed corpus:\n"
            + "\n".join(
                f"  {r['yaml']}: {r['loader_violations']}"
                for r in report["per_yaml"] if r.get("loader_violations")
            )
        )

    def test_report_carries_loader_fields(self):
        mod = _import()
        report = mod.audit()
        assert "total_loader_violations" in report
        for r in report["per_yaml"]:
            assert "loader_violations" in r


class TestScriptCLI:
    def test_exits_zero(self):
        rc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert rc.returncode == 0, (
            f"audit-config-keys CLI failed:\n{rc.stdout}\n{rc.stderr}"
        )

    def test_json_mode(self):
        rc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert rc.returncode == 0
        import json
        out = json.loads(rc.stdout)
        assert out["total_unknown"] == 0
