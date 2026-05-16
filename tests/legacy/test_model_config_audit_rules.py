# SPDX-License-Identifier: Apache-2.0
"""TDD for audit_rules.py — exhaustive 16-rule database."""
from __future__ import annotations

import pytest

from vllm.sndr_core.model_configs import (
    ModelConfig, HardwareSpec, SpecDecodeConfig,
)
from vllm.sndr_core.model_configs.audit_rules import audit, RULES


def _base_cfg(**overrides):
    """Minimal ModelConfig for testing audit rules."""
    defaults = dict(
        key="test-cfg", title="Test", description="Test",
        schema_version=1, maintainer="test",
        model_path="/models/Qwen3.6-35B-A3B-FP8",
        hardware=HardwareSpec(gpu_match_keys=["rtx a5000"], n_gpus=2,
                              min_vram_per_gpu_mib=22000),
        kv_cache_dtype=None,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
        max_num_seqs=2,
        spec_decode=None,
        genesis_env={},
        vllm_pin_required=None,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


class TestRulesDB:
    def test_at_least_19_rules(self):
        # 2026-05-06: added R-019 unresolved ${var} mounts
        assert len(RULES) >= 19

    def test_all_rules_have_id_and_title(self):
        for r in RULES:
            assert r.rule_id and r.rule_id.startswith("R-")
            assert r.title


# ─── R-001: P98 for TQ k8v4 + hybrid ──────────────────────────────────


class TestR001:
    def test_hybrid_tq_without_p98_flagged(self):
        cfg = _base_cfg(
            model_path="/models/Qwen3.6-27B-int4-AutoRound",
            kv_cache_dtype="turboquant_k8v4",
            genesis_env={},
        )
        issues = audit(cfg)
        assert any("R-001" in i[0] for i in issues)

    def test_hybrid_tq_with_p98_clean(self):
        cfg = _base_cfg(
            model_path="/models/Qwen3.6-27B-int4-AutoRound",
            kv_cache_dtype="turboquant_k8v4",
            genesis_env={"GENESIS_ENABLE_P98": "1"},
        )
        issues = audit(cfg)
        assert not any("R-001" in i[0] for i in issues)

    def test_dense_moe_fp8_not_flagged(self):
        # 35B-A3B-FP8 is dense MoE, not hybrid GDN — R-001 should not fire
        cfg = _base_cfg(
            model_path="/models/Qwen3.6-35B-A3B-FP8",
            kv_cache_dtype="turboquant_k8v4",
            genesis_env={},
        )
        issues = audit(cfg)
        assert not any("R-001" in i[0] for i in issues)


# ─── R-005: PN59 for long-ctx hybrid ──────────────────────────────────


class TestR005:
    def test_long_ctx_hybrid_without_pn59_flagged(self):
        cfg = _base_cfg(
            model_path="/models/Qwen3.6-27B-int4-AutoRound",
            max_model_len=131072,
        )
        issues = audit(cfg)
        assert any("R-005" in i[0] for i in issues)

    def test_short_ctx_hybrid_clean(self):
        cfg = _base_cfg(
            model_path="/models/Qwen3.6-27B-int4-AutoRound",
            max_model_len=32768,
        )
        issues = audit(cfg)
        assert not any("R-005" in i[0] for i in issues)


# ─── R-009: prefix-caching DANGER ─────────────────────────────────────


class TestR009:
    def test_prefix_caching_on_hybrid_tq_blocks(self):
        cfg = _base_cfg(
            model_path="/models/Qwen3.6-27B-int4-AutoRound",
            kv_cache_dtype="turboquant_k8v4",
            genesis_env={"GENESIS_ENABLE_P98": "1"},
        )
        cfg.vllm_extra_args = ["--enable-prefix-caching"]
        issues = audit(cfg)
        flagged = [i for i in issues if i[0] == "R-009"]
        assert len(flagged) == 1
        assert flagged[0][1] == "error"  # severity error


# ─── R-010: 27B + TQ + cudagraph FULL ─────────────────────────────────


class TestR010:
    def test_27b_tq_without_validated_pin_warns(self):
        """R-010 v2: 27B INT4 + TQ k8v4 without a validated genesis_pin AND
        without enforce_eager fallback → warning (P67 non-pow-2 GQA fix
        may be absent on older pins).
        """
        cfg = _base_cfg(
            model_path="/models/Qwen3.6-27B-int4-AutoRound",
            kv_cache_dtype="turboquant_k8v4",
            genesis_env={"GENESIS_ENABLE_P98": "1"},
        )
        cfg.enforce_eager = False
        cfg.genesis_pin = "deadbeef"   # not in known-good list
        cfg.vllm_extra_args = []
        issues = audit(cfg)
        assert any(i[0] == "R-010" for i in issues)

    def test_27b_tq_with_validated_genesis_pin_clean(self):
        """R-010 v2 (2026-05-05): with genesis_pin 991dc1a+ the P67 split-M
        non-pow-2 GQA fix is in place — vllm default cudagraph_mode
        FULL_AND_PIECEWISE works fine, no override needed.
        """
        cfg = _base_cfg(
            model_path="/models/Qwen3.6-27B-int4-AutoRound",
            kv_cache_dtype="turboquant_k8v4",
            genesis_env={"GENESIS_ENABLE_P98": "1"},
        )
        cfg.enforce_eager = False
        cfg.genesis_pin = "991dc1a"
        issues = audit(cfg)
        assert not any(i[0] == "R-010" for i in issues)


# ─── R-011: typo in env name ──────────────────────────────────────────


class TestR011:
    def test_typo_env_flagged(self):
        cfg = _base_cfg(
            genesis_env={"GENESIS_ENABLE_PXX9": "1"},  # fake patch
        )
        issues = audit(cfg)
        flagged = [i for i in issues if i[0] == "R-011"]
        assert len(flagged) == 1
        assert "PXX9" in flagged[0][3]

    def test_known_env_clean(self):
        cfg = _base_cfg(
            genesis_env={
                "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL": "1",
                "GENESIS_P67_NUM_KV_SPLITS": "32",  # tunable, allow
            },
        )
        issues = audit(cfg)
        assert not any(i[0] == "R-011" for i in issues)


# ─── R-013: vllm_pin must be in allowlist ─────────────────────────────


class TestR013:
    def test_unknown_pin_flagged(self):
        cfg = _base_cfg(vllm_pin_required="0.99.99-fake")
        issues = audit(cfg)
        assert any(i[0] == "R-013" for i in issues)

    def test_known_pin_clean(self):
        cfg = _base_cfg(vllm_pin_required="0.20.2rc1.dev9+g01d4d1ad3")
        issues = audit(cfg)
        assert not any(i[0] == "R-013" for i in issues)


# ─── Smoke: builtin configs are clean ─────────────────────────────────


class TestBuiltinConfigsClean:
    def test_a5000_2x_35b_prod_audit_clean_or_info_only(self):
        from vllm.sndr_core.model_configs import get
        cfg = get("a5000-2x-35b-prod")
        assert cfg is not None
        issues = audit(cfg)
        # No errors; warnings allowed
        errors = [i for i in issues if i[1] == "error"]
        assert errors == [], f"Builtin 35B config has errors: {errors}"

    def test_a5000_2x_27b_int4_balanced_no_errors(self):
        from vllm.sndr_core.model_configs import get
        cfg = get("a5000-2x-27b-int4-tested")
        issues = audit(cfg)
        errors = [i for i in issues if i[1] == "error"]
        assert errors == [], f"Builtin 27B config has errors: {errors}"


# ─── R-018: hybrid mamba REQUEST_CONSTANT capacity ────────────────────


class TestR018:
    def test_dense_moe_fp8_skipped(self):
        """35B-A3B-FP8 is dense MoE, no mamba — R-018 should not fire."""
        cfg = _base_cfg(
            model_path="/models/Qwen3.6-35B-A3B-FP8",
            max_num_seqs=64,  # extreme value, but no mamba → safe
        )
        issues = audit(cfg)
        assert not any(i[0] == "R-018" for i in issues)

    def test_hybrid_27b_safe_max_num_seqs_clean(self):
        """Sane 27B hybrid config (max_num_seqs=4) on 2×24GB → clean."""
        cfg = _base_cfg(
            model_path="/models/Qwen3.6-27B-int4-AutoRound",
            hardware=HardwareSpec(gpu_match_keys=["rtx a5000"], n_gpus=2,
                                  min_vram_per_gpu_mib=22000),
            gpu_memory_utilization=0.92,
            max_num_seqs=4,
        )
        issues = audit(cfg)
        assert not any(i[0] == "R-018" for i in issues)

    def test_hybrid_27b_extreme_max_num_seqs_flagged(self):
        """max_num_seqs=32 hybrid on 24GB → ~8 GB mamba state alone, flagged."""
        cfg = _base_cfg(
            model_path="/models/Qwen3.6-27B-int4-AutoRound",
            hardware=HardwareSpec(gpu_match_keys=["rtx a5000"], n_gpus=2,
                                  min_vram_per_gpu_mib=22000),
            gpu_memory_utilization=0.92,
            max_num_seqs=32,
        )
        issues = audit(cfg)
        flagged = [i for i in issues if i[0] == "R-018"]
        assert len(flagged) == 1
        assert flagged[0][1] == "warning"
        assert "max_num_seqs=32" in flagged[0][3]

    def test_hybrid_1x_24gb_low_util_flagged(self):
        """Single 24GB with max_num_seqs=8 + low util → tight."""
        cfg = _base_cfg(
            model_path="/models/Qwen3.6-27B-int4-AutoRound",
            hardware=HardwareSpec(gpu_match_keys=["rtx a5000"], n_gpus=1,
                                  min_vram_per_gpu_mib=22000),
            gpu_memory_utilization=0.85,
            max_num_seqs=28,  # 28 × 250 = 7000 MiB > 30% × 18700 = 5610
        )
        issues = audit(cfg)
        assert any(i[0] == "R-018" for i in issues)

    def test_hybrid_dflash_true_clean(self):
        """27B dflash variants are also hybrid — same rule applies."""
        cfg = _base_cfg(
            model_path="/models/Qwen3.6-27B-dflash-true",
            hardware=HardwareSpec(gpu_match_keys=["rtx a5000"], n_gpus=2,
                                  min_vram_per_gpu_mib=22000),
            gpu_memory_utilization=0.90,
            max_num_seqs=2,
        )
        issues = audit(cfg)
        assert not any(i[0] == "R-018" for i in issues)


# ─── Smoke: every builtin config must pass R-018 (no mamba overflow) ──


class TestR018BuiltinClean:
    def test_all_builtin_configs_pass_R018(self):
        """All 8 shipped configs must be clean against R-018 — guards
        against accidentally regressing a curated config below safe budget."""
        from vllm.sndr_core.model_configs import get, list_keys
        for key in list_keys():
            cfg = get(key)
            issues = audit(cfg)
            r018 = [i for i in issues if i[0] == "R-018"]
            assert r018 == [], (
                f"Builtin config {key!r} regressed against R-018 "
                f"(hybrid mamba capacity): {r018[0][3] if r018 else ''}"
            )


# ─── R-018 Phase D empirical-bake — uses model-specific measured value ──


class TestR018EmpiricalBake:
    def _hybrid_cfg_with_baked(self, baked_mib, max_num_seqs=4, util=0.92):
        from vllm.sndr_core.model_configs import ReferenceMetrics
        from dataclasses import replace
        ref = None
        if baked_mib is not None:
            ref = ReferenceMetrics(
                measured_at="2026-05-07T00:00:00Z",
                bench_method="test",
                long_gen_sustained_tps=100.0,
                long_gen_mean_lat_s=10.0,
                tool_call_score="10/10",
                stability_mean_s=2.0,
                stability_cv_pct=1.0,
                vram_used_mib_per_gpu=[20000, 20000],
                vram_total_mib=40000,
                genesis_pin="test",
                vllm_pin="test",
                mamba_state_mib_per_request=baked_mib,
            )
        return _base_cfg(
            model_path="/models/Qwen3.6-27B-int4-AutoRound",
            hardware=HardwareSpec(gpu_match_keys=["rtx a5000"], n_gpus=2,
                                  min_vram_per_gpu_mib=22000),
            gpu_memory_utilization=util,
            max_num_seqs=max_num_seqs,
            reference_metrics=ref,
        )

    def test_empirical_value_used_when_set(self):
        """Empirical-bake 100 MiB allows bigger max_num_seqs than 250 heuristic."""
        # max_num_seqs=50 with 100 MiB baked = 5000 MiB; 30% × 20240 = 6072 → CLEAN
        cfg = self._hybrid_cfg_with_baked(baked_mib=100.0, max_num_seqs=50)
        issues = audit(cfg)
        r018 = [i for i in issues if i[0] == "R-018"]
        # With heuristic 250: 50 × 250 = 12500 MiB → would FIRE
        # With empirical 100: 50 × 100 = 5000 MiB → CLEAN
        assert r018 == [], "empirical 100 MiB should permit max_num_seqs=50"

    def test_empirical_value_overrides_heuristic_to_fire(self):
        """If empirical is HIGHER than heuristic, R-018 fires SOONER."""
        # max_num_seqs=10 with 700 MiB baked = 7000 MiB > 6072 cap → FIRE
        cfg = self._hybrid_cfg_with_baked(baked_mib=700.0, max_num_seqs=10)
        issues = audit(cfg)
        r018 = [i for i in issues if i[0] == "R-018"]
        assert len(r018) == 1
        assert "empirical" in r018[0][3]
        # With heuristic 250: 10 × 250 = 2500 MiB → would NOT fire
        # With empirical 700: 10 × 700 = 7000 MiB → FIRES (good — model-specific)

    def test_message_indicates_source(self):
        """Message should clearly distinguish heuristic vs empirical."""
        cfg = self._hybrid_cfg_with_baked(baked_mib=600.0, max_num_seqs=20)
        issues = audit(cfg)
        r018 = [i for i in issues if i[0] == "R-018"]
        assert len(r018) == 1
        assert "(empirical)" in r018[0][3]

    def test_no_baked_falls_back_to_heuristic(self):
        """When `reference_metrics.mamba_state_mib_per_request` is None,
        R-018 uses the 250 MiB heuristic (preserves prior behavior)."""
        cfg = self._hybrid_cfg_with_baked(baked_mib=None, max_num_seqs=32)
        issues = audit(cfg)
        r018 = [i for i in issues if i[0] == "R-018"]
        # max_num_seqs=32 with heuristic 250 = 8000 MiB > 6072 → FIRE
        assert len(r018) == 1
        # Source label should reflect heuristic fallback (no "empirical")
        assert "(heuristic)" in r018[0][3] or "empirical" not in r018[0][3]

    def test_zero_baked_treated_as_unset(self):
        """0 or negative baked value → fall back to heuristic."""
        cfg = self._hybrid_cfg_with_baked(baked_mib=0.0, max_num_seqs=4)
        issues = audit(cfg)
        # max_num_seqs=4 × 250 heuristic = 1000 MiB < 6072 → CLEAN regardless
        assert not any(i[0] == "R-018" for i in issues)

    def test_no_reference_metrics_attr_safe(self):
        """If cfg has no reference_metrics attribute at all → heuristic."""
        cfg = self._hybrid_cfg_with_baked(baked_mib=None, max_num_seqs=4)
        # Force no ref_metrics
        from dataclasses import replace
        cfg = replace(cfg, reference_metrics=None)
        issues = audit(cfg)
        # Should not crash; heuristic applies
        # max_num_seqs=4 × 250 = 1000 MiB → CLEAN
        assert not any(i[0] == "R-018" for i in issues)


# ─── R-019: unresolved ${var} mounts (W-runtime 2026-05-06) ───────────


class TestR019_SymbolicMounts:
    """Audit catches symbolic mounts referencing vars not in host.yaml."""

    def _cfg_with_mounts(self, mounts):
        from vllm.sndr_core.model_configs.schema import DockerConfig
        return _base_cfg(
            docker=DockerConfig(
                image="vllm/vllm-openai:nightly",
                container_name="test-container",
                port=8000,
                mounts=mounts,
            ),
        )

    def test_no_symbolic_mounts_no_warning(self):
        """Configs with absolute mounts (legacy builtin) → R-019 quiet."""
        cfg = self._cfg_with_mounts([
            "/data/models:/models:ro",
            "/etc/foo:/etc/foo:ro",
        ])
        issues = audit(cfg)
        assert not any(i[0] == "R-019" for i in issues)

    def test_symbolic_var_present_in_host_yaml_no_warning(self, tmp_path,
                                                           monkeypatch):
        """${models_dir} resolves via host.yaml → R-019 quiet."""
        host_yaml = tmp_path / "host.yaml"
        host_yaml.write_text("""
paths:
  models_dir: /data/models
""")
        monkeypatch.setenv("GENESIS_HOME", str(tmp_path))
        cfg = self._cfg_with_mounts(["${models_dir}:/models:ro"])
        issues = audit(cfg)
        assert not any(i[0] == "R-019" for i in issues), \
            f"R-019 false fire on resolvable symbolic mount: {issues}"

    def test_symbolic_var_missing_from_host_yaml_fires(self, tmp_path,
                                                         monkeypatch):
        """${unknown_var} not in host.yaml → R-019 fires with names."""
        host_yaml = tmp_path / "host.yaml"
        host_yaml.write_text("""
paths:
  models_dir: /data/models
""")
        monkeypatch.setenv("GENESIS_HOME", str(tmp_path))
        cfg = self._cfg_with_mounts([
            "${models_dir}:/models:ro",
            "${unknown_var}:/x:ro",
        ])
        issues = audit(cfg)
        r019 = [i for i in issues if i[0] == "R-019"]
        assert r019, f"R-019 should fire on unknown var; issues={issues}"
        assert "unknown_var" in r019[0][3]

    def test_host_yaml_absent_with_symbolic_mounts_fires(self, tmp_path,
                                                           monkeypatch):
        """No host.yaml + symbolic mounts → R-019 fires with helpful tip.

        The test must be hermetic — it cannot depend on whether the
        host running the test has a real `~/.sndr/host.yaml` on disk.
        Earlier versions used a `GENESIS_HOME` env override, but
        `load_host_config()` now reads a fixed path and ignores env,
        so we patch `load_host_config()` (returns an object with no
        paths) and `detect_paths()` (returns an empty dict) instead.
        """
        from vllm.sndr_core.model_configs import host as _host_mod

        class _EmptyHC:
            paths: dict = {}

        monkeypatch.setattr(_host_mod, "load_host_config",
                              lambda: _EmptyHC(), raising=False)
        monkeypatch.setattr(_host_mod, "detect_paths",
                              lambda: {}, raising=False)
        monkeypatch.setenv("GENESIS_HOME", str(tmp_path))
        cfg = self._cfg_with_mounts(["${models_dir}:/models:ro"])
        issues = audit(cfg)
        r019 = [i for i in issues if i[0] == "R-019"]
        assert r019, f"R-019 should fire when host.yaml absent; issues={issues}"
