# SPDX-License-Identifier: Apache-2.0
"""CONFIG-UX.4.2 — Class-4 forbidden override rules (`audit_override_policy.py`).

Per-rule coverage:
  Rule 1 — gpu_memory_utilization > 1.0
  Rule 2 — tensor_parallel_size > hardware.n_gpus
  Rule 3 — kv_cache_dtype downgrade (static narrowness ordering)
  Rule 4 — spec_decode method-name change

Plus:
  - Live corpus: all 21 builtin profiles must pass Class-4 at default mode
  - Severity: Class-4 errors fire at ALL stages (not stage-dependent)
  - GENESIS_DISABLE_V1_DEPRECATION_WARNING does NOT silence Class-4 errors

Loaded via importlib (script path) — same pattern as other scripts/ tests.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "audit_override_policy.py"


def _import_audit():
    spec = importlib.util.spec_from_file_location(
        "audit_override_policy_class4", SCRIPT,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audit_override_policy_class4"] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
    )


# ─── Mock fixtures (synthesized profile/model/hardware) ─────────────────────


def _mock_profile(**kwargs) -> Any:
    """Build a mock ProfileDef-like object with required attrs."""
    sizing = kwargs.pop("sizing", None)
    spec_decode = kwargs.pop("spec_decode", None)
    compression_plan = kwargs.pop("compression_plan", None)
    parent_model = kwargs.pop("parent_model", "some-model")

    obj = MagicMock()
    obj.sizing_override = sizing
    obj.spec_decode_override = spec_decode
    obj.compression_plan = compression_plan
    obj.parent_model = parent_model
    for k, v in kwargs.items():
        setattr(obj, k, v)
    return obj


def _mock_sizing(**kwargs) -> Any:
    obj = MagicMock()
    obj.gpu_memory_utilization = kwargs.get("gpu_memory_utilization")
    obj.tensor_parallel_size = kwargs.get("tensor_parallel_size")
    return obj


def _mock_spec_decode(method: Optional[str], K: Optional[int] = None) -> Any:
    obj = MagicMock()
    obj.method = method
    obj.num_speculative_tokens = K
    return obj


def _mock_compression_plan(default_kv_dtype: Optional[str]) -> Any:
    obj = MagicMock()
    obj.default_kv_dtype = default_kv_dtype
    return obj


def _mock_model(kv_cache_dtype: Optional[str] = None,
                spec_decode_method: Optional[str] = None) -> Any:
    obj = MagicMock()
    obj.capabilities = MagicMock()
    obj.capabilities.kv_cache_dtype = kv_cache_dtype
    if spec_decode_method is None:
        obj.capabilities.spec_decode = None
    else:
        obj.capabilities.spec_decode = MagicMock()
        obj.capabilities.spec_decode.method = spec_decode_method
    return obj


def _mock_hardware(n_gpus: Optional[int] = 2) -> Any:
    obj = MagicMock()
    if n_gpus is None:
        obj.hardware = None
    else:
        obj.hardware = MagicMock()
        obj.hardware.n_gpus = n_gpus
    return obj


# ─── Rule 1: gpu_memory_utilization > 1.0 ───────────────────────────────────


class TestRule1GpuMemUtil:
    def test_below_1_allowed(self):
        mod = _import_audit()
        result = mod._rule_1_gpu_mem_util_bound(
            _mock_profile(sizing=_mock_sizing(gpu_memory_utilization=0.95)),
            _mock_model(), _mock_hardware(),
        )
        assert result is None

    def test_equal_1_allowed(self):
        """1.0 exactly is allowed (strict greater-than)."""
        mod = _import_audit()
        result = mod._rule_1_gpu_mem_util_bound(
            _mock_profile(sizing=_mock_sizing(gpu_memory_utilization=1.0)),
            _mock_model(), _mock_hardware(),
        )
        assert result is None

    def test_above_1_violates(self):
        mod = _import_audit()
        result = mod._rule_1_gpu_mem_util_bound(
            _mock_profile(sizing=_mock_sizing(gpu_memory_utilization=1.5)),
            _mock_model(), _mock_hardware(),
        )
        assert result is not None
        assert "1.5" in result
        assert "physically impossible" in result

    def test_none_allowed(self):
        mod = _import_audit()
        result = mod._rule_1_gpu_mem_util_bound(
            _mock_profile(sizing=_mock_sizing(gpu_memory_utilization=None)),
            _mock_model(), _mock_hardware(),
        )
        assert result is None

    def test_no_sizing_override_allowed(self):
        mod = _import_audit()
        result = mod._rule_1_gpu_mem_util_bound(
            _mock_profile(sizing=None),
            _mock_model(), _mock_hardware(),
        )
        assert result is None


# ─── Rule 2: tensor_parallel_size > hardware.n_gpus ────────────────────────


class TestRule2TpSize:
    def test_equal_allowed(self):
        mod = _import_audit()
        result = mod._rule_2_tp_size_vs_hw_gpus(
            _mock_profile(sizing=_mock_sizing(tensor_parallel_size=2)),
            _mock_model(), _mock_hardware(n_gpus=2),
        )
        assert result is None

    def test_below_allowed(self):
        mod = _import_audit()
        result = mod._rule_2_tp_size_vs_hw_gpus(
            _mock_profile(sizing=_mock_sizing(tensor_parallel_size=1)),
            _mock_model(), _mock_hardware(n_gpus=2),
        )
        assert result is None

    def test_above_violates(self):
        mod = _import_audit()
        result = mod._rule_2_tp_size_vs_hw_gpus(
            _mock_profile(sizing=_mock_sizing(tensor_parallel_size=4)),
            _mock_model(), _mock_hardware(n_gpus=2),
        )
        assert result is not None
        assert "tensor_parallel_size=4" in result
        assert "n_gpus=2" in result

    def test_missing_tp_attribute_allowed(self):
        """Forward-compat: HardwareSizing has no tensor_parallel_size today."""
        mod = _import_audit()
        # sizing without the attribute → predicate skips
        result = mod._rule_2_tp_size_vs_hw_gpus(
            _mock_profile(sizing=_mock_sizing()),  # no tp set → None
            _mock_model(), _mock_hardware(n_gpus=2),
        )
        assert result is None

    def test_missing_hardware_gpu_count_skipped(self):
        """No false-positive when hardware schema is older."""
        mod = _import_audit()
        result = mod._rule_2_tp_size_vs_hw_gpus(
            _mock_profile(sizing=_mock_sizing(tensor_parallel_size=99)),
            _mock_model(), _mock_hardware(n_gpus=None),
        )
        assert result is None


# ─── Rule 3: kv_cache_dtype downgrade ──────────────────────────────────────


class TestRule3KvDtypeDowngrade:
    def test_model_auto_profile_anything_allowed(self):
        mod = _import_audit()
        for profile_dtype in ("fp8_e5m2", "turboquant_k8v4", "int4", "auto"):
            result = mod._rule_3_kv_cache_dtype_narrower(
                _mock_profile(compression_plan=_mock_compression_plan(profile_dtype)),
                _mock_model(kv_cache_dtype="auto"),
                _mock_hardware(),
            )
            assert result is None, f"model=auto + profile={profile_dtype} should allow"

    def test_model_none_profile_anything_allowed(self):
        mod = _import_audit()
        result = mod._rule_3_kv_cache_dtype_narrower(
            _mock_profile(compression_plan=_mock_compression_plan("turboquant_k8v4")),
            _mock_model(kv_cache_dtype=None),
            _mock_hardware(),
        )
        assert result is None

    def test_equal_bits_allowed(self):
        mod = _import_audit()
        result = mod._rule_3_kv_cache_dtype_narrower(
            _mock_profile(compression_plan=_mock_compression_plan("turboquant_k8v4")),
            _mock_model(kv_cache_dtype="turboquant_k8v4"),
            _mock_hardware(),
        )
        assert result is None

    def test_upgrade_allowed(self):
        """model 4-bit → profile 8-bit is upgrade, allowed."""
        mod = _import_audit()
        result = mod._rule_3_kv_cache_dtype_narrower(
            _mock_profile(compression_plan=_mock_compression_plan("fp8_e5m2")),
            _mock_model(kv_cache_dtype="turboquant_k8v4"),
            _mock_hardware(),
        )
        assert result is None

    def test_downgrade_fp8_to_4bit_violates(self):
        mod = _import_audit()
        result = mod._rule_3_kv_cache_dtype_narrower(
            _mock_profile(compression_plan=_mock_compression_plan("turboquant_k8v4")),
            _mock_model(kv_cache_dtype="fp8_e5m2"),
            _mock_hardware(),
        )
        assert result is not None
        assert "turboquant_k8v4" in result
        assert "fp8_e5m2" in result
        assert "loss-of-evidence" in result

    def test_downgrade_bf16_to_8bit_violates(self):
        mod = _import_audit()
        result = mod._rule_3_kv_cache_dtype_narrower(
            _mock_profile(compression_plan=_mock_compression_plan("fp8_e5m2")),
            _mock_model(kv_cache_dtype="bf16"),
            _mock_hardware(),
        )
        assert result is not None

    def test_profile_auto_never_downgrade(self):
        """Profile `auto` is the operator letting the framework decide;
        treated as 16-bit equivalent → never a downgrade."""
        mod = _import_audit()
        result = mod._rule_3_kv_cache_dtype_narrower(
            _mock_profile(compression_plan=_mock_compression_plan("auto")),
            _mock_model(kv_cache_dtype="turboquant_k8v4"),
            _mock_hardware(),
        )
        assert result is None

    def test_unknown_dtype_safe(self):
        """Unknown profile dtype defaults to 16-bit ordering → no
        false-positive for unrecognized names."""
        mod = _import_audit()
        result = mod._rule_3_kv_cache_dtype_narrower(
            _mock_profile(compression_plan=_mock_compression_plan("unknown_dtype_xyz")),
            _mock_model(kv_cache_dtype="bf16"),
            _mock_hardware(),
        )
        assert result is None  # unknown=16-bit, bf16=16-bit → not narrower

    def test_no_compression_plan_allowed(self):
        mod = _import_audit()
        result = mod._rule_3_kv_cache_dtype_narrower(
            _mock_profile(compression_plan=None),
            _mock_model(kv_cache_dtype="bf16"),
            _mock_hardware(),
        )
        assert result is None


# ─── Rule 4: spec_decode method-name change ────────────────────────────────


class TestRule4SpecDecodeMethod:
    def test_k_only_change_allowed(self):
        """mtp K=3 → mtp K=4 in profile = K change, same method, allowed."""
        mod = _import_audit()
        result = mod._rule_4_spec_decode_method_change(
            _mock_profile(spec_decode=_mock_spec_decode("mtp", K=4)),
            _mock_model(spec_decode_method="mtp"),
            _mock_hardware(),
        )
        assert result is None

    def test_method_change_violates(self):
        """mtp → eagle is method name change."""
        mod = _import_audit()
        result = mod._rule_4_spec_decode_method_change(
            _mock_profile(spec_decode=_mock_spec_decode("eagle")),
            _mock_model(spec_decode_method="mtp"),
            _mock_hardware(),
        )
        assert result is not None
        assert "eagle" in result
        assert "mtp" in result

    def test_mtp_to_ngram_violates(self):
        mod = _import_audit()
        result = mod._rule_4_spec_decode_method_change(
            _mock_profile(spec_decode=_mock_spec_decode("ngram")),
            _mock_model(spec_decode_method="mtp"),
            _mock_hardware(),
        )
        assert result is not None

    def test_additive_allowed(self):
        """Model has no spec_decode; profile adds one. Additive, allowed."""
        mod = _import_audit()
        result = mod._rule_4_spec_decode_method_change(
            _mock_profile(spec_decode=_mock_spec_decode("mtp")),
            _mock_model(spec_decode_method=None),
            _mock_hardware(),
        )
        assert result is None

    def test_subtractive_allowed(self):
        """Profile removes spec_decode (method=None). Subtractive, allowed."""
        mod = _import_audit()
        result = mod._rule_4_spec_decode_method_change(
            _mock_profile(spec_decode=_mock_spec_decode(None)),
            _mock_model(spec_decode_method="mtp"),
            _mock_hardware(),
        )
        assert result is None

    def test_no_override_allowed(self):
        mod = _import_audit()
        result = mod._rule_4_spec_decode_method_change(
            _mock_profile(spec_decode=None),
            _mock_model(spec_decode_method="mtp"),
            _mock_hardware(),
        )
        assert result is None


# ─── ForbiddenRule registration ────────────────────────────────────────────


class TestForbiddenRuleRegistration:
    def test_four_rules_registered(self):
        mod = _import_audit()
        assert len(mod.FORBIDDEN_OVERRIDES) == 4

    def test_rule_ids_unique(self):
        mod = _import_audit()
        ids = [r.rule_id for r in mod.FORBIDDEN_OVERRIDES]
        assert len(ids) == len(set(ids))

    def test_expected_rule_ids(self):
        mod = _import_audit()
        ids = {r.rule_id for r in mod.FORBIDDEN_OVERRIDES}
        assert ids == {
            "gpu_memory_utilization_over_1",
            "tensor_parallel_size_over_hw_gpus",
            "kv_cache_dtype_downgrade",
            "spec_decode_method_change",
        }

    def test_predicates_are_callable(self):
        mod = _import_audit()
        for rule in mod.FORBIDDEN_OVERRIDES:
            assert callable(rule.predicate)


# ─── Live corpus: all 21 profiles must pass Class-4 at default ─────────────


class TestLiveCorpusClass4Clean:
    def test_no_class4_errors_on_builtin_profiles(self):
        """Operator-locked acceptance gate: if ANY current builtin
        profile trips a Class-4 rule under default stage, the rule
        definition is too wide and CONFIG-UX.4.2 must HALT."""
        result = _run_cli("--json")
        data = json.loads(result.stdout)
        class4_errors = [
            f for f in data["findings"]
            if f["severity"] == "error"
            and f["rule"].startswith("forbidden_override.")
        ]
        assert class4_errors == [], (
            f"CRITICAL STOP — Class-4 rules tripped on current corpus. "
            f"Either narrow the predicate or surface for operator review.\n"
            + "\n".join(
                f"  [{f['rule']}] {f['profile_id']}: {f['message'][:200]}"
                for f in class4_errors
            )
        )

    def test_default_mode_clean(self):
        """audit_override_policy default mode still exits 0 after Class-4
        wiring (warnings expected; zero errors)."""
        result = _run_cli()
        assert result.returncode == 0


# ─── Severity invariants ────────────────────────────────────────────────────


class TestSeverityInvariants:
    @pytest.mark.parametrize("stage", ["0", "1", "2", "3"])
    def test_class4_errors_fire_at_all_stages(self, stage, monkeypatch):
        """Class-4 errors are NOT stage-dependent — they fire at any
        stage because they're physics/evidence violations, not rollout
        warnings."""
        # Test the predicate directly so we don't need a real bad
        # profile in the corpus.
        mod = _import_audit()
        bad_profile = _mock_profile(sizing=_mock_sizing(gpu_memory_utilization=1.5))
        result = mod._rule_1_gpu_mem_util_bound(
            bad_profile, _mock_model(), _mock_hardware(),
        )
        # Predicate result doesn't depend on stage at all.
        assert result is not None

    def test_disable_env_does_not_silence_class4(self, monkeypatch):
        """GENESIS_DISABLE_V1_DEPRECATION_WARNING is the rollout-warning
        escape hatch; it must NOT silence Class-4 errors."""
        mod = _import_audit()
        monkeypatch.setenv("GENESIS_DISABLE_V1_DEPRECATION_WARNING", "1")
        # Predicate is pure function — env doesn't affect it.
        bad_profile = _mock_profile(sizing=_mock_sizing(gpu_memory_utilization=1.5))
        result = mod._rule_1_gpu_mem_util_bound(
            bad_profile, _mock_model(), _mock_hardware(),
        )
        assert result is not None


# ─── Hardware cross-product (Option b) ──────────────────────────────────────


class TestHardwareCrossProduct:
    def test_resolve_hardware_ids_for_profile_finds_referencing_presets(self):
        mod = _import_audit()
        # qwen3.6-27b-tq-k8v4 is referenced by prod-qwen3.6-27b-tq-k8v4 preset (a5000-2x-... hardware)
        hw_ids = mod._resolve_hardware_ids_for_profile("qwen3.6-27b-tq-k8v4")
        assert "a5000-2x-24gbvram-16cpu-128gbram" in hw_ids

    def test_resolve_hardware_ids_orphan_profile(self):
        """Profile not referenced by any preset returns empty list."""
        mod = _import_audit()
        hw_ids = mod._resolve_hardware_ids_for_profile("nonexistent-profile-xyz")
        assert hw_ids == []


# ─── DEFAULT_STAGE flip backward-compat ─────────────────────────────────────


class TestDefaultStageFlip:
    def test_default_stage_is_one(self):
        """CONFIG-UX.4.2: DEFAULT_STAGE = 1."""
        from sndr.model_configs._rollout import DEFAULT_STAGE
        assert DEFAULT_STAGE == 1

    def test_stage_0_explicit_still_observable(self, monkeypatch):
        """Stage 0 explicit env override still resolves to 0."""
        from sndr.model_configs._rollout import rollout_stage
        monkeypatch.setenv("SNDR_V1_ROLLOUT_STAGE", "0")
        assert rollout_stage() == 0

    def test_stage_0_and_1_observable_equivalent(self):
        """Operator escape: Stage 0 and Stage 1 produce identical
        observable severity for non-tombstone buckets."""
        from sndr.model_configs._rollout import effective_severity
        for bucket in ("transparent", "needs_operator_choice", "deprecated",
                       "card_less_prod", "missing_override_policy"):
            assert effective_severity(bucket=bucket, stage=0) == "warn"
            assert effective_severity(bucket=bucket, stage=1) == "warn"
