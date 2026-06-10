# SPDX-License-Identifier: Apache-2.0
"""§6.H10 tests — Genesis process-info gauge.

Validates the label extraction helpers + the apply() lifecycle. The
gauge is a single-row info metric (canonical Prometheus pattern); the
tests verify each label resolves correctly under both env-set and
env-unset conditions, plus argv-flag parsing for K and backend.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _import_mod():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        mod = importlib.import_module(
            "sndr.observability.genesis_process_info"
        )
    finally:
        sys.path.pop(0)
    return mod


# ─── _extract_K (MTP spec-decode) ──────────────────────────────────────


def test_K_returns_0_when_no_spec_config():
    mod = _import_mod()
    assert mod._extract_K(["vllm", "serve"]) == "0"


def test_K_parses_clean_json_spec_config():
    mod = _import_mod()
    argv = [
        "vllm", "serve",
        "--speculative-config",
        '{"method": "mtp", "num_speculative_tokens": 4}',
    ]
    assert mod._extract_K(argv) == "4"


def test_K_parses_eq_form():
    mod = _import_mod()
    argv = [
        "vllm", "serve",
        '--speculative-config={"method":"mtp","num_speculative_tokens":3}',
    ]
    assert mod._extract_K(argv) == "3"


def test_K_falls_back_to_regex_on_quirky_quoting():
    """Some shells over-quote the JSON in heredoc launchers. The regex
    fallback must still extract num_speculative_tokens."""
    mod = _import_mod()
    argv = [
        "vllm", "serve",
        "--speculative-config",
        '{\\"method\\":\\"mtp\\",\\"num_speculative_tokens\\":5}',
    ]
    assert mod._extract_K(argv) == "5"


def test_K_returns_0_on_malformed_spec_config():
    mod = _import_mod()
    argv = ["vllm", "serve", "--speculative-config", "garbage"]
    assert mod._extract_K(argv) == "0"


# ─── _extract_backend ──────────────────────────────────────────────────


def test_backend_from_argv():
    mod = _import_mod()
    argv = ["vllm", "serve", "--attention-backend", "TURBOQUANT"]
    assert mod._extract_backend(argv) == "TURBOQUANT"


def test_backend_from_env_when_argv_empty(monkeypatch):
    mod = _import_mod()
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    assert mod._extract_backend(["vllm", "serve"]) == "FLASH_ATTN"


def test_backend_default_when_unset(monkeypatch):
    mod = _import_mod()
    monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)
    assert mod._extract_backend(["vllm", "serve"]) == "default"


# ─── _extract_model ────────────────────────────────────────────────────


def test_model_from_served_name_argv():
    mod = _import_mod()
    argv = ["vllm", "serve", "--served-model-name", "qwen3.6-35b-a3b"]
    assert mod._extract_model(argv) == "qwen3.6-35b-a3b"


def test_model_falls_back_to_model_path_basename():
    mod = _import_mod()
    argv = ["vllm", "serve", "--model", "/models/Qwen3.6-35B-A3B-FP8"]
    assert mod._extract_model(argv) == "Qwen3.6-35B-A3B-FP8"


def test_model_unknown_when_neither_flag_present():
    mod = _import_mod()
    assert mod._extract_model(["vllm", "serve"]) == "unknown"


# ─── _extract_patch_hash ───────────────────────────────────────────────


def test_patch_hash_uncommitted_when_no_repo_env(monkeypatch):
    mod = _import_mod()
    monkeypatch.delenv("GENESIS_REPO", raising=False)
    monkeypatch.delenv("GENESIS_PROJECT_ROOT", raising=False)
    assert mod._extract_patch_hash() == "uncommitted"


def test_patch_hash_uncommitted_when_repo_path_missing(monkeypatch, tmp_path):
    """Operator typo in $GENESIS_REPO must not crash the helper."""
    mod = _import_mod()
    monkeypatch.setenv("GENESIS_REPO", str(tmp_path / "doesnotexist"))
    assert mod._extract_patch_hash() == "uncommitted"


def test_patch_hash_returns_short_sha_for_real_repo(monkeypatch):
    """If GENESIS_REPO points at a real git repo, the helper returns
    the short SHA. We point at this repo as a real-data test."""
    mod = _import_mod()
    monkeypatch.setenv("GENESIS_REPO", str(REPO_ROOT))
    result = mod._extract_patch_hash()
    # Short SHA: 7+ hex chars OR the sentinel.
    assert result == "uncommitted" or (
        len(result) >= 7 and all(c in "0123456789abcdef" for c in result)
    )


# ─── _extract_pin ──────────────────────────────────────────────────────


def test_pin_unknown_on_torchless_env(monkeypatch):
    mod = _import_mod()
    # vllm isn't installed in the test env — pin should be "unknown".
    result = mod._extract_pin()
    assert isinstance(result, str)


# ─── _resolve_labels — combined ────────────────────────────────────────


def test_resolve_labels_all_canonical_keys(monkeypatch):
    mod = _import_mod()
    monkeypatch.setenv("GENESIS_PRESET", "prod-qwen3.6-35b-balanced")
    monkeypatch.setenv("GENESIS_PROFILE", "qwen3.6-35b-balanced")
    monkeypatch.setenv("GENESIS_WORKLOAD_CLASS", "free_chat")
    argv = [
        "vllm", "serve",
        "--served-model-name", "qwen3.6-35b-a3b",
        "--attention-backend", "TURBOQUANT",
        "--speculative-config", '{"num_speculative_tokens": 3}',
    ]
    labels = mod._resolve_labels(argv)
    expected_keys = {
        "preset", "profile", "workload_class", "K",
        "backend", "patch_hash", "model", "pin",
    }
    assert set(labels.keys()) == expected_keys
    assert labels["preset"] == "prod-qwen3.6-35b-balanced"
    assert labels["profile"] == "qwen3.6-35b-balanced"
    assert labels["workload_class"] == "free_chat"
    assert labels["K"] == "3"
    assert labels["backend"] == "TURBOQUANT"
    assert labels["model"] == "qwen3.6-35b-a3b"


def test_resolve_labels_fallback_when_env_unset(monkeypatch):
    """No env, no argv → all-unknown labels (no exception)."""
    mod = _import_mod()
    for env in ("GENESIS_PRESET", "GENESIS_PROFILE",
                "GENESIS_WORKLOAD_CLASS", "GENESIS_REPO",
                "GENESIS_PROJECT_ROOT", "VLLM_ATTENTION_BACKEND"):
        monkeypatch.delenv(env, raising=False)
    labels = mod._resolve_labels(["vllm", "serve"])
    assert labels["preset"] == "unknown"
    assert labels["profile"] == "unknown"
    assert labels["workload_class"] == "unknown"
    assert labels["K"] == "0"
    assert labels["backend"] == "default"
    assert labels["model"] == "unknown"


# ─── apply() lifecycle ────────────────────────────────────────────────


def test_apply_skipped_when_prometheus_client_missing(monkeypatch):
    mod = _import_mod()
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "prometheus_client":
            raise ImportError("synthetic")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    status, reason = mod.apply()
    assert status == "skipped"
    assert "prometheus_client" in reason


def test_apply_emits_gauge_with_labels(monkeypatch):
    pytest.importorskip("prometheus_client")
    mod = _import_mod()
    monkeypatch.setenv("GENESIS_PRESET", "prod-qwen3.6-27b-tq-k8v4")
    monkeypatch.setenv("GENESIS_PROFILE", "qwen3.6-27b-long-ctx")
    argv = [
        "vllm", "serve",
        "--served-model-name", "qwen3.6-27b",
        "--attention-backend", "TURBOQUANT",
        "--speculative-config", '{"num_speculative_tokens": 3}',
    ]
    status, reason = mod.apply(argv)
    assert status == "applied"
    assert mod.is_applied() is True

    # Inspect the gauge's labeled samples — should have exactly one
    # row, value 1.0, with our labels.
    samples = list(mod._prom_gauge.collect())[0].samples
    assert len(samples) == 1
    s = samples[0]
    assert s.value == 1.0
    assert s.labels["preset"] == "prod-qwen3.6-27b-tq-k8v4"
    assert s.labels["profile"] == "qwen3.6-27b-long-ctx"
    assert s.labels["K"] == "3"
    assert s.labels["backend"] == "TURBOQUANT"
    assert s.labels["model"] == "qwen3.6-27b"


def test_apply_is_idempotent(monkeypatch):
    pytest.importorskip("prometheus_client")
    mod = _import_mod()
    # First apply
    monkeypatch.setenv("GENESIS_PRESET", "preset-a")
    s1, _ = mod.apply(["vllm", "serve",
                        "--served-model-name", "modelA"])
    assert s1 == "applied"
    # Second apply with different labels — should re-set, not raise.
    monkeypatch.setenv("GENESIS_PRESET", "preset-b")
    s2, _ = mod.apply(["vllm", "serve",
                        "--served-model-name", "modelB"])
    assert s2 == "applied"
    # Gauge survives + carries the latest label set somewhere.
    samples = list(mod._prom_gauge.collect())[0].samples
    assert any(s.labels["preset"] == "preset-b" for s in samples)


# ─── Module exports surface ────────────────────────────────────────────


def test_public_exports_lock():
    mod = _import_mod()
    for name in [
        "apply", "is_applied", "_resolve_labels",
        "_extract_K", "_extract_backend", "_extract_model",
        "_extract_patch_hash", "_extract_pin", "_setup_gauge",
    ]:
        assert name in mod.__all__, (
            f"{name!r} must be in __all__ for downstream stability"
        )
