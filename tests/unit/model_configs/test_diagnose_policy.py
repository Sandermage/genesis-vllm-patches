# SPDX-License-Identifier: Apache-2.0
"""Phase D — diagnose plan-aware comparison.

When an operator launched the container with --policy=safe/minimal,
the running env is the *filtered* set, not cfg.genesis_env raw. The
legacy diagnose path compares against the raw matrix and would flag
every policy-excluded toggle as "missing" — false positive.

This test verifies that ``diagnose_env_exported(cfg, expected_env=...)``
can take an explicit expected-env override (the policy-filtered map)
so the diff stays accurate.
"""
from __future__ import annotations

from unittest.mock import patch

from sndr.model_configs.diagnose import diagnose_env_exported
from sndr.model_configs.schema import (
    DockerConfig,
    HardwareSpec,
    ModelConfig,
)


def _make_cfg_with_docker(
    genesis_env: dict[str, str] | None = None,
) -> ModelConfig:
    return ModelConfig(
        key="diagnose-test", title="t", description="d", schema_version=1,
        maintainer="x", model_path="/m",
        hardware=HardwareSpec(
            gpu_match_keys=["g"], n_gpus=1, min_vram_per_gpu_mib=8000,
        ),
        last_validated=None, genesis_pin=None, vllm_pin_required=None,
        served_model_name=None, quantization=None, kv_cache_dtype=None,
        max_model_len=8192, gpu_memory_utilization=0.9, max_num_seqs=1,
        max_num_batched_tokens=2048, enable_chunked_prefill=False,
        dtype="float16", enforce_eager=False, disable_custom_all_reduce=False,
        language_model_only=True, trust_remote_code=True,
        enable_auto_tool_choice=False,
        tool_call_parser=None, reasoning_parser=None, spec_decode=None,
        genesis_env=genesis_env or {}, system_env={},
        vllm_extra_args=[], cudagraph_mode="auto",
        docker=DockerConfig(image="x", container_name="vllm-diag-test"),
    )


def _stub_docker_inspect(env_pairs: list[str]):
    """Patch the diagnose._run helper to return a synthetic env list."""
    import json
    def _fake_run(argv, *a, **kw):
        # Return (rc, stdout, stderr) — stdout is the JSON env list.
        return (0, json.dumps(env_pairs), "")
    return patch(
        "sndr.model_configs.diagnose._run",
        side_effect=_fake_run,
    )


# ─── Without expected_env: legacy comparison against cfg.genesis_env ─────


class TestLegacyCompare:
    def test_missing_env_flagged_as_error(self):
        cfg = _make_cfg_with_docker(genesis_env={
            "GENESIS_ENABLE_PN17": "1",
            "GENESIS_ENABLE_PN32": "1",
        })
        # Container exports only PN17 — PN32 missing → error.
        with _stub_docker_inspect(["GENESIS_ENABLE_PN17=1"]):
            findings = diagnose_env_exported(cfg)
        errors = [f for f in findings if not f.passed and f.severity == "error"]
        names = [f.name for f in errors]
        assert "env:GENESIS_ENABLE_PN32" in names
        assert "env:GENESIS_ENABLE_PN17" not in names


# ─── With expected_env: policy-filtered comparison ───────────────────────


class TestExpectedEnvOverride:
    def test_expected_env_skips_dropped_toggles(self):
        """When the operator launched with --policy safe (e.g.) PN32
        was excluded. Diagnose with expected_env=plan.env should NOT
        flag PN32 as missing — that's the intended state."""
        cfg = _make_cfg_with_docker(genesis_env={
            "GENESIS_ENABLE_PN17": "1",
            "GENESIS_ENABLE_PN32": "1",
        })
        # Container only carries PN17 (PN32 was dropped by policy).
        expected = {"GENESIS_ENABLE_PN17": "1"}
        with _stub_docker_inspect(["GENESIS_ENABLE_PN17=1"]):
            findings = diagnose_env_exported(cfg, expected_env=expected)
        errors = [f for f in findings if not f.passed and f.severity == "error"]
        names = [f.name for f in errors]
        # PN32 absent from container is expected — no error.
        assert "env:GENESIS_ENABLE_PN32" not in names
        # PN17 present + matches → reported as info-pass.
        assert any(
            f.name == "env:GENESIS_ENABLE_PN17" and f.passed
            for f in findings
        )

    def test_unexpected_extra_env_not_flagged_when_override_used(self):
        """Container can carry extra env keys (system_env, system
        injections). Diagnose with expected_env should only check the
        intersection — not flag extras as ERRORS."""
        cfg = _make_cfg_with_docker(genesis_env={
            "GENESIS_ENABLE_PN17": "1",
        })
        expected = {"GENESIS_ENABLE_PN17": "1"}
        with _stub_docker_inspect([
            "GENESIS_ENABLE_PN17=1",
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        ]):
            findings = diagnose_env_exported(cfg, expected_env=expected)
        assert all(f.passed for f in findings if f.severity == "error")
