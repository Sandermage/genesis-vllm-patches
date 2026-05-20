# SPDX-License-Identifier: Apache-2.0
"""P1.5 unit tests for `sndr profile render-launchers`.

10 operator-facing acceptance gates:

  G01  dry-run renders gemma4-tq-default as valid bash
  G02  default does NOT contain MTP / spec-decode env
  G03  dry-run renders gemma4-tq-mtp-structured-k4 as valid bash
  G04  structured contains skip-list 58,59
  G05  structured contains G4_71b + G4_75 backend routing
  G06  structured contains MTP K=4 (--speculative-config '{"method": "mtp", "num_speculative_tokens": 4, ...}')
  G07  structured does NOT contain PN282/PN283 observability env (auto-emit OFF)
  G08  --output writes to <dir>/start_<id>.sh
  G09  overwrite without --force returns exit 1
  G10  bad backend_plan value raises SchemaError (returns exit 2)
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from vllm.sndr_core.cli.profile import (
    _BACKEND_PLAN_MAP,
    _OBSERVABILITY_OPTIN_ENVS,
    _STRUCTURED_REQUIRED_ENVS,
    _validate_backend_plan_consistency,
    render_profile_launcher,
)


# ─── G01 + G02: default profile ─────────────────────────────────────────


class TestDefaultProfileRender:
    def test_g01_default_renders_valid_bash_shebang(self):
        script = render_profile_launcher("gemma4-tq-default")
        assert script.startswith("#!/bin/bash\n")
        assert "set -e" in script
        assert "docker run -d --name" in script

    def test_g02_default_no_mtp_no_spec_decode(self):
        script = render_profile_launcher("gemma4-tq-default")
        assert "--speculative-config" not in script
        assert "method: mtp" not in script
        assert "method\":\"mtp" not in script

    def test_g02_default_no_skip_list_env(self):
        script = render_profile_launcher("gemma4-tq-default")
        assert "SNDR_G4_TQ_FORCE_SKIP_LAYERS" not in script
        assert "GENESIS_G4_TQ_FORCE_SKIP_LAYERS" not in script

    def test_g02_default_no_structured_backend_routing(self):
        script = render_profile_launcher("gemma4-tq-default")
        assert "GENESIS_ENABLE_G4_71B_DRAFTER_SLIDING_TRITON" not in script
        assert "GENESIS_ENABLE_G4_75_DRAFTER_HEAD512_TRITON" not in script


# ─── G03–G07: structured profile ────────────────────────────────────────


class TestStructuredProfileRender:
    @pytest.fixture(scope="class")
    def script(self):
        return render_profile_launcher("gemma4-tq-mtp-structured-k4")

    def test_g03_structured_renders_valid_bash(self, script):
        assert script.startswith("#!/bin/bash\n")
        assert "set -e" in script
        assert "docker run -d --name" in script
        assert "vllm-gemma4-tq-mtp-structured-k4-k${K}" in script

    def test_g04_structured_skip_list_58_59(self, script):
        assert "SNDR_G4_TQ_FORCE_SKIP_LAYERS=58,59" in script
        # Legacy alias also emitted by compose (one-release migration window)
        assert "GENESIS_G4_TQ_FORCE_SKIP_LAYERS=58,59" in script

    def test_g05_structured_g4_71b_and_g4_75_present(self, script):
        assert "GENESIS_ENABLE_G4_71B_DRAFTER_SLIDING_TRITON=1" in script
        assert "GENESIS_ENABLE_G4_75_DRAFTER_HEAD512_TRITON=1" in script

    def test_g06_structured_mtp_k4_speculative_config(self, script):
        assert "--speculative-config" in script
        # The JSON shape is single-quoted; verify both K=4 and the drafter
        # model path are present together.
        assert '"method": "mtp"' in script
        assert '"num_speculative_tokens": 4' in script
        assert "/models/gemma-4-31B-it-assistant" in script

    def test_g07_no_pn282_pn283_observability_env(self, script):
        for env in _OBSERVABILITY_OPTIN_ENVS:
            # Allow the comment block mentioning these by name to pass —
            # check only the actual -e flag form.
            assert f"-e {env}=" not in script, (
                f"observability env {env} leaked into rendered launcher"
            )

    def test_structured_required_envs_all_present(self, script):
        """All envs the byte-equivalence gate cares about are in the
        rendered launcher."""
        for env in _STRUCTURED_REQUIRED_ENVS:
            assert env in script, (
                f"required structured env {env} missing from render"
            )

    def test_structured_pn274_guard_optin_present(self, script):
        assert "SNDR_ALLOW_SPEC_DECODE_KV_ADAPTER=1" in script

    def test_structured_attention_backend_arg(self, script):
        assert "--attention-backend TURBOQUANT" in script

    # ─── P1.7d byte-equivalence gate extensions ────────────────────────
    #
    # The opt-in rehearsal halt diagnosis (2026-05-20) found that the
    # original P1.5 gate checked env vars but missed THREE CLI args
    # that diverged from the validated start_g4_betaA_k1.sh:
    #   --kv-cache-dtype      auto                  → turboquant_4bit_nc
    #   --max-num-seqs        2 (hardware default)  → 1
    #   --speculative-config  no attention_backend  → "attention_backend":"FLASH_ATTN"
    #
    # The three tests below codify these as part of the byte-
    # equivalence gate so any future regression would fail in CI
    # rather than at server-side rehearsal time.

    def test_p1_7d_structured_kv_cache_dtype_turboquant(self, script):
        """P1.7a + P1.7d: rendered --kv-cache-dtype must be
        turboquant_4bit_nc (driven by profile.compression_plan.default_kv_dtype
        on top of a neutral 'auto' parent ModelDef)."""
        assert "--kv-cache-dtype turboquant_4bit_nc" in script, (
            "structured render must use --kv-cache-dtype turboquant_4bit_nc "
            "(P1.7a kv_cache_dtype promotion)"
        )
        # And NOT carry through the parent's 'auto'
        assert "--kv-cache-dtype auto" not in script

    def test_p1_7d_structured_max_num_seqs_1(self, script):
        """P1.7b + P1.7d: rendered --max-num-seqs must be 1 (driven by
        profile.sizing_override.max_num_seqs=1, NOT hardware default 2)."""
        assert "--max-num-seqs 1" in script, (
            "structured render must use --max-num-seqs 1 (P1.7b "
            "sizing_override matches validated launcher concurrency)"
        )
        # And NOT carry through the hardware default 2
        assert "--max-num-seqs 2" not in script

    def test_p1_7d_structured_max_model_len_4096(self, script):
        """P1.7b + P1.7d: rendered --max-model-len must be 4096 (driven
        by profile.sizing_override.max_model_len=4096, NOT hardware
        default 280000)."""
        assert "--max-model-len 4096" in script

    def test_p1_7d_structured_max_num_batched_tokens_8192(self, script):
        """P1.7b + P1.7d: rendered --max-num-batched-tokens must be 8192
        (driven by profile.sizing_override, NOT hardware default 4096)."""
        assert "--max-num-batched-tokens 8192" in script

    def test_p1_7d_structured_gpu_memory_utilization_0_92(self, script):
        """P1.7b + P1.7d: rendered --gpu-memory-utilization must be 0.92
        (driven by profile.sizing_override, NOT hardware default 0.90)."""
        assert "--gpu-memory-utilization 0.92" in script

    def test_p1_7d_structured_speculative_config_carries_attention_backend(
        self, script,
    ):
        """P1.7c + P1.7d: rendered --speculative-config JSON must
        include "attention_backend": "FLASH_ATTN" (driven by
        profile.spec_decode_override.attention_backend).

        Without this, the drafter falls back to vLLM auto-pick (would
        land on TURBOQUANT on a TQ engine) and breaks the validated
        acceptance distribution."""
        import re
        # Find the --speculative-config arg and parse its JSON
        m = re.search(
            r"--speculative-config '(\{[^']+\})'",
            script,
        )
        assert m, (
            "structured render must contain --speculative-config '{...}'"
        )
        import json as _json
        spec_json = _json.loads(m.group(1))
        assert spec_json["method"] == "mtp"
        assert spec_json["num_speculative_tokens"] == 4
        assert spec_json["model"] == "/models/gemma-4-31B-it-assistant"
        assert spec_json["attention_backend"] == "FLASH_ATTN", (
            f"speculative-config missing attention_backend=FLASH_ATTN; "
            f"got: {spec_json}"
        )

    def test_structured_pr42637_overlay_mounts_present(self, script):
        """structured profile enables G4_60a..k → render must mount the
        8 PR42637 overlay files."""
        assert "upstream_overlay_pr42637/turboquant_attn.py" in script
        assert "upstream_overlay_pr42637/kv_cache_utils.py" in script
        assert "upstream_overlay_pr42637/block_pool.py" in script

    def test_default_pr42637_overlay_mounts_absent(self):
        """default profile does NOT have G4_60a..k → no overlay mounts."""
        script = render_profile_launcher("gemma4-tq-default")
        assert "upstream_overlay_pr42637" not in script


# ─── Backend mapping table tests ────────────────────────────────────────


class TestBackendMapping:
    def test_backend_plan_map_immutable_known_pairs(self):
        """If this fails, somebody added/removed entries in
        _BACKEND_PLAN_MAP. Audit before committing — the map is the
        contract surface."""
        assert _BACKEND_PLAN_MAP[
            ("drafter_sliding", "TRITON_ATTN")
        ] == "GENESIS_ENABLE_G4_71B_DRAFTER_SLIDING_TRITON"
        assert _BACKEND_PLAN_MAP[
            ("drafter_full", "TRITON_ATTN")
        ] == "GENESIS_ENABLE_G4_75_DRAFTER_HEAD512_TRITON"
        assert _BACKEND_PLAN_MAP[("target_default", "TURBOQUANT")] is None
        assert _BACKEND_PLAN_MAP[("target_native_layers", "TRITON_ATTN")] is None

    def test_g10_unknown_backend_value_raises(self, monkeypatch):
        """G10: a backend_plan value not in the mapping table must raise
        SchemaError. Mocks load_profile to return a synthetic profile
        with a bogus drafter_sliding value."""
        from vllm.sndr_core.model_configs.schema_v2 import (
            BackendPlanConfig, ProfileDef, PatchesDelta,
        )
        from vllm.sndr_core.model_configs.schema import SchemaError

        bad_profile = ProfileDef(
            schema_version=2, kind="profile",
            id="synthetic-bad-backend",
            parent_model="gemma-4-31b-it-awq",
            maintainer="tests",
            status="experimental",
            patches_delta=PatchesDelta(),
            role="structured",
            backend_plan=BackendPlanConfig(
                drafter_sliding="MAMBA_ATTN",  # not in mapping
            ),
        )

        def fake_load(pid):
            if pid == bad_profile.id:
                return bad_profile
            from vllm.sndr_core.model_configs import registry_v2 as real
            return real.load_profile(pid)

        monkeypatch.setattr(
            "vllm.sndr_core.model_configs.registry_v2.load_profile",
            fake_load,
        )

        with pytest.raises(SchemaError, match="not in the supported "
                                              "backend mapping table"):
            render_profile_launcher(bad_profile.id)

    def test_mapped_env_must_be_in_genesis_env(self, monkeypatch):
        """If backend_plan says drafter_sliding=TRITON_ATTN but the
        corresponding env is NOT in cfg.genesis_env (operator forgot
        patches_delta.enable), SchemaError fires. This protects
        against silent declarative/runtime divergence."""
        from vllm.sndr_core.model_configs.schema_v2 import (
            BackendPlanConfig, ProfileDef, PatchesDelta,
        )
        from vllm.sndr_core.model_configs.schema import SchemaError

        # Build profile with backend_plan but EMPTY patches_delta —
        # backend declared but not enabled via env.
        bad_profile = ProfileDef(
            schema_version=2, kind="profile",
            id="synthetic-mapped-missing",
            parent_model="gemma-4-31b-it-awq",
            maintainer="tests",
            status="experimental",
            patches_delta=PatchesDelta(),  # empty; will not enable G4_71b env
            role="structured",
            backend_plan=BackendPlanConfig(
                drafter_sliding="TRITON_ATTN",  # mapped, but no env set
            ),
        )

        def fake_load(pid):
            if pid == bad_profile.id:
                return bad_profile
            from vllm.sndr_core.model_configs import registry_v2 as real
            return real.load_profile(pid)

        monkeypatch.setattr(
            "vllm.sndr_core.model_configs.registry_v2.load_profile",
            fake_load,
        )

        with pytest.raises(SchemaError, match="GENESIS_ENABLE_G4_71B"):
            render_profile_launcher(bad_profile.id)


# ─── G08 + G09: output file handling ────────────────────────────────────


class TestOutputFlags:
    def test_g08_output_writes_file(self, tmp_path):
        """--output DIR writes start_<profile_id>.sh into DIR."""
        result = subprocess.run(
            [
                sys.executable, "-m", "vllm.sndr_core.cli",
                "profile", "render-launchers",
                "gemma4-tq-default",
                "--output", str(tmp_path),
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"expected 0; got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        out_file = tmp_path / "start_gemma4-tq-default.sh"
        assert out_file.exists(), f"{out_file} not created"
        assert out_file.stat().st_mode & 0o111, "output file not executable"
        content = out_file.read_text()
        assert content.startswith("#!/bin/bash")

    def test_g09_overwrite_without_force_fails(self, tmp_path):
        """Pre-existing target file + no --force → exit 1, no write."""
        # Pre-create a sentinel file at the target path
        target = tmp_path / "start_gemma4-tq-default.sh"
        target.write_text("# preexisting sentinel\n")
        before_mtime = target.stat().st_mtime

        result = subprocess.run(
            [
                sys.executable, "-m", "vllm.sndr_core.cli",
                "profile", "render-launchers",
                "gemma4-tq-default",
                "--output", str(tmp_path),
                # NO --force
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 1, (
            f"expected exit 1; got {result.returncode}"
        )
        # File untouched
        assert target.read_text() == "# preexisting sentinel\n"

    def test_g09_overwrite_with_force_succeeds(self, tmp_path):
        target = tmp_path / "start_gemma4-tq-default.sh"
        target.write_text("# preexisting sentinel\n")

        result = subprocess.run(
            [
                sys.executable, "-m", "vllm.sndr_core.cli",
                "profile", "render-launchers",
                "gemma4-tq-default",
                "--output", str(tmp_path),
                "--force",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        # Sentinel was overwritten with the real script
        assert target.read_text().startswith("#!/bin/bash")

    def test_dry_run_does_not_write_file(self, tmp_path):
        """--dry-run + --output → still prints to stdout, does NOT
        write the file."""
        result = subprocess.run(
            [
                sys.executable, "-m", "vllm.sndr_core.cli",
                "profile", "render-launchers",
                "gemma4-tq-default",
                "--output", str(tmp_path),
                "--dry-run",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        out_file = tmp_path / "start_gemma4-tq-default.sh"
        assert not out_file.exists(), "dry-run should not write file"
        assert result.stdout.startswith("#!/bin/bash")

    def test_default_is_dry_run(self):
        """No --output, no --dry-run → defaults to stdout (dry-run-like)."""
        result = subprocess.run(
            [
                sys.executable, "-m", "vllm.sndr_core.cli",
                "profile", "render-launchers",
                "gemma4-tq-default",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert result.stdout.startswith("#!/bin/bash")


# ─── Backend consistency check function (direct) ────────────────────────


class TestValidateBackendPlanConsistency:
    def test_no_backend_plan_is_noop(self):
        from vllm.sndr_core.model_configs.schema_v2 import (
            ProfileDef, PatchesDelta,
        )
        p = ProfileDef(
            schema_version=2, kind="profile",
            id="tx", parent_model="gemma-4-31b-it-awq",
            maintainer="t", status="experimental",
            patches_delta=PatchesDelta(),
            role="default",
            backend_plan=None,
        )
        _validate_backend_plan_consistency(p, {})  # no raise

    def test_unmapped_env_none_value_is_ok(self):
        """target_default → None mapping; consistency check passes
        even without an env in genesis_env."""
        from vllm.sndr_core.model_configs.schema_v2 import (
            BackendPlanConfig, ProfileDef, PatchesDelta,
        )
        p = ProfileDef(
            schema_version=2, kind="profile",
            id="tx", parent_model="gemma-4-31b-it-awq",
            maintainer="t", status="experimental",
            patches_delta=PatchesDelta(),
            role="structured",
            backend_plan=BackendPlanConfig(target_default="TURBOQUANT"),
        )
        _validate_backend_plan_consistency(p, {})  # no raise
