# SPDX-License-Identifier: Apache-2.0
"""P1.5 unit tests for `sndr profile render-launchers`.

10 operator-facing acceptance gates:

  G01  dry-run renders gemma4-31b-tq-default as valid bash
  G02  default does NOT contain MTP / spec-decode env
  G03  dry-run renders gemma4-31b-tq-mtp-structured-k4 as valid bash
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

from sndr.cli.legacy.profile import (
    _BACKEND_PLAN_MAP,
    _OBSERVABILITY_OPTIN_ENVS,
    _STRUCTURED_REQUIRED_ENVS,
    _validate_backend_plan_consistency,
    render_profile_launcher,
)


_FIXTURES_DIR = Path(__file__).with_name("fixtures")
_CONTROL_A_ENV_SNAPSHOT = _FIXTURES_DIR / "start_g4_betaA_k1_envs.txt"

# Env lines present in the hand-written control-A launcher but not owned by
# the V2 profile/render contract. They are debug/runtime shell controls, or
# the PN248 trace flag that remains an operator instrumentation choice.
_CONTROL_A_NON_PROFILE_ENVS = {
    "CUDA_LAUNCH_BLOCKING",
    "NCCL_CUMEM_ENABLE",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "GENESIS_ENABLE_PN248_ACCEPTANCE_TRACE",
}

# The hand-written launcher still uses the legacy GENESIS alias here. The V2
# profile intentionally emits the SNDR canonical key; get_sndr_env() accepts
# both at runtime, so the parity test normalizes this one known alias.
_CONTROL_A_ALIAS_EQUIVALENTS = {
    "GENESIS_ALLOW_SPEC_DECODE_KV_ADAPTER": "SNDR_ALLOW_SPEC_DECODE_KV_ADAPTER",
    # The hand-written launcher used a long PN110 name that NO resolver
    # ever read (Class-7 env-name typo — the registry env_flag is the
    # short `GENESIS_ENABLE_PN110`). The V2 profile emits the canonical
    # short key, which is the load-bearing one; parity checks against it.
    "GENESIS_ENABLE_PN110_BLOCK_POOL_FREE_DEDUP": "GENESIS_ENABLE_PN110",
}


def _parse_env_snapshot(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        env[key] = value
    return env


def _parse_rendered_docker_envs(script: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw in script.splitlines():
        line = raw.strip()
        if not line.startswith("-e "):
            continue
        payload = line[3:].strip()
        if payload.endswith("\\"):
            payload = payload[:-1].strip()
        key, value = payload.split("=", 1)
        env[key] = value
    return env


# ─── G01 + G02: default profile ─────────────────────────────────────────


class TestDefaultProfileRender:
    def test_g01_default_renders_valid_bash_shebang(self):
        script = render_profile_launcher("gemma4-31b-tq-default")
        assert script.startswith("#!/bin/bash\n")
        assert "set -e" in script
        assert "docker run -d --name" in script

    def test_g02_default_no_mtp_no_spec_decode(self):
        script = render_profile_launcher("gemma4-31b-tq-default")
        assert "--speculative-config" not in script
        assert "method: mtp" not in script
        assert "method\":\"mtp" not in script

    def test_g02_default_no_skip_list_env(self):
        script = render_profile_launcher("gemma4-31b-tq-default")
        assert "SNDR_G4_TQ_FORCE_SKIP_LAYERS" not in script
        assert "GENESIS_G4_TQ_FORCE_SKIP_LAYERS" not in script

    def test_launcher_stamps_sndr_identity_labels(self):
        # The docker run must carry sndr.preset (authoritative container->preset
        # link), plus served-model / pin / patch-count / role so the Control
        # Center links the engine and shows its identity without an api-key.
        script = render_profile_launcher("gemma4-31b-tq-default")
        assert '--label sndr.preset="gemma4-31b-tq-default"' in script
        assert "--label sndr.pin=" in script
        assert "--label sndr.served-model=" in script
        assert "--label sndr.patch-count=" in script
        assert "--label sndr.role=" in script

    def test_disable_log_stats_emitted_by_default(self):
        # Default (sizing.disable_log_stats=True) preserves the historical
        # launcher: the vLLM stat logger stays off.
        script = render_profile_launcher("gemma4-31b-tq-default")
        assert "--disable-log-stats" in script

    def test_disable_log_stats_omitted_when_sizing_opts_in(self, monkeypatch):
        # sizing.disable_log_stats=False -> launcher omits the flag so vLLM
        # exposes live request/KV-cache/throughput metrics (Inference panel).
        import sndr.model_configs.compose as comp
        real_compose = comp.compose

        def patched(model, hw, profile):
            cfg = real_compose(model, hw, profile)
            cfg.disable_log_stats = False
            return cfg

        monkeypatch.setattr(comp, "compose", patched)
        script = render_profile_launcher("gemma4-31b-tq-default")
        assert "--disable-log-stats" not in script
        # The port line keeps its place and is not left with a dangling backslash.
        assert "--host" in script and "--port" in script

    def test_g02_default_no_structured_backend_routing(self):
        script = render_profile_launcher("gemma4-31b-tq-default")
        assert "GENESIS_ENABLE_G4_71B_DRAFTER_SLIDING_TRITON" not in script
        assert "GENESIS_ENABLE_G4_75_DRAFTER_HEAD512_TRITON" not in script


# ─── G03–G07: structured profile ────────────────────────────────────────


class TestStructuredProfileRender:
    @pytest.fixture(scope="class")
    def script(self):
        return render_profile_launcher("gemma4-31b-tq-mtp-structured-k4")

    def test_g03_structured_renders_valid_bash(self, script):
        assert script.startswith("#!/bin/bash\n")
        assert "set -e" in script
        assert "docker run -d --name" in script
        assert "vllm-gemma4-31b-tq-mtp-structured-k4-k${K}" in script

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

    def test_control_a_env_snapshot_matches_rendered_profile_envs(self, script):
        """Control-A parity gate (2026-05-21).

        The first V2-rendered G-STRUCT-K4 launcher passed boot/guard gates but
        produced corrupt unicode output because it missed load-bearing envs
        from the hand-written `start_g4_betaA_k1.sh`. This fixture snapshots
        that hand-written launcher's env contract and compares key/value pairs
        instead of relying on broad substring checks.

        Some hand-written envs are intentionally outside profile ownership
        (debug shell controls and PN248 trace), and the PN274 guard opt-in is
        normalized from the legacy GENESIS alias to the SNDR canonical key.
        Everything else in the snapshot must be present at the same value.
        """
        expected = _parse_env_snapshot(_CONTROL_A_ENV_SNAPSHOT)
        rendered = _parse_rendered_docker_envs(script)

        missing: dict[str, str] = {}
        mismatched: dict[str, dict[str, str]] = {}
        for key, value in expected.items():
            if key in _CONTROL_A_NON_PROFILE_ENVS:
                continue
            rendered_key = _CONTROL_A_ALIAS_EQUIVALENTS.get(key, key)
            observed = rendered.get(rendered_key)
            if observed is None:
                missing[key] = value
            elif observed != value:
                mismatched[key] = {"expected": value, "observed": observed}

        assert not missing and not mismatched, json.dumps(
            {
                "missing_from_rendered_launcher": missing,
                "value_mismatches": mismatched,
            },
            indent=2,
            sort_keys=True,
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
        assert "overlays/pr42637/turboquant_attn.py" in script
        assert "overlays/pr42637/kv_cache_utils.py" in script
        assert "overlays/pr42637/block_pool.py" in script

    def test_default_pr42637_overlay_mounts_absent(self):
        """default profile does NOT have G4_60a..k → no overlay mounts."""
        script = render_profile_launcher("gemma4-31b-tq-default")
        assert "overlays/pr42637" not in script


# ─── P2.1 image pin routing tests ───────────────────────────────────────


class TestP21ImagePinRouting:
    """P2.1 — rendered launcher's IMAGE line MUST match the hardware YAML's
    runtime.docker.image verbatim, NOT a generic mutable tag.

    The pre-fix renderer hard-coded ``IMAGE="vllm/vllm-openai:nightly"``,
    which silently routed to whichever pin the host last tagged
    ``:nightly``. On 2026-05-21 this produced the Q35-TQ HALT: the rig
    was validated on dev338 but the host's ``:nightly`` was dev371, and
    the cross-pin chunked_fwd kwarg mismatch killed the engine at
    initialization.

    These tests pin the renderer to ``hw.runtime.docker.image`` verbatim.
    """

    @staticmethod
    def _extract_image_line(script: str) -> str | None:
        """Return the IMAGE="..." value from the rendered launcher, or
        None if the line is absent."""
        import re
        m = re.search(r'^IMAGE="([^"]+)"\s*$', script, re.MULTILINE)
        return m.group(1) if m else None

    def test_image_line_present(self):
        """Rendered launcher has exactly one IMAGE="..." line."""
        script = render_profile_launcher(
            "gemma4-31b-tq-mtp-structured-k4",
            hardware_id="a5000-2x-24gbvram-16cpu-128gbram",
        )
        image = self._extract_image_line(script)
        assert image is not None, (
            "rendered launcher missing IMAGE=\"...\" line"
        )

    def test_image_matches_hardware_yaml_a5000_2x(self):
        """gemma4 structured + a5000-2x → rendered IMAGE must equal
        hw.runtime.docker.image (dev338 explicit hash tag, NOT generic
        :nightly)."""
        from sndr.model_configs.registry_v2 import load_hardware
        hw = load_hardware("a5000-2x-24gbvram-16cpu-128gbram")
        expected = hw.runtime.docker.image  # type: ignore[union-attr]
        script = render_profile_launcher(
            "gemma4-31b-tq-mtp-structured-k4",
            hardware_id="a5000-2x-24gbvram-16cpu-128gbram",
        )
        image = self._extract_image_line(script)
        assert image == expected, (
            f"rendered IMAGE={image!r} != hw.runtime.docker.image={expected!r}; "
            f"renderer must use the hardware YAML's pinned image verbatim, "
            f"NOT a generic mutable tag"
        )

    def test_image_is_explicit_hash_not_generic_nightly(self):
        """Defensive: on a5000-2x the hardware YAML pins to the explicit
        dev338 hash tag. The rendered launcher must NOT emit the bare
        :nightly tag (which is mutable on the host)."""
        script = render_profile_launcher(
            "gemma4-31b-tq-mtp-structured-k4",
            hardware_id="a5000-2x-24gbvram-16cpu-128gbram",
        )
        image = self._extract_image_line(script)
        assert image is not None
        # The bare :nightly tag is the failure mode we're guarding
        # against. Any explicit-hash tag (nightly-<sha>) is fine.
        assert image != "vllm/vllm-openai:nightly", (
            "rendered IMAGE must not be the bare mutable :nightly tag; "
            "hardware YAML pins an explicit hash and renderer must "
            "carry it through"
        )

    def test_image_matches_default_profile_too(self):
        """Pin routing is profile-agnostic — also covers the default
        (non-structured) profile to avoid regressions on operator
        smoke launches."""
        from sndr.model_configs.registry_v2 import load_hardware
        hw = load_hardware("a5000-2x-24gbvram-16cpu-128gbram")
        expected = hw.runtime.docker.image  # type: ignore[union-attr]
        script = render_profile_launcher(
            "gemma4-31b-tq-default",
            hardware_id="a5000-2x-24gbvram-16cpu-128gbram",
        )
        image = self._extract_image_line(script)
        assert image == expected

    def test_missing_docker_block_raises_via_compose(self, monkeypatch):
        """When hardware YAML has no docker block but runtime='docker'
        is chosen, compose() raises SchemaError upstream — the renderer
        never reaches its IMAGE line. This regression-tests that the
        renderer relies on compose's invariant rather than a silent
        generic-tag fallback that masked the original Q35-TQ defect."""
        from sndr.model_configs import registry_v2 as r2
        from sndr.model_configs.schema import SchemaError
        hw = r2.load_hardware("a5000-2x-24gbvram-16cpu-128gbram")
        hw.runtime.docker = None

        real_load_hw = r2.load_hardware

        def fake_load_hw(hw_id):
            if hw_id == "a5000-2x-24gbvram-16cpu-128gbram":
                return hw
            return real_load_hw(hw_id)

        monkeypatch.setattr(
            "sndr.model_configs.registry_v2.load_hardware",
            fake_load_hw,
        )
        with pytest.raises(SchemaError, match="hardware.runtime.docker"):
            render_profile_launcher(
                "gemma4-31b-tq-default",
                hardware_id="a5000-2x-24gbvram-16cpu-128gbram",
            )


# ─── KV-cache-dtype emission (DFlash precondition) ─────────────────────


class TestKvCacheDtypeEmission:
    """Q27-DFlash dev371 smoke 2026-05-21 failed at vllm argparse because
    the renderer's unconditional f-string emitted `--kv-cache-dtype None`
    when the ModelDef declared `kv_cache_dtype: null`.

    Fix: when `cfg.kv_cache_dtype` is None / empty, omit the flag
    entirely so vllm uses its model default. These tests pin the
    behavior on both sides — DFlash profiles (null in ModelDef) must
    NOT emit the flag, and TQ/structured profiles must continue to
    emit their explicit dtype.
    """

    @staticmethod
    def _kv_lines(script: str) -> list[str]:
        return [ln for ln in script.splitlines()
                if ln.strip().startswith("--kv-cache-dtype")]

    def test_27b_dflash_omits_kv_cache_dtype(self):
        """Q27-DFlash declares `kv_cache_dtype: null` → renderer omits
        the flag entirely. The pre-fix rendered launcher was
        `--kv-cache-dtype None`, which vllm argparse rejects."""
        script = render_profile_launcher(
            "qwen3.6-27b-dflash",
            hardware_id="a5000-2x-24gbvram-16cpu-128gbram",
        )
        assert "--kv-cache-dtype None" not in script, (
            "renderer must NOT emit `--kv-cache-dtype None` — vllm "
            "argparse rejects the literal string"
        )
        assert self._kv_lines(script) == [], (
            "renderer must omit `--kv-cache-dtype` entirely when the "
            "ModelDef declares null; let vllm pick its model default"
        )

    def test_35b_dflash_omits_kv_cache_dtype(self):
        """Q35-A3B-FP8-DFlash also declares `kv_cache_dtype: null` for
        the same DFlash head_size=256 constraint."""
        script = render_profile_launcher(
            "qwen3.6-35b-dflash",
            hardware_id="a5000-2x-24gbvram-16cpu-128gbram",
        )
        assert "--kv-cache-dtype None" not in script
        assert self._kv_lines(script) == []

    def test_35b_balanced_still_emits_turboquant_k8v4(self):
        """Q35-balanced uses TQ k8v4 KV — the fix must NOT regress
        explicit-dtype profiles."""
        script = render_profile_launcher(
            "qwen3.6-35b-balanced",
            hardware_id="a5000-2x-24gbvram-16cpu-128gbram",
        )
        assert "  --kv-cache-dtype turboquant_k8v4 \\" in script

    def test_27b_tq_k8v4_still_emits_turboquant_k8v4(self):
        script = render_profile_launcher(
            "qwen3.6-27b-tq-k8v4",
            hardware_id="a5000-2x-24gbvram-16cpu-128gbram",
        )
        assert "  --kv-cache-dtype turboquant_k8v4 \\" in script

    def test_gemma4_structured_still_emits_turboquant_4bit_nc(self):
        script = render_profile_launcher(
            "gemma4-31b-tq-mtp-structured-k4",
            hardware_id="a5000-2x-24gbvram-16cpu-128gbram",
        )
        assert "  --kv-cache-dtype turboquant_4bit_nc \\" in script

    def test_rendered_dtype_matches_vllm_argparse_choices(self):
        """When emitted, the dtype string must be one of vllm's
        accepted choices (no Python sentinels, no typos). This is a
        cross-cutting regression guard: a future profile that
        accidentally serializes a non-string sentinel would fail this
        check before the operator hits argparse on the server."""
        # The set is a copy of what dev371's vllm CLI shipped (see
        # P2_Q27_DFLASH_DEV371_SMOKE_RECEIPT). Add new TQ variants
        # here when upstream / Genesis adds them.
        VALID_DTYPES = {
            "auto", "bfloat16", "float16",
            "fp8", "fp8_ds_mla", "fp8_e4m3", "fp8_e5m2",
            "fp8_inc", "fp8_per_token_head", "int8_per_token_head",
            "nvfp4",
            "turboquant_3bit_nc", "turboquant_4bit_nc",
            "turboquant_k3v4_nc", "turboquant_k8v4",
        }
        for profile_id in (
            "qwen3.6-35b-balanced", "qwen3.6-27b-tq-k8v4",
            "gemma4-31b-tq-mtp-structured-k4", "gemma4-31b-tq-default",
        ):
            script = render_profile_launcher(
                profile_id,
                hardware_id="a5000-2x-24gbvram-16cpu-128gbram",
            )
            kv_lines = self._kv_lines(script)
            # Either omitted (null path) or exactly one valid choice
            assert len(kv_lines) <= 1, (
                f"{profile_id!r}: multiple --kv-cache-dtype lines"
            )
            if not kv_lines:
                continue
            # parse the value
            value = kv_lines[0].strip().split()[1]
            assert value in VALID_DTYPES, (
                f"{profile_id!r}: rendered --kv-cache-dtype "
                f"{value!r} not in vllm's accepted set "
                f"{sorted(VALID_DTYPES)}"
            )


# ─── Backend mapping table tests ────────────────────────────────────────


class TestBackendMapping:
    def test_backend_plan_map_immutable_known_pairs(self):
        """If this fails, somebody added/removed entries in
        BACKEND_PLAN_EMISSION_MAP. Audit before committing — the map
        is the contract surface for BOTH compose emission AND render
        consistency validation.

        P1.8 refactored the map shape from str|None to dict[str,str]|None
        to support multi-env mappings (drafter_kv_sharing emits BOTH
        SNDR canonical + GENESIS legacy alias)."""
        assert _BACKEND_PLAN_MAP[
            ("drafter_sliding", "TRITON_ATTN")
        ] == {"GENESIS_ENABLE_G4_71B_DRAFTER_SLIDING_TRITON": "1"}
        assert _BACKEND_PLAN_MAP[
            ("drafter_full", "TRITON_ATTN")
        ] == {"GENESIS_ENABLE_G4_75_DRAFTER_HEAD512_TRITON": "1"}
        assert _BACKEND_PLAN_MAP[("target_default", "TURBOQUANT")] is None
        assert _BACKEND_PLAN_MAP[("target_native_layers", "TRITON_ATTN")] is None
        # P1.8 drafter_kv_sharing
        assert _BACKEND_PLAN_MAP[("drafter_kv_sharing", "physical")] == {
            "SNDR_ENABLE_G4_76_DISABLE_DRAFTER_KV_SHARING": "0",
            "GENESIS_ENABLE_G4_76_DISABLE_DRAFTER_KV_SHARING": "0",
        }
        assert _BACKEND_PLAN_MAP[("drafter_kv_sharing", "disabled")] is None

    def test_g10_unknown_backend_value_raises(self, monkeypatch):
        """G10: a backend_plan value not in the mapping table must raise
        SchemaError. Mocks load_profile to return a synthetic profile
        with a bogus drafter_sliding value."""
        from sndr.model_configs.schema_v2 import (
            BackendPlanConfig, ProfileDef, PatchesDelta,
        )
        from sndr.model_configs.schema import SchemaError

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
            from sndr.model_configs import registry_v2 as real
            return real.load_profile(pid)

        monkeypatch.setattr(
            "sndr.model_configs.registry_v2.load_profile",
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
        from sndr.model_configs.schema_v2 import (
            BackendPlanConfig, ProfileDef, PatchesDelta,
        )
        from sndr.model_configs.schema import SchemaError

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
            from sndr.model_configs import registry_v2 as real
            return real.load_profile(pid)

        monkeypatch.setattr(
            "sndr.model_configs.registry_v2.load_profile",
            fake_load,
        )

        # P1.8 (2026-05-21): compose now AUTO-emits the env from
        # backend_plan declaration via render_backend_env(), so the
        # consistency check no longer fires for the "operator forgot
        # patches_delta.enable" case — that bug class is eliminated
        # by construction. The previous SchemaError-raising path stays
        # as a defensive check for a different failure mode: if some
        # OTHER code path overrides the env to a non-expected value
        # (e.g. patches_delta.disable removes it after compose adds it).
        # Verified by asserting compose's auto-emit puts the env in
        # place and the rendered launcher contains it.
        rendered = render_profile_launcher(bad_profile.id)
        assert "GENESIS_ENABLE_G4_71B_DRAFTER_SLIDING_TRITON=1" in rendered, (
            "P1.8: compose must auto-emit the G4_71b env from "
            "backend_plan.drafter_sliding=TRITON_ATTN even when "
            "patches_delta.enable is empty"
        )


# ─── G08 + G09: output file handling ────────────────────────────────────


class TestOutputFlags:
    def test_g08_output_writes_file(self, tmp_path):
        """--output DIR writes start_<profile_id>.sh into DIR."""
        result = subprocess.run(
            [
                sys.executable, "-m", "sndr.cli.legacy",
                "profile", "render-launchers",
                "gemma4-31b-tq-default",
                "--output", str(tmp_path),
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"expected 0; got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        out_file = tmp_path / "start_gemma4-31b-tq-default.sh"
        assert out_file.exists(), f"{out_file} not created"
        assert out_file.stat().st_mode & 0o111, "output file not executable"
        content = out_file.read_text()
        assert content.startswith("#!/bin/bash")

    def test_g09_overwrite_without_force_fails(self, tmp_path):
        """Pre-existing target file + no --force → exit 1, no write."""
        # Pre-create a sentinel file at the target path
        target = tmp_path / "start_gemma4-31b-tq-default.sh"
        target.write_text("# preexisting sentinel\n")
        before_mtime = target.stat().st_mtime

        result = subprocess.run(
            [
                sys.executable, "-m", "sndr.cli.legacy",
                "profile", "render-launchers",
                "gemma4-31b-tq-default",
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
        target = tmp_path / "start_gemma4-31b-tq-default.sh"
        target.write_text("# preexisting sentinel\n")

        result = subprocess.run(
            [
                sys.executable, "-m", "sndr.cli.legacy",
                "profile", "render-launchers",
                "gemma4-31b-tq-default",
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
                sys.executable, "-m", "sndr.cli.legacy",
                "profile", "render-launchers",
                "gemma4-31b-tq-default",
                "--output", str(tmp_path),
                "--dry-run",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        out_file = tmp_path / "start_gemma4-31b-tq-default.sh"
        assert not out_file.exists(), "dry-run should not write file"
        assert result.stdout.startswith("#!/bin/bash")

    def test_default_is_dry_run(self):
        """No --output, no --dry-run → defaults to stdout (dry-run-like)."""
        result = subprocess.run(
            [
                sys.executable, "-m", "sndr.cli.legacy",
                "profile", "render-launchers",
                "gemma4-31b-tq-default",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert result.stdout.startswith("#!/bin/bash")


# ─── Backend consistency check function (direct) ────────────────────────


# ─── P1.8 regression gate ───────────────────────────────────────────────


class TestP18ArtifactLookupRegression:
    """P1.8 regression gate — would have caught the 2026-05-21 C2 failure.

    The Gemma4 mapping provider's artifact_lookup_keys() has 8 implicit
    predicates; one of them (kv_sharing_on) reads
    GENESIS_ENABLE_G4_76_DISABLE_DRAFTER_KV_SHARING with default="1",
    so the env MUST be explicitly emitted as "0" to opt into the
    validated β'-A K=4 path. Pre-P1.8 the V2 structured profile did
    not declare this, the env was unset, and artifact_lookup_keys()
    returned None at boot — even though the rendered launcher was
    structurally correct otherwise.

    These tests assert the symptom + the fix at compose time, before
    any server-side rehearsal.
    """

    def test_structured_profile_declares_physical_kv_sharing(self):
        """The structured profile YAML MUST declare drafter_kv_sharing:
        physical. Without this declaration the compose layer doesn't
        know to emit the G4_76=0 envs, and the artifact lookup at the
        guard would return None."""
        from sndr.model_configs.registry_v2 import load_profile
        p = load_profile("gemma4-31b-tq-mtp-structured-k4")
        assert p.backend_plan is not None
        assert p.backend_plan.drafter_kv_sharing == "physical", (
            "structured profile must declare drafter_kv_sharing=physical; "
            "without it the Gemma4MappingProvider.artifact_lookup_keys() "
            "returns None at boot and MTP is disabled by the safety guard."
        )

    def test_compose_emits_g4_76_disable_zero(self):
        """compose() must emit BOTH SNDR + GENESIS aliases of the
        G4_76 disable env with value '0'. This is the actual env the
        mapping provider reads to decide kv_sharing_on=True."""
        from sndr.model_configs.compose import compose
        from sndr.model_configs.registry_v2 import (
            load_hardware, load_model, load_profile,
        )
        p = load_profile("gemma4-31b-tq-mtp-structured-k4")
        m = load_model("gemma-4-31b-it-awq")
        hw = load_hardware("a5000-2x-24gbvram-16cpu-128gbram")
        cfg = compose(m, hw, p)
        assert cfg.genesis_env.get(
            "SNDR_ENABLE_G4_76_DISABLE_DRAFTER_KV_SHARING"
        ) == "0"
        assert cfg.genesis_env.get(
            "GENESIS_ENABLE_G4_76_DISABLE_DRAFTER_KV_SHARING"
        ) == "0"

    def test_render_emits_g4_76_disable_zero(self):
        """The rendered launcher (operator-facing) MUST contain both
        G4_76 disable envs at value 0. Without them the C2 verdict
        gate fails at boot."""
        script = render_profile_launcher("gemma4-31b-tq-mtp-structured-k4")
        assert "-e SNDR_ENABLE_G4_76_DISABLE_DRAFTER_KV_SHARING=0" in script
        assert "-e GENESIS_ENABLE_G4_76_DISABLE_DRAFTER_KV_SHARING=0" in script

    def test_disabled_value_emits_no_g4_76_env(self):
        """drafter_kv_sharing=disabled is the explicit opt-out; it must
        NOT emit the G4_76 env (runtime default ='1' is the behavior).
        Verified by constructing a synthetic disabled-sharing profile
        and asserting absence."""
        from sndr.model_configs.compose import compose
        from sndr.model_configs.registry_v2 import (
            load_hardware, load_model,
        )
        from sndr.model_configs.schema_v2 import (
            BackendPlanConfig, PatchesDelta, ProfileDef,
        )
        synthetic = ProfileDef(
            schema_version=2, kind="profile",
            id="synthetic-disabled-sharing",
            parent_model="gemma-4-31b-it-awq",
            maintainer="tests",
            status="experimental",
            patches_delta=PatchesDelta(),
            role="default",
            backend_plan=BackendPlanConfig(
                target_default="TURBOQUANT",
                drafter_kv_sharing="disabled",
            ),
        )
        m = load_model("gemma-4-31b-it-awq")
        hw = load_hardware("a5000-2x-24gbvram-16cpu-128gbram")
        cfg = compose(m, hw, synthetic)
        assert "SNDR_ENABLE_G4_76_DISABLE_DRAFTER_KV_SHARING" not in cfg.genesis_env
        assert "GENESIS_ENABLE_G4_76_DISABLE_DRAFTER_KV_SHARING" not in cfg.genesis_env

    def test_unknown_kv_sharing_value_raises(self, monkeypatch):
        """drafter_kv_sharing must be one of {None, physical, disabled};
        anything else is SchemaError at validate() time."""
        from sndr.model_configs.schema import SchemaError
        from sndr.model_configs.schema_v2 import (
            BackendPlanConfig,
        )
        bp = BackendPlanConfig(drafter_kv_sharing="shared")  # type: ignore[arg-type]
        with pytest.raises(SchemaError, match="drafter_kv_sharing"):
            bp.validate()

    def test_artifact_lookup_keys_would_match(self):
        """Higher-order assertion: after compose, the cfg.genesis_env
        carries the env values that the mapping provider's predicate
        needs. We don't actually call artifact_lookup_keys() here
        (it requires a torch-loaded vllm_config) but we assert the
        envs it reads are all set as expected.

        The predicate chain in gemma4.py:347-360:
          * GENESIS_G4_TQ_FORCE_SKIP_LAYERS == '58,59'         → kv_share_targets+skip_layers
          * GENESIS_ENABLE_G4_71B_DRAFTER_SLIDING_TRITON == '1' → g71b_on
          * GENESIS_ENABLE_G4_75_DRAFTER_HEAD512_TRITON == '1' → g75_on
          * GENESIS_ENABLE_G4_76_DISABLE_DRAFTER_KV_SHARING == '0' → kv_sharing_on
          * GENESIS_ENABLE_G4_78_DRAFTER_TARGET_KV_BRIDGE not '1' → not bridge_on
          * (mtp_k=4 comes from cfg.spec_decode, not env)
        """
        from sndr.model_configs.compose import compose
        from sndr.model_configs.registry_v2 import (
            load_hardware, load_model, load_profile,
        )
        p = load_profile("gemma4-31b-tq-mtp-structured-k4")
        m = load_model("gemma-4-31b-it-awq")
        hw = load_hardware("a5000-2x-24gbvram-16cpu-128gbram")
        cfg = compose(m, hw, p)
        env = cfg.genesis_env

        # All five predicate envs match what the mapping provider expects
        assert env.get("GENESIS_G4_TQ_FORCE_SKIP_LAYERS") == "58,59"
        assert env.get("GENESIS_ENABLE_G4_71B_DRAFTER_SLIDING_TRITON") == "1"
        assert env.get("GENESIS_ENABLE_G4_75_DRAFTER_HEAD512_TRITON") == "1"
        assert env.get("GENESIS_ENABLE_G4_76_DISABLE_DRAFTER_KV_SHARING") == "0"
        # bridge_on must be False — env either unset or != "1"
        assert env.get("GENESIS_ENABLE_G4_78_DRAFTER_TARGET_KV_BRIDGE", "0") != "1"
        # mtp_k=4 from spec_decode
        assert cfg.spec_decode is not None
        assert cfg.spec_decode.num_speculative_tokens == 4


class TestValidateBackendPlanConsistency:
    def test_no_backend_plan_is_noop(self):
        from sndr.model_configs.schema_v2 import (
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
        from sndr.model_configs.schema_v2 import (
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
