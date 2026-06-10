# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr config-keys-*` — canonical env-key registry (§6.7).

Contract: validator catches unknown Genesis/SNDR env keys; non-Genesis
keys (PYTORCH_/NCCL_/VLLM_/etc.) pass through; every committed builtin
YAML validates clean.
"""
from __future__ import annotations

import argparse
import io
import json
import textwrap
from contextlib import redirect_stdout
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _run(handler, opts: argparse.Namespace) -> tuple[int, str]:
    buf = io.StringIO()
    rc_holder = {"rc": None}
    with redirect_stdout(buf):
        rc_holder["rc"] = handler(opts)
    return rc_holder["rc"], buf.getvalue()


# ─── load_canonical_registry ───────────────────────────────────────────


class TestCanonicalRegistry:
    def test_registry_has_keys_from_patch_registry(self):
        from sndr.cli.legacy.config_keys import load_canonical_registry
        canon = load_canonical_registry()
        # Known patch toggle from dispatcher.registry.PATCH_REGISTRY.
        assert "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL" in canon
        meta = canon["GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL"]
        assert meta["source"] == "registry"
        assert "patch_id" in meta
        assert "family" in meta

    def test_registry_has_v2_tuning_knob(self):
        """V2 model yamls add secondary tuning knobs like P67_NUM_KV_SPLITS."""
        from sndr.cli.legacy.config_keys import load_canonical_registry
        canon = load_canonical_registry()
        assert "GENESIS_P67_NUM_KV_SPLITS" in canon
        # Source could be registry (if metadata added) or v2 (yaml-only).
        assert canon["GENESIS_P67_NUM_KV_SPLITS"]["source"] in ("v2", "registry")

    def test_registry_has_policy_key(self):
        """Policy keys (non-patch Genesis env) are in the canonical set."""
        from sndr.cli.legacy.config_keys import load_canonical_registry
        canon = load_canonical_registry()
        assert "GENESIS_VLLM_PIN_POLICY" in canon
        assert canon["GENESIS_VLLM_PIN_POLICY"]["source"] == "policy"

    def test_total_count_sanity(self):
        from sndr.cli.legacy.config_keys import load_canonical_registry
        canon = load_canonical_registry()
        # >130 — patch registry alone has ~135. Sanity check.
        assert len(canon) >= 130


# ─── run_list ──────────────────────────────────────────────────────────


class TestRunList:
    def test_list_text_mode(self):
        from sndr.cli.legacy.config_keys import run_list
        opts = argparse.Namespace(source=None, json=False)
        rc, out = _run(run_list, opts)
        assert rc == 0
        assert "Total:" in out
        assert "registry" in out

    def test_list_json_count_matches_canon(self):
        from sndr.cli.legacy.config_keys import (
            load_canonical_registry, run_list,
        )
        opts = argparse.Namespace(source=None, json=True)
        rc, out = _run(run_list, opts)
        assert rc == 0
        payload = json.loads(out)
        assert payload["count"] == len(load_canonical_registry())

    def test_filter_by_source_registry(self):
        from sndr.cli.legacy.config_keys import run_list
        opts = argparse.Namespace(source="registry", json=True)
        rc, out = _run(run_list, opts)
        assert rc == 0
        payload = json.loads(out)
        # Every returned entry has source=registry.
        assert all(k["source"] == "registry" for k in payload["keys"])

    def test_filter_by_source_policy(self):
        from sndr.cli.legacy.config_keys import run_list
        opts = argparse.Namespace(source="policy", json=True)
        rc, out = _run(run_list, opts)
        assert rc == 0
        payload = json.loads(out)
        ids = {k["key"] for k in payload["keys"]}
        assert "GENESIS_VLLM_PIN_POLICY" in ids


# ─── run_describe ──────────────────────────────────────────────────────


class TestRunDescribe:
    def test_describe_known_key(self):
        from sndr.cli.legacy.config_keys import run_describe
        opts = argparse.Namespace(
            key="GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL", json=False,
        )
        rc, out = _run(run_describe, opts)
        assert rc == 0
        assert "source:" in out
        assert "registry" in out

    def test_describe_unknown_key_with_suggestion(self):
        from sndr.cli.legacy.config_keys import run_describe
        # Typo of the real key — should appear in suggestions.
        opts = argparse.Namespace(
            key="GENESIS_ENABLE_P67_TYPO", json=True,
        )
        rc, out = _run(run_describe, opts)
        assert rc == 1
        payload = json.loads(out)
        assert payload["known"] is False
        # Suggestion should include the real P67 key.
        assert any("P67" in s for s in payload["suggestions"])

    def test_describe_completely_unknown_key(self):
        from sndr.cli.legacy.config_keys import run_describe
        opts = argparse.Namespace(
            key="GENESIS_TOTALLY_NONEXISTENT_XYZ_999", json=True,
        )
        rc, out = _run(run_describe, opts)
        assert rc == 1
        payload = json.loads(out)
        assert payload["known"] is False


# ─── run_validate ──────────────────────────────────────────────────────


class TestRunValidate:
    def test_validate_known_model_yaml(self):
        """A committed V2 model YAML must validate clean."""
        from sndr.cli.legacy.config_keys import run_validate
        opts = argparse.Namespace(
            yaml_file=str(
                REPO_ROOT
                / "vllm" / "sndr_core" / "model_configs" / "builtin"
                / "model" / "qwen3.6-35b-a3b-fp8.yaml"
            ),
            json=True,
        )
        rc, out = _run(run_validate, opts)
        assert rc == 0
        payload = json.loads(out)
        assert payload["passed"] is True
        assert payload["unknown_keys"] == []

    def test_validate_catches_unknown_genesis_key(self, tmp_path):
        from sndr.cli.legacy.config_keys import run_validate
        yaml = textwrap.dedent("""\
            schema_version: 2
            kind: model
            id: test
            patches:
              GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL: '1'
              GENESIS_ENABLE_PXXX_TOTALLY_FAKE_PATCH: '1'
              GENESIS_TYPO_P58: '1'
        """)
        p = tmp_path / "bad.yaml"
        p.write_text(yaml, encoding="utf-8")
        opts = argparse.Namespace(yaml_file=str(p), json=True)
        rc, out = _run(run_validate, opts)
        assert rc == 1
        payload = json.loads(out)
        assert payload["passed"] is False
        # Two unknown Genesis keys caught.
        assert len(payload["unknown_keys"]) == 2
        assert "GENESIS_ENABLE_PXXX_TOTALLY_FAKE_PATCH" in payload["unknown_keys"]
        assert "GENESIS_TYPO_P58" in payload["unknown_keys"]

    def test_validate_ignores_non_genesis_keys(self, tmp_path):
        """PYTORCH_/NCCL_/VLLM_/CUDA_/OMP_/TRITON_ keys are NOT validated
        because they belong to upstream tooling (canonical registry only
        owns Genesis/SNDR keys)."""
        from sndr.cli.legacy.config_keys import run_validate
        yaml = textwrap.dedent("""\
            schema_version: 1
            key: test-cfg
            system_env:
              PYTORCH_CUDA_ALLOC_CONF: 'expandable_segments:True'
              NCCL_P2P_DISABLE: '1'
              VLLM_LOGGING_LEVEL: WARNING
              CUDA_DEVICE_MAX_CONNECTIONS: '8'
              OMP_NUM_THREADS: '1'
              TRITON_CACHE_DIR: /tmp/triton
        """)
        p = tmp_path / "syskeys.yaml"
        p.write_text(yaml, encoding="utf-8")
        opts = argparse.Namespace(yaml_file=str(p), json=True)
        rc, out = _run(run_validate, opts)
        assert rc == 0     # passes — non-Genesis keys ignored
        payload = json.loads(out)
        assert payload["unknown_keys"] == []
        # Counter accounting: every key counted as non-Genesis.
        assert payload["genesis_keys"] == 0
        assert payload["non_genesis_keys"] == 6

    def test_validate_walks_profile_delta(self, tmp_path):
        """V2 profile YAMLs put keys under patches_delta.enable/disable/override."""
        from sndr.cli.legacy.config_keys import run_validate
        yaml = textwrap.dedent("""\
            schema_version: 2
            kind: profile
            id: test-profile
            patches_delta:
              enable:
                GENESIS_ENABLE_PN999_FAKE_NEW: '1'
              disable:
                - GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL
              override:
                GENESIS_TYPO_OVERRIDE: '0.5'
        """)
        p = tmp_path / "profile.yaml"
        p.write_text(yaml, encoding="utf-8")
        opts = argparse.Namespace(yaml_file=str(p), json=True)
        rc, out = _run(run_validate, opts)
        assert rc == 1
        payload = json.loads(out)
        # Two unknown keys (the disable one is real; the typo is unknown,
        # the new one PN999 is unknown).
        unknown = set(payload["unknown_keys"])
        assert "GENESIS_ENABLE_PN999_FAKE_NEW" in unknown
        assert "GENESIS_TYPO_OVERRIDE" in unknown
        # The disabled real key must NOT be flagged.
        assert "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL" not in unknown

    def test_validate_missing_file(self):
        from sndr.cli.legacy.config_keys import run_validate
        opts = argparse.Namespace(
            yaml_file="/tmp/totally-nonexistent-file-xyz.yaml",
            json=False,
        )
        rc, _ = _run(run_validate, opts)
        assert rc == 2


# ─── Sweep — every committed builtin YAML validates clean ─────────────


class TestBuiltinSweep:
    @pytest.mark.parametrize("yaml_path", [
        # All V2 model yamls (6)
        "vllm/sndr_core/model_configs/builtin/model/qwen3.6-27b-dflash.yaml",
        "vllm/sndr_core/model_configs/builtin/model/qwen3.6-27b-int4-autoround-fp8kv.yaml",
        "vllm/sndr_core/model_configs/builtin/model/qwen3.6-27b-int4-autoround-tq-k8v4.yaml",
        "vllm/sndr_core/model_configs/builtin/model/qwen3.6-35b-a3b-fp8-dflash.yaml",
        "vllm/sndr_core/model_configs/builtin/model/qwen3.6-35b-a3b-fp8.yaml",
        "vllm/sndr_core/model_configs/builtin/model/qwen3.6-7b-dense.yaml",
        # All V2 profile yamls (11)
        "vllm/sndr_core/model_configs/builtin/profile/wave9-balanced.yaml",
        "vllm/sndr_core/model_configs/builtin/profile/wave9-qwen3.6-27b-tq-k8v4.yaml",
        "vllm/sndr_core/model_configs/builtin/profile/wave9-qwen3.6-27b-dflash.yaml",
        "vllm/sndr_core/model_configs/builtin/profile/wave9-35b-fp8-dflash.yaml",
        "vllm/sndr_core/model_configs/builtin/profile/wave9-27b-fp8kv-long-ctx.yaml",
        "vllm/sndr_core/model_configs/builtin/profile/qa-27b-fp8kv-tested.yaml",
        "vllm/sndr_core/model_configs/builtin/profile/qa-27b-tq-1x-tested.yaml",
        "vllm/sndr_core/model_configs/builtin/profile/experimental-qwen3.6-27b-tq-dflash-ab.yaml",
        "vllm/sndr_core/model_configs/builtin/profile/path-c-2x-tier-aware-example.yaml",
        "vllm/sndr_core/model_configs/builtin/profile/path-a-3090-cpu-offload-example.yaml",
        "vllm/sndr_core/model_configs/builtin/profile/path-c-3090-tier-aware-example.yaml",
    ])
    def test_yaml_validates_clean(self, yaml_path):
        from sndr.cli.legacy.config_keys import run_validate
        opts = argparse.Namespace(
            yaml_file=str(REPO_ROOT / yaml_path),
            json=True,
        )
        rc, out = _run(run_validate, opts)
        assert rc == 0, (
            f"{yaml_path} did not validate clean; unknown keys: "
            f"{json.loads(out).get('unknown_keys')}"
        )


# ─── Argparser registration ───────────────────────────────────────────


class TestRegistration:
    def test_config_keys_list_registered(self):
        import argparse
        from sndr.cli.legacy.config_keys import add_argparser
        p = argparse.ArgumentParser()
        sub = p.add_subparsers(dest="cmd")
        add_argparser(sub)
        ns = p.parse_args(["config-keys-list", "--json"])
        assert ns.cmd == "config-keys-list"
        assert ns.json is True

    def test_top_level_includes_config_keys(self):
        from sndr import cli as cli_mod
        assert hasattr(cli_mod, "_config_keys_argparser")
        assert callable(cli_mod._config_keys_argparser)
