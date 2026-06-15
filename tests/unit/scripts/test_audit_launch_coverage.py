# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_launch_coverage.py` — V2 hardware schema
coverage gate (Entry 22).

Contract:

  • The 5 canonical mount slots and 7 required env keys are frozen.
  • Every committed V2 hardware YAML must cover every required slot.
  • Synthetic minus-one-slot YAML fails the audit.
  • Path-extraction tolerates quoted entries, comments, and `:mode`
    suffix variations.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_launch_coverage.py"


def _import_script():
    name = "_audit_launch_coverage_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── Canonical schema sanity ──────────────────────────────────────────


class TestCanonicalSchema:
    def test_five_required_mounts(self):
        mod = _import_script()
        # The schema is frozen — if it changes, an entry must document why.
        assert len(mod.REQUIRED_MOUNTS) == 5
        container_paths = {s.container_path for s in mod.REQUIRED_MOUNTS}
        # Spot-check the slots that were dropped in the V2 migration:
        assert "/root/.triton/cache" in container_paths
        assert "/root/.cache/vllm/torch_compile_cache" in container_paths
        assert "/plugin" in container_paths
        # The legacy vllm/sndr_core compat overlay was retired in v12.
        assert (
            "/usr/local/lib/python3.12/dist-packages/vllm/sndr_core"
            not in container_paths
        )

    def test_seven_required_env_keys(self):
        mod = _import_script()
        assert "TRITON_CACHE_DIR" in mod.REQUIRED_ENV_KEYS
        assert "VLLM_ALLOW_LONG_MAX_MODEL_LEN" in mod.REQUIRED_ENV_KEYS
        assert "VLLM_WORKER_MULTIPROC_METHOD" in mod.REQUIRED_ENV_KEYS
        # 7 hard-required keys (was decided in E22 design):
        assert len(mod.REQUIRED_ENV_KEYS) == 7


# ─── Mount-path extraction ────────────────────────────────────────────


class TestExtractContainerPaths:
    def test_simple_ro_mount(self):
        mod = _import_script()
        out = mod._extract_mount_container_paths([
            "${models_dir}:/models:ro",
        ])
        assert out == {"/models"}

    def test_rw_mount_no_mode_suffix(self):
        mod = _import_script()
        out = mod._extract_mount_container_paths([
            "${triton_cache}:/root/.triton/cache",
        ])
        assert out == {"/root/.triton/cache"}

    def test_quoted_entry(self):
        mod = _import_script()
        out = mod._extract_mount_container_paths([
            '"${plugin_src}:/plugin:ro"',
        ])
        assert out == {"/plugin"}

    def test_multiple_entries(self):
        mod = _import_script()
        out = mod._extract_mount_container_paths([
            "${models_dir}:/models:ro",
            "${hf_cache}:/root/.cache/huggingface:ro",
            "${triton_cache}:/root/.triton/cache",
        ])
        assert out == {
            "/models", "/root/.cache/huggingface", "/root/.triton/cache",
        }

    def test_non_string_entry_skipped(self):
        mod = _import_script()
        out = mod._extract_mount_container_paths(["good:/foo:ro", 42, None])
        assert out == {"/foo"}


# ─── Single-YAML audit ────────────────────────────────────────────────


def _write_yaml(p: Path, text: str) -> Path:
    p.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    return p


_ALL_CANONICAL_MOUNTS = (
    '"${models_dir}:/models:ro"',
    '"${hf_cache}:/root/.cache/huggingface:ro"',
    '"${triton_cache}:/root/.triton/cache"',
    '"${compile_cache}:/root/.cache/vllm/torch_compile_cache"',
    '"${plugin_src}:/plugin:ro"',
)

_ALL_CANONICAL_ENVS = {
    "PYTORCH_CUDA_ALLOC_CONF": "'expandable_segments:True'",
    "OMP_NUM_THREADS": "'1'",
    "CUDA_DEVICE_MAX_CONNECTIONS": "'8'",
    "TRITON_CACHE_DIR": "'/root/.triton/cache'",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "'1'",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_NO_USAGE_STATS": "'1'",
}


def _make_yaml(*, mounts: tuple = None, envs: dict = None) -> str:
    """Build a syntactically-valid hardware YAML with the given mount/env
    set. Passing None uses the full canonical set."""
    ms = _ALL_CANONICAL_MOUNTS if mounts is None else mounts
    es = _ALL_CANONICAL_ENVS if envs is None else envs
    mount_lines = "\n".join(f"      - {m}" for m in ms)
    env_lines = "\n".join(f"  {k}: {v}" for k, v in es.items())
    return (
        "schema_version: 2\n"
        "kind: hardware\n"
        "id: synth-hardware\n"
        "runtime:\n"
        "  default: docker\n"
        "  supported: [docker]\n"
        "  docker:\n"
        "    image: vllm/vllm-openai:nightly\n"
        "    mounts:\n"
        f"{mount_lines}\n"
        "system_env:\n"
        f"{env_lines}\n"
    )


def _full_canonical_yaml() -> str:
    return _make_yaml()


class TestAuditOneYaml:
    def test_canonical_complete_passes(self, tmp_path):
        mod = _import_script()
        y = _write_yaml(tmp_path / "h.yaml", _full_canonical_yaml())
        r = mod.audit_one_hardware_yaml(y)
        assert r.passed is True
        assert r.missing_mounts == []
        assert r.missing_envs == []

    def test_missing_triton_mount_fails(self, tmp_path):
        mod = _import_script()
        # Drop only the triton mount; keep everything else canonical.
        kept = tuple(
            m for m in _ALL_CANONICAL_MOUNTS
            if "/root/.triton/cache" not in m
        )
        y = _write_yaml(tmp_path / "h.yaml", _make_yaml(mounts=kept))
        r = mod.audit_one_hardware_yaml(y)
        assert r.parse_error == "", r.parse_error
        assert r.passed is False
        assert "/root/.triton/cache" in r.missing_mounts

    def test_missing_env_key_fails(self, tmp_path):
        mod = _import_script()
        envs = {k: v for k, v in _ALL_CANONICAL_ENVS.items()
                if k != "TRITON_CACHE_DIR"}
        y = _write_yaml(tmp_path / "h.yaml", _make_yaml(envs=envs))
        r = mod.audit_one_hardware_yaml(y)
        assert r.parse_error == "", r.parse_error
        assert r.passed is False
        assert "TRITON_CACHE_DIR" in r.missing_envs

    def test_parse_error_recorded(self, tmp_path):
        mod = _import_script()
        bad = tmp_path / "broken.yaml"
        # Pathological YAML that the parser actually rejects.
        bad.write_text("foo:\n  - [\n", encoding="utf-8")
        r = mod.audit_one_hardware_yaml(bad)
        assert not r.passed
        assert r.parse_error != ""


# ─── E24: env-value invariants ────────────────────────────────────────


class TestEnvValueInvariants:
    def test_canonical_values_pass(self, tmp_path):
        mod = _import_script()
        y = _write_yaml(tmp_path / "h.yaml", _full_canonical_yaml())
        r = mod.audit_one_hardware_yaml(y)
        assert r.env_value_violations == []
        assert r.passed is True

    def test_triton_cache_dir_wrong_path_fails(self, tmp_path):
        mod = _import_script()
        envs = dict(_ALL_CANONICAL_ENVS)
        envs["TRITON_CACHE_DIR"] = "'/wrong/triton'"
        y = _write_yaml(tmp_path / "h.yaml", _make_yaml(envs=envs))
        r = mod.audit_one_hardware_yaml(y)
        assert r.passed is False
        keys = [v[0] for v in r.env_value_violations]
        assert "TRITON_CACHE_DIR" in keys

    def test_multiproc_method_fork_fails(self, tmp_path):
        mod = _import_script()
        envs = dict(_ALL_CANONICAL_ENVS)
        envs["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
        y = _write_yaml(tmp_path / "h.yaml", _make_yaml(envs=envs))
        r = mod.audit_one_hardware_yaml(y)
        assert r.passed is False
        keys = [v[0] for v in r.env_value_violations]
        assert "VLLM_WORKER_MULTIPROC_METHOD" in keys

    def test_long_max_model_len_zero_fails(self, tmp_path):
        mod = _import_script()
        envs = dict(_ALL_CANONICAL_ENVS)
        envs["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "'0'"
        y = _write_yaml(tmp_path / "h.yaml", _make_yaml(envs=envs))
        r = mod.audit_one_hardware_yaml(y)
        assert r.passed is False
        keys = [v[0] for v in r.env_value_violations]
        assert "VLLM_ALLOW_LONG_MAX_MODEL_LEN" in keys

    def test_truthy_variants_accepted(self, tmp_path):
        """Boolean-like keys accept '1' / 'true' / 'True'."""
        mod = _import_script()
        for accepted in ("'1'", "'true'", "'True'"):
            envs = dict(_ALL_CANONICAL_ENVS)
            envs["VLLM_NO_USAGE_STATS"] = accepted
            envs["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = accepted
            y = _write_yaml(tmp_path / "h.yaml", _make_yaml(envs=envs))
            r = mod.audit_one_hardware_yaml(y)
            assert r.passed is True, (
                f"value {accepted!r} not accepted: {r.env_value_violations}"
            )

    def test_normalize_handles_quoting(self):
        mod = _import_script()
        assert mod._normalize_env_value("spawn") == "spawn"
        assert mod._normalize_env_value("'spawn'") == "spawn"
        assert mod._normalize_env_value('"spawn"') == "spawn"
        assert mod._normalize_env_value(1) == "1"
        assert mod._normalize_env_value(True) == "True"

    def test_missing_value_link_key_not_violated(self, tmp_path):
        """If the linked key is missing entirely, that's a missing-env
        failure (E22), not a value violation."""
        mod = _import_script()
        envs = {k: v for k, v in _ALL_CANONICAL_ENVS.items()
                if k != "TRITON_CACHE_DIR"}
        y = _write_yaml(tmp_path / "h.yaml", _make_yaml(envs=envs))
        r = mod.audit_one_hardware_yaml(y)
        # missing_envs handles this, not env_value_violations.
        assert "TRITON_CACHE_DIR" in r.missing_envs
        assert all(k != "TRITON_CACHE_DIR" for k, _, _ in r.env_value_violations)


# ─── Live repo — committed hardware YAMLs must all pass ───────────────


class TestLiveRepo:
    def test_all_committed_hardware_pass(self):
        """After Entry 22 mount restoration, every committed V2 hardware
        YAML must satisfy the canonical schema. This is the regression
        anchor — if a future PR drops a mount, this test breaks."""
        mod = _import_script()
        results = mod.audit_launch_coverage()
        failed = [r for r in results if not r.passed]
        assert failed == [], (
            "V2 hardware files missing canonical slots:\n"
            + "\n".join(
                f"  {r.hardware_id}: mounts={r.missing_mounts}, "
                f"envs={r.missing_envs}"
                for r in failed
            )
        )
        # Sanity: at least 3 hardware files exist (a5000-1x, a5000-2x, 3090).
        assert len(results) >= 3


# ─── Script CLI ────────────────────────────────────────────────────────


class TestScriptCLI:
    def test_cli_exits_zero_on_committed_repo(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout

    def test_cli_json_shape(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert "required_mounts" in payload
        assert "required_env_keys" in payload
        assert "results" in payload
        assert payload["failed"] == 0
        # The schema documentation is part of the JSON output — operators
        # can `cat` it to see what's required.
        cps = {m["container_path"] for m in payload["required_mounts"]}
        assert "/root/.triton/cache" in cps

    def test_cli_synth_broken(self, tmp_path):
        bad = tmp_path / "broken.yaml"
        bad.write_text(textwrap.dedent("""
            id: synth-broken
            runtime:
              docker:
                mounts:
                  - "${models_dir}:/models:ro"
            system_env:
              PYTORCH_CUDA_ALLOC_CONF: 'x'
        """).lstrip("\n"), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--hw-dir", str(tmp_path), "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 1
        payload = json.loads(result.stdout)
        assert payload["failed"] == 1
        r = payload["results"][0]
        assert len(r["missing_mounts"]) == 4   # all except /models
        assert len(r["missing_envs"]) >= 6
