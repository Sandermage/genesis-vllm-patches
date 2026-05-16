# SPDX-License-Identifier: Apache-2.0
"""`sndr model-config new --from-running` — captor contract.

The flag is the docker-inspect-based captor that reverse-engineers a
ModelConfig YAML from a running vLLM container. These tests pin the
operator-facing surface:

  - `model-config new --help` advertises the flag (audit C2 closure
    2026-05-16: was previously hidden via argparse.SUPPRESS while the
    captor was a stub).
  - The captor module is importable, exposes capture_from_running, and
    rejects non-vllm containers cleanly.
  - Argument parsing of a captured `vllm serve …` argv into a
    ModelConfig works end-to-end (synthetic inspect record).
"""
from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout

import pytest


def _run_model_config(argv):
    """Invoke the model-config CLI in-process; capture (rc, stdout, stderr)."""
    from vllm.sndr_core.compat import model_config_cli

    out = io.StringIO()
    err = io.StringIO()
    rc = 0
    try:
        with redirect_stdout(out), redirect_stderr(err):
            rc = model_config_cli.main(argv)
    except SystemExit as e:
        rc = int(e.code) if isinstance(e.code, int) else 2
    return rc, out.getvalue(), err.getvalue()


class TestFromRunningAdvertised:
    def test_new_help_advertises_from_running(self):
        """After audit C2 closure the captor is implemented; --from-running
        must be visible in `model-config new --help`."""
        rc, out, err = _run_model_config(["new", "--help"])
        text = out + err
        assert "--from-running" in text, (
            "--from-running should be visible in --help after the "
            "docker-inspect captor landed (audit C2 closure 2026-05-16)"
        )


class TestCaptorModuleContract:
    def test_module_importable(self):
        from vllm.sndr_core.compat import from_running  # noqa: F401

    def test_exposes_capture_function(self):
        from vllm.sndr_core.compat.from_running import (
            CaptureError, capture_from_running,
        )
        assert callable(capture_from_running)
        assert issubclass(CaptureError, RuntimeError)

    def test_parse_serve_args_extracts_canonical_flags(self):
        from vllm.sndr_core.compat.from_running import _parse_serve_args
        argv = [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", "/models/qwen3-test",
            "--tensor-parallel-size", "2",
            "--gpu-memory-utilization", "0.92",
            "--max-model-len", "65536",
            "--max-num-seqs", "4",
            "--max-num-batched-tokens", "8192",
            "--dtype", "bfloat16",
            "--kv-cache-dtype", "fp8_e4m3",
            "--quantization", "fp8",
            "--tool-call-parser", "qwen3coder",
            "--reasoning-parser", "qwen3",
            "--enable-chunked-prefill",
            "--trust-remote-code",
            "--enable-auto-tool-choice",
            "--port", "8000",
            "--host", "0.0.0.0",
            "--api-key", "EMPTY",
        ]
        parsed = _parse_serve_args(argv)
        assert parsed["model_path"] == "/models/qwen3-test"
        assert parsed["tensor_parallel_size"] == 2
        assert parsed["gpu_memory_utilization"] == 0.92
        assert parsed["max_model_len"] == 65536
        assert parsed["max_num_seqs"] == 4
        assert parsed["max_num_batched_tokens"] == 8192
        assert parsed["dtype"] == "bfloat16"
        assert parsed["kv_cache_dtype"] == "fp8_e4m3"
        assert parsed["quantization"] == "fp8"
        assert parsed["tool_call_parser"] == "qwen3coder"
        assert parsed["reasoning_parser"] == "qwen3"
        assert parsed["enable_chunked_prefill"] is True
        assert parsed["trust_remote_code"] is True
        assert parsed["enable_auto_tool_choice"] is True
        assert parsed["container_port"] == 8000

    def test_parse_env_splits_genesis_and_system(self):
        from vllm.sndr_core.compat.from_running import _parse_env
        genesis, system = _parse_env([
            "GENESIS_ENABLE_P67=1",
            "SNDR_ENABLE_PN95_TIER_AWARE_CACHE=1",
            "GENESIS_BUFFER_MODE=shared",
            "NCCL_DEBUG=WARN",
            "VLLM_LOGGING_LEVEL=INFO",
            "PATH=/usr/bin",       # stripped
            "HOSTNAME=container",  # stripped
            "RANDOM_VAR=ignored",  # unrecognised, dropped
        ])
        # Keys preserved verbatim with full canonical prefix so the
        # launch renderer emits an identical `export <key>=<val>` line.
        assert genesis == {
            "GENESIS_ENABLE_P67": "1",
            "SNDR_ENABLE_PN95_TIER_AWARE_CACHE": "1",
            "GENESIS_BUFFER_MODE": "shared",
        }
        assert system == {
            "NCCL_DEBUG": "WARN",
            "VLLM_LOGGING_LEVEL": "INFO",
        }

    def test_parse_mounts_handles_ro_and_rw(self):
        from vllm.sndr_core.compat.from_running import _parse_mounts
        rec = {
            "Mounts": [
                {"Source": "/host/models", "Destination": "/models",
                 "Mode": "ro"},
                {"Source": "/host/cache", "Destination": "/cache",
                 "Mode": "rw"},
                {"Source": "/tmp/work", "Destination": "/work", "Mode": ""},
            ],
        }
        result = _parse_mounts(rec)
        assert "/host/models:/models:ro" in result
        assert "/host/cache:/cache" in result
        assert "/tmp/work:/work" in result

    def test_parse_gpus_handles_all_and_explicit_devices(self):
        from vllm.sndr_core.compat.from_running import _parse_gpus
        # --gpus all
        rec_all = {
            "HostConfig": {
                "DeviceRequests": [
                    {"Driver": "nvidia", "Count": -1,
                     "Capabilities": [["gpu"]]},
                ],
            },
        }
        assert _parse_gpus(rec_all) == "all"
        # --gpus 'device=0,1'
        rec_dev = {
            "HostConfig": {
                "DeviceRequests": [
                    {"DeviceIDs": ["0", "1"],
                     "Capabilities": [["gpu"]]},
                ],
            },
        }
        assert _parse_gpus(rec_dev) == "device=0,1"
        # No DeviceRequests, env fallback
        rec_env = {
            "HostConfig": {},
            "Config": {"Env": ["NVIDIA_VISIBLE_DEVICES=0"]},
        }
        assert _parse_gpus(rec_env) == "device=0"


class TestCaptorEndToEnd:
    """End-to-end happy path against a synthetic docker inspect record."""

    def test_capture_synthetic_record_round_trips(self, monkeypatch):
        from vllm.sndr_core.compat import from_running
        from vllm.sndr_core.model_configs import dump_yaml

        synthetic = {
            "Config": {
                "Image": "vllm/vllm-openai:nightly",
                "Entrypoint": ["python3", "-m",
                               "vllm.entrypoints.openai.api_server"],
                "Cmd": [
                    "--model", "/models/qwen3.6-35b",
                    "--tensor-parallel-size", "2",
                    "--gpu-memory-utilization", "0.92",
                    "--max-model-len", "32768",
                    "--max-num-seqs", "2",
                    "--max-num-batched-tokens", "4096",
                    "--dtype", "bfloat16",
                    "--trust-remote-code",
                    "--port", "8000",
                ],
                "Env": [
                    "GENESIS_ENABLE_P67=1",
                    "GENESIS_ENABLE_PN95_TIER_AWARE_CACHE=1",
                    "NCCL_DEBUG=WARN",
                    "VLLM_LOGGING_LEVEL=INFO",
                    "PATH=/usr/bin",
                ],
            },
            "HostConfig": {
                "ShmSize": 8 * 1024 * 1024 * 1024,
                "DeviceRequests": [
                    {"Driver": "nvidia", "Count": -1,
                     "Capabilities": [["gpu"]]},
                ],
            },
            "Mounts": [
                {"Source": "/host/models", "Destination": "/models",
                 "Mode": "ro"},
            ],
            "NetworkSettings": {
                "Ports": {"8000/tcp": [{"HostPort": "8101"}]},
            },
        }
        monkeypatch.setattr(
            from_running, "_run_inspect", lambda _container: synthetic,
        )
        cfg = from_running.capture_from_running(
            "vllm-test-container", key="captured-test",
        )
        assert cfg.key == "captured-test"
        assert cfg.model_path == "/models/qwen3.6-35b"
        assert cfg.max_model_len == 32768
        assert cfg.max_num_seqs == 2
        assert cfg.dtype == "bfloat16"
        assert cfg.hardware.n_gpus == 2
        assert cfg.genesis_env["GENESIS_ENABLE_P67"] == "1"
        assert cfg.genesis_env["GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"] == "1"
        assert cfg.system_env["NCCL_DEBUG"] == "WARN"
        assert cfg.docker is not None
        assert cfg.docker.image == "vllm/vllm-openai:nightly"
        assert cfg.docker.container_name == "vllm-test-container"
        assert cfg.docker.host_port == 8101
        assert cfg.docker.container_port == 8000
        assert cfg.docker.shm_size == "8g"
        assert cfg.docker.gpus == "all"
        assert "/host/models:/models:ro" in cfg.docker.mounts

        # YAML round-trip must succeed (the captor only emits objects
        # that dump_yaml() can serialise without errors).
        yaml_text = dump_yaml(cfg)
        assert "model_path: /models/qwen3.6-35b" in yaml_text
        assert "GENESIS_ENABLE_PN95_TIER_AWARE_CACHE" in yaml_text

    def test_capture_rejects_non_vllm_container(self, monkeypatch):
        from vllm.sndr_core.compat import from_running

        synthetic = {
            "Config": {
                "Image": "nginx:latest",
                "Entrypoint": ["/docker-entrypoint.sh"],
                "Cmd": ["nginx", "-g", "daemon off;"],
                "Env": ["PATH=/usr/bin"],
            },
            "HostConfig": {},
            "Mounts": [],
            "NetworkSettings": {"Ports": {}},
        }
        monkeypatch.setattr(
            from_running, "_run_inspect", lambda _container: synthetic,
        )
        with pytest.raises(from_running.CaptureError, match="vLLM"):
            from_running.capture_from_running("nginx-test", key="bad")
