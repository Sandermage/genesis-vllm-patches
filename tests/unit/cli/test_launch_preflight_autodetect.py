# SPDX-License-Identifier: Apache-2.0
"""A1-A9 — launcher-side autodetect PREFLIGHT gate.

The gate runs BEFORE `docker run` / `vllm serve` and surfaces operator-
facing errors/warnings for the common "boots and then dies cryptically"
failure classes:

  A1 GPU count mismatch (nvidia-smi vs config n_gpus)        → warn
  A2 vLLM pin mismatch (image pin vs vllm_pin_required)      → warn
  A3 model path missing                                      → error
  A4 drafter model path missing (spec_decode)               → error (CRITICAL)
  A5 HF cache mount missing / undefaulted                   → warn
  A6 max_model_len > model max_position_embeddings          → warn
  A7 served_model_name unset → defaulted from model id      → (mutation)
  A8 target port already in use                             → error
  A9 SNDR_SRC/GENESIS_REPO unset → resolved fallback        → (resolution)

All host-touching calls are injected through a `HostProbe` so these
tests run with no GPU / no docker / no real filesystem dependency.
"""
from __future__ import annotations

import pytest

from sndr.cli.legacy import preflight as P
from sndr.model_configs.schema import ModelConfig, HardwareSpec, SpecDecodeConfig
from sndr.model_configs.types.docker import DockerConfig


# ─── Fixtures ─────────────────────────────────────────────────────────────


def _cfg(**kw) -> ModelConfig:
    base = dict(
        key="test-cfg",
        title="Test",
        description="test",
        schema_version=1,
        maintainer="tests",
        model_path="/models/Qwen3.6-35B-FP8",
        hardware=HardwareSpec(
            gpu_match_keys=["rtx a5000"], n_gpus=2,
            min_vram_per_gpu_mib=24000,
        ),
        vllm_pin_required="0.20.2rc1.dev338+gbf0d2dc6d",
        served_model_name="qwen35",
        max_model_len=32768,
        docker=DockerConfig(
            image="vllm/vllm-openai:nightly",
            container_name="vllm-test",
            port=8101,
            mounts=[
                "/srv/models:/models:ro",
                "/home/op/.cache/huggingface:/root/.cache/huggingface:ro",
            ],
        ),
    )
    base.update(kw)
    return ModelConfig(**base)


class FakeHost(P.HostProbe):
    """In-memory HostProbe for tests — no nvidia-smi / docker / FS."""

    def __init__(
        self,
        gpu_count=2,
        existing_paths=None,
        ports_in_use=(),
        image_pin=None,
        config_json=None,
        repo_root="/repo/root",
    ):
        self._gpu_count = gpu_count
        self._existing = set(existing_paths or [])
        self._ports = set(ports_in_use)
        self._image_pin = image_pin
        self._config_json = config_json or {}
        self._repo_root = repo_root

    def gpu_count(self):
        return self._gpu_count

    def path_exists(self, path):
        return path in self._existing

    def port_in_use(self, port):
        return port in self._ports

    def image_pin(self, image_ref):
        return self._image_pin

    def read_model_config_json(self, model_dir):
        return self._config_json.get(model_dir)

    def git_toplevel(self):
        return self._repo_root


# ─── A1: GPU count ─────────────────────────────────────────────────────────


class TestA1GpuCount:
    def test_match_no_warning(self):
        host = FakeHost(gpu_count=2, existing_paths=["/srv/models/Qwen3.6-35B-FP8"])
        r = P.run_autodetect_preflight(_cfg(), {}, host=host)
        assert not any(i.code == "A1" for i in r.warnings)

    def test_mismatch_warns(self):
        host = FakeHost(gpu_count=1, existing_paths=["/srv/models/Qwen3.6-35B-FP8"])
        r = P.run_autodetect_preflight(_cfg(), {}, host=host)
        a1 = [i for i in r.warnings if i.code == "A1"]
        assert a1, "GPU count mismatch must warn"
        assert "1" in a1[0].message and "2" in a1[0].message

    def test_unknown_gpu_count_no_warn(self):
        # nvidia-smi missing → gpu_count None → cannot assert, skip silently.
        host = FakeHost(gpu_count=None, existing_paths=["/srv/models/Qwen3.6-35B-FP8"])
        r = P.run_autodetect_preflight(_cfg(), {}, host=host)
        assert not any(i.code == "A1" for i in r.warnings)


# ─── A2: vLLM pin mismatch ─────────────────────────────────────────────────


class TestA2PinMismatch:
    def test_match_no_warn(self):
        host = FakeHost(
            existing_paths=["/srv/models/Qwen3.6-35B-FP8"],
            image_pin="0.20.2rc1.dev338+gbf0d2dc6d",
        )
        r = P.run_autodetect_preflight(_cfg(), {}, host=host)
        assert not any(i.code == "A2" for i in r.warnings)

    def test_mismatch_warns(self):
        host = FakeHost(
            existing_paths=["/srv/models/Qwen3.6-35B-FP8"],
            image_pin="0.20.2rc1.dev371+deadbeef",
        )
        r = P.run_autodetect_preflight(_cfg(), {}, host=host)
        assert any(i.code == "A2" for i in r.warnings)

    def test_no_pin_required_skips(self):
        host = FakeHost(
            existing_paths=["/srv/models/Qwen3.6-35B-FP8"],
            image_pin="anything",
        )
        r = P.run_autodetect_preflight(_cfg(vllm_pin_required=None), {}, host=host)
        assert not any(i.code == "A2" for i in r.warnings)


# ─── A3: model path existence ──────────────────────────────────────────────


class TestA3ModelPath:
    def test_present_ok(self):
        host = FakeHost(existing_paths=["/srv/models/Qwen3.6-35B-FP8"])
        r = P.run_autodetect_preflight(_cfg(), {}, host=host)
        assert not any(i.code == "A3" for i in r.errors)

    def test_missing_errors(self):
        host = FakeHost(existing_paths=[])  # model dir absent
        r = P.run_autodetect_preflight(_cfg(), {}, host=host)
        a3 = [i for i in r.errors if i.code == "A3"]
        assert a3, "missing model path must be an ERROR"
        assert "/srv/models/Qwen3.6-35B-FP8" in a3[0].message


# ─── A4: drafter model existence (CRITICAL) ────────────────────────────────


class TestA4DrafterPath:
    def _spec_cfg(self):
        return _cfg(
            spec_decode=SpecDecodeConfig(
                method="dflash", num_speculative_tokens=4,
                model="/models/Qwen3.6-Drafter",
            ),
        )

    def test_drafter_present_ok(self):
        host = FakeHost(existing_paths=[
            "/srv/models/Qwen3.6-35B-FP8", "/srv/models/Qwen3.6-Drafter",
        ])
        r = P.run_autodetect_preflight(self._spec_cfg(), {}, host=host)
        assert not any(i.code == "A4" for i in r.errors)

    def test_drafter_missing_errors(self):
        host = FakeHost(existing_paths=["/srv/models/Qwen3.6-35B-FP8"])
        r = P.run_autodetect_preflight(self._spec_cfg(), {}, host=host)
        a4 = [i for i in r.errors if i.code == "A4"]
        assert a4, "missing drafter must be an ERROR (MTP fails at engine init)"
        assert "Drafter" in a4[0].message

    def test_no_spec_decode_skips(self):
        host = FakeHost(existing_paths=["/srv/models/Qwen3.6-35B-FP8"])
        r = P.run_autodetect_preflight(_cfg(), {}, host=host)
        assert not any(i.code == "A4" for i in r.errors)

    def test_mtp_without_separate_drafter_skips(self):
        # method=mtp with model=None uses the target's own head → no A4.
        cfg = _cfg(spec_decode=SpecDecodeConfig(
            method="mtp", num_speculative_tokens=4,
        ))
        host = FakeHost(existing_paths=["/srv/models/Qwen3.6-35B-FP8"])
        r = P.run_autodetect_preflight(cfg, {}, host=host)
        assert not any(i.code == "A4" for i in r.errors)


# ─── A5: HF cache mount ────────────────────────────────────────────────────


class TestA5HfCache:
    def test_hf_mount_present_no_warn(self):
        host = FakeHost(existing_paths=["/srv/models/Qwen3.6-35B-FP8"])
        r = P.run_autodetect_preflight(_cfg(), {}, host=host)
        assert not any(i.code == "A5" for i in r.warnings)

    def test_hf_mount_absent_warns(self):
        # No huggingface mount in docker block → warn (downloads can fail).
        cfg = _cfg(docker=DockerConfig(
            image="img", container_name="c", port=8101,
            mounts=["/srv/models:/models:ro"],  # no HF cache mount
        ))
        host = FakeHost(existing_paths=["/srv/models/Qwen3.6-35B-FP8"])
        r = P.run_autodetect_preflight(cfg, {}, host=host)
        assert any(i.code == "A5" for i in r.warnings)


# ─── A6: max-model-len sanity ──────────────────────────────────────────────


class TestA6MaxModelLen:
    def test_within_limit_no_warn(self):
        host = FakeHost(
            existing_paths=["/srv/models/Qwen3.6-35B-FP8"],
            config_json={"/srv/models/Qwen3.6-35B-FP8": {
                "max_position_embeddings": 131072}},
        )
        r = P.run_autodetect_preflight(_cfg(max_model_len=32768), {}, host=host)
        assert not any(i.code == "A6" for i in r.warnings)

    def test_exceeds_limit_warns(self):
        host = FakeHost(
            existing_paths=["/srv/models/Qwen3.6-35B-FP8"],
            config_json={"/srv/models/Qwen3.6-35B-FP8": {
                "max_position_embeddings": 8192}},
        )
        r = P.run_autodetect_preflight(_cfg(max_model_len=32768), {}, host=host)
        a6 = [i for i in r.warnings if i.code == "A6"]
        assert a6, "max_model_len above model limit must warn"
        assert "8192" in a6[0].message

    def test_no_config_json_skips(self):
        host = FakeHost(existing_paths=["/srv/models/Qwen3.6-35B-FP8"])
        r = P.run_autodetect_preflight(_cfg(max_model_len=999999), {}, host=host)
        assert not any(i.code == "A6" for i in r.warnings)


# ─── A6 (llama.cpp lane): GGUF-aware checks ────────────────────────────────
#
# The vLLM A6 reads <model_dir>/config.json (a directory). The GGUF lane's
# model_path is a single .gguf FILE (no config.json), so the orchestrator
# routes engine=="llama-cpp" to _check_a6_gguf: path (.gguf file contract),
# image (pinned b9246 build), and ctx (single-card -c ceiling).

_GGUF = ("/models/qwen3.6-27b-gguf/unsloth-mtp-q4km/"
         "Qwen3.6-27B-Q4_K_M.gguf")
_LLAMACPP_IMG = "ghcr.io/ggml-org/llama.cpp:server-cuda-b9246"


def _llamacpp_cfg(**kw):
    base = dict(
        key="lc", title="t", description="d", schema_version=1,
        maintainer="tests", engine="llama-cpp",
        model_path=_GGUF, served_model_name="qwen3.6-27b",
        max_model_len=131072,
        hardware=HardwareSpec(gpu_match_keys=["rtx 3090"], n_gpus=1,
                              min_vram_per_gpu_mib=22000),
        docker=DockerConfig(image=_LLAMACPP_IMG,
                            container_name="llamacpp-qwen-1x", port=8020,
                            mounts=["/srv/models:/models:ro"]),
    )
    base.update(kw)
    return ModelConfig(**base)


class TestA6GgufLane:
    def test_valid_lane_no_a6_finding(self):
        # .gguf file + b9246 image + 131072 ctx (== ceiling) → no A6 issue.
        host = FakeHost(existing_paths=[_GGUF])
        r = P.run_autodetect_preflight(_llamacpp_cfg(), {}, host=host)
        assert not any(i.code == "A6" for i in r.errors)
        assert not any(i.code == "A6" for i in r.warnings)

    def test_dir_model_path_errors(self):
        host = FakeHost(existing_paths=["/models/Qwen3.6-27B-int4-AutoRound"])
        cfg = _llamacpp_cfg(model_path="/models/Qwen3.6-27B-int4-AutoRound")
        r = P.run_autodetect_preflight(cfg, {}, host=host)
        a6 = [i for i in r.errors if i.code == "A6"]
        assert a6, "a directory model_path must be an A6 ERROR on llama.cpp"
        assert ".gguf" in a6[0].message

    def test_vllm_image_errors(self):
        host = FakeHost(existing_paths=[_GGUF])
        cfg = _llamacpp_cfg(docker=DockerConfig(
            image="vllm/vllm-openai:nightly",
            container_name="llamacpp-qwen-1x", port=8020))
        r = P.run_autodetect_preflight(cfg, {}, host=host)
        a6 = [i for i in r.errors if i.code == "A6"]
        assert a6, "a vLLM image on the llama.cpp lane must be an A6 ERROR"
        assert "vLLM image" in a6[0].message

    def test_unpinned_llamacpp_image_warns(self):
        host = FakeHost(existing_paths=[_GGUF])
        cfg = _llamacpp_cfg(docker=DockerConfig(
            image="ghcr.io/ggml-org/llama.cpp:server-cuda",  # rolling, unpinned
            container_name="llamacpp-qwen-1x", port=8020))
        r = P.run_autodetect_preflight(cfg, {}, host=host)
        a6 = [i for i in r.warnings if i.code == "A6"]
        assert a6, "an unpinned :server-cuda tag must warn"
        assert "server-cuda-b9246" in a6[0].message

    def test_ctx_over_ceiling_warns(self):
        host = FakeHost(existing_paths=[_GGUF])
        cfg = _llamacpp_cfg(max_model_len=200000)  # club-3090: walls at load
        r = P.run_autodetect_preflight(cfg, {}, host=host)
        a6 = [i for i in r.warnings if i.code == "A6"]
        assert a6, "-c above the single-card ceiling must warn"
        assert "200000" in a6[0].message

    def test_vllm_lane_untouched_by_gguf_branch(self):
        # The vLLM lane still runs the config.json A6 check, NOT the GGUF one.
        host = FakeHost(
            existing_paths=["/srv/models/Qwen3.6-35B-FP8"],
            config_json={"/srv/models/Qwen3.6-35B-FP8": {
                "max_position_embeddings": 8192}},
        )
        r = P.run_autodetect_preflight(_cfg(max_model_len=32768), {}, host=host)
        a6 = [i for i in r.warnings if i.code == "A6"]
        assert a6 and "8192" in a6[0].message  # config.json-based, not GGUF


# ─── A7: served-model-name default ─────────────────────────────────────────


class TestA7ServedModelName:
    def test_unset_defaulted_from_model_id(self):
        cfg = _cfg(served_model_name=None)
        host = FakeHost(existing_paths=["/srv/models/Qwen3.6-35B-FP8"])
        P.run_autodetect_preflight(cfg, {}, host=host)
        # Mutated in place to a basename-derived default.
        assert cfg.served_model_name == "Qwen3.6-35B-FP8"

    def test_set_preserved(self):
        cfg = _cfg(served_model_name="my-name")
        host = FakeHost(existing_paths=["/srv/models/Qwen3.6-35B-FP8"])
        P.run_autodetect_preflight(cfg, {}, host=host)
        assert cfg.served_model_name == "my-name"


# ─── A8: port conflict ─────────────────────────────────────────────────────


class TestA8PortConflict:
    def test_port_free_ok(self):
        host = FakeHost(existing_paths=["/srv/models/Qwen3.6-35B-FP8"], ports_in_use=())
        r = P.run_autodetect_preflight(_cfg(), {}, host=host)
        assert not any(i.code == "A8" for i in r.errors)

    def test_port_in_use_errors(self):
        host = FakeHost(
            existing_paths=["/srv/models/Qwen3.6-35B-FP8"], ports_in_use=(8101,))
        r = P.run_autodetect_preflight(_cfg(), {}, host=host)
        a8 = [i for i in r.errors if i.code == "A8"]
        assert a8, "in-use port must be an ERROR"
        assert "8101" in a8[0].message


# ─── A9: SNDR_SRC / GENESIS_REPO resolution ────────────────────────────────


class TestA9RepoResolution:
    def test_env_set_used(self, monkeypatch):
        monkeypatch.setenv("SNDR_SRC", "/explicit/src")
        host = FakeHost(existing_paths=["/srv/models/Qwen3.6-35B-FP8"])
        r = P.run_autodetect_preflight(_cfg(), {}, host=host)
        assert r.resolved_repo == "/explicit/src"

    def test_falls_back_to_git_toplevel(self, monkeypatch):
        monkeypatch.delenv("SNDR_SRC", raising=False)
        monkeypatch.delenv("GENESIS_REPO", raising=False)
        host = FakeHost(
            existing_paths=["/srv/models/Qwen3.6-35B-FP8"],
            repo_root="/git/top/level",
        )
        r = P.run_autodetect_preflight(_cfg(), {}, host=host)
        assert r.resolved_repo == "/git/top/level"


# ─── Aggregate result semantics ────────────────────────────────────────────


class TestResultAggregate:
    def test_ok_when_no_errors(self):
        host = FakeHost(existing_paths=["/srv/models/Qwen3.6-35B-FP8"])
        r = P.run_autodetect_preflight(_cfg(), {}, host=host)
        assert r.ok is True

    def test_not_ok_when_any_error(self):
        host = FakeHost(existing_paths=[])  # model missing → A3 error
        r = P.run_autodetect_preflight(_cfg(), {}, host=host)
        assert r.ok is False


# ─── Wiring into `sndr launch` ─────────────────────────────────────────────


def _launch_opts(**kw):
    import argparse
    from sndr.cli.legacy import launch as L
    parser = argparse.ArgumentParser(prog="sndr")
    sub = parser.add_subparsers()
    L.add_argparser(sub)
    ns = parser.parse_args(["launch", "test-key"])
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


class TestLaunchWiring:
    def test_gate_aborts_live_launch_on_error(self, monkeypatch):
        from sndr.cli.legacy import launch as L
        from sndr.cli.legacy import preflight as PF

        cfg = _cfg()
        monkeypatch.setattr(L, "_resolve_config", lambda *a, **k: (cfg, "test-key"))
        monkeypatch.setattr(L, "_load_host_paths", lambda: {})

        # Force an error result from the gate.
        def _bad(cfg, host_paths=None, *, host=None):
            res = PF.PreflightResult()
            res.error("A3", "model missing")
            return res

        monkeypatch.setattr(PF, "run_autodetect_preflight", _bad)

        # Guard: exec must never be reached.
        def _no_exec(*a, **k):
            raise AssertionError("execvp reached despite preflight error")

        monkeypatch.setattr("os.execvp", _no_exec)

        rc = L.run_launch(_launch_opts())
        assert rc == 2

    def test_skip_autodetect_bypasses_gate(self, monkeypatch):
        from sndr.cli.legacy import launch as L
        from sndr.cli.legacy import preflight as PF

        cfg = _cfg()
        monkeypatch.setattr(L, "_resolve_config", lambda *a, **k: (cfg, "test-key"))
        monkeypatch.setattr(L, "_load_host_paths", lambda: {})

        called = {"n": 0}

        def _track(*a, **k):
            called["n"] += 1
            return PF.PreflightResult()

        monkeypatch.setattr(PF, "run_autodetect_preflight", _track)
        # Stop the launch right after the gate would have run — render fails
        # cleanly so we don't need docker/exec.
        monkeypatch.setattr(
            cfg, "to_launch_script",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")),
        )
        with pytest.raises(SystemExit):
            # render failure → _io.fatal → SystemExit; that's fine, we only
            # assert the gate was skipped.
            L.run_launch(_launch_opts(skip_autodetect=True))
        assert called["n"] == 0, "--skip-autodetect must not invoke the gate"
