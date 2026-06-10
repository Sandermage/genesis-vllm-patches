# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr support-bundle` — §6.H9 of the unified plan.

The verb composes the H7 collect logic, the H8 summarize logic, host
facts, and container facts into a single ``.tar.gz``. Tests stub
``docker``, the catalog scanner, and the docker-cp helper so the
bundling logic can be exercised without a live container.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _run(*args: str, env_extra: dict | None = None,
         text_in: str | None = None) -> subprocess.CompletedProcess:
    import os
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "sndr.cli.legacy", "support-bundle", *args],
        capture_output=True, text=True, cwd=REPO_ROOT,
        env=env, check=False, input=text_in,
    )


# ─── Argparse registration ───────────────────────────────────────────


def test_support_bundle_is_top_level_subcommand() -> None:
    r = _run("--help")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "--container" in out
    assert "--output" in out
    assert "--no-traces" in out
    assert "--json" in out


def test_support_bundle_requires_container_flag() -> None:
    r = _run()
    assert r.returncode != 0
    combined = r.stdout + r.stderr
    assert "--container" in combined.lower() or "required" in combined.lower()


def test_support_bundle_no_docker_returns_3() -> None:
    """Without docker on PATH and without --no-traces, the verb must
    refuse cleanly with exit 3 and a clear message."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        env_extra = {"PATH": tmpdir}
        r = _run("--container", "any", env_extra=env_extra)
    assert r.returncode == 3
    combined = r.stdout + r.stderr
    assert "docker" in combined.lower()


# ─── collect_host_facts helper ───────────────────────────────────────


def test_collect_host_facts_returns_expected_keys() -> None:
    import sndr.cli.legacy.trace as tmod
    facts = tmod.collect_host_facts()
    expected = {"uname", "free_disk", "free_memory",
                "nvidia_smi", "docker_version"}
    assert set(facts.keys()) == expected
    for key, fact in facts.items():
        assert "ok" in fact and "output" in fact
        assert isinstance(fact["ok"], bool)


def test_collect_host_facts_handles_missing_binary(monkeypatch) -> None:
    """Even when every command is absent (no $PATH binaries), the
    function returns a complete dict with `ok: False` everywhere —
    never raises."""
    import sndr.cli.legacy.trace as tmod
    monkeypatch.setattr(tmod.shutil, "which", lambda name: None)
    facts = tmod.collect_host_facts()
    assert set(facts.keys()) == {
        "uname", "free_disk", "free_memory",
        "nvidia_smi", "docker_version",
    }
    for f in facts.values():
        assert f["ok"] is False
        assert "$PATH" in f["output"] or "not on" in f["output"]


# ─── collect_container_facts helper ──────────────────────────────────


def test_collect_container_facts_filters_env_to_relevant_prefixes(
    monkeypatch,
) -> None:
    """The env dump must surface GENESIS_ / SNDR_ / VLLM_ / NCCL_ /
    CUDA_ / TORCH_ / PYTORCH_ / TRITON_ — and only those — so the
    bundle stays small and operator-readable."""
    import sndr.cli.legacy.trace as tmod
    monkeypatch.setattr(tmod.shutil, "which", lambda name: "/usr/bin/docker")

    calls: list[list[str]] = []

    class FakeRun:
        def __init__(self, stdout):
            self.stdout = stdout
            self.stderr = ""
            self.returncode = 0

    def fake_run(argv, **kw):
        calls.append(argv)
        if "--format" in argv and "{{.Config.Image}}" in argv[-1]:
            return FakeRun("vllm/vllm-openai:nightly\nsha256:abc123\n")
        if "--format" in argv and "{{range .Config.Env}}" in argv[-1]:
            return FakeRun(
                "GENESIS_ENABLE_PN286=1\n"
                "HOME=/root\n"
                "VLLM_LOGGING_LEVEL=WARNING\n"
                "PATH=/usr/local/bin\n"
                "NCCL_DEBUG=WARN\n"
                "TRITON_CACHE_DIR=/root/.triton/cache\n"
            )
        if argv[1] == "exec":
            return FakeRun("UID PID PPID CMD\nroot 1 0 vllm-serve\n")
        if argv[1] == "logs":
            return FakeRun("INFO: vllm started\n")
        return FakeRun("")

    monkeypatch.setattr(tmod.subprocess, "run", fake_run)
    facts = tmod.collect_container_facts("vllm-rig")

    assert "GENESIS_ENABLE_PN286=1" in facts["env_filtered"]["output"]
    assert "VLLM_LOGGING_LEVEL=WARNING" in facts["env_filtered"]["output"]
    assert "NCCL_DEBUG=WARN" in facts["env_filtered"]["output"]
    assert "TRITON_CACHE_DIR" in facts["env_filtered"]["output"]
    # Non-genesis env vars must be filtered out.
    assert "HOME=" not in facts["env_filtered"]["output"]
    assert "PATH=" not in facts["env_filtered"]["output"]


def test_collect_container_facts_returns_all_keys_even_on_failure(
    monkeypatch,
) -> None:
    """Every key must always be present in the dict, even when the
    underlying subprocess fails — the bundling code iterates the
    keys and writes one file per."""
    import sndr.cli.legacy.trace as tmod
    monkeypatch.setattr(tmod.shutil, "which", lambda name: "/usr/bin/docker")

    class FakeFail:
        returncode = 1
        stdout = ""
        stderr = "Error: container not running\n"

    monkeypatch.setattr(tmod.subprocess, "run", lambda *a, **kw: FakeFail())
    facts = tmod.collect_container_facts("dead-container")
    assert set(facts.keys()) == {
        "image", "env_filtered", "processes", "logs_tail",
    }


# ─── Bundle build + manifest ─────────────────────────────────────────


def _stub_docker(monkeypatch, live_traces: dict[str, tuple[int, str]]):
    """Set up the common stubs so run_support_bundle can run inside
    a hermetic test: docker present, _container_ls_tmp returns the
    given live trace inventory, docker cp writes a small file with
    valid boot content for the boot log."""
    import sndr.cli.legacy.trace as tmod
    monkeypatch.setattr(tmod.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(
        tmod, "_container_ls_tmp", lambda c: live_traces,
    )

    def fake_cp(container, src, dst):
        # Generate plausible content per filename so the summarize
        # step exercises both boot and generic paths.
        name = src.split("/")[-1]
        if name == "genesis_boot.log":
            content = (
                "[Genesis] PN125: applied\n"
                "[Genesis] PN286: applied\n"
                "[Genesis] PN16: skipped\n"
                "[Genesis] PN999: failed (drift)\n"
            )
        else:
            content = f"sample line in {name}\nsecond line\n"
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write(content)
        return True, "ok"

    monkeypatch.setattr(tmod, "_docker_cp", fake_cp)

    # Stub host + container facts to avoid touching the real host.
    monkeypatch.setattr(
        tmod, "collect_host_facts",
        lambda: {
            "uname": {"ok": True, "output": "Linux x"},
            "free_disk": {"ok": True, "output": "/dev/sda 100GB"},
            "free_memory": {"ok": True, "output": "128GB"},
            "nvidia_smi": {"ok": True, "output": "GPU0 A5000"},
            "docker_version": {"ok": True, "output": "Docker 28.0.0"},
        },
    )
    monkeypatch.setattr(
        tmod, "collect_container_facts",
        lambda c: {
            "image": {"ok": True, "output": "vllm-openai:nightly"},
            "env_filtered": {"ok": True,
                             "output": "GENESIS_ENABLE_PN286=1"},
            "processes": {"ok": True, "output": "PID 1 vllm"},
            "logs_tail": {"ok": True, "output": "INFO: started"},
        },
    )


def test_support_bundle_creates_tar_gz_with_expected_layout(
    monkeypatch, tmp_path,
) -> None:
    import sndr.cli.legacy.trace as tmod
    _stub_docker(monkeypatch, {
        "genesis_boot.log": (12345, "2026-05-30 14:00:00 +0000"),
        "genesis_pn248_acceptance_trace.log":
            (2048, "2026-05-30 14:01:00 +0000"),
    })

    out_path = tmp_path / "bundle.tar.gz"
    import argparse
    args = argparse.Namespace(
        container="vllm-rig", output=str(out_path),
        no_traces=False, json=True, func=tmod.run_support_bundle,
    )
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = tmod.run_support_bundle(args)
    assert rc == 0, buf.getvalue()
    assert out_path.exists()

    # Inspect the bundle contents.
    with tarfile.open(out_path, "r:gz") as tar:
        names = tar.getnames()
    # Top-level wrapper directory present.
    assert any(n == "genesis_support_bundle" or
               n.startswith("genesis_support_bundle/") for n in names)

    def _has(suffix):
        return any(n.endswith(suffix) for n in names)

    assert _has("manifest.json")
    assert _has("traces/genesis_boot.log")
    assert _has("traces/genesis_pn248_acceptance_trace.log")
    # Summaries written for both traces.
    assert _has("summaries/boot.json")
    assert _has("summaries/pn248_acceptance.json")
    # Host facts.
    assert _has("host/uname.txt")
    assert _has("host/nvidia_smi.txt")
    # Container facts.
    assert _has("container/image.txt")
    assert _has("container/env_filtered.txt")


def test_support_bundle_json_report_matches_actual_state(
    monkeypatch, tmp_path,
) -> None:
    import sndr.cli.legacy.trace as tmod
    _stub_docker(monkeypatch, {
        "genesis_boot.log": (12345, "2026-05-30 14:00:00 +0000"),
    })

    out_path = tmp_path / "bundle.tar.gz"
    import argparse
    args = argparse.Namespace(
        container="vllm-rig", output=str(out_path),
        no_traces=False, json=True, func=tmod.run_support_bundle,
    )
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = tmod.run_support_bundle(args)
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["container"] == "vllm-rig"
    assert payload["output"] == str(out_path)
    assert payload["size_bytes"] > 0
    assert payload["trace_collection"]["skipped"] is False
    assert payload["trace_collection"]["collected"] == 1
    assert payload["trace_collection"]["summarized"] == 1
    assert payload["trace_collection"]["failed"] == []


def test_support_bundle_no_traces_skips_collection(monkeypatch, tmp_path) -> None:
    import sndr.cli.legacy.trace as tmod
    # docker present (the --no-traces guard short-circuits PATH check
    # only when docker is missing).
    monkeypatch.setattr(tmod.shutil, "which", lambda name: "/usr/bin/docker")

    def _should_not_be_called(*a, **kw):
        raise AssertionError("_container_ls_tmp must NOT be called "
                              "when --no-traces is set")

    monkeypatch.setattr(tmod, "_container_ls_tmp", _should_not_be_called)
    monkeypatch.setattr(tmod, "_docker_cp", _should_not_be_called)
    monkeypatch.setattr(
        tmod, "collect_host_facts",
        lambda: {"uname": {"ok": True, "output": "Linux"}},
    )
    monkeypatch.setattr(
        tmod, "collect_container_facts",
        lambda c: {"image": {"ok": True, "output": "x"}},
    )

    out_path = tmp_path / "bundle.tar.gz"
    import argparse
    args = argparse.Namespace(
        container="vllm-rig", output=str(out_path),
        no_traces=True, json=True, func=tmod.run_support_bundle,
    )
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = tmod.run_support_bundle(args)
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["trace_collection"]["skipped"] is True
    assert payload["trace_collection"]["collected"] == 0


def test_support_bundle_manifest_records_failed_collects(
    monkeypatch, tmp_path,
) -> None:
    """When a docker cp fails for one trace, the manifest must list it
    under `trace_collection.failed` AND the exit code must be 1 (the
    operator needs to know not everything got through)."""
    import sndr.cli.legacy.trace as tmod
    monkeypatch.setattr(tmod.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(
        tmod, "_container_ls_tmp",
        lambda c: {"genesis_boot.log": (1024, "stamp")},
    )

    def fake_cp_fail(container, src, dst):
        return False, "permission denied"

    monkeypatch.setattr(tmod, "_docker_cp", fake_cp_fail)
    monkeypatch.setattr(
        tmod, "collect_host_facts", lambda: {},
    )
    monkeypatch.setattr(
        tmod, "collect_container_facts", lambda c: {},
    )

    out_path = tmp_path / "bundle.tar.gz"
    import argparse
    args = argparse.Namespace(
        container="vllm-rig", output=str(out_path),
        no_traces=False, json=True, func=tmod.run_support_bundle,
    )
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = tmod.run_support_bundle(args)
    assert rc == 1
    payload = json.loads(buf.getvalue())
    assert len(payload["trace_collection"]["failed"]) == 1
    assert payload["trace_collection"]["failed"][0]["id"] == "boot"
    assert "permission denied" in (
        payload["trace_collection"]["failed"][0]["error"]
    )


# ─── Manifest content lock ───────────────────────────────────────────


def test_manifest_includes_canonical_fields(monkeypatch, tmp_path) -> None:
    """The manifest is the single entry point for downstream tooling.
    Lock the field set so refactors don't silently break that contract."""
    import sndr.cli.legacy.trace as tmod
    _stub_docker(monkeypatch, {
        "genesis_boot.log": (1024, "2026-05-30 14:00:00 +0000"),
    })

    out_path = tmp_path / "bundle.tar.gz"
    import argparse
    args = argparse.Namespace(
        container="vllm-rig", output=str(out_path),
        no_traces=False, json=False, func=tmod.run_support_bundle,
    )
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tmod.run_support_bundle(args)

    with tarfile.open(out_path, "r:gz") as tar:
        manifest_member = next(
            m for m in tar.getmembers() if m.name.endswith("manifest.json")
        )
        f = tar.extractfile(manifest_member)
        assert f is not None
        manifest = json.load(f)

    for key in [
        "container", "created_utc", "trace_collection",
        "host_facts", "container_facts",
    ]:
        assert key in manifest, f"manifest missing {key!r}"
    assert manifest["container"] == "vllm-rig"
    assert manifest["created_utc"].endswith("Z")
    assert isinstance(manifest["host_facts"], list)
    assert isinstance(manifest["container_facts"], list)
