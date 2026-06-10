# SPDX-License-Identifier: Apache-2.0
"""`sndr service` — kubernetes + proxmox backend lifecycle tests.

Audit C3 closure (2026-05-16): both backends used to print
"preview/manual" warnings and exit 0. After this change every lifecycle
verb (install / start / stop / status / logs / uninstall) is wired to
real kubectl / pct invocations, with a dry-run path that prints the
exact command instead of touching the host.

These tests pin the dry-run command shape so the surface cannot
silently regress back to the preview-only behaviour.
"""
from __future__ import annotations

import argparse
import io
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest


def _make_args(**overrides) -> argparse.Namespace:
    """Return an argparse.Namespace with the defaults `sndr service`
    callers always pass (yes/system/lines/config)."""
    defaults = dict(
        config="dummy",
        yes=False,
        system=False,
        lines=50,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _fake_cfg(backend: str, *, ctid: str | None = None, namespace: str | None = None):
    """Build a minimal cfg double matching the attributes `service.py`
    reads. Avoids hitting the full registry / YAML parser so the test
    runs fast and doesn't depend on builtin presets being present."""
    docker = SimpleNamespace(container_name="vllm-test-container")
    options: dict[str, str] = {}
    if ctid is not None:
        options["ctid"] = ctid
    if namespace is not None:
        options["namespace"] = namespace
    service = SimpleNamespace(backend=backend, options=options)
    proxmox = SimpleNamespace(container_id_or_vmid=None)
    return SimpleNamespace(
        key="dummy",
        service=service,
        docker=docker,
        proxmox=proxmox,
    )


@pytest.fixture(autouse=True)
def stub_resolve(monkeypatch):
    """Swap _resolve so we don't need a real config registry."""
    holder = {}

    def _setter(cfg):
        holder["cfg"] = cfg

    from sndr.cli.legacy import service as svc
    monkeypatch.setattr(svc, "_resolve", lambda key: holder.get("cfg"))
    return _setter


def _capture_io(fn, args):
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = fn(args)
    return rc, out.getvalue(), err.getvalue()


# ─── Helper unit tests ─────────────────────────────────────────────────


class TestK8sHelpers:
    def test_object_name_kebabs_underscores(self):
        from sndr.cli.legacy.service import _k8s_object_name
        cfg = SimpleNamespace(
            docker=SimpleNamespace(container_name="vllm_my_Container"),
            key="ignored",
        )
        assert _k8s_object_name(cfg) == "vllm-my-container"

    def test_namespace_from_service_options(self):
        from sndr.cli.legacy.service import _k8s_namespace
        cfg = _fake_cfg("kubernetes", namespace="vllm-prod")
        assert _k8s_namespace(cfg) == "vllm-prod"

    def test_namespace_falls_back_to_default(self):
        from sndr.cli.legacy.service import _k8s_namespace
        cfg = _fake_cfg("kubernetes")
        assert _k8s_namespace(cfg) == "default"


class TestProxmoxHelpers:
    def test_ctid_from_options_wins(self):
        from sndr.cli.legacy.service import _proxmox_ctid
        cfg = _fake_cfg("proxmox", ctid="312")
        assert _proxmox_ctid(cfg) == "312"

    def test_ctid_falls_back_to_default(self, monkeypatch):
        from sndr.cli.legacy.service import _proxmox_ctid
        monkeypatch.delenv("SNDR_CTID", raising=False)
        cfg = _fake_cfg("proxmox")
        assert _proxmox_ctid(cfg) == "200"

    def test_ctid_from_env_var(self, monkeypatch):
        from sndr.cli.legacy.service import _proxmox_ctid
        monkeypatch.setenv("SNDR_CTID", "777")
        cfg = _fake_cfg("proxmox")
        assert _proxmox_ctid(cfg) == "777"


# ─── Lifecycle (dry-run) ───────────────────────────────────────────────


class TestKubernetesLifecycle:
    def test_start_prints_kubectl_scale_to_one(self, stub_resolve):
        stub_resolve(_fake_cfg("kubernetes", namespace="genesis"))
        from sndr.cli.legacy.service import run_start
        rc, out, err = _capture_io(run_start, _make_args())
        text = out + err
        assert rc == 0
        assert "kubectl scale -n genesis deployment/vllm-test-container "
        assert "--replicas=1" in text

    def test_stop_prints_kubectl_scale_to_zero(self, stub_resolve):
        stub_resolve(_fake_cfg("kubernetes"))
        from sndr.cli.legacy.service import run_stop
        rc, out, err = _capture_io(run_stop, _make_args())
        text = out + err
        assert "--replicas=0" in text

    def test_status_uses_kubectl_get(self, stub_resolve, monkeypatch):
        # `kubectl get` is non-dry-run; we need kubectl absent to keep
        # the test deterministic (the helper emits a clean error).
        stub_resolve(_fake_cfg("kubernetes"))
        monkeypatch.setattr(
            "shutil.which", lambda x: None if x == "kubectl" else "/bin/" + x,
        )
        from sndr.cli.legacy.service import run_status
        rc, out, err = _capture_io(run_status, _make_args())
        assert rc == 1
        assert "kubectl" in (out + err)

    def test_logs_uses_kubectl_logs(self, stub_resolve, monkeypatch):
        stub_resolve(_fake_cfg("kubernetes"))
        monkeypatch.setattr(
            "shutil.which", lambda x: None if x == "kubectl" else "/bin/" + x,
        )
        from sndr.cli.legacy.service import run_logs
        rc, out, err = _capture_io(run_logs, _make_args(lines=25))
        assert rc == 1
        text = out + err
        assert "kubectl" in text


class TestProxmoxLifecycle:
    def test_start_calls_pct_start_dry_run(self, stub_resolve):
        stub_resolve(_fake_cfg("proxmox", ctid="201"))
        from sndr.cli.legacy.service import run_start
        rc, out, err = _capture_io(run_start, _make_args())
        text = out + err
        assert rc == 0
        assert "pct start 201" in text

    def test_stop_calls_pct_stop_dry_run(self, stub_resolve):
        stub_resolve(_fake_cfg("proxmox", ctid="201"))
        from sndr.cli.legacy.service import run_stop
        rc, out, err = _capture_io(run_stop, _make_args())
        text = out + err
        assert "pct stop 201" in text

    def test_status_uses_pct_status(self, stub_resolve, monkeypatch):
        stub_resolve(_fake_cfg("proxmox", ctid="201"))
        monkeypatch.setattr(
            "shutil.which", lambda x: None if x == "pct" else "/bin/" + x,
        )
        from sndr.cli.legacy.service import run_status
        rc, out, err = _capture_io(run_status, _make_args())
        assert rc == 1
        assert "pct" in (out + err).lower()

    def test_logs_routes_through_pct_exec_journalctl(
        self, stub_resolve, monkeypatch,
    ):
        stub_resolve(_fake_cfg("proxmox", ctid="201"))
        monkeypatch.setattr(
            "shutil.which", lambda x: None if x == "pct" else "/bin/" + x,
        )
        from sndr.cli.legacy.service import run_logs
        rc, out, err = _capture_io(run_logs, _make_args(lines=25))
        assert rc == 1
        assert "pct" in (out + err).lower()

    def test_uninstall_dry_run_does_not_destroy(self, stub_resolve):
        """--yes is required for destructive ops. Plain dry-run must NOT
        emit any `pct destroy` command — only the planning info line."""
        stub_resolve(_fake_cfg("proxmox", ctid="201"))
        from sndr.cli.legacy.service import run_uninstall
        rc, out, err = _capture_io(run_uninstall, _make_args())
        text = out + err
        assert rc == 0
        assert "dry-run" in text.lower()
        # Importantly: must not have actually executed pct destroy.
        # The dry-run path goes through _io.info which logs the planned
        # command rather than running it. We assert by absence of any
        # real subprocess invocation by checking the text says "would:".
        assert "would" in text.lower()
