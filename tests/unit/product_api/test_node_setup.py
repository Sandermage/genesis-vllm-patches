# SPDX-License-Identifier: Apache-2.0
"""Unit tests for one-button node setup (bundle + gated SSH apply)."""
from __future__ import annotations

import io
import tarfile

from vllm.sndr_core.product_api import node_setup


def test_node_bundle_ships_code_AND_corpus_consistently():
    data = node_setup.node_bundle()
    assert data[:2] == b"\x1f\x8b"  # gzip magic
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        names = tar.getnames()
    # The canonical sndr/ package: daemon code (sndr/product_api/legacy/*.py) AND
    # the corpus (sndr/model_configs/**/*.yaml) — fresh code vs a node's stale
    # corpus is what 500s the catalog. Arcnames are repo-root-relative (sndr/...)
    # so the node script unpacks it next to vllm/.
    assert "sndr/product_api/legacy/http_app.py" in names
    assert "sndr/product_api/legacy/node_setup.py" in names
    assert "sndr/version.py" in names  # the import that crashed the old daemon
    assert any(n.startswith("sndr/model_configs/") and n.endswith(".yaml") for n in names)
    assert all(n.endswith((".py", ".yaml", ".yml")) for n in names)
    assert not any("__pycache__" in n or "web_static" in n for n in names)  # excluded


def test_setup_node_script_is_self_contained():
    s = node_setup.setup_node_script(port=8765, engine_port=8102, admin_password="secret")
    assert "tar -xzf sndr-daemon-bundle.tar.gz" in s        # deploys code + corpus
    assert "find" in s and "*.pyc" in s                      # clears stale bytecode
    assert "docker run -d --name" in s and "--network host" in s
    assert "SNDR_ADMIN_PASSWORD='secret'" in s               # password embedded + quoted
    assert "ENGINE_PORT=8102" in s                           # engine port wired
    assert "SNDR_OPENAI_BASE_URL=http://127.0.0.1:$ENGINE_PORT/v1" in s
    # v12: mounts the canonical sndr/ package next to vllm/ and imports it
    # directly (no vllm namespace), with apply wired from SNDR_ENABLE_APPLY.
    assert '-v "$SNDR_SRC":"$SNDR_DST":ro' in s
    assert "from sndr.product_api.legacy.http_app import run_server" in s
    assert "enable_apply=bool(os.environ.get('SNDR_ENABLE_APPLY'))" in s
    assert "vllm.sndr_core.product_api.http_app import" not in s  # not the shim path


def test_setup_node_mounts_docker_sock_without_auto_exec():
    """The sidecar mounts the docker socket (so it can report + manage the host's
    engine containers), but in-container exec stays OFF by default — the operator
    opts into SNDR_ENABLE_EXEC deliberately."""
    s = node_setup.setup_node_script(admin_password="secret")
    assert "-v /var/run/docker.sock:/var/run/docker.sock" in s
    assert "-e SNDR_ENABLE_EXEC" not in s  # never auto-enabled as an env var


def test_setup_node_password_is_shell_escaped():
    s = node_setup.setup_node_script(admin_password="a'b")
    assert "SNDR_ADMIN_PASSWORD='a'\\''b'" in s              # no shell injection


def test_setup_node_is_double_gated_and_ships_bundle():
    calls = {}

    def run_apply(ssh_target, **kw):
        calls.update(kw)
        return {"ok": True, "steps": [{"cmd": "upload setup-node.sh, sndr-product-api.tar.gz", "rc": 0, "output": ""}]}

    base = dict(ssh_target={"host": "x"}, run_apply=run_apply, admin_password="pw1234")
    # Gate 1: apply off.
    assert node_setup.setup_node(**base, apply_enabled=False, confirm=True)["applied"] is False
    # Gate 2: no confirm.
    assert node_setup.setup_node(**base, apply_enabled=True, confirm=False)["applied"] is False
    # Weak password rejected.
    bad = node_setup.setup_node(ssh_target={}, run_apply=run_apply, admin_password="x", apply_enabled=True, confirm=True)
    assert bad["applied"] is False and "password" in bad["error"]
    # Both gates + good password -> executes, shipping the code bundle.
    ok = node_setup.setup_node(**base, apply_enabled=True, confirm=True)
    assert ok["applied"] is True and ok["ok"] is True
    assert calls["artifact_name"] == "setup-node.sh"
    extra = dict(calls["extra_files"])
    assert "sndr-daemon-bundle.tar.gz" in extra and extra["sndr-daemon-bundle.tar.gz"][:2] == b"\x1f\x8b"
