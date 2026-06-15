# SPDX-License-Identifier: Apache-2.0
"""Tests for the enterprise deployment planner (targets, artifacts, deps).

These exercise the read-only deployment planner that powers the GUI Setup
section: it resolves a preset, renders the deployment artifact for a chosen
target (compose / quadlet / kubernetes / systemd / bare-metal / proxmox),
surfaces the live host inventory, and the dependency install plan. All pure
— no subprocess, no host mutation.
"""
from __future__ import annotations

import pytest

from sndr.product_api.legacy import deployment as dep


def _a_preset() -> str:
    keys = dep.list_preset_keys()
    assert keys, "expected at least one resolvable preset"
    # Prefer a docker-backed preset so artifact rendering has a docker block.
    for k in keys:
        if dep.preset_has_docker(k):
            return k
    return keys[0]


def test_list_targets_covers_enterprise_surface():
    ids = {t["id"] for t in dep.list_targets()}
    assert {"compose", "quadlet", "kubernetes", "systemd", "bare_metal", "proxmox"} <= ids
    for t in dep.list_targets():
        assert t["label"] and t["filename"] and t["kind"] in {"yaml", "ini", "bash"}


def test_host_inventory_shape():
    inv = dep.host_inventory()
    assert {"os", "python", "docker", "nvidia", "vllm"} <= set(inv)
    assert isinstance(inv["python"].get("version"), str)


def test_build_compose_artifact_renders_real_yaml():
    preset = _a_preset()
    out = dep.build_deployment(preset, "compose")
    assert out["preset_id"] == preset
    assert out["target"] == "compose"
    art = out["artifact"]
    assert art["kind"] == "yaml" and art["filename"].endswith(".yml")
    assert "services:" in art["content"]
    # mount placeholders were resolved (no literal ${var} left)
    assert "${" not in art["content"].replace("${VLLM_API_KEY", "")
    # fine launch parameters surfaced from the real cfg
    p = out["parameters"]
    assert p["image"] and p["container_name"] and isinstance(p["argv"], list) and p["argv"]
    assert p["tensor_parallel"] >= 1
    assert p["genesis_env_count"] >= 0
    # dependency plan is real (host has no GPU/docker in CI → blockers)
    deps = out["dependencies"]
    assert {"is_ready", "n_blockers", "n_warnings", "items"} <= set(deps)


def test_image_override_pins_engine_at_install_time():
    preset = _a_preset()
    base = dep.build_deployment(preset, "compose")
    # A tag override replaces the image everywhere — parameters AND the artifact.
    tag = dep.build_deployment(preset, "compose", image_override="vllm/vllm-openai:nightly-PINX")
    assert tag["parameters"]["image"] == "vllm/vllm-openai:nightly-PINX"
    assert tag["image_override"] == "vllm/vllm-openai:nightly-PINX"
    assert "nightly-PINX" in tag["artifact"]["content"]
    assert tag["parameters"]["image"] != base["parameters"]["image"] or "PINX" in base["parameters"]["image"]
    # A digest override is treated as a digest (wins, reproducible).
    dig = dep.build_deployment(preset, "compose", image_override="vllm/vllm-openai@sha256:deadbeef")
    assert dig["parameters"]["image"] == "vllm/vllm-openai@sha256:deadbeef"
    # Empty/None override is a no-op (preset image unchanged).
    assert dep.build_deployment(preset, "compose", image_override="  ")["parameters"]["image"] == base["parameters"]["image"]


def test_with_daemon_container_appends_sidecar():
    preset = _a_preset()
    base = dep.build_deployment(preset, "compose")
    wd = dep.build_deployment(preset, "compose", with_daemon=True)
    assert wd["with_daemon"] is True
    assert len(wd["commands"]) > len(base["commands"])
    joined = "\n".join(wd["commands"])
    assert "run-sndr-daemon.sh" in joined          # sidecar script materialised
    assert "127.0.0.1:8765/api/v1/health" in joined  # health probe appended


def test_with_daemon_bare_metal_uses_native_systemd():
    preset = _a_preset()
    wd = dep.build_deployment(preset, "bare_metal", with_daemon=True)
    joined = "\n".join(wd["commands"])
    assert "/etc/systemd/system/sndr-daemon.service" in joined
    assert "sndr.cli gui-api" in joined   # native daemon entrypoint (canonical, no vllm namespace)
    assert "systemctl enable --now sndr-daemon" in joined


def test_with_daemon_proxmox_embeds_inside_guest():
    preset = _a_preset()
    lxc = dep.build_deployment(preset, "proxmox", with_daemon=True)["artifact"]["content"]
    assert "pct push" in lxc and "run-sndr-daemon.sh" in lxc
    assert 'pct exec "$CTID" -- bash /root/run-sndr-daemon.sh' in lxc
    vm = dep.build_deployment(preset, "proxmox_vm", with_daemon=True)["artifact"]["content"]
    assert "write_files:" in vm and "/root/run-sndr-daemon.sh" in vm
    assert "[bash, /root/run-sndr-daemon.sh]" in vm
    # without the flag the daemon block is absent
    assert "run-sndr-daemon.sh" not in dep.build_deployment(preset, "proxmox")["artifact"]["content"]


def test_mount_vars_overridable():
    preset = _a_preset()
    base = dep.build_deployment(preset, "compose")
    names = {m["name"] for m in base["mount_vars"]}
    if "models_dir" in names:
        out = dep.build_deployment(preset, "compose", host_paths={"models_dir": "/data/weights"})
        assert "/data/weights:/models" in out["artifact"]["content"]


@pytest.mark.parametrize("target,kind,needle", [
    ("quadlet", "ini", "[Container]"),
    ("kubernetes", "yaml", "kind: Deployment"),
    ("systemd", "ini", "[Service]"),
    ("bare_metal", "bash", "vllm serve"),
    ("proxmox", "bash", "pct create"),
])
def test_each_target_renders(target, kind, needle):
    preset = _a_preset()
    out = dep.build_deployment(preset, target)
    assert out["artifact"]["kind"] == kind
    assert needle in out["artifact"]["content"]
    assert out["commands"], "every target should surface deploy commands"


def test_bare_metal_exports_genesis_flags():
    preset = _a_preset()
    out = dep.build_deployment(preset, "bare_metal")
    content = out["artifact"]["content"]
    if out["parameters"]["genesis_env_count"] > 0:
        assert "export GENESIS_" in content or "export VLLM_" in content


def test_unknown_preset_and_target_raise():
    with pytest.raises(KeyError):
        dep.build_deployment("no-such-preset", "compose")
    preset = _a_preset()
    with pytest.raises(ValueError):
        dep.build_deployment(preset, "no-such-target")


# ---- HTTP routes ----

def test_deploy_routes(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    from fastapi.testclient import TestClient
    from sndr.product_api.legacy.http_app import create_app

    client = TestClient(create_app(allowed_origins=()))
    targets = client.get("/api/v1/deploy/targets")
    assert targets.status_code == 200
    body = targets.json()
    assert any(t["id"] == "compose" for t in body["targets"])
    assert "host" in body

    preset = _a_preset()
    plan = client.post("/api/v1/deploy/plan", json={"preset_id": preset, "target": "compose"})
    assert plan.status_code == 200
    assert plan.json()["artifact"]["kind"] == "yaml"

    # bad target -> 400, bad preset -> 404
    assert client.post("/api/v1/deploy/plan", json={"preset_id": preset, "target": "nope"}).status_code == 400
    assert client.post("/api/v1/deploy/plan", json={"preset_id": "nope", "target": "compose"}).status_code == 404


def test_sndr_daemon_target_renders_gui_api_launcher():
    """Path B: deploy the SNDR management daemon onto a server so the GUI can
    connect to its native per-server view (patches/configs/patcher)."""
    from sndr.product_api.legacy import deployment

    assert any(t["id"] == "sndr_daemon" for t in deployment.list_targets())
    script = deployment._sndr_daemon_script(None, {})
    # Launches the Product API directly (immune to per-node cli/ divergence),
    # importing the canonical top-level `sndr` package (no vllm namespace).
    assert "sndr.product_api.legacy.http_app import run_server" in script
    # v12: mounts the canonical sndr/ tree (the top-level sndr package), else
    # the daemon dies with No module 'sndr'.
    assert '-v "${SNDR_SRC}:${SNDR_DEST}:ro"' in script
    assert 'SNDR_SRC="${MNT%%:*}"' in script
    # Runs as a sidecar from the vLLM image, re-mounting the host's sndr package
    # (it's mounted into the engine at runtime, not baked into the image), bound
    # to the LAN so the central GUI can switch to it directly (no tunnel).
    assert "docker run -d" in script and "--entrypoint python3" in script
    assert "--network host" in script and '-e SNDR_BIND="$BIND"' in script
    assert 'BIND="${SNDR_BIND:-0.0.0.0}"' in script  # LAN-bound by default
    assert "/dist-packages/sndr$" in script  # derive paths from canonical sndr mount
    cmds = deployment._commands_sndr_daemon(None)
    assert any("./run-sndr-daemon.sh" in c for c in cmds)
