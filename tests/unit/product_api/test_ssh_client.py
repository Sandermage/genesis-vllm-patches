# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the paramiko-backed SSH/SFTP connectivity layer.

A fake paramiko module is injected so these run without a real SSH server.
"""
from __future__ import annotations

import pytest

from sndr.product_api.legacy import ssh_client


class _FakeChannel:
    def recv_exit_status(self) -> int:
        return 0


class _FakeChannelFile:
    def __init__(self, data: bytes):
        self._data = data
        self.channel = _FakeChannel()

    def read(self) -> bytes:
        return self._data


class _FakeSFTP:
    put_calls: list = []

    def __init__(self, ok=True):
        self._ok = ok

    def normalize(self, path):
        if not self._ok:
            raise OSError("sftp denied")
        return "/home/sander"

    def listdir(self, path="."):
        return ["start_pn95.sh"]

    def putfo(self, fileobj, remotepath):
        type(self).put_calls.append(remotepath)

    def close(self):
        pass


class _FakeSSHClient:
    """Configurable stand-in for paramiko.SSHClient."""

    connect_calls: list[dict] = []
    fail_auth = False
    sftp_ok = True

    host_key_policies: list = []

    def load_system_host_keys(self):
        pass

    def set_missing_host_key_policy(self, policy):
        type(self).host_key_policies.append(policy)

    def connect(self, **kwargs):
        type(self).connect_calls.append(kwargs)
        if type(self).fail_auth:
            raise _FakeAuthError("Authentication failed")

    exec_map: dict = {}

    def exec_command(self, cmd, timeout=None):
        for needle, out in type(self).exec_map.items():
            if needle in cmd:
                return (None, _FakeChannelFile(out.encode()), _FakeChannelFile(b""))
        return (None, _FakeChannelFile(b"Linux 6.2.0"), _FakeChannelFile(b""))

    def open_sftp(self):
        if not type(self).sftp_ok:
            raise OSError("no sftp")
        return _FakeSFTP(self.sftp_ok)

    def get_transport(self):
        class _T:
            def get_banner(self_inner):
                return b"SSH-2.0-OpenSSH_9.0"
        return _T()

    def close(self):
        pass


class _FakeAuthError(Exception):
    pass


class _FakeRejectPolicy:
    pass


class _FakeAutoAddPolicy:
    pass


class _FakeParamiko:
    SSHClient = _FakeSSHClient
    AutoAddPolicy = _FakeAutoAddPolicy
    RejectPolicy = _FakeRejectPolicy

    class ssh_exception:
        AuthenticationException = _FakeAuthError
        SSHException = _FakeAuthError


@pytest.fixture()
def fake_paramiko(monkeypatch):
    _FakeSSHClient.connect_calls = []
    _FakeSSHClient.host_key_policies = []
    _FakeSSHClient.fail_auth = False
    _FakeSSHClient.sftp_ok = True
    _FakeSSHClient.exec_map = {}
    monkeypatch.setattr(ssh_client, "_load_paramiko", lambda: _FakeParamiko)
    return _FakeParamiko


def test_host_key_policy_strict_by_default(fake_paramiko, monkeypatch):
    """Hardening: an unknown host key is REJECTED unless the operator opts
    into TOFU — defends the initial connect against a MITM / DNS-spoof."""
    monkeypatch.delenv("SNDR_SSH_STRICT_HOST_KEYS", raising=False)
    monkeypatch.delenv("SNDR_SSH_HOST_KEY_POLICY", raising=False)
    ssh_client.check_connectivity(
        {"host": "h", "user": "u", "auth_method": "agent"}, timeout=1.0,
    )
    assert _FakeSSHClient.host_key_policies, "no host-key policy was set"
    assert isinstance(_FakeSSHClient.host_key_policies[-1], _FakeRejectPolicy)


def test_host_key_policy_tofu_opt_out(fake_paramiko, monkeypatch):
    """SNDR_SSH_STRICT_HOST_KEYS=0 restores trust-on-first-use for homelabs
    that don't pre-provision known_hosts."""
    monkeypatch.setenv("SNDR_SSH_STRICT_HOST_KEYS", "0")
    ssh_client.check_connectivity(
        {"host": "h", "user": "u", "auth_method": "agent"}, timeout=1.0,
    )
    assert isinstance(_FakeSSHClient.host_key_policies[-1], _FakeAutoAddPolicy)


def test_check_connectivity_success(fake_paramiko):
    out = ssh_client.check_connectivity(
        {"host": "192.0.2.10", "port": 22, "user": "operator", "auth_method": "key", "key_path": "~/.ssh/id_ed25519"}
    )
    assert out["ssh_ok"] is True
    assert out["sftp_ok"] is True
    assert out["error"] is None
    assert "Linux" in (out["uname"] or "")
    assert out["latency_ms"] is not None


def test_check_connectivity_auth_failure(fake_paramiko):
    _FakeSSHClient.fail_auth = True
    out = ssh_client.check_connectivity(
        {"host": "192.0.2.10", "user": "operator", "auth_method": "password", "password": "wrong"}
    )
    assert out["ssh_ok"] is False
    assert "Authentication" in (out["error"] or "")


def test_check_connectivity_sftp_down_but_ssh_up(fake_paramiko):
    _FakeSSHClient.sftp_ok = False
    out = ssh_client.check_connectivity(
        {"host": "h", "user": "u", "auth_method": "agent"}
    )
    assert out["ssh_ok"] is True
    assert out["sftp_ok"] is False


def test_password_pulled_from_secrets_store(fake_paramiko, monkeypatch):
    # When auth=password and no explicit password, the stored one is used.
    monkeypatch.setattr(ssh_client.secrets_store, "get_secret", lambda name: "stored-pw" if name == "ssh:host-x" else None)
    ssh_client.check_connectivity(
        {"host": "h", "user": "u", "auth_method": "password", "secret_id": "ssh:host-x"}
    )
    assert _FakeSSHClient.connect_calls[-1].get("password") == "stored-pw"


def test_paramiko_missing_is_graceful(monkeypatch):
    monkeypatch.setattr(ssh_client, "_load_paramiko", lambda: None)
    out = ssh_client.check_connectivity({"host": "h", "user": "u"})
    assert out["ssh_ok"] is False
    assert out["available"] is False
    assert "paramiko" in (out["error"] or "").lower()


def test_discover_api_key_from_container_env(fake_paramiko):
    _FakeSSHClient.exec_map = {
        "docker ps": "vllm-pn95-2xa5000\nredis\n",
        "docker inspect vllm-pn95-2xa5000": "PATH=/usr/bin\nVLLM_API_KEY=genesis-local\nFOO=bar\n",
    }
    out = ssh_client.discover_api_key({"host": "192.0.2.10", "user": "operator", "auth_method": "agent"})
    assert out["found"] is True
    assert out["key"] == "genesis-local"
    assert out["source"] == "container:vllm-pn95-2xa5000"


def test_discover_api_key_falls_back_to_start_script(fake_paramiko):
    _FakeSSHClient.exec_map = {
        "docker ps": "",  # no containers
        "grep -rhoE": "--api-key genesis-local\n",
    }
    out = ssh_client.discover_api_key({"host": "h", "user": "u", "auth_method": "agent"})
    assert out["found"] is True and out["key"] == "genesis-local" and out["source"] == "start-script"


def test_discover_api_key_not_found(fake_paramiko):
    _FakeSSHClient.exec_map = {"docker ps": "", "grep -rhoE": ""}
    out = ssh_client.discover_api_key({"host": "h", "user": "u", "auth_method": "agent"})
    assert out["found"] is False and out["error"]


def test_discover_host_finds_engines_and_gpus(fake_paramiko):
    _FakeSSHClient.exec_map = {
        "docker ps": (
            "vllm-pn95-2xa5000\t0.0.0.0:8101->8000/tcp, :::8101->8000/tcp\tvllm/vllm-openai:nightly\tUp 3 hours\n"
            "vllm-35b-prod\t0.0.0.0:8102->8000/tcp\tvllm/vllm-openai:nightly-dev354\tUp 1 day\n"
            "redis\t6379/tcp\tredis:7\tUp 2 days\n"
        ),
        "nvidia-smi --query-gpu": "NVIDIA RTX A5000, 24564, 41, 8.6\nNVIDIA RTX A5000, 24564, 0, 8.6\n",
        "nvidia-smi topo -m": "\tGPU0\tGPU1\nGPU0\t X \tPHB\nGPU1\tPHB\t X \n",
    }
    out = ssh_client.discover_host({"host": "192.0.2.10", "user": "operator", "auth_method": "agent"})
    assert out["docker"] is True
    # Only vLLM containers, with their published host ports parsed.
    ports = {e["container"]: e["host_port"] for e in out["engines"]}
    assert ports == {"vllm-pn95-2xa5000": 8101, "vllm-35b-prod": 8102}
    assert len(out["gpus"]) == 2 and out["gpus"][0]["name"] == "NVIDIA RTX A5000"
    # Arch classification + interconnect topology are attached.
    assert out["gpus"][0]["arch"].startswith("Ampere") and out["gpus"][0]["compute_cap"] == 8.6
    assert out["arch_advice"]["fp8_kv_native"] is False
    assert out["interconnect"]["has_nvlink"] is False and out["interconnect"]["worst_link"] == "PCIe"


def test_read_model_config_from_running_container(fake_paramiko):
    cfg = (
        '{"model_type":"qwen3_moe","num_hidden_layers":62,"num_attention_heads":40,'
        '"num_key_value_heads":4,"head_dim":128,"hidden_size":5120,"num_experts":128,'
        '"max_position_embeddings":262144,"quantization_config":{"quant_method":"fp8"}}'
    )
    _FakeSSHClient.exec_map = {
        "docker inspect": 'python3 -m vllm.entrypoints.openai.api_server --model /models/Qwen3.6-35B-A3B-FP8 --port 8102',
        "cat '/models/Qwen3.6-35B-A3B-FP8/config.json'": cfg,
        "du -sb": "37580963840\t/models/Qwen3.6-35B-A3B-FP8\n",
    }
    out = ssh_client.read_model_config({"host": "192.0.2.10", "user": "operator", "auth_method": "agent"},
                                       container="vllm-35b-prod")
    assert out["ok"] is True
    assert out["num_layers"] == 62 and out["num_kv_heads"] == 4 and out["head_dim"] == 128
    assert out["is_moe"] is True and out["num_experts"] == 128
    assert out["max_context"] == 262144
    assert out["weights_bytes"] == 37580963840  # exact size from du, not a guess


def test_read_model_config_rejects_bad_container(fake_paramiko):
    out = ssh_client.read_model_config({"host": "h", "user": "u"}, container="bad name!")
    assert out["ok"] is False and "container" in out["error"]


def test_read_sndr_state_introspects_the_container(fake_paramiko):
    # The in-container python prints a JSON line; we parse the last one.
    _FakeSSHClient.exec_map = {
        "docker ps": "vllm-gemma4\n",
        "base64 -d | python3 -": 'some warning\n{"vllm": "0.21.1rc1.dev354", "sndr": "11.0.0", "configs": 11, "patches": 134}\n',
    }
    out = ssh_client.read_sndr_state({"host": "192.0.2.10", "user": "operator", "auth_method": "agent"})
    assert out["ok"] is True and out["container"] == "vllm-gemma4"
    assert out["sndr_version"] == "11.0.0" and out["vllm_version"] == "0.21.1rc1.dev354"
    assert out["configs"] == 11 and out["patches"] == 134


def test_read_sndr_state_rejects_bad_container(fake_paramiko):
    out = ssh_client.read_sndr_state({"host": "h", "user": "u"}, container="bad name!")
    assert out["ok"] is False and "container" in out["error"]


def test_run_apply_uploads_artifact_and_runs_commands(fake_paramiko):
    _FakeSFTP.put_calls = []
    out = ssh_client.run_apply(
        {"host": "192.0.2.10", "user": "operator", "auth_method": "agent"},
        artifact_name="run-sndr-daemon.sh", artifact_content="#!/bin/sh\necho hi\n",
        commands=["chmod +x run-sndr-daemon.sh", "echo started"],
    )
    assert out["ok"] is True and out["error"] is None
    assert _FakeSFTP.put_calls == ["sndr-install/run-sndr-daemon.sh"]  # SFTP'd into the workdir
    cmds = [s["cmd"] for s in out["steps"]]
    assert cmds == ["upload run-sndr-daemon.sh", "chmod +x run-sndr-daemon.sh", "echo started"]


def test_run_apply_rejects_bad_artifact_name(fake_paramiko):
    out = ssh_client.run_apply({"host": "h", "user": "u"}, artifact_name="../evil.sh",
                               artifact_content="x", commands=[])
    assert out["ok"] is False and "artifact" in out["error"]


def test_discover_host_graceful_without_docker(fake_paramiko):
    _FakeSSHClient.exec_map = {"docker ps": "", "nvidia-smi": ""}
    # docker ps returns empty (rc 0) -> docker True but no engines; nvidia empty.
    out = ssh_client.discover_host({"host": "h", "user": "u", "auth_method": "agent"})
    assert out["engines"] == [] and out["gpus"] == []


def test_discover_host_captures_active_genesis_flags(fake_paramiko):
    # The running container's GENESIS_ENABLE_*=1 env are the host's live patches.
    _FakeSSHClient.exec_map = {
        "docker ps": "vllm-35b-prod\t0.0.0.0:8102->8000/tcp\tvllm/vllm-openai:nightly\tUp 1 day\n",
        "docker inspect": "PATH=/usr/bin\nGENESIS_ENABLE_P94=1\nGENESIS_ENABLE_PN95=0\nGENESIS_ENABLE_P82=1\nVLLM_API_KEY=x\n",
        "nvidia-smi": "",
    }
    out = ssh_client.discover_host({"host": "192.0.2.10", "user": "operator", "auth_method": "agent"})
    flags = out["engines"][0]["genesis_flags"]
    assert flags == ["GENESIS_ENABLE_P94", "GENESIS_ENABLE_P82"]  # only the =1 ones


def test_engine_port_candidates_prefers_api_over_metrics():
    f = ssh_client._engine_port_candidates
    # Canonical -p 8102:8000 (+ IPv6 dup) → single API port.
    assert f("0.0.0.0:8102->8000/tcp, :::8102->8000/tcp") == [8102]
    # Metrics mapping listed first must NOT win; container-8000 is API-first.
    assert f("0.0.0.0:8103->8001/tcp, 0.0.0.0:8102->8000/tcp") == [8102, 8103]
    # Custom -p 8102:8102 layout (no container-8000) → still surfaced.
    assert f("0.0.0.0:8102->8102/tcp") == [8102]
    # Custom API + metrics: API-first, metrics last.
    assert f("0.0.0.0:8102->8102/tcp, 0.0.0.0:9102->8001/tcp") == [8102, 9102]
    # No published ports.
    assert f("") == []
    assert f("8000/tcp") == []  # exposed but not published → no host port
