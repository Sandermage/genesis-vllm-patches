# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr doctor --full` extended sections — T1.4 / audit §18.1.

Verifies the 6 new sections (wsl, image, mounts, license, engine,
remote) added by collect_report(full=True). Each is dict-shaped, never
raises on a no-vllm host, and back-compat default mode (full=False)
still produces the original 12-section shape.
"""
from __future__ import annotations

import json
import os

import pytest


def test_default_mode_omits_extended_sections():
    """Back-compat: collect_report() with no args matches old 12-section shape."""
    from sndr.compat.doctor import collect_report
    report = collect_report()
    for section in ("wsl", "image", "mounts", "license", "engine", "remote"):
        assert section not in report, (
            f"section {section} leaked into default mode — breaks back-compat"
        )


def test_full_mode_includes_extended_sections():
    from sndr.compat.doctor import collect_report
    report = collect_report(full=True)
    for section in ("wsl", "image", "mounts", "license", "engine", "remote"):
        assert section in report, f"--full missing section: {section}"


def test_full_report_json_serializable():
    from sndr.compat.doctor import collect_report
    report = collect_report(full=True)
    s = json.dumps(report, default=str)
    decoded = json.loads(s)
    assert "wsl" in decoded


# ─── individual section tests ───────────────────────────────────────────


class TestSectionWsl:
    def test_returns_dict_with_required_keys(self):
        from sndr.compat.doctor import _section_wsl
        out = _section_wsl()
        for k in ("is_wsl", "kernel", "distro", "pin_memory_ok",
                  "docker_gpu_runtime", "recommendations", "errors"):
            assert k in out
        assert isinstance(out["recommendations"], list)
        assert isinstance(out["errors"], list)

    def test_is_wsl_is_bool(self):
        from sndr.compat.doctor import _section_wsl
        assert isinstance(_section_wsl()["is_wsl"], bool)


class TestSectionImage:
    def test_skipped_without_container_env(self, monkeypatch):
        monkeypatch.delenv("SNDR_DOCTOR_CONTAINER", raising=False)
        from sndr.compat.doctor import _section_image
        out = _section_image()
        assert out["status"].startswith("skipped")
        assert out["actual_digest"] is None

    def test_returns_required_keys(self):
        from sndr.compat.doctor import _section_image
        out = _section_image()
        for k in ("container", "expected_digest", "actual_digest",
                  "drift", "allowlist_status", "status"):
            assert k in out


class TestSectionMounts:
    def test_returns_dict(self):
        from sndr.compat.doctor import _section_mounts
        out = _section_mounts()
        assert "mounts" in out
        assert "writability_violations" in out
        assert isinstance(out["mounts"], list)
        assert isinstance(out["writability_violations"], list)


class TestSectionLicense:
    def test_returns_required_keys(self):
        from sndr.compat.doctor import _section_license
        out = _section_license()
        for k in ("trust_anchor", "license_present", "license_status",
                  "legacy_mode_active", "engine_tier_eligible", "reason"):
            assert k in out

    def test_legacy_mode_reflects_env(self, monkeypatch):
        from sndr.compat.doctor import _section_license
        monkeypatch.delenv("SNDR_ENABLE_TIER_OVERRIDE", raising=False)
        monkeypatch.delenv("GENESIS_ENABLE_TIER_OVERRIDE", raising=False)
        assert _section_license()["legacy_mode_active"] is False
        monkeypatch.setenv("SNDR_ENABLE_TIER_OVERRIDE", "1")
        assert _section_license()["legacy_mode_active"] is True


class TestSectionEngine:
    def test_returns_required_keys(self):
        from sndr.compat.doctor import _section_engine
        out = _section_engine()
        for k in ("engine_available", "overlay_packages", "version", "errors"):
            assert k in out
        assert isinstance(out["overlay_packages"], list)

    def test_engine_available_is_bool(self):
        from sndr.compat.doctor import _section_engine
        assert isinstance(_section_engine()["engine_available"], bool)


class TestSectionRemote:
    def test_returns_required_keys(self):
        from sndr.compat.doctor import _section_remote_capability
        out = _section_remote_capability()
        for k in ("ssh_keys_present", "ssh_agent_running",
                  "can_resolve_remote_targets"):
            assert k in out

    def test_target_passes_through_env(self, monkeypatch):
        from sndr.compat.doctor import _section_remote_capability
        monkeypatch.setenv("SNDR_DOCTOR_REMOTE", "ops@gpu.example.com")
        out = _section_remote_capability()
        assert out.get("target") == "ops@gpu.example.com"


# ─── CLI flag wiring ────────────────────────────────────────────────────


class TestCliFlags:
    def test_full_flag_in_argparse(self, capsys):
        """--full + --json + --container parse without error."""
        from sndr.compat import doctor as D
        # main(argv=...) returns int; we just verify no SystemExit on parse
        rc = D.main(["--full", "--json"])
        assert isinstance(rc, int)
        # JSON output captured
        captured = capsys.readouterr().out
        data = json.loads(captured)
        assert "wsl" in data

    def test_redact_flag_masks_output(self, capsys, monkeypatch):
        """--redact swaps IPs / hostnames in the JSON dump."""
        from sndr.compat import doctor as D
        # Inject a fake env var that would surface unredacted on a real host
        monkeypatch.setenv("SNDR_DOCTOR_REMOTE", "ops@192.168.1.50")
        D.main(["--full", "--redact", "--json"])
        out = capsys.readouterr().out
        # Either the IP is masked or the rendering omits it; the contract
        # is "raw IP must not appear anywhere in --redact output".
        assert "192.168.1.50" not in out
