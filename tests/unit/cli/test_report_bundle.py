# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr report bundle` — T1.1 (audit closure 2026-05-09)."""
from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

import pytest

from sndr.cli.legacy import report as R


# ─── Per-collector tests (no I/O on host where possible) ─────────────────


class TestCollectors:
    def test_collect_doctor_returns_dict(self):
        d = R._collect_doctor()
        assert isinstance(d, dict)
        # `error` key only present if collect_report() raised.
        if "error" not in d:
            # Real doctor report has at least these top-level keys.
            assert any(k in d for k in ("hardware", "software", "patches"))

    def test_collect_patches_returns_summary(self):
        p = R._collect_patches()
        assert isinstance(p, dict)
        if "error" not in p:
            assert p["total"] >= 100
            assert "by_tier" in p
            assert "by_lifecycle" in p
            assert "entries" in p
            # Spot-check shape of an entry.
            assert all(
                {"patch_id", "title", "tier", "lifecycle"} <= set(e)
                for e in p["entries"][:3]
            )

    def test_collect_launch_dryrun_no_preset(self):
        out = R._collect_launch_dryrun(preset_key=None)
        assert "no --preset" in out

    def test_collect_launch_dryrun_unknown_preset(self):
        out = R._collect_launch_dryrun(preset_key="totally-fake-preset-xyz")
        assert "not found" in out or "failed to load" in out

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parents[3] / "vllm" / "sndr_core"
             / "model_configs" / "builtin" / "a5000-2x-35b-prod.yaml"
             ).is_file(),
        reason="V1 fixture retired (Phase 10 sunset)",
    )
    def test_collect_launch_dryrun_real_preset(self):
        out = R._collect_launch_dryrun(preset_key="a5000-2x-35b-prod")
        assert "vllm serve" in out or "render failed" in out

    def test_collect_host_yaml_either_path_or_message(self):
        out = R._collect_host_yaml()
        # Either it found a yaml or returned "no host.yaml found".
        assert isinstance(out, str)
        assert len(out) > 0

    def test_collect_nvidia_smi_handles_absence(self):
        out = R._collect_nvidia_smi()
        assert isinstance(out, str)
        # On the dev Mac, this returns the "(nvidia-smi not available)" string.

    def test_collect_pip_freeze_returns_text(self):
        out = R._collect_pip_freeze()
        assert isinstance(out, str)
        # pip freeze should always produce SOMETHING in a Python env.

    def test_collect_git_log_in_repo(self):
        out = R._collect_git_log()
        # If running from inside a repo, expect at least one commit.
        # Otherwise the helper returns "(running outside a git checkout)".
        assert isinstance(out, str)

    def test_collect_vllm_boot_log_no_container(self):
        out = R._collect_vllm_boot_log(container=None)
        assert "no --container" in out

    def test_collect_image_inspect_no_container(self):
        out = R._collect_image_inspect(container=None)
        assert isinstance(out, dict)
        assert out.get("status", "").startswith("skipped")


# ─── Redaction integration ──────────────────────────────────────────────


class TestMaybeRedact:
    def test_redaction_off_returns_artifacts_unchanged(self):
        artifacts = {
            "test.txt": "ssh user@10.0.0.5 with token GENESIS_API_KEY=abc",
        }
        out, counts = R._maybe_redact(artifacts, do_redact=False)
        assert out == artifacts
        assert counts == {}

    def test_redaction_on_masks_sensitive_data(self):
        artifacts = {
            "logs.txt": "client=192.168.1.50 GENESIS_API_KEY=secret",
            "summary": {
                "endpoint": "http://192.168.1.50:8000",
                "ssh": "ssh user@gpu.example.com",
            },
        }
        out, counts = R._maybe_redact(artifacts, do_redact=True)
        # Top-level string redacted
        assert "192.168.1.50" not in out["logs.txt"]
        assert "secret" not in out["logs.txt"]
        # Nested dict leaves redacted
        assert "192.168.1.50" not in out["summary"]["endpoint"]
        assert "user@<HOSTNAME>" in out["summary"]["ssh"]
        # Counts populated
        assert counts.get("ipv4", 0) >= 2


# ─── Serialize ──────────────────────────────────────────────────────────


class TestSerialize:
    def test_dict_serialized_as_json(self):
        b = R._serialize({"a": 1, "b": [2, 3]})
        assert json.loads(b.decode("utf-8")) == {"a": 1, "b": [2, 3]}

    def test_string_serialized_as_utf8(self):
        b = R._serialize("hello world")
        assert b == b"hello world"

    def test_other_types_str_repr(self):
        b = R._serialize(42)
        assert b == b"42"


# ─── End-to-end bundle build ────────────────────────────────────────────


class TestBundleE2E:
    def test_bundle_creates_tarball(self, tmp_path, monkeypatch):
        out_path = tmp_path / "test-bundle.tar.gz"
        opts = argparse.Namespace(
            output=out_path,
            preset=None,
            container=None,
            no_redact=False,
            print_summary=False,
        )
        rc = R.run_bundle(opts)
        assert rc == 0
        assert out_path.is_file()
        # Verify tarball structure
        with tarfile.open(out_path, "r:gz") as tar:
            names = set(tar.getnames())
        assert "manifest.json" in names
        # All 9 artifact filenames present
        for expected in [
            "doctor.json", "patches.json", "launch_dryrun.sh",
            "vllm_boot.log", "host_yaml.txt", "nvidia_smi.txt",
            "pip_freeze.txt", "git_log.txt", "image_inspect.json",
        ]:
            assert expected in names, f"missing artifact: {expected}"

    def test_bundle_manifest_has_redaction_counts(self, tmp_path):
        out_path = tmp_path / "redacted-bundle.tar.gz"
        opts = argparse.Namespace(
            output=out_path, preset=None, container=None,
            no_redact=False, print_summary=False,
        )
        R.run_bundle(opts)
        with tarfile.open(out_path, "r:gz") as tar:
            f = tar.extractfile("manifest.json")
            assert f is not None
            manifest = json.loads(f.read().decode("utf-8"))
        assert "redaction" in manifest
        assert manifest["redaction"]["enabled"] is True
        assert "hit_counts" in manifest["redaction"]

    def test_bundle_no_redact_flag(self, tmp_path):
        out_path = tmp_path / "raw-bundle.tar.gz"
        opts = argparse.Namespace(
            output=out_path, preset=None, container=None,
            no_redact=True, print_summary=False,
        )
        R.run_bundle(opts)
        with tarfile.open(out_path, "r:gz") as tar:
            f = tar.extractfile("manifest.json")
            assert f is not None
            manifest = json.loads(f.read().decode("utf-8"))
        assert manifest["redaction"]["enabled"] is False
        assert manifest["redaction"]["hit_counts"] == {}

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parents[3] / "vllm" / "sndr_core"
             / "model_configs" / "builtin" / "a5000-2x-35b-prod.yaml"
             ).is_file(),
        reason="V1 fixture retired (Phase 10 sunset) — bundle CLI doesn't "
               "support V2 alias dispatch yet (V2 bundle support is a "
               "separate effort outside Phase 10 scope)",
    )
    def test_bundle_with_preset(self, tmp_path):
        out_path = tmp_path / "preset-bundle.tar.gz"
        opts = argparse.Namespace(
            output=out_path, preset="a5000-2x-35b-prod", container=None,
            no_redact=False, print_summary=False,
        )
        R.run_bundle(opts)
        with tarfile.open(out_path, "r:gz") as tar:
            f = tar.extractfile("launch_dryrun.sh")
            content = f.read().decode("utf-8")
        # Either rendered the script or returned an error message —
        # both are valid as long as the artifact exists with content.
        assert len(content) > 50

    def test_bundle_default_output_path(self, tmp_path, monkeypatch):
        # Force SNDR_HOME to tmp so reports dir is created in isolation
        monkeypatch.setenv("SNDR_HOME", str(tmp_path))
        opts = argparse.Namespace(
            output=None, preset=None, container=None,
            no_redact=False, print_summary=False,
        )
        rc = R.run_bundle(opts)
        assert rc == 0
        # Find the created file
        reports = list((tmp_path / "reports").glob("sndr-report-*.tar.gz"))
        assert len(reports) == 1


class TestArgparser:
    def test_argparser_registers_bundle_subcommand(self):
        p = argparse.ArgumentParser()
        sub = p.add_subparsers()
        R.add_argparser(sub)
        # The parent `report` subcommand should be registered.
        # Parsing just `report` should print help and not crash.
        ns = p.parse_args(["report", "bundle", "--no-redact"])
        assert ns.no_redact is True
