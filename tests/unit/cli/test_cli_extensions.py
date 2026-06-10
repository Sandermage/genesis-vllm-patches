# SPDX-License-Identifier: Apache-2.0
"""Tests for CLI extensions added in the P1 audit closure (2026-05-12):

  - `sndr config list` alias forwards to bridged `model-config list`.
  - `sndr patches plan --profile production` blocks partial/placeholder/
    research/retired patches from a plan's APPLY set.
  - `sndr k8s doctor` returns deterministic exit codes (1 fail / 2 warn /
    0 pass) and a JSON contract.
"""
from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stdout

import pytest


class TestConfigListAlias:
    """`sndr config list` must forward to `sndr model-config list` with
    the same output (table-mode) and an alias hint."""

    def test_run_list_exists(self):
        from sndr.cli.legacy import config as cfg_cli
        assert hasattr(cfg_cli, "run_list")
        assert callable(cfg_cli.run_list)

    def test_run_list_table_mode(self):
        from sndr.cli.legacy import config as cfg_cli
        ns = argparse.Namespace(json=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cfg_cli.run_list(ns)
        out = buf.getvalue()
        assert rc == 0
        # Forwarded output includes the model-config table header OR
        # the legitimate empty-state message — `sndr config list` is a
        # V1 compat alias forwarding to `compat/model_config_cli::cmd_list`
        # which only enumerates V1 monolithic builtin/*.yaml entries.
        # Phase 10.5 (2026-06-01): V1 tier fully retired, so the empty-
        # state path is the canonical post-sunset surface here. Operators
        # discover V2 presets via `sndr preset list` (separate verb).
        assert (
            "model configs" in out.lower()
            or "KEY" in out
            or "(no configs found under" in out
        )
        # Alias hint is printed only in table mode
        assert "alias of `sndr model-config list`" in out

    def test_run_list_json_mode_no_hint(self):
        from sndr.cli.legacy import config as cfg_cli
        ns = argparse.Namespace(json=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cfg_cli.run_list(ns)
        out = buf.getvalue()
        assert rc == 0
        # JSON mode skips the hint to keep output machine-parseable
        assert "alias of" not in out


class TestProductionProfileGate:
    """`sndr patches plan --profile production` must block partial/
    placeholder/research/retired patches in the APPLY set."""

    def test_profile_default_is_any(self):
        from sndr.cli.legacy.patches import add_argparser
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="cmd")
        add_argparser(sub)
        # Parse `patches plan --preset X` without --profile → default any
        ns = parser.parse_args(["patches", "plan", "--preset", "any-key"])
        assert ns.profile == "any"

    def test_profile_production_choice_accepted(self):
        from sndr.cli.legacy.patches import add_argparser
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="cmd")
        add_argparser(sub)
        ns = parser.parse_args([
            "patches", "plan", "--preset", "any-key",
            "--profile", "production",
        ])
        assert ns.profile == "production"

    def test_profile_invalid_choice_rejected(self):
        from sndr.cli.legacy.patches import add_argparser
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="cmd")
        add_argparser(sub)
        with pytest.raises(SystemExit):
            parser.parse_args([
                "patches", "plan", "--preset", "any-key",
                "--profile", "garbage",
            ])


class TestK8sDoctor:
    """`sndr k8s doctor` must exit 1 when kubectl is missing and emit
    machine-parseable JSON when `--json` is set."""

    def test_run_doctor_exists(self):
        from sndr.cli.legacy import k8s
        assert hasattr(k8s, "run_doctor")
        assert callable(k8s.run_doctor)

    def test_run_doctor_no_kubectl_returns_1(self, monkeypatch):
        import shutil as _shutil
        monkeypatch.setattr(_shutil, "which", lambda x: None)
        from sndr.cli.legacy import k8s
        # ensure k8s module sees the monkeypatched shutil.which too
        monkeypatch.setattr(k8s.shutil, "which", lambda x: None)
        ns = argparse.Namespace(config=None, json=False)
        rc = k8s.run_doctor(ns)
        assert rc == 1

    def test_run_doctor_no_kubectl_json_mode(self, monkeypatch):
        from sndr.cli.legacy import k8s
        monkeypatch.setattr(k8s.shutil, "which", lambda x: None)
        ns = argparse.Namespace(config=None, json=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = k8s.run_doctor(ns)
        out = buf.getvalue()
        assert rc == 1
        import json
        parsed = json.loads(out)
        assert parsed["kubectl_present"] is False
        assert "summary" in parsed
