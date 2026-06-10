# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr launch --extra-env KEY=VALUE` — operator probe env
injection without preset/profile YAML edit.

Phase 7.G4.CLI-EXTRA-ENV (2026-05-23): added to support ad-hoc
GENESIS_ENABLE_* / SNDR_* probe overrides at launch time. The primary
use case is enabling per-step trace patches (PN248 acceptance trace,
etc.) for a single bench run without polluting the canonical preset
that's used in PROD.

Contract:

  • `--extra-env KEY=VALUE` is repeatable (argparse action='append').
  • Empty list is the default — same env block as before this flag.
  • KEY starting with GENESIS_/SNDR_ lands in cfg.genesis_env.
  • Other KEYs land in cfg.system_env.
  • Conflicts with preset's own env are last-wins (override) and
    logged as a warn so the diff is auditable.
  • Missing `=` or empty KEY exits the CLI with code 2.
  • VALUE may contain further `=` characters (JSON payloads etc.).
"""
from __future__ import annotations

import argparse
import pytest

from sndr.cli.legacy import launch as L


# ─── Argparse surface ────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sndr")
    sub = parser.add_subparsers()
    L.add_argparser(sub)
    return parser


class TestArgparserFlag:
    def test_extra_env_default_empty(self):
        ns = _build_parser().parse_args(["launch", "x"])
        assert ns.extra_env == []

    def test_single_extra_env(self):
        ns = _build_parser().parse_args([
            "launch", "x", "--extra-env", "FOO=bar",
        ])
        assert ns.extra_env == ["FOO=bar"]

    def test_repeated_extra_env_accumulates(self):
        ns = _build_parser().parse_args([
            "launch", "x",
            "--extra-env", "FOO=bar",
            "--extra-env", "BAZ=qux",
        ])
        assert ns.extra_env == ["FOO=bar", "BAZ=qux"]


# ─── _parse_extra_env helper ─────────────────────────────────────────────


class TestParseExtraEnv:
    def test_empty_list_returns_empty_dict(self):
        assert L._parse_extra_env([]) == {}

    def test_single_pair(self):
        assert L._parse_extra_env(["FOO=bar"]) == {"FOO": "bar"}

    def test_multiple_pairs(self):
        assert L._parse_extra_env(["A=1", "B=2"]) == {"A": "1", "B": "2"}

    def test_empty_value_allowed(self):
        # Some env vars are presence-checked; empty value is OK.
        assert L._parse_extra_env(["FOO="]) == {"FOO": ""}

    def test_value_with_equals_preserved(self):
        # Split only on the FIRST `=` — JSON-like values must pass intact.
        assert L._parse_extra_env(['CFG={"x":1}']) == {"CFG": '{"x":1}'}

    def test_missing_equals_fatal(self):
        with pytest.raises(SystemExit) as e:
            L._parse_extra_env(["NOPE"])
        assert e.value.code == 2

    def test_empty_key_fatal(self):
        with pytest.raises(SystemExit) as e:
            L._parse_extra_env(["=value"])
        assert e.value.code == 2


# ─── _apply_extra_env helper ─────────────────────────────────────────────


class _FakeCfg:
    def __init__(self, system_env=None, genesis_env=None):
        self.system_env = dict(system_env or {})
        self.genesis_env = dict(genesis_env or {})


class TestApplyExtraEnv:
    def test_empty_dict_is_noop(self):
        cfg = _FakeCfg(system_env={"A": "1"}, genesis_env={"GENESIS_X": "1"})
        L._apply_extra_env(cfg, {})
        assert cfg.system_env == {"A": "1"}
        assert cfg.genesis_env == {"GENESIS_X": "1"}

    def test_genesis_key_lands_in_genesis_env(self):
        cfg = _FakeCfg()
        L._apply_extra_env(cfg, {"GENESIS_ENABLE_PN248_ACCEPTANCE_TRACE": "1"})
        assert cfg.genesis_env == {
            "GENESIS_ENABLE_PN248_ACCEPTANCE_TRACE": "1",
        }
        assert cfg.system_env == {}

    def test_sndr_key_lands_in_genesis_env(self):
        cfg = _FakeCfg()
        L._apply_extra_env(cfg, {"SNDR_TRACE_LEVEL": "debug"})
        assert cfg.genesis_env == {"SNDR_TRACE_LEVEL": "debug"}
        assert cfg.system_env == {}

    def test_other_key_lands_in_system_env(self):
        cfg = _FakeCfg()
        L._apply_extra_env(cfg, {"VLLM_LOGGING_LEVEL": "DEBUG"})
        assert cfg.system_env == {"VLLM_LOGGING_LEVEL": "DEBUG"}
        assert cfg.genesis_env == {}

    def test_override_existing_key(self):
        cfg = _FakeCfg(genesis_env={"GENESIS_X": "0"})
        L._apply_extra_env(cfg, {"GENESIS_X": "1"})
        assert cfg.genesis_env["GENESIS_X"] == "1"

    def test_no_genesis_env_attr_initialised(self):
        cfg = _FakeCfg()
        cfg.genesis_env = None
        L._apply_extra_env(cfg, {"GENESIS_X": "1"})
        assert cfg.genesis_env == {"GENESIS_X": "1"}

    def test_no_system_env_attr_initialised(self):
        cfg = _FakeCfg()
        cfg.system_env = None
        L._apply_extra_env(cfg, {"FOO": "bar"})
        assert cfg.system_env == {"FOO": "bar"}

    def test_mixed_routing(self):
        cfg = _FakeCfg()
        L._apply_extra_env(cfg, {
            "GENESIS_ENABLE_PN248_ACCEPTANCE_TRACE": "1",
            "VLLM_LOGGING_LEVEL": "DEBUG",
            "SNDR_PROBE": "x",
        })
        assert cfg.genesis_env == {
            "GENESIS_ENABLE_PN248_ACCEPTANCE_TRACE": "1",
            "SNDR_PROBE": "x",
        }
        assert cfg.system_env == {"VLLM_LOGGING_LEVEL": "DEBUG"}
