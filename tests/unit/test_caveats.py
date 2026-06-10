# SPDX-License-Identifier: Apache-2.0
"""Y13 + C22 (UNIFIED_CONFIG plan 2026-05-09) — caveats registry + CLI tests."""
from __future__ import annotations

import json
import argparse

import pytest

from sndr.caveats import (
    Caveat, KNOWN_CAVEATS, match_caveats, get_caveat, list_caveat_ids,
)


# ─── Caveat dataclass

def test_caveat_dataclass_constructible():
    c = Caveat(
        id="test", severity="info", title="t", message="m",
        match_fn=lambda f: f.get("trigger", False),
    )
    assert c.matches({"trigger": True})
    assert not c.matches({})


def test_caveat_no_match_fn_never_fires():
    c = Caveat(id="x", severity="info", title="t", message="m")
    assert c.matches({"anything": True}) is False


def test_caveat_match_fn_exception_treated_as_no_match():
    def boom(_facts):
        raise RuntimeError("oops")
    c = Caveat(id="x", severity="info", title="t", message="m",
               match_fn=boom)
    assert c.matches({}) is False


# ─── KNOWN_CAVEATS shape

def test_known_caveats_nonempty():
    assert len(KNOWN_CAVEATS) >= 4


def test_known_caveats_unique_ids():
    ids = [c.id for c in KNOWN_CAVEATS]
    assert len(set(ids)) == len(ids), f"duplicate caveat ids: {ids}"


def test_known_caveats_all_have_match_fn():
    for c in KNOWN_CAVEATS:
        assert c.match_fn is not None, f"caveat {c.id!r} missing match_fn"


def test_known_caveats_severity_valid():
    for c in KNOWN_CAVEATS:
        assert c.severity in ("info", "warning", "error"), (
            f"caveat {c.id!r} has bad severity {c.severity!r}"
        )


# ─── Match scenarios

def test_match_proxmox_lxc_kernel_617_fires():
    facts = {"virtualization": "lxc",
              "os": {"system": "Linux", "release": "6.17.2-pve"}}
    triggered = match_caveats(facts)
    ids = [c.id for c in triggered]
    assert "proxmox_lxc_kernel_617" in ids


def test_match_proxmox_lxc_kernel_616_does_not_fire():
    facts = {"virtualization": "lxc",
              "os": {"system": "Linux", "release": "6.16.0-pve"}}
    triggered = match_caveats(facts)
    ids = [c.id for c in triggered]
    assert "proxmox_lxc_kernel_617" not in ids


def test_match_single_24g_no_pn95_fires():
    facts = {
        "nvidia": {"n_gpus": 1, "gpu_total_vram_mib": [24564]},
        "genesis_env": {},  # PN95 not set → caveat fires
    }
    triggered = match_caveats(facts)
    ids = [c.id for c in triggered]
    assert "single_3090_long_ctx_vision_no_pn95" in ids


def test_match_single_24g_with_pn95_does_not_fire():
    facts = {
        "nvidia": {"n_gpus": 1, "gpu_total_vram_mib": [24564]},
        "genesis_env": {"GENESIS_ENABLE_PN95_TIER_AWARE_CACHE": "1"},
    }
    triggered = match_caveats(facts)
    ids = [c.id for c in triggered]
    assert "single_3090_long_ctx_vision_no_pn95" not in ids


def test_match_two_gpus_does_not_fire_single_24g():
    facts = {
        "nvidia": {"n_gpus": 2, "gpu_total_vram_mib": [24564, 24564]},
        "genesis_env": {},
    }
    triggered = match_caveats(facts)
    ids = [c.id for c in triggered]
    assert "single_3090_long_ctx_vision_no_pn95" not in ids


def test_match_docker_no_nvidia_runtime_fires():
    facts = {
        "docker": {
            "installed": True, "daemon_running": True,
            "nvidia_runtime_present": False,
        },
    }
    triggered = match_caveats(facts)
    ids = [c.id for c in triggered]
    assert "docker_no_nvidia_runtime" in ids


def test_match_vllm_pin_drift_info_fires():
    facts = {
        "vllm": {"installed": True, "version": "0.20.99rc999.dev500"},
        "vllm_pin_in_allowlist": False,
    }
    triggered = match_caveats(facts)
    ids = [c.id for c in triggered]
    assert "vllm_pin_drift_from_genesis_known_good" in ids


# ─── get_caveat / list_caveat_ids

def test_get_caveat_known_id():
    c = get_caveat("proxmox_lxc_kernel_617")
    assert c is not None
    assert c.id == "proxmox_lxc_kernel_617"


def test_get_caveat_unknown_id_returns_none():
    assert get_caveat("nonexistent-caveat-xyz") is None


def test_list_caveat_ids_matches_registry():
    ids = list_caveat_ids()
    assert len(ids) == len(KNOWN_CAVEATS)


# ─── CLI

def _parse(args: list[str]) -> argparse.Namespace:
    from sndr.cli.legacy.caveats import add_argparser
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    add_argparser(sub)
    return parser.parse_args(args)


def test_cli_argparser_list_subcommand():
    ns = _parse(["caveats", "list"])
    assert ns.caveats_cmd == "list"


def test_cli_argparser_check_strict():
    ns = _parse(["caveats", "check", "--strict", "--json"])
    assert ns.caveats_cmd == "check"
    assert ns.strict is True
    assert ns.json is True


def test_cli_argparser_explain_requires_id():
    with pytest.raises(SystemExit):
        _parse(["caveats", "explain"])


def test_cli_run_list_human(capsys):
    from sndr.cli.legacy.caveats import run_list
    ns = _parse(["caveats", "list"])
    rc = run_list(ns)
    assert rc == 0
    out = capsys.readouterr().out
    assert "KNOWN_CAVEATS" in out
    assert "proxmox_lxc_kernel_617" in out


def test_cli_run_list_json(capsys):
    from sndr.cli.legacy.caveats import run_list
    ns = _parse(["caveats", "list", "--json"])
    rc = run_list(ns)
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert isinstance(data, list)
    assert len(data) >= 4
    assert all("id" in c and "severity" in c for c in data)


def test_cli_run_explain_known(capsys):
    from sndr.cli.legacy.caveats import run_explain
    ns = _parse(["caveats", "explain", "proxmox_lxc_kernel_617"])
    rc = run_explain(ns)
    assert rc == 0
    out = capsys.readouterr().out
    assert "proxmox_lxc_kernel_617" in out


def test_cli_run_explain_unknown_returns_2(capsys):
    from sndr.cli.legacy.caveats import run_explain
    ns = _parse(["caveats", "explain", "nonexistent-xyz"])
    rc = run_explain(ns)
    assert rc == 2


def test_cli_top_level_dispatches_caveats():
    from sndr.cli.legacy import cli_main
    rc = cli_main(["caveats", "list", "--json"])
    assert rc == 0
