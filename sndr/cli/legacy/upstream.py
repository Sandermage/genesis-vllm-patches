# SPDX-License-Identifier: Apache-2.0
"""C17 (UNIFIED_CONFIG plan 2026-05-09) — `sndr upstream` subcommand tree.

Surfaces the Y11 `UpstreamPinPolicy` schema block + the project-wide
`KNOWN_GOOD_VLLM_PINS` allowlist via a CLI. Read-only today (the
allowlist is a hardcoded tuple in `detection/guards.py` — promotion
requires a code edit + PR; this CLI just SHOWS the state).

Subcommands:

  sndr upstream check
      Detect the running vllm pin (if importable). Print whether it's
      in KNOWN_GOOD_VLLM_PINS. If a preset is given, also check
      against `cfg.upstream` Y11 policy.

  sndr upstream show <preset_key>
      Print the preset's upstream policy: required_pin, allowed_pins,
      blocked_pins, and how the running pin compares.

  sndr upstream list
      List every entry in KNOWN_GOOD_VLLM_PINS with relative-age info.

(Future: `sndr upstream watch` polls upstream nightly tags + flags
known-bad pin candidates. Out of scope for the read-only first cut.)
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Optional

from . import _io


__all__ = ["add_argparser", "run_check", "run_show", "run_list"]


def add_argparser(subparsers: Any) -> None:
    """Register the `sndr upstream` subcommand tree."""
    p = subparsers.add_parser(
        "upstream",
        help="vLLM pin allowlist + per-config upstream policy (UNIFIED_CONFIG C17).",
        description=(
            "Check whether the running vllm pin is in the project's "
            "KNOWN_GOOD_VLLM_PINS allowlist and/or a per-config "
            "Y11 UpstreamPinPolicy block. Read-only — does not "
            "modify the allowlist."
        ),
    )
    sub = p.add_subparsers(dest="upstream_cmd", required=True)

    p_check = sub.add_parser(
        "check",
        help="Check the running vllm pin against allowlist + optional preset.",
    )
    p_check.add_argument(
        "--config", default=None,
        help="Also validate against this preset's upstream policy.",
    )
    p_check.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON.",
    )
    p_check.add_argument(
        "--strict", action="store_true",
        help="Exit 1 on any policy violation (default 0 — read-only view).",
    )
    p_check.set_defaults(func=run_check)

    p_show = sub.add_parser(
        "show",
        help="Show one preset's upstream policy + comparison vs running pin.",
    )
    p_show.add_argument("config", help="model_config preset key")
    p_show.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON.")
    p_show.set_defaults(func=run_show)

    p_list = sub.add_parser(
        "list",
        help="List every pin in KNOWN_GOOD_VLLM_PINS with age info.",
    )
    p_list.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON.")
    p_list.set_defaults(func=run_list)


def _running_pin() -> Optional[str]:
    """Return the running vllm pin, or None if not importable."""
    try:
        from sndr.engines.vllm.detection.guards import (
            get_vllm_full_version_string,
        )
        return get_vllm_full_version_string()
    except Exception:
        return None


def _known_good() -> tuple[str, ...]:
    """Return the project KNOWN_GOOD_VLLM_PINS allowlist."""
    try:
        from sndr.engines.vllm.detection.guards import KNOWN_GOOD_VLLM_PINS
        return KNOWN_GOOD_VLLM_PINS
    except Exception:
        return ()


def _resolve_cfg(key: str):
    from sndr.model_configs.registry import get
    cfg = get(key)
    if cfg is None:
        _io.warn(f"unknown preset key {key!r}")
        try:
            from sndr.model_configs.registry import list_keys
            _io.info(f"available: {', '.join(sorted(list_keys()))}")
        except Exception:
            pass
    return cfg


# ─── check

def run_check(args: argparse.Namespace) -> int:
    pin = _running_pin()
    allowlist = _known_good()
    in_allowlist = (pin is not None) and (pin in allowlist)
    cfg_msg: Optional[str] = None
    cfg_violation = False

    cfg = None
    if args.config:
        cfg = _resolve_cfg(args.config)
        if cfg is None:
            return 2
        if cfg.upstream is not None and pin is not None:
            cfg_msg = cfg.upstream.check(pin)
            cfg_violation = cfg_msg is not None

    if args.json:
        out: dict = {
            "running_pin": pin,
            "in_known_good_allowlist": in_allowlist,
            "known_good_count": len(allowlist),
        }
        if cfg is not None:
            out["preset"] = args.config
            out["preset_violation"] = cfg_violation
            out["preset_message"] = cfg_msg
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print("sndr upstream check")
        print("─" * 50)
        print(f"  Running pin:       {pin or '<vllm not importable>'}")
        if pin is not None:
            mark = "✓" if in_allowlist else "✗"
            print(f"  In allowlist:      {mark} {in_allowlist} "
                  f"({len(allowlist)} pins known-good)")
        if cfg is not None:
            print()
            print(f"  Preset '{args.config}' policy:")
            if cfg.upstream is None:
                print("    (no Y11 upstream block declared)")
            elif cfg_msg is None:
                print(f"    ✓ pin {pin!r} accepted by preset policy")
            else:
                print(f"    ✗ {cfg_msg}")

    if args.strict:
        if not in_allowlist:
            return 1
        if cfg_violation:
            return 1
    return 0


# ─── show

def run_show(args: argparse.Namespace) -> int:
    cfg = _resolve_cfg(args.config)
    if cfg is None:
        return 2
    pin = _running_pin()

    pol = cfg.upstream
    if args.json:
        out = {
            "preset": args.config,
            "running_pin": pin,
            "policy": None if pol is None else {
                "required_pin": pol.required_pin,
                "allowed_pins": list(pol.allowed_pins),
                "blocked_pins": list(pol.blocked_pins),
                "notes": pol.notes,
            },
            "policy_check": (pol.check(pin) if pol is not None and pin else None),
        }
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    print(f"sndr upstream show '{args.config}'")
    print("─" * 50)
    print(f"  Running pin:   {pin or '<vllm not importable>'}")
    if pol is None:
        print("  Policy:        (no Y11 upstream block declared)")
        return 0
    print(f"  required_pin:  {pol.required_pin or '_unset_'}")
    print(f"  allowed_pins:  {', '.join(pol.allowed_pins) or '_empty_'}")
    print(f"  blocked_pins:  {', '.join(pol.blocked_pins) or '_empty_'}")
    if pol.notes:
        print(f"  notes:         {pol.notes}")
    if pin is not None:
        print()
        msg = pol.check(pin)
        if msg is None:
            print(f"  ✓ running pin {pin!r} accepted by this policy")
        else:
            print(f"  ✗ {msg}")
    return 0


# ─── list

def run_list(args: argparse.Namespace) -> int:
    allowlist = _known_good()
    if args.json:
        print(json.dumps({"known_good_vllm_pins": list(allowlist)},
                         indent=2, sort_keys=True))
        return 0

    print("sndr upstream list — KNOWN_GOOD_VLLM_PINS")
    print("─" * 50)
    if not allowlist:
        print("  (allowlist empty — sndr.engines.vllm.detection.guards not importable)")
        return 0
    running = _running_pin()
    for i, pin in enumerate(allowlist):
        mark = " ← running" if pin == running else ""
        print(f"  {i+1:2d}. {pin}{mark}")
    print()
    print(f"  Total: {len(allowlist)} pins")
    return 0
