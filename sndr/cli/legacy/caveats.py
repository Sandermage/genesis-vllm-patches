# SPDX-License-Identifier: Apache-2.0
"""C22 (UNIFIED_CONFIG plan 2026-05-09) — `sndr caveats` subcommand.

Surfaces the Y13 runtime caveats registry. Three subcommands:

  sndr caveats list
      Print every known caveat with severity + title.

  sndr caveats check
      Snapshot host inventory, match against caveats, print triggered.
      Exit 1 if any 'error'-severity caveat fires; 0 otherwise (info
      and warning are non-fatal).

  sndr caveats explain <id>
      Print full caveat detail (message, docs URL).
"""
from __future__ import annotations

import argparse
import json
from typing import Any

from . import _io


__all__ = ["add_argparser", "run_list", "run_check", "run_explain"]


def add_argparser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "caveats",
        help="Runtime caveats registry — known host-condition issues (UNIFIED_CONFIG C22).",
        description=(
            "Inspect Genesis's known runtime caveats: list every "
            "registered caveat, snapshot the host and match, or "
            "explain a single caveat's detail."
        ),
    )
    sub = p.add_subparsers(dest="caveats_cmd", required=True)

    p_list = sub.add_parser("list", help="List all known caveat ids + titles.")
    p_list.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON.")
    p_list.set_defaults(func=run_list)

    p_check = sub.add_parser("check",
                              help="Match caveats against current host.")
    p_check.add_argument("--json", action="store_true",
                          help="Emit machine-readable JSON.")
    p_check.add_argument("--strict", action="store_true",
                          help="Exit 1 on any caveat fire (default 0 unless 'error' severity).")
    p_check.set_defaults(func=run_check)

    p_explain = sub.add_parser("explain",
                                help="Print full detail of one caveat.")
    p_explain.add_argument("caveat_id", help="caveat id (e.g. 'proxmox_lxc_kernel_617')")
    p_explain.add_argument("--json", action="store_true",
                            help="Emit machine-readable JSON.")
    p_explain.set_defaults(func=run_explain)


def _facts_from_inventory() -> dict:
    """Build the `facts` dict that caveat match_fns expect."""
    from sndr.deps.checkers import inspect_host
    from sndr.engines.vllm.detection.guards import KNOWN_GOOD_VLLM_PINS
    inv = inspect_host()
    facts = inv.to_dict()
    # Augment with vllm pin allowlist membership
    pin = facts.get("vllm", {}).get("version")
    facts["vllm_pin_in_allowlist"] = (
        pin in KNOWN_GOOD_VLLM_PINS if pin else None
    )
    # virtualization (best-effort detect)
    import shutil
    import subprocess
    if shutil.which("systemd-detect-virt"):
        try:
            r = subprocess.run(["systemd-detect-virt"],
                                capture_output=True, text=True, timeout=2)
            facts["virtualization"] = r.stdout.strip() if r.returncode == 0 else ""
        except Exception:
            facts["virtualization"] = ""
    return facts


# ─── list

def run_list(args: argparse.Namespace) -> int:
    from sndr.caveats import KNOWN_CAVEATS
    if args.json:
        out = [
            {"id": c.id, "severity": c.severity, "title": c.title,
             "docs_url": c.docs_url}
            for c in KNOWN_CAVEATS
        ]
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0
    print("sndr caveats — KNOWN_CAVEATS")
    print("─" * 60)
    if not KNOWN_CAVEATS:
        print("  (registry empty)")
        return 0
    for c in KNOWN_CAVEATS:
        sev_pad = c.severity.upper().ljust(7)
        print(f"  [{sev_pad}]  {c.id}")
        print(f"             {c.title}")
    print()
    print(f"  Total: {len(KNOWN_CAVEATS)} caveats")
    return 0


# ─── check

def run_check(args: argparse.Namespace) -> int:
    from sndr.caveats import match_caveats
    facts = _facts_from_inventory()
    triggered = match_caveats(facts)

    if args.json:
        out = {
            "facts": {
                k: facts[k] for k in (
                    "os", "vllm", "docker", "nvidia",
                    "virtualization", "vllm_pin_in_allowlist",
                ) if k in facts
            },
            "triggered": [
                {"id": c.id, "severity": c.severity, "title": c.title,
                 "message": c.message, "docs_url": c.docs_url}
                for c in triggered
            ],
        }
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print("sndr caveats check")
        print("─" * 60)
        print(f"  Host: {facts.get('os', {}).get('system', '?')}")
        print(f"  Triggered: {len(triggered)}")
        if not triggered:
            print()
            print("  ✓ No caveats apply.")
            return 0
        print()
        for c in triggered:
            mark = {"info": "ℹ", "warning": "⚠", "error": "✗"}.get(c.severity, "·")
            print(f"  {mark} [{c.severity.upper()}] {c.title}")
            print(f"     {c.id}")
            for line in c.message.split(". "):
                print(f"     {line.strip().rstrip('.')}.")
            if c.docs_url:
                print(f"     docs: {c.docs_url}")
            print()

    has_error = any(c.severity == "error" for c in triggered)
    if has_error:
        return 1
    if args.strict and triggered:
        return 1
    return 0


# ─── explain

def run_explain(args: argparse.Namespace) -> int:
    from sndr.caveats import get_caveat, list_caveat_ids
    c = get_caveat(args.caveat_id)
    if c is None:
        _io.warn(f"unknown caveat id {args.caveat_id!r}")
        _io.info(f"available: {', '.join(list_caveat_ids())}")
        return 2

    if args.json:
        out = {
            "id": c.id, "severity": c.severity, "title": c.title,
            "message": c.message, "docs_url": c.docs_url,
        }
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    print(f"sndr caveats explain '{c.id}'")
    print("─" * 60)
    print(f"  severity:  {c.severity}")
    print(f"  title:     {c.title}")
    print(f"  message:   {c.message}")
    if c.docs_url:
        print(f"  docs:      {c.docs_url}")
    return 0
