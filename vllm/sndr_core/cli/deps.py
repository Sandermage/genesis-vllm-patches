# SPDX-License-Identifier: Apache-2.0
"""C2 (UNIFIED_CONFIG plan 2026-05-09) — `sndr deps` subcommand trio.

Wires the Tier 2 `vllm.sndr_core.deps` package (P3) to the user-facing
CLI. No install side effects in this module — `check` and `plan` are
pure inspection; `--write-report` writes JSON+MD to a destination dir
but never modifies the host's runtime stack.

Subcommands:

  sndr deps check [--config <key>] [--json] [--write-report]
      Inspect host inventory; if --config is set, derive the plan and
      print readiness verdict. Exit 0 if ready (or no config given);
      exit 1 if blockers exist.

  sndr deps plan --config <key> [--json] [--write-report]
      Show the plan items (blockers + warnings) for the given config
      against the current host inventory. Exit 0 always (this is a
      view, not a gate); use --strict to flip to exit 1 on blockers.

(Future: `sndr deps install` — runs the plan; gated behind separate
installer module with --yes / --scope / dry-run.)
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Optional

from . import _io


__all__ = ["add_argparser", "run_check", "run_plan"]


def add_argparser(subparsers: Any) -> None:
    """Register the `sndr deps` subcommand tree."""
    p = subparsers.add_parser(
        "deps",
        help="Host dependency inventory + per-config plan (UNIFIED_CONFIG C2).",
        description=(
            "Inspect Docker/NVIDIA/Python/vllm on the host and, given "
            "a model config, derive what changes the host needs. Pure "
            "inspection — never installs anything."
        ),
    )
    sub = p.add_subparsers(dest="deps_cmd", required=True)

    p_check = sub.add_parser(
        "check",
        help="Inspect host inventory; optionally validate against a config.",
    )
    p_check.add_argument(
        "--config", default=None,
        help="Validate inventory against this model_config key.",
    )
    p_check.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON instead of human prose.",
    )
    p_check.add_argument(
        "--write-report", action="store_true",
        help="Also write JSON+MD reports to ~/.sndr/reports/.",
    )
    p_check.add_argument(
        "--report-dir", default=None,
        help="Override report destination directory.",
    )
    p_check.set_defaults(func=run_check)

    p_plan = sub.add_parser(
        "plan",
        help="Show what host changes the given config requires.",
    )
    p_plan.add_argument(
        "--config", required=True,
        help="model_config key to plan for.",
    )
    p_plan.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON instead of human prose.",
    )
    p_plan.add_argument(
        "--write-report", action="store_true",
        help="Also write JSON+MD reports to ~/.sndr/reports/.",
    )
    p_plan.add_argument(
        "--report-dir", default=None,
        help="Override report destination directory.",
    )
    p_plan.add_argument(
        "--strict", action="store_true",
        help="Exit 1 on any blocker (default: always exit 0).",
    )
    p_plan.set_defaults(func=run_plan)


def _resolve_config(key: str):
    """Resolve a config key via the registry; print a friendly error
    and return None if the key isn't found."""
    from vllm.sndr_core.model_configs.registry import get
    cfg = get(key)
    if cfg is None:
        _io.warn(f"unknown config key {key!r}")
        try:
            from vllm.sndr_core.model_configs.registry import list_keys
            keys = list_keys()
            _io.info(f"available keys: {', '.join(sorted(keys))}")
        except Exception:
            pass
    return cfg


def _maybe_report(*, report: bool, plan_obj, inventory, dest_str: Optional[str]):
    """Optional report writers; called from both `check` and `plan`."""
    if not report:
        return
    from pathlib import Path
    from vllm.sndr_core.deps import report_inventory, report_plan
    dest = Path(dest_str).expanduser() if dest_str else None
    inv_json, inv_md = report_inventory(inventory, dest=dest)
    _io.info(f"wrote inventory: {inv_md}")
    if plan_obj is not None:
        pj, pm = report_plan(plan_obj, dest=dest)
        _io.info(f"wrote plan:      {pm}")


def run_check(args: argparse.Namespace) -> int:
    from vllm.sndr_core.deps import inspect_host, plan_changes
    inv = inspect_host()
    plan = None
    cfg = None
    if args.config:
        cfg = _resolve_config(args.config)
        if cfg is None:
            return 2
        plan = plan_changes(cfg, inv)

    if args.json:
        out: dict = {"inventory": inv.to_dict()}
        if plan is not None:
            out["plan"] = plan.to_dict()
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        _print_inventory_summary(inv)
        if plan is not None:
            print()
            _print_plan_summary(plan)

    _maybe_report(report=args.write_report, plan_obj=plan,
                  inventory=inv, dest_str=args.report_dir)

    if plan is not None and not plan.is_ready():
        return 1
    return 0


def run_plan(args: argparse.Namespace) -> int:
    from vllm.sndr_core.deps import inspect_host, plan_changes
    cfg = _resolve_config(args.config)
    if cfg is None:
        return 2

    inv = inspect_host()
    plan = plan_changes(cfg, inv)

    if args.json:
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    else:
        _print_plan_summary(plan)

    _maybe_report(report=args.write_report, plan_obj=plan,
                  inventory=inv, dest_str=args.report_dir)

    if args.strict and not plan.is_ready():
        return 1
    return 0


# ─── Pretty printing ───────────────────────────────────────────────────


def _print_inventory_summary(inv) -> None:
    print("Host inventory")
    print("─" * 40)
    print(f"  OS:      {inv.os.system} {inv.os.release}")
    if inv.os.distro:
        print(f"           {inv.os.distro}")
    print(f"  Python:  {inv.python.version} ({inv.python.implementation})")
    print(f"           pip {inv.python.pip_version or '_missing_'}")
    if inv.docker.installed:
        daemon = "running" if inv.docker.daemon_running else "STOPPED"
        nvr = "yes" if inv.docker.nvidia_runtime_present else "NO"
        print(f"  Docker:  {inv.docker.version or '?'}  daemon={daemon}  "
              f"nvidia-runtime={nvr}")
    else:
        print("  Docker:  not installed")
    if inv.nvidia.installed:
        print(f"  NVIDIA:  driver {inv.nvidia.driver_version}  "
              f"CUDA {inv.nvidia.cuda_version}  "
              f"GPUs={inv.nvidia.n_gpus}")
        for i, name in enumerate(inv.nvidia.gpu_names):
            mib = (inv.nvidia.gpu_total_vram_mib[i]
                   if i < len(inv.nvidia.gpu_total_vram_mib) else 0)
            print(f"           [{i}] {name} ({mib} MiB)")
    else:
        print("  NVIDIA:  not detected")
    if inv.vllm.installed:
        print(f"  vLLM:    {inv.vllm.version}")
    else:
        print("  vLLM:    not installed in current Python")


def _print_plan_summary(plan) -> None:
    blockers = plan.blockers()
    warnings = plan.warnings()
    verdict = "READY" if plan.is_ready() else "NOT READY"
    print(f"Plan for config '{plan.config_key}'  →  {verdict}")
    print("─" * 40)
    print(f"  blockers: {len(blockers)}")
    print(f"  warnings: {len(warnings)}")
    if blockers:
        print()
        print("Blockers:")
        for item in blockers:
            print(f"  ✗ [{item.scope}] {item.target}")
            print(f"      {item.reason}")
            if item.suggested_command:
                print(f"      → {item.suggested_command}")
    if warnings:
        print()
        print("Warnings:")
        for item in warnings:
            print(f"  ! [{item.scope}] {item.target}")
            print(f"      {item.reason}")
            if item.suggested_command:
                print(f"      → {item.suggested_command}")
    if plan.notes:
        print()
        for n in plan.notes:
            print(f"  · {n}")
