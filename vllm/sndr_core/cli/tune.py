# SPDX-License-Identifier: Apache-2.0
"""C14 (UNIFIED_CONFIG plan 2026-05-09) — `sndr tune` GPU tuning CLI.

Wraps `nvidia-smi -pl/-pm/-lgc/-lmc` per the preset's Y8 GpuTuningConfig
declarations. Strict safety:
  - Default --dry-run; --yes required for actual nvidia-smi calls
  - power_limit/clocks REFUSED unless cfg.gpu_tuning.unsafe_apply=True
  - Sanity range checks (refuse < 50W power_limit, etc.)

Subcommands:
  sndr tune plan <key>      — print what would be applied (no action)
  sndr tune apply <key>     — apply tuning (--yes required)
  sndr tune revert <key>    — restore default values (best-effort)
  sndr tune sweep <key>     — run a power-limit sweep over [low..high]
  sndr tune report <key>    — current nvidia-smi state vs declared
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from typing import Any, Optional

from . import _io


__all__ = ["add_argparser", "run_plan", "run_apply", "run_revert",
           "run_sweep", "run_report"]


def add_argparser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "tune",
        help="GPU tuning wrapper around Y8 GpuTuningConfig (UNIFIED_CONFIG C14).",
        description=(
            "Apply GPU power/clock tuning from a preset's Y8 gpu_tuning "
            "block. Default --dry-run; --yes to actually run nvidia-smi."
        ),
    )
    sub = p.add_subparsers(dest="tune_cmd", required=True)

    for cmd, helper, fn in (
        ("plan", "Print the planned nvidia-smi commands without running them",
         run_plan),
        ("apply", "Apply Y8 gpu_tuning settings via nvidia-smi", run_apply),
        ("revert", "Best-effort restore to default GPU clocks/power", run_revert),
        ("report", "Print current nvidia-smi state vs Y8 declared", run_report),
    ):
        sp = sub.add_parser(cmd, help=helper)
        sp.add_argument("config", help="model_config preset key")
        sp.add_argument("--yes", action="store_true",
                          help="Actually call nvidia-smi (default: dry-run).")
        sp.add_argument("--gpu-id", type=int, default=None,
                          help="Restrict to a single GPU index "
                               "(default: all GPUs).")
        sp.set_defaults(func=fn)

    sweep = sub.add_parser(
        "sweep",
        help="Run a power-limit sweep [low..high in step] (operator must "
             "supply --bench-cmd).",
    )
    sweep.add_argument("config", help="model_config preset key")
    sweep.add_argument("--low", type=int, required=True, help="low watts")
    sweep.add_argument("--high", type=int, required=True, help="high watts")
    sweep.add_argument("--step", type=int, default=20, help="step watts")
    sweep.add_argument("--bench-cmd", required=True,
                         help="bench command per arm")
    sweep.add_argument("--yes", action="store_true",
                         help="Actually run; default dry-run")
    sweep.set_defaults(func=run_sweep)


def _resolve(key: str):
    from vllm.sndr_core.model_configs.registry import get
    cfg = get(key)
    if cfg is None:
        _io.warn(f"unknown preset key {key!r}")
        return None
    if cfg.gpu_tuning is None:
        _io.warn(f"preset {key!r} has no Y8 gpu_tuning block")
        return None
    return cfg


def _planned_commands(cfg, gpu_id: Optional[int] = None) -> list[list[str]]:
    """Compose the list of nvidia-smi commands the apply path would run."""
    g = cfg.gpu_tuning
    cmds: list[list[str]] = []
    gpu_arg: list[str] = []
    if gpu_id is not None:
        gpu_arg = ["-i", str(gpu_id)]

    # Safe knobs first
    if g.persistence_mode is True:
        cmds.append(["nvidia-smi", *gpu_arg, "-pm", "1"])
    elif g.persistence_mode is False:
        cmds.append(["nvidia-smi", *gpu_arg, "-pm", "0"])

    # Unsafe knobs gated behind unsafe_apply=True
    if g.unsafe_apply:
        if g.power_limit_watts is not None:
            cmds.append(["nvidia-smi", *gpu_arg, "-pl", str(g.power_limit_watts)])
        if g.clocks_gfx_mhz is not None:
            cmds.append(["nvidia-smi", *gpu_arg, "-lgc",
                          f"0,{g.clocks_gfx_mhz}"])
        if g.clocks_mem_mhz is not None:
            cmds.append(["nvidia-smi", *gpu_arg, "-lmc",
                          f"0,{g.clocks_mem_mhz}"])
    return cmds


def _execute_one(cmd: list[str], dry_run: bool) -> int:
    if dry_run:
        _io.info(f"[dry-run] would: {' '.join(cmd)}")
        return 0
    if shutil.which("nvidia-smi") is None:
        _io.error("nvidia-smi not on PATH")
        return 1
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if r.stdout:
        print(r.stdout.rstrip())
    if r.stderr:
        print(r.stderr.rstrip())
    return r.returncode


# ─── plan

def run_plan(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    cmds = _planned_commands(cfg, gpu_id=args.gpu_id)
    print(f"sndr tune plan '{args.config}'")
    print("─" * 60)
    if not cmds:
        print("  (no commands planned — gpu_tuning declares only ulimits/THP)")
        return 0
    for c in cmds:
        print(f"  $ {' '.join(c)}")
    if cfg.gpu_tuning.unsafe_apply:
        print()
        _io.warn("unsafe_apply=True — power_limit / clocks will be applied "
                  "with --yes")
    if cfg.gpu_tuning.notes:
        print()
        print(f"  notes: {cfg.gpu_tuning.notes}")
    return 0


# ─── apply

def run_apply(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    cmds = _planned_commands(cfg, gpu_id=args.gpu_id)
    if not cmds:
        _io.info("no commands to apply")
        return 0
    dry_run = not args.yes
    rc_max = 0
    for c in cmds:
        rc = _execute_one(c, dry_run=dry_run)
        rc_max = max(rc_max, rc)
        if rc != 0 and not dry_run:
            _io.error(f"abort — command failed: {' '.join(c)}")
            return rc_max
    return rc_max


# ─── revert

def run_revert(args: argparse.Namespace) -> int:
    """Best-effort: reset power-limit to default, unlock clocks."""
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    dry_run = not args.yes
    gpu_arg = ["-i", str(args.gpu_id)] if args.gpu_id is not None else []
    cmds = [
        ["nvidia-smi", *gpu_arg, "-pl", "0"],   # 0 = reset to factory default
        ["nvidia-smi", *gpu_arg, "-rgc"],        # reset GPU clocks
        ["nvidia-smi", *gpu_arg, "-rmc"],        # reset memory clocks
    ]
    rc_max = 0
    for c in cmds:
        rc = _execute_one(c, dry_run=dry_run)
        rc_max = max(rc_max, rc)
    return rc_max


# ─── sweep

def run_sweep(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    if not cfg.gpu_tuning.unsafe_apply:
        _io.error(f"preset {args.config!r} gpu_tuning.unsafe_apply=False; "
                   f"power-limit sweeps require unsafe_apply=True")
        return 2
    if args.low < 50 or args.high < args.low:
        _io.error(f"invalid range: low={args.low} high={args.high}")
        return 2
    dry_run = not args.yes
    gpu_arg = ["-i", str(args.gpu_id)] if args.gpu_id is not None else []
    print(f"sndr tune sweep '{args.config}'")
    print(f"  range: {args.low}W..{args.high}W step={args.step}W")
    print(f"  bench: {args.bench_cmd}")
    print()
    for w in range(args.low, args.high + 1, args.step):
        print(f"━━━ ARM @ {w}W ━━━")
        rc = _execute_one(["nvidia-smi", *gpu_arg, "-pl", str(w)],
                            dry_run=dry_run)
        if rc != 0:
            _io.error(f"failed to set power-limit {w}W; abort sweep")
            return rc
        if dry_run:
            _io.info(f"[dry-run] would run: {args.bench_cmd}")
        else:
            r = subprocess.run(["/bin/bash", "-c", args.bench_cmd],
                                timeout=600)
            if r.returncode != 0:
                _io.warn(f"bench at {w}W returned rc={r.returncode}")
        print()
    return 0


# ─── report

def run_report(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    if shutil.which("nvidia-smi") is None:
        _io.error("nvidia-smi not on PATH")
        return 1
    gpu_arg = ["-i", str(args.gpu_id)] if args.gpu_id is not None else []
    r = subprocess.run([
        "nvidia-smi", *gpu_arg,
        "--query-gpu=index,name,power.draw,power.limit,clocks.current.graphics,"
        "clocks.current.memory,persistence_mode",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, timeout=10)
    print(f"sndr tune report '{args.config}'")
    print("─" * 70)
    print("  Y8 declared:")
    g = cfg.gpu_tuning
    print(f"    persistence_mode: {g.persistence_mode}")
    print(f"    power_limit_W:    {g.power_limit_watts} (unsafe={g.unsafe_apply})")
    print(f"    clocks_gfx_MHz:   {g.clocks_gfx_mhz}")
    print(f"    clocks_mem_MHz:   {g.clocks_mem_mhz}")
    print()
    print("  Live nvidia-smi:")
    if r.returncode != 0:
        print(f"    (nvidia-smi failed: {r.stderr})")
        return 1
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 7:
            print(f"    GPU {parts[0]} {parts[1]}:")
            print(f"      power: {parts[2]}W / limit {parts[3]}W")
            print(f"      clocks: gfx {parts[4]}MHz mem {parts[5]}MHz")
            print(f"      persistence: {parts[6]}")
    return 0
