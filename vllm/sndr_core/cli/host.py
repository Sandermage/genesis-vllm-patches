# SPDX-License-Identifier: Apache-2.0
"""`sndr host` — host profile manager (P3-B audit closure 2026-05-12).

Manages `~/.sndr/host.yaml` (or `~/.genesis/host.yaml` legacy) — the
single source of truth for host-specific paths the launcher resolves
into Docker mount strings. Operators run:

  sndr host detect   - probe current host (GPUs, dirs, runtimes)
  sndr host init     - write a starter host.yaml from detection
  sndr host doctor   - validate the current host.yaml (paths exist, etc.)
  sndr host edit     - open host.yaml in $EDITOR / print path
  sndr host show     - print current host.yaml content

The launcher reads host.yaml via `model_configs.host.load_host_config()`;
this CLI is the operator-facing writer/inspector of the same file.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from . import _io

__all__ = [
    "add_argparser",
    "run_detect",
    "run_init",
    "run_doctor",
    "run_edit",
    "run_show",
]


def add_argparser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "host",
        help="Host profile manager — manages `~/.sndr/host.yaml`.",
        description=(
            "Probe host (GPUs, paths, runtimes), write a starter "
            "`~/.sndr/host.yaml`, validate it, or open in $EDITOR. "
            "host.yaml feeds the launcher's symbolic-mount resolution."
        ),
    )
    sub = p.add_subparsers(dest="host_cmd", required=True)

    p_detect = sub.add_parser("detect",
                                 help="Probe host without writing anything.")
    p_detect.add_argument("--json", action="store_true")
    p_detect.set_defaults(func=run_detect)

    p_init = sub.add_parser("init",
                               help="Write a starter host.yaml from detection.")
    p_init.add_argument("--force", action="store_true",
                          help="Overwrite existing host.yaml.")
    p_init.add_argument("--path", default=None,
                          help="Custom output path (default: ~/.sndr/host.yaml).")
    p_init.set_defaults(func=run_init)

    p_doctor = sub.add_parser("doctor",
                                 help="Validate current host.yaml.")
    p_doctor.add_argument("--json", action="store_true")
    p_doctor.set_defaults(func=run_doctor)

    p_edit = sub.add_parser("edit",
                                help="Open host.yaml in $EDITOR or print path.")
    p_edit.add_argument("--print", action="store_true",
                          help="Just print the resolved path; don't open editor.")
    p_edit.set_defaults(func=run_edit)

    p_show = sub.add_parser("show",
                                help="Print current host.yaml content.")
    p_show.set_defaults(func=run_show)


# ─── Resolve path
def _host_yaml_path() -> Path:
    """Return the canonical host.yaml path. Honors `$SNDR_HOST_YAML` env
    override; otherwise `~/.sndr/host.yaml`."""
    override = os.environ.get("SNDR_HOST_YAML")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".sndr" / "host.yaml"


# ─── Detect
def _probe_host() -> dict:
    """Return a structured host profile: GPUs, paths, runtimes."""
    profile: dict[str, Any] = {"detected_at": "", "paths": {}, "gpus": [],
                                 "runtimes": {}}
    profile["detected_at"] = subprocess.run(
        ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
        capture_output=True, text=True,
    ).stdout.strip()

    # Paths — try sensible defaults
    home = Path.home()
    candidates = {
        "models_dir": [home / "models", Path("/models"), Path("/nfs/models")],
        "hf_cache": [home / ".cache" / "huggingface"],
        "triton_cache": [home / ".triton" / "cache"],
        "compile_cache": [home / ".cache" / "vllm" / "torch_compile_cache"],
        "genesis_src": [],  # operator must set
        "plugin_src": [],   # operator must set
    }
    for key, paths in candidates.items():
        chosen = None
        for c in paths:
            if c.is_dir():
                chosen = str(c)
                break
        profile["paths"][key] = chosen

    # GPUs via nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                for line in r.stdout.strip().splitlines():
                    fields = [s.strip() for s in line.split(",")]
                    if len(fields) >= 4:
                        profile["gpus"].append({
                            "name": fields[0],
                            "vram_mib": int(fields[1]) if fields[1].isdigit() else None,
                            "driver": fields[2],
                            "compute_cap": fields[3],
                        })
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # Runtimes
    profile["runtimes"] = {
        "docker": shutil.which("docker") is not None,
        "podman": shutil.which("podman") is not None,
        "kubectl": shutil.which("kubectl") is not None,
        "systemctl": shutil.which("systemctl") is not None,
    }
    return profile


def run_detect(args: argparse.Namespace) -> int:
    p = _probe_host()
    if args.json:
        print(json.dumps(p, indent=2))
        return 0
    _io.banner("sndr host detect",
                 f"{len(p['gpus'])} GPU(s) · "
                 f"{sum(1 for v in p['runtimes'].values() if v)} runtime(s)")
    _io.info(f"  detected_at: {p['detected_at']}")
    _io.info("  paths:")
    for k, v in p["paths"].items():
        mark = "✓" if v else "?"
        _io.info(f"    {mark} {k}: {v or '(not set)'}")
    _io.info("  gpus:")
    if not p["gpus"]:
        _io.warn("    (none detected — nvidia-smi missing or no GPUs)")
    for i, g in enumerate(p["gpus"]):
        _io.info(
            f"    [{i}] {g['name']:<24} VRAM={g.get('vram_mib')} MiB  "
            f"driver={g['driver']}  SM={g['compute_cap']}"
        )
    _io.info("  runtimes:")
    for k, v in p["runtimes"].items():
        mark = "✓" if v else "·"
        _io.info(f"    {mark} {k}: {'available' if v else 'not on PATH'}")
    return 0


# ─── Init
def run_init(args: argparse.Namespace) -> int:
    out_path = Path(args.path).expanduser() if args.path else _host_yaml_path()
    if out_path.exists() and not args.force:
        _io.error(
            f"host.yaml already exists at {out_path} — use --force to overwrite."
        )
        return 1
    profile = _probe_host()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = _render_host_yaml(profile)
    out_path.write_text(body)
    _io.success(f"wrote {out_path}")
    _io.info("  next steps:")
    _io.info("    1. fill in genesis_src + plugin_src paths (operator-specific)")
    _io.info("    2. validate: sndr host doctor")
    _io.info("    3. launch:   sndr launch <preset>")
    return 0


def _render_host_yaml(profile: dict) -> str:
    paths = profile.get("paths", {})
    gpus = profile.get("gpus", [])
    runtimes = profile.get("runtimes", {})
    n_gpu = len(gpus)
    lines = [
        "# host.yaml — operator host profile for sndr (Genesis) launcher.",
        f"# Generated from `sndr host detect` on {profile.get('detected_at')}.",
        "",
        "paths:",
    ]
    for key in ("models_dir", "hf_cache", "triton_cache",
                  "compile_cache", "genesis_src", "plugin_src"):
        v = paths.get(key)
        if v:
            lines.append(f"  {key}: {v}")
        else:
            lines.append(f"  # {key}: /path/to/{key.replace('_', '-')}   "
                           f"# REQUIRED — operator must fill")
    lines.extend([
        "",
        f"# Detected {n_gpu} GPU(s):",
    ])
    for i, g in enumerate(gpus):
        lines.append(
            f"#   [{i}] {g['name']} VRAM={g.get('vram_mib')} MiB "
            f"driver={g['driver']} SM={g['compute_cap']}"
        )
    lines.extend([
        "",
        "# Runtimes present (informational; launcher chooses based on preset):",
    ])
    for k, v in runtimes.items():
        lines.append(f"#   {k}: {'yes' if v else 'no'}")
    lines.append("")
    return "\n".join(lines)


# ─── Doctor
def run_doctor(args: argparse.Namespace) -> int:
    p = _host_yaml_path()
    findings: list[dict] = []
    if not p.exists():
        findings.append({
            "severity": "FAIL", "name": "host_yaml_present",
            "message": f"{p} does not exist — run `sndr host init`",
        })
    else:
        try:
            import yaml
            data = yaml.safe_load(p.read_text()) or {}
        except Exception as e:
            findings.append({
                "severity": "FAIL", "name": "host_yaml_parseable",
                "message": f"YAML parse failed: {e}",
            })
            data = None
        else:
            findings.append({
                "severity": "PASS", "name": "host_yaml_present",
                "message": str(p),
            })
            paths = data.get("paths", {}) or {}
            for key in ("models_dir", "genesis_src", "plugin_src"):
                v = paths.get(key)
                if not v:
                    findings.append({
                        "severity": "FAIL", "name": f"paths.{key}",
                        "message": "missing — launcher cannot resolve mounts",
                    })
                elif not Path(v).expanduser().is_dir():
                    findings.append({
                        "severity": "WARN", "name": f"paths.{key}",
                        "message": f"path does not exist: {v}",
                    })
                else:
                    findings.append({
                        "severity": "PASS", "name": f"paths.{key}",
                        "message": v,
                    })
            for key in ("hf_cache", "triton_cache", "compile_cache"):
                v = paths.get(key)
                if v and not Path(v).expanduser().is_dir():
                    findings.append({
                        "severity": "WARN", "name": f"paths.{key}",
                        "message": f"optional path missing: {v}",
                    })

    fails = sum(1 for f in findings if f["severity"] == "FAIL")
    warns = sum(1 for f in findings if f["severity"] == "WARN")
    if args.json:
        print(json.dumps({
            "host_yaml": str(p), "findings": findings,
            "fails": fails, "warns": warns,
        }, indent=2))
        return 1 if fails else (2 if warns else 0)
    _io.banner("sndr host doctor",
                 f"{len(findings)} checks · {fails} fail · {warns} warn")
    for f in findings:
        mark = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}[f["severity"]]
        line = f"  {mark} [{f['severity']:<4}] {f['name']:<24} {f['message']}"
        if f["severity"] == "FAIL":
            _io.error(line)
        elif f["severity"] == "WARN":
            _io.warn(line)
        else:
            _io.info(line)
    if fails:
        return 1
    if warns:
        return 2
    return 0


# ─── Edit
def run_edit(args: argparse.Namespace) -> int:
    p = _host_yaml_path()
    if args.print:
        print(p)
        return 0
    if not p.exists():
        _io.error(f"{p} does not exist — run `sndr host init` first.")
        return 1
    editor = os.environ.get("EDITOR", "vi")
    return subprocess.call([editor, str(p)])


# ─── Show
def run_show(args: argparse.Namespace) -> int:
    p = _host_yaml_path()
    if not p.exists():
        _io.error(f"{p} does not exist — run `sndr host init`.")
        return 1
    print(p.read_text())
    return 0
