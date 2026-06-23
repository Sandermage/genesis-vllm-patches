# SPDX-License-Identifier: Apache-2.0
"""C1 (UNIFIED_CONFIG plan 2026-05-09) — `sndr doctor-system` extended host diagnostic.

Composes:
  - `sndr.deps.inspect_host()`   — full host inventory
  - `sndr.caveats.match_caveats()` — runtime caveats matcher
  - vllm pin allowlist check
  - Optional Y11 upstream policy match (when --config given)

Output: human prose by default; `--json` for CI/dashboards.
Exit code:
  0 → green
  1 → caveats fired (warning) OR upstream pin drift
  2 → blockers / errors

Distinct from the bridged `sndr doctor` which is the comprehensive
patches+lifecycle+models walker — `sndr doctor-system` focuses on
host/runtime concerns (the operator-facing "is my machine ready?"
question).
"""
from __future__ import annotations

import argparse
import json
from typing import Any

from . import _io


__all__ = ["add_argparser", "run_doctor_system"]


def add_argparser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "doctor-system",
        help="Extended host/runtime diagnostic (UNIFIED_CONFIG C1).",
        description=(
            "Snapshot host inventory + runtime caveats + vllm pin allowlist. "
            "Distinct from `sndr doctor` (which is the patches+lifecycle "
            "walker); `doctor-system` is the operator's first-look "
            "machine-readiness check."
        ),
    )
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON.")
    p.add_argument("--config", default=None,
                   help="Optional preset key — extends check with Y11 upstream "
                        "policy + Y3 artifacts presence verification.")
    p.add_argument("--strict", action="store_true",
                   help="Exit 1 on any warning (default 0 unless errors).")
    p.add_argument(
        "--logs", action="store_true",
        help=(
            "Additional host-side log forensics: last OOM-kill, "
            "NVRM Xid errors, restarting containers, recent dmesg "
            "warnings. Read-only — performs no actions, only reports. "
            "Audit P2-1 (2026-05-12)."
        ),
    )
    p.add_argument(
        "--logs-hours", type=int, default=24,
        help="Scan window for --logs in hours (default 24).",
    )
    # Etap 3.2 (audit 2026-05-12): container name prefix filter — by
    # default only Genesis-related containers count as fatal signals.
    p.add_argument(
        "--logs-container-prefix", action="append", default=None,
        metavar="PREFIX",
        help=(
            "Only consider restarting containers whose name starts with "
            "PREFIX (repeatable). Default: vllm, genesis, sndr. "
            "Use --logs-all-containers to disable the filter."
        ),
    )
    p.add_argument(
        "--logs-all-containers", action="store_true",
        help="Disable the container-name prefix filter (show every "
             "restart loop on the host as a fatal signal).",
    )
    # Etap 3.3 (audit 2026-05-12): strict window mode.
    p.add_argument(
        "--logs-strict-window", action="store_true",
        help="Drop OOM/Xid events with unknown timestamps instead of "
             "including them as 'possibly recent'.",
    )
    p.set_defaults(func=run_doctor_system)


def _build_facts(cfg=None) -> dict:
    """Compose the full facts dict that downstream matchers consume."""
    from sndr.deps.checkers import inspect_host
    from sndr.engines.vllm.detection.guards import KNOWN_GOOD_VLLM_PINS
    inv = inspect_host()
    facts = inv.to_dict()
    pin = facts.get("vllm", {}).get("version")
    facts["vllm_pin_in_allowlist"] = (
        pin in KNOWN_GOOD_VLLM_PINS if pin else None
    )
    # virtualization
    import os
    import shutil
    import subprocess
    if shutil.which("systemd-detect-virt"):
        try:
            r = subprocess.run(["systemd-detect-virt"],
                                capture_output=True, text=True, timeout=2)
            facts["virtualization"] = (
                r.stdout.strip() if r.returncode == 0 else "")
        except Exception:
            facts["virtualization"] = ""

    # WSL2 probes (audit closure post-MASTER_REMEDIATION_PLAN): operators
    # on Windows-WSL2 hit class-specific issues (NVRM compat, kernel
    # version, /dev/nvidia* visibility from Linux side). Detect early so
    # caveats matcher can warn rather than waiting for vllm boot crash.
    wsl_info: dict[str, object] = {"detected": False}
    proc_version = "/proc/version"
    try:
        if os.path.isfile(proc_version):
            with open(proc_version, "r") as f:
                kver = f.read()
            wsl_info["detected"] = (
                "microsoft" in kver.lower() or "wsl" in kver.lower()
            )
            if wsl_info["detected"]:
                # WSL2 specifically (vs WSL1) — gpus only work on 2
                wsl_info["wsl2"] = "wsl2" in kver.lower()
                wsl_info["kernel_release"] = kver.strip()[:200]
                wsl_info["nvidia_devices"] = [
                    p for p in ("/dev/nvidia0", "/dev/nvidiactl",
                                  "/dev/nvidia-uvm", "/dev/dxg")
                    if os.path.exists(p)
                ]
    except Exception:
        # Defensive: detection should never crash doctor-system
        pass
    facts["wsl"] = wsl_info
    if cfg is not None:
        facts["genesis_env"] = dict(getattr(cfg, "genesis_env", {}) or {})
        # Y11 upstream policy check
        if getattr(cfg, "upstream", None) is not None and pin:
            msg = cfg.upstream.check(pin)
            facts["upstream_violation"] = msg
        # Y3 artifacts presence
        arts = getattr(cfg, "artifacts", None)
        if arts is not None and arts.models:
            facts["artifacts_problems"] = [
                {"hf_id": m.hf_id, "problems": m.verify()}
                for m in arts.models
            ]
    return facts


def run_doctor_system(args: argparse.Namespace) -> int:
    cfg = None
    if args.config:
        from sndr.model_configs.registry import get
        cfg = get(args.config)
        if cfg is None:
            _io.warn(f"unknown preset key {args.config!r}")
            return 2

    facts = _build_facts(cfg)

    from sndr.caveats import match_caveats
    triggered = match_caveats(facts)

    has_error = any(c.severity == "error" for c in triggered)
    has_warning = any(c.severity == "warning" for c in triggered)
    upstream_violation = facts.get("upstream_violation")
    artifacts_problems = facts.get("artifacts_problems") or []
    n_artifact_failures = sum(
        1 for a in artifacts_problems if a["problems"]
    )

    # Audit P2-1 (2026-05-12): host log forensics (optional).
    log_forensics = None
    if getattr(args, "logs", False):
        from .doctor_logs import collect_log_forensics
        prefixes_arg = getattr(args, "logs_container_prefix", None)
        log_forensics = collect_log_forensics(
            window_hours=getattr(args, "logs_hours", 24),
            container_prefixes=tuple(prefixes_arg) if prefixes_arg else None,
            all_containers=getattr(args, "logs_all_containers", False),
            strict_window=getattr(args, "logs_strict_window", False),
        )
        facts["log_forensics"] = log_forensics.to_dict()
        # OOM / fatal Xid / restart loop are escalated as error.
        if log_forensics.has_fatal_signals:
            has_error = True

    if args.json:
        out = {
            "facts": {
                k: v for k, v in facts.items()
                if k in ("os", "python", "docker", "nvidia", "vllm",
                          "virtualization", "vllm_pin_in_allowlist",
                          "upstream_violation", "artifacts_problems",
                          "log_forensics", "wsl")
            },
            "caveats_triggered": [
                {"id": c.id, "severity": c.severity, "title": c.title,
                 "message": c.message, "docs_url": c.docs_url}
                for c in triggered
            ],
            "verdict": (
                "error" if has_error or upstream_violation or n_artifact_failures > 0
                else ("warning" if has_warning else "green")
            ),
        }
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print("sndr doctor-system — extended host diagnostic")
        print("─" * 60)
        os_ = facts.get("os", {})
        print(f"  OS:       {os_.get('system', '?')} {os_.get('release', '')}")
        if os_.get("distro"):
            print(f"            {os_['distro']}")
        print(f"  Python:   {facts.get('python', {}).get('version', '?')} "
              f"({facts.get('python', {}).get('implementation', '?')})")
        d = facts.get("docker", {})
        if d.get("installed"):
            print(f"  Docker:   {d.get('version', '?')} "
                  f"daemon={'running' if d.get('daemon_running') else 'STOPPED'} "
                  f"nvidia-runtime={'yes' if d.get('nvidia_runtime_present') else 'NO'}")
        else:
            print("  Docker:   not installed")
        n = facts.get("nvidia", {})
        if n.get("installed"):
            print(f"  NVIDIA:   driver {n.get('driver_version')} CUDA "
                  f"{n.get('cuda_version')} GPUs={n.get('n_gpus')}")
        else:
            print("  NVIDIA:   not detected")
        v = facts.get("vllm", {})
        if v.get("installed"):
            mark = "✓" if facts.get("vllm_pin_in_allowlist") else "✗"
            print(f"  vLLM:     {v.get('version')} (allowlist={mark})")
        else:
            print("  vLLM:     not installed in current Python")
        if facts.get("virtualization"):
            print(f"  Virt:     {facts['virtualization']}")
        wsl = facts.get("wsl") or {}
        if wsl.get("detected"):
            kind = "WSL2" if wsl.get("wsl2") else "WSL1"
            devs = wsl.get("nvidia_devices") or []
            print(f"  WSL:      {kind} detected — "
                  f"nvidia devices visible: {len(devs)}/4")

        if cfg is not None:
            print()
            print(f"  Preset:   {args.config}")
            if upstream_violation:
                print(f"    ✗ upstream: {upstream_violation}")
            else:
                print("    ✓ upstream policy OK")
            if artifacts_problems:
                print("    Artifacts:")
                for a in artifacts_problems:
                    if a["problems"]:
                        print(f"      ✗ {a['hf_id']}: {a['problems']}")
                    else:
                        print(f"      ✓ {a['hf_id']}: present")

        print()
        print(f"  Caveats:  {len(triggered)} triggered")
        for c in triggered:
            mark = {"info": "ℹ", "warning": "⚠", "error": "✗"}.get(c.severity, "·")
            print(f"    {mark} [{c.severity.upper()}] {c.id}")
            print(f"        {c.title}")

        # Audit P2-1 (2026-05-12): log forensics output
        if log_forensics is not None:
            from .doctor_logs import summarize_for_text
            print()
            for line in summarize_for_text(log_forensics):
                print(line)

        # Verdict line
        print()
        if has_error or n_artifact_failures > 0 or upstream_violation:
            print("  → VERDICT: NOT READY (errors / blockers present)")
        elif has_warning:
            print("  → VERDICT: WARNING (operational with caveats)")
        else:
            print("  → VERDICT: GREEN")

    if has_error or upstream_violation or n_artifact_failures > 0:
        return 2
    if args.strict and triggered:
        return 1
    return 0
