# SPDX-License-Identifier: Apache-2.0
"""JSON + Markdown report writers for inventory + plan.

Outputs go to `~/.sndr/reports/` by default; callers can pass an
explicit destination directory. The report bundle is what `sndr report
bundle --scope deps` will share with operators (P4 in the plan).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .checkers import HostInventory
from .planners import DepsPlan


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_reports_dir() -> Path:
    """Returns `$SNDR_HOME/reports/` or `~/.sndr/reports/` by default."""
    home = os.environ.get("SNDR_HOME")
    if home:
        return Path(home) / "reports"
    return Path.home() / ".sndr" / "reports"


# ─── Inventory ─────────────────────────────────────────────────────────


def report_inventory(
    inventory: HostInventory,
    dest: Optional[Path] = None,
    *,
    name: str = "inventory",
) -> tuple[Path, Path]:
    """Write inventory JSON + MD; return (json_path, md_path)."""
    dest = dest or _default_reports_dir()
    dest.mkdir(parents=True, exist_ok=True)
    ts = _now_iso().replace(":", "-")
    json_path = dest / f"{name}-{ts}.json"
    md_path = dest / f"{name}-{ts}.md"

    body = {
        "kind": "host_inventory",
        "generated_at": _now_iso(),
        "inventory": inventory.to_dict(),
    }
    json_path.write_text(json.dumps(body, indent=2, sort_keys=True))
    md_path.write_text(_render_inventory_md(inventory))
    return json_path, md_path


def _render_inventory_md(inv: HostInventory) -> str:
    lines = [
        "# SNDR Host Inventory",
        "",
        f"_Generated: {_now_iso()}_",
        "",
        "## OS",
        f"- system:  `{inv.os.system}` ({inv.os.release})",
        f"- distro:  {inv.os.distro or '_unknown_'}",
        f"- arch:    `{inv.os.arch}`",
        "",
        "## Python",
        f"- binary:  `{inv.python.binary_path}`",
        f"- version: `{inv.python.version}` ({inv.python.implementation})",
        f"- venv:    {'yes' if inv.python.venv_active else 'no'}",
        f"- pip:     {inv.python.pip_version or '_missing_'}",
        "",
        "## Docker",
    ]
    if inv.docker.installed:
        lines += [
            f"- binary:  `{inv.docker.binary_path}`",
            f"- version: `{inv.docker.version or '?'}`",
            f"- daemon:  {'running' if inv.docker.daemon_running else 'stopped'}",
            f"- server:  `{inv.docker.server_version or '_n/a_'}`",
            f"- nvidia runtime: {'present' if inv.docker.nvidia_runtime_present else 'MISSING'}",
        ]
    else:
        lines += ["- _not installed_"]
    if inv.docker.notes:
        lines += [f"- notes: {inv.docker.notes}"]

    lines += ["", "## NVIDIA"]
    if inv.nvidia.installed:
        lines += [
            f"- driver:  `{inv.nvidia.driver_version or '?'}`",
            f"- CUDA:    `{inv.nvidia.cuda_version or '?'}`",
            f"- GPUs:    {inv.nvidia.n_gpus}",
        ]
        for i, name in enumerate(inv.nvidia.gpu_names):
            mib = (inv.nvidia.gpu_total_vram_mib[i]
                   if i < len(inv.nvidia.gpu_total_vram_mib) else 0)
            lines.append(f"  - GPU {i}: {name} ({mib} MiB)")
    else:
        lines += ["- _not installed_"]

    lines += ["", "## vLLM"]
    if inv.vllm.installed:
        lines += [
            f"- version:  `{inv.vllm.version}`",
            f"- location: `{inv.vllm.location}`",
        ]
    else:
        lines += ["- _not installed in current Python_"]
    lines.append("")
    return "\n".join(lines)


# ─── Plan ──────────────────────────────────────────────────────────────


def report_plan(
    plan: DepsPlan,
    dest: Optional[Path] = None,
    *,
    name: str = "deps-plan",
) -> tuple[Path, Path]:
    """Write plan JSON + MD; return (json_path, md_path)."""
    dest = dest or _default_reports_dir()
    dest.mkdir(parents=True, exist_ok=True)
    ts = _now_iso().replace(":", "-")
    json_path = dest / f"{name}-{ts}.json"
    md_path = dest / f"{name}-{ts}.md"

    body = {
        "kind": "deps_plan",
        "generated_at": _now_iso(),
        "plan": plan.to_dict(),
    }
    json_path.write_text(json.dumps(body, indent=2, sort_keys=True))
    md_path.write_text(_render_plan_md(plan))
    return json_path, md_path


def _render_plan_md(plan: DepsPlan) -> str:
    lines = [
        "# SNDR Deps Plan",
        "",
        f"_Generated: {_now_iso()}_",
        "",
        f"- config:    `{plan.config_key or '_global_'}`",
        f"- ready:     {'yes' if plan.is_ready() else 'NO'}",
        f"- blockers:  {len(plan.blockers())}",
        f"- warnings:  {len(plan.warnings())}",
        "",
    ]
    if not plan.items:
        lines += ["_No changes required — host is ready._"]
        return "\n".join(lines) + "\n"

    # Group by severity
    for severity in ("blocker", "warning", "info"):
        bucket = [i for i in plan.items if i.severity == severity]
        if not bucket:
            continue
        lines += [f"## {severity.capitalize()}s", ""]
        for item in bucket:
            lines.append(
                f"- **{item.scope}/{item.action}**: {item.target}"
            )
            lines.append(f"  - reason: {item.reason}")
            if item.suggested_command:
                lines.append(f"  - hint:   `{item.suggested_command}`")
        lines.append("")

    if plan.notes:
        lines += ["## Notes", ""]
        for n in plan.notes:
            lines.append(f"- {n}")
        lines.append("")

    return "\n".join(lines)
