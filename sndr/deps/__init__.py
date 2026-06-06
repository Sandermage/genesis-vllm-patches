# SPDX-License-Identifier: Apache-2.0
"""SNDR deps — system dependency inventory + planning (P3).

UNIFIED_CONFIG_AUTOMATION_PLAN 2026-05-09: this package is the
foundational layer beneath `sndr deps check / plan / install` (C2),
`sndr install --config <key> --prepare` (C5), and the bootstrap
sequence (C12).

Design principles:
  - **Pure inspection** in `checkers.py` — no installs, no privileged
    operations, no network calls beyond `which` / `--version` probes.
  - **Pure planning** in `planners.py` — given a config + an inventory,
    compute what needs to change. Plan is a typed dataclass; no IO.
  - **Side-effecting installers** stay in `installers.py` (separate
    module, opt-in import) so unit tests cover checkers/planners with
    zero risk of accidentally `apt install`-ing on a dev machine.

Public API (this `__init__`):
  - `inspect_host()`  → returns `HostInventory` snapshot
  - `plan_changes(cfg, inventory)` → returns `DepsPlan`
  - `report_inventory(inventory, dest)` → writes JSON+MD reports
  - `report_plan(plan, dest)` → writes JSON+MD reports

Future (not yet implemented):
  - `installers.apply(plan, *, dry_run, scope, yes)` — runs the plan;
    has its own scope-gating and rollback support.
"""
from __future__ import annotations

from .checkers import (
    HostInventory,
    DockerInfo,
    NvidiaInfo,
    PythonInfo,
    VLLMInfo,
    inspect_host,
)
from .planners import (
    DepsPlan,
    PlanItem,
    plan_changes,
)
from .report import (
    report_inventory,
    report_plan,
)
from .sources import (
    SourceDecision,
    resolve_source,
    list_safe_channels,
)
from .installers import (
    InstallResult,
    ApplyOutcome,
    apply,
)

__all__ = [
    "HostInventory",
    "DockerInfo",
    "NvidiaInfo",
    "PythonInfo",
    "VLLMInfo",
    "inspect_host",
    "DepsPlan",
    "PlanItem",
    "plan_changes",
    "report_inventory",
    "report_plan",
    # Tier 2 P3 full (UNIFIED_CONFIG plan 2026-05-09)
    "SourceDecision",
    "resolve_source",
    "list_safe_channels",
    "InstallResult",
    "ApplyOutcome",
    "apply",
]
