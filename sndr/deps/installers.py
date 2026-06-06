# SPDX-License-Identifier: Apache-2.0
"""P3 (UNIFIED_CONFIG plan 2026-05-09) — side-effecting installer.

Walks a `DepsPlan` and applies each `PlanItem` via the appropriate
channel (apt / dnf / pip / etc). Strict gating:

  - Default mode = `--dry-run` — print what WOULD run, do nothing
  - `--yes` required to actually run anything
  - `--scope` filter: only items in the named scope (deps/launch/quality)
    get applied
  - Refuses `kind='curl_pipe_bash'` unless `allow_third_party=True` in
    the underlying source declaration

This module is deliberately separate from `checkers.py` and
`planners.py` so unit tests for those never accidentally execute
side effects.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Optional

from .planners import DepsPlan, PlanItem


@dataclass
class InstallResult:
    item: PlanItem
    status: str            # 'applied' | 'skipped' | 'failed' | 'dry-run'
    reason: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""


@dataclass
class ApplyOutcome:
    """Aggregate output of installers.apply()."""
    results: list[InstallResult]
    n_applied: int = 0
    n_skipped: int = 0
    n_failed: int = 0
    n_dry_run: int = 0


def apply(
    plan: DepsPlan,
    *,
    dry_run: bool = True,
    yes: bool = False,
    scope_filter: Optional[set[str]] = None,
    timeout_per_item: int = 600,
) -> ApplyOutcome:
    """Apply each PlanItem in `plan.items`.

    `dry_run=True` (default) → print + skip. `yes=True` required for
    real execution. `scope_filter` (optional set of `PlanItem.scope`
    values) restricts which items get applied.

    Returns an `ApplyOutcome` summarizing per-item results.
    """
    results: list[InstallResult] = []
    for item in plan.items:
        if scope_filter is not None and item.scope not in scope_filter:
            results.append(InstallResult(
                item=item, status="skipped",
                reason=f"scope {item.scope!r} not in filter",
            ))
            continue
        if dry_run or not yes:
            results.append(InstallResult(
                item=item, status="dry-run",
                reason=item.suggested_command or "(no command)",
            ))
            continue
        # Real execution path
        if item.suggested_command is None:
            results.append(InstallResult(
                item=item, status="skipped",
                reason="no suggested_command (manual step)",
            ))
            continue
        # Safety: detect curl|bash patterns + refuse
        cmd_lower = item.suggested_command.lower()
        if "curl" in cmd_lower and ("|sh" in cmd_lower or "| sh" in cmd_lower
                                      or "|bash" in cmd_lower
                                      or "| bash" in cmd_lower):
            results.append(InstallResult(
                item=item, status="failed",
                reason=("curl|bash pattern in suggested_command — "
                         "refused (this installer never auto-pipes "
                         "scripts to a shell)"),
            ))
            continue
        # Execute with timeout
        try:
            r = subprocess.run(
                ["/bin/bash", "-c", item.suggested_command],
                capture_output=True, text=True, timeout=timeout_per_item,
            )
            if r.returncode == 0:
                results.append(InstallResult(
                    item=item, status="applied",
                    reason="rc=0",
                    stdout_tail=r.stdout[-300:],
                ))
            else:
                results.append(InstallResult(
                    item=item, status="failed",
                    reason=f"rc={r.returncode}",
                    stdout_tail=r.stdout[-300:],
                    stderr_tail=r.stderr[-300:],
                ))
        except subprocess.TimeoutExpired:
            results.append(InstallResult(
                item=item, status="failed",
                reason=f"timeout after {timeout_per_item}s",
            ))
        except Exception as e:
            results.append(InstallResult(
                item=item, status="failed",
                reason=f"exception: {e}",
            ))

    n_applied = sum(1 for r in results if r.status == "applied")
    n_skipped = sum(1 for r in results if r.status == "skipped")
    n_failed = sum(1 for r in results if r.status == "failed")
    n_dry_run = sum(1 for r in results if r.status == "dry-run")
    return ApplyOutcome(
        results=results,
        n_applied=n_applied,
        n_skipped=n_skipped,
        n_failed=n_failed,
        n_dry_run=n_dry_run,
    )
