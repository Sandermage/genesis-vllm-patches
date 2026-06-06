# SPDX-License-Identifier: Apache-2.0
"""C12 (UNIFIED_CONFIG plan 2026-05-09) — `sndr bootstrap` universal installer.

Reads a preset's Y7 `bootstrap` block + Y2 `package_sources` and orchestrates
the install steps per declared scope. Composes:

  - `vllm.sndr_core.deps.inspect_host()`     — what's already there
  - `vllm.sndr_core.deps.plan_changes()`     — what needs to change
  - `vllm.sndr_core.deps.sources.resolve_source()` — pick channel
  - `vllm.sndr_core.deps.installers.apply()` — run installs (with safety)

Subcommands:
  sndr bootstrap doctor <key>               — read-only diagnostic
  sndr bootstrap plan <key>                 — print the plan
  sndr bootstrap apply <key> --scope <X>    — run the plan (--yes required)
  sndr bootstrap status <key>               — current state

`--scope` is comma-separated (or 'all'): os-packages, gpu-runtime,
python-runtime, container-runtime, model-artifacts, service.
"""
from __future__ import annotations

import argparse
from typing import Any

from . import _io


__all__ = ["add_argparser", "run_doctor", "run_plan", "run_apply",
           "run_status"]


def add_argparser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "bootstrap",
        help="Universal installer wrapper around Y7 BootstrapConfig (UNIFIED_CONFIG C12).",
        description=(
            "Bootstrap a host to run a Genesis preset: install missing "
            "dependencies per declared scope, honoring Y2 package_sources "
            "channel policy. Default --dry-run; --yes to execute."
        ),
    )
    sub = p.add_subparsers(dest="bootstrap_cmd", required=True)

    for cmd, helper, fn in (
        ("doctor", "Read-only diagnostic — what's missing for the preset", run_doctor),
        ("plan",   "Print the install plan (no execution)",                 run_plan),
        ("apply",  "Run the install plan (--yes required)",                 run_apply),
        ("status", "Current host state vs preset declared state",           run_status),
    ):
        sp = sub.add_parser(cmd, help=helper)
        sp.add_argument("config", help="model_config preset key")
        sp.add_argument("--scope", default="all",
                          help="comma-separated subset (default: all)")
        sp.add_argument("--yes", action="store_true",
                          help="Actually install (default: dry-run preview)")
        sp.add_argument("--json", action="store_true",
                          help="JSON output where applicable")
        sp.set_defaults(func=fn)


def _resolve(key: str):
    from vllm.sndr_core.model_configs.registry import get
    cfg = get(key)
    if cfg is None:
        _io.warn(f"unknown preset key {key!r}")
        return None
    if cfg.bootstrap is None:
        _io.warn(f"preset {key!r} has no Y7 bootstrap block; "
                  f"add `bootstrap:` to the YAML to use this CLI.")
        return None
    return cfg


def _scope_set(scope_arg: str, cfg) -> set[str]:
    """Parse --scope arg. Falls back to cfg.bootstrap.scopes if 'all'."""
    parts = [s.strip() for s in scope_arg.split(",") if s.strip()]
    if "all" in parts or not parts:
        # Use cfg-declared scopes (or all known scopes if cfg has none)
        if cfg.bootstrap.scopes:
            return set(cfg.bootstrap.scopes)
        return {"os-packages", "gpu-runtime", "python-runtime",
                 "container-runtime", "model-artifacts", "service"}
    return set(parts)


def _scope_to_plan_scope(s: set[str]) -> set[str]:
    """Map Y7 scope names to PlanItem.scope values used by deps.planners."""
    out: set[str] = set()
    if "os-packages" in s:
        out.update({"os"})
    if "gpu-runtime" in s:
        out.update({"nvidia", "docker"})  # nvidia toolkit lives in docker scope
    if "python-runtime" in s:
        out.update({"python"})
    if "container-runtime" in s:
        out.update({"docker"})
    if "model-artifacts" in s:
        out.update({"model"})  # not yet a real PlanItem.scope, but reserved
    return out


def _unsupported_plan_scopes(s: set[str]) -> set[str]:
    """Y7 scopes that the planner does not yet emit PlanItems for.

    All current scopes (`model-artifacts`, `service`, plus the legacy
    set) have planner coverage — see `deps/planners.py`. This stays as
    an empty hook so a future schema scope addition without a planner
    surfaces here instead of silently no-op'ing.
    """
    return set()


# ─── doctor

def run_doctor(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    from vllm.sndr_core.deps import inspect_host, plan_changes
    inv = inspect_host()
    plan = plan_changes(cfg, inv)
    scope = _scope_set(args.scope, cfg)

    if args.json:
        import json
        print(json.dumps({
            "preset": args.config,
            "bootstrap_scopes": sorted(scope),
            "host_ready_for_preset": plan.is_ready(),
            "n_blockers": len(plan.blockers()),
            "n_warnings": len(plan.warnings()),
        }, indent=2))
        return 0

    print(f"sndr bootstrap doctor '{args.config}'")
    print("─" * 60)
    print(f"  Y7 declared scopes: {sorted(cfg.bootstrap.scopes)}")
    print(f"  --scope filter:     {sorted(scope)}")
    print(f"  apply_policy:       {cfg.bootstrap.apply_policy}")
    print(f"  privilege:          {cfg.bootstrap.privilege}")
    print()
    print(f"  Host plan readiness: "
          f"{'READY' if plan.is_ready() else 'NOT READY'}")
    print(f"    blockers: {len(plan.blockers())}")
    print(f"    warnings: {len(plan.warnings())}")
    if plan.blockers():
        print()
        for b in plan.blockers():
            print(f"    ✗ [{b.scope}] {b.target}")
    return 0 if plan.is_ready() else 1


# ─── plan

def run_plan(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    from vllm.sndr_core.deps import inspect_host, plan_changes
    inv = inspect_host()
    plan = plan_changes(cfg, inv)
    scope = _scope_set(args.scope, cfg)
    plan_scope = _scope_to_plan_scope(scope)
    unsupported = _unsupported_plan_scopes(scope)

    print(f"sndr bootstrap plan '{args.config}'  scope={sorted(scope)}")
    print("─" * 60)
    if unsupported:
        _io.warn(
            "scope(s) not yet covered by deps planner: "
            + ", ".join(sorted(unsupported))
        )
    items_in_scope = [
        i for i in plan.items
        if i.scope in plan_scope
    ]
    if not items_in_scope:
        print("  (no items in scope — host is already ready for these scopes)")
        return 0
    for item in items_in_scope:
        print(f"  [{item.severity.upper()}] {item.scope}: {item.target}")
        if item.suggested_command:
            print(f"      → {item.suggested_command}")
        else:
            print(f"      (manual step — see {item.reason})")
    return 0


# ─── apply

def run_apply(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    from vllm.sndr_core.deps import inspect_host, plan_changes, apply
    inv = inspect_host()
    plan = plan_changes(cfg, inv)
    scope = _scope_set(args.scope, cfg)
    plan_scope = _scope_to_plan_scope(scope)

    # Honor cfg.bootstrap.apply_policy:
    #   ask      → require --yes
    #   auto-yes → execute without a CLI confirmation flag
    #   never    → refuse to apply
    yes = args.yes
    if cfg.bootstrap.apply_policy == "never":
        _io.error(f"preset {args.config!r} bootstrap.apply_policy='never' "
                   f"— refusing to install. Use a different preset.")
        return 2
    if cfg.bootstrap.apply_policy == "auto-yes":
        yes = True
    if cfg.bootstrap.apply_policy == "ask" and not yes:
        _io.warn("--yes required for apply_policy='ask' (running dry-run)")
        # fall through to dry-run path

    unsupported = _unsupported_plan_scopes(scope)
    if unsupported:
        _io.warn(
            "scope(s) not yet covered by deps planner: "
            + ", ".join(sorted(unsupported))
        )

    out = apply(plan, dry_run=not yes, yes=yes, scope_filter=plan_scope)

    print(f"sndr bootstrap apply '{args.config}' scope={sorted(scope)}")
    print("─" * 60)
    print(f"  applied:  {out.n_applied}")
    print(f"  skipped:  {out.n_skipped}")
    print(f"  failed:   {out.n_failed}")
    print(f"  dry-run:  {out.n_dry_run}")
    if out.n_failed > 0:
        print()
        print("  Failures:")
        for r in out.results:
            if r.status == "failed":
                print(f"    ✗ {r.item.scope}: {r.item.target} — {r.reason}")
        return 1
    return 0


# ─── status

def run_status(args: argparse.Namespace) -> int:
    """Re-uses doctor, but reports OK/NOT-OK only (terse)."""
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    from vllm.sndr_core.deps import inspect_host, plan_changes
    inv = inspect_host()
    plan = plan_changes(cfg, inv)
    if plan.is_ready():
        _io.success(f"preset {args.config!r}: host READY")
        return 0
    _io.warn(f"preset {args.config!r}: host NOT READY "
              f"({len(plan.blockers())} blockers)")
    return 1
