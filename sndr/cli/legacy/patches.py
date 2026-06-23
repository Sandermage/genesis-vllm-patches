# SPDX-License-Identifier: Apache-2.0
"""SNDR Core CLI — `sndr patches` registry/bundle browser + planner.

T1.2 (audit closure 2026-05-09 / production roadmap §18.2).

Subcommands:

  sndr patches list [--tier ...] [--lifecycle ...] [--default-on]
                    [--family ...] [--has-upstream] [--changed-since vX.Y]
                    [--json]
        — flat tabular dump filtered by registry attributes.

  sndr patches explain <PATCH_ID>
        — full metadata (title, tier, family, env_flag, default_on,
          lifecycle, applies_to, requires_patches, conflicts_with,
          upstream_pr, credit, category, apply_module dotted path).

  sndr patches doctor [--strict]
        — registry validator + apply-module coverage + dispatcher audit.
          Exit non-zero on ERROR-class issues when --strict is set.

  sndr patches plan --preset <KEY> [--json]
        — load preset YAML, simulate `should_apply()` for every registry
          entry, group into would-APPLY / would-SKIP buckets with
          per-patch reason codes. Mirrors what live `sndr launch` would
          decide without booting vllm.

  sndr patches diff-upstream
        — surface patches whose `upstream_pr` is in vllm pin's MERGED
          set or whose lifecycle == 'merged_upstream'. Used for sprint
          drift triage.

  sndr patches bundles list
  sndr patches bundles explain <NAME>
        — bundle-level browser parallel to `list` / `explain`.

Design notes
─────────────
This CLI is the **canonical operator-facing reflection** of
`PATCH_REGISTRY`. The legacy `compat.cli explain` / `categories`
subcommands stay for back-compat (bridged through `sndr <name>` per
DA-006), but they predate the spec/audit/coverage modules added in
PR38 and don't surface implementation_status, apply_module coverage,
or plan-time `should_apply()` simulation. New work should target this
module.

All output paths support `--json` for machine-readable export so
operators can pipe into jq / build issue templates without scraping
ANSI colors.

Author: Sandermage(Sander)-Barzov Aleksandr.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Optional

from . import _io

# M.6.1–.3 (2026-05-27): pure-data query layer extracted to
# ``sndr.product_api.legacy.patches``. This module is the thin
# argparse + rendering shell — every handler delegates to the API
# layer and prints the result. See
# ``sndr_private/planning/audits/M6_CLI_THIN_SHELL_R_2026-05-27_RU.md``.
#
# Submodules imported explicitly (not via package __init__) so the
# module reference doesn't get shadowed by its re-exported function of
# the same name (e.g. ``product_api.patches.diff_upstream`` is both a
# module and a function).
from sndr.product_api.legacy.patches import bench_attach as _bench_attach
from sndr.product_api.legacy.patches import bundles as _bundles
from sndr.product_api.legacy.patches import diff_upstream as _diff_upstream
from sndr.product_api.legacy.patches import doctor as _doctor
from sndr.product_api.legacy.patches import explain as _explain
from sndr.product_api.legacy.patches import listing as _listing
from sndr.product_api.legacy.patches import plan as _plan
from sndr.product_api.legacy.patches import pn95 as _pn95
from sndr.product_api.legacy.patches import proof_status as _proof_status
from sndr.product_api.legacy.patches import prove as _prove
from sndr.product_api.legacy.patches import release_check as _release_check


# ─── `sndr patches list` ─────────────────────────────────────────────────


def _run_list(opts: argparse.Namespace) -> int:
    typed_rows = _listing.list_patches(
        tier=opts.tier,
        lifecycle=opts.lifecycle,
        family=opts.family,
        default_on=(True if opts.default_on
                    else (False if opts.opt_in else None)),
        has_upstream=(True if opts.has_upstream
                      else (False if opts.no_upstream else None)),
    )
    rows: list[dict[str, Any]] = [asdict(r) for r in typed_rows]

    if opts.json:
        print(json.dumps({"count": len(rows), "patches": rows}, indent=2))
        return 0

    if not rows:
        _io.warn("no patches matched the filter set")
        return 0

    # ASCII table — fixed columns: id | tier | lc | def | env_flag | title
    cols = [
        ("Patch", "patch_id", 10),
        ("Tier", "tier", 9),
        ("Lifecycle", "lifecycle", 12),
        ("Def", "default_on", 5),
        ("Upstream", "upstream_pr", 9),
        ("Family", "family", 22),
        ("Title", "title", 50),
    ]
    header = " | ".join(name.ljust(width) for name, _, width in cols)
    print(header)
    print("-+-".join("-" * width for _, _, width in cols))
    for r in rows:
        cells = []
        for _name, key, width in cols:
            val = r[key]
            if val is None:
                rendered = ""
            elif isinstance(val, bool):
                rendered = "yes" if val else "no"
            elif key == "upstream_pr":
                rendered = f"#{val}"
            else:
                rendered = str(val)[:width]
            cells.append(rendered.ljust(width))
        print(" | ".join(cells))
    print(f"\n  {len(rows)} patches matched")
    return 0


# ─── `sndr patches explain` ──────────────────────────────────────────────


def _run_explain(opts: argparse.Namespace) -> int:
    view = _explain.explain_patch(opts.patch_id)
    if view is None:
        _io.error(f"patch_id {opts.patch_id!r} not found in PATCH_REGISTRY")
        candidates = _explain.suggest_candidates(opts.patch_id)
        if candidates:
            _io.info(f"did you mean: {', '.join(candidates)}")
        return 2

    pid = view.patch_id
    meta = view.meta
    spec = view.spec

    if opts.json:
        # Convert spec dataclass into a plain dict for JSON dump.
        out = asdict(spec)
        # Add raw registry fields not modeled in PatchSpec yet
        # (credit, category, implementation_status, deprecated reason, etc).
        for extra in ("credit", "category", "implementation_status",
                      "deprecated", "deprecation_reason", "notes"):
            if extra in meta:
                out[extra] = meta[extra]
        print(json.dumps(out, indent=2, default=str))
        return 0

    # Pretty print
    _io.banner(f"Patch {pid}", meta.get("title", "")[:60])

    def _row(label: str, value: Any) -> None:
        if value is None or value == "" or value == ():
            _io.info(f"  {label:<22} (unset)")
            return
        _io.info(f"  {label:<22} {value}")

    _row("Tier", spec.tier)
    _row("Family", spec.family)
    _row("Lifecycle", spec.lifecycle)
    _row("Implementation", meta.get("implementation_status", "(unset)"))
    _row("Default-on", "yes" if spec.default_on else "no")
    _row("Env flag", spec.env_flag)
    _row("Upstream PR", f"vllm#{spec.upstream_pr}" if spec.upstream_pr else None)
    if spec.related_upstream_prs:
        _row("Related PRs",
             ", ".join(f"vllm#{p}" for p in spec.related_upstream_prs))
    _row("Apply module", spec.apply_module or "(no on-disk module)")
    _row("Category", meta.get("category"))
    _row("Credit", meta.get("credit"))

    if spec.requires_patches:
        _io.info("")
        _io.info("  Requires patches:")
        for ref in spec.requires_patches:
            _io.info(f"    - {ref}")

    if spec.conflicts_with:
        _io.info("")
        _io.info("  Conflicts with:")
        for ref in spec.conflicts_with:
            _io.info(f"    - {ref}")

    if spec.applies_to:
        _io.info("")
        _io.info("  applies_to:")
        for k, v in spec.applies_to.items():
            _io.info(f"    {k}: {v}")

    if meta.get("deprecated"):
        _io.warn(
            f"  This patch is marked DEPRECATED: "
            f"{meta.get('deprecation_reason', '(no reason)')}"
        )

    # Live decision — if the API layer was able to probe ``should_apply``
    # it returns the verdict + reason; otherwise the field reads
    # ``(unavailable: <ExceptionType>)`` to match the historical CLI
    # output on hosts without a vllm runtime.
    _io.info("")
    if view.live_decision is not None:
        applied, reason = view.live_decision
        verdict = "APPLY" if applied else "SKIP"
        _row("Live decision", f"{verdict} — {reason}")
    else:
        err = view.live_decision_error or "unknown"
        _row("Live decision", f"(unavailable: {err})")

    return 0


# ─── `sndr patches pn95-status` ─────────────────────────────────────────


def _run_pn95_status(opts: argparse.Namespace) -> int:
    path = getattr(opts, "stats_file", "/tmp/pn95_stats.json")
    report = _pn95.read_pn95_status(path)
    if not report.available:
        if report.parse_error:
            if opts.json:
                print(json.dumps(
                    {"available": False, "reason": report.reason},
                    indent=2,
                ))
            else:
                _io.warn(
                    f"PN95 stats file at {path} is not parseable: "
                    f"{report.reason.removeprefix('parse error: ')}"
                )
            return 2
        # missing-file path
        if opts.json:
            print(json.dumps(
                {"available": False, "reason": report.reason},
                indent=2,
            ))
        else:
            _io.warn(report.reason)
        return 1

    stats = report.stats
    disk_stats = report.disk_tier
    hints = list(report.hints)

    if opts.json:
        print(json.dumps(
            {
                "available": True,
                "stats": stats,
                "disk_tier": disk_stats,
                "hints": hints,
            },
            indent=2, sort_keys=True,
        ))
        return 0

    _io.banner(
        "sndr patches pn95-status",
        f"stats={path}  ticks={stats['ticks_total']}  "
        f"demotes={stats['blocks_demoted_total']}  "
        f"prefix_store={stats['prefix_store_entries']} entries",
    )
    print("")
    print(f"  ticks_total:              {stats['ticks_total']}")
    print(f"  ticks_pressure_check:     {stats['ticks_pressure_check']}")
    print(f"  ticks_demote_triggered:   {stats['ticks_demote_triggered']}")
    print(f"  blocks_demoted_total:     {stats['blocks_demoted_total']}")
    print(f"  blocks_promoted_total:    {stats['blocks_promoted_total']}")
    print(f"  last_free_mib:            {stats['last_free_mib']}")
    print(f"  prefix_store_entries:     {stats['prefix_store_entries']}")
    print(f"  prefix_store_promote_hits:{stats['prefix_store_promote_hits']}")
    print(f"  async_demote_count:       {stats.get('async_demote_count', 0)}")
    print(f"  worker_proactive_calls:   {stats.get('worker_proactive_calls', 0)}")
    print(f"  worker_proactive_captured:{stats.get('worker_proactive_captured', 0)}")
    print(f"  ram_to_disk_spills:       {stats.get('ram_to_disk_spills_total', 0)}")
    print(f"  disk_to_ram_promotes:     {stats.get('disk_to_ram_promotes_total', 0)}")
    print(f"  timestamp:                {stats.get('timestamp', '-')}")
    if disk_stats and "error" not in disk_stats:
        print("")
        print("  ── disk tier (Tier 3) ──")
        print(f"  dir:           {disk_stats.get('disk_dir')}")
        print(f"  entries:       {disk_stats.get('disk_entries', 0)}")
        print(f"  bytes_on_disk: {disk_stats.get('disk_bytes_on_disk', 0)}")
        print(f"  capacity:      {disk_stats.get('disk_capacity_bytes', 0)}")
        print(f"  writes_total:  {disk_stats.get('disk_writes_total', 0)}")
        print(f"  read_hits:     {disk_stats.get('disk_read_hits_total', 0)}")
        print(f"  evictions:     {disk_stats.get('disk_evictions_total', 0)}")
        err = disk_stats.get('disk_last_io_error')
        if err:
            print(f"  last_io_error: {err}")
    print("")
    if hints:
        print("  Self-diagnosis:")
        for h in hints:
            sym = {"warn": "⚠", "ok": "✓", "err": "✗"}.get(h["severity"], "·")
            print(f"  {sym} [{h['severity']}] {h['message']}")
    else:
        print("  ✓ no anomalies detected")
    print("")
    return 0


# ─── `sndr patches doctor` ───────────────────────────────────────────────


def _run_doctor(opts: argparse.Namespace) -> int:
    report = _doctor.run_doctor()
    issues = list(report.issues)
    coverage = report.coverage
    registry_size = report.registry_size

    if opts.json:
        print(json.dumps({
            "registry_size": registry_size,
            "validation": [
                {"severity": i.severity, "patch_id": i.patch_id,
                 "message": i.message}
                for i in issues
            ],
            "apply_module_coverage": {
                "total": coverage.total,
                "mapped": coverage.mapped,
                "unmapped_count": len(coverage.unmapped),
                "unmapped": coverage.unmapped[:30],
            },
        }, indent=2))
        return _exit_for_issues(issues, opts.strict)

    _io.banner("sndr patches doctor",
               f"{registry_size} entries, "
               f"{coverage.mapped}/{coverage.total} have apply_module")

    # Validation summary
    err_count = sum(1 for i in issues if i.severity == "ERROR")
    warn_count = sum(1 for i in issues if i.severity == "WARNING")
    info_count = sum(1 for i in issues if i.severity == "INFO")

    _io.info(
        f"  Validator: ERROR={err_count}  WARNING={warn_count}  INFO={info_count}"
    )

    # Show ERROR + WARNING in full; cap INFO
    shown_info = 0
    INFO_CAP = 5
    for issue in issues:
        line = f"    [{issue.severity}] {issue.patch_id} — {issue.message}"
        if issue.severity == "ERROR":
            _io.error(line)
        elif issue.severity == "WARNING":
            _io.warn(line)
        else:
            shown_info += 1
            if shown_info <= INFO_CAP:
                _io.info(line)
    remaining_info = info_count - min(info_count, INFO_CAP)
    if remaining_info > 0:
        _io.info(f"    … and {remaining_info} more INFO entries")

    # Coverage detail. `unmapped` is the real-residual-gap list;
    # `intentionally_unmapped` is the legacy/marker/coordinator/retired
    # set that has no apply_module by design (Phase 3A.7+8, 2026-05-22).
    _io.info("")
    _io.info(
        f"  apply_module coverage: {coverage.mapped}/{coverage.total} "
        f"({len(coverage.unmapped)} unmapped, "
        f"{len(coverage.intentionally_unmapped)} intentionally unmapped)"
    )
    if coverage.unmapped:
        cap = 12
        sample = coverage.unmapped[:cap]
        _io.info(f"    sample unmapped (follow-up): {', '.join(sample)}")
        if len(coverage.unmapped) > cap:
            _io.info(f"    … and {len(coverage.unmapped) - cap} more")
    if coverage.intentionally_unmapped:
        cap = 12
        sample = coverage.intentionally_unmapped[:cap]
        _io.info(f"    sample intentionally unmapped: {', '.join(sample)}")
        if len(coverage.intentionally_unmapped) > cap:
            _io.info(
                f"    … and {len(coverage.intentionally_unmapped) - cap} more"
            )

    return _exit_for_issues(issues, opts.strict)


def _exit_for_issues(issues, strict: bool) -> int:
    """Return appropriate exit code given validator output + --strict flag."""
    err_count = sum(1 for i in issues if i.severity == "ERROR")
    if err_count > 0 and strict:
        return 1
    return 0


# ─── `sndr patches plan` ─────────────────────────────────────────────────


def _run_plan(opts: argparse.Namespace) -> int:
    """Simulate `should_apply()` decisions for every registry entry against
    a preset's env (`genesis_env` + `system_env`). This shows the operator
    the projected APPLY/SKIP set BEFORE booting vllm — a dry-run for the
    dispatcher rather than the launch script."""
    if not opts.preset:
        _io.fatal("--preset is required for `sndr patches plan`", 2)

    profile = getattr(opts, "profile", "any")
    policy = getattr(opts, "policy", None)
    explain = bool(getattr(opts, "explain", False))

    try:
        report = _plan.simulate_plan(
            opts.preset,
            profile=profile,
            policy=policy,
            explain=explain,
        )
    except _plan.PresetNotFoundError as e:
        _io.fatal(f"preset {e.preset_key!r} not found ({e.reason})", 2)

    apply_rows: list[dict[str, Any]] = list(report.apply)
    skip_rows: list[dict[str, Any]] = list(report.skip)
    error_rows: list[dict[str, Any]] = list(report.errors)
    profile_violations: list[dict[str, Any]] = list(report.profile_violations)
    resolver_payload = report.resolver_payload
    advisory_warnings: tuple[str, ...] = report.advisory_warnings

    # Late re-import for human renderer (`upstream_pr` lookup).
    from sndr.dispatcher import PATCH_REGISTRY

    if opts.json:
        out = {
            "preset": opts.preset,
            "profile": profile,
            "apply_count": len(apply_rows),
            "skip_count": len(skip_rows),
            "error_count": len(error_rows),
            "profile_violations": profile_violations,
            "apply": apply_rows,
            "skip": skip_rows,
            "errors": error_rows,
        }
        if resolver_payload is not None:
            out["resolver"] = resolver_payload
        if advisory_warnings:
            out["resolver_warnings"] = list(advisory_warnings)
        print(json.dumps(out, indent=2, default=str))
        return 2 if profile_violations else 0

    _io.banner(f"Plan: preset={opts.preset}",
               f"{len(apply_rows)} APPLY · {len(skip_rows)} SKIP "
               f"· {len(error_rows)} ERR")
    _io.info("  ✓ APPLIED")
    for r in apply_rows:
        upstream = ""
        meta = PATCH_REGISTRY.get(r["patch_id"], {})
        if meta.get("upstream_pr"):
            upstream = f"  ←  vllm#{meta['upstream_pr']}"
        _io.info(f"    + {r['patch_id']:<10} {r['title']:<60}{upstream}")
    if skip_rows:
        _io.info("")
        _io.info("  ⊘ SKIPPED")
        # Group by simple reason class for readability
        by_class: dict[str, list[dict[str, Any]]] = {}
        for r in skip_rows:
            cls = _classify_skip(r["reason"])
            by_class.setdefault(cls, []).append(r)
        for cls in sorted(by_class):
            items = by_class[cls]
            _io.info(f"    {cls} ({len(items)}):")
            for r in items[:8]:
                _io.info(f"      - {r['patch_id']:<10} {r['title'][:60]}")
            if len(items) > 8:
                _io.info(f"      … and {len(items) - 8} more")
    if error_rows:
        _io.info("")
        _io.warn(f"  ⚠ {len(error_rows)} dispatcher errors")
        for r in error_rows[:5]:
            _io.warn(f"    {r['patch_id']:<10} {r['error'][:80]}")
        if len(error_rows) > 5:
            _io.info(f"    … and {len(error_rows) - 5} more")

    if profile_violations:
        _io.info("")
        _io.error(
            f"  ✗ PRODUCTION PROFILE: {len(profile_violations)} blocker(s)"
        )
        for v in profile_violations:
            _io.warn(
                f"    {v['patch_id']:<10} {v['title']:<50} "
                f"[{', '.join(v['reasons'])}]"
            )
        _io.info("")
        _io.warn(
            "  Either fix the preset (disable the offending patches), "
            "raise the patches' lifecycle/status, or drop "
            "`--profile production` to allow this plan."
        )
        return 2

    # Phase B human renderer — only when --policy was passed.
    if resolver_payload is not None:
        _io.info("")
        _io.banner(
            f"Resolver: policy={resolver_payload['policy']}",
            f"{len(resolver_payload['included'])} included · "
            f"{len(resolver_payload['excluded'])} excluded",
        )
        for d in resolver_payload["included"]:
            line = f"  + {d['patch_id']:<10} role={d['role']:<22} {d['env_flag']}"
            _io.info(line)
            if explain and d.get("note"):
                _io.info(f"      note: {d['note'][:160]}")
            if explain and d.get("bench_evidence"):
                _io.info(f"      bench: {d['bench_evidence'][:160]}")
        if resolver_payload["excluded"]:
            _io.info("")
            _io.info("  ⊘ excluded by policy:")
            for d in resolver_payload["excluded"]:
                _io.info(
                    f"    - {d['patch_id']:<10} role={d['role']:<22} "
                    f"{d['env_flag']} — {d['reason'][:80]}"
                )
        if resolver_payload["warnings"]:
            _io.info("")
            _io.warn(f"  ⚠ {len(resolver_payload['warnings'])} warning(s):")
            for w in resolver_payload["warnings"]:
                _io.warn(f"    {w}")
    # Phase D refinement — surface resolver advisory warnings even
    # when --policy is NOT set, so legacy operators see conflict /
    # candidate_when mismatches they'd otherwise miss.
    if resolver_payload is None and advisory_warnings:
        _io.info("")
        _io.warn(
            f"  ⚠ {len(advisory_warnings)} advisory warning(s) from "
            "patch_plan resolver:"
        )
        for w in advisory_warnings:
            _io.warn(f"    {w}")
        _io.info(
            "  (pass `--policy compat --explain` for the full resolver view)"
        )
    return 0


def _classify_skip(reason: str) -> str:
    """Bucket a skip-reason string into one of the canonical classes."""
    r = (reason or "").lower()
    if "tier=engine" in r:
        return "engine-gated"
    if "opt-in" in r and "set " in r:
        return "opt-in (env unset)"
    if "model-compat" in r or "applies_to" in r:
        return "model-incompatible"
    if "deprecated" in r:
        return "deprecated"
    if "config_detect" in r:
        return "config-detect:skip"
    if "merged" in r and "upstream" in r:
        return "upstream-merged"
    return "other"


# ─── `sndr patches diff-upstream` ────────────────────────────────────────


def _run_diff_upstream(opts: argparse.Namespace) -> int:
    """Surface patches likely retiring because upstream merged the fix.

    Two signals:
      1. `lifecycle == "merged_upstream"` (operator already flipped)
      2. `upstream_pr` set AND it's in the vllm pin's MERGED set
         (heuristic: pin metadata not always present, fallback = list
         all patches with `upstream_pr` so the operator can audit).
    """
    report = _diff_upstream.diff_upstream()
    merged_upstream: list[dict[str, Any]] = list(report.merged_upstream)
    has_upstream_pr: list[dict[str, Any]] = list(report.has_upstream_pr)

    if opts.json:
        print(json.dumps({
            "merged_upstream": merged_upstream,
            "has_upstream_pr": has_upstream_pr,
            "merged_upstream_count": len(merged_upstream),
            "has_upstream_pr_count": len(has_upstream_pr),
        }, indent=2))
        return 0

    _io.banner("Upstream drift triage",
               f"{len(merged_upstream)} retired · "
               f"{len(has_upstream_pr)} still active w/ upstream PR")
    if merged_upstream:
        _io.info("  Lifecycle = merged_upstream (already retired):")
        for r in merged_upstream:
            pr = f"vllm#{r['upstream_pr']}" if r["upstream_pr"] else "(no PR ref)"
            _io.info(f"    - {r['patch_id']:<10} {pr:<14} {r['title']}")
    if has_upstream_pr:
        _io.info("")
        _io.info("  Active patches with upstream_pr (audit candidates):")
        for r in has_upstream_pr[:30]:
            pr = f"vllm#{r['upstream_pr']}"
            _io.info(
                f"    - {r['patch_id']:<10} {pr:<14} "
                f"lifecycle={r['lifecycle']:<14} {r['title']}"
            )
        if len(has_upstream_pr) > 30:
            _io.info(f"    … and {len(has_upstream_pr) - 30} more")
    return 0


# ─── `sndr patches bundles ...` ──────────────────────────────────────────


def _run_bundles_list(opts: argparse.Namespace) -> int:
    specs = _bundles.list_bundles()
    if opts.json:
        print(json.dumps([
            {
                "name": b.name,
                "umbrella_flag": b.umbrella_flag,
                "tier": b.tier,
                "description": b.description,
            }
            for b in specs
        ], indent=2))
        return 0

    _io.banner("SNDR Bundles", f"{len(specs)} atomic multi-patch orchestrators")
    for b in specs:
        _io.info(f"  • {b.name}  [{b.tier}]")
        _io.info(f"      flag:  SNDR_ENABLE_{b.umbrella_flag}=1")
        _io.info(f"      desc:  {b.description}")
        _io.info("")
    return 0


def _run_bundles_explain(opts: argparse.Namespace) -> int:
    target = opts.name
    spec = _bundles.explain_bundle(target)
    if spec is None:
        _io.error(f"bundle {target!r} not found")
        available = ", ".join(b[0] for b in _bundles.BUNDLES_CATALOG)
        _io.info(f"available: {available}")
        return 2

    if spec.import_error is not None:
        _io.warn(f"  bundle module failed to import: {spec.import_error}")

    if opts.json:
        print(json.dumps({
            "name": spec.name,
            "umbrella_flag": spec.umbrella_flag,
            "tier": spec.tier,
            "description": spec.description,
            "module": spec.module_path,
            "has_apply": bool(spec.has_apply),
        }, indent=2))
        return 0

    _io.banner(f"Bundle: {spec.name}", spec.description[:60])
    _io.info(f"  Tier:           {spec.tier}")
    _io.info(
        f"  Umbrella flag:  SNDR_ENABLE_{spec.umbrella_flag}=1  "
        f"(or GENESIS_ENABLE_{spec.umbrella_flag}=1)"
    )
    _io.info(f"  Module:         {spec.module_path}")
    _io.info(f"  apply():        {'callable' if spec.has_apply else 'MISSING'}")
    _io.info("")
    _io.info("  Description:")
    _io.info(f"    {spec.description}")
    return 0


# ─── argparse plumbing ───────────────────────────────────────────────────


def add_argparser(subparsers: Any) -> None:
    """Register `sndr patches ...` parent + subcommands."""
    parent = subparsers.add_parser(
        "patches",
        help="Browse, plan, and validate the patch registry.",
        description=(
            "`sndr patches` — registry browser + plan/doctor/diff-upstream "
            "tooling. Replaces the legacy `sndr explain` / `categories` "
            "for new work; bridged compat commands stay for back-compat."
        ),
    )
    sub = parent.add_subparsers(dest="patches_cmd", title="Subcommands",
                                metavar="{list,explain,doctor,plan,diff-upstream,bundles}")

    # list
    p_list = sub.add_parser("list", help="Filter + tabulate registry entries.")
    p_list.add_argument("--tier", choices=("community", "engine"))
    p_list.add_argument("--lifecycle",
                        help="One of stable/experimental/deprecated/legacy/...")
    p_list.add_argument("--family", help="Substring match against family field.")
    p_list.add_argument("--default-on", action="store_true",
                        dest="default_on",
                        help="Only patches that are default-on.")
    p_list.add_argument("--opt-in", action="store_true",
                        dest="opt_in",
                        help="Only opt-in patches (default_on=False).")
    p_list.add_argument("--has-upstream", action="store_true",
                        dest="has_upstream",
                        help="Only patches with an upstream_pr field.")
    p_list.add_argument("--no-upstream", action="store_true",
                        dest="no_upstream",
                        help="Only patches WITHOUT an upstream_pr field.")
    p_list.add_argument("--json", action="store_true",
                        help="Emit JSON instead of ASCII table.")
    p_list.set_defaults(func=_run_list)

    # explain
    p_explain = sub.add_parser("explain",
                               help="Print full metadata for one patch.")
    p_explain.add_argument("patch_id", help="e.g. P67, PN82, p61c.")
    p_explain.add_argument("--json", action="store_true")
    p_explain.set_defaults(func=_run_explain)

    # doctor
    p_doctor = sub.add_parser(
        "doctor",
        help="Validate registry shape + apply_module coverage.",
    )
    p_doctor.add_argument(
        "--strict", action="store_true",
        help="Exit non-zero if any ERROR-class issues are reported.",
    )
    p_doctor.add_argument("--json", action="store_true")
    p_doctor.set_defaults(func=_run_doctor)

    # pn95-status — live PN95 runtime diagnostic
    p_pn95 = sub.add_parser(
        "pn95-status",
        help="Live PN95 runtime stats: ticks, pressure checks, demote count, "
             "prefix store size — and a self-diagnosis of common gaps "
             "(multiproc TM gap, no eligible attention layers, etc.).",
    )
    p_pn95.add_argument(
        "--stats-file", default="/tmp/pn95_stats.json",
        help="Path to the PN95 stats file written by the worker process "
             "(default: /tmp/pn95_stats.json). On a running prod container "
             "this is the canonical observability surface.",
    )
    p_pn95.add_argument("--json", action="store_true")
    p_pn95.set_defaults(func=_run_pn95_status)

    # prove (§6.8 patch proof gate / R1 mitigation, Phase 4.5+)
    p_prove = sub.add_parser(
        "prove",
        help="Static-check coverage + proof-artefact writer (§6.8).",
        description=(
            "Verify that a patch is wired up correctly: registered, "
            "apply_module importable, no shadow orphan, env_flag canonical, "
            "dependencies resolve. Writes `evidence/patch_proof/<id>__<vllm_pin>.json` "
            "with static-check results. Bench-delta evidence slots in later."
        ),
    )
    p_prove.add_argument(
        "patch_id", nargs="?", default=None,
        help="Patch id to prove (e.g. P67b). Omit with --all or --dead-detect.",
    )
    p_prove.add_argument(
        "--all", action="store_true", dest="prove_all",
        help="Sweep every PATCH_REGISTRY entry; report coverage %%.",
    )
    p_prove.add_argument(
        "--dead-detect", action="store_true", dest="dead_detect",
        help="List patches with no proof artefact (no static-check run).",
    )
    p_prove.add_argument(
        "--out-dir", default=None,
        help="Override evidence/patch_proof/ output directory.",
    )
    p_prove.add_argument(
        "--no-write", action="store_true",
        help="Run checks but don't persist artefact (dry-run).",
    )
    p_prove.add_argument("--json", action="store_true")
    p_prove.set_defaults(func=_run_prove)

    # bench-attach (§6.8 bench-delta evidence ingestion, Entry 19)
    p_ba = sub.add_parser(
        "bench-attach",
        help="Attach a bench-suite JSON to a patch proof artefact (§6.8).",
        description=(
            "Ingest a bench-suite result JSON (one operator runs on GPU, "
            "this runs anywhere) and write its headline metrics into the "
            "patch's `evidence/patch_proof/<id>__<vllm_pin>.json` artefact "
            "as `bench_delta`. Optional `--baseline` computes percent "
            "deltas vs a baseline bench JSON."
        ),
    )
    p_ba.add_argument("patch_id", help="Patch id (e.g. PN90).")
    p_ba.add_argument("bench_path",
                      help="Path to the bench-suite result JSON.")
    p_ba.add_argument("--baseline", default=None,
                      help="Optional baseline bench JSON for delta computation.")
    p_ba.add_argument("--out-dir", default=None,
                      help="Override evidence/patch_proof/ output directory.")
    p_ba.add_argument("--json", action="store_true",
                      help="Emit machine-readable JSON summary.")
    p_ba.set_defaults(func=_run_bench_attach)

    # proof-status (§6.8 read-side reporting, Entry 20)
    p_ps = sub.add_parser(
        "proof-status",
        help="Per-patch proof-artefact bucket summary (§6.8 readout).",
        description=(
            "Walk PATCH_REGISTRY and classify each patch's proof "
            "artefact into one of five buckets: bench_with_baseline / "
            "bench_attached / static_only / static_failed / dead. "
            "Operator-visible at-a-glance of §6.8 evidence health."
        ),
    )
    p_ps.add_argument(
        "--out-dir", default=None,
        help="Override evidence/patch_proof/ artefact directory.",
    )
    p_ps.add_argument(
        "--bucket", action="append", default=None,
        help="Only show patches in this bucket (repeatable). "
             "Valid: bench_with_baseline, bench_attached, static_only, "
             "static_failed, dead.",
    )
    p_ps.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON.",
    )
    p_ps.set_defaults(func=_run_proof_status)

    # release-check (§6.8 release-gate consumer, Entry 21)
    p_rc = sub.add_parser(
        "release-check",
        help="Decide release-readiness from proof artefacts (§6.8).",
        description=(
            "Apply a release policy across every PATCH_REGISTRY entry "
            "and decide whether release is blocked. Combines static-check "
            "bucket (Entry 17) + bench_delta presence (Entry 19) + bucket "
            "summary (Entry 20) into a single release/no-release verdict. "
            "Mode 'report' never blocks; tighter modes (require-static / "
            "require-bench / require-baseline) block when patches don't "
            "meet the bar. `--max-regression-pct N` also blocks when any "
            "bench_with_baseline patch has a regression beyond N%. "
            "The current public release gate is `require-static` — see "
            "docs/RELEASE_POLICY.md for the cutover procedure between modes."
        ),
    )
    p_rc.add_argument(
        "--mode", default="report",
        choices=["report", "require-static", "require-bench", "require-baseline"],
        help="Policy strictness. Default: report (never blocks).",
    )
    p_rc.add_argument(
        "--max-regression-pct", type=float, default=None,
        help="Block when a bench_with_baseline patch has a TPS drop or "
             "latency rise beyond this percent (e.g. 5.0 = ±5%%).",
    )
    p_rc.add_argument(
        "--patch", action="append", default=None,
        help="Restrict to these patch ids (repeatable).",
    )
    p_rc.add_argument(
        "--tier", action="append", default=None,
        help="Restrict to these tiers (repeatable, e.g. release, community).",
    )
    p_rc.add_argument(
        "--scope", default="all",
        choices=["all", "production-subset"],
        help=(
            "Restrict evaluation to a named subset. 'production-subset' "
            "is the union of patches enabled by any prod-* V2 preset + "
            "every default_on=True entry — the practical scope for "
            "hardened-release bench/baseline gating. Default: all."
        ),
    )
    p_rc.add_argument(
        "--out-dir", default=None,
        help="Override evidence/patch_proof/ artefact directory.",
    )
    p_rc.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON.",
    )
    p_rc.add_argument(
        "--show-passing", action="store_true",
        help="In human mode, also print passing verdicts (default: failed only).",
    )
    p_rc.set_defaults(func=_run_release_check)

    # plan
    p_plan = sub.add_parser(
        "plan",
        help="Simulate dispatcher decisions for a preset.",
        description=(
            "Loads the preset's genesis_env+system_env, runs should_apply() "
            "for every registry entry, groups into APPLY/SKIP buckets with "
            "per-patch reasons. Use to verify a preset BEFORE booting vllm."
        ),
    )
    p_plan.add_argument("--preset", required=True,
                        help="model_config key (e.g. a5000-2x-35b-prod).")
    p_plan.add_argument("--json", action="store_true")
    p_plan.add_argument(
        "--profile",
        choices=("any", "production"),
        default="any",
        help=(
            "`production` blocks the plan when it would APPLY a patch with "
            "implementation_status ∈ {partial, placeholder} or lifecycle "
            "∈ {research, retired}. Use before live launch to guarantee "
            "no half-finished code reaches PROD."
        ),
    )
    p_plan.add_argument(
        "--policy",
        choices=("compat", "safe", "minimal"),
        default=None,
        help=(
            "run the patch_plan resolver alongside "
            "the dispatcher simulator. compat passes every truthy "
            "genesis_env flag through; safe drops role=='no_op'; minimal "
            "additionally drops role in {suspected_regression, unknown}. "
            "Omit the flag to keep legacy output (simulator only)."
        ),
    )
    p_plan.add_argument(
        "--explain",
        action="store_true",
        help=(
            "Include role / note / bench_evidence from patches_attribution "
            "in the resolver output. No effect without --policy."
        ),
    )
    p_plan.set_defaults(func=_run_plan)

    # diff-upstream
    p_diff = sub.add_parser(
        "diff-upstream",
        help="Surface patches whose upstream PR may have merged.",
    )
    p_diff.add_argument("--json", action="store_true")
    p_diff.set_defaults(func=_run_diff_upstream)

    # bundles
    p_bundles = sub.add_parser(
        "bundles", help="Browse atomic multi-patch bundles.",
    )
    bsub = p_bundles.add_subparsers(dest="bundles_cmd",
                                    metavar="{list,explain}")
    p_blist = bsub.add_parser("list", help="List all bundles.")
    p_blist.add_argument("--json", action="store_true")
    p_blist.set_defaults(func=_run_bundles_list)
    p_bex = bsub.add_parser("explain", help="Explain one bundle.")
    p_bex.add_argument("name", help="Bundle module name (no .py).")
    p_bex.add_argument("--json", action="store_true")
    p_bex.set_defaults(func=_run_bundles_explain)

    # Default for `sndr patches` (no subcommand) — show help.
    parent.set_defaults(func=lambda _ns: parent.print_help() or 0)


# ─── prove (§6.8 patch proof gate) ─────────────────────────────────────


def _run_prove(args: argparse.Namespace) -> int:
    """Dispatcher for `sndr patches prove [id|--all|--dead-detect]`."""
    from pathlib import Path

    out_dir = Path(args.out_dir) if args.out_dir else None

    if args.dead_detect:
        return _run_prove_dead_detect(args, out_dir)
    if args.prove_all:
        return _run_prove_all(args, out_dir)
    if not args.patch_id:
        _io.warn("provide a patch_id, or use --all / --dead-detect")
        return 2
    return _run_prove_one(args, out_dir)


def _run_prove_one(args: argparse.Namespace, out_dir) -> int:
    """`sndr patches prove <id>` — verify one patch + write artefact."""
    import json as _json

    result = _prove.prove_one(
        args.patch_id, out_dir=out_dir, no_write=bool(args.no_write),
    )
    proof = result.proof
    path = result.artefact_path

    if args.json:
        payload = {
            "patch_id": proof.patch_id,
            "vllm_pin": proof.vllm_pin,
            "genesis_pin": proof.genesis_pin,
            "commit_sha": proof.commit_sha,
            "host": proof.host,
            "measured_at": proof.measured_at,
            "static_checks": [
                {"rule": c.rule, "passed": c.passed, "message": c.message}
                for c in proof.static_checks
            ],
            "static_passed": proof.static_passed,
            "artefact_path": str(path) if path else None,
        }
        print(_json.dumps(payload, indent=2, sort_keys=True))
        return 0 if proof.static_passed else 1

    print(f"sndr patches prove '{proof.patch_id}'")
    print(f"  vllm_pin: {proof.vllm_pin}  commit: {proof.commit_sha}  "
          f"host: {proof.host}")
    print(f"  measured: {proof.measured_at}")
    print("─" * 70)
    for c in proof.static_checks:
        sym = "✓" if c.passed else "✗"
        print(f"  {sym} [{c.rule}] {c.message}")
    print()
    if path:
        print(f"  artefact: {path}")
    elif args.no_write:
        print("  (artefact write skipped — --no-write)")
    else:
        print("  (no artefact written — P-1 failed)")
    print()
    if proof.static_passed:
        print(f"  ✓ static checks passed ({len(proof.static_checks)}/{len(proof.static_checks)})")
    else:
        n_fail = len(proof.static_errors)
        print(f"  ✗ static checks FAILED ({n_fail}/{len(proof.static_checks)} failing)")
    return 0 if proof.static_passed else 1


def _run_prove_all(args: argparse.Namespace, out_dir) -> int:
    """`sndr patches prove --all` — sweep all PATCH_REGISTRY entries."""
    import json as _json

    sweep = _prove.prove_all(out_dir=out_dir, no_write=bool(args.no_write))
    results = list(sweep.results)
    total = sweep.total
    passed = sweep.passed
    coverage = (passed / total) if total else 1.0

    if args.json:
        print(_json.dumps({
            "total": total,
            "passed": passed,
            "failed": sweep.failed,
            "coverage_pct": sweep.coverage_pct,
            "results": results,
        }, indent=2, sort_keys=True))
        return 0 if passed == total else 1

    print(f"sndr patches prove --all   ({total} patches)")
    print("─" * 70)
    for r in results:
        sym = "✓" if r["passed"] else "✗"
        print(f"  {sym} {r['patch_id']}")
        if not r["passed"]:
            for e in r["errors"][:3]:
                print(f"      [{e['rule']}] {e['message']}")
    print()
    print(f"  Coverage: {passed}/{total} ({coverage*100:.1f}%)")
    return 0 if passed == total else 1


def _run_prove_dead_detect(args: argparse.Namespace, out_dir) -> int:
    """`sndr patches prove --dead-detect` — list patches without artefacts."""
    import json as _json

    detect = _prove.dead_detect(out_dir=out_dir)
    total = detect.total_patches
    proven = detect.proven
    dead = list(detect.dead_patches)
    coverage = (proven / total) if total else 1.0

    if args.json:
        print(_json.dumps({
            "total_patches": total,
            "proven": proven,
            "dead": detect.dead_count,
            "coverage_pct": detect.coverage_pct,
            "dead_patches": dead,
        }, indent=2, sort_keys=True))
        return 0

    print(f"sndr patches prove --dead-detect   ({total} patches in registry)")
    print("─" * 70)
    print(f"  proven (has passing static artefact): {proven}")
    print(f"  dead   (no passing artefact):         {detect.dead_count}")
    print(f"  coverage:                             {coverage*100:.1f}%")
    print()
    if dead:
        print("  Patches with no proof artefact:")
        for d in dead[:20]:
            arts = f" [stale: {len(d['artefacts_found'])}]" if d["artefacts_found"] else ""
            print(f"    - {d['patch_id']}  (lifecycle={d['lifecycle']}, "
                  f"tier={d['tier']}, family={d['family']}){arts}")
        if len(dead) > 20:
            print(f"    ... ({len(dead) - 20} more)")
    return 0


def _run_bench_attach(args: argparse.Namespace) -> int:
    """`sndr patches bench-attach <patch_id> <bench.json> [--baseline X.json]`."""
    import json as _json
    from pathlib import Path
    from sndr.proof.bench_attach import BenchAttachError

    out_dir = Path(args.out_dir) if args.out_dir else None
    bench = Path(args.bench_path)
    baseline = Path(args.baseline) if args.baseline else None

    try:
        result = _bench_attach.attach_bench(
            args.patch_id,
            bench,
            baseline_path=baseline,
            out_dir=out_dir,
        )
    except BenchAttachError as e:
        _io.warn(str(e))
        return 2

    target = result.artefact_path
    bench_delta = result.bench_delta

    if args.json:
        print(_json.dumps({
            "patch_id": args.patch_id,
            "artefact_path": str(target),
            "bench_delta": bench_delta,
        }, indent=2, sort_keys=True))
        return 0

    print(f"sndr patches bench-attach '{args.patch_id}'")
    print(f"  bench:    {bench}")
    if baseline:
        print(f"  baseline: {baseline}")
    print(f"  artefact: {target}")
    print("─" * 70)
    if not bench_delta:
        print("  ⚠ no metrics extracted — bench JSON may be empty / unrecognised shape")
        return 1
    for k in ("median_tps", "p95_tps", "decode_tpot_ms", "ttft_ms",
              "cv_pct", "tool_call_score"):
        v = bench_delta.get(k)
        if v is None:
            continue
        # Two specific delta keys we render: median_tps + p95_tps + decode_tpot + ttft
        pretty_delta = ""
        if k == "median_tps":
            d = bench_delta.get("median_tps_delta_pct")
            if d is not None:
                pretty_delta = f"  ({d:+.2f}% vs baseline)"
        elif k == "p95_tps":
            d = bench_delta.get("p95_tps_delta_pct")
            if d is not None:
                pretty_delta = f"  ({d:+.2f}% vs baseline)"
        elif k == "decode_tpot_ms":
            d = bench_delta.get("decode_tpot_delta_pct")
            if d is not None:
                pretty_delta = f"  ({d:+.2f}% vs baseline)"
        elif k == "ttft_ms":
            d = bench_delta.get("ttft_delta_pct")
            if d is not None:
                pretty_delta = f"  ({d:+.2f}% vs baseline)"
        print(f"  {k:18s} = {v}{pretty_delta}")
    print()
    print(f"  ✓ bench_delta attached to {target.name}")
    return 0


def _run_proof_status(args: argparse.Namespace) -> int:
    """`sndr patches proof-status` — bucket summary of patch proof state."""
    import json as _json
    from pathlib import Path
    from sndr.proof import DEFAULT_PROOF_DIR, PROOF_STATUS_BUCKETS

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_PROOF_DIR
    bucket_filter: Optional[list[str]] = list(args.bucket) if args.bucket else None
    try:
        result = _proof_status.proof_status(
            out_dir=out_dir, bucket_filter=bucket_filter,
        )
    except _proof_status.UnknownBucketError as e:
        _io.warn(
            f"unknown bucket(s): {e.unknown!r}. "
            f"Valid: {e.valid}"
        )
        return 2

    filter_set: Optional[set[str]] = (
        set(result.filter_buckets) if result.filter_buckets is not None else None
    )
    filtered = list(result.patches)

    if args.json:
        payload = {
            "total": result.total,
            "counts": result.counts,
            "patches": filtered,
            "filter_buckets": (
                sorted(result.filter_buckets)
                if result.filter_buckets is not None else None
            ),
        }
        print(_json.dumps(payload, indent=2, sort_keys=True))
        return 0

    counts = result.counts
    total = result.total
    print(f"sndr patches proof-status — {total} patches, {out_dir}")
    print("─" * 70)
    for b in PROOF_STATUS_BUCKETS:
        n = counts.get(b, 0)
        pct = (n / total * 100.0) if total else 0.0
        print(f"  {b:24s} {n:4d}  ({pct:5.1f}%)")
    print("─" * 70)

    if filter_set is not None:
        print(f"Filtered to buckets: {sorted(filter_set)}")
        print(f"  → {len(filtered)} patch(es)")
        print()
        for p in filtered:
            print(
                f"  [{p['bucket']:22s}] {p['patch_id']:8s} "
                f"family={p['family']} tier={p['tier']} "
                f"lifecycle={p['lifecycle']}"
            )
    return 0


def _run_release_check(args: argparse.Namespace) -> int:
    """`sndr patches release-check` — release-gate consumer.

    Exit codes:
      0  policy passed (or mode=report)
      1  policy failed — at least one patch blocks release
      2  bad CLI input
    """
    import json as _json
    from pathlib import Path
    from sndr.proof import DEFAULT_PROOF_DIR
    from sndr.proof.release_check import ReleaseCheckError

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_PROOF_DIR

    # --patch wins over --scope production-subset (operator override);
    # both flags are surfaced to the product_api which performs the
    # production-subset expansion when ``patch_filter`` is unset.
    patch_filter = list(args.patch) if args.patch else None
    tier_filter = list(args.tier) if args.tier else None
    scope = getattr(args, "scope", "all")

    try:
        result = _release_check.release_check(
            mode=args.mode,
            out_dir=out_dir,
            max_regression_pct=args.max_regression_pct,
            patch_filter=patch_filter,
            tier_filter=tier_filter,
            scope=scope,
        )
    except ReleaseCheckError as e:
        _io.warn(str(e))
        return 2

    report = result.raw

    if args.json:
        print(_json.dumps(report, indent=2, sort_keys=True))
        return 1 if report["release_blocked"] else 0

    pol = report["policy"]
    print(f"sndr patches release-check — mode={pol['mode']}")
    if pol["max_regression_pct"] is not None:
        print(f"  max_regression_pct: ±{pol['max_regression_pct']}%")
    if pol["patch_filter"]:
        print(f"  patch filter: {pol['patch_filter']}")
    if pol["tier_filter"]:
        print(f"  tier filter:  {pol['tier_filter']}")
    print(f"  artefact dir: {out_dir}")
    print("─" * 70)
    print(
        f"  considered={report['considered']}/{report['total']}  "
        f"passed={report['passed_count']}  failed={report['failed_count']}"
    )
    print()

    failed = [v for v in report["verdicts"] if not v["passed"]]
    passing = [v for v in report["verdicts"] if v["passed"]]

    if failed:
        print(f"  ✗ {len(failed)} patch(es) block release:")
        for v in failed[:40]:
            print(
                f"    [{v['bucket']:22s}] {v['patch_id']:8s} "
                f"family={v['family']} tier={v['tier']}"
            )
            for r in v["reasons"]:
                print(f"        - {r}")
            for reg in v["regressions"]:
                print(
                    f"        regression: {reg['metric']} "
                    f"{reg['delta_pct']:+.2f}% ({reg['polarity']})"
                )
        if len(failed) > 40:
            print(f"    ... ({len(failed) - 40} more)")
        print()

    if args.show_passing and passing:
        print(f"  ✓ {len(passing)} passing:")
        for v in passing[:40]:
            print(
                f"    [{v['bucket']:22s}] {v['patch_id']:8s} "
                f"family={v['family']} tier={v['tier']}"
            )
        if len(passing) > 40:
            print(f"    ... ({len(passing) - 40} more)")
        print()

    if report["release_blocked"]:
        print(f"  ✗ RELEASE BLOCKED — policy={pol['mode']!r}")
        return 1
    if pol["mode"] == "report":
        print("  · report-only mode (no blocking)")
    else:
        print(f"  ✓ release policy {pol['mode']!r} satisfied")
    return 0


__all__ = ["add_argparser"]
