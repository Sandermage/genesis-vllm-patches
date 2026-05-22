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


# ─── Filtering helpers (shared by `list` + `plan`) ───────────────────────


def _coerce_iter(value: Any) -> tuple:
    """Loosely turn a registry tuple/list/string into an iterable for display."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(value)
    if isinstance(value, str):
        return (value,)
    return (value,)


def _matches_filters(
    spec,
    *,
    tier: Optional[str] = None,
    lifecycle: Optional[str] = None,
    family: Optional[str] = None,
    default_on: Optional[bool] = None,
    has_upstream: Optional[bool] = None,
) -> bool:
    """Return True if `spec` matches every non-None filter."""
    if tier is not None and spec.tier != tier:
        return False
    if lifecycle is not None and spec.lifecycle != lifecycle:
        return False
    if family is not None and family not in (spec.family or ""):
        return False
    if default_on is True and not spec.default_on:
        return False
    if default_on is False and spec.default_on:
        return False
    if has_upstream is True and not spec.upstream_pr:
        return False
    if has_upstream is False and spec.upstream_pr:
        return False
    return True


def _spec_to_row(spec) -> dict[str, Any]:
    """Convert PatchSpec → flat dict for table/json rendering.

    Adds a derived ``production_default`` field that flags entries
    where ``default_on=True`` but the patch is only a marker
    (``implementation_status='marker_only'``) — these have no
    apply module, so the operator-facing label "Default-on" can
    mislead. Honest values:

      "applied"          default_on + full apply_module
      "marker"           default_on + marker_only (no runtime effect)
      "opt-in"           default_on=False
      "blocked"          implementation_status in {partial, placeholder}
                         or lifecycle in {retired, research}
    """
    impl = getattr(spec, "implementation_status", "full") or "full"
    if impl in ("partial", "placeholder"):
        prod_default = "blocked"
    elif spec.lifecycle in ("retired", "research"):
        prod_default = "blocked"
    elif spec.default_on and impl == "marker_only":
        prod_default = "marker"
    elif spec.default_on:
        prod_default = "applied"
    else:
        prod_default = "opt-in"
    return {
        "patch_id": spec.patch_id,
        "tier": spec.tier,
        "lifecycle": spec.lifecycle,
        "family": spec.family,
        "default_on": spec.default_on,
        "production_default": prod_default,
        "implementation_status": impl,
        "env_flag": spec.env_flag or "",
        "upstream_pr": spec.upstream_pr,
        "title": (spec.title or "")[:80],
        "apply_module": spec.apply_module or "",
    }


# ─── `sndr patches list` ─────────────────────────────────────────────────


def _run_list(opts: argparse.Namespace) -> int:
    from vllm.sndr_core.dispatcher.spec import iter_patch_specs

    rows: list[dict[str, Any]] = []
    for spec in iter_patch_specs():
        if not _matches_filters(
            spec,
            tier=opts.tier,
            lifecycle=opts.lifecycle,
            family=opts.family,
            default_on=(True if opts.default_on
                        else (False if opts.opt_in else None)),
            has_upstream=(True if opts.has_upstream
                          else (False if opts.no_upstream else None)),
        ):
            continue
        rows.append(_spec_to_row(spec))

    rows.sort(key=lambda r: r["patch_id"])

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
    from vllm.sndr_core.dispatcher import PATCH_REGISTRY
    from vllm.sndr_core.dispatcher.spec import patch_spec_for

    pid = opts.patch_id
    meta = PATCH_REGISTRY.get(pid)
    if meta is None:
        # Try case-insensitive lookup so operators can type `p67` or `pn82`.
        for key in PATCH_REGISTRY:
            if key.lower() == pid.lower():
                meta = PATCH_REGISTRY[key]
                pid = key
                break
    if meta is None:
        _io.error(f"patch_id {pid!r} not found in PATCH_REGISTRY")
        # Helpful suggestion: list closest prefix matches
        prefix = pid[:2].upper()
        candidates = sorted(
            k for k in PATCH_REGISTRY if k.startswith(prefix)
        )[:8]
        if candidates:
            _io.info(f"did you mean: {', '.join(candidates)}")
        return 2

    spec = patch_spec_for(pid, meta)

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

    # Live decision — if dispatcher can run on this host, show what it
    # would decide right now. Wrapped in try because should_apply pulls in
    # model_detect / config_detect which may fail on Mac / no-vllm hosts.
    _io.info("")
    try:
        from vllm.sndr_core.dispatcher import should_apply
        applied, reason = should_apply(pid)
        verdict = "APPLY" if applied else "SKIP"
        _row("Live decision", f"{verdict} — {reason}")
    except Exception as e:
        _row("Live decision", f"(unavailable: {type(e).__name__})")

    return 0


# ─── `sndr patches pn95-status` ─────────────────────────────────────────


_PN95_STATUS_HINTS = (
    # (predicate, severity, message)
    (
        lambda s: s["ticks_total"] == 0,
        "warn",
        "Zero scheduler ticks recorded. Likely SITE5 anchor missed the "
        "vllm Scheduler.schedule() entry — re-apply via "
        "`python3 -m vllm.sndr_core.apply` after a fresh container boot.",
    ),
    (
        lambda s: s["ticks_pressure_check"] > 0 and s["ticks_demote_triggered"] == 0 and s["blocks_demoted_total"] == 0,
        "warn",
        "Pressure checks running but no demotes ever fired. Most common "
        "cause is the multiproc gap: scheduler_tick runs in EngineCore "
        "process whose _PN95_BLOCK_POOL_REFS is empty (pools live in "
        "Worker processes). The fall-through eviction-driven path "
        "(SITE7 demote-on-evict) still works on natural vllm prefix "
        "eviction. To get proactive coverage call "
        "vllm.sndr_core.cache._pn95_runtime.worker_side_proactive_demote "
        "from a Worker-side hook (BlockPool.get_new_blocks or similar).",
    ),
    (
        lambda s: s.get("last_free_mib", -1) >= 0 and s.get("last_free_mib", 0) < 200,
        "warn",
        "GPU free memory below 200 MiB — kernel scratch allocations "
        "(Marlin GEMM, FlashAttention) are at risk of CUDA OOM. PN95 "
        "manages KV-cache bytes only; this looks like an activation-buffer "
        "budget issue. Lower --gpu-memory-utilization (e.g. 0.92 → 0.88) "
        "or reduce --max-num-batched-tokens.",
    ),
    (
        lambda s: s["blocks_demoted_total"] > 0 and s["prefix_store_entries"] == 0,
        "warn",
        "Demote counter incremented but prefix store is empty — the CPU "
        "slab eviction TTL may be too short, or compression is dropping "
        "entries. Inspect _PN95_PREFIX_STORE in a worker REPL.",
    ),
    (
        lambda s: s["prefix_store_promote_hits"] > 0,
        "ok",
        'Prefix store is actively serving cache hits — multi-turn '
        'workloads are benefiting from CPU offload.',
    ),
)


def _run_pn95_status(opts: argparse.Namespace) -> int:
    import os
    path = getattr(opts, "stats_file", "/tmp/pn95_stats.json")
    if not os.path.isfile(path):
        msg = (
            f"PN95 stats file not found at {path}. "
            "Either PN95 is not enabled in this deployment "
            "(GENESIS_ENABLE_PN95_TIER_AWARE_CACHE=1) or the worker "
            "hasn't dumped stats yet — stats land every "
            "GENESIS_PN95_STATS_INTERVAL ticks (default 100)."
        )
        if opts.json:
            print(json.dumps({"available": False, "reason": msg}, indent=2))
        else:
            _io.warn(msg)
        return 1
    try:
        with open(path, "r") as fh:
            stats = json.load(fh)
    except (OSError, ValueError) as e:
        if opts.json:
            print(json.dumps(
                {"available": False, "reason": f"parse error: {e}"},
                indent=2,
            ))
        else:
            _io.warn(f"PN95 stats file at {path} is not parseable: {e}")
        return 2

    hints: list[dict] = []
    for predicate, severity, msg in _PN95_STATUS_HINTS:
        try:
            hit = predicate(stats)
        except (KeyError, TypeError):
            hit = False
        if hit:
            hints.append({"severity": severity, "message": msg})

    # Disk-tier stats are best-effort: the module reads env at first
    # access; we can probe it without forcing init when disabled.
    disk_stats: dict = {}
    try:
        from vllm.sndr_core.cache import _pn95_disk_tier as _dt
        disk_stats = _dt.disk_tier_stats()
    except Exception as e:
        disk_stats = {"error": str(e)}

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
    from vllm.sndr_core.dispatcher import (
        PATCH_REGISTRY,
        validate_registry,
    )
    from vllm.sndr_core.dispatcher.spec import (
        validate_apply_module_coverage,
    )

    issues = validate_registry()
    coverage = validate_apply_module_coverage()

    if opts.json:
        print(json.dumps({
            "registry_size": len(PATCH_REGISTRY),
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
               f"{len(PATCH_REGISTRY)} entries, "
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

    # accept either V1 monolithic key or V2 alias
    # so `sndr patches plan --preset prod-35b` works alongside the legacy
    # `--preset a5000-2x-35b-prod`. memory.py already exports the same
    # resolver — re-use to avoid divergent lookup paths.
    try:
        from vllm.sndr_core.cli.memory import _resolve_preset_v1_or_v2
        cfg = _resolve_preset_v1_or_v2(opts.preset)
    except Exception as e:
        _io.fatal(f"preset {opts.preset!r} not found ({e})", 2)
    if cfg is None:
        _io.fatal(f"preset {opts.preset!r} not found", 2)

    # Snapshot current env, overlay preset's env, run should_apply, restore.
    overlay: dict[str, str] = {}
    overlay.update(getattr(cfg, "system_env", {}) or {})
    overlay.update(getattr(cfg, "genesis_env", {}) or {})

    saved: dict[str, Optional[str]] = {}
    for k, v in overlay.items():
        saved[k] = os.environ.get(k)
        os.environ[k] = str(v)

    apply_rows: list[dict[str, Any]] = []
    skip_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    try:
        from vllm.sndr_core.dispatcher import (
            PATCH_REGISTRY,
            should_apply,
        )
        for pid in sorted(PATCH_REGISTRY):
            meta = PATCH_REGISTRY[pid]
            if not isinstance(meta, dict):
                continue
            try:
                applied, reason = should_apply(pid)
            except Exception as e:
                error_rows.append({
                    "patch_id": pid,
                    "title": meta.get("title", ""),
                    "tier": meta.get("tier", ""),
                    "error": f"{type(e).__name__}: {e}",
                })
                continue
            row = {
                "patch_id": pid,
                "title": (meta.get("title") or "")[:80],
                "tier": meta.get("tier", "community"),
                "default_on": bool(meta.get("default_on", False)),
                "lifecycle": meta.get("lifecycle"),
                "reason": reason[:160],
            }
            if applied:
                apply_rows.append(row)
            else:
                skip_rows.append(row)
    finally:
        # Restore env
        for k, prev in saved.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev

    # Production-profile gate. `partial`/`placeholder` impl_status =
    # wiring stub only; `research`/`retired` lifecycle = should not reach
    # production. Opt-in via `--profile production`.
    profile = getattr(opts, "profile", "any")
    profile_violations: list[dict[str, Any]] = []
    if profile == "production":
        forbidden_status = {"partial", "placeholder"}
        forbidden_lifecycle = {"research", "retired"}
        for r in apply_rows:
            meta = PATCH_REGISTRY.get(r["patch_id"]) or {}
            impl = meta.get("implementation_status")
            lc = meta.get("lifecycle")
            reasons = []
            if impl in forbidden_status:
                reasons.append(f"implementation_status={impl}")
            if lc in forbidden_lifecycle:
                reasons.append(f"lifecycle={lc}")
            if reasons:
                profile_violations.append({
                    "patch_id": r["patch_id"],
                    "title": r["title"],
                    "reasons": reasons,
                })

    # optional patch_plan resolver layer.
    # Decoupled from the dispatcher simulator above — operators can
    # request both views in one JSON payload by passing --policy.
    #
    # even when --policy is NOT
    # passed, we still run the resolver under compat just to collect
    # advisory warnings (conflicts_with + candidate_when). These are
    # surfaced as `resolver_warnings` in JSON output and as a
    # standalone "⚠ advisory" block in human output, so legacy
    # operators see misconfigurations they'd otherwise miss.
    resolver_payload = None
    advisory_warnings: tuple[str, ...] = ()
    policy = getattr(opts, "policy", None)

    if policy is None:
        try:
            from vllm.sndr_core.model_configs.patch_plan import (
                resolve_patch_plan,
            )
            advisory_plan = resolve_patch_plan(cfg, policy="compat")
            advisory_warnings = advisory_plan.warnings
        except Exception:
            # Resolver failure must not break the simulator output —
            # advisory layer is best-effort.
            advisory_warnings = ()

    if policy is not None:
        from vllm.sndr_core.model_configs.patch_plan import (
            resolve_patch_plan,
        )
        plan = resolve_patch_plan(cfg, policy=policy)
        explain = bool(getattr(opts, "explain", False))

        def _decision_dict(d) -> dict[str, Any]:
            base = {
                "patch_id": d.patch_id,
                "env_flag": d.env_flag,
                "value": d.value,
                "decision": d.decision,
                "role": d.role,
                "reason": d.reason,
            }
            if explain:
                base["note"] = d.note
                base["bench_evidence"] = d.bench_evidence
            return base

        resolver_payload = {
            "policy": plan.policy,
            "included": [_decision_dict(d) for d in plan.included],
            "excluded": [_decision_dict(d) for d in plan.excluded],
            "warnings": list(plan.warnings),
            # Non-toggle parameter keys (GENESIS_BUFFER_MODE,
            # GENESIS_PN95_CONFIG_KEY, …) pass through every policy
            # so dependent patches don't silently noop. Expose
            # separately for diff tools + traceability.
            "passthrough": dict(plan.passthrough),
            "env": plan.env,
        }

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
    from vllm.sndr_core.dispatcher import PATCH_REGISTRY
    from vllm.sndr_core.dispatcher.spec import iter_patch_specs

    merged_upstream: list[dict[str, Any]] = []
    has_upstream_pr: list[dict[str, Any]] = []

    for spec in iter_patch_specs():
        meta = PATCH_REGISTRY.get(spec.patch_id) or {}
        if spec.lifecycle == "merged_upstream":
            merged_upstream.append({
                "patch_id": spec.patch_id,
                "title": (spec.title or "")[:80],
                "upstream_pr": spec.upstream_pr,
                "credit": meta.get("credit", ""),
            })
            continue
        if spec.upstream_pr:
            has_upstream_pr.append({
                "patch_id": spec.patch_id,
                "title": (spec.title or "")[:80],
                "upstream_pr": spec.upstream_pr,
                "lifecycle": spec.lifecycle,
                "default_on": spec.default_on,
            })

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


# Bundle catalog mirrors tests/bundles/test_stage7_bundles_smoke.py.
# Kept here as a typed list because there's no runtime BUNDLES catalog
# in `vllm/sndr_core/bundles/__init__.py` yet — module imports are
# ordered by file, but that's not a stable enumeration source for CLI.
_BUNDLES: list[tuple[str, str, str, str]] = [
    # (module_name, umbrella_flag, tier, description)
    (
        "tool_parsing_qwen3coder",
        "BUNDLE_TOOL_PARSING_QWEN3CODER",
        "community",
        "P15 + P61c + P64(×2) + PN56 — Qwen3-coder tool-parser fixes.",
    ),
    (
        "reasoning_qwen3",
        "BUNDLE_REASONING_QWEN3",
        "community",
        "P12 + P27 + P59 + P61 + P61b + PN51 — Qwen3 reasoning parser.",
    ),
    (
        "attention_gdn_spec",
        "BUNDLE_ATTENTION_GDN_SPEC",
        "community",
        "P60 + P60b — GDN spec-decode pipeline atomic apply.",
    ),
    (
        "attention_tq_multi_query",
        "BUNDLE_ATTENTION_TQ_MULTI_QUERY",
        "community",
        "P67 + P67b — TQ multi-query kernel + spec verify routing.",
    ),
    (
        "spec_decode_async_cleanup",
        "BUNDLE_SPEC_DECODE_ASYNC_CLEANUP",
        "community",
        "P79b + P79c + P79d — async cleanup of spec-decode artifacts.",
    ),
]


def _run_bundles_list(opts: argparse.Namespace) -> int:
    if opts.json:
        print(json.dumps([
            {
                "name": name,
                "umbrella_flag": flag,
                "tier": tier,
                "description": desc,
            }
            for name, flag, tier, desc in _BUNDLES
        ], indent=2))
        return 0

    _io.banner("SNDR Bundles", f"{len(_BUNDLES)} atomic multi-patch orchestrators")
    for name, flag, tier, desc in _BUNDLES:
        _io.info(f"  • {name}  [{tier}]")
        _io.info(f"      flag:  SNDR_ENABLE_{flag}=1")
        _io.info(f"      desc:  {desc}")
        _io.info("")
    return 0


def _run_bundles_explain(opts: argparse.Namespace) -> int:
    target = opts.name
    matches = [b for b in _BUNDLES if b[0] == target]
    if not matches:
        _io.error(f"bundle {target!r} not found")
        _io.info(f"available: {', '.join(b[0] for b in _BUNDLES)}")
        return 2
    name, flag, tier, desc = matches[0]

    # Try import + describe its component patches via reflection.
    try:
        mod = __import__(
            f"vllm.sndr_core.bundles.{name}", fromlist=["apply"]
        )
        has_apply = callable(getattr(mod, "apply", None))
    except Exception as e:
        mod = None
        has_apply = False
        _io.warn(f"  bundle module failed to import: {type(e).__name__}: {e}")

    if opts.json:
        print(json.dumps({
            "name": name,
            "umbrella_flag": flag,
            "tier": tier,
            "description": desc,
            "module": f"vllm.sndr_core.bundles.{name}",
            "has_apply": has_apply,
        }, indent=2))
        return 0

    _io.banner(f"Bundle: {name}", desc[:60])
    _io.info(f"  Tier:           {tier}")
    _io.info(f"  Umbrella flag:  SNDR_ENABLE_{flag}=1  (or GENESIS_ENABLE_{flag}=1)")
    _io.info(f"  Module:         vllm.sndr_core.bundles.{name}")
    _io.info(f"  apply():        {'callable' if has_apply else 'MISSING'}")
    _io.info("")
    _io.info("  Description:")
    _io.info(f"    {desc}")
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
    from vllm.sndr_core.proof import DEFAULT_PROOF_DIR

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_PROOF_DIR

    if args.dead_detect:
        return _run_prove_dead_detect(args, out_dir)
    if args.prove_all:
        return _run_prove_all(args, out_dir)
    if not args.patch_id:
        from vllm.sndr_core.cli import _io
        _io.warn("provide a patch_id, or use --all / --dead-detect")
        return 2
    return _run_prove_one(args, out_dir)


def _run_prove_one(args: argparse.Namespace, out_dir) -> int:
    """`sndr patches prove <id>` — verify one patch + write artefact."""
    import json as _json
    from vllm.sndr_core.proof import (
        build_proof_for_patch, write_proof_artefact,
    )

    proof = build_proof_for_patch(args.patch_id)
    if not args.no_write and proof.static_checks and proof.static_checks[0].passed:
        # Only write artefact when P-1 (patch in registry) passes — otherwise
        # we'd be persisting "patch not found" as evidence.
        path = write_proof_artefact(proof, out_dir)
    else:
        path = None

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
    from vllm.sndr_core.proof import (
        build_proof_for_patch, write_proof_artefact,
    )
    from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY

    results = []
    passed = 0
    for patch_id in PATCH_REGISTRY:
        proof = build_proof_for_patch(patch_id)
        if not args.no_write and proof.static_checks[0].passed:
            write_proof_artefact(proof, out_dir)
        ok = proof.static_passed
        if ok:
            passed += 1
        results.append({
            "patch_id": patch_id,
            "passed": ok,
            "errors": [
                {"rule": c.rule, "message": c.message}
                for c in proof.static_errors
            ],
        })

    total = len(results)
    coverage = (passed / total) if total else 1.0

    if args.json:
        print(_json.dumps({
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "coverage_pct": round(coverage * 100, 1),
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
    from vllm.sndr_core.proof import list_dead_patches
    from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY

    dead = list_dead_patches(out_dir=out_dir)
    total = len(PATCH_REGISTRY)
    proven = total - len(dead)
    coverage = (proven / total) if total else 1.0

    if args.json:
        print(_json.dumps({
            "total_patches": total,
            "proven": proven,
            "dead": len(dead),
            "coverage_pct": round(coverage * 100, 1),
            "dead_patches": dead,
        }, indent=2, sort_keys=True))
        return 0

    print(f"sndr patches prove --dead-detect   ({total} patches in registry)")
    print("─" * 70)
    print(f"  proven (has passing static artefact): {proven}")
    print(f"  dead   (no passing artefact):         {len(dead)}")
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
    from vllm.sndr_core.cli import _io
    from vllm.sndr_core.proof import DEFAULT_PROOF_DIR, load_proof_artefact
    from vllm.sndr_core.proof.bench_attach import (
        BenchAttachError, attach_bench,
    )

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_PROOF_DIR
    bench = Path(args.bench_path)
    baseline = Path(args.baseline) if args.baseline else None

    try:
        target = attach_bench(
            args.patch_id, bench,
            baseline_path=baseline,
            out_dir=out_dir,
        )
    except BenchAttachError as e:
        _io.warn(str(e))
        return 2

    data = load_proof_artefact(target)
    bench_delta = data.get("bench_delta", {}) or {}

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
    from vllm.sndr_core.proof import (
        DEFAULT_PROOF_DIR, PROOF_STATUS_BUCKETS, summarize_proof_status,
    )

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_PROOF_DIR
    summary = summarize_proof_status(out_dir=out_dir)

    # Optional bucket filter.
    filter_set: Optional[set[str]] = None
    if args.bucket:
        unknown = [b for b in args.bucket if b not in PROOF_STATUS_BUCKETS]
        if unknown:
            from vllm.sndr_core.cli import _io
            _io.warn(
                f"unknown bucket(s): {unknown!r}. "
                f"Valid: {list(PROOF_STATUS_BUCKETS)}"
            )
            return 2
        filter_set = set(args.bucket)
        filtered = [p for p in summary["patches"] if p["bucket"] in filter_set]
    else:
        filtered = summary["patches"]

    if args.json:
        payload = {
            "total": summary["total"],
            "counts": summary["counts"],
            "patches": filtered,
            "filter_buckets": sorted(filter_set) if filter_set else None,
        }
        print(_json.dumps(payload, indent=2, sort_keys=True))
        return 0

    counts = summary["counts"]
    total = summary["total"]
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
    from vllm.sndr_core.cli import _io
    from vllm.sndr_core.proof import DEFAULT_PROOF_DIR
    from vllm.sndr_core.proof.release_check import (
        ReleaseCheckError, ReleasePolicy, evaluate_release,
    )

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_PROOF_DIR

    # Build the patch filter: --patch is the explicit operator-driven
    # list; --scope production-subset programmatically widens it to the
    # canonical hardened-release scope (every patch enabled by any prod-*
    # preset + default_on=True entries). When both are given, --patch
    # wins (operator override).
    patch_filter = frozenset(args.patch) if args.patch else None
    if patch_filter is None and getattr(args, "scope", "all") == "production-subset":
        from vllm.sndr_core.proof.production_subset import get_production_subset
        patch_filter = get_production_subset()

    try:
        policy = ReleasePolicy(
            mode=args.mode,
            max_regression_pct=args.max_regression_pct,
            patch_filter=patch_filter,
            tier_filter=frozenset(args.tier) if args.tier else None,
        )
    except ReleaseCheckError as e:
        _io.warn(str(e))
        return 2

    report = evaluate_release(policy, out_dir=out_dir)

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
