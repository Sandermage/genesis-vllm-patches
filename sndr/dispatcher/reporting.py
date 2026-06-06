# SPDX-License-Identifier: Apache-2.0
"""SNDR Core dispatcher — apply matrix + structured boot summary.

Reporting helpers that turn the registry decision tree into:
  - get_apply_matrix() — list of dicts with per-patch status
  - dump_apply_matrix() — ASCII table for CLI
  - dump_structured_boot_summary() — multi-tier diagnostic output
  - log_apply_matrix() / log_structured_boot_summary() — logger-targeted

Used by:
  - `apply/orchestrator.run()` — emit summary at boot
  - CLI `sndr doctor`, `sndr explain`, `sndr list-patches`

Migration history:
  - Original location: vllm/_genesis/dispatcher.py (Stage 0).
  - Stage 3 (CURRENT): split into dispatcher/reporting.py.
"""
from __future__ import annotations

import logging
from typing import Any

from .decision import _DECISIONS, _live_registry
from .registry import PATCH_REGISTRY  # noqa: F401  (re-exported)

# Logger name kept as `genesis.dispatcher` for back-compat with operator
# log filters. Stage 12 (brand swap) will introduce dual-emit to both
# `genesis.*` and `sndr_core.*` once operator migration paths are defined.
log = logging.getLogger("genesis.dispatcher")


def get_apply_matrix() -> list[dict[str, Any]]:
    """Return the recorded apply matrix for this boot.

    Useful for tests + diagnostic dump.
    """
    return list(_DECISIONS)


def dump_apply_matrix() -> str:
    """Format the apply matrix as ASCII table (string for printing).

    Columns: patch_id, status, title, reason (truncated), credit.
    """
    if not _DECISIONS:
        return "(no decisions recorded — Genesis Dispatcher hasn't been used yet)"

    # Compute column widths
    rows = [
        (
            d["patch_id"],
            "APPLY" if d["applied"] else "SKIP",
            d["title"][:45],
            d["reason"][:60],
            d.get("credit", "")[:30],
        )
        for d in _DECISIONS
    ]
    widths = [max(len(r[i]) for r in rows) for i in range(5)]
    widths = [max(w, len(h)) for w, h in zip(widths,
              ["Patch", "Status", "Title", "Reason", "Credit"])]

    def _fmt_row(r):
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(r))

    lines = []
    lines.append(_fmt_row(["Patch", "Status", "Title", "Reason", "Credit"]))
    lines.append("-+-".join("-" * w for w in widths))
    for r in rows:
        lines.append(_fmt_row(r))
    return "\n".join(lines)


def log_apply_matrix() -> None:
    """Emit the apply matrix as a multi-line INFO block.

    Called by apply_all at end of boot to give operator a single readable
    summary instead of grep-ing through scattered INFO lines.
    """
    matrix = dump_apply_matrix()
    log.info(
        "[Genesis Dispatcher v2] apply matrix:\n%s",
        matrix,
    )


def dump_structured_boot_summary() -> str:
    """Emit a structured, table-formatted boot summary.

    Sections:
      1. System info — GPU, vllm pin, Genesis version, model class
      2. Per-category APPLY/SKIP/FAIL counters
      3. APPLIED patches table (grouped by category)
      4. SKIPPED patches (grouped by reason class: env-disabled / model-incompat
         / upstream-merged / conflict / other)
      5. FAILED patches (highlighted, none expected in healthy boot)
      6. Active warnings (regression-flagged enabled patches)

    Designed for readability by operators who tail container logs. Replaces
    the scattered per-patch INFO lines + the bare apply matrix.
    """
    if not _DECISIONS:
        return "(no Genesis decisions recorded — patcher not active or first call)"

    # Dedup: keep last decision per patch_id (handles multi-worker boot
    # where apply_all runs once per TP rank — the second call typically
    # logs `already applied (idempotent)` for the same patches).
    _seen: dict[str, dict[str, Any]] = {}
    for d in _DECISIONS:
        _seen[d["patch_id"]] = d
    decisions = list(_seen.values())

    lines: list[str] = []

    # ─── 1. System info header ────────────────────────────────────────────
    lines.append("═" * 78)
    lines.append("Genesis vLLM Patcher — boot summary")
    lines.append("═" * 78)

    # Genesis version
    try:
        from vllm.sndr_core.version import __version__ as _gver
        _gver_str = _gver.lstrip("v")  # avoid "vv7.63.x" if module already prefixes
        lines.append(f"  Genesis:  v{_gver_str}")
    except Exception:
        lines.append("  Genesis:  (version unavailable)")

    # vllm pin
    try:
        import vllm as _vllm
        lines.append(f"  vLLM:     {getattr(_vllm, '__version__', 'unknown')}")
    except Exception:
        lines.append("  vLLM:     (import failed)")

    # GPU + compute capability
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            lines.append(
                f"  GPU:      {n}× {gpu_name} (sm_{cap[0]}{cap[1]})"
            )
    except Exception:
        pass

    # Model profile (if loaded)
    try:
        from vllm.sndr_core.detection.model_detect import get_model_profile
        profile = get_model_profile()
        if profile.get("resolved", False):
            mc = profile.get("model_class", "unknown")
            qf = profile.get("quant_format", "unknown")
            kv = profile.get("kv_cache_dtype", "unknown")
            hyb = "hybrid" if profile.get("hybrid") else "dense"
            lines.append(
                f"  Model:    {mc} | quant={qf} | kv={kv} | {hyb}"
            )
    except Exception:
        pass

    # ─── 2. Counters ──────────────────────────────────────────────────────
    n_apply = sum(1 for d in decisions if d["applied"])
    n_skip = sum(1 for d in decisions if not d["applied"])
    lines.append("─" * 78)
    lines.append(
        f"  Patches:  {len(decisions)} total  →  "
        f"{n_apply} APPLY  |  {n_skip} SKIP"
    )

    # Per-category breakdown
    cat_counts: dict[str, dict[str, int]] = {}
    for d in decisions:
        meta = _live_registry().get(d["patch_id"], {})
        cat = meta.get("category", "uncategorized")
        bucket = cat_counts.setdefault(cat, {"apply": 0, "skip": 0})
        bucket["apply" if d["applied"] else "skip"] += 1

    if cat_counts:
        lines.append("  By category:")
        for cat in sorted(cat_counts):
            c = cat_counts[cat]
            lines.append(
                f"    • {cat:<22} APPLY={c['apply']:>3}  SKIP={c['skip']:>3}"
            )

    # ─── Pretty category labels ──────────────────────────────────────────
    # Friendly human-readable description for each registry category.
    CATEGORY_LABELS = {
        "compile_safety":      "Compile / cudagraph safety",
        "hybrid":              "Hybrid GDN / Mamba (qwen3_5/3_6)",
        "kernel":              "Kernel correctness (Marlin / TQ)",
        "kernel_perf":         "Kernel performance tuning",
        "kernel_safety":       "Kernel-level safety guards",
        "kv_cache":            "KV cache management",
        "memory_hotfix":       "Memory hotfix (Cliff 2 / OOM)",
        "memory_pool":         "Memory pool / scratch buffers",
        "memory_savings":      "Memory savings (defensive)",
        "model_correctness":   "Model correctness (load / dtype)",
        "perf_hotfix":         "Performance hotfix (defensive)",
        "perf_kernel":         "Performance kernel rewrite",
        "quantization":        "Quantization (AutoRound / FP8)",
        "request_middleware":  "Request middleware",
        "spec_decode":         "Speculative decoding (MTP / ngram)",
        "stability":           "Stability / DX safeguards",
        "structured_output":   "Structured output / Qwen3 parser",
        "uncategorized":       "Uncategorized",
    }

    def _cat_label(cat: str) -> str:
        return CATEGORY_LABELS.get(cat, cat)

    # ─── 3. APPLIED patches grouped by category ──────────────────────────
    applied_by_cat: dict[str, list[dict[str, Any]]] = {}
    for d in decisions:
        if not d["applied"]:
            continue
        meta = _live_registry().get(d["patch_id"], {})
        cat = meta.get("category", "uncategorized")
        applied_by_cat.setdefault(cat, []).append(d)

    if applied_by_cat:
        lines.append("─" * 78)
        lines.append(f"  ✓ APPLIED ({n_apply})")
        for cat in sorted(applied_by_cat):
            label = _cat_label(cat)
            count = len(applied_by_cat[cat])
            lines.append("")
            lines.append(f"  ╔═══ {label} ({count})")
            for d in applied_by_cat[cat]:
                upstream = ""
                meta = _live_registry().get(d["patch_id"], {})
                if meta.get("upstream_pr"):
                    upstream = f"  ←  vllm#{meta['upstream_pr']}"
                lines.append(
                    f"  ║   • {d['patch_id']:<10}  {d['title'][:90]}{upstream}"
                )

    # ─── 4. SKIPPED patches grouped by reason class ──────────────────────
    skip_classes = {
        "upstream_merged": [],
        "env_disabled": [],
        "model_incompat": [],
        "conflict": [],
        "other": [],
    }
    for d in decisions:
        if d["applied"]:
            continue
        reason = d["reason"].lower()
        if "upstream" in reason and ("merged" in reason or "drift" in reason):
            cls = "upstream_merged"
        elif "opt-in" in reason or "set genesis_enable" in reason:
            cls = "env_disabled"
        elif "applies_to" in reason or "incompatible" in reason or \
                "model_class" in reason or "no gdn" in reason:
            cls = "model_incompat"
        elif "conflict" in reason or "mutually exclusive" in reason or \
                "skipped — p" in reason:
            cls = "conflict"
        else:
            cls = "other"
        skip_classes[cls].append(d)

    SKIP_LABELS = {
        "upstream_merged": "Upstream merged in current pin (auto-skip)",
        "env_disabled":    "Opt-in (env flag disabled by operator)",
        "model_incompat":  "Model architecture incompatible (applies_to)",
        "conflict":        "Conflict / mutual-exclusion with active patch",
        "other":           "Other / config-neutral",
    }

    if n_skip > 0:
        lines.append("")
        lines.append("─" * 78)
        lines.append(f"  ⊘ SKIPPED ({n_skip}) — grouped by reason")
        for cls, items in skip_classes.items():
            if not items:
                continue
            label = SKIP_LABELS.get(cls, cls)
            lines.append("")
            lines.append(f"  ╔═══ {label} ({len(items)})")
            for d in items[:12]:  # cap per-class to keep summary readable
                lines.append(
                    f"  ║   • {d['patch_id']:<10}  {d['title'][:90]}"
                )
            if len(items) > 12:
                lines.append(f"  ║   … and {len(items) - 12} more")

    # ─── 5. FAILED (highlighted) ─────────────────────────────────────────
    failed = [d for d in decisions
              if not d["applied"] and "fail" in d["reason"].lower()]
    if failed:
        lines.append("─" * 78)
        lines.append(f"  ⚠ FAILED ({len(failed)}) — investigate before serving traffic")
        for d in failed:
            lines.append(
                f"    {d['patch_id']:<8}  {d['title'][:50]}"
            )
            lines.append(f"             reason: {d['reason'][:65]}")

    lines.append("═" * 78)
    return "\n".join(lines)


def log_structured_boot_summary() -> None:
    """Emit the structured boot summary as a single multi-line INFO block.

    Drop-in replacement for `log_apply_matrix()`. Called once at end of
    apply_all.run() boot. Operator-friendly: tables, counters, system info,
    grouped by category and skip-reason class.
    """
    summary = dump_structured_boot_summary()
    log.info(
        "[Genesis] structured boot summary:\n%s",
        summary,
    )


# ─── A3/D2 — PATCH_REGISTRY dependency / conflict validator ───────────────
# Two layers:
#   1. validate_registry()      — static structural check (boot-time)
#   2. validate_apply_plan(set) — runtime check on actual decisions
#
# Patch metadata may declare:
#   "requires_patches": ["P60"]      — list of patch_ids that must also apply
#   "conflicts_with":   ["P65"]      — list of patch_ids that MUST NOT apply
# Both fields default to [] when absent (no relationship declared).


