# SPDX-License-Identifier: Apache-2.0
"""SNDR Core dispatcher — typed `PatchSpec` over `PATCH_REGISTRY`.

PR38 Day 4 (2026-05-08): introduces a typed contract that lifts the
hand-rolled `_per_patch_dispatch.apply_patch_X` parking lot into a
data-driven structure. Every `PATCH_REGISTRY` entry maps to one
`PatchSpec`; the `apply_module` field is auto-derived by walking the
canonical `sndr/engines/vllm/patches/<family>/p<id>_*.py` tree.

Architecture
─────────────
The PR38 plan §5.2 calls for a single source of truth for patch
metadata + dispatch. Today the system has two:

  - `dispatcher.PATCH_REGISTRY`  — metadata (tier, family, lifecycle, ...)
  - `apply._per_patch_dispatch`   — 124 hand-written `apply_patch_X`
                                     functions calling `_wiring_text_patch(stem)`

This module exposes `iter_patch_specs()` which yields a `PatchSpec` per
registry entry with `apply_module` derived from the on-disk filename.
The next step (Day 6-8) is to teach `apply.orchestrator.run()` to
iterate specs directly, retiring `_per_patch_dispatch.py` as a parking
lot.

Usage
─────

    from sndr.dispatcher.spec import iter_patch_specs

    for spec in iter_patch_specs():
        if spec.apply_module is None:
            log.warning("patch %s has no apply_module", spec.patch_id)
            continue
        mod = importlib.import_module(spec.apply_module)
        result = mod.apply()
        ...

Author: Sandermage (Sander) Barzov Aleksandr.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

log = logging.getLogger("genesis.dispatcher.spec")


# Audit closure 2026-05-08 (P1-2 noonghunna): registry metadata
# enrichment fields. Inference defaults are derived from existing
# tier/family/lifecycle/upstream_pr when an entry doesn't set them
# explicitly. Schema validator warns on imprecise defaults so new
# patches are nudged to set them deliberately.

# `category` enumerates the operational subsystem the patch targets.
# Used by `sndr patches plan` and release notes to group patches.
#
# Audit closure 2026-05-08 (P1-2): the original audit suggested a 10-item
# canonical enum. The actual registry has been using a richer descriptive
# vocabulary for ~6 months (memory_savings, perf_hotfix, kernel_safety,
# etc.) so we ratify these as canonical here rather than force-rename
# 65+ existing entries. New patches should pick from this set.
VALID_CATEGORIES = (
    # Audit's canonical enum (kept as primary reference)
    "memory",
    "spec_decode",
    "structured_output",
    "quantization",
    "gdn",
    "moe",
    "launcher",
    "security",
    "observability",
    "research",
    "uncategorized",
    # Subsystem-flavoured (more granular than audit enum)
    "attention",
    "tool_parsing",
    "reasoning",
    "kernel",
    "kv_cache",
    "compile_safety",
    "loader",
    "scheduler",
    # Operational descriptors used in registry pre-audit
    "memory_savings",     # patches that free VRAM (PN12, PN17, PN77)
    "memory_pool",        # patches preallocating buffers (P22, P26, etc.)
    "memory_hotfix",      # P103-class single-card cliff fixes
    "kernel_perf",        # kernel-level perf wins (P67, P67b/c, P40)
    "kernel_safety",      # kernel-level correctness (P14, PN14)
    "perf_kernel",        # alias for kernel_perf (legacy entries)
    "perf_hotfix",        # narrow regression patches (PN29, PN51, …)
    "model_correctness",  # output-correctness fixes (P31, PN11, PN30)
    "request_middleware", # request-path hooks (PN16, PN65)
    "stability",          # boot/runtime stability guards (P34, PN61)
    "hybrid",             # GDN/Mamba hybrid path (PN32, PN59)
    "config_auto_tune",   # auto-tune config/profile from detected hw (PN296, PN302)
    # 2026-06-10 vendor-wave descriptors (PN346/PN347/PN349/PN362/PN364)
    "correctness",        # output/cache correctness fixes (PN346, PN347, PN349)
    "bench_methodology",  # bench reproducibility aids (PN362 autotune determinism)
    "ttft_warmup",        # startup warmup reducing first-request latency (PN364)
)

# `implementation_status` describes how the patch is wired into vllm.
# `live` and `text_patch` and `runtime_hook` are real; `metadata_only`
# / `retired` / `research` / `blocked` / `upstream_merged` are no-ops
# at apply time but still tracked in the registry.
VALID_IMPLEMENTATION_STATUSES = (
    "live",              # generic active patch (default fallback)
    "full",              # explicitly "fully wired" production-ready (stable lifecycle)
    "text_patch",        # text-edit on vllm source
    "runtime_hook",      # class/method monkey-patch
    "middleware",        # FastAPI/ASGI/logging middleware
    "metadata_only",     # informational entry, no apply
    "marker_only",       # alias of metadata_only — operator-facing doctor uses this name
    "placeholder",       # entry exists but apply path TBD
    "partial",           # honest "Phase 1 of 2" — instrumentation/wire-in present
                         # but full feature pending next phase. Used by PN95 v0.5.
    "scaffold",          # code is present but without production validation — research-tier
    "coordinator",       # bundles/forward-shim, no apply (P5b pattern)
    "experimental",      # impl exists but lacks A/B / cross-rig validation
    "retired",           # superseded by another patch
    "research",          # opt-in research path, not for PROD
    "blocked",           # known-broken / waiting on upstream
    "upstream_merged",   # auto-skip — vllm pin already has the fix
)

# `source` describes where the patch idea came from. Used in CREDITS
# and release-notes attribution.
VALID_SOURCES = (
    "genesis_original",
    "vllm_pr_backport",
    "club_3090_adapted",
    "cross_engine_research",
    # Phase 5.3.D (2026-05-22): "vendor_backport" captures Gemma 4
    # checkpoint-format adaptations that came from a downstream vendor
    # fork (Google's Gemma-specific path), not from the canonical vllm
    # repo. Distinct provenance from `vllm_pr_backport` (=upstream
    # vllm PR). Four current consumers: G4_04 / G4_05 / G4_06 / G4_18
    # (Gemma 4 AWQ MoE keys remap + sibling loader-class entries).
    "vendor_backport",
)

# `upstream_pr_relationship` describes the SEMANTIC relationship between
# the Genesis patch and the upstream PR it cites via `upstream_pr`. The
# audit script (`audit_upstream_status.py`) uses this to route the patch
# to the correct decision bucket (retire candidate vs. waived vs. WATCH).
#
# Schema rule (enforced after Phase 5.1.C cleanup):
#   * REQUIRED when `upstream_pr` is an integer.
#   * FORBIDDEN when `upstream_pr` is None.
# During the Phase 5.1.A migration window, missing field is treated as
# implicit `"backport"` for back-compat.
#
# Introduced 2026-05-22 (Phase 5.1.A) — see
# `sndr_private/planning/audits/PHASE_5_1_RELATIONSHIP_SCHEMA_DESIGN_2026-05-22_RU.md`
# for the full design rationale, migration set, and audit-routing changes.
#
# Extension protocol: to add a new relationship value, follow the 8-step
# protocol in `scripts/audit_upstream_status.py` →
# `_PURE_UPSTREAM_RELATIONSHIPS` docstring. The minimum changes are
# (a) appending the new value to this tuple with a trailing one-line
# semantic comment, (b) deciding whether the value belongs in
# `_PURE_UPSTREAM_RELATIONSHIPS` (status-based retire eligible), and
# (c) adding a test case that pins the resulting bucket routing.
VALID_UPSTREAM_PR_RELATIONSHIPS = (
    "backport",                  # Genesis mirrors upstream (default)
    "counter_regression",        # Genesis corrects a regression introduced by the cited PR
    "intentional_inverse",       # Genesis deliberately reverses cited PR's behavior for our shape
    "enables_upstream",          # Genesis turns on the upstream feature on opt-in
    "related_not_superseding",   # Genesis lives at a different layer; coverage doesn't overlap
    "defensive_overlay",         # Genesis is a defensive lower-layer guard alongside upstream's primary fix
)

# Mapping from `family` (registry field) to canonical `category`.
# Families not listed default to "uncategorized" + warning.
_FAMILY_TO_CATEGORY: dict[str, str] = {
    "attention": "attention",
    "attention.gdn": "gdn",
    "attention.turboquant": "quantization",
    "attention.kv": "kv_cache",
    "spec_decode": "spec_decode",
    "spec-decode": "spec_decode",
    "structured_output": "structured_output",
    "tool_parsing": "tool_parsing",
    "reasoning": "reasoning",
    "moe": "moe",
    "kernel": "kernel",
    "kernels": "kernel",
    "kv_cache": "kv_cache",
    "memory": "memory",
    "compile_safety": "compile_safety",
    "loader": "loader",
    "scheduler": "scheduler",
    "middleware": "observability",
    "serving": "observability",
    "multimodal": "memory",
    "lora": "loader",
    "quantization": "quantization",
    "worker": "kernel",
    "gdn": "gdn",
    # v11.3.0 BUG #12 fix: add 4 families that 29 patches use but the
    # original map (master plan §1.1 P0.2) didn't cover. Resolved
    # `infer_category()` → "uncategorized" warnings for 18 gemma4 + 4
    # observability + 4 streaming + 3 offload patches.
    "gemma4": "model_correctness",  # model-family specific compat patches
    "observability": "observability",  # logging/metrics/tracing patches
    "streaming": "kv_cache",  # PN200-203 streaming KV cache (per P0.2)
    "offload": "memory",  # P104/P105/PN102 CPU/disk offload (per P0.2)
    # v12 follow-up (2026-06-10): two families shipped without map
    # entries (caught by test_no_uncategorized_families):
    "detection": "config_auto_tune",  # PN296/PN302 arch+model profile
    #   boot initializers, PN300 autotune wrapper (its explicit
    #   category='kernel_perf' field still wins for PN300 itself).
    "model_compat": "model_correctness",  # root for model_compat.* —
    #   covers `model_compat.gemma4` (PN349) and future per-model
    #   subfamilies via the prefix fallback in infer_category().
}


def infer_category(family: str) -> str:
    """Derive `category` from `family` field. Returns "uncategorized"
    for unknown families so caller can warn."""
    if not family:
        return "uncategorized"
    # Exact match wins over prefix match (e.g. "attention.gdn" before "attention")
    if family in _FAMILY_TO_CATEGORY:
        return _FAMILY_TO_CATEGORY[family]
    # Family prefix fallback: "attention.<x>" → check root "attention"
    root = family.split(".", 1)[0]
    return _FAMILY_TO_CATEGORY.get(root, "uncategorized")


def infer_implementation_status(meta: dict, patch_id: str = "") -> str:
    """Derive `implementation_status` via the registry_metadata overlay.

    Delegates to `dispatcher.registry_metadata.derive_metadata`, which
    takes EXPLICIT_OVERRIDES + lifecycle + filesystem test detection
    into account. The old lifecycle-only fallback is kept for legacy
    call sites without a patch_id.
    """
    explicit = meta.get("implementation_status")
    if isinstance(explicit, str) and explicit:
        return explicit
    if patch_id:
        try:
            from sndr.dispatcher.registry_metadata import (
                derive_metadata,
            )
            return derive_metadata(patch_id, meta)["implementation_status"]
        except Exception:
            pass
    lifecycle = str(meta.get("lifecycle", "")).lower()
    if lifecycle in ("retired", "deprecated"):
        return "retired"
    if lifecycle == "research":
        return "research"
    if lifecycle == "stable":
        return "full"
    if lifecycle == "coordinator":
        return "coordinator"
    return "live"


def infer_source(meta: dict) -> str:
    """Derive `source` from upstream_pr presence + credit text.

    Rules:
      explicit `source` field          → respect verbatim
      `upstream_pr` is an int          → "vllm_pr_backport"
      `related_upstream_prs` non-empty → "vllm_pr_backport"
      "club-3090" / "noonghunna" in credit → "club_3090_adapted"
      "SGLang" / "TRT-LLM" / "llama.cpp" in credit → "cross_engine_research"
      else                              → "genesis_original"
    """
    explicit = meta.get("source")
    if isinstance(explicit, str) and explicit:
        return explicit
    if isinstance(meta.get("upstream_pr"), int):
        return "vllm_pr_backport"
    if meta.get("related_upstream_prs"):
        return "vllm_pr_backport"
    credit = str(meta.get("credit", "")).lower()
    if "club-3090" in credit or "noonghunna" in credit or "club3090" in credit:
        return "club_3090_adapted"
    if any(x in credit for x in ("sglang", "trt-llm", "tensorrt-llm",
                                  "llama.cpp", "tabby")):
        return "cross_engine_research"
    return "genesis_original"


@dataclass(frozen=True)
class PatchSpec:
    """Typed view over one `PATCH_REGISTRY` entry.

    Fields mirror the registry dict keys, plus a derived `apply_module`
    that points at the canonical patch implementation under
    `sndr.engines.vllm.patches.<family>.<filename>`. `apply_module=None`
    means the registry entry has no on-disk implementation (informational
    entries, legacy stubs, plugin entries).

    Audit closure 2026-05-08 (P1-2): added `category`,
    `implementation_status`, `source` so `sndr patches plan` can group
    entries by operational subsystem and provenance. Inference defaults
    derived from existing metadata when the entry doesn't set them
    explicitly.
    """
    patch_id: str
    title: str
    tier: str  # "community" | "engine"
    family: str  # subsystem family (e.g. "attention.gdn", "spec_decode")
    env_flag: Optional[str]
    default_on: bool
    lifecycle: str  # "stable" | "experimental" | "deprecated" | ...
    upstream_pr: Optional[int]
    apply_module: Optional[str]  # dotted path or None
    # Audit P1-2 enrichment fields:
    category: str = "uncategorized"
    implementation_status: str = "live"
    source: str = "genesis_original"
    applies_to: dict[str, Any] = field(default_factory=dict)
    requires_patches: tuple[str, ...] = field(default_factory=tuple)
    conflicts_with: tuple[str, ...] = field(default_factory=tuple)
    related_upstream_prs: tuple[int, ...] = field(default_factory=tuple)
    # Phase 5.1.A (2026-05-22) — relationship between Genesis patch and
    # the cited upstream_pr. Default `"backport"` for back-compat during
    # the migration window. After Phase 5.1.C cleanup the default will
    # be removed and the field will be REQUIRED when upstream_pr is set.
    upstream_pr_relationship: str = "backport"


# ─── apply_module derivation ──────────────────────────────────────────────


def _patch_ids_from_stem(stem: str) -> list[str]:
    """Extract canonical patch_id(s) from a filename stem.

    Returns a list because compound files like `p68_69_*` represent two
    distinct registry ids (P68 + P69) sharing one wiring module.

    Examples:
      `pn14_tq_decode_oob_clamp`         → ["PN14"]
      `p67_tq_multi_query_kernel`         → ["P67"]
      `p67b_spec_verify_routing`          → ["P67b"]
      `p68_69_long_ctx_tool_adherence`    → ["P68", "P69"]
    """
    if stem.startswith("__"):
        return []
    # Compound first: p<NUM>_<NUM>_*
    m_comp = re.match(r"^p(\d+[a-z]?)_(\d+[a-z]?)_", stem)
    if m_comp:
        return ["P" + m_comp.group(1), "P" + m_comp.group(2)]
    # Single: p[n]<digits>[<letter>]_
    m = re.match(r"^p(n)?(\d+[a-z]?)(?:_|$)", stem)
    if not m:
        return []
    is_pn = m.group(1) == "n"
    body = m.group(2)
    return [("PN" if is_pn else "P") + body]


def _patch_id_from_stem(stem: str) -> Optional[str]:
    """Back-compat: return the FIRST extracted patch_id, or None.

    Kept for callers that only want the canonical primary id (e.g. when
    rendering a per-module label). Internal code should prefer
    `_patch_ids_from_stem` for compound-aware handling.
    """
    ids = _patch_ids_from_stem(stem)
    return ids[0] if ids else None


# Explicit overrides for registry IDs whose canonical filename doesn't
# follow the `p<id>_*.py` / `pn<id>_*.py` convention. Examples:
#   - hyphenated registry keys (PN40-classifier shares pn40_workload_classifier_hook.py)
#   - registry sub-IDs that share their parent's file (PN26b → PN26's file)
_REGISTRY_ID_TO_STEM_OVERRIDES: dict[str, str] = {
    # PN40 has TWO files; the classifier sub-ID maps to the workload hook.
    "PN40-classifier": "pn40_workload_classifier_hook",
}


_APPLY_MODULE_MAP_CACHE: Optional[dict[str, str]] = None


def _resolve_patches_dir() -> Optional[Path]:
    """Locate the on-disk patches tree.

    Canonical (v12.0): ``sndr/engines/vllm/patches/`` — the relocated
    runtime-integration overlay tree. The pre-v12 repo-root
    ``vllm/sndr_core/integrations/`` (and older ``vllm/sndr_core/patches/``)
    shim tree is GONE; we no longer look there (legacy-path elimination
    2026-06-15 — nothing under the product should reference or create a
    ``vllm/sndr_core`` path). We anchor at the repo root (``parents[2]``)
    because the dotted module paths the caller derives are relative to it.

    Returns the resolved patches directory, or ``None`` if the canonical
    tree is absent (logs a warning so the empty map is diagnosable).
    """
    _repo_root = Path(__file__).resolve().parents[2]
    canonical_dir = _repo_root / "sndr" / "engines" / "vllm" / "patches"
    if canonical_dir.is_dir():
        return canonical_dir
    log.warning(
        "[PatchSpec] canonical patches dir not found at %s "
        "— apply_module map empty",
        canonical_dir,
    )
    return None


def _is_patch_impl_file(f: Path) -> bool:
    """Patch-impl files start with ``p<digit>`` or ``pn<digit>``;
    skip dunder modules + ``__pycache__`` artefacts + non-patch
    helpers like ``upstream_compat.py``."""
    if f.name.startswith("__") or "__pycache__" in f.parts:
        return False
    return re.match(r"^p(n)?\d", f.stem) is not None


def _register_variants(
    out: dict[str, str],
    duplicates: list[tuple[str, str, str]],
    pid: str,
    dotted: str,
) -> None:
    """Register the canonical patch id under every casing variant the
    registry might use (e.g. ``"P15B"`` vs ``"P15b"``). First writer
    wins; subsequent dotted-path collisions on the same variant are
    recorded in ``duplicates`` and surfaced via a DEBUG log."""
    for variant in {pid, pid.upper(), pid.lower(),
                    pid[0] + pid[1:].upper()}:
        if variant in out:
            if out[variant] != dotted:
                duplicates.append((variant, out[variant], dotted))
        else:
            out[variant] = dotted


def _walk_patch_impl_files(
    patches_dir: Path, repo_root: Path,
) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    """Primary map construction — walk the patches tree, derive
    ``patch_id``s from each impl file's stem, and register every
    casing variant. Returns ``(out, duplicates)``."""
    out: dict[str, str] = {}
    duplicates: list[tuple[str, str, str]] = []
    for f in sorted(patches_dir.rglob("*.py")):
        if not _is_patch_impl_file(f):
            continue
        pids = _patch_ids_from_stem(f.stem)
        if not pids:
            continue
        rel = f.relative_to(repo_root)
        dotted = ".".join(list(rel.parts[:-1]) + [f.stem])
        for pid in pids:
            _register_variants(out, duplicates, pid, dotted)
    return out, duplicates


def _build_stem_to_dotted_index(
    patches_dir: Path, repo_root: Path,
) -> dict[str, str]:
    """Build a ``stem → dotted_module_path`` lookup so explicit
    registry-id overrides can resolve to the right file without
    re-walking. Includes EVERY ``.py`` under ``patches_dir`` except
    dunder / cache artefacts, not only patch-impl files — overrides
    point at non-patch-impl modules sometimes (e.g. workload hooks)."""
    stem_to_dotted: dict[str, str] = {}
    for f in patches_dir.rglob("*.py"):
        if f.name.startswith("__") or "__pycache__" in f.parts:
            continue
        rel = f.relative_to(repo_root)
        stem_to_dotted[f.stem] = ".".join(list(rel.parts[:-1]) + [f.stem])
    return stem_to_dotted


def _apply_registry_id_overrides(
    out: dict[str, str], stem_to_dotted: dict[str, str],
) -> None:
    """Apply explicit ``registry_id → stem`` overrides for entries
    whose canonical filename doesn't follow the ``p<id>_*.py`` /
    ``pn<id>_*.py`` convention (e.g. ``PN40-classifier``). Stale
    overrides (target stem absent from the tree) surface as a
    WARNING."""
    for registry_id, stem in _REGISTRY_ID_TO_STEM_OVERRIDES.items():
        if stem in stem_to_dotted:
            out[registry_id] = stem_to_dotted[stem]
        else:
            log.warning(
                "[PatchSpec] override registry_id=%r → stem=%r: stem "
                "not found in patches/ — override is stale",
                registry_id, stem,
            )


def _build_apply_module_map() -> dict[str, str]:
    """Walk ``sndr/engines/vllm/patches/<family>/<file>.py`` and build
    a ``patch_id → dotted_module_path`` map. Cached on first call.

    M.1.1.T1.C restructure (2026-05-27): the original 85-LOC monolithic
    body is split into private helpers above. Cache semantics, walk
    order, dedup policy, override resolution, and log messages are
    preserved byte-identical;
    ``tests/unit/dispatcher/fixtures/spec_set.json`` +
    ``apply_module_coverage.json`` (228 + 17 entries) are the
    byte-identity guards.
    """
    global _APPLY_MODULE_MAP_CACHE
    if _APPLY_MODULE_MAP_CACHE is not None:
        return _APPLY_MODULE_MAP_CACHE

    patches_dir = _resolve_patches_dir()
    if patches_dir is None:
        out: dict[str, str] = {}
        _APPLY_MODULE_MAP_CACHE = out
        return out

    # Anchor at the repo root (same anchor `_resolve_patches_dir` builds
    # its candidates from) so the dotted path keeps the full package
    # prefix for every supported layout. The pre-v12 `parent.parent.parent`
    # walk-up assumed the 3-level `vllm/sndr_core/integrations` tree and
    # silently dropped the `sndr.` prefix on the 4-level
    # `sndr/engines/vllm/patches` layout.
    repo_root = Path(__file__).resolve().parents[2]

    out, duplicates = _walk_patch_impl_files(patches_dir, repo_root)

    # v12 moved retired impl files from ``patches/_retired/`` (covered by
    # the walk above) to the sibling ``engines/vllm/_archive/`` — keep
    # them in the coverage map so retired patch ids still resolve.
    archive_dir = patches_dir.parent / "_archive"
    if archive_dir.is_dir():
        archive_out, archive_dups = _walk_patch_impl_files(archive_dir, repo_root)
        for pid, dotted in archive_out.items():
            out.setdefault(pid, dotted)
        duplicates.extend(archive_dups)

    if duplicates:
        for pid, first, second in duplicates[:3]:
            log.debug(
                "[PatchSpec] duplicate apply_module for %s: %s vs %s "
                "(keeping first)", pid, first, second,
            )

    # Apply explicit overrides for registry-ID → stem mappings that
    # don't follow the auto-derived convention (e.g. PN40-classifier).
    stem_to_dotted = _build_stem_to_dotted_index(patches_dir, repo_root)
    _apply_registry_id_overrides(out, stem_to_dotted)

    _APPLY_MODULE_MAP_CACHE = out
    return out


def reset_apply_module_cache() -> None:
    """Drop the cached `_APPLY_MODULE_MAP_CACHE`. Used by tests that
    mutate the on-disk patches/ tree (e.g. add a synthetic patch) and
    need a re-walk on the next `iter_patch_specs()` call."""
    global _APPLY_MODULE_MAP_CACHE
    _APPLY_MODULE_MAP_CACHE = None


# ─── Spec construction from registry ──────────────────────────────────────


def _coerce_tuple(value: Any) -> tuple:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(value)
    if isinstance(value, str):
        return (value,)
    return ()


def patch_spec_for(
    patch_id: str,
    meta: dict[str, Any],
    apply_module_map: Optional[dict[str, str]] = None,
) -> PatchSpec:
    """Build one `PatchSpec` from a registry entry.

    Args:
        patch_id: registry key (e.g. "PN14")
        meta: registry value dict
        apply_module_map: optional override for testing. If None, the
            module-level map is used (lazy-built on first call).
    """
    if apply_module_map is None:
        apply_module_map = _build_apply_module_map()

    # Prefer explicit `apply_module` field on the entry if present
    # (registry data wins over auto-derivation).
    explicit = meta.get("apply_module")
    derived = apply_module_map.get(patch_id)
    apply_module = explicit if isinstance(explicit, str) else derived

    # Audit P1-2 enrichment: derive category/implementation_status/source
    # from existing fields when entry doesn't set them explicitly.
    family = str(meta.get("family", "uncategorized"))
    category = (
        meta.get("category")
        if isinstance(meta.get("category"), str) and meta.get("category")
        else infer_category(family)
    )
    implementation_status = infer_implementation_status(meta, patch_id)
    source = infer_source(meta)

    # Phase 5.1.C (2026-05-22) — derive upstream_pr_relationship.
    # All 72 upstream_pr-bearing entries carry an explicit value after
    # the 5.1.B migration. Entries without an explicit field default to
    # "backport" — which is meaningless when upstream_pr is None (the
    # 154 entries without an upstream link); the registry validator
    # catches "upstream_pr set without relationship" as an ERROR so
    # the next operator adding a backport sets the field explicitly.
    # The legacy `enables_upstream_feature: True` boolean fallback was
    # removed in 5.1.C — P75 and P99 carry the explicit
    # `upstream_pr_relationship: "enables_upstream"` field set in 5.1.B.
    rel_explicit = meta.get("upstream_pr_relationship")
    if isinstance(rel_explicit, str) and rel_explicit:
        upstream_pr_relationship = rel_explicit
    else:
        upstream_pr_relationship = "backport"

    return PatchSpec(
        patch_id=patch_id,
        title=str(meta.get("title", "")),
        tier=str(meta.get("tier", "community")),
        family=family,
        env_flag=meta.get("env_flag"),
        default_on=bool(meta.get("default_on", False)),
        lifecycle=str(meta.get("lifecycle", "stable")),
        upstream_pr=normalize_upstream_pr(meta.get("upstream_pr")),
        apply_module=apply_module,
        category=category,
        implementation_status=implementation_status,
        source=source,
        applies_to=meta.get("applies_to") or {},
        requires_patches=_coerce_tuple(meta.get("requires_patches")),
        conflicts_with=_coerce_tuple(meta.get("conflicts_with")),
        related_upstream_prs=_coerce_tuple(meta.get("related_upstream_prs")),
        upstream_pr_relationship=upstream_pr_relationship,
    )


_UPSTREAM_URL_PR_RE = re.compile(
    r"github\.com/vllm-project/vllm/pull/(\d+)"
)
_UPSTREAM_URL_ISSUE_RE = re.compile(
    r"github\.com/vllm-project/vllm/issues/(\d+)"
)


def normalize_upstream_pr(value: Any) -> Optional[int]:
    """Coerce `upstream_pr` registry value to `Optional[int]`.

    The registry stores three forms historically:
      - int (canonical, e.g. 41043)
      - str int ("41043") — accepted from JSON-typed YAML
      - URL str (e.g. "https://github.com/vllm-project/vllm/pull/40886"
        — 4 G4_* entries; or .../issues/39407 — 24 G4_* entries point
        at the issue tracking the problem, NOT at a PR)
      - None (Genesis-original, no upstream link)

    v11.3.0 BUG #13 fix: this normalizer extracts the PR number from
    `pull/<N>` URLs so consumers (audit-upstream-status, PatchSpec,
    docs generators) see a clean int. Issue URLs (no PR number) are
    deliberately returned as None — issue ≠ PR; the registry should
    use `related_upstream_prs` for tracking-by-issue, or a separate
    `upstream_issue` field. Existing 24 issue-URL entries are flagged
    by `test_no_issue_url_in_upstream_pr_field` (advisory).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None  # bool is technically int subclass — exclude
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.isdigit():
            return int(s)
        m = _UPSTREAM_URL_PR_RE.search(s)
        if m:
            return int(m.group(1))
        # Issue URL → not a PR, return None (advisory test surfaces this)
        return None
    return None


def is_issue_url_not_pr(value: Any) -> bool:
    """Return True if `value` is a github.com issue URL — i.e. it
    points at an issue rather than a PR. Used by the BUG #13 audit
    to surface advisory warnings."""
    if not isinstance(value, str):
        return False
    return bool(_UPSTREAM_URL_ISSUE_RE.search(value.strip()))


def _topological_order(
    registry: dict[str, dict[str, Any]],
) -> list[str]:
    """Return registry IDs in a stable topological order respecting
    `requires_patches` (dependency must apply BEFORE dependent).

    Uses Kahn's algorithm with insertion-order tie-breaking — minimizes
    diff against the natural registry order. Cycles are reported as
    `RuntimeError` (the registry's `test_iron_rule_11_enforcement.py`
    + the dep-graph preflight already gate against cycles at PR time;
    this is a defensive runtime backstop).

    v11.3.0 BUG #11 fix: `iter_patch_specs(topo_sort=True)` uses this
    so the spec-driven apply path applies dependencies before
    dependents — fixes 6 known order violations (PN105/PN104,
    PN34/PN33, G4_75/G4_74, G4_70/G4_69, PN256/G4_67, G4_69/G4_60K)
    at v11.3.0 baseline.
    """
    # Build in-degree + adjacency
    insertion_order = list(registry.keys())
    pos = {pid: i for i, pid in enumerate(insertion_order)}
    incoming: dict[str, set[str]] = {pid: set() for pid in insertion_order}
    outgoing: dict[str, set[str]] = {pid: set() for pid in insertion_order}
    for pid, meta in registry.items():
        if not isinstance(meta, dict):
            continue
        for req in (meta.get("requires_patches") or []):
            if req not in registry:
                continue  # unknown ref — silently ignore (other audits surface this)
            # `pid requires req` ⇒ edge req → pid (req must come first)
            incoming[pid].add(req)
            outgoing[req].add(pid)
    # Kahn — ready queue sorted by original insertion order (stable)
    ready = sorted(
        (pid for pid, deps in incoming.items() if not deps),
        key=lambda p: pos[p],
    )
    out: list[str] = []
    while ready:
        pid = ready.pop(0)
        out.append(pid)
        # Remove this node from its dependents
        for dep in sorted(outgoing[pid], key=lambda p: pos[p]):
            incoming[dep].discard(pid)
            if not incoming[dep]:
                # Insert keeping insertion-order ordering of ready
                _insert_sorted(ready, dep, pos)
    if len(out) != len(insertion_order):
        # Cycle — find unfinished
        remaining = [p for p in insertion_order if p not in out]
        raise RuntimeError(
            f"requires_patches DAG contains a cycle — cannot topological-sort. "
            f"Remaining unsorted: {remaining[:10]}"
        )
    return out


def _insert_sorted(
    queue: list[str], item: str, pos: dict[str, int],
) -> None:
    """Insert `item` into `queue` keeping it sorted by pos[item]."""
    target = pos[item]
    lo, hi = 0, len(queue)
    while lo < hi:
        mid = (lo + hi) // 2
        if pos[queue[mid]] < target:
            lo = mid + 1
        else:
            hi = mid
    queue.insert(lo, item)


def iter_patch_specs(
    registry: Optional[dict[str, dict[str, Any]]] = None,
    *,
    topo_sort: bool = False,
) -> Iterator[PatchSpec]:
    """Yield `PatchSpec` for every registry entry.

    Args:
        registry: defaults to `sndr.dispatcher.PATCH_REGISTRY`
            (the canonical source of truth).
        topo_sort: if True, yield specs in a stable topological order
            respecting `requires_patches` (dependency-first). Default
            False for backward compatibility — applies in
            registry-insertion order. The spec-driven apply
            orchestrator opts in via env `SNDR_TOPO_SORT_SPECS=1`.

            v11.3.0 BUG #11: at the v11.3.0 baseline, registry
            insertion order has 6 known requires_patches violations
            where a dependent appears BEFORE its required dependency.
            With topo_sort=True these are corrected before dispatch.
    """
    if registry is None:
        from sndr.dispatcher import PATCH_REGISTRY
        registry = PATCH_REGISTRY
    apply_map = _build_apply_module_map()
    if topo_sort:
        ordered_ids = _topological_order(registry)
        for pid in ordered_ids:
            meta = registry[pid]
            if not isinstance(meta, dict):
                continue
            yield patch_spec_for(pid, meta, apply_module_map=apply_map)
    else:
        for pid, meta in registry.items():
            if not isinstance(meta, dict):
                continue
            yield patch_spec_for(pid, meta, apply_module_map=apply_map)


# ─── Coverage diagnostic ──────────────────────────────────────────────────


# Lifecycle / implementation_status values that legitimately have NO
# `apply_module` field — they are registry-only entries by design:
#
#   lifecycle=legacy       → pre-dispatcher patches; auto-apply via the
#                            legacy text-patch wiring under
#                            `sndr/engines/vllm/wiring/`. The dispatcher
#                            does not need to know about them.
#   lifecycle=coordinator  → bundle/forward-shim row (e.g. PN274). Has
#                            no apply path of its own; coordinates
#                            other patches via env_flag wiring.
#   lifecycle=retired      → no-op entry preserved for audit-trail.
#   implementation_status in {marker_only, metadata_only, placeholder,
#                             advisory, research}
#                          → informational; env consumed inside another
#                            patch's apply_module or by the runtime
#                            directly, not via the dispatcher loop.
#
# The coverage report excludes these from the `unmapped` list so the
# operator's `patches doctor` view doesn't conflate intentional
# registry-only entries with real-residual-gap patches that need
# follow-up wiring.
_INTENTIONALLY_UNMAPPED_LIFECYCLES = frozenset({"legacy", "coordinator", "retired"})
_INTENTIONALLY_UNMAPPED_IMPL_STATUSES = frozenset({
    "marker_only", "metadata_only", "placeholder", "advisory", "research",
})


@dataclass(frozen=True)
class CoverageReport:
    """Snapshot of `PatchSpec.apply_module` coverage across the registry.

    `unmapped` is the list of `patch_id` values whose registry entry has
    no `apply_module` AND is not in any of the intentionally-unmapped
    categories above. Real-residual-gap patches surface here so the
    operator can audit which need follow-up wiring.

    `intentionally_unmapped` is the parallel list of registry-only
    entries that are NOT a gap by design (legacy auto-apply,
    coordinator bundles, marker-only metadata, etc.).
    """
    total: int
    mapped: int
    unmapped: list[str]
    intentionally_unmapped: list[str] = field(default_factory=list)


def validate_apply_module_coverage(
    registry: Optional[dict[str, dict[str, Any]]] = None,
) -> CoverageReport:
    """Walk every registry entry; report which ones have no apply_module,
    splitting between real gaps and intentional registry-only entries."""
    total = 0
    mapped = 0
    unmapped: list[str] = []
    intentional: list[str] = []
    for spec in iter_patch_specs(registry):
        total += 1
        if spec.apply_module is not None:
            mapped += 1
            continue
        # Inspect the raw registry meta to classify the unmapped entry.
        # `PatchSpec` doesn't carry lifecycle / impl_status directly,
        # so re-read from the registry dict.
        from sndr.dispatcher.registry import PATCH_REGISTRY as _REG
        meta = _REG.get(spec.patch_id, {}) if registry is None else (registry.get(spec.patch_id, {}))
        lc = meta.get("lifecycle")
        impl = meta.get("implementation_status")
        if (lc in _INTENTIONALLY_UNMAPPED_LIFECYCLES
                or impl in _INTENTIONALLY_UNMAPPED_IMPL_STATUSES):
            intentional.append(spec.patch_id)
        else:
            unmapped.append(spec.patch_id)
    return CoverageReport(
        total=total, mapped=mapped, unmapped=unmapped,
        intentionally_unmapped=intentional,
    )


__all__ = [
    "PatchSpec",
    "patch_spec_for",
    "iter_patch_specs",
    "CoverageReport",
    "validate_apply_module_coverage",
    "reset_apply_module_cache",
]
