# SPDX-License-Identifier: Apache-2.0
"""SNDR Core apply — orchestration state (shared across orchestrator + dispatch).

Why this module exists:
  At Stage 3 we split the 5273-LOC apply_all.py monolith into 4 submodules:
    - _state.py             (this) — mutable boot state shared by everyone
    - _per_patch_dispatch.py        — 95 apply_patch_X functions (parking
                                       lot until Stage 6 reorg moves them
                                       to per-subsystem modules)
    - orchestrator.py               — run() main loop + main()
    - verify.py                     — verify_live_rebinds()

  Module-level mutable state (`_APPLY_MODE` flag, `PATCH_REGISTRY` list)
  must be SHARED across these modules. Naive `from x import _APPLY_MODE`
  takes a snapshot at import time — when run() does `_APPLY_MODE = True`,
  the imported references stay False. Putting state in this dedicated
  module + having other modules access via `_state._APPLY_MODE` (attribute
  read) keeps the mutation visible.

This module contains:
  - PatchResult / PatchStats dataclasses (used everywhere)
  - _APPLY_MODE module-level flag (mutated by run())
  - PATCH_REGISTRY list (populated by @register_patch decorators)
  - register_patch decorator
  - _applied / _skipped / _failed factory helpers
  - _resolve_wiring_module / _wiring_text_patch generic dispatchers

Migration history:
  - Original location: vllm/_genesis/patches/apply_all.py (Stage 0).
  - Stage 3 (CURRENT): extracted into apply/_state.py.
"""
from __future__ import annotations

import json  # noqa: F401  (used by some patch functions via from-import)
import logging
import sys  # noqa: F401
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("genesis.apply_all")


# ═══════════════════════════════════════════════════════════════════════════
#                        ORCHESTRATION STATE / CLASSES
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
#                          ORCHESTRATION STATE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PatchResult:
    """Outcome of a single patch attempt."""
    name: str
    status: str           # "applied" | "skipped" | "failed"
    reason: str = ""      # short explanation


@dataclass
class PatchStats:
    """Accumulates per-run statistics for reporting."""
    results: list[PatchResult] = field(default_factory=list)
    # [Genesis T4.6] compile-watchdog: total apply_all elapsed seconds.
    # Set by run() at end. 0.0 if not measured (e.g. dry-run via CLI).
    compile_elapsed_sec: float = 0.0

    @property
    def applied(self) -> list[PatchResult]:
        return [r for r in self.results if r.status == "applied"]

    @property
    def skipped(self) -> list[PatchResult]:
        return [r for r in self.results if r.status == "skipped"]

    @property
    def failed(self) -> list[PatchResult]:
        return [r for r in self.results if r.status == "failed"]

    @property
    def applied_count(self) -> int:
        return len(self.applied)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    @property
    def failed_count(self) -> int:
        return len(self.failed)

    @property
    def partial_apply_warnings(self) -> list[PatchResult]:
        """Skipped patches whose reason signals a real problem (drift,
        ambiguous anchor, anchor-missing — NOT opt-in-OFF, upstream-merged,
        or platform-mismatch which are all expected).

        Surfaced separately from `skipped_count` so noonghunna's "silent
        skip class" diagnosis (club-3090 discussion #19) is impossible to
        miss in the boot summary. Cliff 8 hardening, v7.65.
        """
        # Reasons that indicate a benign/expected skip
        BENIGN = (
            "opt-in",   # matches "opt-in only", "opt-in:", "opt-in env"
            "default off",
            "upstream_merged",
            "upstream_already",
            "upstream_already_contains",
            "upstream may have absorbed",
            "upstream pr",  # "redundant: upstream PR ..."
            "platform mismatch",
            "platform_skip",
            "config: opt-in",
            "config: opt-out",
            "config: skipped",
            "config: neutral",
            "already applied",
            "marker present",
            "soft_skip",
            "no-op",
            "dry-run",
            "vllm install root not discoverable",
            "target file not resolvable",
            "is_pn",
            "unsupported",
            "not applicable",
            "auto-disabled",
            "auto-skip",
            "deprecated",
            "obsolete",
            "redundant",
            "deferred",
            "disabled",           # Stage 8 — bundle skip message ("bundle X disabled")
            "incompatible with",  # P7 deferred reason
            "retired",            # explicitly retired patches (P8 → 2026-05-04)
            "kernel disabled",    # P67b when P67 kernel disabled (companion patch design)
            "dispatch unused",    # ditto
        )
        warnings = []
        for r in self.skipped:
            reason_lower = (r.reason or "").lower()
            if not any(b.lower() in reason_lower for b in BENIGN):
                warnings.append(r)
        return warnings

    @property
    def partial_apply_warnings_count(self) -> int:
        return len(self.partial_apply_warnings)

    def summary(self) -> dict[str, Any]:
        return {
            "applied": self.applied_count,
            "skipped": self.skipped_count,
            "failed": self.failed_count,
            "partial_apply_warnings": self.partial_apply_warnings_count,
            "details": {
                "applied": [(r.name, r.reason) for r in self.applied],
                "skipped": [(r.name, r.reason) for r in self.skipped],
                "failed": [(r.name, r.reason) for r in self.failed],
                "partial_apply_warnings": [
                    (r.name, r.reason) for r in self.partial_apply_warnings
                ],
            },
        }

    def __str__(self) -> str:
        base = (
            f"Results: {self.applied_count} applied, "
            f"{self.skipped_count} skipped, {self.failed_count} failed"
        )
        warns = self.partial_apply_warnings_count
        if warns:
            base += f", {warns} ⚠️ partial-apply warning(s)"
        return base


# ═══════════════════════════════════════════════════════════════════════════
#                           PATCH REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

# Each patch function returns a PatchResult describing the outcome.
PATCH_REGISTRY: list[tuple[str, Callable[[], PatchResult]]] = []


def register_patch(name: str):
    """Decorator to register a patch function.

    Wave 7 (2026-05-09): wrap the registered ``fn`` in
    ``measure_patch_apply`` so each apply() call captures elapsed_ms +
    rss_delta_kb when ``GENESIS_OBSERVABILITY=1``. The instrumentation
    is a no-op pass-through when the env is unset, preserving the
    default boot path's zero-overhead posture (verified by
    tests/unit/observability/test_patch_metrics.py::TestDefaultOff).
    """
    def decorator(fn: Callable[[], PatchResult]) -> Callable[[], PatchResult]:
        # Late import to avoid a circular dep during module init —
        # observability lives outside apply/.
        from sndr.observability import measure_patch_apply

        def _instrumented_apply() -> PatchResult:
            with measure_patch_apply(name) as _metric:
                result = fn()
                # Best-effort propagate to the metric so observers get
                # the right status/reason. Never raises — defensive.
                try:
                    _metric.status = result.status
                    _metric.reason = result.reason or ""
                except Exception:
                    pass
                return result

        # Preserve __name__/__doc__ so test introspection + audit
        # tooling still work against the registered functions.
        _instrumented_apply.__name__ = getattr(fn, "__name__", name)
        _instrumented_apply.__doc__ = getattr(fn, "__doc__", None)
        _instrumented_apply.__wrapped__ = fn  # type: ignore[attr-defined]
        PATCH_REGISTRY.append((name, _instrumented_apply))
        return fn
    return decorator


def _applied(name: str, reason: str = "") -> PatchResult:
    return PatchResult(name=name, status="applied", reason=reason)


def _skipped(name: str, reason: str) -> PatchResult:
    return PatchResult(name=name, status="skipped", reason=reason)


def _failed(name: str, reason: str) -> PatchResult:
    return PatchResult(name=name, status="failed", reason=reason)


# ═══════════════════════════════════════════════════════════════════════════
#                       PATCH IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════

# Module-level state: are we in dry-run or apply mode for this run?
# Set by run(apply=True/False). Dry-run only diagnoses; apply performs the
# actual text-patch / monkey-patch wiring.
_APPLY_MODE: bool = False


_WIRING_STEM_INDEX: dict[str, str] | None = None


def _resolve_wiring_module(stem: str) -> str:
    """Resolve a wiring filename stem to its full dotted module path.

    PR38 cleanup (2026-05-08): `_genesis/wiring/` removed. Walk the
    canonical `sndr_core/integrations/<family>/` tree with the new naming
    convention (`p67_*.py`, `pn14_*.py`). Build BOTH a stem→dotted index
    AND a legacy-stem alias (`patch_67_*` → `p67_*`, `patch_N14_*` →
    `pn14_*`) so `_per_patch_dispatch.py` callers that still pass legacy
    stem names continue to resolve correctly.
    """
    global _WIRING_STEM_INDEX
    if _WIRING_STEM_INDEX is None:
        from sndr.engines.vllm.locations.project_paths import wiring_dir as _wiring_dir
        wd = _wiring_dir()
        idx: dict[str, str] = {}
        if wd is not None and wd.is_dir():
            # v12.0 (2026-06-04): wiring tree may live at either
            #   vllm/sndr_core/integrations/ (legacy)  → dotted prefix vllm.sndr_core.integrations
            #   sndr/engines/vllm/patches/   (canonical) → dotted prefix sndr.engines.vllm.patches
            # The previous logic ("dotted starts at vllm_root.parent") only
            # worked for the legacy layout. Detect which layout by walking
            # up from the wiring dir until we hit a directory that contains
            # an ``__init__.py`` in its parent (the top-level package root).
            # The dotted path is then constructed relative to that root's
            # parent so the top-level package name is included.
            def _top_package_parent(p: Path) -> Path:
                """Return the parent of the top-level Python package
                containing ``p`` (used for ``relative_to`` math)."""
                current = p
                # Climb while current is inside a Python package
                # (parent has __init__.py).
                while (current.parent / "__init__.py").exists():
                    current = current.parent
                return current.parent

            pkg_parent = _top_package_parent(wd)

            # v12.0 (2026-06-06): also walk sibling ``_archive/`` (retired
            # patch corpus — pn19, g4_05, p61 etc. live here). The
            # dispatcher still calls into them so retired stubs can return
            # a structured SKIPPED, but the resolver was only walking the
            # active wiring tree and reported every retired stem as
            # ``UNRESOLVED.<stem>``. Walking _archive (when present)
            # closes that.
            walk_roots = [wd]
            archive_dir = wd.parent / "_archive"
            if archive_dir.is_dir():
                walk_roots.append(archive_dir)

            # Canonical filenames: p<NUM>[<LETTER>]_*.py / pn<NUM>_*.py.
            # Walk every .py and key by its stem.
            for root in walk_roots:
                for f in root.rglob("*.py"):
                    if f.name.startswith("__") or "__pycache__" in f.parts:
                        continue
                    stem_name = f.stem
                    rel_parts = f.relative_to(pkg_parent).parts
                    dotted = ".".join(list(rel_parts[:-1]) + [stem_name])
                    # Active patches win over archived stems on stem
                    # collision (active tree walked first).
                    idx.setdefault(stem_name, dotted)

                    # Build legacy-name alias for back-compat with
                    # _per_patch_dispatch.py callers passing pre-flip names:
                    #   p67_tq_multi_query_kernel  ←  patch_67_tq_multi_query_kernel
                    #   pn14_tq_decode_oob_clamp   ←  patch_N14_tq_decode_oob_clamp
                    #   p67b_spec_verify_routing   ←  patch_67b_spec_verify_routing
                    if stem_name.startswith("pn"):
                        legacy_alias = "patch_N" + stem_name[2:]
                        idx.setdefault(legacy_alias, dotted)
                    elif stem_name.startswith("p") and len(stem_name) > 1 and stem_name[1].isdigit():
                        legacy_alias = "patch_" + stem_name[1:]
                        idx.setdefault(legacy_alias, dotted)
        _WIRING_STEM_INDEX = idx
    # If unresolvable, return an obviously-wrong dotted path so the
    # subsequent `import_module` raises ImportError with the bad name
    # in the message — easier to diagnose than silently returning an
    # alias that doesn't exist.
    return _WIRING_STEM_INDEX.get(
        stem, f"sndr.engines.vllm.patches.UNRESOLVED.{stem}"
    )


def _wiring_text_patch(name: str, wiring_module_name: str) -> PatchResult:
    """Generic helper for dry-run / live dispatch of a text-patch wiring module."""
    try:
        import importlib
        mod = importlib.import_module(
            _resolve_wiring_module(wiring_module_name)
        )
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")

    if not _APPLY_MODE:
        return _applied(name, "dry-run: wiring ready (pass apply=True to execute)")

    try:
        status, reason = mod.apply()
    except Exception as e:
        return _failed(name, f"wiring raised (should not happen): {e}")

    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)
