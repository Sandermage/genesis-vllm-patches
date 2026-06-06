# SPDX-License-Identifier: Apache-2.0
"""Genesis categories — patch navigation by category.

The categories API answers operator questions like:

  - "What category does PN14 belong to?"  → kernel_safety
  - "What patches are in spec_decode?"     → ['P56', 'P58', 'P60', ...]
  - "What's the wiring module for P67?"    → vllm._genesis.wiring.patch_67_*

Categories are derived from PATCH_REGISTRY's `category` field — there
is no separate manual table that could drift. Adding `category=...` to a
new patch entry automatically makes it discoverable here.

Also exposes a CLI:
  python3 -m vllm.sndr_core.compat.categories
  python3 -m vllm.sndr_core.compat.categories --category spec_decode
  python3 -m vllm.sndr_core.compat.categories --json

This is the **navigation surface for Phase 2** of the Compat Layer
overhaul. The physical disk reorganization (moving wiring/patch_*.py
files into category subdirs) is Phase 2.1, deferred until needed —
this module provides the logical reorganization without touching disk.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import re
import sys
from pathlib import Path

log = logging.getLogger("genesis.compat.categories")


# ─── Filename → module mapping ──────────────────────────────────────────


# PR38 cleanup (2026-05-08): legacy `_genesis/wiring/` removed. Walk the
# canonical `sndr_core/integrations/` tree with the post-flip naming
# convention (`p67_*.py`, `pn14_*.py`).
_WIRING_DIR = (
    Path(__file__).resolve().parent.parent / "integrations"
)


def _build_module_index() -> dict[str, str]:
    """Walk wiring/ for patch_*.py files, build a normalized index keyed
    by patch number/letter for fast lookup.

    Walks recursively so post-Phase-2.1 category subdirs work the same
    as the legacy flat layout. Module paths are computed from the file's
    relative location under wiring/, e.g.:

      Flat layout (legacy):
        wiring/patch_67_tq_multi_query_kernel.py
          → 'vllm._genesis.wiring.patch_67_tq_multi_query_kernel'

      Categorical layout (Phase 2.1):
        wiring/spec_decode/patch_67_tq_multi_query_kernel.py
          → 'vllm._genesis.wiring.spec_decode.patch_67_tq_multi_query_kernel'

    Index key format examples:
      'PN14'  → '...wiring.kernels.patch_N14_tq_decode_oob_clamp'
      'P67'   → '...wiring.spec_decode.patch_67_tq_multi_query_kernel'
      'P67b'  → '...wiring.spec_decode.patch_67b_spec_verify_routing'
      'P68'   → '...wiring.structured_output.patch_68_69_long_ctx_tool_adherence' (shared)
      'P69'   → same as above
    """
    index: dict[str, str] = {}
    if not _WIRING_DIR.is_dir():
        return index

    # First pass — collect candidates per pid (file_basename, module_path).
    # We don't write to `index` immediately because some pids (e.g. PN26
    # vs PN26b) have two files matching the same numeric stem; the
    # registry's `env_flag` is the only signal to disambiguate.
    candidates: dict[str, list[tuple[str, str]]] = {}

    # Match canonical filenames: `p67_*.py`, `pn14_*.py`, `p67b_*.py`,
    # `p68_69_*.py` (compound). Numbers can be followed by an optional
    # letter (suffix variants like `b`, `c`).
    for f in sorted(_WIRING_DIR.rglob("p*.py")):
        if f.name.startswith("__"):
            continue
        stem = f.stem  # e.g. "p67_tq_multi_query_kernel"
        # Extract the patch identifier(s):
        #   p<NUM>[<LETTER>]_* → P<NUM>[<LETTER>]   (numeric P-series)
        #   pn<NUM>[<LETTER>]_* → PN<NUM>[<LETTER>]   (PN-series + suffix)
        #   p<A>_<B>_*         → covers two (P_A AND P_B both → this file)
        m = re.match(r"^p(n\d+\w*?|\d+\w*?)_", stem)
        if not m:
            continue
        first_id_token = m.group(1)
        if first_id_token.startswith("n"):
            # pn14_* → PN14 ; pn26b_* → PN26b (case-preserving) + PN26B
            primary_lc = "PN" + first_id_token[1:]
            primary_uc = "PN" + first_id_token[1:].upper()
            ids = list({primary_lc, primary_uc})
        else:
            # Numeric form. Could be plain (p56_) or compound (p68_69_).
            # Suffix letter stays lowercase in canonical filenames
            # (`p60b`, `p67c`) but registry IDs use the same case the
            # operator types — `P60b` not `P60B`. Index BOTH forms so
            # `module_for("P60b")` and `module_for("P60B")` resolve.
            primary_lc = "P" + first_id_token  # `P60b`
            primary_uc = "P" + first_id_token.upper()  # `P60B`
            ids = list({primary_lc, primary_uc})
            compound_m = re.match(r"^p(\d+\w*?)_(\d+\w*?)_", stem)
            if compound_m:
                comp_token = compound_m.group(2)
                ids.append("P" + comp_token)
                ids.append("P" + comp_token.upper())

        # Compute dotted module path from path parts.
        # _WIRING_DIR = repo_root/vllm/sndr_core/integrations
        # parents:    [.../patches, .../sndr_core, .../vllm, repo_root]
        rel = f.relative_to(_WIRING_DIR.parent.parent.parent)  # repo-root anchored
        # rel example: vllm/sndr_core/integrations/spec_decode/p67_*.py
        parts = list(rel.parts[:-1]) + [stem]
        module_path = ".".join(parts)
        for pid in ids:
            candidates.setdefault(pid, []).append((stem, module_path))

    # Second pass — disambiguate collisions via registry env_flag suffix.
    # When two files claim the same pid (legacy convention: `pn26_*` for
    # both PN26 and PN26b), the registry's `env_flag` suffix word picks
    # the right file. Example: PN26's env_flag is
    # GENESIS_ENABLE_PN26_TQ_UNIFIED → prefer file whose stem contains
    # `tq_unified`. Also: registry may have `PN26b` (letter-suffix
    # variant) whose actual on-disk filename is the legacy `pn26_*` form
    # without the letter — we propagate the bare-numeric candidate pool
    # to those variants so they can still resolve.
    try:
        from vllm.sndr_core.dispatcher import PATCH_REGISTRY
    except Exception:
        PATCH_REGISTRY = {}  # type: ignore[assignment]

    def _disambiguate(pid: str, cands: list[tuple[str, str]]) -> str:
        flag = (PATCH_REGISTRY.get(pid, {}) or {}).get("env_flag", "") or ""
        # Strip `GENESIS_ENABLE_<PID>_` prefix → keep discriminator
        # suffix. E.g. `GENESIS_ENABLE_PN26_TQ_UNIFIED` → `tq_unified`.
        suffix_tokens = [
            t.lower() for t in flag.split("_")
            if t.lower() not in {"genesis", "enable", pid.lower()}
            and not t.lower().startswith(("pn", "p"))
        ]
        for stem, mod in cands:
            stem_lower = stem.lower()
            if suffix_tokens and all(tok in stem_lower for tok in suffix_tokens):
                return mod
        return sorted(cands)[0][1]  # deterministic fallback

    for pid, cands in candidates.items():
        index[pid] = cands[0][1] if len(cands) == 1 else _disambiguate(pid, cands)

    # Letter-suffix registry variants without their own files. PN26b's
    # filename is the legacy `pn26_sparse_v_kernel.py` (no `b` in stem),
    # so first-pass indexing only put it under PN26. Walk the registry,
    # find PID variants that share a numeric stem with an existing index
    # entry, and route them via the parent's candidate pool.
    pattern = re.compile(r"^(P[N]?\d+)([a-zA-Z].*)$")
    for pid in PATCH_REGISTRY:
        if pid in index:
            continue
        m_variant = pattern.match(pid)
        if not m_variant:
            continue
        parent_pid = m_variant.group(1)
        parent_cands = candidates.get(parent_pid) or candidates.get(parent_pid.upper())
        if not parent_cands:
            continue
        index[pid] = _disambiguate(pid, parent_cands)

    return index


_MODULE_INDEX: dict[str, str] | None = None


def _module_index() -> dict[str, str]:
    global _MODULE_INDEX
    if _MODULE_INDEX is None:
        _MODULE_INDEX = _build_module_index()
    return _MODULE_INDEX


# ─── Category → patches mapping ─────────────────────────────────────────


def _build_categories() -> dict[str, list[str]]:
    """Group PATCH_REGISTRY entries by their `category` field.

    Patches with no `category` are grouped under 'uncategorized'.
    """
    try:
        from vllm.sndr_core.dispatcher import PATCH_REGISTRY
    except Exception as e:
        log.debug("PATCH_REGISTRY import failed: %s", e)
        return {}

    out: dict[str, list[str]] = {}
    for pid, meta in PATCH_REGISTRY.items():
        cat = meta.get("category", "uncategorized")
        out.setdefault(cat, []).append(pid)

    # Sort each category's patch list deterministically
    for cat in out:
        out[cat].sort()
    return out


# CATEGORIES is built lazily on first access so it picks up monkey-patched
# PATCH_REGISTRY in tests.
def _get_categories_dict() -> dict[str, list[str]]:
    """Always rebuild from current PATCH_REGISTRY — no caching, so tests
    that monkey-patch the registry get the right view."""
    return _build_categories()


# Compatibility surface: at import time, populate CATEGORIES from current
# registry. Test fixtures that monkey-patch PATCH_REGISTRY can call
# `refresh()` to recompute.
CATEGORIES: dict[str, list[str]] = _get_categories_dict()


def refresh() -> None:
    """Rebuild CATEGORIES from current PATCH_REGISTRY. Called by tests
    that monkey-patch the registry."""
    global CATEGORIES
    CATEGORIES = _get_categories_dict()


# ─── Public lookup helpers ──────────────────────────────────────────────


def category_for(patch_id: str) -> str | None:
    """Return the category for `patch_id`, or None if not in registry."""
    cats = _get_categories_dict()
    for cat, patches in cats.items():
        if patch_id in patches:
            return cat
    return None


def patches_in(category: str) -> list[str]:
    """Return the list of patch IDs in `category` (empty if unknown)."""
    return list(_get_categories_dict().get(category, []))


def module_for(patch_id: str) -> str | None:
    """Return the wiring module path for `patch_id`, or None if no
    wiring module is found.

    Examples:
      'PN14' → 'vllm._genesis.wiring.patch_N14_tq_decode_oob_clamp'
      'P67'  → 'vllm._genesis.wiring.patch_67_tq_multi_query_kernel'
    """
    return _module_index().get(patch_id)


def import_module_for(patch_id: str):
    """Resolve `patch_id` to a wiring module + import it.

    Returns the loaded module object, or None if no wiring file
    matches. Raises ImportError if found but import fails.
    """
    mod_path = module_for(patch_id)
    if mod_path is None:
        return None
    return importlib.import_module(mod_path)


# ─── CLI ─────────────────────────────────────────────────────────────────


def _format_text(cats: dict[str, list[str]],
                 filter_category: str | None = None) -> list[str]:
    L = ["=" * 72,
         f"Genesis patch categories — {sum(len(p) for p in cats.values())} "
         f"total patches in {len(cats)} categories",
         "=" * 72, ""]

    items = sorted(cats.items()) if filter_category is None \
            else [(filter_category, cats.get(filter_category, []))]
    for cat, patches in items:
        L.append(f"  [{cat}]  ({len(patches)} patches)")
        for p in patches:
            mod = module_for(p) or "(no wiring file)"
            # Strip the long common prefix so the dotted path is readable.
            # v10 rename moved canonical from `vllm._genesis.wiring.` to
            # `vllm.sndr_core.integrations.`; legacy prefix kept for
            # backwards-compat output during transition.
            short = mod
            for prefix in (
                "vllm.sndr_core.integrations.",
                "vllm._genesis.wiring.",
            ):
                if short.startswith(prefix):
                    short = short[len(prefix):]
                    break
            L.append(f"    • {p:<8} → {short}")
        L.append("")
    L.append("=" * 72)
    return L


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m vllm.sndr_core.compat.categories",
        description="Browse Genesis patches by category.",
    )
    parser.add_argument("--category", default=None,
                        help="Filter to one category (e.g. spec_decode)")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    args = parser.parse_args(argv)

    cats = _get_categories_dict()

    if args.category and args.category not in cats:
        print(f"unknown category: {args.category!r}", file=sys.stderr)
        print(f"available: {sorted(cats.keys())}", file=sys.stderr)
        return 2

    if args.json:
        # Structured for machine consumers
        out = {
            "categories": {
                c: [
                    {"patch_id": p, "module": module_for(p)}
                    for p in cats[c]
                ]
                for c in (sorted(cats) if not args.category else [args.category])
            },
            "total_patches": sum(len(p) for p in cats.values()),
            "total_categories": len(cats),
        }
        print(json.dumps(out, indent=2, default=str))
    else:
        for line in _format_text(cats, filter_category=args.category):
            print(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
