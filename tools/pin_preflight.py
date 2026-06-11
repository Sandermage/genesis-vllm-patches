#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pin-bump preflight — read-only readiness report for a CANDIDATE vllm tree.

Given a pristine vllm package tree extracted from a candidate image (see
``tools/extract_candidate_tree.sh``), evaluate EVERY active Genesis patch
against it WITHOUT applying anything and WITHOUT touching PROD:

  - Text-patchers: build via each wiring module's ``_make*patcher()``
    builders under a redirected install root, then mirror the apply()
    decision layers READ-ONLY (Layer 2 marker / Layer 3 drift markers /
    Layer 5 anchor scan via a pure reimplementation of
    ``TextPatcher._apply_layer5_legacy`` — behavior parity is pinned by
    ``tests/unit/tools/test_pin_preflight.py::TestLayer5Parity``).
    ``TextPatcher.apply()`` is NEVER called — it writes to disk.
  - Runtime-hook modules (no text-patcher builder): static AST extraction
    of their ``vllm.*`` import bindings, checked for file/symbol presence
    in the candidate tree. Static only — call-site liveness still needs
    in-container verification (rows carry ``VERIFY_IN_CONTAINER``).
  - Tree-wide passes: ``UPSTREAM_MARKERS`` merge detection, per-patch
    ``applies_to.vllm_version_range`` gating against the candidate's
    internal version, anchor-manifest staleness, and a SELF_COLLISION
    static lint (the PN369 class: a patcher's own replacement text or
    marker matching one of its upstream_drift_markers).

Verdicts per patcher:
  OK                  — every required anchor matches exactly once
  DRIFT_ANCHOR        — a required anchor is absent (count == 0)
  AMBIGUOUS_ANCHOR    — an anchor matches more than once
  DRIFT_FILE_MOVED    — target file missing, or builder returned None
                        with a resolve target actually absent
                        (report carries up to 3 moved-to candidates)
  UPSTREAM_MERGED     — patcher-level upstream_drift_marker present
  SUB_UPSTREAM_MERGED — per-sub upstream_merged_markers fired
  STALE_RESIDUE       — patch marker already present in a PRISTINE tree
                        (impossible unless residue or marker collision)
  ENV_GATED_ABSTAIN   — builder returned None even with its registry
                        env_flag forced to "1" AND every
                        resolve_vllm_file literal exists — the abstain
                        is gated on something beyond the flag, not on
                        file drift (v1.2, non-actionable)
  KNOWN_OPTIONAL_RETIRED — all-required=False patcher with zero anchor
                        matches whose every sub name is declared
                        retired-by-design via the wiring module's
                        ``*_RETIRED_SUBS`` attr (v1.2, non-actionable;
                        the P64 serving-patcher steady state)
  EXPECTED_ALTERNATE  — zero-match patcher whose every sub is declared
                        a per-pin variant via the wiring module's
                        ``*_ANCHOR_ALTERNATION`` dict attr AND whose
                        declared alternate sub matched in a sibling
                        row of the same module (v1.3, non-actionable;
                        the P91B dual-factory dev338/dev371 class —
                        exactly one inc.py anchor matches on any live
                        pin by design)
  UNBUILDABLE         — builder has required params we will not guess
                        (v1.2: a param <name> IS fillable when the
                        module exposes a callable ``_read_<name>`` —
                        the P82 apply() convention)
  IMPORT_FAIL         — wiring module failed to import
  RUNTIME_BINDING     — no text-patcher; static binding check only

v1.2 (2026-06-11, residual-triage action plan §4): rows built only
after an env-forced builder retry carry ``env_forced: true``; rows
whose required builder params were filled through the ``_read_<name>``
convention carry ``params_filled_via``.

v1.3 (2026-06-11, P91B alternation manifest): module-declared
anchor-alternation via ``*_ANCHOR_ALTERNATION`` (sub name → alternate
sub names). Reclassifies the non-matching per-pin factory variant from
DRIFT_ANCHOR to EXPECTED_ALTERNATE when the OTHER variant matched;
rows carry ``alternate_matched``. A miss on EVERY variant stays
DRIFT_ANCHOR — fail-noisy, never fail-silent.

Usage:
    python3 tools/pin_preflight.py <candidate_root> \
        [--provenance <path>] [--json-out <path>] [--fail-fast]

``<candidate_root>`` is the directory CONTAINING the vllm package
contents (i.e. ``<candidate_root>/v1`` must exist — typically
``<staging>/vllm`` produced by extract_candidate_tree.sh).

Exit codes:
    0 — all patches OK / runtime bindings resolve / no new merges
    1 — at least one actionable verdict (drift, merge, residue, ...)
    2 — invocation error (bad root, unreadable provenance, ...)

Fully offline by default — no gh API, no docker, no SSH.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import os

# MUST be set before any sndr import — kills the Layer-0 file_cache
# fast-path that could otherwise report false IDEMPOTENT from a stale
# cache entry recorded against a previous tree at the same path.
os.environ["GENESIS_NO_PATCH_CACHE"] = "1"

import argparse
import ast
import hashlib
import importlib
import inspect
import json
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ─── Verdict constants ────────────────────────────────────────────────────

OK = "OK"
DRIFT_ANCHOR = "DRIFT_ANCHOR"
CHAINED_ANCHOR = "CHAINED_ANCHOR"
AMBIGUOUS_ANCHOR = "AMBIGUOUS_ANCHOR"
DRIFT_FILE_MOVED = "DRIFT_FILE_MOVED"
UPSTREAM_MERGED = "UPSTREAM_MERGED"
SUB_UPSTREAM_MERGED = "SUB_UPSTREAM_MERGED"
STALE_RESIDUE = "STALE_RESIDUE"
ENV_GATED_ABSTAIN = "ENV_GATED_ABSTAIN"
KNOWN_OPTIONAL_RETIRED = "KNOWN_OPTIONAL_RETIRED"
EXPECTED_ALTERNATE = "EXPECTED_ALTERNATE"
UNBUILDABLE = "UNBUILDABLE"
IMPORT_FAIL = "IMPORT_FAIL"
RUNTIME_BINDING = "RUNTIME_BINDING"

BINDING_OK = "BINDING_OK"
BINDING_FILE_MISSING = "BINDING_FILE_MISSING"
BINDING_SYMBOL_MISSING = "BINDING_SYMBOL_MISSING"
BINDING_UNRESOLVED = "BINDING_UNRESOLVED"

# Patch-level verdicts that demand operator action before promoting the
# candidate pin (when the patch is in version range for the candidate).
# Deliberately excluded: CHAINED_ANCHOR (apply-order chain, judge the
# provider), ENV_GATED_ABSTAIN and KNOWN_OPTIONAL_RETIRED (v1.2 —
# documented non-drift abstain states, fail-noisy in the report only),
# EXPECTED_ALTERNATE (v1.3 — declared per-pin factory alternation; the
# matching variant's own row is the one to judge).
ACTIONABLE_VERDICTS = frozenset({
    DRIFT_ANCHOR, AMBIGUOUS_ANCHOR, DRIFT_FILE_MOVED, UPSTREAM_MERGED,
    SUB_UPSTREAM_MERGED, STALE_RESIDUE, UNBUILDABLE, IMPORT_FAIL,
})

# Implementation states whose wiring actually applies against the vllm
# tree (mirrors tools/check_upstream_drift.py::_list_wiring_modules).
TEXT_PATCH_STATES = frozenset({"live", "full", "text_patch", "runtime_hook"})

# Genesis-owned namespaces inside the vllm tree — NOT upstream bindings
# (installed by us at deploy time; absent from a pristine candidate).
_SELF_PACKAGE_PREFIXES = ("vllm.sndr_core", "vllm._genesis")


# ─── Layer 5 pure mirror ──────────────────────────────────────────────────


@dataclass
class Layer5Outcome:
    """Pure read-only mirror of ``TextPatcher._apply_layer5_legacy``.

    status   — "success" (some sub applied) | "skipped"
    reason   — skip reason string, byte-identical vocabulary to the real
               method: required_anchor_missing | ambiguous_anchor |
               sub_upstream_merged_abort_bundle | no_applicable_sub_patches
    detail   — human detail mirroring the real TextPatchFailure.detail
    applied  — sub-patch names whose anchor matched (in order)
    sub_merged — (sub_name, marker) pairs whose per-sub upstream marker
               fired (skip_silently / warn modes)
    modified — the would-be post-Layer-5 content on success (no marker
               prepend — that is Layer 6), None on skip
    """
    status: str
    reason: Optional[str] = None
    detail: Optional[str] = None
    applied: list[str] = field(default_factory=list)
    sub_merged: list[tuple[str, str]] = field(default_factory=list)
    modified: Optional[str] = None


def evaluate_layer5(sub_patches: Iterable[Any], content: str) -> Layer5Outcome:
    """Reimplementation of ``TextPatcher._apply_layer5_legacy`` (pure,
    never raises, never writes). Kept line-for-line equivalent in
    semantics; parity pinned by TestLayer5Parity. Adds ``sub_merged``
    bookkeeping the original only logs."""
    modified = content
    applied: list[str] = []
    sub_merged: list[tuple[str, str]] = []

    for sp in sub_patches:
        markers = getattr(sp, "upstream_merged_markers", None) or []
        sub_drift_match = next((um for um in markers if um in modified), None)
        if sub_drift_match is not None:
            if getattr(sp, "on_upstream_merge", "skip_silently") == "abort_bundle":
                return Layer5Outcome(
                    status="skipped",
                    reason="sub_upstream_merged_abort_bundle",
                    detail=(
                        f"sub-patch {sp.name!r}: upstream-merge marker "
                        f"{sub_drift_match!r} fired with "
                        f"on_upstream_merge=abort_bundle — patcher aborts"
                    ),
                    sub_merged=sub_merged + [(sp.name, sub_drift_match)],
                )
            sub_merged.append((sp.name, sub_drift_match))
            continue  # sibling subs continue with current `modified`

        if sp.anchor not in modified:
            if sp.required:
                return Layer5Outcome(
                    status="skipped",
                    reason="required_anchor_missing",
                    detail=f"sub-patch {sp.name!r}: anchor not found in file",
                    applied=applied, sub_merged=sub_merged,
                )
            continue

        if modified.count(sp.anchor) != 1:
            return Layer5Outcome(
                status="skipped",
                reason="ambiguous_anchor",
                detail=(
                    f"sub-patch {sp.name!r}: anchor appears "
                    f"{modified.count(sp.anchor)} times (expected 1)"
                ),
                applied=applied, sub_merged=sub_merged,
            )

        modified = modified.replace(sp.anchor, sp.replacement, 1)
        applied.append(sp.name)

    if not applied:
        return Layer5Outcome(
            status="skipped",
            reason="no_applicable_sub_patches",
            detail="every sub-patch anchor absent — file may be post-upstream-fix",
            sub_merged=sub_merged,
        )

    return Layer5Outcome(status="success", applied=applied,
                         sub_merged=sub_merged, modified=modified)


# ─── per-patcher verdict (read-only) ──────────────────────────────────────


def _anchor_signature_line(anchor: str) -> Optional[str]:
    """Most distinctive line of an anchor = its longest stripped line."""
    lines = [ln.strip() for ln in (anchor or "").splitlines() if ln.strip()]
    if not lines:
        return None
    return max(lines, key=len)


def find_moved_candidates(
    candidate_root: Path,
    *,
    anchor: Optional[str] = None,
    missing_rel: Optional[str] = None,
    limit: int = 3,
) -> list[str]:
    """Locate where a moved target may live now (the gdn/-split class).

    Two strategies, merged in order:
      1. anchor signature — grep candidate ``*.py`` for the FIRST
         sub-patch anchor's most distinctive (longest) line;
      2. basename — files elsewhere in the tree sharing the missing
         target's basename.
    """
    out: list[str] = []
    sig = _anchor_signature_line(anchor) if anchor else None
    base = Path(missing_rel).name if missing_rel else None
    stem = Path(missing_rel).stem if missing_rel else None
    try:
        for f in sorted(candidate_root.rglob("*.py")):
            rel = f.relative_to(candidate_root).as_posix()
            if missing_rel and rel == missing_rel:
                continue
            hit = False
            if base and f.name == base:
                hit = True
            # Rename/split class (gdn_linear_attn.py →
            # gdn/qwen_gdn_linear_attn.py): old stem survives as a
            # substring of the new filename (or vice versa). BOTH stems
            # must be >= 8 chars — short generic stems ("linear",
            # "utils") produce junk matches.
            if (not hit and stem and len(stem) >= 8
                    and len(f.stem) >= 8
                    and (stem in f.stem or f.stem in stem)):
                hit = True
            if not hit and sig:
                try:
                    if sig in f.read_text(encoding="utf-8", errors="ignore"):
                        hit = True
                except OSError:
                    continue
            if hit and rel not in out:
                out.append(rel)
                if len(out) >= limit:
                    break
    except OSError:
        pass
    return out


def self_collision_findings(patcher: Any) -> list[dict]:
    """SELF_COLLISION static lint (the PN369 class).

    A drift marker matching the patcher's OWN replacement text or its
    idempotency marker means: after apply (or via another patch baking
    the same string), Layer 3 reads the marker as 'upstream merged' and
    silently skips — deterministic false-skip class observed on PN369
    2026-06-10. Markers starting with "[Genesis" are flagged
    ``defended=True``: by convention custom apply() wrappers (PN353A)
    skip them, but the stock ``TextPatcher.apply()`` Layer 3 does NOT.
    """
    findings: list[dict] = []
    drift_markers = getattr(patcher, "upstream_drift_markers", None) or []
    own_marker = getattr(patcher, "marker", "") or ""
    subs = getattr(patcher, "sub_patches", None) or []
    for dm in drift_markers:
        if not dm:
            continue
        collides_with = None
        if dm in own_marker or own_marker and own_marker in dm:
            collides_with = "marker"
        else:
            for sp in subs:
                if dm in (getattr(sp, "replacement", "") or ""):
                    collides_with = "replacement"
                    break
        if collides_with:
            findings.append({
                "patch_name": getattr(patcher, "patch_name", "?"),
                "marker": dm,
                "collides_with": collides_with,
                "defended": dm.startswith("[Genesis"),
            })
    return findings


def evaluate_patcher(patcher: Any, candidate_root: Path) -> dict:
    """READ-ONLY verdict for one TextPatcher against the candidate tree.

    Mirrors ``TextPatcher.apply()`` decision layers on pristine content:
    Layer 1 (existence) → Layer 2 (marker ⇒ STALE_RESIDUE on a pristine
    tree) → Layer 3 (patcher-level drift markers ⇒ UPSTREAM_MERGED) →
    Layer 5 mirror (anchors). Never calls ``.apply()``.
    """
    target = Path(getattr(patcher, "target_file", "") or "")
    try:
        rel = target.relative_to(candidate_root).as_posix()
    except ValueError:
        rel = str(target)
    row: dict[str, Any] = {
        "patch_name": getattr(patcher, "patch_name", "?"),
        "target": rel,
        "verdict": None,
        "detail": "",
        "applied_subs": [],
        "sub_merged": [],
    }

    if not target.is_file():
        first_anchor = None
        subs = getattr(patcher, "sub_patches", None) or []
        if subs:
            first_anchor = getattr(subs[0], "anchor", None)
        row["verdict"] = DRIFT_FILE_MOVED
        row["detail"] = f"target file missing: {rel}"
        row["moved_to_candidates"] = find_moved_candidates(
            candidate_root, anchor=first_anchor, missing_rel=rel)
        return row

    try:
        content = target.read_text(encoding="utf-8", errors="ignore")
    except OSError as e:
        row["verdict"] = DRIFT_FILE_MOVED
        row["detail"] = f"target unreadable: {e}"
        return row

    # Private field for the chain reclassify pass (stripped before report
    # assembly): what this patcher EMITS — it may provide the anchors a
    # later patcher on the same file targets (P18B-on-PN119 class).
    row["_replacement_blob"] = "\n".join(
        getattr(s, "replacement", "") or ""
        for s in (getattr(patcher, "sub_patches", None) or []))

    marker = getattr(patcher, "marker", "") or ""
    if marker and marker in content:
        row["verdict"] = STALE_RESIDUE
        row["detail"] = (
            f"idempotency marker {marker!r} present in PRISTINE candidate "
            "file — extraction residue or marker/upstream-text collision"
        )
        return row

    for dm in getattr(patcher, "upstream_drift_markers", None) or []:
        if dm and dm in content:
            row["verdict"] = UPSTREAM_MERGED
            row["detail"] = f"upstream drift marker {dm!r} present in {rel}"
            return row

    subs = getattr(patcher, "sub_patches", None) or []
    outcome = evaluate_layer5(subs, content)
    row["applied_subs"] = outcome.applied
    row["sub_merged"] = [list(t) for t in outcome.sub_merged]
    row["_missing_anchors"] = [
        getattr(s, "anchor", "") for s in subs
        if getattr(s, "anchor", "") and getattr(s, "anchor", "") not in content
    ]

    if outcome.status == "success":
        row["verdict"] = SUB_UPSTREAM_MERGED if outcome.sub_merged else OK
        if outcome.sub_merged:
            row["detail"] = (
                "per-sub upstream markers fired: "
                + ", ".join(f"{n} via {m!r}" for n, m in outcome.sub_merged)
            )
        return row

    if outcome.reason == "ambiguous_anchor":
        row["verdict"] = AMBIGUOUS_ANCHOR
    elif outcome.reason == "sub_upstream_merged_abort_bundle":
        row["verdict"] = SUB_UPSTREAM_MERGED
    elif outcome.reason == "no_applicable_sub_patches" and outcome.sub_merged:
        # Every sub either merged upstream or missed; merges dominate.
        row["verdict"] = SUB_UPSTREAM_MERGED
    else:
        # required_anchor_missing | no_applicable_sub_patches (pure miss)
        row["verdict"] = DRIFT_ANCHOR
    row["detail"] = f"{outcome.reason}: {outcome.detail or ''}".strip(": ")
    return row


def _diff_post_apply_text(diff_text: str) -> str:
    """Reconstruct the post-apply content segments of a unified diff:
    added ('+') and context (' ') lines, removed ('-') lines and
    headers dropped. Good enough for substring chain-matching — NOT a
    full patch engine."""
    out: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith(("+++", "---", "@@", "diff ", "index ")):
            continue
        if line.startswith("+"):
            out.append(line[1:])
        elif line.startswith(" "):
            out.append(line[1:])
        elif line == "":
            out.append("")
    return "\n".join(out)


def evaluate_md5_diff_patch(mod: Any, candidate_root: Path) -> Optional[dict]:
    """READ-ONLY verdict for a PN119-class md5-gated diff patch.

    Convention (see PIN_BUMP_PLAYBOOK): the module exposes
    ``<NAME>_PRE_PATCH_MD5`` (hex str), ``<NAME>_DIFF_PATH`` (Path to a
    bundled unified diff) and ``_target_path()`` (resolves through
    ``resolve_vllm_file`` so the preflight's alternate-root seam
    redirects it into the candidate tree). Markers are module attrs
    named ``GENESIS_*_MARKER``.

    Returns None when the module doesn't follow the convention (caller
    falls through to the runtime-binding row).
    """
    md5_attrs = [a for a in dir(mod) if a.endswith("_PRE_PATCH_MD5")]
    diff_attrs = [a for a in dir(mod) if a.endswith("_DIFF_PATH")]
    target_fn = getattr(mod, "_target_path", None)
    if not (md5_attrs and diff_attrs and callable(target_fn)):
        return None

    expected_md5 = str(getattr(mod, md5_attrs[0]))
    diff_path = Path(getattr(mod, diff_attrs[0]))
    row: dict[str, Any] = {
        "patch_name": f"{mod.__name__} (md5-gated diff)",
        "verdict": None, "detail": "", "applied_subs": [],
        "sub_merged": [],
    }

    try:
        target = target_fn()
    except Exception as e:  # noqa: BLE001
        row["verdict"] = DRIFT_FILE_MOVED
        row["detail"] = f"_target_path() raised {type(e).__name__}: {e}"
        return row
    if target is None or not Path(target).is_file():
        row["verdict"] = DRIFT_FILE_MOVED
        row["detail"] = "target not found under candidate root"
        return row
    target = Path(target)
    try:
        row["target"] = target.relative_to(candidate_root).as_posix()
    except ValueError:
        row["target"] = str(target)

    if diff_path.is_file():
        row["_replacement_blob"] = _diff_post_apply_text(
            diff_path.read_text(encoding="utf-8", errors="ignore"))
    else:
        row["verdict"] = UNBUILDABLE
        row["detail"] = f"bundled diff missing: {diff_path}"
        return row

    content = target.read_text(encoding="utf-8", errors="ignore")
    for attr in dir(mod):
        if attr.startswith("GENESIS_") and "_MARKER" in attr:
            val = getattr(mod, attr)
            if isinstance(val, str) and val.strip() and val.strip() in content:
                row["verdict"] = STALE_RESIDUE
                row["detail"] = (
                    f"marker {val.strip()!r} present in PRISTINE candidate "
                    "file — extraction residue or collision")
                return row

    actual_md5 = hashlib.md5(target.read_bytes()).hexdigest()
    if actual_md5 == expected_md5:
        row["verdict"] = OK
        row["detail"] = "md5 gate matches — bundled diff will apply"
    else:
        row["verdict"] = DRIFT_ANCHOR
        row["detail"] = (
            f"md5 gate mismatch: candidate {actual_md5} != expected "
            f"{expected_md5} — patch self-retires on this pin; regenerate "
            "diff + md5 (and re-verify dependents via the chain pass)")
    return row


# ─── module evaluation (builders / import / runtime bindings) ─────────────


def builder_names(mod: Any) -> list[str]:
    """All text-patcher builder callables in a wiring module: canonical
    ``_make_patcher`` plus variants ``_make*patcher`` (e.g.
    ``_make_serving_patcher``, ``_make_gdn_patcher``)."""
    return sorted(
        n for n in dir(mod)
        if n.startswith("_make") and n.endswith("patcher")
        and callable(getattr(mod, n, None))
    )


def unfillable_params(fn: Any) -> list[str]:
    """Required parameters (no default) we refuse to guess values for.

    The legacy drift tool guessed kwargs by annotation substring — a
    known wart that produced nonsense patchers. Preflight reports
    UNBUILDABLE instead so the operator wires an explicit probe.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return []
    out = []
    for name, p in sig.parameters.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL,
                      inspect.Parameter.VAR_KEYWORD):
            continue
        if p.default is inspect.Parameter.empty:
            out.append(name)
    return out


_VLLM_DOTTED_RE = re.compile(r"^vllm(\.[A-Za-z_][A-Za-z0-9_]*)+$")


def extract_bindings(source: str) -> list[dict]:
    """Static AST extraction of upstream vllm bindings from wiring source.

    Recognized forms:
      - ``from vllm.x.y import A, B``        → (vllm.x.y, A), (vllm.x.y, B)
      - ``import vllm.x.y``                  → (vllm.x.y, None)
      - ``importlib.import_module("vllm.x")`` → (vllm.x, None)
      - ``*_MODULE_PATHS``/``*CANDIDATE*`` list constants of dotted paths
      - dynamic ``import_module(<expr>)``    → module=None (UNRESOLVED)

    Genesis-owned namespaces (vllm.sndr_core / vllm._genesis) are
    excluded — they are installed at deploy time, never present in a
    pristine candidate tree.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [{"module": None, "attr": None, "form": "unparseable"}]

    found: list[dict] = []
    seen: set[tuple[Optional[str], Optional[str]]] = set()

    def _add(module: Optional[str], attr: Optional[str], form: str) -> None:
        if module is not None:
            if not module.startswith("vllm"):
                return
            if module == "vllm" and attr is None:
                return  # bare `import vllm` — no binding to verify
            if any(module == p or module.startswith(p + ".")
                   for p in _SELF_PACKAGE_PREFIXES):
                return
        key = (module, attr)
        if key in seen:
            return
        seen.add(key)
        found.append({"module": module, "attr": attr, "form": form})

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "vllm" or node.module.startswith("vllm."):
                for alias in node.names:
                    _add(node.module, alias.name, "from_import")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("vllm."):
                    _add(alias.name, None, "import")
        elif isinstance(node, ast.Call):
            fn = node.func
            is_import_module = (
                (isinstance(fn, ast.Attribute) and fn.attr == "import_module")
                or (isinstance(fn, ast.Name) and fn.id == "import_module")
            )
            if is_import_module and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value.startswith("vllm"):
                        _add(arg.value, None, "import_module")
                else:
                    _add(None, None, "import_module_dynamic")
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if not any(("MODULE_PATHS" in n) or ("CANDIDATE" in n)
                       for n in names):
                continue
            for sub in ast.walk(node.value):
                if (isinstance(sub, ast.Constant)
                        and isinstance(sub.value, str)
                        and _VLLM_DOTTED_RE.match(sub.value)):
                    _add(sub.value, None, "module_paths_constant")
    return found


def check_binding(
    candidate_root: Path, module_path: Optional[str], attr: Optional[str],
) -> tuple[str, str]:
    """Static existence check of one (vllm module, attr) binding against
    the candidate tree. Returns (verdict, detail).

    NOTE: static check only — a symbol may exist while its call-site
    contract changed. Liveness needs in-container verification
    (VERIFY_IN_CONTAINER leg, planned v1.1).
    """
    if module_path is None:
        return BINDING_UNRESOLVED, "dynamic import — module path not a constant"
    parts = module_path.split(".")[1:]  # drop leading "vllm"
    base = candidate_root.joinpath(*parts) if parts else candidate_root
    mod_file = base.with_suffix(".py") if parts else None
    pkg_init = base / "__init__.py"
    target: Optional[Path] = None
    if mod_file is not None and mod_file.is_file():
        target = mod_file
    elif pkg_init.is_file():
        target = pkg_init
    elif not parts:
        target = candidate_root / "__init__.py"
        if not target.is_file():
            return BINDING_FILE_MISSING, "vllm/__init__.py absent"
    if target is None:
        return BINDING_FILE_MISSING, f"{'/'.join(parts)}(.py|/__init__.py) absent"

    if not attr:
        return BINDING_OK, f"module file present: {target.name}"

    # Submodule-as-attr: `from vllm.x import submodule`
    if base.is_dir():
        sub = base / f"{attr}.py"
        sub_pkg = base / attr / "__init__.py"
        if sub.is_file() or sub_pkg.is_file():
            return BINDING_OK, f"submodule {attr} present"

    try:
        text = target.read_text(encoding="utf-8", errors="ignore")
    except OSError as e:
        return BINDING_UNRESOLVED, f"unreadable: {e}"

    esc = re.escape(attr)
    patterns = (
        rf"\b(?:def|class)\s+{esc}\b",        # definition
        rf"^\s*{esc}\s*[:=]",                  # assignment / annotation
        rf"\bimport\s+{esc}\b",                # re-export `from .x import attr`
        rf"\bas\s+{esc}\b",                    # aliased import
        rf"[\"']{esc}[\"']",                    # __all__ / lazy-attr maps
    )
    for pat in patterns:
        if re.search(pat, text, re.MULTILINE):
            return BINDING_OK, f"symbol {attr} present (static)"
    return BINDING_SYMBOL_MISSING, f"symbol {attr} not found in {target.name}"


def _expand_patchers(result: Any) -> list[Any]:
    """A builder may return one patcher, None, or a list/tuple of them."""
    if result is None:
        return []
    if isinstance(result, (list, tuple)):
        return [p for p in result if p is not None]
    return [result]


_ENV_TRUTHY = ("1", "true", "yes", "on")


def _unset_env_flags(specs: Optional[Iterable[Any]]) -> list[str]:
    """Registry env flags from the module's specs whose current value
    would NOT pass the canonical wiring-module gate (absent or set to a
    non-truthy string — see e.g. pn204_dual_stream_inproj._enabled).
    An operator-disabled "0" still counts: preflight judges patch
    health against the candidate tree, not the opt-in state."""
    out: list[str] = []
    for spec in specs or []:
        flag = getattr(spec, "env_flag", None)
        if not flag or flag in out:
            continue
        if os.environ.get(flag, "").strip().lower() not in _ENV_TRUTHY:
            out.append(flag)
    return out


def _call_with_env_forced(fn: Any, kwargs: dict, flags: list[str]) -> Any:
    """Re-invoke a builder with every flag forced to "1", restoring the
    exact prior environment (including absence) in a finally block."""
    saved = {f: os.environ.get(f) for f in flags}
    try:
        for f in flags:
            os.environ[f] = "1"
        return fn(**kwargs)
    finally:
        for f, old in saved.items():
            if old is None:
                os.environ.pop(f, None)
            else:
                os.environ[f] = old


def _retired_sub_names(mod: Any) -> dict[str, str]:
    """v1.2 KNOWN_OPTIONAL_RETIRED convention: a wiring module declares
    subs retired-by-design through a ``*_RETIRED_SUBS`` attr holding the
    sub-patch names (e.g. P64_RETIRED_SUBS in p64_qwen3coder_mtp_
    streaming — anchors died with the dev259 serving.py refactor; zero
    matches is the journaled steady state, journal 2026-06-09 / commit
    630283ac). Returns sub name → declaring attr name."""
    out: dict[str, str] = {}
    for attr in dir(mod):
        if not attr.endswith("_RETIRED_SUBS"):
            continue
        val = getattr(mod, attr, None)
        if isinstance(val, (list, tuple, set, frozenset)):
            for n in val:
                if isinstance(n, str):
                    out[n] = attr
    return out


def _maybe_optional_retired(row: dict, patcher: Any,
                            retired_map: dict[str, str]) -> None:
    """Reclassify an all-optional zero-match DRIFT_ANCHOR row to
    KNOWN_OPTIONAL_RETIRED when EVERY sub is declared retired-by-design
    (partial declarations stay DRIFT_ANCHOR — fail-noisy)."""
    if row.get("verdict") != DRIFT_ANCHOR or row.get("applied_subs"):
        return
    subs = getattr(patcher, "sub_patches", None) or []
    if not subs or any(getattr(s, "required", True) for s in subs):
        return
    names = [getattr(s, "name", "") for s in subs]
    if not all(n in retired_map for n in names):
        return
    attrs = sorted({retired_map[n] for n in names})
    row["verdict"] = KNOWN_OPTIONAL_RETIRED
    row["detail"] = (
        "all optional sub-patches "
        f"{names} declared retired-by-design via module attr "
        f"{', '.join(attrs)} — zero anchor matches is the documented "
        "steady state, not drift"
    )


def _anchor_alternation_map(mod: Any) -> dict[str, dict[str, Any]]:
    """v1.3 EXPECTED_ALTERNATE convention: a wiring module with multiple
    per-pin factory variants of the SAME code site declares the
    alternation through a ``*_ANCHOR_ALTERNATION`` dict attr mapping
    each sub name to the sub names of its alternate variants (e.g.
    P91B_ANCHOR_ALTERNATION in p91b_autoround_row_group_cdiv_multi_
    scheme — dual inc.py factories dev338/dev371 by design, exactly one
    anchor matches on any live pin; Option A from the Step 0 anchor
    manifest). Returns sub name → {"alternates": tuple, "attr": name}."""
    out: dict[str, dict[str, Any]] = {}
    for attr in dir(mod):
        if not attr.endswith("_ANCHOR_ALTERNATION"):
            continue
        val = getattr(mod, attr, None)
        if not isinstance(val, dict):
            continue
        for name, alts in val.items():
            if not isinstance(name, str):
                continue
            if isinstance(alts, str):
                alts = (alts,)
            if not isinstance(alts, (list, tuple, set, frozenset)):
                continue
            clean = tuple(a for a in alts if isinstance(a, str))
            if clean:
                out[name] = {"alternates": clean, "attr": attr}
    return out


def _reclassify_expected_alternate(
    pairs: list[tuple[dict, Any]], alt_map: dict[str, dict[str, Any]],
) -> None:
    """Flip a zero-match DRIFT_ANCHOR row to EXPECTED_ALTERNATE when the
    module declares the alternation AND a declared alternate sub
    actually matched in a sibling row of the same module (the P91B
    dual-factory class). A miss on EVERY declared variant stays
    DRIFT_ANCHOR — fail-noisy, never fail-silent."""
    if not alt_map:
        return
    matched: set[str] = set()
    for row, _patcher in pairs:
        matched.update(row.get("applied_subs") or [])
    for row, patcher in pairs:
        if row.get("verdict") != DRIFT_ANCHOR or row.get("applied_subs"):
            continue
        subs = getattr(patcher, "sub_patches", None) or []
        names = [getattr(s, "name", "") for s in subs]
        if not names or not all(n in alt_map for n in names):
            continue
        hits: list[str] = []
        attrs: set[str] = set()
        for n in names:
            alt_hit = next(
                (a for a in alt_map[n]["alternates"] if a in matched), None)
            if alt_hit is None:
                hits = []
                break
            if alt_hit not in hits:
                hits.append(alt_hit)
            attrs.add(alt_map[n]["attr"])
        if not hits:
            continue
        row["verdict"] = EXPECTED_ALTERNATE
        row["alternate_matched"] = hits
        row["detail"] = (
            "anchor variant for another pin — alternate sub(s) "
            + ", ".join(repr(h) for h in hits)
            + " matched in this sweep; per-pin alternation declared via "
            "module attr " + ", ".join(sorted(attrs))
            + ", not drift"
        )


def evaluate_module(
    module_name: str, patch_ids: list[str], candidate_root: Path,
    specs: Optional[list[Any]] = None,
) -> list[dict]:
    """Evaluate one wiring module → list of report rows.

    Rows: one per built patcher (or one per builder for None/UNBUILDABLE),
    or a single IMPORT_FAIL / RUNTIME_BINDING row for the whole module.

    ``specs`` (v1.2): the module's PatchSpec objects, used for the
    env-forced builder retry — many builders early-exit on an unset
    GENESIS_* flag BEFORE resolving targets, which v1.1 misread as
    DRIFT_FILE_MOVED (triage 2026-06-11: PN96/PN73/PN91/PN92/PN204/
    SNDR_WORKSPACE_001 all healthy and PROD-applied).
    """
    base = {"patch_ids": patch_ids, "module": module_name}
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:  # noqa: BLE001 — report, never crash the sweep
        return [{**base, "verdict": IMPORT_FAIL,
                 "detail": f"{type(e).__name__}: {e}"}]

    builders = builder_names(mod)
    if not builders:
        md5_row = evaluate_md5_diff_patch(mod, candidate_root)
        if md5_row is not None:
            md5_row.update(base)
            return [md5_row]
        return [_runtime_binding_row(base, mod, candidate_root)]

    unset_flags = _unset_env_flags(specs)
    retired_map = _retired_sub_names(mod)
    alternation_map = _anchor_alternation_map(mod)

    rows: list[dict] = []
    alt_pairs: list[tuple[dict, Any]] = []
    for bname in builders:
        fn = getattr(mod, bname)
        missing = unfillable_params(fn)
        fill_kwargs: dict[str, Any] = {}
        filled_via: dict[str, str] = {}
        if missing:
            # v1.2 (P82 class): a required param <name> is fillable when
            # the module exposes a callable _read_<name> — the same
            # probe its own apply() uses (_read_threshold /
            # _read_min_draft_pos). Anything else stays UNBUILDABLE.
            readers = {n: getattr(mod, f"_read_{n}", None) for n in missing}
            unreadable = [n for n, f in readers.items() if not callable(f)]
            if unreadable:
                rows.append({
                    **base, "builder": bname, "verdict": UNBUILDABLE,
                    "detail": (
                        f"required params {unreadable} have no defaults and "
                        "no module _read_<param> probe — refusing to guess "
                        "(legacy-tool wart)"
                    ),
                })
                continue
            try:
                fill_kwargs = {n: f() for n, f in readers.items()}
            except Exception as e:  # noqa: BLE001
                rows.append({
                    **base, "builder": bname, "verdict": UNBUILDABLE,
                    "detail": (
                        "module _read_<param> probe raised: "
                        + ", ".join(f"_read_{n}" for n in missing)
                        + f" → {type(e).__name__}: {e}"
                    ),
                })
                continue
            filled_via = {n: f"_read_{n}" for n in missing}
        try:
            built = fn(**fill_kwargs)
        except Exception as e:  # noqa: BLE001
            rows.append({
                **base, "builder": bname, "verdict": UNBUILDABLE,
                "detail": f"builder raised {type(e).__name__}: {e}",
            })
            continue
        patchers = _expand_patchers(built)

        # v1.2: builder abstained and the registry flag is unset → the
        # abstain may be the module's own env gate, not file drift.
        # Retry once with the flag(s) forced, environment restored.
        env_forced = False
        if not patchers and unset_flags:
            env_forced = True
            try:
                built = _call_with_env_forced(fn, fill_kwargs, unset_flags)
            except Exception as e:  # noqa: BLE001
                rows.append({
                    **base, "builder": bname, "verdict": UNBUILDABLE,
                    "env_forced": True,
                    "detail": (f"builder raised under env-forced retry "
                               f"({unset_flags}): {type(e).__name__}: {e}"),
                })
                continue
            patchers = _expand_patchers(built)

        if not patchers:
            row = _builder_none_row(
                base, bname, mod, candidate_root,
                env_forced=env_forced, forced_flags=unset_flags)
            rows.append(row)
            continue
        for p in patchers:
            row = evaluate_patcher(p, candidate_root)
            row.update(base)
            row["builder"] = bname
            if env_forced:
                row["env_forced"] = True
                row["env_forced_flags"] = list(unset_flags)
            if filled_via:
                row["params_filled_via"] = filled_via
            _maybe_optional_retired(row, p, retired_map)
            alt_pairs.append((row, p))
            rows.append(row)
    _reclassify_expected_alternate(alt_pairs, alternation_map)
    return rows


def _builder_none_row(
    base: dict, bname: str, mod: Any, candidate_root: Path,
    *, env_forced: bool, forced_flags: list[str],
) -> dict:
    """Verdict for a builder that abstained (returned None) even after
    any env-forced retry. DRIFT_FILE_MOVED only when a resolve target
    is actually absent (plan §4.2); a post-retry abstain with every
    target present is ENV_GATED_ABSTAIN — gated on something beyond the
    registry flag, non-actionable."""
    resolved, unresolved = _resolve_target_status(mod, candidate_root)

    if env_forced and not unresolved:
        targets_note = (
            f"all resolve targets exist: {resolved}" if resolved
            else "no resolve_vllm_file literals found in module source"
        )
        return {
            **base, "builder": bname, "verdict": ENV_GATED_ABSTAIN,
            "env_forced": True, "env_forced_flags": list(forced_flags),
            "detail": (
                f"builder returned None even with {forced_flags} forced "
                f"to \"1\"; {targets_note} — abstain gated beyond the "
                "registry env_flag, not file drift"
            ),
        }

    detail = "builder returned None"
    if unresolved:
        detail += f" — unresolved target(s): {unresolved}"
        if resolved:
            # v1.2: the v1.1 canned "target not found" detail lied for
            # fallback-resolving modules (PN204 class) — say which side
            # is which.
            detail += f"; resolves via fallback: {resolved}"
    elif resolved:
        detail += (f" — all resolve targets exist: {resolved} "
                   "(non-file abstain; no registry env_flag to force)")
    else:
        detail += " — no resolve_vllm_file literals found in module source"
    if env_forced:
        detail += f"; env-forced retry with {forced_flags} also returned None"

    row = {**base, "builder": bname, "verdict": DRIFT_FILE_MOVED,
           "detail": detail}
    if env_forced:
        row["env_forced"] = True
        row["env_forced_flags"] = list(forced_flags)
    cands: list[str] = []
    for rel in unresolved:
        cands.extend(find_moved_candidates(
            candidate_root, missing_rel=rel, limit=3 - len(cands)))
    row["moved_to_candidates"] = cands[:3]
    return row


def _resolve_target_status(
    mod: Any, candidate_root: Path,
) -> tuple[list[str], list[str]]:
    """For a builder that returned None: statically pull the
    ``resolve_vllm_file("<rel>")`` string constants out of the module
    source and split them by existence under the candidate root.
    Returns (resolved, unresolved) in source order, deduplicated."""
    try:
        source = inspect.getsource(mod)
    except (OSError, TypeError):
        return [], []
    rels = re.findall(r"resolve_vllm_file\(\s*[\"']([^\"']+)[\"']", source)
    resolved: list[str] = []
    unresolved: list[str] = []
    for r in dict.fromkeys(rels):
        (resolved if (candidate_root / r).is_file() else unresolved).append(r)
    return resolved, unresolved


def _unresolved_resolve_targets(mod: Any, candidate_root: Path) -> list[str]:
    """v1.1-compat wrapper: relative targets that do NOT exist under the
    candidate root — the usual reason a builder abstains."""
    return _resolve_target_status(mod, candidate_root)[1]


def _runtime_binding_row(base: dict, mod: Any, candidate_root: Path) -> dict:
    """No text-patcher builder → static binding verdicts for the module."""
    row = {**base, "verdict": RUNTIME_BINDING, "bindings": [],
           "binding_ok": True,
           "detail": ("static binding check only — call-site liveness "
                      "needs VERIFY_IN_CONTAINER")}
    try:
        source = inspect.getsource(mod)
    except (OSError, TypeError) as e:
        row["binding_ok"] = True  # cannot judge — do not block on it
        row["detail"] = f"source unavailable for AST pass: {e}"
        return row
    for b in extract_bindings(source):
        verdict, detail = check_binding(candidate_root, b["module"], b["attr"])
        row["bindings"].append({**b, "verdict": verdict, "detail": detail})
        if verdict in (BINDING_FILE_MISSING, BINDING_SYMBOL_MISSING):
            row["binding_ok"] = False
    return row


# ─── tree-wide passes ─────────────────────────────────────────────────────


def check_upstream_markers(
    candidate_root: Path, markers: Optional[dict] = None,
) -> list[dict]:
    """UPSTREAM_MARKERS pass against the candidate tree. Mirrors
    ``tools/check_upstream_drift.py::_check_markers`` but resolves files
    against the extracted package root (``candidate_root/<rel>``)."""
    if markers is None:
        from sndr.engines.vllm.upstream_compat import UPSTREAM_MARKERS
        markers = UPSTREAM_MARKERS

    results: list[dict] = []
    for key, info in markers.items():
        files = info.get("files") or ([info["file"]] if "file" in info else [])
        marker_strings = []
        if info.get("marker"):
            marker_strings.append(info["marker"])
        for k in ("marker_decode", "marker_store"):
            if k in info:
                marker_strings.append(info[k])
        if not marker_strings or not files:
            continue

        per_marker = []
        for m in marker_strings:
            found_in: list[str] = []
            for rel in files:
                target = candidate_root / rel
                if not target.is_file():
                    continue
                try:
                    content = target.read_text(encoding="utf-8",
                                               errors="ignore")
                except OSError:
                    continue
                if m in content:
                    found_in.append(rel)
            per_marker.append({"marker": m, "found_in": found_in})

        verified_keys = [k for k in info if k.startswith("verified_in_main_")]
        already_known = any(info.get(k, False) for k in verified_keys)
        currently_present = any(m["found_in"] for m in per_marker)
        results.append({
            "key": key,
            "files": files,
            "marker_results": per_marker,
            "currently_present": currently_present,
            "already_known_merged": already_known,
            "newly_merged": currently_present and not already_known,
        })
    return results


def check_version_range(
    spec: Any, candidate_version: Optional[str],
) -> tuple[Optional[bool], str]:
    """Evaluate ``applies_to.vllm_version_range`` (tuple of specifiers or
    a single/comma string) against the candidate's internal version.

    Delegates to ``sndr.compat.version_check.check_version_constraints``
    so the verdict predicts EXACTLY what the dispatcher will do on the
    candidate pin (including its malformed-specifier = skip behavior).
    Returns (True | False | None-unknown, reason).
    """
    if candidate_version is None:
        return None, "candidate version unknown — provenance missing"
    try:
        from sndr.compat.version_check import (
            VersionProfile,
            check_version_constraints,
        )
        profile = VersionProfile(vllm=candidate_version)
        ok, results = check_version_constraints(
            {"vllm_version_range": spec}, profile=profile)
        reason = results[0].reason if results else ""
        return bool(ok), reason
    except Exception as e:  # noqa: BLE001 — degrade to unknown
        return None, f"version check failed: {e}"


def manifest_staleness(
    candidate_version: Optional[str],
    manifest_path: Optional[Path] = None,
) -> dict:
    """anchor_manifest.json pins.vllm vs candidate version. A stale
    manifest is harmless for preflight (we force GENESIS_NO_PATCH_CACHE
    and never use the offset fast-path) but the operator must regenerate
    it before promoting the pin."""
    if manifest_path is None:
        manifest_path = REPO_ROOT / "sndr" / "manifests" / "anchor_manifest.json"
    out: dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "manifest_vllm_pin": None,
        "candidate_version": candidate_version,
        "stale": None,
    }
    try:
        data = json.loads(Path(manifest_path).read_text())
        out["manifest_vllm_pin"] = (data.get("pins") or {}).get("vllm")
    except (OSError, ValueError) as e:
        out["error"] = f"manifest unreadable: {e}"
        return out
    if candidate_version and out["manifest_vllm_pin"]:
        out["stale"] = out["manifest_vllm_pin"] != candidate_version
    return out


def reclassify_chained(rows: list[dict]) -> int:
    """Patch-chain pass (P18B-on-PN119 class, found 2026-06-10).

    A patcher whose anchors target ANOTHER patcher's post-apply output
    misses every anchor on a pristine tree and reads as DRIFT_ANCHOR —
    a false positive. For each DRIFT_ANCHOR row, if EVERY missing
    anchor is a substring of a same-target sibling's replacement text,
    flip the verdict to CHAINED_ANCHOR (informational, not actionable)
    and record the provider patch ids in ``chained_on``.

    Strips the private ``_replacement_blob`` / ``_missing_anchors``
    working fields from all rows. Returns the number reclassified.

    Limitation (v1): providers that early-return before Layer 5 with
    dynamically-built replacements may not expose a blob; a chained
    dependent then stays DRIFT_ANCHOR (fail-noisy, never fail-silent).
    """
    providers: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("_replacement_blob"):
            providers.setdefault(r.get("target") or "", []).append(r)

    n = 0
    for r in rows:
        if r.get("verdict") != DRIFT_ANCHOR:
            continue
        missing = r.get("_missing_anchors") or []
        if not missing:
            continue
        sibs = [p for p in providers.get(r.get("target") or "", [])
                if p is not r]
        chained_on: list[str] = []
        for anchor in missing:
            hit = next(
                (p for p in sibs if anchor in p["_replacement_blob"]), None)
            if hit is None:
                chained_on = []
                break
            name = (",".join(hit.get("patch_ids") or [])
                    or hit.get("patch_name") or "?")
            if name not in chained_on:
                chained_on.append(name)
        if chained_on:
            r["verdict"] = CHAINED_ANCHOR
            r["chained_on"] = chained_on
            r["detail"] = (
                "anchors target post-apply content of: "
                + ", ".join(chained_on)
                + " — apply-order chain, not upstream drift; verify the "
                "provider's own verdict instead"
            )
            n += 1

    for r in rows:
        r.pop("_replacement_blob", None)
        r.pop("_missing_anchors", None)
    return n


def classify_out_of_range(
    pids: list[str], registry: Optional[dict] = None,
) -> list[dict]:
    """Annotate out-of-range patch ids with the dispatcher's ACTUAL
    enforcement semantics (dispatcher/decision.py should_apply, rule 1):

    - default_on=True  → STRICT_SKIP: the version gate hard-skips the
      patch on the candidate pin. It is silently disabled.
    - default_on=False → ENV_OVERRIDE_POSSIBLE: a truthy env flag wins
      over applies_to, so the patch STILL APPLIES wherever the operator
      opted in (live counter-example: P67/P82 on 0.22.1 with <0.22.0
      ranges, PROD 2026-06-10). The stale range only degrades doctor /
      recommend diagnostics — bump it after validating on the new pin.
    """
    if registry is None:
        from sndr.dispatcher.registry import PATCH_REGISTRY
        registry = PATCH_REGISTRY
    out = []
    for pid in pids:
        meta = registry.get(pid)
        if meta is None:
            out.append({"patch_id": pid, "default_on": None,
                        "env_flag": None, "lifecycle": None,
                        "enforcement": "UNKNOWN_PATCH"})
            continue
        default_on = bool(meta.get("default_on", False))
        out.append({
            "patch_id": pid,
            "default_on": default_on,
            "env_flag": meta.get("env_flag"),
            "lifecycle": meta.get("lifecycle"),
            "enforcement": ("STRICT_SKIP" if default_on
                            else "ENV_OVERRIDE_POSSIBLE"),
        })
    return out


# ─── summary / driver ─────────────────────────────────────────────────────


def count_actionable(rows: list[dict], markers: list[dict]) -> int:
    """Actionable = needs operator action before the candidate can be
    promoted. Out-of-version-range patches are excluded (the dispatcher
    will skip them on the candidate pin); BINDING_UNRESOLVED is
    informational (static-analysis limitation, not candidate breakage);
    newly-merged upstream markers feed the iron-rule-#11 retire queue."""
    n = 0
    for r in rows:
        if r.get("in_version_range") is False:
            continue
        v = r.get("verdict")
        if v in ACTIONABLE_VERDICTS:
            n += 1
        elif v == RUNTIME_BINDING and r.get("binding_ok") is False:
            n += 1
    n += sum(1 for m in markers if m.get("newly_merged"))
    return n


def _load_provenance(candidate_root: Path,
                     explicit: Optional[str]) -> Optional[dict]:
    paths = []
    if explicit:
        paths.append(Path(explicit))
    else:
        paths.append(candidate_root / "PROVENANCE.json")
        paths.append(candidate_root.parent / "PROVENANCE.json")
    for p in paths:
        if p.is_file():
            try:
                return json.loads(p.read_text())
            except ValueError as e:
                raise SystemExit(
                    f"pin_preflight: provenance {p} is not valid JSON: {e}")
    if explicit:
        raise SystemExit(f"pin_preflight: provenance not found: {explicit}")
    return None


# Lifecycles that never apply at runtime regardless of the (sometimes
# contradictory) explicit implementation_status field. Observed on the
# current pin: 7 _archive entries carry lifecycle=retired WITH explicit
# implementation_status=full — they would otherwise pollute the sweep.
_EXCLUDED_LIFECYCLES = frozenset({"retired", "deprecated"})


def spec_in_scope(spec: Any) -> bool:
    """True iff a PatchSpec participates in the preflight sweep."""
    if not getattr(spec, "apply_module", None):
        return False
    if getattr(spec, "lifecycle", "") in _EXCLUDED_LIFECYCLES:
        return False
    return getattr(spec, "implementation_status", None) in TEXT_PATCH_STATES


def _iter_filtered_specs() -> tuple[dict[str, dict[str, Any]], list[str]]:
    from sndr.dispatcher.spec import iter_patch_specs
    by_module: dict[str, dict[str, Any]] = {}
    retired_excluded: list[str] = []
    for spec in iter_patch_specs():
        if (getattr(spec, "lifecycle", "") in _EXCLUDED_LIFECYCLES
                and spec.implementation_status in TEXT_PATCH_STATES
                and spec.apply_module):
            retired_excluded.append(spec.patch_id)
        if not spec_in_scope(spec):
            continue
        entry = by_module.setdefault(
            spec.apply_module, {"patch_ids": [], "specs": []})
        entry["patch_ids"].append(spec.patch_id)
        entry["specs"].append(spec)
    return by_module, retired_excluded


def run_preflight(
    candidate_root: Path,
    provenance: Optional[dict] = None,
    fail_fast: bool = False,
) -> dict:
    """Full sweep → report dict (no I/O besides reads)."""
    import sndr.engines.vllm.detection.guards as guards

    candidate_root = Path(candidate_root).resolve()
    candidate_version = (provenance or {}).get("internal_version")

    # Redirect EVERY wiring module's target resolution to the candidate
    # tree. resolve_vllm_file() looks vllm_install_root up through the
    # guards module attribute (sys.modules self-dispatch, guards.py
    # resolve_vllm_file), so one assignment covers all 218 modules.
    guards.vllm_install_root = lambda: str(candidate_root)

    by_module, retired_excluded = _iter_filtered_specs()
    rows: list[dict] = []
    lint: list[dict] = []
    for module_name in sorted(by_module):
        info = by_module[module_name]
        module_rows = evaluate_module(
            module_name, sorted(set(info["patch_ids"])), candidate_root,
            specs=info["specs"])

        # version-range gate: a module row is in range if ANY of its
        # patch_ids is in range for the candidate (compound modules).
        ranges = {}
        for spec in info["specs"]:
            vr = (spec.applies_to or {}).get("vllm_version_range")
            if vr is None:
                ranges[spec.patch_id] = (True, "no range declared")
            else:
                ranges[spec.patch_id] = check_version_range(
                    vr, candidate_version)
        in_range_vals = [v for v, _ in ranges.values()]
        module_in_range: Optional[bool]
        if any(v is True for v in in_range_vals):
            module_in_range = True
        elif all(v is False for v in in_range_vals):
            module_in_range = False
        else:
            module_in_range = None
        for r in module_rows:
            r["in_version_range"] = module_in_range
            r["version_ranges"] = {
                pid: {"in_range": v, "reason": reason}
                for pid, (v, reason) in ranges.items()
            }
        rows.extend(module_rows)

        # Self-collision lint on every buildable patcher (re-runs the
        # builders — cheap, and import is already cached).
        try:
            mod = importlib.import_module(module_name)
            for bname in builder_names(mod):
                if unfillable_params(getattr(mod, bname)):
                    continue
                try:
                    for p in _expand_patchers(getattr(mod, bname)()):
                        for f in self_collision_findings(p):
                            lint.append({**f, "module": module_name,
                                         "builder": bname,
                                         "patch_ids": sorted(set(info["patch_ids"]))})
                except Exception:  # noqa: BLE001 — builders already reported
                    pass
        except Exception:  # noqa: BLE001 — import already reported
            pass

        if fail_fast and count_actionable(rows, []):
            break

    chained = reclassify_chained(rows)

    markers = check_upstream_markers(candidate_root)
    out_of_range = sorted({
        pid for r in rows
        for pid, st in (r.get("version_ranges") or {}).items()
        if st["in_range"] is False
    })

    buckets: dict[str, int] = {}
    for r in rows:
        buckets[r["verdict"]] = buckets.get(r["verdict"], 0) + 1
    binding_fail = sum(
        1 for r in rows
        if r.get("verdict") == RUNTIME_BINDING and r.get("binding_ok") is False
    )

    report = {
        "tool": "pin_preflight v1.3",
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidate_root": str(candidate_root),
        "provenance": provenance,
        "manifest": manifest_staleness(candidate_version),
        "rows": rows,
        "self_collision_lint": lint,
        "upstream_markers": markers,
        "summary": {
            "modules_checked": len(by_module),
            "retired_excluded": sorted(retired_excluded),
            "rows": len(rows),
            "buckets": dict(sorted(buckets.items())),
            "binding_failures": binding_fail,
            "self_collision_findings": len(lint),
            "self_collision_undefended": sum(
                1 for f in lint if not f.get("defended")),
            "upstream_markers_checked": len(markers),
            "newly_merged_markers": sum(
                1 for m in markers if m.get("newly_merged")),
            "out_of_range_patches": out_of_range,
            "out_of_range_detail": classify_out_of_range(out_of_range),
            "chained_anchors": chained,
            "actionable": count_actionable(rows, markers),
        },
        "notes": [
            "READ-ONLY: TextPatcher.apply() never invoked; candidate tree unmodified.",
            "RUNTIME_BINDING rows are static-only; call-site liveness "
            "requires the VERIFY_IN_CONTAINER leg (v1.1).",
            "Out-of-range enforcement differs by default_on (decision.py "
            "rule 1): default_on=True → STRICT_SKIP (silently disabled on "
            "the candidate); opt-in + truthy env flag → STILL APPLIES "
            "(operator override wins). Bump ranges after validating.",
        ],
    }
    return report


def _print_human(report: dict) -> None:
    s = report["summary"]
    err = sys.stderr
    print("=" * 72, file=err)
    prov = report.get("provenance") or {}
    print(f"Pin preflight — candidate {prov.get('image_ref', '?')} "
          f"version {prov.get('internal_version', '?')}", file=err)
    print(f"root: {report['candidate_root']}", file=err)
    m = report["manifest"]
    print(f"anchor manifest pin: {m.get('manifest_vllm_pin')} "
          f"(stale={m.get('stale')})", file=err)
    print("=" * 72, file=err)
    print(f"{'BUCKET':24} COUNT", file=err)
    for k, v in s["buckets"].items():
        print(f"{k:24} {v}", file=err)
    print("-" * 72, file=err)
    for r in report["rows"]:
        if r["verdict"] in (OK,):
            continue
        if r["verdict"] == RUNTIME_BINDING and r.get("binding_ok", True):
            continue
        ids = ",".join(r.get("patch_ids", []))
        print(f"  {r['verdict']:20} {ids:14} {r.get('module', '')}", file=err)
        detail = r.get("detail", "")
        if detail:
            print(f"      {detail[:200]}", file=err)
        for c in r.get("moved_to_candidates", []) or []:
            print(f"      moved-to candidate: {c}", file=err)
        if r["verdict"] == RUNTIME_BINDING:
            for b in r.get("bindings", []):
                if b["verdict"] != BINDING_OK:
                    print(f"      {b['verdict']}: {b['module']}"
                          f"{'.' + b['attr'] if b['attr'] else ''}", file=err)
    if report["self_collision_lint"]:
        print("-" * 72, file=err)
        print("SELF_COLLISION_RISK lint:", file=err)
        for f in report["self_collision_lint"]:
            tag = "defended" if f["defended"] else "RISK"
            print(f"  [{tag}] {','.join(f['patch_ids'])} marker "
                  f"{f['marker']!r} in own {f['collides_with']}", file=err)
    newly = [mk for mk in report["upstream_markers"] if mk.get("newly_merged")]
    if newly:
        print("-" * 72, file=err)
        print("Newly-merged upstream markers (iron-rule-#11 queue):", file=err)
        for mk in newly:
            print(f"  {mk['key']}", file=err)
    if s["out_of_range_patches"]:
        print("-" * 72, file=err)
        detail = s.get("out_of_range_detail") or []
        strict = [d["patch_id"] for d in detail
                  if d["enforcement"] == "STRICT_SKIP"]
        overridable = [d["patch_id"] for d in detail
                       if d["enforcement"] == "ENV_OVERRIDE_POSSIBLE"]
        if strict:
            print(f"Out-of-range, default_on — SILENTLY DISABLED on "
                  f"candidate: {', '.join(strict)}", file=err)
        if overridable:
            print(f"Out-of-range, opt-in — env flag still overrides "
                  f"(stale metadata; bump after validation): "
                  f"{', '.join(overridable)}", file=err)
    print("=" * 72, file=err)
    print(f"modules={s['modules_checked']} rows={s['rows']} "
          f"actionable={s['actionable']}", file=err)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only pin-bump preflight against a candidate vllm tree.")
    parser.add_argument("candidate_root",
                        help="dir containing the vllm package contents "
                             "(<root>/v1 must exist)")
    parser.add_argument("--provenance", default=None,
                        help="PROVENANCE.json path (default: "
                             "<root>/PROVENANCE.json or <root>/../PROVENANCE.json)")
    parser.add_argument("--json-out", default=None,
                        help="also write the JSON report to this path")
    parser.add_argument("--fail-fast", action="store_true",
                        help="stop the sweep at the first actionable verdict")
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        return 2

    candidate_root = Path(args.candidate_root).resolve()
    if not (candidate_root / "v1").is_dir():
        print(f"pin_preflight: {candidate_root} does not look like an "
              "extracted vllm package (no v1/ subdir). Pass the directory "
              "that CONTAINS the vllm package contents, e.g. "
              "<staging>/vllm from extract_candidate_tree.sh.",
              file=sys.stderr)
        return 2

    try:
        provenance = _load_provenance(candidate_root, args.provenance)
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 2

    try:
        report = run_preflight(candidate_root, provenance=provenance,
                               fail_fast=args.fail_fast)
    except Exception:  # noqa: BLE001 — invocation-level failure
        traceback.print_exc()
        return 2

    payload = json.dumps(report, indent=2, default=str)
    print(payload)
    if args.json_out:
        Path(args.json_out).write_text(payload + "\n")
    _print_human(report)
    return 1 if report["summary"]["actionable"] else 0


if __name__ == "__main__":
    sys.exit(main())
