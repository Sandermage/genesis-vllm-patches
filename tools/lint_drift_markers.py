#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Self-collision drift-marker lint (the PN369 false-skip class).

Per the §6 action plan
(docs/superpowers/journal/2026-06-11-preflight-residual-triage-action-plan.md,
"Actions" item 1): a patcher whose ``upstream_drift_markers`` contain a
string that the patch ITSELF writes into the target file will read its
own output as "upstream merged" on the next boot and silently skip —
``TextPatcher.apply()`` checks idempotency (Layer 2) before drift
(Layer 3) with no ``[Genesis`` skip, and the marker line it prepends is
(``sndr/kernel/text_patch.py``, Layer 6, quoted for evidence):

    marker_line = f"# [Genesis wiring marker: {self.marker}]\\n"

So this lint REJECTS any ``upstream_drift_marker`` that is a substring of

  (a) any of the patcher's own sub-patch replacement texts, or
  (b) its idempotency marker LINE (the Layer-6 prepend above),

EXCEPT markers starting with ``[Genesis`` — the defended convention
(custom apply() wrappers skip them; the PN353A fix standardizes on it).
Observed live instances of the class: PN369/P71 (2026-06-10), PN353A
(§1), PN54, PN55 (§2), and the PN118 marker re-plant.

Standalone by design — ``tools/pin_preflight.py`` carries an
informational copy of this check inside its sweep, but THIS tool is the
enforcing gate (exit 1) with allowlist support so remediation of the
existing backlog can land incrementally. The enumeration helpers mirror
``tools/pin_preflight.py`` (builder discovery + the
``guards.vllm_install_root`` redirection seam) instead of importing it,
to keep the two tools free of file-level coupling.

Allowlist: ``tools/lint_drift_markers_allowlist.txt`` — one exact marker
string per line; every entry MUST sit under a justification comment
block (lines starting with ``#``); a blank line ends the block. Entries
without a justification are a hard config error (exit 2).

Usage:
    python3 tools/lint_drift_markers.py <candidate_root> \
        [--allowlist <path>] [--json-out <path>]

``<candidate_root>`` is the directory CONTAINING the vllm package
contents (``<candidate_root>/v1`` must exist — same contract as
``tools/pin_preflight.py``); builders resolve their targets against it.

Exit codes:
    0 — no violations (clean, or every finding allowlisted)
    1 — at least one undefended, non-allowlisted collision
    2 — invocation/config error (bad root, malformed allowlist,
        zero patchers built — a lint that checks nothing must not pass)

Fully offline — no gh API, no docker, no SSH; never writes to the tree.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import os

# MUST be set before any sndr import — kills the Layer-0 file_cache
# fast-path (same preamble as tools/pin_preflight.py).
os.environ["GENESIS_NO_PATCH_CACHE"] = "1"

import argparse
import importlib
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Defended convention: ``[Genesis``-prefixed drift markers are the
# sanctioned self-referencing form — exempt from this lint.
DEFENDED_PREFIX = "[Genesis"

DEFAULT_ALLOWLIST = HERE / "lint_drift_markers_allowlist.txt"

# Cross-patch anchor-overlap allowlist (deep-audit #2). One ``PID_A|PID_B``
# pair (patch ids sorted) per line; ``#``-comment justification blocks per the
# same format as the drift-marker allowlist. A pair listed here is a reviewed,
# order-dependent or otherwise-tolerated overlap — surfaced by the lint but not
# a build failure. Pairs that genuinely cannot coexist should instead get a
# registry conflicts_with declaration (which the lint excludes automatically).
DEFAULT_CROSS_PATCH_ALLOWLIST = HERE / "lint_cross_patch_allowlist.txt"


def cross_patch_pair_key(patch_a: str, patch_b: str) -> str:
    """Canonical ``PID_A|PID_B`` key with the two ids sorted."""
    return "|".join(sorted((patch_a, patch_b)))

# Mirrors tools/pin_preflight.py — implementation states whose wiring
# actually applies against the vllm tree, and lifecycles that never do.
TEXT_PATCH_STATES = frozenset({"live", "full", "text_patch", "runtime_hook"})
EXCLUDED_LIFECYCLES = frozenset({"retired", "deprecated"})


# ─── core lint ────────────────────────────────────────────────────────────


def idempotency_marker_line(marker: str) -> str:
    """The exact line TextPatcher Layer 6 prepends to a patched file —
    byte-for-byte mirror of sndr/kernel/text_patch.py:
    ``marker_line = f"# [Genesis wiring marker: {self.marker}]\\n"``.
    Pinned by the test_marker_line_fixed_prefix_collision contract."""
    return f"# [Genesis wiring marker: {marker}]\n"


def collisions_for_patcher(patcher: Any) -> list[dict]:
    """All self-collisions for one patcher (one finding per marker).

    A drift marker is a collision when it is a substring of the
    patcher's own emitted text: any sub-patch replacement, or the
    Layer-6 idempotency marker LINE (NOT just the raw marker — the
    line's constant prefix is emitted text too). ``[Genesis``-prefixed
    markers are exempt (defended convention)."""
    findings: list[dict] = []
    marker = getattr(patcher, "marker", "") or ""
    mline = idempotency_marker_line(marker) if marker else ""
    subs = getattr(patcher, "sub_patches", None) or []
    for dm in getattr(patcher, "upstream_drift_markers", None) or []:
        if not dm or dm.startswith(DEFENDED_PREFIX):
            continue
        collides_with: list[str] = []
        if mline and dm in mline:
            collides_with.append("idempotency_marker_line")
        colliding_subs = [
            getattr(sp, "name", "?") for sp in subs
            if dm in (getattr(sp, "replacement", "") or "")
        ]
        if colliding_subs:
            collides_with.append("replacement")
        if collides_with:
            findings.append({
                "patch_name": getattr(patcher, "patch_name", "?"),
                "marker": dm,
                "collides_with": collides_with,
                "colliding_subs": colliding_subs,
            })
    return findings


# ─── cross-patch anchor-overlap detection (deep-audit 2026-06-14 #2) ──────
#
# collisions_for_patcher() above catches a patcher colliding with ITSELF.
# This section catches a DIFFERENT, previously-invisible class: two SEPARATE
# patches whose REQUIRED anchors overlap the same byte span in one upstream
# file. Whichever registers first rewrites the shared region, so the other's
# required anchor vanishes and that patch boots FAILED with the file
# half-patched (the P23_WIRE / PN368 marlin_moe.py case). The orchestrator
# dependency validator and audit.validate_apply_plan are conflicts_with-only,
# so an UNDECLARED overlap is invisible to every existing guard. This makes
# it a build failure unless the pair is declared mutually exclusive.


def _declared_conflict_pairs() -> set:
    """Symmetric ``{patch_a, patch_b}`` frozenset pairs declared via
    ``conflicts_with`` anywhere in the registry."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    pairs: set = set()
    for pid, meta in PATCH_REGISTRY.items():
        for other in meta.get("conflicts_with", []) or []:
            pairs.add(frozenset((pid, other)))
    return pairs


def _required_subpatches(patcher: Any) -> list[dict]:
    """(anchor, replacement, name) of a patcher's REQUIRED sub-patches —
    failure of one aborts the group, so it is a required anchor's destruction
    that causes the silent half-patched boot."""
    out: list[dict] = []
    for sp in getattr(patcher, "sub_patches", None) or []:
        if getattr(sp, "required", False):
            a = getattr(sp, "anchor", "") or ""
            if a:
                out.append({
                    "anchor": a,
                    "replacement": getattr(sp, "replacement", "") or "",
                    "name": getattr(sp, "name", "?"),
                })
    return out


def locate_anchor_spans(file_content: str, anchors: list[dict]) -> list[dict]:
    """Locate each anchor's UNIQUE byte span in the pristine file content.

    ``anchors``: ``[{patch_id, label, anchor}]``. Returns only anchors that
    occur EXACTLY once, each annotated with ``start``/``end``. Anchors absent
    (drift) or ambiguous (count != 1) are dropped — those are other lints'
    concern, and a non-located anchor cannot be reasoned about positionally.
    """
    located: list[dict] = []
    for a in anchors:
        s = a["anchor"]
        if file_content.count(s) != 1:
            continue
        start = file_content.find(s)
        located.append({**a, "start": start, "end": start + len(s)})
    return located


def find_destructive_collisions(
    file_content: str, patches: list[dict], declared: set,
) -> list[dict]:
    """Pure: cross-patch collisions that are ORDER-DEPENDENT FRAGILE.

    A mere span overlap is NOT enough — TextPatcher applies sub-patches
    sequentially, so two patches whose anchors overlap compose fine as long as
    the earlier one's REPLACEMENT preserves the later one's anchor (intentional
    chaining). The real footgun is when applying one patch removes the other's
    (unique) anchor: then whichever registers first silently makes the other
    boot FAILED. We test BOTH orders against the pristine file.

    ``patches``: ``[{patch_id, label, anchor, replacement}]``. Only anchors
    that occur EXACTLY once in the pristine file are considered (a non-unique
    anchor is a separate lint). A pair declared mutually exclusive
    (``conflicts_with``) is handled by the dispatcher and excluded.
    """
    located = [p for p in patches if file_content.count(p["anchor"]) == 1]
    findings: list[dict] = []
    n = len(located)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = located[i], located[j]
            if a["patch_id"] == b["patch_id"]:
                continue
            if frozenset((a["patch_id"], b["patch_id"])) in declared:
                continue
            after_a = file_content.replace(a["anchor"], a["replacement"], 1)
            after_b = file_content.replace(b["anchor"], b["replacement"], 1)
            a_breaks_b = after_a.count(b["anchor"]) != 1
            b_breaks_a = after_b.count(a["anchor"]) != 1
            if a_breaks_b or b_breaks_a:
                findings.append({
                    "patch_a": a["patch_id"], "anchor_a": a["label"],
                    "patch_b": b["patch_id"], "anchor_b": b["label"],
                    "a_breaks_b": a_breaks_b, "b_breaks_a": b_breaks_a,
                })
    return findings


def collect_cross_patch_collisions(
    entries: list, declared: set,
) -> list[dict]:
    """Read each target file once, locate every patch's required anchors, and
    return undeclared cross-patch span overlaps. Needs the pristine tree
    (``target_file`` resolved into the candidate_root by the caller)."""
    by_file: dict[str, list[dict]] = {}
    for module_name, bname, patch_ids, patcher in entries:
        target = getattr(patcher, "target_file", None)
        if not target:
            continue
        pid = patch_ids[0] if patch_ids else module_name
        for sub in _required_subpatches(patcher):
            by_file.setdefault(target, []).append({
                "patch_id": pid, "label": sub["name"],
                "anchor": sub["anchor"], "replacement": sub["replacement"],
            })
    findings: list[dict] = []
    for target, patches in by_file.items():
        try:
            content = Path(target).read_text()
        except OSError:
            continue
        for f in find_destructive_collisions(content, patches, declared):
            findings.append({**f, "file": target})
    return findings


# ─── enumeration (mirrors tools/pin_preflight.py builder discovery) ───────


def _builder_names(mod: Any) -> list[str]:
    """Text-patcher builder callables: ``_make*patcher`` convention."""
    return sorted(
        n for n in dir(mod)
        if n.startswith("_make") and n.endswith("patcher")
        and callable(getattr(mod, n, None))
    )


def _unfillable_params(fn: Any) -> list[str]:
    """Required parameters with no default — we refuse to guess values."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return []
    return [
        name for name, p in sig.parameters.items()
        if p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD)
        and p.default is inspect.Parameter.empty
    ]


def _expand_patchers(result: Any) -> list[Any]:
    """A builder may return one patcher, None, or a list/tuple of them."""
    if result is None:
        return []
    if isinstance(result, (list, tuple)):
        return [p for p in result if p is not None]
    return [result]


def iter_buildable_patchers(
    candidate_root: Path,
    failures: Optional[dict[str, list[str]]] = None,
) -> Iterator[tuple[str, str, list[str], Any]]:
    """Yield (module, builder, patch_ids, patcher) for every in-scope
    spec whose wiring builds torch-less against ``candidate_root``.

    Redirects ``guards.vllm_install_root`` so every builder's
    ``resolve_vllm_file()`` resolves into the candidate tree (the same
    one-assignment seam pin_preflight uses). Import/builder failures
    are collected into ``failures`` (informational here — the preflight
    owns their reporting) and never abort the sweep."""
    import sndr.engines.vllm.detection.guards as guards
    guards.vllm_install_root = lambda: str(candidate_root)

    from sndr.dispatcher.spec import iter_patch_specs

    by_module: dict[str, list[str]] = {}
    for spec in iter_patch_specs():
        if not getattr(spec, "apply_module", None):
            continue
        if getattr(spec, "lifecycle", "") in EXCLUDED_LIFECYCLES:
            continue
        if getattr(spec, "implementation_status", None) not in TEXT_PATCH_STATES:
            continue
        by_module.setdefault(spec.apply_module, []).append(spec.patch_id)

    for module_name in sorted(by_module):
        patch_ids = sorted(set(by_module[module_name]))
        try:
            mod = importlib.import_module(module_name)
        except Exception as e:  # noqa: BLE001 — collect, never crash
            if failures is not None:
                failures.setdefault("import", []).append(
                    f"{module_name}: {type(e).__name__}: {e}")
            continue
        for bname in _builder_names(mod):
            fn = getattr(mod, bname)
            if _unfillable_params(fn):
                continue  # UNBUILDABLE — preflight reports these
            try:
                built = fn()
            except Exception as e:  # noqa: BLE001
                if failures is not None:
                    failures.setdefault("builder", []).append(
                        f"{module_name}.{bname}: {type(e).__name__}: {e}")
                continue
            for p in _expand_patchers(built):
                yield module_name, bname, patch_ids, p


# ─── allowlist ────────────────────────────────────────────────────────────


def parse_allowlist(path: Path) -> tuple[list[str], list[str]]:
    """Parse the allowlist → (markers, errors).

    Format: one exact marker string per line (trailing newline stripped,
    inner whitespace preserved). Lines whose first non-blank character
    is ``#`` are justification comments; a comment block covers every
    marker line until the next blank line. A marker with NO active
    justification block is an error — unexplained suppressions defeat
    the lint. Limitations (by design, keep the format trivial): markers
    that themselves start with ``#`` or contain a newline cannot be
    allowlisted here — fix the patch instead."""
    markers: list[str] = []
    errors: list[str] = []
    if not path.is_file():
        return markers, errors
    justified = False
    for lineno, raw in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            justified = False
            continue
        if raw.lstrip().startswith("#"):
            justified = True
            continue
        if not justified:
            errors.append(
                f"{path.name}:{lineno}: entry {raw!r} has no justification "
                "comment — add a '#' block above it")
            continue
        markers.append(raw.rstrip("\n"))
    return markers, errors


# ─── report assembly / exit semantics ─────────────────────────────────────


def run_lint(
    entries: Iterable[tuple[str, str, list[str], Any]],
    allowlist: Iterable[str],
    candidate_root: Optional[Path] = None,
    failures: Optional[dict[str, list[str]]] = None,
) -> dict:
    """Lint every enumerated patcher → report dict (pure, no I/O)."""
    allowset = set(allowlist)
    violations: list[dict] = []
    allowlisted: list[dict] = []
    seen_markers: set[str] = set()
    patchers_checked = 0
    for module_name, bname, patch_ids, patcher in entries:
        patchers_checked += 1
        for f in collisions_for_patcher(patcher):
            row = {**f, "module": module_name, "builder": bname,
                   "patch_ids": patch_ids}
            seen_markers.add(f["marker"])
            (allowlisted if f["marker"] in allowset else violations).append(row)

    failures = failures or {}
    return {
        "tool": "lint_drift_markers v1",
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidate_root": str(candidate_root) if candidate_root else None,
        "violations": violations,
        "allowlisted": allowlisted,
        # Allowlist entries that matched nothing this run — remediated
        # or renamed markers; prune them (informational, never fatal).
        "stale_allowlist": sorted(allowset - seen_markers),
        "summary": {
            "patchers_checked": patchers_checked,
            "findings": len(violations) + len(allowlisted),
            "violations": len(violations),
            "allowlisted": len(allowlisted),
            "import_failures": failures.get("import", []),
            "builder_failures": failures.get("builder", []),
        },
    }


def decide_exit(report: dict) -> int:
    """0 clean, 1 violations, 2 empty sweep (checked nothing ≠ passed).

    Either self-collision violations OR undeclared cross-patch anchor
    overlaps fail the lint."""
    if report["summary"]["patchers_checked"] == 0:
        return 2
    if report["summary"]["violations"]:
        return 1
    if report["summary"].get("cross_patch_collisions", 0):
        return 1
    return 0


def _print_human(report: dict) -> None:
    s = report["summary"]
    err = sys.stderr
    print("=" * 72, file=err)
    print(f"Drift-marker self-collision lint — root {report['candidate_root']}",
          file=err)
    print(f"patchers={s['patchers_checked']} findings={s['findings']} "
          f"violations={s['violations']} allowlisted={s['allowlisted']}",
          file=err)
    print(f"cross-patch: violations={s.get('cross_patch_collisions', 0)} "
          f"allowlisted={s.get('cross_patch_allowlisted', 0)}", file=err)
    if report["violations"]:
        print("-" * 72, file=err)
        print(f"{'PATCH_IDS':18} {'COLLIDES_WITH':34} MARKER", file=err)
        for f in report["violations"]:
            where = ",".join(f["collides_with"])
            if f["colliding_subs"]:
                where += f" ({','.join(f['colliding_subs'])})"
            print(f"{','.join(f['patch_ids']):18} {where:34} "
                  f"{f['marker']!r}", file=err)
    cross = report.get("cross_patch_collisions") or []
    if cross:
        print("-" * 72, file=err)
        print(f"UNDECLARED cross-patch anchor overlaps ({len(cross)}) — two "
              "patches rewrite the same byte span; whichever applies first "
              "leaves the other half-patched. Declare conflicts_with or "
              "disambiguate the anchor:", file=err)
        for c in cross:
            print(f"  {c['patch_a']} ({c['anchor_a']}) ⨯ {c['patch_b']} "
                  f"({c['anchor_b']})\n      file: {c['file']} "
                  f"overlap@{c['overlap']}", file=err)
    if report["stale_allowlist"]:
        print("-" * 72, file=err)
        print("Stale allowlist entries (no longer collide — prune):", file=err)
        for m in report["stale_allowlist"]:
            print(f"  {m!r}", file=err)
    for kind in ("import_failures", "builder_failures"):
        if s[kind]:
            print("-" * 72, file=err)
            print(f"{kind} (informational — pin_preflight reports these):",
                  file=err)
            for line in s[kind]:
                print(f"  {line}", file=err)
    print("=" * 72, file=err)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reject upstream_drift_markers that match the "
                    "patcher's own emitted text (PN369 false-skip class).")
    parser.add_argument("candidate_root",
                        help="dir containing the vllm package contents "
                             "(<root>/v1 must exist)")
    parser.add_argument("--allowlist", default=str(DEFAULT_ALLOWLIST),
                        help="allowlist path (default: "
                             "tools/lint_drift_markers_allowlist.txt)")
    parser.add_argument("--json-out", default=None,
                        help="also write the JSON report to this path")
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        return 2

    candidate_root = Path(args.candidate_root).resolve()
    if not (candidate_root / "v1").is_dir():
        print(f"lint_drift_markers: {candidate_root} does not look like an "
              "extracted vllm package (no v1/ subdir) — same contract as "
              "tools/pin_preflight.py.", file=sys.stderr)
        return 2

    allow_markers, allow_errors = parse_allowlist(Path(args.allowlist))
    if allow_errors:
        for e in allow_errors:
            print(f"lint_drift_markers: allowlist error: {e}", file=sys.stderr)
        return 2

    failures: dict[str, list[str]] = {}
    # Materialize so the same enumeration feeds both the self-collision lint
    # and the cross-patch anchor-overlap check.
    entries = list(iter_buildable_patchers(candidate_root, failures=failures))
    report = run_lint(iter(entries), allow_markers,
                      candidate_root=candidate_root, failures=failures)

    # Cross-patch anchor-overlap check (deep-audit #2). conflicts_with pairs
    # are excluded in the detector; a reviewed allowlist tolerates the rest.
    declared = _declared_conflict_pairs()
    cross_all = collect_cross_patch_collisions(entries, declared)
    xp_allow, xp_allow_errors = parse_allowlist(
        Path(DEFAULT_CROSS_PATCH_ALLOWLIST)
    ) if Path(DEFAULT_CROSS_PATCH_ALLOWLIST).exists() else ([], [])
    if xp_allow_errors:
        for e in xp_allow_errors:
            print(f"lint_drift_markers: cross-patch allowlist error: {e}",
                  file=sys.stderr)
        return 2
    xp_allowset = set(xp_allow)
    cross_violations, cross_allowlisted = [], []
    for c in cross_all:
        key = cross_patch_pair_key(c["patch_a"], c["patch_b"])
        (cross_allowlisted if key in xp_allowset
         else cross_violations).append({**c, "pair": key})
    report["cross_patch_collisions"] = cross_violations
    report["cross_patch_allowlisted"] = cross_allowlisted
    report["summary"]["cross_patch_collisions"] = len(cross_violations)
    report["summary"]["cross_patch_allowlisted"] = len(cross_allowlisted)

    payload = json.dumps(report, indent=2, default=str)
    print(payload)
    if args.json_out:
        Path(args.json_out).write_text(payload + "\n")
    _print_human(report)
    return decide_exit(report)


if __name__ == "__main__":
    sys.exit(main())
