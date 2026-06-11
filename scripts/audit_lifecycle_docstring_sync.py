#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit gate: registry `lifecycle` vs integration-file docstring markers.

Catches drift between a patch's docstring (which authors update first)
and the PATCH_REGISTRY `lifecycle` field (which audits read). Closes
Follow-up #2 from `docs/_internal/CONSOLIDATED_ROADMAP_2026-05-13_RU.md`
documented as "Add docstring lifecycle parser" in plan section 13.

Why this matters — real precedent: PN108 had its docstring marked
TOMBSTONED with an explicit reason ("fla recurrent kernel cannot serve
single-seq prefill") while the registry still said `lifecycle:
experimental`. The mismatch was caught only by accident during the
2026-05-15 session. With this gate, the same shape of drift fails CI.

Drift directions checked:

  - Docstring says RETIRED/TOMBSTONED/DEPRECATED → registry should
    have `lifecycle: retired` (or be missing from registry entirely,
    in which case the file is dead code that should be in _retired/).
  - Registry has `lifecycle: retired` → docstring should mention
    RETIRED/TOMBSTONED at least once, so a future reader of the
    source file sees the lifecycle stamp without needing to grep
    registry.py.

Files in `vllm/sndr_core/integrations/_retired/` are exempt from both
directions (the path alone signals retired).

Coordinator/aggregator wirings (e.g. `_per_patch_dispatch.py`) are
exempt — they wire many patches and shouldn't claim any single
lifecycle.

Exit codes:
  0 — no drift (or all drift is in `IGNORE_PATCH_IDS`)
  1 — at least one un-ignored drift
  2 — internal error
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
# v12.x moved the patch modules to sndr/engines/vllm/patches/; the old
# vllm/sndr_core/integrations tree is now re-export shims whose docstrings
# all read "Backward-compatibility shim" — never the RETIRED/lifecycle
# stamp the real module carries. Audit the canonical source.
INTEGRATIONS_DIR = REPO_ROOT / "sndr" / "engines" / "vllm" / "patches"
if not INTEGRATIONS_DIR.is_dir():
    INTEGRATIONS_DIR = REPO_ROOT / "sndr" / "engines" / "vllm" / "patches"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Tokens we treat as lifecycle markers in module docstrings. Case-sensitive
# to avoid matching normal English prose like "retired" in a sentence
# fragment.
RETIRED_MARKERS: tuple[str, ...] = (
    "TOMBSTONED",
    "RETIRED",
    "DEPRECATED",
    "DEAD-CODE",
)


# Patch IDs we intentionally allow to drift (operator-documented). Keep
# this list short and add a comment for each entry.
IGNORE_PATCH_IDS: dict[str, str] = {
    # Example shape — add real entries here if operator decides to
    # waive a specific drift:
    # "PN999": "operator-allowlisted: drift acknowledged in YAML",
    #
    # 2026-06-08 archive-drift forensics retired individual SUB-patches
    # inside these still-active parents; the inline "sub-patch RETIRED"
    # annotations are deliberate and do NOT mean the module is retired.
    "P98": "sub-patch p98_decode_workspace_revert retired 2026-06-08; parent active",
    "P5": "sub-patch p5_import_math retired 2026-06-08; parent active",
    "PN95": "anchor #13 worker_proactive_demote retired on dev259+; parent active",
    "P12": "sub-patches retired 2026-06-08 (archive-drift forensics); parent active",
    "P27": "sub-patch p27_nonstream_return_baseline retired 2026-06-08; parent active",
    "P64": "sub-patches p64_safety_net_widen / p64_callsite_guard retired; parent active",
    # 2026-06-11 preflight-residual-triage fix-drifts batch: sub-variant
    # retirements inside still-active parents (plan section 1b).
    "P59": "wrap variants A/B retired 2026-06-11 (dead residue anchors); parent active",
    "PN38": "Site B retired 2026-06-11 (upstream-native since dev259); parent active",
}


# Module filenames that are dispatchers / coordinators / aggregators,
# not single patches. Exempt from both drift directions.
COORDINATOR_NAMES: frozenset[str] = frozenset({
    "__init__.py",
    "_per_patch_dispatch.py",
    "_family_init.py",
    "_dedupe_helpers.py",
})


PATCH_ID_FROM_FILENAME = re.compile(
    r"^(?P<pid>P[Nn]?\d+[A-Za-z]?|sndr_workspace_\d+|sprint\d+_[A-Za-z0-9_]+|g4_\d+[A-Za-z]?|G\d+_T\d+|G\d+_\d+[A-Za-z]?)_",
    re.IGNORECASE,
)


@dataclass
class FileInspection:
    path: Path
    patch_id: str = ""
    has_retire_marker: bool = False
    registry_lifecycle: str = ""
    error: str = ""
    drift: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.error and not self.drift


def _extract_source(path: Path) -> str:
    """Return the raw source text of `path`, or empty string on error."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _extract_module_docstring(source: str) -> str:
    """Return the first module-level docstring from `source`."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    return ast.get_docstring(tree) or ""


def _patch_id_from_filename(path: Path) -> str:
    """Derive a likely registry patch_id from a file name."""
    m = PATCH_ID_FROM_FILENAME.match(path.stem)
    if not m:
        return ""
    pid = m.group("pid")
    # Normalize: PN132 stays PN132; pn132 → PN132.
    if pid.lower().startswith("p"):
        # Keep first letter as-is then uppercase the rest of the
        # leading 1-2 letters.
        if pid[:2].lower() == "pn":
            return "PN" + pid[2:]
        if pid[0].lower() == "p":
            return "P" + pid[1:]
    if pid.lower().startswith("g") and "_" in pid:
        return pid.upper().replace("_", "_")
    return pid.upper()


def _has_retire_marker(text: str) -> bool:
    """Whether `text` (raw source) carries a lifecycle retirement marker.

    Checked against the raw source — covers both module-level docstrings
    and per-function docstrings (e.g. `apply()` carrying a RETIRED note).
    Case-sensitive to avoid matching English prose like "retired".
    """
    return any(marker in text for marker in RETIRED_MARKERS)


def _registry_lifecycle(patch_id: str) -> str:
    try:
        from sndr.dispatcher.registry import PATCH_REGISTRY
    except ImportError:
        return "?import-error"
    meta = PATCH_REGISTRY.get(patch_id)
    if not meta:
        return "?missing"
    return str(meta.get("lifecycle", "?unset"))


def inspect_file(path: Path) -> FileInspection:
    if path.name in COORDINATOR_NAMES:
        return FileInspection(path=path, patch_id="<coordinator>")
    source = _extract_source(path)
    patch_id = _patch_id_from_filename(path)
    if not patch_id:
        return FileInspection(path=path, patch_id="<unmatched>")
    has_marker = _has_retire_marker(source)
    lifecycle = _registry_lifecycle(patch_id)
    inspection = FileInspection(
        path=path,
        patch_id=patch_id,
        has_retire_marker=has_marker,
        registry_lifecycle=lifecycle,
    )
    if patch_id in IGNORE_PATCH_IDS:
        return inspection
    if lifecycle == "?import-error":
        inspection.error = "registry import failed"
        return inspection
    # Drift direction 1: docstring marker, registry not retired.
    if has_marker and lifecycle not in ("retired", "?missing"):
        inspection.drift.append(
            f"docstring marks RETIRED/TOMBSTONED but registry "
            f"`lifecycle: {lifecycle}` — update registry to `retired` "
            f"or remove the marker from the docstring"
        )
    # Drift direction 2: registry retired, docstring silent.
    if lifecycle == "retired" and not has_marker:
        inspection.drift.append(
            "registry `lifecycle: retired` but docstring does not "
            "mention RETIRED/TOMBSTONED — add a line near the top of "
            "the module docstring so future readers see the lifecycle "
            "stamp without grepping registry.py"
        )
    return inspection


def iter_integration_files() -> list[Path]:
    out: list[Path] = []
    for path in sorted(INTEGRATIONS_DIR.rglob("*.py")):
        # Skip everything under _retired/ — path alone signals retired,
        # docstring and registry are allowed to be silent.
        if any(part == "_retired" for part in path.parts):
            continue
        # Skip dead-patches subtree (historical archive).
        if any(part.startswith("dead_") for part in path.parts):
            continue
        # Skip __pycache__.
        if "__pycache__" in path.parts:
            continue
        out.append(path)
    return out


def audit() -> list[FileInspection]:
    files = iter_integration_files()
    return [inspect_file(p) for p in files]


def render_text(results: list[FileInspection]) -> str:
    drift_count = sum(1 for r in results if r.drift)
    err_count = sum(1 for r in results if r.error)
    relevant = [r for r in results if r.drift or r.error]
    lines = [
        f"audit-lifecycle-docstring-sync: {len(results)} integration files scanned",
        f"  drift entries: {drift_count}",
        f"  errors:        {err_count}",
        f"  ignored:       {len(IGNORE_PATCH_IDS)}",
        "─" * 70,
    ]
    if not relevant:
        lines.append("  ✓ all files in sync with registry lifecycle")
        return "\n".join(lines)
    for r in relevant:
        rel = r.path.relative_to(REPO_ROOT)
        if r.error:
            lines.append(f"  ✗ {rel}: {r.error}")
            continue
        lines.append(f"  ✗ {rel} (id={r.patch_id}, registry={r.registry_lifecycle})")
        for d in r.drift:
            lines.append(f"      → {d}")
    lines.append("─" * 70)
    lines.append(
        f"  ✗ {drift_count + err_count} files need attention"
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true",
                    help="Exit 1 on any drift (default: report only)")
    args = ap.parse_args()
    try:
        results = audit()
    except Exception as e:
        print(f"audit-lifecycle-docstring-sync: internal error — {e}",
              file=sys.stderr)
        return 2
    print(render_text(results))
    has_drift = any(r.drift or r.error for r in results)
    return 1 if (has_drift and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
