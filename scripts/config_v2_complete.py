#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§4.2 V2 hardware-schema auto-completer (Entry 23 closeout).

Reads what `audit-launch-coverage` reports as missing (per-file
`missing_mounts` + `missing_envs`) and writes the canonical entries
back into the YAML in-place, preserving existing comments + ordering
of unrelated fields.

The schema is single-source-of-truth from `audit_launch_coverage.py`:

    • `REQUIRED_MOUNTS`   — 5 canonical slots
                            (container_path, mode, description, host_var)
    • `REQUIRED_ENV_KEYS` — 7 keys
    • `ENV_DEFAULTS`      — canonical YAML-literal values per key

Why not rewrite the whole YAML with PyYAML?
  → PyYAML's default dumper destroys comments + reflows whitespace. The
    V2 hardware YAMLs are operator-curated documentation as much as
    config; reflowing them = losing operator intent. Line-injection
    preserves the source exactly, modifying only the two anchor blocks
    (`runtime.docker.mounts` + `system_env`).

Modes:

    python3 scripts/config_v2_complete.py                  # check every V2 hardware YAML, print diff, exit 1 if drift
    python3 scripts/config_v2_complete.py --write          # rewrite drifted files
    python3 scripts/config_v2_complete.py --file PATH      # restrict to one file
    python3 scripts/config_v2_complete.py --json           # machine-readable summary

Exit codes:
    0  every YAML already canonical (or --write succeeded everywhere)
    1  drift detected in at least one YAML (in --check / default mode)
    2  internal error (parse failure, missing anchor blocks)
"""
from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
HW_DIR = REPO_ROOT / "sndr" / "model_configs" / "builtin" / "hardware"


def _import_audit():
    """Load `scripts/audit_launch_coverage.py` as a module so we share
    its REQUIRED_MOUNTS / REQUIRED_ENV_KEYS / ENV_DEFAULTS schema."""
    name = "_audit_launch_coverage_for_completer"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / "audit_launch_coverage.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── Result types ─────────────────────────────────────────────────────


class CompletionStatus(str, Enum):
    CLEAN = "clean"                 # already canonical
    WOULD_WRITE = "would_write"     # drift detected; not written (check mode)
    WRITTEN = "written"             # drift detected; written
    ERROR = "error"                 # internal error (no write performed)


@dataclass
class CompletionResult:
    yaml_path: Path
    hardware_id: str
    status: CompletionStatus
    missing_mounts: list[str] = field(default_factory=list)
    missing_envs: list[str] = field(default_factory=list)
    diff: str = ""
    error: str = ""


# ─── Line-injection helpers ───────────────────────────────────────────


def _find_anchor_block(
    lines: list[str], anchor: str,
) -> Optional[tuple[int, int, int]]:
    """Find the line index of `anchor:`, the index of the last
    list/map item under it, and the inferred indent prefix length for
    new items.

    Returns `(anchor_line, last_item_line, item_indent)` or None.

    Detection logic:
      • anchor_line: the line ending with `anchor:` (possibly with comments)
      • last_item_line: the last line below anchor_line whose indent is
        strictly deeper than anchor_line's indent (until a line with
        ≤ anchor_line indent appears)
      • item_indent: number of leading spaces on the items
    """
    anchor_token = anchor.strip() + ":"
    anchor_line = -1
    anchor_indent = -1
    for i, line in enumerate(lines):
        stripped = line.lstrip(" ")
        # Skip empty + comment-only lines.
        if not stripped or stripped.startswith("#"):
            continue
        # Match `anchor:` (with or without trailing inline comment).
        # The anchor itself is at any indent — we want the FIRST one.
        if stripped.startswith(anchor_token):
            anchor_line = i
            anchor_indent = len(line) - len(stripped)
            break
    if anchor_line < 0:
        return None

    last_item = anchor_line
    item_indent = anchor_indent + 2   # fallback default

    for j in range(anchor_line + 1, len(lines)):
        line = lines[j]
        stripped = line.lstrip(" ")
        # Empty + comment-only lines don't terminate the block (operator
        # may have left blank lines between entries).
        if not stripped or stripped.startswith("#"):
            continue
        cur_indent = len(line) - len(stripped)
        if cur_indent <= anchor_indent:
            break
        # An item belonging to the block.
        last_item = j
        if item_indent == anchor_indent + 2:
            # First real item seen — adopt its indent.
            item_indent = cur_indent
    return (anchor_line, last_item, item_indent)


def _render_mount_entry(slot, indent: int) -> str:
    """Produce a single canonical mount YAML entry line, matching the
    style of existing entries: `      - "${host}:/container[:mode]"`."""
    mode_suffix = f":{slot.mode}" if slot.mode in ("ro", "rw") else ""
    # `rw` mode is the YAML default; V1 omits it. Keep mode suffix
    # explicit only for `ro` to match the V1 convention.
    if slot.mode == "rw":
        mode_suffix = ""
    pad = " " * indent
    return (
        f'{pad}- "{slot.host_var}:{slot.container_path}{mode_suffix}"'
        f'  # E23 auto-added: {slot.description}'
    )


def _render_env_entry(key: str, value: str, indent: int) -> str:
    pad = " " * indent
    return f"{pad}{key}: {value}   # E23 auto-added"


def _inject_lines(
    src_lines: list[str], anchor: str, new_lines: list[str],
) -> Optional[list[str]]:
    """Insert `new_lines` immediately after the last item under
    `anchor`. Returns the new line list, or None if the anchor isn't
    found (caller decides whether that's an error)."""
    found = _find_anchor_block(src_lines, anchor)
    if found is None:
        return None
    _, last_item, _ = found
    return (
        src_lines[: last_item + 1]
        + new_lines
        + src_lines[last_item + 1:]
    )


# ─── Per-YAML completion ──────────────────────────────────────────────


def complete_one_yaml(
    path: Path, *, write: bool = False,
) -> CompletionResult:
    audit_mod = _import_audit()
    audit = audit_mod.audit_one_hardware_yaml(path)

    if audit.parse_error:
        return CompletionResult(
            yaml_path=path,
            hardware_id=audit.hardware_id,
            status=CompletionStatus.ERROR,
            error=audit.parse_error,
        )

    if audit.passed:
        return CompletionResult(
            yaml_path=path,
            hardware_id=audit.hardware_id,
            status=CompletionStatus.CLEAN,
        )

    src_text = path.read_text(encoding="utf-8")
    src_lines = src_text.splitlines()

    # Inject missing mounts.
    new_lines = src_lines
    if audit.missing_mounts:
        cp_to_slot = {s.container_path: s for s in audit_mod.REQUIRED_MOUNTS}
        # Use the indent inferred from the existing mounts block.
        mount_block = _find_anchor_block(new_lines, "mounts")
        if mount_block is None:
            return CompletionResult(
                yaml_path=path,
                hardware_id=audit.hardware_id,
                status=CompletionStatus.ERROR,
                error="`mounts:` anchor not found — V2 schema variant?",
            )
        _, _, mount_indent = mount_block
        rendered = [
            _render_mount_entry(cp_to_slot[cp], mount_indent)
            for cp in audit.missing_mounts
        ]
        new_lines = _inject_lines(new_lines, "mounts", rendered)
        if new_lines is None:
            return CompletionResult(
                yaml_path=path,
                hardware_id=audit.hardware_id,
                status=CompletionStatus.ERROR,
                error="failed to inject mounts",
            )

    # Inject missing envs.
    if audit.missing_envs:
        env_block = _find_anchor_block(new_lines, "system_env")
        if env_block is None:
            return CompletionResult(
                yaml_path=path,
                hardware_id=audit.hardware_id,
                status=CompletionStatus.ERROR,
                error="`system_env:` anchor not found — V2 schema variant?",
            )
        _, _, env_indent = env_block
        rendered = [
            _render_env_entry(k, audit_mod.ENV_DEFAULTS[k], env_indent)
            for k in audit.missing_envs
            if k in audit_mod.ENV_DEFAULTS
        ]
        # Skip any keys for which we don't have canonical defaults (shouldn't
        # happen with current schema, but defensive).
        if rendered:
            new_lines = _inject_lines(new_lines, "system_env", rendered)
            if new_lines is None:
                return CompletionResult(
                    yaml_path=path,
                    hardware_id=audit.hardware_id,
                    status=CompletionStatus.ERROR,
                    error="failed to inject envs",
                )

    new_text = "\n".join(new_lines)
    if src_text.endswith("\n") and not new_text.endswith("\n"):
        new_text += "\n"

    diff = "".join(difflib.unified_diff(
        src_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile=str(path) + " (before)",
        tofile=str(path) + " (after E23)",
        n=2,
    ))

    if write:
        path.write_text(new_text, encoding="utf-8")
        status = CompletionStatus.WRITTEN
    else:
        status = CompletionStatus.WOULD_WRITE

    return CompletionResult(
        yaml_path=path,
        hardware_id=audit.hardware_id,
        status=status,
        missing_mounts=audit.missing_mounts,
        missing_envs=audit.missing_envs,
        diff=diff,
    )


# ─── Whole-directory sweep ────────────────────────────────────────────


def complete_directory(
    hw_dir: Path = HW_DIR, *, write: bool = False,
) -> list[CompletionResult]:
    if not hw_dir.is_dir():
        return []
    return [
        complete_one_yaml(p, write=write)
        for p in sorted(hw_dir.glob("*.yaml"))
    ]


# ─── Renderers ────────────────────────────────────────────────────────


def _render_text(results: list[CompletionResult]) -> str:
    lines = []
    lines.append(f"config-v2-complete: {len(results)} V2 hardware YAML(s)")
    lines.append("─" * 70)
    drift = [r for r in results if r.status != CompletionStatus.CLEAN]
    clean = [r for r in results if r.status == CompletionStatus.CLEAN]
    errors = [r for r in results if r.status == CompletionStatus.ERROR]

    for r in results:
        if r.status == CompletionStatus.CLEAN:
            lines.append(f"  ✓ {r.hardware_id} (canonical, no changes)")
            continue
        if r.status == CompletionStatus.ERROR:
            lines.append(f"  ! {r.hardware_id} ({r.error})")
            continue
        verb = "would add" if r.status == CompletionStatus.WOULD_WRITE else "added"
        lines.append(
            f"  ✎ {r.hardware_id}: {verb} "
            f"{len(r.missing_mounts)} mount(s) + {len(r.missing_envs)} env(s)"
        )
        for cp in r.missing_mounts:
            lines.append(f"      + mount  {cp}")
        for k in r.missing_envs:
            lines.append(f"      + env    {k}")

    lines.append("─" * 70)
    if errors:
        lines.append(f"  ! {len(errors)} error(s); fix the YAML manually")
    if drift and clean is not None:
        verb = "needed" if any(
            r.status == CompletionStatus.WOULD_WRITE for r in results
        ) else "applied"
        lines.append(f"  {len(drift) - len(errors)} file(s): completion {verb}")
        lines.append(f"  {len(clean)} file(s): already canonical")
    else:
        lines.append(f"  {len(results)} file(s): all canonical")
    return "\n".join(lines)


def _render_json(results: list[CompletionResult]) -> str:
    payload = {
        "total": len(results),
        "clean": sum(1 for r in results if r.status == CompletionStatus.CLEAN),
        "would_write": sum(
            1 for r in results if r.status == CompletionStatus.WOULD_WRITE
        ),
        "written": sum(
            1 for r in results if r.status == CompletionStatus.WRITTEN
        ),
        "errors": sum(1 for r in results if r.status == CompletionStatus.ERROR),
        "results": [
            {
                "yaml": _yaml_label(r.yaml_path),
                "hardware_id": r.hardware_id,
                "status": r.status.value,
                "missing_mounts": r.missing_mounts,
                "missing_envs": r.missing_envs,
                "error": r.error or None,
            }
            for r in results
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _yaml_label(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="Actually rewrite drifted YAMLs (default: check-only).")
    ap.add_argument("--file", default=None,
                    help="Process only this YAML (default: every file in hardware/).")
    ap.add_argument("--hw-dir", default=None,
                    help="Override V2 hardware YAML directory.")
    ap.add_argument("--json", action="store_true",
                    help="Emit JSON summary.")
    ap.add_argument("--show-diff", action="store_true",
                    help="In text mode, also print unified diff per drifted file.")
    args = ap.parse_args()

    if args.file:
        results = [complete_one_yaml(Path(args.file), write=args.write)]
    else:
        hw_dir = Path(args.hw_dir) if args.hw_dir else HW_DIR
        results = complete_directory(hw_dir, write=args.write)

    if args.json:
        print(_render_json(results))
    else:
        print(_render_text(results))
        if args.show_diff:
            for r in results:
                if r.diff:
                    print()
                    print(r.diff)

    drift_count = sum(
        1 for r in results
        if r.status in (CompletionStatus.WOULD_WRITE, CompletionStatus.ERROR)
    )
    return 0 if drift_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
