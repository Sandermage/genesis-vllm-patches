#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""audit_shim_window.py — verify historical-path compatibility shims
remain in valid shim state.

A *shim* in Genesis is a directory that exists ONLY to preserve a
historical filesystem path while the real code lives at a new
canonical location. The shim's only legitimate contents are:

  1. A README.md documenting the historical path + retirement contract.
  2. One or more symlinks redirecting the historical path to the
     canonical location.
  3. No other files, subdirectories, or cache entries.

This audit runs over the ``SHIM_MANIFEST`` below; each entry declares
the shim's expected structure (allowed entries, symlink targets,
overlay sentinel files, retirement-wording markers in the README).

Current shim manifest (single entry, 2026-05-26):

  - ``vllm/sndr_core/integrations/gemma4/`` — historical path for the
    PR42637 turboquant overlay. The canonical location is
    ``vllm/sndr_core/integrations/attention/turboquant/overlays/pr42637/``.
    Retirement gated on (a) all ``~/start_g4_*.sh`` launchers
    re-baselined or archived, (b) launcher md5 invariant retired or
    re-baselined, (c) PR42637 / current-pin decision no longer
    requires the historical bind-mount path.

Rules enforced per shim (E.1–E.5):

  - E.1 raw tree = exactly the declared entries (no extras, no
    missing required entries).
  - E.2 each declared symlink resolves to the declared target path.
  - E.3 the resolved symlink target exists and is a directory.
  - E.4 the resolved target contains all declared sentinel files
    (catches accidental overlay-file deletion that would silently
    break the historical path).
  - E.5 README.md contains every declared retirement-wording anchor
    (catches accidental loss of the retirement-contract narrative
    that would let a future operator "just delete the empty folder").

Exit codes:

  0 — all shims valid (CI green).
  1 — at least one rule failed (CI red).
  2 — internal error / manifest unloadable.

Modes:

  python3 scripts/audit_shim_window.py            # human-readable
  python3 scripts/audit_shim_window.py --json     # machine-readable

Adding a new shim:

  1. Append a ``ShimSpec`` to ``SHIM_MANIFEST`` with the historical
     path, allowed entries, symlink targets, sentinel files, and
     retirement-wording anchors.
  2. Add a test case in ``tests/unit/scripts/test_audit_shim_window.py``
     covering valid + each rule's drift case.
  3. The audit auto-picks up the new entry; no other wiring needed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ShimSpec:
    """Declarative shim contract — what the directory must contain."""

    # Path to the shim directory, relative to REPO_ROOT.
    shim_dir: str

    # Path to the README.md (typically `<shim_dir>/README.md`).
    readme_path: str

    # Required files that must appear in the shim dir (besides symlinks).
    required_files: tuple[str, ...]

    # Required symlinks: {entry_name: expected_target_path_relative_to_shim_dir}.
    required_symlinks: dict[str, str]

    # Whether to forbid any extra entries beyond declared files + symlinks.
    forbid_extra_entries: bool = True

    # For each declared symlink, sentinel files that MUST exist in the
    # resolved target directory (catches overlay-file deletion).
    # Format: {symlink_name: (sentinel_file_1, sentinel_file_2, ...)}.
    overlay_sentinels: dict[str, tuple[str, ...]] = field(default_factory=dict)

    # Wording anchors that must appear in the README (substring match).
    # These guard the retirement-contract narrative.
    readme_retirement_anchors: tuple[str, ...] = ()


SHIM_MANIFEST: tuple[ShimSpec, ...] = (
    ShimSpec(
        shim_dir="vllm/sndr_core/integrations/gemma4",
        readme_path="vllm/sndr_core/integrations/gemma4/README.md",
        required_files=("README.md",),
        required_symlinks={
            "upstream_overlay_pr42637": (
                "../attention/turboquant/overlays/pr42637"
            ),
        },
        forbid_extra_entries=True,
        overlay_sentinels={
            "upstream_overlay_pr42637": (
                "turboquant_attn.py",
                "triton_turboquant_store.py",
                "turboquant_config.py",
                "__init__.py",
            ),
        },
        readme_retirement_anchors=(
            "historical",
            "retirement",
            "launcher",
        ),
    ),
)


def _check_shim(
    spec: ShimSpec, root: Optional[Path] = None
) -> list[str]:
    """Validate one shim against its declared contract.

    Returns a list of human-readable issue strings (empty = valid).
    The ``root`` parameter lets tests point at a synthetic tree.
    """
    base = root if root is not None else REPO_ROOT
    shim_path = base / spec.shim_dir
    issues: list[str] = []

    # Existence
    if not shim_path.is_dir():
        return [
            f"shim {spec.shim_dir}: directory does not exist "
            "(the shim itself is missing — retirement happened "
            "without updating SHIM_MANIFEST?)"
        ]

    # E.1 — raw tree integrity
    expected_entries = set(spec.required_files) | set(spec.required_symlinks)
    actual_entries = {p.name for p in shim_path.iterdir()}

    missing_files = set(spec.required_files) - actual_entries
    for entry in sorted(missing_files):
        issues.append(
            f"E.1 shim {spec.shim_dir}: required file {entry!r} missing"
        )

    missing_symlinks = set(spec.required_symlinks) - actual_entries
    for entry in sorted(missing_symlinks):
        issues.append(
            f"E.1 shim {spec.shim_dir}: required symlink {entry!r} missing"
        )

    if spec.forbid_extra_entries:
        extras = actual_entries - expected_entries
        for entry in sorted(extras):
            full = shim_path / entry
            kind = (
                "subdirectory" if full.is_dir() and not full.is_symlink()
                else "file" if full.is_file() and not full.is_symlink()
                else "symlink" if full.is_symlink()
                else "entry"
            )
            issues.append(
                f"E.1 shim {spec.shim_dir}: unexpected {kind} {entry!r} "
                "(shim must contain only declared README + symlinks)"
            )

    # E.2 / E.3 / E.4 — per-symlink checks
    for link_name, expected_target in spec.required_symlinks.items():
        link_path = shim_path / link_name
        if link_name in missing_symlinks:
            # Already reported in E.1; skip further checks for this symlink.
            continue
        if not link_path.is_symlink():
            issues.append(
                f"E.2 shim {spec.shim_dir}: {link_name!r} exists but is "
                "not a symlink (must be a symbolic link to the canonical "
                "overlay path)"
            )
            continue
        actual_target = os.readlink(link_path)
        if actual_target != expected_target:
            issues.append(
                f"E.2 shim {spec.shim_dir}: {link_name!r} points to "
                f"{actual_target!r}, expected {expected_target!r}"
            )
            # Continue to E.3/E.4 anyway — the resolved target may still
            # be valid even if the readlink string differs cosmetically;
            # but the cosmetic divergence itself is a contract violation.

        resolved = link_path.resolve()
        if not resolved.exists():
            issues.append(
                f"E.3 shim {spec.shim_dir}: {link_name!r} target "
                f"{actual_target!r} does not resolve (broken symlink)"
            )
            continue
        if not resolved.is_dir():
            issues.append(
                f"E.3 shim {spec.shim_dir}: {link_name!r} resolves to "
                f"{resolved} which is not a directory"
            )
            continue

        # E.4 — sentinel files
        sentinels = spec.overlay_sentinels.get(link_name, ())
        for sentinel in sentinels:
            sentinel_path = resolved / sentinel
            if not sentinel_path.is_file():
                issues.append(
                    f"E.4 shim {spec.shim_dir}: {link_name!r} target "
                    f"{resolved.name}/ is missing sentinel file "
                    f"{sentinel!r} (overlay may have been gutted; the "
                    "shim now points at an incomplete canonical location)"
                )

    # E.5 — README retirement wording
    if spec.readme_retirement_anchors:
        readme_full = base / spec.readme_path
        if not readme_full.is_file():
            issues.append(
                f"E.5 shim {spec.shim_dir}: README at "
                f"{spec.readme_path} not found (cannot verify retirement "
                "wording)"
            )
        else:
            try:
                text = readme_full.read_text(encoding="utf-8")
            except OSError as e:
                issues.append(
                    f"E.5 shim {spec.shim_dir}: cannot read README: {e}"
                )
            else:
                lowered = text.lower()
                for anchor in spec.readme_retirement_anchors:
                    if anchor.lower() not in lowered:
                        issues.append(
                            f"E.5 shim {spec.shim_dir}: README missing "
                            f"retirement-wording anchor {anchor!r} "
                            "(the historical / retirement / launcher narrative "
                            "must remain so the shim is not removed by "
                            "mistake)"
                        )

    return issues


def run_all(root: Optional[Path] = None) -> dict[str, list[str]]:
    """Validate every shim in the manifest. Returns
    ``{shim_dir: [issues]}`` (empty list = valid).
    """
    return {spec.shim_dir: _check_shim(spec, root=root) for spec in SHIM_MANIFEST}


def _print_human(results: dict[str, list[str]]) -> None:
    total_shims = len(results)
    total_issues = sum(len(v) for v in results.values())
    bar = "─" * 70

    print("╭" + bar + "╮")
    print(f"│ audit_shim_window: {total_shims} shim(s) scanned"
          + " " * (70 - 25 - len(str(total_shims))) + "│")
    print("╰" + bar + "╯")

    for shim_dir, issues in results.items():
        if not issues:
            print(f"  ✓ {shim_dir}")
        else:
            print(f"  ✗ {shim_dir} ({len(issues)} issue(s))")
            for issue in issues:
                print(f"      {issue}")

    print()
    if total_issues == 0:
        print(f"  ✓ all {total_shims} shim(s) valid")
    else:
        print(f"  ✗ {total_issues} issue(s) across {total_shims} shim(s)")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit historical-path compatibility shims."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human output.",
    )
    args = parser.parse_args(argv)

    try:
        results = run_all()
    except Exception as e:  # noqa: BLE001
        print(f"audit_shim_window: internal error: {e}", file=sys.stderr)
        return 2

    total_issues = sum(len(v) for v in results.values())

    if args.json:
        payload = {
            "total_shims": len(results),
            "total_issues": total_issues,
            "results": {k: list(v) for k, v in results.items()},
            "status": "OK" if total_issues == 0 else "DRIFT",
        }
        print(json.dumps(payload, indent=2))
    else:
        _print_human(results)

    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
