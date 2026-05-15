# SPDX-License-Identifier: Apache-2.0
"""Phase 5 community SDK — `sndr community` CLI surface.

Subcommands:

  sndr community list [--json]
      Enumerate every discoverable community patch (filesystem +
      entry-points), sorted by (namespace, id).

  sndr community validate [--root <path>] [--json]
      Walk `plugins/community/` and run release-tier validator rules
      (R-1 anchor md5, R-2 requires_patches, R-3 conflicts_with,
       R-4 apply importable, R-5 tests_required, R-6 id uniqueness,
       R-7 default_on publish_state). Exit 0 on clean, 1 on errors.

  sndr community new-patch --id PN999 --author <handle> --family <name>
                           [--type runtime_hook|text_patch|composite]
                           [--title "..."] [--root <path>]
      Scaffold a working draft plugin tree.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import _io


__all__ = ["add_argparser", "run_list", "run_validate", "run_new_patch"]


def add_argparser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "community",
        help="Community patch SDK (Phase 5) — list/validate/new-patch.",
        description=(
            "Discover, validate, and scaffold community patches under "
            "`plugins/community/`. Release-tier validator catches schema "
            "violations, anchor md5 drift, cross-reference typos, "
            "missing test harnesses, and default_on publish-state mismatches."
        ),
    )
    sub = p.add_subparsers(dest="community_cmd", required=True)

    p_list = sub.add_parser("list",
                            help="Enumerate discoverable community patches.")
    p_list.add_argument("--root", default=None,
                        help="Override plugins/community/ root.")
    p_list.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON.")
    p_list.set_defaults(func=run_list)

    p_val = sub.add_parser("validate",
                           help="Run release-tier validation rules.")
    p_val.add_argument("--root", default=None,
                       help="Override plugins/community/ root.")
    p_val.add_argument("--json", action="store_true",
                       help="Emit machine-readable JSON.")
    p_val.set_defaults(func=run_validate)

    p_new = sub.add_parser("new-patch",
                           help="Scaffold a draft community patch plugin tree.")
    p_new.add_argument("--id", required=True, dest="patch_id",
                       help="Patch id (e.g. PN999, P107_RETRY).")
    p_new.add_argument("--author", required=True,
                       help="Lowercase author/user handle.")
    p_new.add_argument("--family", required=True,
                       help="Patch family (e.g. spec_decode, memory, tool_call).")
    p_new.add_argument("--title", default="Untitled community patch",
                       help="Human-readable patch title.")
    p_new.add_argument("--type", default="runtime_hook",
                       choices=("runtime_hook", "text_patch", "composite"),
                       dest="patch_type",
                       help="Patch type. Default: runtime_hook.")
    p_new.add_argument("--license", default="apache-2.0",
                       dest="license_str",
                       help="SPDX license identifier. Default: apache-2.0.")
    p_new.add_argument("--root", default=None,
                       help="Override plugins/community/ root.")
    p_new.set_defaults(func=run_new_patch)


# ─── Helpers ───────────────────────────────────────────────────────────


def _resolve_root(opts: argparse.Namespace) -> Path:
    """If `--root` is passed, use it. Otherwise the SDK default
    (`plugins/community/` under the repo root)."""
    if opts.root:
        return Path(opts.root).expanduser().resolve()
    from vllm.sndr_core.community.manifest import DEFAULT_PLUGINS_DIR
    return DEFAULT_PLUGINS_DIR


def _manifest_summary(m) -> dict:
    return {
        "namespace": m.namespace,
        "id": m.id,
        "title": m.title,
        "maintainer": m.maintainer,
        "version": m.version,
        "type": m.type,
        "family": m.family,
        "lifecycle": m.lifecycle,
        "implementation_status": m.implementation_status,
        "publish_state": m.publish_state,
        "default_on": m.default_on,
    }


# ─── list ──────────────────────────────────────────────────────────────


def run_list(opts: argparse.Namespace) -> int:
    from vllm.sndr_core.community import discover_all

    root = _resolve_root(opts)
    manifests = discover_all(root)

    if opts.json:
        print(json.dumps(
            {"manifests": [_manifest_summary(m) for m in manifests],
             "count": len(manifests)},
            indent=2, sort_keys=True,
        ))
        return 0

    print("sndr community list — discoverable patches")
    print("─" * 70)
    if not manifests:
        print(f"  (no community patches found under {root})")
        print("  Create one with `sndr community new-patch --id PN999 ...`")
        return 0
    for m in manifests:
        flag = " (default_on)" if m.default_on else ""
        print(f"  {m.namespace}:{m.id}")
        print(f"      {m.title}  [v{m.version}]")
        print(f"      type={m.type}  family={m.family}  "
              f"impl={m.implementation_status}  publish={m.publish_state}"
              f"{flag}")
    print()
    print(f"  Total: {len(manifests)} community patches")
    return 0


# ─── validate ──────────────────────────────────────────────────────────


def run_validate(opts: argparse.Namespace) -> int:
    from vllm.sndr_core.community import validate_directory

    root = _resolve_root(opts)
    result = validate_directory(root)

    if opts.json:
        payload = {
            "root": str(root),
            "manifests": [_manifest_summary(m) for m in result.manifests],
            "issues": [asdict(i) for i in result.issues],
            "errors": len(result.errors),
            "warnings": len(result.warnings),
            "passed": result.passed,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if result.passed else 1

    print(f"sndr community validate — {root}")
    print("─" * 70)
    print(f"  manifests: {len(result.manifests)}")
    print(f"  errors:    {len(result.errors)}")
    print(f"  warnings:  {len(result.warnings)}")
    print()
    if result.issues:
        # Group by severity for readability.
        for sev in ("error", "warning", "info"):
            rows = [i for i in result.issues if i.severity == sev]
            if not rows:
                continue
            sym = {"error": "✗", "warning": "⚠", "info": "ℹ"}[sev]
            print(f"  {sym} {sev.upper()} ({len(rows)}):")
            for i in rows:
                where = f"  [{i.path}]" if i.path else ""
                print(f"    [{i.rule}] {i.message}{where}")
            print()
    if result.passed:
        print("  ✓ release-tier validation passed")
    else:
        print(f"  ✗ release-tier validation FAILED ({len(result.errors)} errors)")
    return 0 if result.passed else 1


# ─── new-patch ─────────────────────────────────────────────────────────


def run_new_patch(opts: argparse.Namespace) -> int:
    from vllm.sndr_core.community.scaffold import (
        ScaffoldError,
        ScaffoldRequest,
        scaffold_patch,
    )

    req = ScaffoldRequest(
        patch_id=opts.patch_id,
        author=opts.author,
        family=opts.family,
        title=opts.title,
        type=opts.patch_type,
        license_str=opts.license_str,
        root=Path(opts.root) if opts.root else None,
    )

    try:
        target = scaffold_patch(req)
    except ScaffoldError as e:
        _io.warn(f"scaffold failed: {e}")
        return 2

    print("sndr community new-patch — scaffold ready at:")
    print(f"  {target}")
    print()
    print("  Files created:")
    print("    manifest.yaml")
    print("    __init__.py")
    print("    patch.py        (apply() stub — replace with real logic)")
    print("    tests/__init__.py")
    print(f"    tests/test_{opts.patch_id.lower()}.py")
    print()
    print("  Next steps:")
    print("    1. Edit manifest.yaml — set compatibility gates + family details")
    print("    2. Implement patch.py:apply() with the real patch logic")
    print(f"    3. Run `sndr community validate --root {target.parents[2]}`")
    print("    4. Flip publish_state to `review` when ready for promotion")
    return 0
