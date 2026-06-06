# SPDX-License-Identifier: Apache-2.0
"""Genesis self-test — operator sanity check.

Quick CI-style verification that runs after a git pull or pin bump:

  1. VERSION constant readable + sane
  2. All compat modules import cleanly
  3. All wiring modules import cleanly
  4. PATCH_REGISTRY validates against schema
  5. Lifecycle audit clean (no unknown states)
  6. Categories index builds without errors
  7. Predicates evaluator works on the real registry
  8. JSON schema file present + parseable

Exit code:
  0 = all critical checks passed
  1 = at least one failure (operator action required)

Usage:
  python3 -m vllm.sndr_core.compat.self_test
  python3 -m vllm.sndr_core.compat.self_test --json
  python3 -m vllm.sndr_core.compat.self_test --quiet

This is the "is Genesis itself working?" tool. Different from doctor,
which is "is my SYSTEM healthy?". A doctor failure can be hardware /
config; a self-test failure is a Genesis bug.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("genesis.compat.self_test")


# ─── Check dataclass shape ───────────────────────────────────────────────


def _check(name: str, fn: Callable[[], tuple[str, str]]) -> dict[str, str]:
    """Run a check function. Returns {name, status, message}.

    Check function returns (status, message) where status in
    {pass, fail, warn, skip}.
    """
    try:
        status, message = fn()
    except Exception as e:
        # Self-test must NEVER crash; failed import / unexpected error
        # = "fail" status with traceback summary
        status = "fail"
        message = f"{type(e).__name__}: {e}"
    return {"name": name, "status": status, "message": message}


# ─── Individual checks ───────────────────────────────────────────────────


def _check_version() -> tuple[str, str]:
    """VERSION constant present + readable."""
    from vllm.sndr_core.version import __version__
    if not isinstance(__version__, str):
        return "fail", f"__version__ is {type(__version__).__name__}, want str"
    if not __version__:
        return "fail", "__version__ is empty"
    return "pass", f"version: {__version__}"


def _check_compat_imports() -> tuple[str, str]:
    """Every compat/* module imports without error."""
    modules = [
        "vllm.sndr_core.compat.predicates",
        "vllm.sndr_core.compat.version_check",
        "vllm.sndr_core.compat.lifecycle",
        "vllm.sndr_core.compat.schema_validator",
        "vllm.sndr_core.compat.lifecycle_audit_cli",
        "vllm.sndr_core.compat.categories",
        "vllm.sndr_core.compat.explain",
        "vllm.sndr_core.compat.recipes",
        "vllm.sndr_core.compat.plugins",
        "vllm.sndr_core.compat.telemetry",
        "vllm.sndr_core.compat.update_channel",
        "vllm.sndr_core.compat.cli",
        "vllm.sndr_core.compat.bench",
        "vllm.sndr_core.compat.doctor",
        "vllm.sndr_core.compat.init_wizard",
        "vllm.sndr_core.compat.migrate",
        "vllm.sndr_core.compat.models.registry",
        "vllm.sndr_core.compat.models.list_cli",
        "vllm.sndr_core.compat.models.pull",
    ]
    failed = []
    for m in modules:
        try:
            importlib.import_module(m)
        except Exception as e:
            failed.append(f"{m}: {type(e).__name__}: {e}")
    if failed:
        return "fail", f"{len(failed)} compat module(s) failed to import:\n  " + \
                       "\n  ".join(failed)
    return "pass", f"{len(modules)} compat modules import cleanly"


def _check_wiring_imports() -> tuple[str, str]:
    """Every patch module imports without error.

    v10 (2026-05-07): canonical location is `sndr_core/integrations/`
    with `p<NN>_*.py` / `pn<NN>_*.py` naming. Legacy `wiring/patch_*.py`
    fallback kept for compat. Some patches require resolve_vllm_file
    (i.e. a real vllm install); those are SKIPPED rather than failed
    when vllm isn't importable here.
    """
    from vllm.sndr_core.locations.project_paths import wiring_dir as _wiring_dir

    root = _wiring_dir()
    if root is None or not root.is_dir():
        return "skip", "wiring_dir() returned None or non-existent path"

    # Try canonical naming first (p*.py / pn*.py under integrations/).
    files = sorted(
        list(root.rglob("p[0-9]*.py")) + list(root.rglob("pn[0-9]*.py"))
    )
    # Legacy fallback (wiring/patch_*.py) if no canonical files found.
    if not files:
        files = sorted(root.rglob("patch_*.py"))
    # Filter out tests and __init__.
    files = [f for f in files if not f.name.startswith("test_") and f.name != "__init__.py"]
    failed = []
    skipped_count = 0
    for f in files:
        # Compute dotted module path from the file's location under
        # the vllm/ namespace root — walk upward to find the `vllm` parent.
        parts = f.resolve().parts
        try:
            vllm_idx = len(parts) - 1 - list(reversed(parts)).index("vllm")
        except ValueError:
            failed.append(f"{f}: could not locate 'vllm' in path")
            continue
        rel_parts = parts[vllm_idx:]
        mod_name = ".".join(list(rel_parts[:-1]) + [f.stem])
        try:
            importlib.import_module(mod_name)
        except ModuleNotFoundError as e:
            # vllm not installed → skip rather than fail
            if "vllm" in str(e) and "_genesis" not in str(e) and "sndr_core" not in str(e):
                skipped_count += 1
                continue
            failed.append(f"{mod_name}: {type(e).__name__}: {e}")
        except Exception as e:
            failed.append(f"{mod_name}: {type(e).__name__}: {e}")

    if failed:
        return "fail", f"{len(failed)}/{len(files)} wiring modules broken:\n  " + \
                       "\n  ".join(failed[:5])  # cap at 5 for readability
    if skipped_count > 0:
        return "warn", (
            f"{len(files) - skipped_count}/{len(files)} wiring modules "
            f"imported; {skipped_count} skipped (vllm not installed in this env)"
        )
    return "pass", f"{len(files)} wiring modules import cleanly"


def _check_schema_validator() -> tuple[str, str]:
    """PATCH_REGISTRY schema-validates."""
    from vllm.sndr_core.compat.schema_validator import validate_registry
    from vllm.sndr_core.dispatcher import PATCH_REGISTRY
    issues = validate_registry(PATCH_REGISTRY)
    errors = [i for i in issues if i.severity == "ERROR"]
    if errors:
        return "fail", (
            f"{len(errors)} schema error(s) in PATCH_REGISTRY: "
            + "; ".join(f"{i.patch_id}.{i.field}" for i in errors[:5])
        )
    if issues:
        return "warn", f"{len(issues)} non-error schema issue(s)"
    return "pass", f"all {len(PATCH_REGISTRY)} entries schema-clean"


def _check_lifecycle_audit() -> tuple[str, str]:
    """Lifecycle audit clean (no unknown states)."""
    from vllm.sndr_core.compat.lifecycle import audit_registry
    from vllm.sndr_core.dispatcher import PATCH_REGISTRY
    entries = audit_registry(PATCH_REGISTRY)
    errors = [e for e in entries if e.severity == "error"]
    if errors:
        return "fail", (
            f"{len(errors)} lifecycle error(s): "
            + "; ".join(f"{e.patch_id}: {e.note}" for e in errors[:3])
        )
    return "pass", f"{len(entries)} entries — no unknown lifecycle states"


def _check_categories_build() -> tuple[str, str]:
    """Categories index builds without errors + every patch placed."""
    from vllm.sndr_core.compat.categories import _build_categories
    from vllm.sndr_core.dispatcher import PATCH_REGISTRY

    cats = _build_categories()
    placed = sum(len(v) for v in cats.values())
    if placed != len(PATCH_REGISTRY):
        return "fail", (
            f"category placement mismatch: registry={len(PATCH_REGISTRY)}, "
            f"categorized={placed}"
        )
    return "pass", f"{len(PATCH_REGISTRY)} patches → {len(cats)} categories"


def _check_predicates_evaluate() -> tuple[str, str]:
    """Predicates evaluator works on real entries."""
    from vllm.sndr_core.compat.predicates import evaluate
    from vllm.sndr_core.dispatcher import PATCH_REGISTRY

    failures = []
    for pid, meta in PATCH_REGISTRY.items():
        applies_to = meta.get("applies_to")
        if not applies_to:
            continue
        try:
            ok, _reason = evaluate(applies_to, {})
            assert isinstance(ok, bool)
        except Exception as e:
            failures.append(f"{pid}: {type(e).__name__}: {e}")
    if failures:
        return "fail", (
            f"{len(failures)} predicate eval failures:\n  "
            + "\n  ".join(failures[:5])
        )
    return "pass", "predicates evaluator works on every applies_to in registry"


def _check_schema_file() -> tuple[str, str]:
    """schemas/patch_entry.schema.json file present + parseable.

    P1-2 fix (audit 2026-05-08): the schema is now package data inside
    `vllm/sndr_core/schemas/`, so the canonical resolution uses
    `importlib.resources`. Repo-root and env fallbacks remain for v10.x
    operator workflows where only the source tree is available.
    """
    import os

    candidates: list[Path] = []
    # 1. Canonical: package data via importlib.resources
    try:
        from importlib import resources
        ref = (
            resources.files("vllm.sndr_core.schemas")
            / "patch_entry.schema.json"
        )
        candidates.append(Path(str(ref)))
    except (ModuleNotFoundError, ImportError, FileNotFoundError):
        pass
    # 2. Explicit override via env (lets sysadmins point at any location)
    env_root = os.environ.get("GENESIS_REPO_ROOT")
    if env_root:
        candidates.append(Path(env_root) / "schemas" / "patch_entry.schema.json")
    # 3. Repo-root relative to this file (works in a git checkout)
    candidates.append(
        Path(__file__).resolve().parents[3] / "schemas" / "patch_entry.schema.json"
    )
    # 4. Cwd-relative (works when invoked from the repo root)
    candidates.append(Path.cwd() / "schemas" / "patch_entry.schema.json")

    schema_path = next((p for p in candidates if p.is_file()), None)
    if schema_path is None:
        return "skip", (
            "schema file not findable — package data missing AND no "
            "repo source tree visible. Set GENESIS_REPO_ROOT to override."
        )
    try:
        with open(schema_path) as f:
            data = json.load(f)
    except Exception as e:
        return "fail", f"schema file not valid JSON: {e}"
    required_keys = ("$schema", "title", "type", "properties")
    missing = [k for k in required_keys if k not in data]
    if missing:
        return "warn", f"schema missing keys: {missing}"
    return "pass", "schema file parseable + has required keys"


_CHECKS: list[tuple[str, Callable[[], tuple[str, str]]]] = [
    ("version constant",       _check_version),
    ("compat imports",          _check_compat_imports),
    ("wiring imports",          _check_wiring_imports),
    ("schema validator",        _check_schema_validator),
    ("lifecycle audit",         _check_lifecycle_audit),
    ("categories build",        _check_categories_build),
    ("predicates evaluator",    _check_predicates_evaluate),
    ("schema file",             _check_schema_file),
]


# ─── Driver ──────────────────────────────────────────────────────────────


def run_self_test() -> dict[str, Any]:
    """Run all checks. Returns a dict with `checks` (list) and
    `summary` (counts)."""
    results = []
    for name, fn in _CHECKS:
        results.append(_check(name, fn))

    summary = {"passed": 0, "failed": 0, "warned": 0,
               "skipped": 0, "total": len(results)}
    for r in results:
        if r["status"] == "pass":
            summary["passed"] += 1
        elif r["status"] == "fail":
            summary["failed"] += 1
        elif r["status"] == "warn":
            summary["warned"] += 1
        elif r["status"] == "skip":
            summary["skipped"] += 1

    return {"checks": results, "summary": summary}


def _format_check(c: dict[str, str]) -> str:
    icon = {"pass": "✓", "fail": "✗", "warn": "⚠", "skip": "•"}.get(c["status"], "?")
    return f"  {icon} [{c['status'].upper():<4}] {c['name']:<25} {c['message']}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m vllm.sndr_core.compat.self_test",
        description="Run Genesis structural self-tests.",
    )
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--quiet", action="store_true",
                        help="Show only fail / warn rows")
    args = parser.parse_args(argv)

    result = run_self_test()

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        s = result["summary"]
        if not args.quiet:
            print("=" * 72)
            print("Genesis self-test")
            print("=" * 72)
        for c in result["checks"]:
            if args.quiet and c["status"] == "pass":
                continue
            print(_format_check(c))
        if not args.quiet:
            print("=" * 72)
            print(f"Summary: {s['passed']} pass, {s['failed']} fail, "
                  f"{s['warned']} warn, {s['skipped']} skip")
            print("=" * 72)

    return 1 if result["summary"]["failed"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
