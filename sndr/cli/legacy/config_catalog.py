# SPDX-License-Identifier: Apache-2.0
"""CONFIG-UX.5.2 — `sndr config-catalog` low-level derived-catalog CLI.

Operator-product surface for the derived catalog: inventory, debug,
and export. **Derived catalog**, NOT a source of truth. Source of
truth remains the V2 YAML tree plus public baseline JSONs; the
catalog is regenerable from those at any time.

Four leaves (operator-locked at CONFIG-UX.5.2.R §3):

  build   — thin wrapper over scripts/generate_config_catalog.py
  verify  — thin wrapper over scripts/audit_generated_config_catalog.py
  show    — one-row inspection (preset / profile / model / hardware / baseline)
  query   — narrow filter DSL (5 fixed flags, AND-only; never mini-SQL)

Hard rules (locked):
  - No torch import in this path (CLI runs without a GPU)
  - Output redaction: terminal -> "[REDACTED private evidence]";
    JSON output -> `{"redacted": true, ...}` marker per the
    generator's RedactedEvidenceRef shape
  - Every --help string includes the phrase "derived catalog"
  - --from <path> honors the operator's file choice; if stale and
    --strict-fresh is set, exit nonzero (never silently regenerate)
  - Bare row IDs accepted only when unambiguous across row_types;
    collisions error with candidate list
  - No SQLite anywhere
  - No CLI auto-regeneration hooks
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from . import _io


__all__ = [
    "add_argparser",
    "run_build",
    "run_verify",
    "run_show",
    "run_query",
]


_REPO_ROOT = Path(__file__).resolve().parents[3]


# ─── Generator + audit module loading (lazy, no torch) ──────────────────────


def _load_generator():
    """Load generate_config_catalog.py as a module. Lazy to keep CLI
    import fast and to keep the torch-free guarantee honest."""
    script = _REPO_ROOT / "scripts" / "generate_config_catalog.py"
    spec = importlib.util.spec_from_file_location(
        "_config_catalog_cli_generator", script,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_config_catalog_cli_generator"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_audit():
    """Load audit_generated_config_catalog.py as a module."""
    script = _REPO_ROOT / "scripts" / "audit_generated_config_catalog.py"
    spec = importlib.util.spec_from_file_location(
        "_config_catalog_cli_audit", script,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_config_catalog_cli_audit"] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── Argparse registration ──────────────────────────────────────────────────


def add_argparser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "config-catalog",
        help=(
            "Low-level derived catalog inventory, debug, and export surface."
        ),
        description=(
            "Inspect and export the derived catalog generated from the V2 "
            "YAML tree + public baseline JSONs. The catalog is a derived "
            "API, NOT a source of truth — every leaf can regenerate the "
            "catalog in-memory by default. See `sndr preset` for the "
            "operator-product surface that picks presets for workloads; "
            "this command is for inventory / debug / scripting."
        ),
    )
    sub = p.add_subparsers(dest="config_catalog_cmd", required=True)

    # ── build ───────────────────────────────────────────────────────────
    p_build = sub.add_parser(
        "build",
        help="Build the derived catalog JSON from the V2 YAML tree + baselines.",
        description=(
            "Build the derived catalog JSON. Default writes to "
            "build/config_catalog/config_catalog.json. The committed-JSON "
            "artifact is not required by any leaf — `show`, `query`, and "
            "`verify` regenerate in-memory by default."
        ),
    )
    p_build.add_argument(
        "--stdout", action="store_true",
        help="Emit the derived catalog JSON to stdout instead of writing.",
    )
    p_build.add_argument(
        "--check", action="store_true",
        help="Verify deterministic regeneration (two consecutive runs match)."
             " Exit 1 on drift.",
    )
    p_build.set_defaults(func=run_build)

    # ── verify ──────────────────────────────────────────────────────────
    p_verify = sub.add_parser(
        "verify",
        help="Verify derived catalog determinism + redaction discipline.",
        description=(
            "Thin wrapper over audit_generated_config_catalog.py. Verifies "
            "the derived catalog regenerates deterministically and that no "
            "private paths leak into the output. Informational by default."
        ),
    )
    p_verify.add_argument(
        "--strict", action="store_true",
        help="Exit 1 on any finding (warning OR error). Default mode is "
             "informational — only errors fail.",
    )
    p_verify.add_argument(
        "--json", action="store_true",
        help="Machine-readable JSON output.",
    )
    p_verify.set_defaults(func=run_verify)

    # ── show ────────────────────────────────────────────────────────────
    p_show = sub.add_parser(
        "show",
        help="Display one row from the derived catalog (by preset/profile/model/hardware/baseline id).",
        description=(
            "Display one derived catalog row. Row id forms: prefixed "
            "(e.g. `preset/prod-qwen3.6-35b-balanced`, `profile/qwen3.6-27b-tq-k8v4`) or bare "
            "(e.g. `prod-qwen3.6-35b-balanced`) when unambiguous. Bare-id collisions "
            "across row types error with the candidate list."
        ),
    )
    p_show.add_argument("row_id", help="Row id (prefixed or bare).")
    p_show.add_argument(
        "--json", action="store_true",
        help="Machine-readable JSON output (full row dict).",
    )
    p_show.add_argument(
        "--from", dest="from_path", default=None,
        help="Read derived catalog from <path> instead of regenerating "
             "in-memory. Stale file (older than youngest source YAML) "
             "warns by default; use --strict-fresh to exit nonzero.",
    )
    p_show.add_argument(
        "--strict-fresh", action="store_true",
        help="With --from, exit nonzero if the file is stale.",
    )
    p_show.set_defaults(func=run_show)

    # ── query ───────────────────────────────────────────────────────────
    p_query = sub.add_parser(
        "query",
        help="Query the derived catalog with a narrow fixed filter DSL (5 flags).",
        description=(
            "Query the derived catalog. Filter DSL is intentionally narrow: "
            "5 fixed flags, AND-only, no JSONPath, no joins, no sort. "
            "Operator-locked: extensions tempt toward mini-SQL and are "
            "deferred to a separate phase. Pipe `--json` output through "
            "`jq` for richer transforms."
        ),
    )
    p_query.add_argument(
        "--row-type", required=True,
        choices=["preset", "profile", "model", "hardware", "baseline", "any"],
        help="Restrict query to rows of this type (or `any` for all).",
    )
    p_query.add_argument(
        "--field", default=None,
        help="Catalog row field to filter on (e.g. override_class, card_status, match_quality).",
    )
    p_query.add_argument(
        "--equals", default=None,
        help="Field value must equal this (string comparison after str()).",
    )
    p_query.add_argument(
        "--contains", default=None,
        help="Field value must contain this substring (for str fields) "
             "or include this element (for list fields).",
    )
    p_query.add_argument(
        "--expires-before", default=None,
        help="ISO date (YYYY-MM-DD); field value (an ISO date) must be earlier.",
    )
    p_query.add_argument(
        "--json", action="store_true",
        help="Machine-readable JSON output (list of full rows).",
    )
    p_query.add_argument(
        "--from", dest="from_path", default=None,
        help="Read derived catalog from <path> instead of regenerating "
             "in-memory.",
    )
    p_query.add_argument(
        "--strict-fresh", action="store_true",
        help="With --from, exit nonzero if the file is stale.",
    )
    p_query.set_defaults(func=run_query)


# ─── Catalog loading (in-memory regen by default, --from honors operator) ──


def _newest_source_mtime() -> float:
    """Return the youngest mtime across all source YAMLs + baseline JSONs.

    Used to detect a stale --from file: if file mtime < youngest source
    mtime, the file is stale relative to the actual corpus state.
    """
    builtin = _REPO_ROOT / "sndr" / "model_configs" / "builtin"
    baselines = _REPO_ROOT / "tests" / "integration" / "baselines"
    mtimes: list[float] = [0.0]
    if builtin.is_dir():
        for path in builtin.rglob("*.yaml"):
            try:
                mtimes.append(path.stat().st_mtime)
            except OSError:
                continue
    if baselines.is_dir():
        for path in baselines.glob("*.json"):
            try:
                mtimes.append(path.stat().st_mtime)
            except OSError:
                continue
    return max(mtimes)


def _load_catalog(
    *,
    from_path: Optional[str],
    strict_fresh: bool,
) -> list[dict]:
    """Return catalog rows.

    Default (`from_path=None`): regenerate in-memory via the generator's
    `build_catalog()`. No file dependency.

    With `--from <path>`: read the pre-built JSON. If the file is older
    than the youngest source YAML/baseline, emit a stale-warning. If
    `strict_fresh=True`, exit nonzero rather than continuing.

    Raises:
      FileNotFoundError if `from_path` doesn't exist.
      SystemExit(1) with --strict-fresh and stale file.
    """
    if from_path is None:
        gen = _load_generator()
        return gen.build_catalog()

    path = Path(from_path)
    if not path.is_file():
        _io.error(f"--from {from_path!r}: file not found")
        sys.exit(2)

    # Staleness check before loading
    file_mtime = path.stat().st_mtime
    source_mtime = _newest_source_mtime()
    if file_mtime < source_mtime:
        msg = (
            f"--from {from_path!r}: file mtime ({datetime.fromtimestamp(file_mtime).isoformat()}) "
            f"is older than youngest source YAML/baseline "
            f"({datetime.fromtimestamp(source_mtime).isoformat()}) — "
            f"catalog may be stale relative to actual corpus state"
        )
        if strict_fresh:
            _io.error(msg + " (--strict-fresh)")
            sys.exit(1)
        _io.warn(msg)

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        _io.error(f"--from {from_path!r}: invalid JSON: {e}")
        sys.exit(2)

    if not isinstance(data, dict) or "rows" not in data:
        _io.error(f"--from {from_path!r}: expected top-level dict with `rows` key")
        sys.exit(2)
    return list(data["rows"])


# ─── Row-id resolution (prefixed or bare, collision-aware) ──────────────────


def _resolve_row_id(rows: list[dict], row_id: str) -> dict:
    """Find one row by id. Accepts prefixed (`preset/prod-qwen3.6-35b-balanced`) or bare
    (`prod-qwen3.6-35b-balanced`) forms; bare-id collisions across row types raise with
    the candidate list.

    Returns the matching row dict. Exits nonzero on not-found OR ambiguity.
    """
    if "/" in row_id:
        prefix, bare = row_id.split("/", 1)
        matches = [
            r for r in rows
            if r.get("row_type") == prefix and r.get("id") == bare
        ]
        if not matches:
            _io.error(f"row not found: {row_id!r}")
            sys.exit(1)
        return matches[0]

    # Bare form — search across row types
    matches = [r for r in rows if r.get("id") == row_id]
    if not matches:
        _io.error(f"row not found: {row_id!r} (tried all row types)")
        sys.exit(1)
    if len(matches) > 1:
        candidates = [
            f"{r.get('row_type')}/{r.get('id')}" for r in matches
        ]
        _io.error(
            f"row id {row_id!r} ambiguous; candidates: "
            + ", ".join(candidates)
            + "\n  use the prefixed form (e.g. `preset/{row_id}`) to disambiguate."
        )
        sys.exit(1)
    return matches[0]


# ─── Redaction at output time (terminal vs JSON) ────────────────────────────


def _redact_for_terminal(row: dict) -> dict:
    """Replace `{redacted: true, ...}` markers in a row with human
    `[REDACTED private evidence]` strings for terminal display.

    JSON output keeps the generator's marker shape verbatim (consumers
    of `--json` need the structured form).
    """
    out = dict(row)
    refs = out.get("card_evidence_refs")
    if isinstance(refs, list):
        new_refs = []
        for ref in refs:
            if isinstance(ref, dict) and ref.get("redacted") is True:
                new_refs.append("[REDACTED private evidence]")
            else:
                new_refs.append(ref)
        out["card_evidence_refs"] = new_refs
    return out


# ─── Leaf: build ────────────────────────────────────────────────────────────


def run_build(args) -> int:
    """Thin wrapper over scripts/generate_config_catalog.py main()."""
    gen = _load_generator()
    # Translate argparse namespace into the generator's argv
    argv = []
    if getattr(args, "stdout", False):
        argv.append("--stdout")
    if getattr(args, "check", False):
        argv.append("--check")
    # Reuse the generator's main() to keep behavior identical
    return _run_module_main(gen, argv)


# ─── Leaf: verify ───────────────────────────────────────────────────────────


def run_verify(args) -> int:
    """Thin wrapper over scripts/audit_generated_config_catalog.py main()."""
    audit = _load_audit()
    argv = []
    if getattr(args, "strict", False):
        argv.append("--strict")
    if getattr(args, "json", False):
        argv.append("--json")
    return _run_module_main(audit, argv)


def _run_module_main(mod, argv: list[str]) -> int:
    """Invoke `mod.main()` with a synthesised sys.argv. Restores the
    real sys.argv after the call."""
    saved = sys.argv
    try:
        sys.argv = [getattr(mod, "__file__", "module"), *argv]
        return mod.main()
    finally:
        sys.argv = saved


# ─── Leaf: show ─────────────────────────────────────────────────────────────


def run_show(args) -> int:
    rows = _load_catalog(
        from_path=getattr(args, "from_path", None),
        strict_fresh=getattr(args, "strict_fresh", False),
    )
    row = _resolve_row_id(rows, args.row_id)

    if getattr(args, "json", False):
        # JSON keeps generator marker shape verbatim
        print(json.dumps(row, indent=2, sort_keys=True, default=str))
        return 0

    # Terminal: human-readable, redacted private evidence
    row_display = _redact_for_terminal(row)
    _render_row_human(row_display)
    return 0


def _render_row_human(row: dict) -> None:
    """Render a row in human-readable sectioned form. Generic across
    row types; section choice is driven by the row's `row_type`."""
    row_type = row.get("row_type", "unknown")
    rid = row.get("id", "?")
    print(f"\n  derived catalog row: {row_type}/{rid}")
    print(f"    source: {row.get('source_path')}")
    print(f"    source sha256: {row.get('source_sha256')}")
    if row.get("updated_from_git_commit"):
        print(f"    last commit: {row['updated_from_git_commit']}")
    print(f"    generated at: {row.get('generated_at')}")

    skip = {
        "schema_version", "row_type", "id", "source_path", "source_sha256",
        "updated_from_git_commit", "generated_at",
    }
    other = {k: v for k, v in row.items() if k not in skip}
    if not other:
        return
    print(f"\n  fields:")
    for k in sorted(other.keys()):
        v = other[k]
        if isinstance(v, (list, dict)):
            print(f"    {k}: {json.dumps(v, default=str)}")
        else:
            print(f"    {k}: {v}")
    print()


# ─── Leaf: query ────────────────────────────────────────────────────────────


def _matches_filter(row: dict, *, field: Optional[str],
                    equals: Optional[str], contains: Optional[str],
                    expires_before: Optional[datetime]) -> bool:
    """AND-only filter evaluation. All three value filters (--equals,
    --contains, --expires-before) require --field."""
    if field is None:
        # No field-level filter; row passes
        return True
    if field not in row:
        # Field absent → cannot match any value filter
        return False
    value = row[field]

    if equals is not None:
        if str(value) != str(equals):
            return False

    if contains is not None:
        if isinstance(value, str):
            if contains not in value:
                return False
        elif isinstance(value, (list, tuple)):
            if contains not in [str(x) for x in value]:
                return False
        else:
            # Cannot apply --contains to non-string/list
            return False

    if expires_before is not None:
        if value is None:
            return False
        try:
            row_date = datetime.fromisoformat(str(value)[:10])
        except (TypeError, ValueError):
            return False
        if row_date >= expires_before:
            return False

    return True


def _valid_fields_for_row(row: dict) -> list[str]:
    return sorted(row.keys())


def run_query(args) -> int:
    rows = _load_catalog(
        from_path=getattr(args, "from_path", None),
        strict_fresh=getattr(args, "strict_fresh", False),
    )

    row_type = args.row_type
    if row_type != "any":
        rows = [r for r in rows if r.get("row_type") == row_type]

    field = args.field
    equals = args.equals
    contains = args.contains
    expires_before_raw = args.expires_before

    # Validation: value filters require --field
    if (equals or contains or expires_before_raw) and field is None:
        _io.error(
            "--equals / --contains / --expires-before require --field"
        )
        return 2

    # Parse ISO date
    expires_before_dt: Optional[datetime] = None
    if expires_before_raw is not None:
        try:
            expires_before_dt = datetime.fromisoformat(expires_before_raw)
        except ValueError:
            _io.error(
                f"--expires-before {expires_before_raw!r}: not a valid ISO date (YYYY-MM-DD)"
            )
            return 2

    # Validate --field against row type — clear error per operator §10.1
    if field is not None and rows:
        sample = rows[0]
        if field not in sample:
            valid = _valid_fields_for_row(sample)
            _io.error(
                f"--field {field!r}: no such field on row_type={row_type!r}. "
                f"valid fields: {', '.join(valid)}"
            )
            return 2

    matches = [
        r for r in rows
        if _matches_filter(
            r, field=field, equals=equals,
            contains=contains, expires_before=expires_before_dt,
        )
    ]

    if getattr(args, "json", False):
        print(json.dumps(matches, indent=2, sort_keys=True, default=str))
        return 0

    # Terminal: compact rows
    if not matches:
        _io.info(f"no rows match (row_type={row_type})")
        return 0

    for r in matches:
        rt = r.get("row_type", "?")
        rid = r.get("id", "?")
        # One-line summary per row
        summary_field = field if field else "status"
        summary_value = r.get(summary_field, "")
        print(f"  {rt}/{rid}  {summary_field}={summary_value}")
    print(f"\n  matched {len(matches)} row(s)")
    return 0
