# SPDX-License-Identifier: Apache-2.0
"""Findings filesystem registry — load, list, discover, staleness check.

External findings are review notes on upstream activity (vLLM PRs,
issues, papers) that the maintainer tracks privately. The default
directory is resolved at import time:

1. ``GENESIS_FINDINGS_DIR`` env var (absolute or repo-relative path),
   if set — wins outright.
2. The first directory pattern matched under the repo root that
   carries a ``*.yaml`` finding file. This lets the maintainer keep
   the actual notes outside the public tree without the loader
   hard-coding any specific layout.
3. The retired ``docs/_internal/external_findings`` namespace, kept
   as a final fallback so legacy checkouts keep working.

The loader is YAML-only (no entry-points yet — kept simple to match
the deferred design from EXTERNAL_FINDINGS_PIPELINE §3).
"""
from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from .schema import Finding


__all__ = [
    "DEFAULT_FINDINGS_DIR",
    "load_finding",
    "list_finding_paths",
    "discover_findings",
    "is_due_for_review",
]


log = logging.getLogger("genesis.findings.registry")


REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_findings_dir() -> Path:
    """Pick the active findings root using the precedence in the module
    docstring. Falls back to the legacy ``docs/_internal/external_findings``
    location even if it does not exist on disk — callers that walk the
    directory simply see an empty list, which is the documented
    "no-findings" state for a public clone."""
    env_dir = os.environ.get("GENESIS_FINDINGS_DIR")
    if env_dir:
        p = Path(env_dir)
        if not p.is_absolute():
            p = REPO_ROOT / p
        return p
    for candidate in REPO_ROOT.glob("*/planning/external_findings"):
        if candidate.is_dir():
            return candidate
    return REPO_ROOT / "docs" / "_internal" / "external_findings"


DEFAULT_FINDINGS_DIR = _resolve_findings_dir()


def _yaml_safe_load(path: Path) -> dict:
    import yaml
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: top-level YAML must be a mapping, got {type(data).__name__}"
        )
    return data


def _finding_from_dict(data: dict) -> Finding:
    """Materialize a Finding from a parsed YAML dict.

    Lenient — unknown keys are dropped, missing optional keys get
    dataclass defaults. The validator catches required-field gaps.
    """
    # Whitelist only declared dataclass fields so the loader stays
    # forward-compatible if the YAML grows extra metadata.
    allowed = {
        "schema_version", "id", "source", "url", "title", "discovered_at",
        "category", "status", "risk", "acceptance",
        "last_reviewed", "review_cadence",
        "relevance", "affected_genesis_paths", "target",
        "risk_notes", "notes",
    }
    clean = {k: v for k, v in data.items() if k in allowed}
    # Required fields with no default — populate explicit "" / sane fallbacks
    # so dataclass construction doesn't TypeError on incomplete YAML. The
    # validator then catches the empty values.
    for field_name, default in (
        ("schema_version", 0),
        ("id", ""),
        ("source", ""),
        ("url", ""),
        ("title", ""),
        ("discovered_at", ""),
        ("category", ""),
        ("status", ""),
        ("risk", ""),
        ("acceptance", ""),
        ("last_reviewed", ""),
        ("review_cadence", ""),
    ):
        clean.setdefault(field_name, default)
    # Lists must not be None.
    for list_field in ("affected_genesis_paths", "target", "notes"):
        if clean.get(list_field) is None:
            clean[list_field] = []
    return Finding(**clean)


def load_finding(path: Path) -> Finding:
    """Load and instantiate one finding YAML. Does NOT run validate()
    — caller chooses whether to enforce shape (validator.validate_finding)
    or just inspect the loaded data."""
    data = _yaml_safe_load(path)
    return _finding_from_dict(data)


def list_finding_paths(root: Path = DEFAULT_FINDINGS_DIR) -> list[Path]:
    """Walk the findings root and return every `*.yaml`.
    Files / dirs starting with `_` are skipped (e.g. `_template.yaml`)."""
    if not root.is_dir():
        return []
    paths: list[Path] = []
    for p in sorted(root.glob("*.yaml")):
        if p.name.startswith("_"):
            continue
        paths.append(p)
    return paths


def discover_findings(
    root: Path = DEFAULT_FINDINGS_DIR,
) -> list[tuple[Path, Finding]]:
    """Walk + load every finding under root. Findings that fail to parse
    are LOGGED and SKIPPED (matches community SDK discovery pattern).
    Callers wanting strict checks should use `validate_directory()`."""
    out: list[tuple[Path, Finding]] = []
    for path in list_finding_paths(root):
        try:
            f = load_finding(path)
        except Exception as e:
            log.warning("findings discovery: %s: %s", path, e)
            continue
        out.append((path, f))
    return out


# ─── Staleness ─────────────────────────────────────────────────────────


_CADENCE_DAYS: dict[str, Optional[int]] = {
    "weekly": 7,
    "biweekly": 14,
    "on-pin-bump": None,        # event-triggered, never time-stale
    "retired": None,            # closed findings don't go stale
}


def is_due_for_review(finding: Finding, today: Optional[date] = None) -> bool:
    """Return True if `finding.last_reviewed` is older than its cadence.

    Returns False when the cadence is event-triggered (`on-pin-bump`)
    or terminal (`retired`).
    """
    cadence_days = _CADENCE_DAYS.get(finding.review_cadence)
    if cadence_days is None:
        return False
    try:
        last = date.fromisoformat(finding.last_reviewed)
    except (ValueError, TypeError):
        # Malformed date → treat as stale so the operator notices.
        return True
    cutoff = (today or date.today()) - timedelta(days=cadence_days)
    return last < cutoff
