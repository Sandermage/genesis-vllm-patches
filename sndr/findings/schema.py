# SPDX-License-Identifier: Apache-2.0
"""Findings schema dataclass (matches external_findings/*.yaml).

Mirrors the schema in EXTERNAL_FINDINGS_PIPELINE_2026-05-12_RU.md §1.
Validation is done by `validator.py` so the dataclass stays simple and
deserialization stays lenient (forward-compat).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ─── Vocabularies (single source of truth) ─────────────────────────────


VALID_STATUSES: frozenset[str] = frozenset({
    "watch",
    "skip",
    "needs-reproducer",
    "needs-bench",
    "backport-now",
    "doctor-rule",
    "config-recipe",
    "retire-local-patch",
    "done",
})

VALID_SOURCES: frozenset[str] = frozenset({
    "vllm-pr",
    "vllm-issue",
    "club-3090",
    "sglang",
    "lmcache",
    "paper",
    "blog",
    "reddit",
    "other",
})

VALID_CATEGORIES: frozenset[str] = frozenset({
    "memory-cache",
    "spec-decode",
    "tool-call",
    "sampling",
    "scheduler",
    "tracing",
    "quantization",
    "kernel",
    "misc",
})

VALID_RISKS: frozenset[str] = frozenset({"low", "medium", "high"})

VALID_CADENCES: frozenset[str] = frozenset({
    "weekly", "biweekly", "on-pin-bump", "retired",
})

VALID_TARGETS: frozenset[str] = frozenset({
    "patch-backport",
    "doctor-rule",
    "config-recipe",
    "retire-local-patch",
    "documentation",
    "test-only",
})


# State machine — which transitions are legal. Captures the diagram from
# EXTERNAL_FINDINGS_PIPELINE_2026-05-12_RU.md §2.
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "watch": frozenset({
        "skip", "needs-reproducer", "needs-bench",
        "backport-now", "doctor-rule", "config-recipe",
    }),
    "needs-reproducer": frozenset({
        "needs-bench", "backport-now", "skip", "watch",
    }),
    "needs-bench": frozenset({
        "backport-now", "doctor-rule", "config-recipe", "skip", "watch",
    }),
    "backport-now": frozenset({"done", "retire-local-patch", "skip"}),
    "doctor-rule": frozenset({"done", "skip"}),
    "config-recipe": frozenset({"done", "skip"}),
    "done": frozenset({"retire-local-patch"}),
    "retire-local-patch": frozenset(),    # terminal
    "skip": frozenset({"watch"}),         # operator can reopen
}


# ─── ISO-8601 dates (compact) ───────────────────────────────────────────


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _is_iso_date(s: str) -> bool:
    return bool(_ISO_DATE_RE.match(s))


# ─── Dataclass ─────────────────────────────────────────────────────────


@dataclass
class FindingValidationIssue:
    """One row of validator output."""
    rule: str
    severity: str                       # "error" | "warning"
    message: str


@dataclass
class Finding:
    """Structured external finding. One per `<id>.yaml`."""
    schema_version: int
    id: str
    source: str
    url: str
    title: str
    discovered_at: str                   # ISO date
    category: str
    status: str
    risk: str
    acceptance: str
    last_reviewed: str                   # ISO date
    review_cadence: str

    # Optional fields with explicit defaults.
    relevance: Optional[str] = None
    affected_genesis_paths: list[str] = field(default_factory=list)
    target: list[str] = field(default_factory=list)
    risk_notes: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    # ─── Lightweight invariant checks (called by validator) ──────────

    def validate(self) -> list[FindingValidationIssue]:
        out: list[FindingValidationIssue] = []
        if self.schema_version != 1:
            out.append(FindingValidationIssue(
                "schema", "error",
                f"schema_version={self.schema_version} unsupported (expected 1)",
            ))
        if not self.id:
            out.append(FindingValidationIssue(
                "schema", "error", "id is required and must be non-empty",
            ))
        if self.source not in VALID_SOURCES:
            out.append(FindingValidationIssue(
                "schema", "error",
                f"source={self.source!r} not in {sorted(VALID_SOURCES)}",
            ))
        if self.category not in VALID_CATEGORIES:
            out.append(FindingValidationIssue(
                "schema", "error",
                f"category={self.category!r} not in {sorted(VALID_CATEGORIES)}",
            ))
        if self.status not in VALID_STATUSES:
            out.append(FindingValidationIssue(
                "schema", "error",
                f"status={self.status!r} not in {sorted(VALID_STATUSES)}",
            ))
        if self.risk not in VALID_RISKS:
            out.append(FindingValidationIssue(
                "schema", "error",
                f"risk={self.risk!r} not in {sorted(VALID_RISKS)}",
            ))
        if self.review_cadence not in VALID_CADENCES:
            out.append(FindingValidationIssue(
                "schema", "error",
                f"review_cadence={self.review_cadence!r} not in {sorted(VALID_CADENCES)}",
            ))
        if not _is_iso_date(self.discovered_at):
            out.append(FindingValidationIssue(
                "schema", "error",
                f"discovered_at={self.discovered_at!r} must be YYYY-MM-DD",
            ))
        if not _is_iso_date(self.last_reviewed):
            out.append(FindingValidationIssue(
                "schema", "error",
                f"last_reviewed={self.last_reviewed!r} must be YYYY-MM-DD",
            ))
        if not self.url.startswith(("http://", "https://", "file://")):
            out.append(FindingValidationIssue(
                "schema", "warning",
                f"url={self.url!r} does not start with http(s)/file scheme",
            ))
        if not self.acceptance.strip():
            out.append(FindingValidationIssue(
                "schema", "error",
                "acceptance criterion is required and must be non-empty",
            ))
        for t in self.target:
            if t not in VALID_TARGETS:
                out.append(FindingValidationIssue(
                    "schema", "warning",
                    f"target {t!r} not in canonical set {sorted(VALID_TARGETS)} "
                    f"— accepted but operator may have intended a typo",
                ))
        return out
