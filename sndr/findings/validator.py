# SPDX-License-Identifier: Apache-2.0
"""Findings validator — schema + cross-finding rules.

Per-finding rules live on `Finding.validate()`; this module layers:

  F-1  id uniqueness across the directory
  F-2  state-machine transition legality (when an update happens)
  F-3  acceptance non-empty
  F-4  staleness (last_reviewed past cadence — warning)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .registry import (
    DEFAULT_FINDINGS_DIR,
    discover_findings,
    is_due_for_review,
)
from .schema import (
    ALLOWED_TRANSITIONS,
    Finding,
    FindingValidationIssue,
)


__all__ = [
    "ValidationResult",
    "validate_finding",
    "validate_directory",
    "is_valid_transition",
]


log = logging.getLogger("genesis.findings.validator")


@dataclass
class ValidationResult:
    findings: list[Finding] = field(default_factory=list)
    issues: list[FindingValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[FindingValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[FindingValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def passed(self) -> bool:
        return not self.errors


def is_valid_transition(from_status: str, to_status: str) -> bool:
    """Returns True iff `from_status → to_status` is a documented edge
    of the state machine. Same-status (no-op) is always valid."""
    if from_status == to_status:
        return True
    allowed = ALLOWED_TRANSITIONS.get(from_status, frozenset())
    return to_status in allowed


def validate_finding(
    finding: Finding,
    *,
    known_ids: Optional[set[str]] = None,
    today: Optional[object] = None,
) -> list[FindingValidationIssue]:
    """Run schema + staleness checks against ONE finding."""
    issues = list(finding.validate())

    # F-3 acceptance non-empty already enforced by Finding.validate().
    # F-4 staleness — warning, not error.
    if is_due_for_review(finding, today=today):
        issues.append(FindingValidationIssue(
            "F-4", "warning",
            f"finding {finding.id!r} is past its {finding.review_cadence!r} "
            f"cadence (last_reviewed={finding.last_reviewed}) — review needed",
        ))
    return issues


def validate_directory(root: Path = DEFAULT_FINDINGS_DIR) -> ValidationResult:
    """Walk the findings directory and return aggregate validation result.

    Cross-finding rules (F-1 id uniqueness) need the full set visible.
    """
    result = ValidationResult()
    loaded = discover_findings(root)

    # F-1 id uniqueness
    seen: dict[str, Path] = {}
    for path, f in loaded:
        if not f.id:
            continue                    # schema error already caught per-finding
        if f.id in seen:
            result.issues.append(FindingValidationIssue(
                "F-1", "error",
                f"duplicate finding id {f.id!r}: also declared by {seen[f.id]}",
            ))
        else:
            seen[f.id] = path
        result.findings.append(f)

    # Per-finding schema + staleness
    for _path, f in loaded:
        for issue in validate_finding(f):
            result.issues.append(issue)
    return result
