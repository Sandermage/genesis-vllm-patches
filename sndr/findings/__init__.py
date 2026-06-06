# SPDX-License-Identifier: Apache-2.0
"""External findings pipeline (Roadmap §5 deferred Phase 5 deliverable).

Structured tracking for upstream vLLM PRs/issues, club-3090 reports,
SGLang/LMCache observations, and research papers. Each finding is a
self-contained YAML under `docs/_internal/external_findings/<id>.yaml`.

Design: `docs/_internal/EXTERNAL_FINDINGS_PIPELINE_2026-05-12_RU.md`.

CLI surface (cli/findings.py):

    sndr findings list [--status <s>] [--due-for-review] [--json]
    sndr findings add --source ... --url ... --category ... ...
    sndr findings update <id> --status ... [--notes ...]
    sndr findings validate
"""
from __future__ import annotations

from .schema import (
    Finding,
    FindingValidationIssue,
    VALID_STATUSES,
    VALID_SOURCES,
    VALID_CATEGORIES,
    VALID_RISKS,
    VALID_CADENCES,
    VALID_TARGETS,
)
from .registry import (
    DEFAULT_FINDINGS_DIR,
    load_finding,
    list_finding_paths,
    discover_findings,
    is_due_for_review,
)
from .validator import (
    validate_finding,
    validate_directory,
    is_valid_transition,
)


__all__ = [
    # Schema
    "Finding",
    "FindingValidationIssue",
    "VALID_STATUSES",
    "VALID_SOURCES",
    "VALID_CATEGORIES",
    "VALID_RISKS",
    "VALID_CADENCES",
    "VALID_TARGETS",
    # Registry
    "DEFAULT_FINDINGS_DIR",
    "load_finding",
    "list_finding_paths",
    "discover_findings",
    "is_due_for_review",
    # Validator
    "validate_finding",
    "validate_directory",
    "is_valid_transition",
]
