# SPDX-License-Identifier: Apache-2.0
"""CONFIG-UX.4.1 — rollout stage + severity helpers.

Stage-aware severity resolver shared by V1 deprecation surface,
card-less preset warnings, and the future override-policy enforcement.
At Stage 0/1 (the only modes that ship in CONFIG-UX.4.1), behavior is
observability-only — no caller sees a new ERROR severity unless they
opt into Stage 2+ explicitly via `SNDR_V1_ROLLOUT_STAGE`.

Operator-visible env vars:

  SNDR_V1_ROLLOUT_STAGE=0  current (warnings only, fully informational)
  SNDR_V1_ROLLOUT_STAGE=1  warnings + sndr doctor surface (CONFIG-UX.4.1)
  SNDR_V1_ROLLOUT_STAGE=2  warnings + --strict errors (CONFIG-UX.4.2)
  SNDR_V1_ROLLOUT_STAGE=3  ERROR default (CONFIG-UX.4.3)

  GENESIS_DISABLE_V1_DEPRECATION_WARNING=1  silences emitted warnings
                                            (existing escape hatch;
                                            does NOT silence ERROR
                                            severity at Stage 3+)

Bucket × stage matrix locked in CONFIG_UX_R §6.1; this module is the
single authority for "what severity should this event produce".
"""
from __future__ import annotations

import os
from typing import Literal


__all__ = [
    "Bucket",
    "Severity",
    "BUCKETS",
    "SEVERITIES",
    "rollout_stage",
    "effective_severity",
    "is_disabled",
    "DEFAULT_STAGE",
]


Bucket = Literal[
    "transparent",
    "needs_operator_choice",
    "deprecated",
    "tombstone",
    # CONFIG-UX.4.1 also reuses this severity helper for two non-V1
    # surfaces — card-less prod-* presets and missing-override-policy
    # profiles. We add synthetic bucket names so the same matrix
    # applies. Card-less non-prod stays "info" forever per operator
    # decision (CONFIG-UX.4.R §10.3).
    "card_less_prod",
    "card_less_non_prod",
    "missing_override_policy",
]

Severity = Literal["info", "warn", "error"]

BUCKETS: tuple[str, ...] = (
    "transparent",
    "needs_operator_choice",
    "deprecated",
    "tombstone",
    "card_less_prod",
    "card_less_non_prod",
    "missing_override_policy",
)

SEVERITIES: tuple[str, ...] = ("info", "warn", "error")

# Default stage in source — operator/CI opts in to higher stages via env.
# CONFIG-UX.4.1 shipped Stage 0 default; CONFIG-UX.4.2 (2026-05-24) flips
# to Stage 1. Operators reverting via `SNDR_V1_ROLLOUT_STAGE=0` see
# functionally identical observable output (severity matrix is unchanged
# between Stage 0 and Stage 1; the flip is preparation for Stage 2/3
# escalation in CONFIG-UX.4.3, not a behavioral change).
DEFAULT_STAGE: int = 1

# Stages allowed by the matrix. Operator-supplied values outside this
# range fall back to DEFAULT_STAGE with a one-time warning.
_VALID_STAGES: frozenset[int] = frozenset({0, 1, 2, 3})

_INVALID_STAGE_WARNED: set[str] = set()


def rollout_stage(*, env_value: str | None = None) -> int:
    """Resolve the current rollout stage from the process env.

    Args:
        env_value: testing hook to bypass os.environ. None reads
            `SNDR_V1_ROLLOUT_STAGE` from the live env.

    Returns:
        Integer stage in [0, 3]. Invalid values fall back to
        DEFAULT_STAGE with a one-time warning.
    """
    raw = env_value if env_value is not None else os.environ.get("SNDR_V1_ROLLOUT_STAGE", "")
    if not raw:
        return DEFAULT_STAGE
    try:
        n = int(raw)
    except ValueError:
        _warn_invalid_stage(raw)
        return DEFAULT_STAGE
    if n not in _VALID_STAGES:
        _warn_invalid_stage(raw)
        return DEFAULT_STAGE
    return n


def _warn_invalid_stage(raw: str) -> None:
    """One-time warning per invalid value seen."""
    if raw in _INVALID_STAGE_WARNED:
        return
    _INVALID_STAGE_WARNED.add(raw)
    import warnings
    warnings.warn(
        f"SNDR_V1_ROLLOUT_STAGE={raw!r} is invalid; expected 0/1/2/3. "
        f"Falling back to stage {DEFAULT_STAGE}.",
        UserWarning,
        stacklevel=3,
    )


def is_disabled() -> bool:
    """True if the global deprecation-silencer env is set.

    `GENESIS_DISABLE_V1_DEPRECATION_WARNING=1` silences all rollout
    warnings (informational + warn-severity). ERROR severity at
    Stage 2+/3 is NOT silenced — that's by design per CONFIG-UX.R §6.4.
    """
    return bool(os.environ.get("GENESIS_DISABLE_V1_DEPRECATION_WARNING"))


def effective_severity(
    *,
    bucket: Bucket,
    stage: int | None = None,
    strict_mode: bool = False,
) -> Severity:
    """Resolve effective severity for a rollout event.

    Args:
        bucket: classification of the event.
        stage: explicit stage override (CLI / tests). None reads env.
        strict_mode: True when an audit is invoked with --strict.

    Returns:
        Severity level — caller decides whether to print, raise,
        or suppress based on this.

    Matrix (locked in CONFIG_UX_R §6.1 + .4.R §2.2):

      bucket                    | s0   | s1   | s2 default | s2 strict | s3+
      ------------------------- + ---- + ---- + ---------- + --------- + ----
      transparent               | warn | warn | warn       | warn      | warn
      needs_operator_choice     | warn | warn | warn       | error     | error
      deprecated                | warn | warn | warn       | error     | error
      tombstone                 | error|error | error      | error     | error
      card_less_prod            | warn | warn | warn       | error     | error
      card_less_non_prod        | info | info | info       | info      | info
      missing_override_policy   | warn | warn | warn       | error     | error
    """
    if stage is None:
        stage = rollout_stage()

    # Tombstone is always ERROR regardless of stage.
    if bucket == "tombstone":
        return "error"

    # Transparent V1 keys stay WARN forever (regression guard).
    if bucket == "transparent":
        return "warn"

    # Card-less non-prod presets stay INFO indefinitely
    # (operator decision CONFIG-UX.4.R §10.3 — non-prod escalation
    # deferred to CONFIG-UX.2b separately).
    if bucket == "card_less_non_prod":
        return "info"

    # Buckets that follow the standard escalation curve:
    #   needs_operator_choice, deprecated, card_less_prod,
    #   missing_override_policy
    if stage <= 1:
        return "warn"
    if stage == 2:
        return "error" if strict_mode else "warn"
    # Stage 3+ — ERROR by default.
    return "error"
