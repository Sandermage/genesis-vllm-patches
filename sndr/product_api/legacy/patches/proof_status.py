# SPDX-License-Identifier: Apache-2.0
"""Pure-API layer for ``sndr patches proof-status`` — M.6.2.

Wraps :func:`vllm.sndr_core.proof.summarize_proof_status` with explicit
bucket-filter validation. Unknown bucket names raise
:class:`UnknownBucketError` so the CLI renders the same operator
message it did pre-M.6.2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


class UnknownBucketError(ValueError):
    """Raised when a bucket filter contains an unrecognised name.

    Attributes:
        unknown: The buckets the operator passed that weren't valid.
        valid: The canonical bucket list, for the operator-facing
            error message.
    """

    def __init__(self, unknown: list[str], valid: list[str]) -> None:
        super().__init__(
            f"unknown bucket(s): {unknown!r}. Valid: {valid}"
        )
        self.unknown = list(unknown)
        self.valid = list(valid)


@dataclass(frozen=True)
class ProofStatusResult:
    """Bucket summary of every patch's proof-artefact state."""

    total: int
    counts: dict[str, int]
    patches: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    filter_buckets: Optional[tuple[str, ...]] = None


def proof_status(
    *,
    out_dir: Optional[Path] = None,
    bucket_filter: Optional[Iterable[str]] = None,
) -> ProofStatusResult:
    """Summarise patches' proof-artefact buckets, optionally filtered.

    Raises :class:`UnknownBucketError` if ``bucket_filter`` contains a
    name not in :data:`vllm.sndr_core.proof.PROOF_STATUS_BUCKETS`.
    """
    from vllm.sndr_core.proof import (
        DEFAULT_PROOF_DIR,
        PROOF_STATUS_BUCKETS,
        summarize_proof_status,
    )

    target_dir = Path(out_dir) if out_dir is not None else DEFAULT_PROOF_DIR
    summary = summarize_proof_status(out_dir=target_dir)

    filter_buckets: Optional[tuple[str, ...]] = None
    if bucket_filter is not None:
        provided = list(bucket_filter)
        unknown = [b for b in provided if b not in PROOF_STATUS_BUCKETS]
        if unknown:
            raise UnknownBucketError(unknown, list(PROOF_STATUS_BUCKETS))
        filter_set = set(provided)
        patches = tuple(
            p for p in summary["patches"] if p["bucket"] in filter_set
        )
        filter_buckets = tuple(sorted(filter_set))
    else:
        patches = tuple(summary["patches"])

    return ProofStatusResult(
        total=summary["total"],
        counts=dict(summary["counts"]),
        patches=patches,
        filter_buckets=filter_buckets,
    )
