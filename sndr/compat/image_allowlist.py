# SPDX-License-Identifier: Apache-2.0
"""KNOWN_GOOD_IMAGES allowlist — Wave 4.1 (audit closure 2026-05-09).

Closes club-3090 issue #60: nightly image swaps trigger Marlin repack
OOM. Operators pull `vllm/vllm-openai:nightly` blindly and end up
with a vllm/torch combo we never tested. This module ships an
explicit allowlist of (image_digest, vllm_pin, torch_pin, validated_at)
tuples so `sndr doctor --full` can say "this image is on the
known-good list" or "this image has never been benched here".

Format
──────
Each entry is a `KnownGoodImage` dataclass with:
  - `image_repo`: `vllm/vllm-openai`
  - `image_digest`: full sha256 (`vllm/vllm-openai@sha256:abc...`)
  - `vllm_pin`: `0.20.2rc1.dev93+g51f22dcfd`
  - `torch_pin`: `2.11.0` (None if not pinned)
  - `validated_at`: ISO 8601
  - `validated_on`: hardware tag (`a5000-2x` etc.)
  - `bench_url`: link to bench result (or repo-relative path)
  - `notes`: free text

Update protocol:
  1. New nightly image gets pulled.
  2. Run `sndr launch <preset>` + canonical bench.
  3. If bench passes verify_tolerances → add an entry HERE via PR.
  4. Otherwise: mark as `NOT_TESTED`; warn operators via doctor.

Without this module, the audit trail is "we hope nightly works".
With this module, every accepted image has a paper trail.

Author: Sandermage(Sander)-Barzov Aleksandr.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class KnownGoodImage:
    """One validated container image entry."""
    image_repo: str
    image_digest: str           # vllm/vllm-openai@sha256:...
    vllm_pin: str               # 0.20.2rc1.devNN+gXXXX
    validated_at: str           # ISO 8601 (e.g. 2026-05-09T14:32:39Z)
    validated_on: str           # hardware tag
    bench_url: str              # repo-relative path or external link
    torch_pin: Optional[str] = None
    triton_pin: Optional[str] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.image_digest.startswith(self.image_repo):
            raise ValueError(
                f"image_digest {self.image_digest!r} does not start with "
                f"image_repo {self.image_repo!r}"
            )
        if "@sha256:" not in self.image_digest:
            raise ValueError(
                f"image_digest {self.image_digest!r} must include "
                "'@sha256:' (use docker inspect RepoDigests, not a tag)"
            )
        # Validate ISO 8601 by parsing
        try:
            _dt.datetime.fromisoformat(
                self.validated_at.replace("Z", "+00:00")
            )
        except ValueError as e:
            raise ValueError(
                f"validated_at {self.validated_at!r} is not ISO 8601: {e}"
            )


# ─── The allowlist ──────────────────────────────────────────────────────


KNOWN_GOOD_IMAGES: tuple[KnownGoodImage, ...] = (
    KnownGoodImage(
        image_repo="vllm/vllm-openai",
        image_digest=(
            "vllm/vllm-openai@sha256:"
            "9b534fe66daf152e8ceca8a7f8e14c18105aaf6ddabc61eb17730d85b4c7c194"
        ),
        vllm_pin="0.20.2rc1.dev93+g51f22dcfd",
        torch_pin="2.11.0",
        validated_at="2026-05-09T14:32:39Z",
        validated_on="a5000-2x",
        bench_url=(
            "Genesis_internal_docs/bench_2026_05_09/wave1+2_canonical/"
            "35b_PROD_v11.json"
        ),
        notes=(
            "v11.0.0 + Wave 1+2 patches enabled. "
            "Bench: 35B FP8 PROD wall_TPS=231.41 (+2.3% over Wave 0 baseline), "
            "27B INT4+TQ k8v4 wall_TPS=116.28 (no regression). "
            "P82, P107, PN17, PN56, PN67, P95 all verified APPLY/idempotent. "
            "P71 enabled in 35B only (27B no measurable benefit). "
            "PN52, PN66, PN9 self-retired upstream-merged."
        ),
    ),
    # Historical — kept for reproducibility:
    KnownGoodImage(
        image_repo="vllm/vllm-openai",
        image_digest=(
            "vllm/vllm-openai@sha256:"
            # Synthesized from memory entry "v759 PROD" (2026-04-28).
            # Real digest unknown; placeholder ALL-zero indicates an
            # entry whose origin predates digest-pinning. Mark as
            # `historical=True` for filtering.
            "0000000000000000000000000000000000000000000000000000000000000000"
        ),
        vllm_pin="0.20.1rc1.dev16+g7a1eb8ac2",
        torch_pin="2.11.0",
        validated_at="2026-04-28T00:00:00Z",
        validated_on="a5000-2x",
        bench_url="historical/v759_320k_validated",
        notes=(
            "HISTORICAL — digest unknown. v7.59 baseline 320K context "
            "promoted to PROD. No machine-verifiable digest pin from "
            "this era; entry kept for vllm_pin allowlist consistency. "
            "DO NOT use this digest as a verify target — it's a "
            "documentation marker only."
        ),
    ),
)


# Mark which entries are historical / non-verifiable
_HISTORICAL_DIGESTS: frozenset[str] = frozenset({
    "vllm/vllm-openai@sha256:" + "0" * 64,
})


def lookup_by_digest(digest: str) -> Optional[KnownGoodImage]:
    """Return the entry matching `digest` or None."""
    for entry in KNOWN_GOOD_IMAGES:
        if entry.image_digest == digest:
            return entry
    return None


def lookup_by_vllm_pin(pin: str) -> tuple[KnownGoodImage, ...]:
    """All entries that record `pin` as their vllm version. Multiple
    images can carry the same vllm pin (e.g. base + nightly variants)."""
    return tuple(e for e in KNOWN_GOOD_IMAGES if e.vllm_pin == pin)


def is_known_good(digest: str) -> bool:
    """Boolean check excluding historical placeholders."""
    if digest in _HISTORICAL_DIGESTS:
        return False
    return any(e.image_digest == digest for e in KNOWN_GOOD_IMAGES)


def list_active() -> tuple[KnownGoodImage, ...]:
    """All non-historical entries."""
    return tuple(
        e for e in KNOWN_GOOD_IMAGES
        if e.image_digest not in _HISTORICAL_DIGESTS
    )


def find_for_pin(vllm_pin: str) -> Optional[KnownGoodImage]:
    """Return the freshest `KnownGoodImage` for a given vllm pin, or None."""
    matches = lookup_by_vllm_pin(vllm_pin)
    active = [m for m in matches if m.image_digest not in _HISTORICAL_DIGESTS]
    if not active:
        return None
    return max(active, key=lambda e: e.validated_at)


def status_for(digest: str, vllm_pin: str) -> str:
    """Classify a (digest, vllm_pin) pair into one of:
        - "known_good"  — exact digest match
        - "pin_match"   — vllm_pin matches but digest unknown
        - "unknown"     — neither matches
        - "historical"  — only in historical placeholder list
    """
    if digest in _HISTORICAL_DIGESTS:
        return "historical"
    if is_known_good(digest):
        return "known_good"
    if lookup_by_vllm_pin(vllm_pin):
        return "pin_match"
    return "unknown"


__all__ = [
    "KnownGoodImage",
    "KNOWN_GOOD_IMAGES",
    "lookup_by_digest",
    "lookup_by_vllm_pin",
    "is_known_good",
    "list_active",
    "find_for_pin",
    "status_for",
]
