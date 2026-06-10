# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr.compat.image_allowlist` — Wave 4.1.

Closes club-3090 #60 (Marlin repack OOM after blind nightly pull).
The allowlist is the audit trail; tests pin the contract so future
sessions can't accidentally drop validated entries or accept malformed
ones.
"""
from __future__ import annotations

import pytest

from sndr.compat.image_allowlist import (
    KNOWN_GOOD_IMAGES,
    KnownGoodImage,
    find_for_pin,
    is_known_good,
    list_active,
    lookup_by_digest,
    lookup_by_vllm_pin,
    status_for,
)


# ─── KnownGoodImage dataclass ───────────────────────────────────────────


class TestDataclass:
    def test_canonical_construction(self):
        e = KnownGoodImage(
            image_repo="vllm/vllm-openai",
            image_digest="vllm/vllm-openai@sha256:" + "a" * 64,
            vllm_pin="0.20.2rc1.dev93+g51f22dcfd",
            validated_at="2026-05-09T14:32:39Z",
            validated_on="a5000-2x",
            bench_url="bench/x.json",
        )
        assert e.image_repo == "vllm/vllm-openai"

    def test_rejects_repo_mismatch(self):
        with pytest.raises(ValueError, match="does not start with"):
            KnownGoodImage(
                image_repo="vllm/vllm-openai",
                image_digest="other/repo@sha256:" + "a" * 64,
                vllm_pin="0.20.2rc1.dev93+g51f22dcfd",
                validated_at="2026-05-09T14:32:39Z",
                validated_on="a5000-2x",
                bench_url="x",
            )

    def test_rejects_tag_only(self):
        with pytest.raises(ValueError, match="@sha256:"):
            KnownGoodImage(
                image_repo="vllm/vllm-openai",
                image_digest="vllm/vllm-openai:nightly",
                vllm_pin="0.20.2rc1.dev93+g51f22dcfd",
                validated_at="2026-05-09T14:32:39Z",
                validated_on="a5000-2x",
                bench_url="x",
            )

    def test_rejects_invalid_iso_date(self):
        with pytest.raises(ValueError, match="ISO 8601"):
            KnownGoodImage(
                image_repo="vllm/vllm-openai",
                image_digest="vllm/vllm-openai@sha256:" + "a" * 64,
                vllm_pin="0.20.2rc1.dev93+g51f22dcfd",
                validated_at="not-a-date",
                validated_on="a5000-2x",
                bench_url="x",
            )

    def test_immutable(self):
        e = KnownGoodImage(
            image_repo="vllm/vllm-openai",
            image_digest="vllm/vllm-openai@sha256:" + "a" * 64,
            vllm_pin="0.20.2rc1.dev93+g51f22dcfd",
            validated_at="2026-05-09T14:32:39Z",
            validated_on="a5000-2x",
            bench_url="x",
        )
        with pytest.raises(Exception):  # frozen=True
            e.vllm_pin = "different"  # type: ignore[misc]


# ─── Allowlist contents ─────────────────────────────────────────────────


class TestAllowlistContents:
    def test_at_least_one_active_entry(self):
        active = list_active()
        assert len(active) >= 1, "allowlist must have at least one validated image"

    def test_dev93_image_present(self):
        """Wave 1+2 baseline image must be in allowlist."""
        digest = (
            "vllm/vllm-openai@sha256:"
            "9b534fe66daf152e8ceca8a7f8e14c18105aaf6ddabc61eb17730d85b4c7c194"
        )
        entry = lookup_by_digest(digest)
        assert entry is not None, "dev93 + Wave 1+2 image must be allowlisted"
        assert entry.vllm_pin == "0.20.2rc1.dev93+g51f22dcfd"

    def test_all_entries_have_validated_at(self):
        for entry in KNOWN_GOOD_IMAGES:
            assert entry.validated_at, f"entry {entry.image_digest} missing validated_at"

    def test_no_duplicate_digests(self):
        digests = [e.image_digest for e in KNOWN_GOOD_IMAGES]
        assert len(digests) == len(set(digests)), "duplicate digests not allowed"


# ─── Lookup helpers ─────────────────────────────────────────────────────


class TestLookups:
    def test_lookup_by_digest_present(self):
        digest = (
            "vllm/vllm-openai@sha256:"
            "9b534fe66daf152e8ceca8a7f8e14c18105aaf6ddabc61eb17730d85b4c7c194"
        )
        e = lookup_by_digest(digest)
        assert e is not None
        assert e.image_digest == digest

    def test_lookup_by_digest_absent(self):
        assert lookup_by_digest("vllm/vllm-openai@sha256:" + "x" * 64) is None

    def test_lookup_by_vllm_pin_returns_tuple(self):
        results = lookup_by_vllm_pin("0.20.2rc1.dev93+g51f22dcfd")
        assert isinstance(results, tuple)
        assert len(results) >= 1

    def test_lookup_by_vllm_pin_no_match(self):
        results = lookup_by_vllm_pin("0.0.0.fake999")
        assert results == ()

    def test_is_known_good_true_for_active(self):
        digest = (
            "vllm/vllm-openai@sha256:"
            "9b534fe66daf152e8ceca8a7f8e14c18105aaf6ddabc61eb17730d85b4c7c194"
        )
        assert is_known_good(digest) is True

    def test_is_known_good_false_for_historical(self):
        # The all-zero historical placeholder must NOT count as known good
        historical = "vllm/vllm-openai@sha256:" + "0" * 64
        assert is_known_good(historical) is False

    def test_is_known_good_false_for_unknown(self):
        assert is_known_good("vllm/vllm-openai@sha256:" + "f" * 64) is False

    def test_find_for_pin_returns_freshest(self):
        e = find_for_pin("0.20.2rc1.dev93+g51f22dcfd")
        assert e is not None
        assert e.vllm_pin == "0.20.2rc1.dev93+g51f22dcfd"

    def test_find_for_pin_skips_historical(self):
        # Historical entry has all-zero digest; if filter works, it's excluded
        e = find_for_pin("0.20.1rc1.dev16+g7a1eb8ac2")
        assert e is None  # historical entries excluded from active lookup


# ─── status_for() ───────────────────────────────────────────────────────


class TestStatusFor:
    def test_known_good(self):
        digest = (
            "vllm/vllm-openai@sha256:"
            "9b534fe66daf152e8ceca8a7f8e14c18105aaf6ddabc61eb17730d85b4c7c194"
        )
        assert status_for(
            digest=digest, vllm_pin="0.20.2rc1.dev93+g51f22dcfd",
        ) == "known_good"

    def test_pin_match_unknown_digest(self):
        # Unknown digest, but vllm_pin matches a known entry
        assert status_for(
            digest="vllm/vllm-openai@sha256:" + "f" * 64,
            vllm_pin="0.20.2rc1.dev93+g51f22dcfd",
        ) == "pin_match"

    def test_unknown(self):
        assert status_for(
            digest="vllm/vllm-openai@sha256:" + "f" * 64,
            vllm_pin="0.0.0.unknown",
        ) == "unknown"

    def test_historical(self):
        historical = "vllm/vllm-openai@sha256:" + "0" * 64
        assert status_for(
            digest=historical,
            vllm_pin="0.20.1rc1.dev16+g7a1eb8ac2",
        ) == "historical"
