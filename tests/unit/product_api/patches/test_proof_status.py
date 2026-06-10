# SPDX-License-Identifier: Apache-2.0
"""Tests for ``product_api.patches.proof_status`` — M.6.2."""
from __future__ import annotations

import pytest

from sndr.product_api.legacy.patches import proof_status
from sndr.product_api.legacy.patches.proof_status import (
    ProofStatusResult,
    UnknownBucketError,
)


class TestProofStatus:
    def test_returns_result(self, tmp_path):
        result = proof_status.proof_status(out_dir=tmp_path)
        assert isinstance(result, ProofStatusResult)
        assert result.total >= 100
        assert isinstance(result.counts, dict)

    def test_filter_buckets_none_when_unfiltered(self, tmp_path):
        result = proof_status.proof_status(out_dir=tmp_path)
        assert result.filter_buckets is None
        # Without filter, every bucketed patch shows up.
        assert len(result.patches) == result.total

    def test_known_bucket_filter(self, tmp_path):
        # Empty dir → all patches fall into the ``dead`` bucket; the
        # filter must reduce the patches tuple to those entries.
        result = proof_status.proof_status(
            out_dir=tmp_path, bucket_filter=["dead"],
        )
        assert result.filter_buckets == ("dead",)
        for p in result.patches:
            assert p["bucket"] == "dead"

    def test_unknown_bucket_raises(self, tmp_path):
        with pytest.raises(UnknownBucketError) as excinfo:
            proof_status.proof_status(
                out_dir=tmp_path, bucket_filter=["not-a-bucket"],
            )
        err = excinfo.value
        assert err.unknown == ["not-a-bucket"]
        assert isinstance(err.valid, list)
        assert len(err.valid) > 0

    def test_multi_bucket_filter_sorted(self, tmp_path):
        from sndr.proof import PROOF_STATUS_BUCKETS

        # Pick two valid bucket names in non-sorted order; result must
        # report them in sorted form.
        b1, b2 = list(PROOF_STATUS_BUCKETS)[1], list(PROOF_STATUS_BUCKETS)[0]
        result = proof_status.proof_status(
            out_dir=tmp_path, bucket_filter=[b1, b2],
        )
        assert result.filter_buckets == tuple(sorted({b1, b2}))
