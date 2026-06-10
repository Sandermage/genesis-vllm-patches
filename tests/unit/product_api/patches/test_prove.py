# SPDX-License-Identifier: Apache-2.0
"""Tests for ``product_api.patches.prove`` — M.6.2."""
from __future__ import annotations

from sndr.product_api.legacy.patches import prove
from sndr.product_api.legacy.patches.prove import (
    DeadDetectResult,
    ProveAllResult,
    ProveOneResult,
)


class TestProveOne:
    def test_known_patch_no_write(self, tmp_path):
        result = prove.prove_one("P67", out_dir=tmp_path, no_write=True)
        assert isinstance(result, ProveOneResult)
        assert result.proof.patch_id == "P67"
        # ``no_write=True`` must guarantee no artefact creation.
        assert result.artefact_path is None
        assert not list(tmp_path.iterdir())

    def test_known_patch_writes_artefact(self, tmp_path):
        result = prove.prove_one("P67", out_dir=tmp_path, no_write=False)
        # P67 is registered + apply_module is importable, so P-1 passes
        # and the artefact gets written.
        assert result.artefact_path is not None
        assert result.artefact_path.exists()
        # File lives under the requested out_dir.
        assert tmp_path in result.artefact_path.parents

    def test_unknown_patch_does_not_write(self, tmp_path):
        result = prove.prove_one(
            "PXXXX_NOT_REAL", out_dir=tmp_path, no_write=False,
        )
        # P-1 (patch in registry) fails → we explicitly skip the write to
        # avoid persisting "patch not found" as evidence.
        assert result.artefact_path is None
        assert result.static_passed is False
        assert not list(tmp_path.iterdir())


class TestProveAll:
    def test_returns_sweep(self, tmp_path):
        sweep = prove.prove_all(out_dir=tmp_path, no_write=True)
        assert isinstance(sweep, ProveAllResult)
        assert sweep.total >= 100
        assert sweep.passed + sweep.failed == sweep.total
        assert 0.0 <= sweep.coverage_pct <= 100.0

    def test_no_write_creates_no_artefacts(self, tmp_path):
        prove.prove_all(out_dir=tmp_path, no_write=True)
        assert not list(tmp_path.iterdir())

    def test_results_shape(self, tmp_path):
        sweep = prove.prove_all(out_dir=tmp_path, no_write=True)
        for r in sweep.results:
            assert set(r.keys()) >= {"patch_id", "passed", "errors"}


class TestDeadDetect:
    def test_returns_report(self, tmp_path):
        report = prove.dead_detect(out_dir=tmp_path)
        assert isinstance(report, DeadDetectResult)
        assert report.total_patches >= 100
        assert report.proven + report.dead_count == report.total_patches

    def test_dead_patch_shape(self, tmp_path):
        report = prove.dead_detect(out_dir=tmp_path)
        for d in report.dead_patches:
            assert "patch_id" in d
            assert "lifecycle" in d
            assert "tier" in d
            assert "family" in d
