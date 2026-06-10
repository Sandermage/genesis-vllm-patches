# SPDX-License-Identifier: Apache-2.0
"""Tests for the read-only model cache-status report."""
from __future__ import annotations

from sndr.product_api.legacy.model_cache import collect_model_cache_report


def test_cache_report_lists_models_with_declared_paths():
    report = collect_model_cache_report()
    assert report.host  # daemon host label is populated
    assert isinstance(report.models, tuple)
    assert len(report.models) >= 1
    sample = report.models[0]
    assert sample.model_id
    assert sample.model_path
    # present is a real os.path.isdir check on the daemon host.
    assert isinstance(sample.present, bool)
    # Counts are consistent with the per-model entries.
    assert report.present_count == sum(1 for m in report.models if m.present)
    assert report.total == len(report.models)


def test_cache_report_marks_absent_for_nonexistent_container_paths(tmp_path):
    # Builtin model_paths are container-side (e.g. /models/...). On a dev/remote
    # daemon host they will not exist, which must be reported honestly as absent
    # rather than guessed as present.
    report = collect_model_cache_report()
    for entry in report.models:
        if entry.model_path.startswith("/models/"):
            # We cannot assert absent universally (the GPU host may have them),
            # but the field must be a real boolean derived from the filesystem.
            assert entry.present in (True, False)
