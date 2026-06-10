# SPDX-License-Identifier: Apache-2.0
"""Tests for the hardware fit / requirements report."""
from __future__ import annotations

import pytest

from sndr.product_api.legacy.memory import estimate_fit


def test_estimate_fit_prod_35b_on_a5000_is_compatible():
    report = estimate_fit(
        model_id="qwen3.6-35b-a3b-fp8",
        hardware_id="a5000-2x-24gbvram-16cpu-128gbram",
    )
    assert report.model_id == "qwen3.6-35b-a3b-fp8"
    assert report.compatible is True
    by_id = {c.id: c for c in report.checks}
    assert by_id["gpu_count"].ok is True
    assert "cuda" in by_id
    # VRAM is informational (catalog gives a conservative floor, not real VRAM).
    assert report.vram["model_min_mib"] >= 1
    assert report.vram["rig_floor_mib"] >= 1


def test_estimate_fit_unknown_raises():
    with pytest.raises(Exception):
        estimate_fit(model_id="not-a-model", hardware_id="a5000-2x-24gbvram-16cpu-128gbram")
