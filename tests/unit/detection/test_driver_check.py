# SPDX-License-Identifier: Apache-2.0
"""S-05 (2026-05-08) — driver_check.probe_driver()."""
from __future__ import annotations

import sndr.engines.vllm.detection.driver_check as M


def test_no_nvidia_smi_returns_skip_recommendation(monkeypatch):
    """When nvidia-smi is missing, probe reports skipped check."""
    monkeypatch.setattr("shutil.which", lambda _: None)
    info = M.probe_driver()
    assert info.nvidia_smi_present is False
    assert info.raw_driver_version is None
    assert info.below_recommended is False
    assert "nvidia-smi not found" in info.recommendation


def test_old_driver_below_recommended(monkeypatch):
    """Driver 535.x → flagged below_recommended (≥580 required)."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(M, "_read_driver_version", lambda: "535.183.06")
    monkeypatch.setattr(M, "_read_cuda_header", lambda: "12.4")
    info = M.probe_driver()
    assert info.nvidia_smi_present is True
    assert info.raw_driver_version == "535.183.06"
    assert info.driver_major == 535
    assert info.below_recommended is True
    assert "≥580" in info.recommendation


def test_modern_driver_passes(monkeypatch):
    """Driver 580.x passes the check."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(M, "_read_driver_version", lambda: "580.42.01")
    monkeypatch.setattr(M, "_read_cuda_header", lambda: "13.0")
    info = M.probe_driver()
    assert info.nvidia_smi_present is True
    assert info.driver_major == 580
    assert info.below_recommended is False
    assert "CUDA 13.0 compatible" in info.recommendation


def test_unparseable_driver_string(monkeypatch):
    """Garbage driver string → driver_major=None, no false-positive flag."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(M, "_read_driver_version", lambda: "")
    monkeypatch.setattr(M, "_read_cuda_header", lambda: None)
    info = M.probe_driver()
    assert info.nvidia_smi_present is True
    assert info.driver_major is None
    assert info.below_recommended is False


def test_major_int_parser():
    """Helper extracts integer major from version strings."""
    assert M._major_int("580.42.01") == 580
    assert M._major_int("535.183.06") == 535
    assert M._major_int("470") == 470
    assert M._major_int("") is None
    assert M._major_int("abc") is None
