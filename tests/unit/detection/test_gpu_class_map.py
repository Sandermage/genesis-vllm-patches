# SPDX-License-Identifier: Apache-2.0
"""S-05 (2026-05-08) — gpu_class_map.classify_gpu()."""
from __future__ import annotations

import pytest

from sndr.detection.gpu_class_map import classify_gpu


@pytest.mark.parametrize("name,expected", [
    # Ampere
    ("NVIDIA RTX A5000", "rtx a5000"),
    ("NVIDIA GeForce RTX 3090", "rtx 3090"),
    ("NVIDIA A100-SXM4-80GB", "a100"),
    # Ada — specific BEFORE general matters
    ("NVIDIA GeForce RTX 4060 Ti", "rtx 4060 ti"),
    ("NVIDIA GeForce RTX 4060", "rtx 4060"),
    ("NVIDIA GeForce RTX 4070 Ti SUPER", "rtx 4070 ti super"),
    ("NVIDIA GeForce RTX 4070 Ti", "rtx 4070 ti"),
    ("NVIDIA GeForce RTX 4070 SUPER", "rtx 4070 super"),
    ("NVIDIA GeForce RTX 4070", "rtx 4070"),
    ("NVIDIA GeForce RTX 4080 SUPER", "rtx 4080 super"),
    ("NVIDIA GeForce RTX 4080", "rtx 4080"),
    ("NVIDIA GeForce RTX 4090", "rtx 4090"),
    # Hopper
    ("NVIDIA H100 80GB HBM3", "h100"),
    ("NVIDIA H200 141GB HBM3e", "h200"),
    # Blackwell consumer
    ("NVIDIA GeForce RTX 5090", "rtx 5090"),
    ("NVIDIA GeForce RTX 5070 Ti", "rtx 5070 ti"),
    # Blackwell pro — Max-Q is more specific than base
    ("NVIDIA RTX PRO 6000 Blackwell Max-Q", "rtx pro 6000 blackwell max-q"),
    ("NVIDIA RTX PRO 6000 Blackwell", "rtx pro 6000 blackwell"),
    # Edge cases
    ("", ""),
    ("AMD MI300X", ""),
    ("Apple M3 Pro", ""),
])
def test_classify_gpu_known_patterns(name: str, expected: str) -> None:
    assert classify_gpu(name) == expected


def test_classify_gpu_case_insensitive() -> None:
    assert classify_gpu("NVIDIA RTX A5000") == "rtx a5000"
    assert classify_gpu("nvidia rtx a5000") == "rtx a5000"
    assert classify_gpu("NVIDIA RTX a5000") == "rtx a5000"


def test_classify_gpu_specific_before_general_4060() -> None:
    """4060 Ti string matches 4060 Ti, not bare 4060 (order matters)."""
    assert classify_gpu("RTX 4060 Ti") == "rtx 4060 ti"


def test_classify_gpu_specific_before_general_blackwell() -> None:
    """Max-Q match must beat the bare Blackwell match."""
    assert (
        classify_gpu("RTX PRO 6000 Blackwell Max-Q")
        == "rtx pro 6000 blackwell max-q"
    )
