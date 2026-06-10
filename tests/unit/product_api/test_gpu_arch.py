# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the GPU architecture advisor."""
from __future__ import annotations

from sndr.product_api.legacy import gpu_arch


def test_ampere_a5000_is_storage_only_fp8():
    c = gpu_arch.classify(name="NVIDIA RTX A5000")
    assert c["arch"].startswith("Ampere")
    assert c["compute_cap"] == 8.6
    assert c["fp8_kv_native"] is False and c["flash_attention"] == 2
    assert any("storage-only" in r["text"] for r in c["recommendations"])


def test_ada_4090_native_fp8():
    c = gpu_arch.classify(name="NVIDIA GeForce RTX 4090")
    assert c["arch"] == "Ada Lovelace"
    assert c["fp8_kv_native"] is True and c["fp8_weights_native"] is True


def test_blackwell_5090_has_fp4_and_fa3():
    c = gpu_arch.classify(name="NVIDIA GeForce RTX 5090")
    assert c["arch"] == "Blackwell"
    assert c["fp4_weights"] is True and c["flash_attention"] == 3


def test_explicit_compute_cap_wins():
    c = gpu_arch.classify(name="Mystery GPU", compute_cap="9.0")
    assert c["arch"] == "Hopper" and c["fp8_kv_native"] is True


def test_unknown_gpu_is_graceful():
    c = gpu_arch.classify(name="Some Random Card")
    assert c["arch"] == "unknown" and c["compute_cap"] is None
    assert any("Could not determine" in r["text"] for r in c["recommendations"])
