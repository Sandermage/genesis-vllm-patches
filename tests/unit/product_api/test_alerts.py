# SPDX-License-Identifier: Apache-2.0
"""Tests for the hardware alert rules + deduplicating store."""
from sndr.product_api.legacy import alerts as A


def _tele(**gpu):
    base = {"temp_gpu": 60, "ecc_uncorrected": "0", "mem_used": 1000, "mem_total": 24000, "name": "RTX A5000"}
    base.update(gpu)
    return {"gpus": [base], "system": {}}


def test_temp_thresholds():
    assert A.evaluate_hardware("h", _tele(temp_gpu=60)) == []
    warn = A.evaluate_hardware("h", _tele(temp_gpu=82))
    assert warn and warn[0]["level"] == "warn" and warn[0]["category"] == "gpu"
    crit = A.evaluate_hardware("h", _tele(temp_gpu=90))
    assert crit[0]["level"] == "critical"


def test_ecc_and_vram():
    ecc = A.evaluate_hardware("h", _tele(ecc_uncorrected="3"))
    assert any(a["level"] == "critical" and "ECC" in a["title"] for a in ecc)
    vram = A.evaluate_hardware("h", _tele(mem_used=23800, mem_total=24000))
    assert any(a["key"].endswith("vram") and a["level"] == "warn" for a in vram)


def test_disk_and_ram():
    t = {"gpus": [], "system": {"disk": {"mount": "/", "used_pct": 94, "free_gb": 50},
                                "ram_total_gb": 64, "ram_used_gb": 62}}
    fired = A.evaluate_hardware("h", t)
    assert any(a["category"] == "disk" and a["level"] == "critical" for a in fired)
    assert any(a["category"] == "host" and a["level"] == "warn" for a in fired)


def test_store_dedup_and_resolve():
    s = A.AlertStore()
    fired = A.evaluate_hardware("h", _tele(temp_gpu=90))
    s.update(fired, now=100.0)
    snap1 = s.snapshot()
    assert len(snap1["active"]) == 1 and snap1["counts"]["critical"] == 1
    first_seen = snap1["active"][0]["first_seen"]

    # Same condition again → still one active, first_seen preserved.
    s.update(A.evaluate_hardware("h", _tele(temp_gpu=91)), now=104.0)
    snap2 = s.snapshot()
    assert len(snap2["active"]) == 1
    assert snap2["active"][0]["first_seen"] == first_seen
    assert snap2["active"][0]["last_seen"] == 104.0

    # Condition clears → moves to recent (resolved).
    s.update([], now=108.0)
    snap3 = s.snapshot()
    assert snap3["active"] == []
    assert snap3["recent"] and snap3["recent"][0]["resolved_at"] == 108.0
