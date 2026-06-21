"""Phase 3 — per-pin manifest resolver tests (local; dev148 anchors.json is committed)."""
from sndr.engines.vllm.wiring.anchor_manifest import (
    normalize_pin,
    per_pin_manifest_path,
    is_pin_supported,
    list_supported_pins,
)

DEV148 = "0.23.1rc1.dev148+gb4c80ec0f"


def test_normalize_pin():
    assert normalize_pin(DEV148) == "0.23.1_b4c80ec0f"
    assert normalize_pin("0.21.1rc0+g626fa9bba566") == "0.21.1_626fa9bba"
    assert normalize_pin(None) is None
    assert normalize_pin("no-sha-here") is None


def test_per_pin_path_points_at_committed_manifest():
    p = per_pin_manifest_path(DEV148)
    assert p is not None
    assert p.name == "anchors.json"
    assert "0.23.1_b4c80ec0f" in str(p)


def test_is_pin_supported_dev148_true_unknown_false():
    assert is_pin_supported(DEV148) is True            # we generated it
    assert is_pin_supported("9.9.9+gdeadbeef0") is False
    assert is_pin_supported(None) is False


def test_list_supported_pins_includes_dev148():
    assert "0.23.1_b4c80ec0f" in list_supported_pins()
