# SPDX-License-Identifier: Apache-2.0
"""Pin-support behaviour of the vLLM engine adapter.

Locks the fix for the former ``# TODO Phase 7`` stubs that made
``is_pin_supported()`` always return True and ``list_supported_pins()``
always return ``()``. The adapter now answers from the real
``sndr/engines/vllm/pins/`` directory (delegating to the canonical
``wiring.anchor_manifest`` per-pin resolver), so these tests assert
against the committed pin set rather than a constant.
"""
from __future__ import annotations

from sndr.engines.vllm import wiring
from sndr.engines.vllm.adapter import VllmEngine
from sndr.engines.vllm.wiring import anchor_manifest


def _adapter() -> VllmEngine:
    # The pin-support methods are filesystem-only; they never touch a
    # SndrConfig or an installed vllm, so an uninstantiated-config adapter
    # is fine. ``EngineAdapter.__init__`` only stores the config.
    return VllmEngine.__new__(VllmEngine)


def test_list_supported_pins_matches_real_pins_dir():
    """The adapter enumerates exactly the committed per-pin dirs that
    carry an ``anchors.json`` — the canonical 'supported' marker."""
    adapter = _adapter()
    listed = adapter.list_supported_pins()

    # Sorted tuple, agrees with the canonical resolver (single source of
    # truth shared with drift_check / anchor_manifest_gen).
    assert listed == anchor_manifest.list_supported_pins()
    assert isinstance(listed, tuple)
    assert listed == tuple(sorted(listed))

    # The three 0.23.1 pins ship an anchors.json today.
    assert "0.23.1_b4c80ec0f" in listed
    assert "0.23.1_3f5a1e173" in listed
    assert "0.23.1_04c2a8dea" in listed


def test_list_supported_pins_is_not_empty():
    """Regression guard: the old stub returned ``()`` unconditionally."""
    assert _adapter().list_supported_pins() != ()


def test_is_pin_supported_true_for_known_full_version():
    """A full vllm version that normalizes onto a committed pin dir is
    supported (the old stub returned True for *everything*, including
    None — this proves it now discriminates)."""
    adapter = _adapter()
    # 0.23.1rc1.dev148+gb4c80ec0f -> 0.23.1_b4c80ec0f (has anchors.json)
    assert adapter.is_pin_supported("0.23.1rc1.dev148+gb4c80ec0f") is True


def test_is_pin_supported_true_for_normalized_pin_tag():
    """The already-normalized directory name is accepted too."""
    assert _adapter().is_pin_supported("0.23.1_b4c80ec0f") is True


def test_is_pin_supported_false_for_unknown_pin():
    """Regression guard: the old stub returned True for unknown pins."""
    adapter = _adapter()
    assert adapter.is_pin_supported("9.9.9+gdeadbeef0") is False
    assert adapter.is_pin_supported("0.99.0_nonexistent") is False


def test_is_pin_supported_false_for_none():
    assert _adapter().is_pin_supported(None) is False


def test_is_pin_supported_false_for_manifest_only_legacy_pin():
    """The 0.21.1 / 0.22.1 dirs carry only a legacy ``manifest.yaml`` (no
    ``anchors.json``), so they are NOT in the supported anchor-SOT set —
    the adapter must agree with the canonical resolver and reject them."""
    adapter = _adapter()
    assert adapter.is_pin_supported("0.21.1rc0+g626fa9bba566") is False
    assert "0.21.1_626fa9bba" not in adapter.list_supported_pins()


def test_wiring_reexports_pin_resolvers():
    """The adapter delegates to the wiring resolver; that resolver must be
    importable from the package so the delegation cannot silently break."""
    assert hasattr(wiring.anchor_manifest, "is_pin_supported")
    assert hasattr(wiring.anchor_manifest, "list_supported_pins")
