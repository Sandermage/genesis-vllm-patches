# SPDX-License-Identifier: Apache-2.0
"""SGLang engine adapter — proves the multi-engine EngineAdapter abstraction
holds for a real, non-vllm engine.

These tests run with or without sglang installed: the adapter detects a live
install when present and degrades gracefully (None / empty / typed error)
when absent, so they assert the contract either way.
"""
from __future__ import annotations

import pytest

from sndr.config import SndrConfig
from sndr.engines import get_engine, list_engines
from sndr.engines.base import EngineAdapter
from sndr.engines.sglang import SglangEngine
from sndr.exceptions import EngineNotInstalledError


def _adapter() -> SglangEngine:
    return SglangEngine(SndrConfig.from_env())


def _sglang_installed() -> bool:
    try:
        import sglang  # noqa: F401
        return True
    except ImportError:
        return False


def test_sglang_registered_and_subclasses_adapter():
    assert "sglang" in list_engines()
    assert get_engine("sglang") is SglangEngine
    assert issubclass(SglangEngine, EngineAdapter)


def test_name():
    assert _adapter().name == "sglang"


def test_instantiable_no_abstract_methods_left():
    # Would raise TypeError if any abstractmethod were unimplemented.
    assert isinstance(_adapter(), EngineAdapter)


def test_version_and_root_track_install_state():
    a = _adapter()
    if _sglang_installed():
        assert a.install_root() is not None
        assert isinstance(a.detect_version(), str)
    else:
        assert a.install_root() is None
        with pytest.raises(EngineNotInstalledError):
            a.detect_version()


def test_resolve_file_none_without_root():
    a = _adapter()
    if a.install_root() is None:
        assert a.resolve_file("anything/here.py") is None


def test_no_pins_or_patches_ship_yet():
    a = _adapter()
    assert a.list_supported_pins() == ()
    assert a.list_patches() == []


def test_is_pin_supported_rejects_empty_and_unknown():
    a = _adapter()
    assert a.is_pin_supported(None) is False
    assert a.is_pin_supported("") is False
    assert a.is_pin_supported("0.4.6_nonexistent") is False


def test_runtime_introspection_is_none_until_first_pin():
    a = _adapter()
    assert a.get_runtime_config() is None
    assert a.get_model_profile() is None


@pytest.mark.parametrize("version,expected", [
    ("0.4.6rc1.dev10+gabcdef1234", "0.4.6_abcdef123"),
    ("0.4.6+gdeadbeef99", "0.4.6_deadbeef9"),
    ("1.2.3.dev5+g0123456789ab", "1.2.3_012345678"),
    ("0.4.6", "0.4.6"),            # no sha → unchanged
    ("weird-version", "weird-version"),
])
def test_normalize_pin(version, expected):
    assert _adapter()._normalize_pin(version) == expected
