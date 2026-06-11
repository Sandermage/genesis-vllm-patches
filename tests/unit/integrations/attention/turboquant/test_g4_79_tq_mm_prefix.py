# SPDX-License-Identifier: Apache-2.0
"""G4_79 — TQ backend supports_mm_prefix injection (Gemma 4 MM unblock).

Torch-less unit tests: the wiring injects a ``supports_mm_prefix``
classmethod returning True onto a backend class, idempotently, and
keeps a revert handle. The real target
(``vllm.v1.attention.backends.turboquant_attn.TurboQuantAttentionBackend``)
is import-gated at apply time; tests exercise the pure injection helper
against a stand-in class with the upstream-default behavior.
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture()
def mod():
    return importlib.import_module(
        "sndr.engines.vllm.patches.attention.turboquant."
        "g4_79_tq_mm_prefix_support"
    )


def _fake_backend_cls():
    class _Base:
        @classmethod
        def supports_mm_prefix(cls) -> bool:
            return False

    class FakeTurboQuantBackend(_Base):
        pass

    return FakeTurboQuantBackend


class TestInjection:
    def test_injects_true(self, mod):
        cls = _fake_backend_cls()
        assert cls.supports_mm_prefix() is False
        changed = mod.inject_mm_prefix_support(cls)
        assert changed is True
        assert cls.supports_mm_prefix() is True

    def test_idempotent(self, mod):
        cls = _fake_backend_cls()
        assert mod.inject_mm_prefix_support(cls) is True
        assert mod.inject_mm_prefix_support(cls) is False  # second = no-op
        assert cls.supports_mm_prefix() is True

    def test_marker_attr_set(self, mod):
        cls = _fake_backend_cls()
        mod.inject_mm_prefix_support(cls)
        assert getattr(cls, mod.GENESIS_G4_79_MARKER_ATTR) is True

    def test_revert(self, mod):
        cls = _fake_backend_cls()
        mod.inject_mm_prefix_support(cls)
        assert cls.supports_mm_prefix() is True
        mod.revert_mm_prefix_support(cls)
        assert cls.supports_mm_prefix() is False
        assert not getattr(cls, mod.GENESIS_G4_79_MARKER_ATTR, False)

    def test_is_applied_reflects_state(self, mod):
        cls = _fake_backend_cls()
        assert mod._cls_is_patched(cls) is False
        mod.inject_mm_prefix_support(cls)
        assert mod._cls_is_patched(cls) is True
