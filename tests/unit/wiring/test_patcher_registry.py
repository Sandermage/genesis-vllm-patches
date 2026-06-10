# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr.engines.vllm.wiring.patcher_registry` — P2.1 Node 2.

Contract:

  1. register_text_patcher rejects non-str / empty patch_id.
  2. register_text_patcher rejects patcher missing target_file/marker/
     sub_patches attributes (duck-typed shape check).
  3. Same patch_id + same patcher = idempotent re-register (allows
     module reimport during testing).
  4. Same patch_id + different patcher = ValueError (programming error).
  5. get_registered_patcher returns None when patch_id absent.
  6. iter_registered_patchers yields snapshot (concurrent register/
     iterate doesn't trip — caller gets list copy).
  7. registered_count tracks live registrations.
  8. clear_registry wipes everything (tests only).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from sndr.engines.vllm.wiring import patcher_registry as pr


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Each test runs against a fresh empty registry."""
    pr.clear_registry()
    yield
    pr.clear_registry()


def _make_patcher(target_file: str = "/x.py", marker: str = "M",
                  sub_patches: list | None = None):
    """Mock TextPatcher-shaped object."""
    return SimpleNamespace(
        target_file=target_file,
        marker=marker,
        sub_patches=sub_patches or [],
    )


# ─── register_text_patcher input validation ───────────────────────────


class TestRegisterValidation:
    def test_rejects_non_string_id(self):
        with pytest.raises(ValueError, match="non-empty str"):
            pr.register_text_patcher(123, _make_patcher())

    def test_rejects_empty_id(self):
        with pytest.raises(ValueError, match="non-empty str"):
            pr.register_text_patcher("", _make_patcher())

    def test_rejects_none_id(self):
        with pytest.raises(ValueError, match="non-empty str"):
            pr.register_text_patcher(None, _make_patcher())

    def test_rejects_patcher_missing_target_file(self):
        bad = SimpleNamespace(marker="M", sub_patches=[])
        with pytest.raises(ValueError, match="target_file"):
            pr.register_text_patcher("PN1", bad)

    def test_rejects_patcher_missing_marker(self):
        bad = SimpleNamespace(target_file="/x.py", sub_patches=[])
        with pytest.raises(ValueError, match="marker"):
            pr.register_text_patcher("PN1", bad)

    def test_rejects_patcher_missing_sub_patches(self):
        bad = SimpleNamespace(target_file="/x.py", marker="M")
        with pytest.raises(ValueError, match="sub_patches"):
            pr.register_text_patcher("PN1", bad)


# ─── Idempotency + collision ──────────────────────────────────────────


class TestIdempotency:
    def test_same_id_same_object_is_idempotent(self):
        p = _make_patcher()
        pr.register_text_patcher("PN1", p)
        pr.register_text_patcher("PN1", p)  # no-op
        assert pr.registered_count() == 1

    def test_same_id_different_object_raises(self):
        p1 = _make_patcher(target_file="/a.py")
        p2 = _make_patcher(target_file="/b.py")
        pr.register_text_patcher("PN1", p1)
        with pytest.raises(ValueError, match="already registered"):
            pr.register_text_patcher("PN1", p2)

    def test_distinct_ids_coexist(self):
        pr.register_text_patcher("PN1", _make_patcher("/a.py"))
        pr.register_text_patcher("PN2", _make_patcher("/b.py"))
        assert pr.registered_count() == 2


# ─── Lookup ───────────────────────────────────────────────────────────


class TestLookup:
    def test_get_returns_registered_patcher(self):
        p = _make_patcher(target_file="/x.py")
        pr.register_text_patcher("PN1", p)
        assert pr.get_registered_patcher("PN1") is p

    def test_get_returns_none_when_absent(self):
        assert pr.get_registered_patcher("NEVER") is None

    def test_get_returns_none_after_clear(self):
        pr.register_text_patcher("PN1", _make_patcher())
        pr.clear_registry()
        assert pr.get_registered_patcher("PN1") is None


# ─── Iterator semantics ───────────────────────────────────────────────


class TestIteration:
    def test_yields_all_registered_pairs(self):
        pr.register_text_patcher("PN1", _make_patcher("/a.py"))
        pr.register_text_patcher("PN2", _make_patcher("/b.py"))
        pairs = list(pr.iter_registered_patchers())
        ids = {pid for pid, _ in pairs}
        assert ids == {"PN1", "PN2"}

    def test_empty_when_none_registered(self):
        assert list(pr.iter_registered_patchers()) == []

    def test_snapshot_semantics_isolates_from_concurrent_changes(self):
        """Iterator works on a list copy — modifying the registry mid-iter
        does NOT affect the iteration."""
        pr.register_text_patcher("PN1", _make_patcher("/a.py"))
        pr.register_text_patcher("PN2", _make_patcher("/b.py"))
        it = pr.iter_registered_patchers()
        # Add a third entry AFTER the iterator is grabbed
        pr.register_text_patcher("PN3", _make_patcher("/c.py"))
        seen = [pid for pid, _ in it]
        # Iterator yields the snapshot at iter() time — PN3 not included
        assert seen == ["PN1", "PN2"]


# ─── Count + clear ────────────────────────────────────────────────────


class TestCountAndClear:
    def test_count_tracks_registrations(self):
        assert pr.registered_count() == 0
        pr.register_text_patcher("PN1", _make_patcher("/a.py"))
        assert pr.registered_count() == 1
        pr.register_text_patcher("PN2", _make_patcher("/b.py"))
        assert pr.registered_count() == 2

    def test_clear_resets_to_zero(self):
        pr.register_text_patcher("PN1", _make_patcher())
        pr.register_text_patcher("PN2", _make_patcher())
        pr.clear_registry()
        assert pr.registered_count() == 0
