# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr.engines.vllm.wiring.file_cache` — P2.2 Layer 0 fast-path.

Contract:

  1. _resolve_cache_path honors GENESIS_FILE_CACHE_PATH > XDG > HOME > /tmp.
  2. _validate_cache rejects non-dict, schema_version mismatch, missing
     pins/files keys.
  3. _check_pins_match returns True only when cached pins match detected.
  4. _new_empty_cache returns canonical shape.
  5. is_marker_cached_present requires: cache hit + mtime_ns + size_bytes +
     marker in markers list.
  6. record_apply_result writes entry with markers list; appends if exists.
  7. invalidate_file removes single entry.
  8. clear_cache wipes everything (in-memory).
  9. _save_cache_atomic + _load_cache_from_disk round-trip.
"""
from __future__ import annotations

import json

import pytest

from sndr.engines.vllm.wiring import file_cache as fc


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Each test runs against an isolated cache file under tmp_path."""
    cache_path = tmp_path / "files_md5.json"
    monkeypatch.setenv("GENESIS_FILE_CACHE_PATH", str(cache_path))
    fc._reset_for_tests()
    yield
    fc._reset_for_tests()


# ─── Cache path resolution ────────────────────────────────────────────


class TestResolveCachePath:
    def test_env_override_wins(self, tmp_path, monkeypatch):
        custom = tmp_path / "custom_cache.json"
        monkeypatch.setenv("GENESIS_FILE_CACHE_PATH", str(custom))
        assert fc._resolve_cache_path() == custom

    def test_xdg_cache_home_fallback(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GENESIS_FILE_CACHE_PATH", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        result = fc._resolve_cache_path()
        assert str(result).startswith(str(tmp_path))
        assert result.name == "files_md5.json"

    def test_home_fallback(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GENESIS_FILE_CACHE_PATH", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        result = fc._resolve_cache_path()
        assert ".cache/genesis" in str(result)

    def test_tmp_fallback_when_home_unwritable(self, monkeypatch):
        monkeypatch.delenv("GENESIS_FILE_CACHE_PATH", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setenv("HOME", "/nonexistent-dir-xyz")
        result = fc._resolve_cache_path()
        assert str(result).startswith("/tmp/")


# ─── Cache schema validation ──────────────────────────────────────────


class TestValidateCache:
    def test_rejects_non_dict(self):
        assert not fc._validate_cache("not a dict")
        assert not fc._validate_cache(None)
        assert not fc._validate_cache(42)

    def test_rejects_wrong_schema_version(self):
        assert not fc._validate_cache({
            "cache_version": 999, "pins": {}, "files": {},
        })

    def test_rejects_missing_pins(self):
        assert not fc._validate_cache({
            "cache_version": fc.CACHE_SCHEMA_VERSION, "files": {},
        })

    def test_rejects_non_dict_files(self):
        assert not fc._validate_cache({
            "cache_version": fc.CACHE_SCHEMA_VERSION,
            "pins": {}, "files": [],
        })

    def test_accepts_canonical_shape(self):
        assert fc._validate_cache({
            "cache_version": fc.CACHE_SCHEMA_VERSION,
            "pins": {"vllm": "", "genesis": ""},
            "updated_at": "2026-05-30T00:00:00Z",
            "files": {},
        })


# ─── New empty cache shape ────────────────────────────────────────────


class TestNewEmptyCache:
    def test_has_required_keys(self):
        c = fc._new_empty_cache()
        assert c["cache_version"] == fc.CACHE_SCHEMA_VERSION
        assert "pins" in c
        assert "files" in c
        assert "updated_at" in c
        assert c["files"] == {}

    def test_pins_dict_shape(self):
        c = fc._new_empty_cache()
        assert "vllm" in c["pins"]
        assert "genesis" in c["pins"]


# ─── Layer 0 fast-path check ─────────────────────────────────────────


class TestIsMarkerCachedPresent:
    def test_missing_file_returns_false(self, tmp_path):
        assert not fc.is_marker_cached_present(
            str(tmp_path / "nope.py"), "MARKER",
        )

    def test_no_cache_entry_returns_false(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("content")
        assert not fc.is_marker_cached_present(str(f), "MARKER")

    def test_matching_entry_returns_true(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("# Genesis MARKER applied\n")
        fc.record_apply_result(str(f), "MARKER")
        assert fc.is_marker_cached_present(str(f), "MARKER")

    def test_wrong_marker_returns_false(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("# Genesis MARKER applied\n")
        fc.record_apply_result(str(f), "MARKER")
        assert not fc.is_marker_cached_present(str(f), "DIFFERENT")

    def test_mtime_change_returns_false(self, tmp_path):
        """Layer 0 invariant: any mtime change invalidates the cache hit."""
        import os
        f = tmp_path / "x.py"
        f.write_text("# Genesis MARKER\n")
        fc.record_apply_result(str(f), "MARKER")
        # Touch file with new mtime
        new_mtime = os.stat(f).st_mtime + 100
        os.utime(f, (new_mtime, new_mtime))
        assert not fc.is_marker_cached_present(str(f), "MARKER")


# ─── record_apply_result ──────────────────────────────────────────────


class TestRecordApplyResult:
    def test_creates_entry_on_first_record(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("content\n")
        fc.record_apply_result(str(f), "MARKER")
        entry = fc.get_cache_entry(str(f))
        assert entry is not None
        assert entry["markers"] == ["MARKER"]
        assert entry["size_bytes"] == len("content\n")

    def test_appends_marker_on_subsequent_record(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("content\n")
        fc.record_apply_result(str(f), "MARKER1")
        fc.record_apply_result(str(f), "MARKER2")
        entry = fc.get_cache_entry(str(f))
        assert sorted(entry["markers"]) == ["MARKER1", "MARKER2"]

    def test_no_duplicate_marker_appended(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("content\n")
        fc.record_apply_result(str(f), "MARKER")
        fc.record_apply_result(str(f), "MARKER")
        entry = fc.get_cache_entry(str(f))
        assert entry["markers"] == ["MARKER"]

    def test_missing_target_file_no_raise(self):
        # Should not raise even if file doesn't exist
        fc.record_apply_result("/nonexistent.py", "MARKER")

    def test_post_apply_content_is_md5d(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("on_disk\n")
        # Pass explicit post-apply content — md5 uses THAT, not disk
        fc.record_apply_result(str(f), "MARKER",
                                post_apply_content="from-arg\n")
        entry = fc.get_cache_entry(str(f))
        import hashlib
        expected = hashlib.md5(b"from-arg\n").hexdigest()
        assert entry["md5_post_apply"] == expected


# ─── invalidate + clear ──────────────────────────────────────────────


class TestInvalidate:
    def test_invalidate_removes_entry(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("content\n")
        fc.record_apply_result(str(f), "MARKER")
        assert fc.get_cache_entry(str(f)) is not None
        fc.invalidate_file(str(f))
        assert fc.get_cache_entry(str(f)) is None

    def test_invalidate_unknown_no_raise(self, tmp_path):
        # Should not raise
        fc.invalidate_file(str(tmp_path / "never_cached.py"))

    def test_clear_cache_resets_all(self, tmp_path):
        f1 = tmp_path / "a.py"
        f1.write_text("a\n")
        f2 = tmp_path / "b.py"
        f2.write_text("b\n")
        fc.record_apply_result(str(f1), "M1")
        fc.record_apply_result(str(f2), "M2")
        fc.clear_cache()
        assert fc.get_cache_entry(str(f1)) is None
        assert fc.get_cache_entry(str(f2)) is None


# ─── Atomic save + load round-trip ───────────────────────────────────


class TestSaveLoadRoundTrip:
    def test_save_and_load(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("content\n")
        fc.record_apply_result(str(f), "MARKER")
        # record_apply_result mutates only the in-memory cache since ce6c174d
        # (the O(N^2)->O(N) boot fix moved disk persistence to a single
        # end-of-boot flush). Flush before resetting so the round-trip reads
        # the persisted entry back from disk.
        fc.flush_file_cache()
        # Reset in-memory cache → next access loads from disk
        fc._reset_for_tests()
        entry = fc.get_cache_entry(str(f))
        assert entry is not None
        assert "MARKER" in entry["markers"]
