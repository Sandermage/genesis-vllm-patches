# SPDX-License-Identifier: Apache-2.0
"""Tests for `wiring/file_cache.py` — P2.2 of patcher evolution plan
(2026-05-07). Persistent file mtime+size+marker cache used by Layer 0
fast-path in TextPatcher.apply().

Tests cover:
  TestCachePathResolution — env override + XDG + HOME + tmp fallback
  TestPinDetection — vllm/genesis pin extraction
  TestEmptyCache — _new_empty_cache shape + validation
  TestLoadFromDisk — absent / corrupted / pin mismatch / valid
  TestGetCacheEntry — basic lookup + miss
  TestIsMarkerCachedPresent — composite check (mtime + size + marker)
  TestRecordApplyResult — write-back + multiple markers per file + idempotent
  TestInvalidateFile — single-entry removal
  TestClearCache — full wipe
  TestPinInvalidation — wipe on pin mismatch
  TestAtomicWrite — temp file cleanup
  TestNoRaiseInvariant — no method ever raises (graceful degrade)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    """Each test gets a unique cache file in tmp_path. Prevents test
    pollution of ~/.cache/genesis/files_md5.json. Also resets in-memory
    cache state so each test starts clean.
    """
    test_cache = tmp_path / "test_files_md5.json"
    monkeypatch.setenv("GENESIS_FILE_CACHE_PATH", str(test_cache))
    monkeypatch.delenv("GENESIS_NO_PATCH_CACHE", raising=False)
    from sndr.engines.vllm.wiring import file_cache
    file_cache._reset_for_tests()
    yield test_cache
    file_cache._reset_for_tests()


# ═════════════════════════════════════════════════════════════════════════
# 1. TestCachePathResolution
# ═════════════════════════════════════════════════════════════════════════


class TestCachePathResolution:

    def test_env_override_wins(self, monkeypatch, tmp_path):
        from sndr.engines.vllm.wiring import file_cache
        monkeypatch.setenv("GENESIS_FILE_CACHE_PATH",
                           str(tmp_path / "custom.json"))
        assert file_cache._resolve_cache_path() == tmp_path / "custom.json"

    def test_xdg_cache_home_when_no_env(self, monkeypatch, tmp_path):
        from sndr.engines.vllm.wiring import file_cache
        monkeypatch.delenv("GENESIS_FILE_CACHE_PATH", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert file_cache._resolve_cache_path() == \
            tmp_path / "genesis" / "files_md5.json"

    def test_home_fallback_when_no_xdg(self, monkeypatch, tmp_path):
        from sndr.engines.vllm.wiring import file_cache
        monkeypatch.delenv("GENESIS_FILE_CACHE_PATH", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        assert file_cache._resolve_cache_path() == \
            tmp_path / ".cache" / "genesis" / "files_md5.json"


# ═════════════════════════════════════════════════════════════════════════
# 2. TestEmptyCache + Pin Detection
# ═════════════════════════════════════════════════════════════════════════


class TestEmptyCacheAndPins:

    def test_new_empty_has_required_fields(self):
        from sndr.engines.vllm.wiring.file_cache import _new_empty_cache
        c = _new_empty_cache()
        assert c["cache_version"] == 1
        assert "pins" in c
        assert "vllm" in c["pins"]
        assert "genesis" in c["pins"]
        assert c["files"] == {}

    def test_validate_cache_accepts_well_formed(self):
        from sndr.engines.vllm.wiring.file_cache import _new_empty_cache
        from sndr.engines.vllm.wiring.file_cache import _validate_cache
        assert _validate_cache(_new_empty_cache()) is True

    @pytest.mark.parametrize("bad", [
        None, "string", 42, [],
        {"cache_version": 99},  # wrong version
        {"cache_version": 1, "pins": "not-dict"},
        {"cache_version": 1, "pins": {}, "files": "not-dict"},
    ])
    def test_validate_cache_rejects_malformed(self, bad):
        from sndr.engines.vllm.wiring.file_cache import _validate_cache
        assert _validate_cache(bad) is False


# ═════════════════════════════════════════════════════════════════════════
# 3. TestLoadFromDisk
# ═════════════════════════════════════════════════════════════════════════


class TestLoadFromDisk:

    def test_absent_returns_none(self, _isolated_cache):
        from sndr.engines.vllm.wiring.file_cache import _load_cache_from_disk
        # Cache file path set by fixture but file doesn't exist
        assert _load_cache_from_disk() is None

    def test_corrupted_json_returns_none(self, _isolated_cache):
        from sndr.engines.vllm.wiring.file_cache import _load_cache_from_disk
        _isolated_cache.write_text("not valid json {")
        assert _load_cache_from_disk() is None

    def test_invalid_schema_returns_none(self, _isolated_cache):
        from sndr.engines.vllm.wiring.file_cache import _load_cache_from_disk
        _isolated_cache.write_text(json.dumps({"cache_version": 99}))
        assert _load_cache_from_disk() is None

    def test_pin_mismatch_returns_none(self, _isolated_cache):
        from sndr.engines.vllm.wiring.file_cache import _load_cache_from_disk
        _isolated_cache.write_text(json.dumps({
            "cache_version": 1,
            "pins": {"vllm": "WRONG_PIN", "genesis": "WRONG"},
            "files": {},
        }))
        # Real pins won't match dummy ones
        assert _load_cache_from_disk() is None


# ═════════════════════════════════════════════════════════════════════════
# 4. TestGetCacheEntry
# ═════════════════════════════════════════════════════════════════════════


class TestGetCacheEntry:

    def test_miss_returns_none(self):
        from sndr.engines.vllm.wiring.file_cache import get_cache_entry
        assert get_cache_entry("/nonexistent/file.py") is None

    def test_after_record_returns_entry(self, tmp_path):
        from sndr.engines.vllm.wiring.file_cache import get_cache_entry
        from sndr.engines.vllm.wiring.file_cache import record_apply_result
        target = tmp_path / "f.py"
        target.write_text("hello")
        record_apply_result(str(target), "MARKER",
                            post_apply_content="hello")
        entry = get_cache_entry(str(target))
        assert entry is not None
        assert entry["size_bytes"] == 5
        assert "MARKER" in entry["markers"]


# ═════════════════════════════════════════════════════════════════════════
# 5. TestIsMarkerCachedPresent
# ═════════════════════════════════════════════════════════════════════════


class TestIsMarkerCachedPresent:

    def test_returns_true_when_all_match(self, tmp_path):
        from sndr.engines.vllm.wiring.file_cache import is_marker_cached_present
        from sndr.engines.vllm.wiring.file_cache import record_apply_result
        target = tmp_path / "f.py"
        target.write_text("body")
        record_apply_result(str(target), "M", post_apply_content="body")
        assert is_marker_cached_present(str(target), "M") is True

    def test_false_when_no_entry(self, tmp_path):
        from sndr.engines.vllm.wiring.file_cache import is_marker_cached_present
        target = tmp_path / "f.py"
        target.write_text("body")
        assert is_marker_cached_present(str(target), "M") is False

    def test_false_when_marker_unknown(self, tmp_path):
        from sndr.engines.vllm.wiring.file_cache import is_marker_cached_present
        from sndr.engines.vllm.wiring.file_cache import record_apply_result
        target = tmp_path / "f.py"
        target.write_text("body")
        record_apply_result(str(target), "M1", post_apply_content="body")
        assert is_marker_cached_present(str(target), "M2") is False

    def test_false_when_mtime_changed(self, tmp_path):
        from sndr.engines.vllm.wiring.file_cache import is_marker_cached_present
        from sndr.engines.vllm.wiring.file_cache import record_apply_result
        target = tmp_path / "f.py"
        target.write_text("v1")
        record_apply_result(str(target), "M", post_apply_content="v1")
        # Modify file (changes mtime)
        time.sleep(0.01)
        target.write_text("v2")
        assert is_marker_cached_present(str(target), "M") is False

    def test_false_when_size_changed(self, tmp_path):
        from sndr.engines.vllm.wiring.file_cache import is_marker_cached_present
        from sndr.engines.vllm.wiring.file_cache import record_apply_result
        target = tmp_path / "f.py"
        target.write_text("short")
        record_apply_result(str(target), "M", post_apply_content="short")
        # Manually corrupt cache to simulate size mismatch (mtime stays
        # if we just overwrite the cache JSON with a wrong size_bytes)
        from sndr.engines.vllm.wiring import file_cache
        cache = file_cache._ensure_loaded()
        cache["files"][str(target)]["size_bytes"] = 999
        assert is_marker_cached_present(str(target), "M") is False

    def test_false_when_file_disappeared(self, tmp_path):
        from sndr.engines.vllm.wiring.file_cache import is_marker_cached_present
        from sndr.engines.vllm.wiring.file_cache import record_apply_result
        target = tmp_path / "gone.py"
        target.write_text("x")
        record_apply_result(str(target), "M", post_apply_content="x")
        target.unlink()
        assert is_marker_cached_present(str(target), "M") is False


# ═════════════════════════════════════════════════════════════════════════
# 6. TestRecordApplyResult
# ═════════════════════════════════════════════════════════════════════════


class TestRecordApplyResult:

    def test_creates_entry_with_md5(self, tmp_path):
        from sndr.engines.vllm.wiring.file_cache import get_cache_entry
        from sndr.engines.vllm.wiring.file_cache import record_apply_result
        target = tmp_path / "f.py"
        target.write_text("payload")
        record_apply_result(str(target), "M", post_apply_content="payload")
        entry = get_cache_entry(str(target))
        assert entry is not None
        assert len(entry["md5_post_apply"]) == 32

    def test_multiple_markers_accumulate(self, tmp_path):
        from sndr.engines.vllm.wiring.file_cache import get_cache_entry
        from sndr.engines.vllm.wiring.file_cache import record_apply_result
        target = tmp_path / "f.py"
        target.write_text("x")
        record_apply_result(str(target), "M1", post_apply_content="x")
        record_apply_result(str(target), "M2", post_apply_content="x")
        record_apply_result(str(target), "M3", post_apply_content="x")
        markers = get_cache_entry(str(target))["markers"]
        assert "M1" in markers
        assert "M2" in markers
        assert "M3" in markers

    def test_duplicate_marker_not_added_twice(self, tmp_path):
        from sndr.engines.vllm.wiring.file_cache import get_cache_entry
        from sndr.engines.vllm.wiring.file_cache import record_apply_result
        target = tmp_path / "f.py"
        target.write_text("x")
        record_apply_result(str(target), "M", post_apply_content="x")
        record_apply_result(str(target), "M", post_apply_content="x")
        markers = get_cache_entry(str(target))["markers"]
        assert markers.count("M") == 1

    def test_disk_persistence(self, tmp_path, _isolated_cache):
        """After record_apply_result, restart-equivalent (re-load from
        disk) sees the entry."""
        from sndr.engines.vllm.wiring import file_cache
        target = tmp_path / "f.py"
        target.write_text("hi")
        file_cache.record_apply_result(str(target), "M",
                                       post_apply_content="hi")
        # boot-opt §4.1 (2026-06-17): record_apply_result now defers disk
        # persistence to flush_file_cache() (the orchestrator calls it once
        # at end-of-boot). Flush here to exercise the restart-equivalent path.
        file_cache.flush_file_cache()
        # Simulate restart — reset in-memory state, force reload
        file_cache._reset_for_tests()
        entry = file_cache.get_cache_entry(str(target))
        assert entry is not None, "cache should persist to disk"
        assert "M" in entry["markers"]

    def test_no_raise_on_missing_target(self, tmp_path):
        from sndr.engines.vllm.wiring.file_cache import record_apply_result
        # Target doesn't exist — must not raise
        record_apply_result(str(tmp_path / "nonexistent.py"), "M")


# ═════════════════════════════════════════════════════════════════════════
# 7. TestInvalidate + Clear
# ═════════════════════════════════════════════════════════════════════════


class TestInvalidate:

    def test_invalidate_removes_entry(self, tmp_path):
        from sndr.engines.vllm.wiring.file_cache import get_cache_entry
        from sndr.engines.vllm.wiring.file_cache import invalidate_file
        from sndr.engines.vllm.wiring.file_cache import record_apply_result
        target = tmp_path / "f.py"
        target.write_text("x")
        record_apply_result(str(target), "M", post_apply_content="x")
        assert get_cache_entry(str(target)) is not None
        invalidate_file(str(target))
        assert get_cache_entry(str(target)) is None

    def test_invalidate_unknown_path_no_raise(self):
        from sndr.engines.vllm.wiring.file_cache import invalidate_file
        invalidate_file("/totally/fake/path.py")  # no exception expected

    def test_clear_wipes_all(self, tmp_path):
        from sndr.engines.vllm.wiring.file_cache import clear_cache
        from sndr.engines.vllm.wiring.file_cache import get_cache_entry
        from sndr.engines.vllm.wiring.file_cache import record_apply_result
        for n in range(3):
            t = tmp_path / f"f{n}.py"
            t.write_text(str(n))
            record_apply_result(str(t), f"M{n}", post_apply_content=str(n))
        clear_cache()
        for n in range(3):
            assert get_cache_entry(str(tmp_path / f"f{n}.py")) is None


# ═════════════════════════════════════════════════════════════════════════
# 8. TestNoRaiseInvariant
# ═════════════════════════════════════════════════════════════════════════


class TestNoRaiseInvariant:
    """Cache module's contract: NEVER raise to caller. apply() is hot
    path — even bizarre filesystem state must produce graceful False/None."""

    def test_is_marker_cached_present_no_raise(self):
        from sndr.engines.vllm.wiring.file_cache import is_marker_cached_present
        # Various malformed inputs — none should raise
        assert is_marker_cached_present("", "M") is False
        assert is_marker_cached_present("/dev/null", "M") is False
        assert is_marker_cached_present("/proc/self/mem", "M") is False

    def test_record_apply_result_no_raise(self):
        from sndr.engines.vllm.wiring.file_cache import record_apply_result
        # Bad inputs — never raise
        record_apply_result("", "M")
        record_apply_result("/dev/null", "M")

    def test_get_entry_no_raise(self):
        from sndr.engines.vllm.wiring.file_cache import get_cache_entry
        assert get_cache_entry("/dev/null") is None


# ═════════════════════════════════════════════════════════════════════════
# 9. TestAtomicWrite
# ═════════════════════════════════════════════════════════════════════════


class TestAtomicWrite:

    def test_no_tmp_after_save(self, tmp_path, _isolated_cache):
        from sndr.engines.vllm.wiring import file_cache
        target = tmp_path / "f.py"
        target.write_text("x")
        file_cache.record_apply_result(str(target), "M",
                                       post_apply_content="x")
        file_cache.flush_file_cache()  # §4.1: persist is deferred to flush
        # Verify .tmp file cleaned up
        cache_path = _isolated_cache
        assert cache_path.exists()
        assert not cache_path.with_suffix(cache_path.suffix + ".tmp").exists()

    def test_overwrites_atomically(self, tmp_path, _isolated_cache):
        from sndr.engines.vllm.wiring import file_cache
        target = tmp_path / "f.py"
        target.write_text("v1")
        file_cache.record_apply_result(str(target), "M",
                                       post_apply_content="v1")
        # Make sure file is valid JSON, even after second write
        time.sleep(0.01)
        target.write_text("v2")
        file_cache.record_apply_result(str(target), "M",
                                       post_apply_content="v2")
        file_cache.flush_file_cache()  # §4.1: persist is deferred to flush
        with open(_isolated_cache) as f:
            payload = json.load(f)  # raises if invalid
        assert payload["cache_version"] == 1
