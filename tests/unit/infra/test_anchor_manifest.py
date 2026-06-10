# SPDX-License-Identifier: Apache-2.0
"""Tests for Site Map (anchor offset manifest) — Nodes 1+2 of P2.1.

Coverage:
  - TestComputeAnchorMeta — single-anchor offset/length/md5 extraction
  - TestBuildFileEntry — per-file manifest fragment with multiple sub-patches
  - TestAssembleManifest — full manifest from multiple files
  - TestSchemaValidation — required fields / types / format
  - TestVerifyAgainstSource — md5 reality check
  - TestPersistence — atomic write + safe load + corruption
  - TestPinInvalidation — load_manifest_for_pins refuses on mismatch
  - TestRegistry — patcher_registry register/lookup/iter/clear
  - TestPN79Integration — uses real PN79 anchors from pristine_fixtures/

These tests are torch-less and pure-Python — run on Mac CI gate.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def pristine_chunk_py() -> str:
    """Read the committed pristine fixture for chunk.py."""
    p = (Path(__file__).resolve().parents[3] / "tests" / "legacy" / "pristine_fixtures" / "chunk.py")
    return p.read_text(encoding="utf-8")


@pytest.fixture
def pristine_gdn_py() -> str:
    p = (Path(__file__).resolve().parents[3] / "tests" / "legacy" / "pristine_fixtures" / "gdn_linear_attn.py")
    return p.read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def _clear_registry():
    """Each test starts with empty registry (test isolation)."""
    from sndr.engines.vllm.wiring.patcher_registry import clear_registry
    clear_registry()
    yield
    clear_registry()


def _make_fake_patcher(target_file: str, marker: str,
                       sub_patches: list) -> object:
    """Build a TextPatcher-shaped duck-typed object for registry tests
    without importing the real class (avoids triggering apply_all
    side-effects in test setup)."""
    from sndr.kernel.text_patch import TextPatch, TextPatcher
    return TextPatcher(
        patch_name="fake",
        target_file=target_file,
        marker=marker,
        sub_patches=[TextPatch(name=n, anchor=a, replacement=r, required=True)
                     for n, a, r in sub_patches],
    )


# ─────────────────────────────────────────────────────────────────────────
# 1. TestComputeAnchorMeta
# ─────────────────────────────────────────────────────────────────────────


class TestComputeAnchorMeta:

    def test_basic_unique_anchor(self):
        from sndr.engines.vllm.wiring.anchor_manifest import compute_anchor_meta
        src = "abc\ndef\nghi\n"
        meta = compute_anchor_meta(src, "def\n", "DEF\n")
        assert meta is not None
        assert meta["byte_offset"] == 4
        assert meta["byte_length"] == 4
        assert len(meta["anchor_md5"]) == 32
        assert len(meta["replacement_md5"]) == 32
        # Different anchor and replacement should hash differently
        assert meta["anchor_md5"] != meta["replacement_md5"]

    def test_no_replacement_omits_replacement_md5(self):
        from sndr.engines.vllm.wiring.anchor_manifest import compute_anchor_meta
        meta = compute_anchor_meta("hello world", "world")
        assert meta is not None
        assert "replacement_md5" not in meta

    def test_anchor_not_found_returns_none(self):
        from sndr.engines.vllm.wiring.anchor_manifest import compute_anchor_meta
        assert compute_anchor_meta("hello", "world") is None

    def test_anchor_ambiguous_returns_none(self):
        """Manifest entry only valid if anchor uniquely identifies a region."""
        from sndr.engines.vllm.wiring.anchor_manifest import compute_anchor_meta
        assert compute_anchor_meta("abcabc", "abc") is None  # 2 occurrences

    def test_empty_inputs_return_none(self):
        from sndr.engines.vllm.wiring.anchor_manifest import compute_anchor_meta
        # Empty src — anchor cannot be found OR str.find may return 0 for ""
        # we want None for unsafe inputs
        assert compute_anchor_meta("", "anchor") is None

    def test_non_string_inputs_return_none(self):
        from sndr.engines.vllm.wiring.anchor_manifest import compute_anchor_meta
        assert compute_anchor_meta(123, "x") is None  # type: ignore[arg-type]
        assert compute_anchor_meta("text", 42) is None  # type: ignore[arg-type]

    def test_byte_offset_matches_actual_bytes(self, pristine_chunk_py):
        """For real PN79 anchor, byte_offset must point at correct slice."""
        from sndr.engines.vllm.wiring.anchor_manifest import compute_anchor_meta
        from sndr.engines.vllm.patches.attention.gdn import pn79_inplace_ssm_state as M
        anchor = M.ANCHOR_1B_FWD_SIG_OLD
        meta = compute_anchor_meta(pristine_chunk_py, anchor, M.ANCHOR_1B_FWD_SIG_NEW)
        assert meta is not None
        # Slice using byte_offset+byte_length should equal anchor bytes
        src_bytes = pristine_chunk_py.encode("utf-8")
        actual_slice = src_bytes[
            meta["byte_offset"]:meta["byte_offset"] + meta["byte_length"]
        ]
        assert actual_slice == anchor.encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────
# 2. TestBuildFileEntry
# ─────────────────────────────────────────────────────────────────────────


class TestBuildFileEntry:

    def test_single_patch_single_anchor(self):
        from sndr.engines.vllm.wiring.anchor_manifest import PatcherManifestInput
        from sndr.engines.vllm.wiring.anchor_manifest import build_file_entry
        src = "alpha\nbravo\ncharlie\n"
        entry = build_file_entry(src, [
            PatcherManifestInput(
                patch_id="DEMO",
                rel_path="some/file.py",
                sub_patches=[("S1", "bravo\n", "BRAVO\n")],
            ),
        ])
        assert entry is not None
        assert entry["size_bytes"] == len(src.encode("utf-8"))
        assert "md5_pristine" in entry
        assert "DEMO" in entry["patches"]
        assert "S1" in entry["patches"]["DEMO"]["anchors"]

    def test_two_patches_same_file(self):
        from sndr.engines.vllm.wiring.anchor_manifest import PatcherManifestInput
        from sndr.engines.vllm.wiring.anchor_manifest import build_file_entry
        src = "a\nb\nc\n"
        entry = build_file_entry(src, [
            PatcherManifestInput("PA", "x.py",
                                 [("a1", "a\n", "A\n")]),
            PatcherManifestInput("PB", "x.py",
                                 [("b1", "b\n", "B\n")]),
        ])
        assert entry is not None
        assert set(entry["patches"].keys()) == {"PA", "PB"}

    def test_missing_anchor_skipped_silently(self, caplog):
        from sndr.engines.vllm.wiring.anchor_manifest import PatcherManifestInput
        from sndr.engines.vllm.wiring.anchor_manifest import build_file_entry
        src = "only this content"
        entry = build_file_entry(src, [
            PatcherManifestInput("PX", "x.py", [
                ("hit", "this", "THIS"),
                ("miss", "absent", "ABSENT"),
            ]),
        ])
        assert entry is not None
        assert "hit" in entry["patches"]["PX"]["anchors"]
        assert "miss" not in entry["patches"]["PX"]["anchors"]

    def test_all_anchors_missing_returns_none(self):
        from sndr.engines.vllm.wiring.anchor_manifest import PatcherManifestInput
        from sndr.engines.vllm.wiring.anchor_manifest import build_file_entry
        entry = build_file_entry("real content", [
            PatcherManifestInput("PX", "x.py",
                                 [("miss", "absent", "x")]),
        ])
        assert entry is None

    def test_empty_source_returns_none(self):
        from sndr.engines.vllm.wiring.anchor_manifest import PatcherManifestInput
        from sndr.engines.vllm.wiring.anchor_manifest import build_file_entry
        assert build_file_entry("", [
            PatcherManifestInput("PX", "x.py", [("s1", "x", "y")])
        ]) is None


# ─────────────────────────────────────────────────────────────────────────
# 3. TestAssembleManifest
# ─────────────────────────────────────────────────────────────────────────


class TestAssembleManifest:

    def test_full_assembly_two_files(self):
        from sndr.engines.vllm.wiring.anchor_manifest import PatcherManifestInput
        from sndr.engines.vllm.wiring.anchor_manifest import assemble_manifest
        from sndr.engines.vllm.wiring.anchor_manifest import MANIFEST_SCHEMA_VERSION
        manifest = assemble_manifest(
            vllm_pin="0.20.2rc1.dev9+g01d4d1ad3",
            genesis_pin="v7.72.2",
            file_to_inputs={
                "f1.py": ("hello world", [
                    PatcherManifestInput("PA", "f1.py",
                                         [("h1", "hello", "HELLO")]),
                ]),
                "f2.py": ("foo bar baz", [
                    PatcherManifestInput("PB", "f2.py",
                                         [("b1", "bar", "BAR")]),
                ]),
            },
        )
        assert manifest["manifest_version"] == MANIFEST_SCHEMA_VERSION
        assert manifest["pins"]["vllm"] == "0.20.2rc1.dev9+g01d4d1ad3"
        assert manifest["pins"]["genesis"] == "v7.72.2"
        assert set(manifest["files"].keys()) == {"f1.py", "f2.py"}

    def test_files_with_no_anchors_omitted(self):
        from sndr.engines.vllm.wiring.anchor_manifest import PatcherManifestInput
        from sndr.engines.vllm.wiring.anchor_manifest import assemble_manifest
        manifest = assemble_manifest(
            vllm_pin="x",
            genesis_pin="y",
            file_to_inputs={
                "good.py": ("hello", [
                    PatcherManifestInput("P", "good.py",
                                         [("h", "hello", "HELLO")]),
                ]),
                "bad.py": ("blah", [
                    PatcherManifestInput("P", "bad.py",
                                         [("miss", "absent", "x")]),
                ]),
            },
        )
        assert "good.py" in manifest["files"]
        assert "bad.py" not in manifest["files"]


# ─────────────────────────────────────────────────────────────────────────
# 4. TestSchemaValidation
# ─────────────────────────────────────────────────────────────────────────


class TestSchemaValidation:

    def _good_manifest(self):
        from sndr.engines.vllm.wiring.anchor_manifest import PatcherManifestInput
        from sndr.engines.vllm.wiring.anchor_manifest import assemble_manifest
        return assemble_manifest(
            vllm_pin="x", genesis_pin="y",
            file_to_inputs={
                "f.py": ("hello world", [
                    PatcherManifestInput("P", "f.py",
                                         [("a", "hello", "HELLO")]),
                ]),
            },
        )

    def test_valid_manifest_no_errors(self):
        from sndr.engines.vllm.wiring.anchor_manifest import validate_manifest_schema
        assert validate_manifest_schema(self._good_manifest()) == []

    def test_missing_top_level_key(self):
        from sndr.engines.vllm.wiring.anchor_manifest import validate_manifest_schema
        m = self._good_manifest()
        del m["pins"]
        errors = validate_manifest_schema(m)
        assert any("pins" in e for e in errors)

    def test_wrong_manifest_version(self):
        from sndr.engines.vllm.wiring.anchor_manifest import validate_manifest_schema
        m = self._good_manifest()
        m["manifest_version"] = 99
        errors = validate_manifest_schema(m)
        assert any("manifest_version 99" in e for e in errors)

    def test_non_dict_input(self):
        from sndr.engines.vllm.wiring.anchor_manifest import validate_manifest_schema
        assert validate_manifest_schema("not a dict") == [
            "manifest must be dict, got str"
        ]
        assert validate_manifest_schema(None) == [
            "manifest must be dict, got NoneType"
        ]

    def test_md5_wrong_length(self):
        from sndr.engines.vllm.wiring.anchor_manifest import validate_manifest_schema
        m = self._good_manifest()
        m["files"]["f.py"]["md5_pristine"] = "tooshort"
        errors = validate_manifest_schema(m)
        assert any("md5_pristine wrong length" in e for e in errors)

    def test_byte_offset_negative(self):
        from sndr.engines.vllm.wiring.anchor_manifest import validate_manifest_schema
        m = self._good_manifest()
        m["files"]["f.py"]["patches"]["P"]["anchors"]["a"]["byte_offset"] = -1
        errors = validate_manifest_schema(m)
        assert any("byte_offset must be >= 0" in e for e in errors)


# ─────────────────────────────────────────────────────────────────────────
# 5. TestVerifyAgainstSource
# ─────────────────────────────────────────────────────────────────────────


class TestVerifyAgainstSource:

    def test_match_returns_empty_errors(self):
        from sndr.engines.vllm.wiring.anchor_manifest import PatcherManifestInput
        from sndr.engines.vllm.wiring.anchor_manifest import assemble_manifest
        from sndr.engines.vllm.wiring.anchor_manifest import verify_manifest_against_source
        src = "hello world"
        manifest = assemble_manifest(
            vllm_pin="x", genesis_pin="y",
            file_to_inputs={"f.py": (src, [
                PatcherManifestInput("P", "f.py",
                                     [("a", "hello", "HELLO")]),
            ])},
        )
        errors = verify_manifest_against_source(manifest, lambda p: src)
        assert errors == []

    def test_md5_mismatch_reported(self):
        from sndr.engines.vllm.wiring.anchor_manifest import PatcherManifestInput
        from sndr.engines.vllm.wiring.anchor_manifest import assemble_manifest
        from sndr.engines.vllm.wiring.anchor_manifest import verify_manifest_against_source
        manifest = assemble_manifest(
            vllm_pin="x", genesis_pin="y",
            file_to_inputs={"f.py": ("hello world", [
                PatcherManifestInput("P", "f.py",
                                     [("a", "hello", "HELLO")]),
            ])},
        )
        # Loader returns DIFFERENT content
        errors = verify_manifest_against_source(manifest, lambda p: "different")
        assert len(errors) == 1
        assert "md5 mismatch" in errors[0]

    def test_source_unloadable(self):
        from sndr.engines.vllm.wiring.anchor_manifest import PatcherManifestInput
        from sndr.engines.vllm.wiring.anchor_manifest import assemble_manifest
        from sndr.engines.vllm.wiring.anchor_manifest import verify_manifest_against_source
        manifest = assemble_manifest(
            vllm_pin="x", genesis_pin="y",
            file_to_inputs={"f.py": ("hi", [
                PatcherManifestInput("P", "f.py", [("a", "hi", "HI")]),
            ])},
        )
        errors = verify_manifest_against_source(manifest, lambda p: None)
        assert any("not loadable" in e for e in errors)


# ─────────────────────────────────────────────────────────────────────────
# 6. TestPersistence (atomic write + safe load)
# ─────────────────────────────────────────────────────────────────────────


class TestPersistence:

    def _good_manifest(self):
        from sndr.engines.vllm.wiring.anchor_manifest import PatcherManifestInput
        from sndr.engines.vllm.wiring.anchor_manifest import assemble_manifest
        return assemble_manifest(
            vllm_pin="vllm-1.0", genesis_pin="v7.72.2",
            file_to_inputs={"f.py": ("hello world", [
                PatcherManifestInput("P", "f.py",
                                     [("a", "hello", "HELLO")]),
            ])},
        )

    def test_round_trip_preserves_shape(self, tmp_path):
        from sndr.engines.vllm.wiring.anchor_manifest import load_manifest
        from sndr.engines.vllm.wiring.anchor_manifest import write_manifest_atomic
        path = tmp_path / "m.json"
        m = self._good_manifest()
        write_manifest_atomic(path, m)
        loaded = load_manifest(path)
        assert loaded == m

    def test_atomic_write_no_tmp_file_after(self, tmp_path):
        from sndr.engines.vllm.wiring.anchor_manifest import write_manifest_atomic
        path = tmp_path / "m.json"
        write_manifest_atomic(path, self._good_manifest())
        # Verify .tmp file cleaned up
        assert not (tmp_path / "m.json.tmp").exists()
        assert path.exists()

    def test_overwrite_existing(self, tmp_path):
        from sndr.engines.vllm.wiring.anchor_manifest import load_manifest
        from sndr.engines.vllm.wiring.anchor_manifest import write_manifest_atomic
        path = tmp_path / "m.json"
        m1 = self._good_manifest()
        write_manifest_atomic(path, m1)
        m2 = dict(m1)
        m2["pins"] = {"vllm": "v2", "genesis": "v7.72.2"}
        write_manifest_atomic(path, m2)
        loaded = load_manifest(path)
        assert loaded["pins"]["vllm"] == "v2"

    def test_load_missing_returns_none(self, tmp_path):
        from sndr.engines.vllm.wiring.anchor_manifest import load_manifest
        assert load_manifest(tmp_path / "absent.json") is None

    def test_load_corrupted_json_returns_none(self, tmp_path):
        from sndr.engines.vllm.wiring.anchor_manifest import load_manifest
        path = tmp_path / "corrupt.json"
        path.write_text("{this is not valid json")
        assert load_manifest(path) is None

    def test_load_invalid_schema_returns_none(self, tmp_path):
        from sndr.engines.vllm.wiring.anchor_manifest import load_manifest
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"manifest_version": 99}))
        assert load_manifest(path) is None

    def test_load_non_dict_returns_none(self, tmp_path):
        from sndr.engines.vllm.wiring.anchor_manifest import load_manifest
        path = tmp_path / "list.json"
        path.write_text(json.dumps([1, 2, 3]))
        assert load_manifest(path) is None


# ─────────────────────────────────────────────────────────────────────────
# 7. TestPinInvalidation
# ─────────────────────────────────────────────────────────────────────────


class TestPinInvalidation:

    def _build(self, tmp_path):
        from sndr.engines.vllm.wiring.anchor_manifest import PatcherManifestInput
        from sndr.engines.vllm.wiring.anchor_manifest import assemble_manifest
        from sndr.engines.vllm.wiring.anchor_manifest import write_manifest_atomic
        m = assemble_manifest(
            vllm_pin="vllm-X", genesis_pin="genesis-Y",
            file_to_inputs={"f.py": ("hello", [
                PatcherManifestInput("P", "f.py",
                                     [("a", "hello", "HELLO")]),
            ])},
        )
        path = tmp_path / "m.json"
        write_manifest_atomic(path, m)
        return path

    def test_match_returns_manifest(self, tmp_path):
        from sndr.engines.vllm.wiring.anchor_manifest import load_manifest_for_pins
        path = self._build(tmp_path)
        assert load_manifest_for_pins(
            path, vllm_pin="vllm-X", genesis_pin="genesis-Y"
        ) is not None

    def test_vllm_pin_mismatch_returns_none(self, tmp_path):
        from sndr.engines.vllm.wiring.anchor_manifest import load_manifest_for_pins
        path = self._build(tmp_path)
        assert load_manifest_for_pins(
            path, vllm_pin="WRONG", genesis_pin="genesis-Y"
        ) is None

    def test_genesis_pin_mismatch_returns_none(self, tmp_path):
        from sndr.engines.vllm.wiring.anchor_manifest import load_manifest_for_pins
        path = self._build(tmp_path)
        assert load_manifest_for_pins(
            path, vllm_pin="vllm-X", genesis_pin="WRONG"
        ) is None

    def test_unspecified_pin_skips_check(self, tmp_path):
        from sndr.engines.vllm.wiring.anchor_manifest import load_manifest_for_pins
        path = self._build(tmp_path)
        # Only vllm specified, genesis None — passes
        assert load_manifest_for_pins(
            path, vllm_pin="vllm-X", genesis_pin=None
        ) is not None


# ─────────────────────────────────────────────────────────────────────────
# 8. TestRegistry
# ─────────────────────────────────────────────────────────────────────────


class TestRegistry:

    def test_register_and_lookup(self):
        from sndr.engines.vllm.wiring.patcher_registry import register_text_patcher
        from sndr.engines.vllm.wiring.patcher_registry import get_registered_patcher
        p = _make_fake_patcher("/tmp/x.py", "marker", [("s1", "old", "new")])
        register_text_patcher("PX.Sub-1", p)
        assert get_registered_patcher("PX.Sub-1") is p

    def test_lookup_missing_returns_none(self):
        from sndr.engines.vllm.wiring.patcher_registry import get_registered_patcher
        assert get_registered_patcher("not-registered") is None

    def test_iter_pairs(self):
        from sndr.engines.vllm.wiring.patcher_registry import register_text_patcher
        from sndr.engines.vllm.wiring.patcher_registry import iter_registered_patchers
        p1 = _make_fake_patcher("/x", "m1", [("s", "a", "b")])
        p2 = _make_fake_patcher("/y", "m2", [("s", "c", "d")])
        register_text_patcher("A", p1)
        register_text_patcher("B", p2)
        pairs = list(iter_registered_patchers())
        ids = [pid for pid, _ in pairs]
        assert "A" in ids and "B" in ids

    def test_count(self):
        from sndr.engines.vllm.wiring.patcher_registry import register_text_patcher
        from sndr.engines.vllm.wiring.patcher_registry import registered_count
        assert registered_count() == 0
        register_text_patcher("X", _make_fake_patcher("/x", "m", [("s", "a", "b")]))
        assert registered_count() == 1

    def test_register_same_object_idempotent(self):
        from sndr.engines.vllm.wiring.patcher_registry import register_text_patcher
        p = _make_fake_patcher("/x", "m", [("s", "a", "b")])
        register_text_patcher("X", p)
        register_text_patcher("X", p)  # same id + same object — no error

    def test_register_different_object_same_id_raises(self):
        from sndr.engines.vllm.wiring.patcher_registry import register_text_patcher
        p1 = _make_fake_patcher("/x", "m1", [("s", "a", "b")])
        p2 = _make_fake_patcher("/y", "m2", [("s", "a", "b")])
        register_text_patcher("X", p1)
        with pytest.raises(ValueError, match="already registered"):
            register_text_patcher("X", p2)

    def test_invalid_patch_id_raises(self):
        from sndr.engines.vllm.wiring.patcher_registry import register_text_patcher
        p = _make_fake_patcher("/x", "m", [("s", "a", "b")])
        with pytest.raises(ValueError, match="patch_id"):
            register_text_patcher("", p)
        with pytest.raises(ValueError, match="patch_id"):
            register_text_patcher(123, p)  # type: ignore[arg-type]

    def test_non_textpatcher_shape_raises(self):
        from sndr.engines.vllm.wiring.patcher_registry import register_text_patcher
        with pytest.raises(ValueError, match="missing required attribute"):
            register_text_patcher("X", "not a patcher")  # type: ignore[arg-type]

    def test_clear_registry(self):
        from sndr.engines.vllm.wiring.patcher_registry import register_text_patcher
        from sndr.engines.vllm.wiring.patcher_registry import registered_count
        from sndr.engines.vllm.wiring.patcher_registry import clear_registry
        register_text_patcher("X",
            _make_fake_patcher("/x", "m", [("s", "a", "b")]))
        assert registered_count() == 1
        clear_registry()
        assert registered_count() == 0


# ─────────────────────────────────────────────────────────────────────────
# 9. TestPN79Integration — real anchors against real fixtures
# ─────────────────────────────────────────────────────────────────────────


class TestPN79Integration:
    """End-to-end: build manifest for PN79's 7 chunk.py anchors against
    pristine_fixtures/chunk.py, validate schema, verify md5 against same
    fixture. This is the proof-of-concept that demonstrates manifest
    builder works on actual Genesis patches.
    """

    def test_pn79_chunk_py_full_pipeline(self, pristine_chunk_py):
        from sndr.engines.vllm.wiring.anchor_manifest import PatcherManifestInput
        from sndr.engines.vllm.wiring.anchor_manifest import assemble_manifest
        from sndr.engines.vllm.wiring.anchor_manifest import validate_manifest_schema
        from sndr.engines.vllm.wiring.anchor_manifest import verify_manifest_against_source
        from sndr.engines.vllm.patches.attention.gdn import pn79_inplace_ssm_state as M

        chunk_subs = [
            ("1B", M.ANCHOR_1B_FWD_SIG_OLD, M.ANCHOR_1B_FWD_SIG_NEW),
            ("1C", M.ANCHOR_1C_FWD_INTERNAL_OLD, M.ANCHOR_1C_FWD_INTERNAL_NEW),
            ("1D_DECORATOR", M.ANCHOR_1D_DECORATOR_OLD,
             M.ANCHOR_1D_DECORATOR_NEW),
            ("1D_FORWARD_SIG", M.ANCHOR_1D_FORWARD_SIG_OLD,
             M.ANCHOR_1D_FORWARD_SIG_NEW),
            ("1D_FORWARD_CALL", M.ANCHOR_1D_FORWARD_CALL_OLD,
             M.ANCHOR_1D_FORWARD_CALL_NEW),
            ("1E_SIG", M.ANCHOR_1E_SIG_OLD, M.ANCHOR_1E_SIG_NEW),
            ("1E_VAL", M.ANCHOR_1E_VAL_OLD, M.ANCHOR_1E_VAL_NEW),
            ("1E_APPLY_CALL", M.ANCHOR_1E_APPLY_CALL_OLD,
             M.ANCHOR_1E_APPLY_CALL_NEW),
        ]
        manifest = assemble_manifest(
            vllm_pin="0.20.2rc1.dev9+g01d4d1ad3",
            genesis_pin="v7.72.2",
            file_to_inputs={
                "model_executor/layers/fla/ops/chunk.py": (
                    pristine_chunk_py,
                    [PatcherManifestInput("PN79.Sub-1",
                        "model_executor/layers/fla/ops/chunk.py", chunk_subs)],
                ),
            },
        )
        # Schema valid
        assert validate_manifest_schema(manifest) == []
        # All 7 PN79 Sub-1 anchors recorded
        anchors = manifest["files"][
            "model_executor/layers/fla/ops/chunk.py"
        ]["patches"]["PN79.Sub-1"]["anchors"]
        assert set(anchors.keys()) == {"1B", "1C", "1D_DECORATOR",
                                       "1D_FORWARD_SIG", "1D_FORWARD_CALL",
                                       "1E_SIG", "1E_VAL", "1E_APPLY_CALL"}
        # Verify against pristine — must be clean
        errors = verify_manifest_against_source(
            manifest, lambda p: pristine_chunk_py
        )
        assert errors == []

    def test_pn79_byte_offsets_actually_point_to_anchors(
            self, pristine_chunk_py):
        """Spot-check: each manifest entry's offset+length slice equals
        the original anchor. This is the sanity check that runtime path
        will rely on."""
        from sndr.engines.vllm.wiring.anchor_manifest import PatcherManifestInput
        from sndr.engines.vllm.wiring.anchor_manifest import assemble_manifest
        from sndr.engines.vllm.patches.attention.gdn import pn79_inplace_ssm_state as M

        anchors = {
            "1B": M.ANCHOR_1B_FWD_SIG_OLD,
            "1D_FORWARD_SIG": M.ANCHOR_1D_FORWARD_SIG_OLD,
        }
        manifest = assemble_manifest(
            vllm_pin="x", genesis_pin="y",
            file_to_inputs={
                "chunk.py": (pristine_chunk_py, [
                    PatcherManifestInput("PN79.Sub-1", "chunk.py",
                        [(name, a, "REPL") for name, a in anchors.items()]),
                ]),
            },
        )
        src_bytes = pristine_chunk_py.encode("utf-8")
        recorded = manifest["files"]["chunk.py"]["patches"][
            "PN79.Sub-1"]["anchors"]
        for name, anchor in anchors.items():
            offset = recorded[name]["byte_offset"]
            length = recorded[name]["byte_length"]
            assert src_bytes[offset:offset + length] == anchor.encode("utf-8")
