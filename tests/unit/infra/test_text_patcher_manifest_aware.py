# SPDX-License-Identifier: Apache-2.0
"""Tests for TextPatcher manifest-aware fast-path — P2.1 Phase 3.

Critical invariant: manifest path output ≡ legacy path output for any
pristine source. Both write same bytes to file.

Coverage:
  TestPatchIdField — backwards compat, patch_id default None
  TestRelPathDerivation — vllm path stripping for various shapes
  TestManifestCache — load once, sentinel on failure
  TestManifestFastPath — apply via manifest fully works
  TestManifestFallback — each of 7 abstain gates → legacy
  TestEquivalenceWithLegacy — manifest vs legacy byte-identical
  TestNoPatchCacheEnvDisables — GENESIS_NO_PATCH_CACHE=1 forces legacy
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


@pytest.fixture(autouse=True)
def _clear_manifest_cache():
    """Each test starts with empty manifest cache."""
    from sndr.kernel.text_patch import _reset_manifest_cache_for_tests
    _reset_manifest_cache_for_tests()
    yield
    _reset_manifest_cache_for_tests()


@pytest.fixture(autouse=True)
def _clear_no_patch_cache_env(monkeypatch):
    """Each test starts with GENESIS_NO_PATCH_CACHE unset."""
    monkeypatch.delenv("GENESIS_NO_PATCH_CACHE", raising=False)


# ═════════════════════════════════════════════════════════════════════════
# 1. TestPatchIdField — backwards-compat default
# ═════════════════════════════════════════════════════════════════════════


class TestPatchIdField:

    def test_default_none(self):
        from sndr.kernel.text_patch import TextPatch, TextPatcher
        p = TextPatcher(
            patch_name="test", target_file="/tmp/x", marker="m",
            sub_patches=[TextPatch(name="s", anchor="a", replacement="b",
                                   required=True)],
        )
        assert p.patch_id is None

    def test_explicit_patch_id(self):
        from sndr.kernel.text_patch import TextPatch, TextPatcher
        p = TextPatcher(
            patch_name="test", target_file="/tmp/x", marker="m",
            sub_patches=[TextPatch(name="s", anchor="a", replacement="b",
                                   required=True)],
            patch_id="PN79.Sub-1",
        )
        assert p.patch_id == "PN79.Sub-1"


# ═════════════════════════════════════════════════════════════════════════
# 2. TestRelPathDerivation
# ═════════════════════════════════════════════════════════════════════════


class TestRelPathDerivation:

    def test_canonical_install_path(self):
        from sndr.kernel.text_patch import _derive_rel_path_from_target
        rel = _derive_rel_path_from_target(
            "/usr/local/lib/python3.12/dist-packages/vllm/"
            "model_executor/layers/fla/ops/chunk.py"
        )
        assert rel == "model_executor/layers/fla/ops/chunk.py"

    def test_genesis_repo_path_lasts_vllm(self):
        """In Genesis repo there are TWO `vllm` segments — outer (repo
        clone dir) and inner (Python package overlay). We want the
        LAST one, which means the path under the overlay package.

        PR38 cleanup (2026-05-08): pristine fixtures live under
        `vllm/sndr_core/tests/...` after the `_genesis/` shim removal.
        Before they lived at `vllm/_genesis/tests/pristine_fixtures/`.
        """
        from sndr.kernel.text_patch import _derive_rel_path_from_target
        rel = _derive_rel_path_from_target(
            "/home/dev/genesis-vllm-patches/vllm/sndr_core/tests/"
            "pristine_fixtures/chunk.py"
        )
        # Last `vllm` is the repo's overlay package — after it the path
        # is sndr_core/tests/...
        assert rel == "sndr_core/tests/pristine_fixtures/chunk.py"

    def test_no_vllm_segment_returns_none(self):
        from sndr.kernel.text_patch import _derive_rel_path_from_target
        assert _derive_rel_path_from_target("/tmp/random/path.py") is None

    def test_empty_path_returns_none(self):
        from sndr.kernel.text_patch import _derive_rel_path_from_target
        assert _derive_rel_path_from_target("") is None

    def test_just_vllm_no_suffix(self):
        from sndr.kernel.text_patch import _derive_rel_path_from_target
        # `/usr/.../vllm` with nothing after — None (degenerate)
        assert _derive_rel_path_from_target("/path/to/vllm") is None


# ═════════════════════════════════════════════════════════════════════════
# 3. TestManifestCache
# ═════════════════════════════════════════════════════════════════════════


class TestManifestCache:

    def test_first_load_then_cached(self, monkeypatch, tmp_path):
        """Loader called once; second call returns cached dict."""
        from sndr.engines.vllm.wiring import text_patch as tp_module

        # Monkey-patch load_manifest_for_pins to count calls
        call_count = {"n": 0}
        sample_manifest = {
            "manifest_version": 1, "generated_at": "x",
            "pins": {"vllm": "x", "genesis": "y"}, "files": {},
        }

        def fake_load(path, vllm_pin=None, genesis_pin=None):
            call_count["n"] += 1
            return sample_manifest

        monkeypatch.setattr(
            "sndr.engines.vllm.wiring.anchor_manifest.load_manifest_for_pins",
            fake_load,
        )

        m1 = tp_module._cached_load_manifest()
        m2 = tp_module._cached_load_manifest()
        m3 = tp_module._cached_load_manifest()
        assert m1 is m2 is m3 is sample_manifest
        assert call_count["n"] == 1, f"expected 1 load, got {call_count['n']}"

    def test_load_failure_caches_invalid_sentinel(self, monkeypatch):
        """If first load returns None, subsequent calls return None
        without re-attempting load (avoids retry storm)."""
        from sndr.engines.vllm.wiring import text_patch as tp_module

        call_count = {"n": 0}

        def fake_load(path, vllm_pin=None, genesis_pin=None):
            call_count["n"] += 1
            return None

        monkeypatch.setattr(
            "sndr.engines.vllm.wiring.anchor_manifest.load_manifest_for_pins",
            fake_load,
        )

        assert tp_module._cached_load_manifest() is None
        assert tp_module._cached_load_manifest() is None
        assert tp_module._cached_load_manifest() is None
        assert call_count["n"] == 1, "load should NOT be retried after failure"


# ═════════════════════════════════════════════════════════════════════════
# 4. TestManifestFastPath — successful manifest-driven apply
# ═════════════════════════════════════════════════════════════════════════


def _setup_manifest_for_test(tmp_path: Path, monkeypatch,
                             pristine_src: str, patch_id: str,
                             sub_patches: list[tuple[str, str, str]],
                             rel_path: str = "x.py"):
    """Build a manifest covering pristine_src with given sub_patches and
    monkey-patch the loader to return it."""
    from sndr.engines.vllm.wiring.anchor_manifest import PatcherManifestInput
    from sndr.engines.vllm.wiring.anchor_manifest import assemble_manifest

    manifest = assemble_manifest(
        vllm_pin="test-vllm-pin",
        genesis_pin="test-genesis-pin",
        file_to_inputs={
            rel_path: (pristine_src, [
                PatcherManifestInput(
                    patch_id=patch_id, rel_path=rel_path,
                    sub_patches=sub_patches,
                ),
            ]),
        },
    )

    def fake_load(path, vllm_pin=None, genesis_pin=None):
        return manifest

    monkeypatch.setattr(
        "sndr.engines.vllm.wiring.anchor_manifest.load_manifest_for_pins",
        fake_load,
    )
    return manifest


class TestManifestFastPath:

    def test_apply_via_manifest_full_success(self, tmp_path, monkeypatch):
        """Build manifest covering all anchors, apply patcher, verify
        file modified correctly + IDEMPOTENT on second call."""
        from sndr.kernel.text_patch import (
            TextPatch, TextPatcher, TextPatchResult,
        )
        pristine = "alpha\nbravo\ncharlie\ndelta\n"
        rel_path = "vllm/some/file.py"  # important: must end up as "some/file.py"

        _setup_manifest_for_test(
            tmp_path, monkeypatch, pristine, "TEST.Sub-1",
            sub_patches=[
                ("s1", "alpha\n", "ALPHA\n"),
                ("s2", "charlie\n", "CHARLIE\n"),
            ],
            rel_path="some/file.py",
        )

        # Write pristine to disk under a path containing `vllm`
        target = tmp_path / "vllm" / "some" / "file.py"
        target.parent.mkdir(parents=True)
        target.write_text(pristine)

        patcher = TextPatcher(
            patch_name="test patch",
            target_file=str(target),
            marker="TEST_MARKER",
            sub_patches=[
                TextPatch(name="s1", anchor="alpha\n", replacement="ALPHA\n",
                          required=True),
                TextPatch(name="s2", anchor="charlie\n",
                          replacement="CHARLIE\n", required=True),
            ],
            patch_id="TEST.Sub-1",
        )
        result, failure = patcher.apply()
        assert result == TextPatchResult.APPLIED
        assert failure is None

        body = target.read_text()
        assert "ALPHA" in body
        assert "CHARLIE" in body
        assert "TEST_MARKER" in body  # marker prepended

        # Re-apply — IDEMPOTENT
        result2, _ = patcher.apply()
        assert result2 == TextPatchResult.IDEMPOTENT


# ═════════════════════════════════════════════════════════════════════════
# 5. TestManifestFallback — each abstain gate
# ═════════════════════════════════════════════════════════════════════════


class TestManifestFallback:
    """Each gate failure should fall through to legacy Layer 5 path,
    NOT crash. Legacy path must produce the same output."""

    def _make_patcher(self, target, patch_id=None,
                      sub_patches=None) -> object:
        from sndr.kernel.text_patch import TextPatch, TextPatcher
        sp = sub_patches or [
            TextPatch(name="s", anchor="alpha", replacement="ALPHA",
                      required=True),
        ]
        return TextPatcher(
            patch_name="test", target_file=str(target), marker="MARK",
            sub_patches=sp, patch_id=patch_id,
        )

    def test_no_patch_id_falls_back(self, tmp_path):
        """patch_id None → manifest path skipped, legacy Layer 5 used."""
        from sndr.kernel.text_patch import TextPatchResult
        target = tmp_path / "f.py"
        target.write_text("alpha")
        patcher = self._make_patcher(target, patch_id=None)
        result, _ = patcher.apply()
        assert result == TextPatchResult.APPLIED
        # Legacy path applied — file modified
        assert "ALPHA" in target.read_text()

    def test_no_manifest_loaded_falls_back(self, tmp_path, monkeypatch):
        """Manifest absent (loader returns None) → legacy Layer 5."""
        from sndr.kernel.text_patch import TextPatchResult

        monkeypatch.setattr(
            "sndr.engines.vllm.wiring.anchor_manifest.load_manifest_for_pins",
            lambda *a, **kw: None,
        )
        target = tmp_path / "f.py"
        target.write_text("alpha")
        patcher = self._make_patcher(target, patch_id="TEST.Sub-1")
        result, _ = patcher.apply()
        assert result == TextPatchResult.APPLIED
        assert "ALPHA" in target.read_text()

    def test_md5_mismatch_falls_back(self, tmp_path, monkeypatch):
        """File on disk doesn't match manifest pristine — fall through.
        Verify: Layer 5 still APPLIES (anchor present even if file changed)."""
        from sndr.kernel.text_patch import TextPatchResult

        # Manifest built for "alpha\n", but file on disk is "MODIFIED alpha\n"
        _setup_manifest_for_test(
            tmp_path, monkeypatch, "alpha\n", "TEST.Sub-1",
            sub_patches=[("s", "alpha\n", "ALPHA\n")],
            rel_path="x.py",
        )
        target = tmp_path / "vllm" / "x.py"
        target.parent.mkdir()
        target.write_text("MODIFIED alpha\n")  # ← modified vs manifest
        patcher = self._make_patcher(
            target, patch_id="TEST.Sub-1",
            sub_patches=[__import__("sndr.kernel.text_patch",
                fromlist=["TextPatch"]).TextPatch(
                name="s", anchor="alpha\n", replacement="ALPHA\n",
                required=True)],
        )
        result, _ = patcher.apply()
        # Still APPLIED via legacy (anchor "alpha\n" exists in file)
        assert result == TextPatchResult.APPLIED
        body = target.read_text()
        assert "ALPHA" in body
        assert "MODIFIED" in body  # legacy preserved the rest

    def test_no_patch_cache_env_falls_back(self, tmp_path, monkeypatch):
        """GENESIS_NO_PATCH_CACHE=1 → manifest gate 1 fail → legacy used."""
        from sndr.kernel.text_patch import TextPatchResult

        # Manifest IS available, but env disables it
        _setup_manifest_for_test(
            tmp_path, monkeypatch, "alpha\n", "TEST.Sub-1",
            sub_patches=[("s", "alpha\n", "ALPHA\n")],
            rel_path="x.py",
        )
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")

        target = tmp_path / "vllm" / "x.py"
        target.parent.mkdir()
        target.write_text("alpha\n")
        patcher = self._make_patcher(target, patch_id="TEST.Sub-1")
        from sndr.kernel.text_patch import TextPatch
        patcher.sub_patches = [
            TextPatch(name="s", anchor="alpha\n", replacement="ALPHA\n",
                      required=True),
        ]
        result, _ = patcher.apply()
        assert result == TextPatchResult.APPLIED
        assert "ALPHA" in target.read_text()


# ═════════════════════════════════════════════════════════════════════════
# 6. TestEquivalenceWithLegacy — byte-identical output
# ═════════════════════════════════════════════════════════════════════════


class TestEquivalenceWithLegacy:
    """The CRITICAL test: manifest path and legacy path must produce
    byte-identical output for the same pristine source.

    If this fails, manifest path has a bug — refuse to ship until fixed.
    """

    def _apply_to_copy(self, pristine: str, patcher_factory,
                       tmp_path: Path) -> str:
        """Write pristine to a fresh tmp file, run patcher_factory(target)
        which returns a TextPatcher, apply, return resulting file content."""
        target = tmp_path / "f.py"
        target.write_text(pristine)
        patcher = patcher_factory(target)
        result, _ = patcher.apply()
        from sndr.kernel.text_patch import TextPatchResult
        assert result == TextPatchResult.APPLIED, f"apply failed: {result}"
        return target.read_text()

    def test_pn79_chunk_py_manifest_equals_legacy(
            self, tmp_path, monkeypatch):
        """Apply PN79 Sub-1 to pristine chunk.py via BOTH paths,
        compare byte-for-byte.
        """
        from sndr.kernel.text_patch import (
            TextPatch, TextPatcher, _reset_manifest_cache_for_tests,
        )
        from sndr.engines.vllm.patches.attention.gdn import pn79_inplace_ssm_state as M

        # Load real pristine fixture
        pristine = (
            Path(__file__).resolve().parents[3] / "tests" / "legacy" / "pristine_fixtures" / "chunk.py"
        ).read_text(encoding="utf-8")

        sub_patches = [
            TextPatch(name="1C", anchor=M.ANCHOR_1C_FWD_INTERNAL_OLD,
                      replacement=M.ANCHOR_1C_FWD_INTERNAL_NEW, required=True),
            TextPatch(name="1B", anchor=M.ANCHOR_1B_FWD_SIG_OLD,
                      replacement=M.ANCHOR_1B_FWD_SIG_NEW, required=True),
            TextPatch(name="1E_VAL", anchor=M.ANCHOR_1E_VAL_OLD,
                      replacement=M.ANCHOR_1E_VAL_NEW, required=True),
        ]

        # Path 1: legacy. Force fallback via env.
        def factory_legacy(target):
            return TextPatcher(
                patch_name="PN79 Sub-1 (legacy)", target_file=str(target),
                marker="GENESIS_PN79_TEST", sub_patches=sub_patches,
                patch_id=None,  # no manifest path
            )

        # Path 2: manifest. Set up cache and use patch_id.
        def factory_manifest_setup():
            _setup_manifest_for_test(
                tmp_path, monkeypatch, pristine, "TEST.Sub-1",
                sub_patches=[
                    ("1C", M.ANCHOR_1C_FWD_INTERNAL_OLD,
                     M.ANCHOR_1C_FWD_INTERNAL_NEW),
                    ("1B", M.ANCHOR_1B_FWD_SIG_OLD, M.ANCHOR_1B_FWD_SIG_NEW),
                    ("1E_VAL", M.ANCHOR_1E_VAL_OLD, M.ANCHOR_1E_VAL_NEW),
                ],
                rel_path="chunk.py",
            )

        def factory_manifest(target):
            return TextPatcher(
                patch_name="PN79 Sub-1 (manifest)", target_file=str(target),
                marker="GENESIS_PN79_TEST", sub_patches=sub_patches,
                patch_id="TEST.Sub-1",
            )

        # Apply legacy
        legacy_dir = tmp_path / "legacy"
        legacy_dir.mkdir()
        target_legacy = legacy_dir / "f.py"
        target_legacy.write_text(pristine)
        result_legacy = factory_legacy(target_legacy).apply()
        from sndr.kernel.text_patch import TextPatchResult
        assert result_legacy[0] == TextPatchResult.APPLIED
        legacy_output = target_legacy.read_text()

        # Apply manifest. NB: manifest's load_manifest_for_pins was
        # mocked in factory_manifest_setup() — but we called
        # _reset_manifest_cache_for_tests via fixture autouse already.
        _reset_manifest_cache_for_tests()
        # Need target path to contain "vllm" so _derive_rel_path works
        manifest_dir = tmp_path / "vllm"
        manifest_dir.mkdir()
        target_manifest = manifest_dir / "chunk.py"
        target_manifest.write_text(pristine)
        factory_manifest_setup()
        result_manifest = factory_manifest(target_manifest).apply()
        assert result_manifest[0] == TextPatchResult.APPLIED
        manifest_output = target_manifest.read_text()

        assert manifest_output == legacy_output, (
            f"manifest path output ≠ legacy output. "
            f"legacy len={len(legacy_output)}, "
            f"manifest len={len(manifest_output)}"
        )


# ═════════════════════════════════════════════════════════════════════════
# 7. TestNoPatchCacheEnvDisables — operator escape hatch enforcement
# ═════════════════════════════════════════════════════════════════════════


class TestNoPatchCacheEnvDisables:
    """Even when manifest is perfectly available, GENESIS_NO_PATCH_CACHE=1
    must force legacy path. Operator-controlled escape hatch.
    """

    def test_env_set_forces_legacy(self, tmp_path, monkeypatch):
        from sndr.kernel.text_patch import (
            TextPatch, TextPatcher, TextPatchResult,
        )

        # Set up perfect manifest
        _setup_manifest_for_test(
            tmp_path, monkeypatch, "alpha\n", "TEST.Sub-1",
            sub_patches=[("s", "alpha\n", "ALPHA\n")],
            rel_path="x.py",
        )

        target = tmp_path / "vllm" / "x.py"
        target.parent.mkdir()
        target.write_text("alpha\n")

        patcher = TextPatcher(
            patch_name="test", target_file=str(target),
            marker="MARK",
            sub_patches=[TextPatch(name="s", anchor="alpha\n",
                                   replacement="ALPHA\n", required=True)],
            patch_id="TEST.Sub-1",
        )

        # WITHOUT env — manifest would fire
        monkeypatch.delenv("GENESIS_NO_PATCH_CACHE", raising=False)
        # Quick sanity: try_apply_via_manifest returns non-None
        ans = patcher._try_apply_via_manifest("alpha\n")
        assert ans is not None  # manifest path would activate

        # WITH env=1 — manifest path must abstain
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        ans2 = patcher._try_apply_via_manifest("alpha\n")
        assert ans2 is None  # gate 1 fail → legacy
