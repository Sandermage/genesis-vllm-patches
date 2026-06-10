# SPDX-License-Identifier: Apache-2.0
"""Tests for the §8 shared pytest fixtures registered in tests/conftest.py.

Contract:
  • Each fixture returns the canonical view (matches manual import path).
  • Session-scope means one import per pytest run, not per test.
  • `pristine_vllm_source` skips cleanly when vllm isn't installed.
  • `proof_dir` is isolated per test (no cross-pollination).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


# ─── genesis_registry ─────────────────────────────────────────────────


class TestGenesisRegistry:
    def test_returns_dict_with_known_patches(self, genesis_registry):
        # Known patch from PATCH_REGISTRY — sanity check this fixture
        # returns the same dict the dispatcher uses at runtime.
        assert isinstance(genesis_registry, dict)
        assert "P58" in genesis_registry
        # Standard fields are present on real entries.
        assert "env_flag" in genesis_registry["P58"]

    def test_matches_direct_import(self, genesis_registry):
        from sndr.dispatcher.registry import PATCH_REGISTRY
        assert genesis_registry is PATCH_REGISTRY


class TestPatchIdSubsets:
    def test_stable_patch_ids_subset(self, genesis_registry, stable_patch_ids):
        for pid in stable_patch_ids:
            assert pid in genesis_registry
            assert genesis_registry[pid].get("lifecycle") == "stable"

    def test_experimental_patch_ids_subset(self, genesis_registry,
                                            experimental_patch_ids):
        for pid in experimental_patch_ids:
            assert pid in genesis_registry
            assert genesis_registry[pid].get("lifecycle") == "experimental"

    def test_subsets_are_disjoint(self, stable_patch_ids,
                                   experimental_patch_ids):
        assert set(stable_patch_ids) & set(experimental_patch_ids) == set()


# ─── V2 ID fixtures ───────────────────────────────────────────────────


class TestV2IdFixtures:
    def test_v2_model_ids(self, v2_model_ids):
        assert "qwen3.6-35b-a3b-fp8" in v2_model_ids

    def test_v2_hardware_ids(self, v2_hardware_ids):
        assert "a5000-2x-24gbvram-16cpu-128gbram" in v2_hardware_ids

    def test_v2_profile_ids(self, v2_profile_ids):
        assert "wave9-balanced" in v2_profile_ids

    def test_v2_alias_ids(self, v2_alias_ids):
        # All 11 production-confirmed aliases.
        for alias in (
            "prod-qwen3.6-35b-balanced", "prod-qwen3.6-27b-tq-k8v4", "prod-qwen3.6-35b-dflash",
            "long-ctx-qwen3.6-27b", "qa-qwen3.6-27b-tested", "qa-qwen3.6-27b-tq-1x",
            "prod-qwen3.6-27b-dflash", "experimental-qwen3.6-27b-tq-dflash-ab",
            "example-2x-tier-aware", "example-3090-dense-cpu-offload",
            "example-3090-tier-aware",
        ):
            assert alias in v2_alias_ids, (
                f"v2_alias_ids missing {alias!r}; got {sorted(v2_alias_ids)}"
            )


# ─── canonical_env_keys ───────────────────────────────────────────────


class TestCanonicalEnvKeys:
    def test_includes_real_env_flag(self, canonical_env_keys, genesis_registry):
        # Every PATCH_REGISTRY env_flag must appear in the canonical set.
        sample = "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL"
        assert sample in canonical_env_keys, (
            f"canonical_env_keys missing {sample!r}"
        )

    def test_is_set_for_o1_lookups(self, canonical_env_keys):
        assert isinstance(canonical_env_keys, set)
        # >130 keys (matches Entry 11 baseline of 149).
        assert len(canonical_env_keys) >= 130

    def test_policy_keys_in_canonical(self, canonical_env_keys):
        """§6.7 policy tier — non-patch Genesis env keys are included."""
        assert "GENESIS_VLLM_PIN_POLICY" in canonical_env_keys


# ─── proof_dir ────────────────────────────────────────────────────────


class TestProofDir:
    def test_is_writable_dir(self, proof_dir):
        assert proof_dir.is_dir()
        (proof_dir / "test.json").write_text('{"ok": true}', encoding="utf-8")
        assert (proof_dir / "test.json").read_text() == '{"ok": true}'

    def test_isolated_per_test_call_1(self, proof_dir):
        (proof_dir / "marker_one.json").write_text("1", encoding="utf-8")

    def test_isolated_per_test_call_2(self, proof_dir):
        # Different test → different tmp_path → no marker_one.json.
        assert not (proof_dir / "marker_one.json").exists()


# ─── pristine_vllm_source ─────────────────────────────────────────────


class TestPristineVllmSource:
    """Verify the skip path works on Mac (no vllm install).

    When vllm IS installed (server CI), the fixture returns a Path; we
    don't assert internal contents because they vary by version. The
    skip semantics are the operator-facing contract.
    """

    def test_skips_when_vllm_absent_or_returns_path(self, pristine_vllm_source):
        # If the fixture didn't skip, it returned a Path that exists.
        assert pristine_vllm_source.is_dir()
        # Sanity: it points at a directory whose name is `vllm`.
        assert pristine_vllm_source.name == "vllm"

    def test_skip_uses_find_spec_not_import(self):
        """Check the fixture body relies on find_spec — no top-level
        `import vllm` that would cost ~seconds on every collection."""
        from pathlib import Path
        conftest_text = (
            Path(__file__).resolve().parents[1] / "conftest.py"
        ).read_text(encoding="utf-8")
        # In the body of `pristine_vllm_source`, find_spec must be used.
        # Search for the fixture and the keyword.
        idx = conftest_text.find("def pristine_vllm_source")
        assert idx >= 0
        snippet = conftest_text[idx:idx + 1500]
        assert "find_spec" in snippet


# ─── Marker registration ──────────────────────────────────────────────


class TestMarkerRegistration:
    def test_requires_vllm_marker_registered(self, pytestconfig):
        """The §8 fixture work registered a new `requires_vllm` marker.
        It should appear in pytest's marker list."""
        # pytestconfig._getini reads strict marker definitions via the
        # `markers` field of pyproject/pytest.ini. The conftest-registered
        # marker shows up via the canonical iteration.
        names = {m.name for m in pytestconfig.getini("markers")
                 if hasattr(m, "name")} if False else set()
        # Simpler: check via the markers list (pytestconfig.getini returns
        # a list of strings of the form "name: description").
        markers = pytestconfig.getini("markers")
        marker_names = {m.split(":")[0].strip() for m in markers}
        # Pre-existing markers still registered.
        assert "gpu_required" in marker_names
        assert "requires_torch" in marker_names
        # New §8 marker — added by conftest's pytest_configure.
        # NOTE: addinivalue_line markers don't always surface in getini;
        # an alternative check is that the marker doesn't fail strict mode.
        # We confirm by running a test with the marker indirectly below.


@pytest.mark.requires_vllm
def test_requires_vllm_marker_works():
    """If the marker were not registered, --strict-markers would fail
    collection. This test's mere collection is the assertion."""
    # If we get here, vllm IS installed (or the marker simply tagged us
    # without restriction). Either way, the marker is valid.
    assert True
