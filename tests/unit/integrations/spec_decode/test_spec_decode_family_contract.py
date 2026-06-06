# SPDX-License-Identifier: Apache-2.0
"""Spec_decode family contract tests — Theme 4 expansion (audit 2026-05-11).

Background: spec_decode family had 6/15 patches with dedicated tests
(P71, P77, P82, P94, PN72, PN90). Family-level contract covers all 15
parameterized patches. Largest single-family expansion this iteration.

Spec_decode family characteristics:
  - Heavy registry mix: experimental + research + retired (PN9, P94)
  - PN40 has 2 files on disk (omnibus + workload_classifier_hook) —
    we map registry PN40 to pn40_dflash_omnibus.py (main file)
  - Several patches are PROD-active (P82, PN90, P70, P94*) — contract
    must accept retired patches gracefully

These tests run torch-less. Verify:
  1. Module importable (skip on torch/triton missing)
  2. Genesis marker (const OR attr pattern)
  3. apply() or should_apply() callable
  4. env_flag from registry referenced in source (xfail synthetic legacy)
  5. No top-level torch import
  6. family field is "spec_decode"
  7. Registry has entries + filesystem files match (drift detector)

Coverage gap closure: spec_decode 6/15 → 15/15 covered by contract.
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

# (module path, patch ID). PN40's main file is pn40_dflash_omnibus
# (pn40_workload_classifier_hook is a sub-D component of the same patch).
SPEC_DECODE_PATCHES = [
    ("sndr.engines.vllm.patches.spec_decode.p70_auto_strict_ngram", "P70"),
    ("sndr.engines.vllm.patches.spec_decode.p71_block_verify", "P71"),
    ("sndr.engines.vllm.patches.spec_decode.p75_suffix_decoding_enable", "P75"),
    ("sndr.engines.vllm.patches.spec_decode.p77_adaptive_ngram_k", "P77"),
    ("sndr.engines.vllm.patches.spec_decode.p82_sglang_acceptance_threshold", "P82"),
    ("sndr.engines.vllm.patches.spec_decode.p86_ngram_batch_propose_linear", "P86"),
    ("sndr.engines.vllm._archive.p94_spec_decode_zero_alloc", "P94"),
    ("sndr.engines.vllm._archive.pn9_independent_drafter_attn_backend", "PN9"),
    ("sndr.engines.vllm.patches.spec_decode.pn21_dflash_swa_support", "PN21"),
    ("sndr.engines.vllm.patches.spec_decode.pn22_local_argmax_tp", "PN22"),
    ("sndr.engines.vllm.patches.spec_decode.pn23_dflash_combine_hidden_dtype", "PN23"),
    ("sndr.engines.vllm.patches.spec_decode.pn38_dflash_quant_drafter", "PN38"),
    ("sndr.engines.vllm.patches.spec_decode.pn40_dflash_omnibus", "PN40"),
    ("sndr.engines.vllm.patches.spec_decode.pn72_frequency_ngram_drafter", "PN72"),
    ("sndr.engines.vllm.patches.spec_decode.pn90_probabilistic_draft_rejection", "PN90"),
    # Phase 3 bucket 3 (2026-05-21): drafter-routing patches relocated from gemma4/.
    # PIN.R-G4_05-RETIRE.1 (2026-05-24): G4_05 retired — superseded by vllm#39930; moved to _retired/.
    ("sndr.engines.vllm._archive.g4_05_dflash_backend_autoselect", "G4_05"),
    ("sndr.engines.vllm.patches.spec_decode.g4_71_drafter_native_attn_backend", "G4_71"),
    ("sndr.engines.vllm.patches.spec_decode.g4_71b_drafter_sliding_triton", "G4_71B"),
    ("sndr.engines.vllm.patches.spec_decode.g4_72_drafter_native_kv_cache_spec", "G4_72"),
    ("sndr.engines.vllm.patches.spec_decode.g4_73_drafter_profile_skip", "G4_73"),
    ("sndr.engines.vllm.patches.spec_decode.g4_74_drafter_hnd_layout", "G4_74"),
    ("sndr.engines.vllm.patches.spec_decode.g4_75_drafter_head512_triton", "G4_75"),
    ("sndr.engines.vllm.patches.spec_decode.g4_76_disable_drafter_kv_sharing", "G4_76"),
    # Retired in Phase 3 bucket 3 (superseded by P1.8 A2 declarative kv_sharing).
    ("sndr.engines.vllm._archive.g4_78_drafter_target_kv_bridge", "G4_78"),
]


def _get_registry_field(patch_id: str, field: str) -> str | None:
    registry_path = (
        Path(__file__).resolve().parents[4]
        / "sndr" / "dispatcher" / "registry.py"
    )
    text = registry_path.read_text()
    m = re.search(rf'"{patch_id}":\s*\{{(.*?)^    \}}', text, flags=re.M | re.S)
    if not m:
        return None
    body = m.group(1)
    f_match = re.search(rf'"{field}":\s*"([^"]+)"', body)
    return f_match.group(1) if f_match else None


@pytest.mark.parametrize("module_path,patch_id", SPEC_DECODE_PATCHES)
class TestSpecDecodePatchContract:
    """Family-level invariants enforced across every spec_decode patch."""

    def test_module_importable(self, module_path, patch_id):
        if module_path in sys.modules:
            del sys.modules[module_path]
        try:
            importlib.import_module(module_path)
        except ImportError as e:
            if "torch" in str(e) or "triton" in str(e):
                pytest.skip(f"{patch_id} requires torch/triton: {e}")
            raise

    def test_genesis_marker_exists(self, module_path, patch_id):
        """Accept any of three marker conventions:
        - `GENESIS_*_MARKER` constant (text-patch convention)
        - `<PID>_MARKER_*` constant (PN90-style)
        - setattr-based `_genesis_<id>_wrapped` / `_genesis_<id>_marker`
          (runtime monkey-patch idempotency guard)
        """
        mod = importlib.import_module(module_path)
        src_text = Path(mod.__file__).read_text()
        has_const_marker = any(
            a for a in dir(mod) if "MARKER" in a.upper()
            and (a.startswith("GENESIS_") or a.startswith(patch_id.upper())
                 or "_MARKER" in a)
        )
        has_attr_marker = bool(re.search(
            r"_genesis_\w+_(wrapped|marker)\b|_GENESIS_\w+_MARKER_ATTR\b",
            src_text, flags=re.IGNORECASE,
        ))
        assert has_const_marker or has_attr_marker, (
            f"{patch_id}: no Genesis marker found in {module_path}"
        )

    def test_apply_function_exists(self, module_path, patch_id):
        mod = importlib.import_module(module_path)
        has_apply = hasattr(mod, "apply") and callable(mod.apply)
        has_should_apply = (
            hasattr(mod, "should_apply") and callable(mod.should_apply)
        )
        assert has_apply or has_should_apply, (
            f"{patch_id}: no apply() or should_apply() in {module_path}"
        )

    def test_env_flag_documented(self, module_path, patch_id):
        """Patch source mentions its env_flag from registry. Xfail for
        synthetic GENESIS_LEGACY_* flags. Accept env_flag mentioned in
        companion files (kernels/, etc) — many patches split wiring +
        kernel, env read happens in kernel file (e.g. PN40)."""
        env_flag = _get_registry_field(patch_id, "env_flag")
        assert env_flag, f"{patch_id}: env_flag not found in registry"
        lifecycle = _get_registry_field(patch_id, "lifecycle")
        mod = importlib.import_module(module_path)
        src_text = Path(mod.__file__).read_text()
        if env_flag in src_text:
            return  # operator-grep finds it, all good
        is_synthetic_legacy_flag = (
            lifecycle in ("legacy", "retired")
            and env_flag.startswith("GENESIS_LEGACY_")
        )
        if is_synthetic_legacy_flag:
            pytest.xfail(
                f"{patch_id}: synthetic legacy env_flag {env_flag!r} "
                f"(lifecycle={lifecycle}) — STILL not in source/companion "
                f"(operator-grep gap; tech debt: add doc comment OR refactor)"
            )
        if env_flag in src_text:
            return  # primary case: in wiring file
        # Companion case: check kernels/ + other sndr_core dirs.
        sndr_root = Path(mod.__file__).resolve().parents[2]
        # patch_id stem heuristic — find files matching the patch id
        patch_stem = patch_id.lower()
        for companion in sndr_root.rglob(f"*{patch_stem}*.py"):
            if companion == Path(mod.__file__):
                continue
            try:
                if env_flag in companion.read_text():
                    return  # found in companion (e.g. kernels/)
            except OSError:
                continue
        pytest.fail(
            f"{patch_id}: env_flag {env_flag!r} not found in wiring file "
            f"OR companion files matching '*{patch_stem}*.py' under "
            f"{sndr_root}. Add to docstring (operator-grep continuity) "
            f"OR self-gate via vllm_envs."
        )

    def test_module_torch_less_import_safety(self, module_path, patch_id):
        mod = importlib.import_module(module_path)
        src_text = Path(mod.__file__).read_text()
        top_level_torch = re.findall(
            r"^(?:import torch|from torch)", src_text, flags=re.M
        )
        if top_level_torch:
            for line in src_text.splitlines():
                if line.startswith(("import torch", "from torch")):
                    pytest.fail(
                        f"{patch_id}: top-level torch import in {module_path}"
                    )

    def test_family_is_spec_decode(self, module_path, patch_id):
        family = _get_registry_field(patch_id, "family")
        assert family == "spec_decode", (
            f"{patch_id}: registry family={family!r}, expected 'spec_decode'"
        )


class TestSpecDecodeFamilyRegistry:
    """Family-level invariants verified once."""

    def test_all_patches_listed_in_registry(self):
        registry_path = (
            Path(__file__).resolve().parents[4]
            / "sndr" / "dispatcher" / "registry.py"
        )
        text = registry_path.read_text()
        for _module_path, patch_id in SPEC_DECODE_PATCHES:
            assert f'"{patch_id}":' in text, (
                f"{patch_id}: no entry in PATCH_REGISTRY"
            )

    def test_family_count_matches_filesystem(self):
        from vllm.sndr_core.dispatcher import PATCH_REGISTRY
        fam_dir = (
            Path(__file__).resolve().parents[4]
            / "vllm" / "sndr_core" / "integrations" / "spec_decode"
        )
        retired_dir = fam_dir.parent / "_retired"
        files = {f.stem for f in fam_dir.glob("*.py") if f.name != "__init__.py"}
        retired_files = (
            {f.stem for f in retired_dir.glob("*.py") if f.name != "__init__.py"}
            if retired_dir.exists() else set()
        )
        for module_path, patch_id in SPEC_DECODE_PATCHES:
            file_stem = module_path.rsplit(".", 1)[-1]
            meta = PATCH_REGISTRY.get(patch_id, {})
            if meta.get("lifecycle") == "retired":
                if not meta.get("apply_module"):
                    continue
                assert file_stem in retired_files, (
                    f"{patch_id}: retired but {file_stem}.py not in {retired_dir}"
                )
                continue
            assert file_stem in files, (
                f"{patch_id}: expected file {file_stem}.py in {fam_dir}"
            )