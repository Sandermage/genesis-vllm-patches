# SPDX-License-Identifier: Apache-2.0
"""Worker family contract tests — Theme 4 expansion (audit 2026-05-11).

Background: worker family had 3/10 patches with dedicated tests (PN52,
PN55, PN82). Family-level contract covers all 9 patches actually living
under `integrations/worker/`. The historical "10th entry" referenced in
the original audit was the predecessor of PN122 (formerly
`SPRINT26_CG_DISPATCH_TRACE`, renamed 2026-05-14), which had a
registry/filesystem mismatch: declared `family="worker"` while its
source lived under `integrations/observability/`. The 2026-05-11 audit
resolved it by changing the registry `family` to `"observability"`
(matching the actual filesystem location); PN122 is now correctly out
of scope for THIS worker contract.

Worker family characteristics:
  - High mix of PROD-active (P72 profile_run cap, PN67 thinking-budget,
    PN52 retired with byte-equiv upstream, PN82 Mamba prefill)
  - Several patches involve gpu_model_runner integration (PN52, PN82,
    PN24, PN35) — anchor-sensitive
  - PN55 wakes-up hybrid KV (hybrid model edge case)

Tests run torch-less. Verify per-patch:
  1. Module importable (skip on torch/triton)
  2. Genesis marker (const OR attr OR PID_MARKER_*)
  3. apply() or should_apply() callable
  4. env_flag from registry referenced in source (xfail synthetic legacy)
  5. No top-level torch import
  6. family field is "worker"
Family-level: all listed patches in registry + filesystem.

Coverage gap closure: worker 3/9 → 9/9 covered by contract.
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

# (module path, patch ID) — env_flag from registry source-of-truth
WORKER_PATCHES = [
    ("sndr.engines.vllm.patches.worker.p72_profile_run_cap", "P72"),
    ("sndr.engines.vllm.patches.worker.p79b_async_proposer_sync", "P79b"),
    ("sndr.engines.vllm.patches.worker.pn24_dflash_aux_layer_indexing", "PN24"),
    ("sndr.engines.vllm.patches.worker.pn33_spec_decode_warmup_k", "PN33"),
    ("sndr.engines.vllm.patches.worker.pn35_inputs_embeds_optional", "PN35"),
    ("sndr.engines.vllm._archive.pn52_prompt_logprobs_eviction", "PN52"),
    ("sndr.engines.vllm.patches.worker.pn55_wake_up_hybrid_kv", "PN55"),
    ("sndr.engines.vllm._archive.pn67_thinking_budget_inverted_bool", "PN67"),
    ("sndr.engines.vllm._archive.pn82_mamba_cudagraph_prefill_zero", "PN82"),
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


@pytest.mark.parametrize("module_path,patch_id", WORKER_PATCHES)
class TestWorkerPatchContract:
    """Family-level invariants enforced across every worker patch."""

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
        """Accept any marker convention: GENESIS_*_MARKER const,
        <PID>_MARKER_* const, or setattr _genesis_<id>_wrapped/_marker."""
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
        """env_flag from registry referenced in wiring file or any
        companion file (kernels/, etc). Xfail for synthetic
        GENESIS_LEGACY_* on legacy/retired patches."""
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
            return
        # Companion file fallback
        sndr_root = Path(mod.__file__).resolve().parents[2]
        patch_stem = patch_id.lower()
        for companion in sndr_root.rglob(f"*{patch_stem}*.py"):
            if companion == Path(mod.__file__):
                continue
            try:
                if env_flag in companion.read_text():
                    return
            except OSError:
                continue
        pytest.fail(
            f"{patch_id}: env_flag {env_flag!r} not found in wiring or "
            f"companion files matching '*{patch_stem}*.py' under {sndr_root}"
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

    def test_family_is_worker(self, module_path, patch_id):
        family = _get_registry_field(patch_id, "family")
        assert family == "worker", (
            f"{patch_id}: registry family={family!r}, expected 'worker'"
        )


class TestWorkerFamilyRegistry:
    """Family-level invariants verified once."""

    def test_all_patches_listed_in_registry(self):
        registry_path = (
            Path(__file__).resolve().parents[4]
            / "sndr" / "dispatcher" / "registry.py"
        )
        text = registry_path.read_text()
        for _module_path, patch_id in WORKER_PATCHES:
            assert f'"{patch_id}":' in text, (
                f"{patch_id}: no entry in PATCH_REGISTRY"
            )

    def test_family_count_matches_filesystem(self):
        from sndr.dispatcher import PATCH_REGISTRY
        fam_dir = (
            Path(__file__).resolve().parents[4]
            / "sndr" / "engines" / "vllm" / "patches" / "worker"
        )
        retired_dirs = [fam_dir.parent / "_retired", fam_dir.parents[1] / "_archive"]
        files = {f.stem for f in fam_dir.glob("*.py") if f.name != "__init__.py"}
        retired_files = {
                f.stem for rd in retired_dirs if rd.exists()
                for f in rd.glob("*.py") if f.name != "__init__.py"
            }
        for module_path, patch_id in WORKER_PATCHES:
            file_stem = module_path.rsplit(".", 1)[-1]
            meta = PATCH_REGISTRY.get(patch_id, {})
            if meta.get("lifecycle") == "retired":
                if not meta.get("apply_module"):
                    continue
                assert file_stem in retired_files, (
                    f"{patch_id}: retired but {file_stem}.py not in {retired_dirs}"
                )
                continue
            assert file_stem in files, (
                f"{patch_id}: expected file {file_stem}.py in {fam_dir}"
            )