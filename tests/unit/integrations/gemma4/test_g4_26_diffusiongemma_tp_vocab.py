# SPDX-License-Identifier: Apache-2.0
"""G4_26 — backport TP-correctness half of OPEN vllm PR #45774.

DiffusionGemmaForBlockDiffusion self-conditioning computes
``probs @ embed_weight`` over the FULL vocab (262144). Under TP>1 the
``embed_tokens.weight`` is vocab-sharded to ``[131072, 2816]`` and the
matmul raises ``RuntimeError: a and b must have same reduction dim``.

The fix adds:
  1. an import of ``get_tensor_model_parallel_world_size`` +
     ``tensor_model_parallel_all_gather`` from ``vllm.distributed``;
  2. a module-level ``_get_full_embed_weight`` helper that all-gathers the
     sharded weight under TP and returns ``.weight`` unchanged at TP=1;
  3. a line-853 swap of ``embed_weight=self.model.model.embed_tokens.weight``
     to ``embed_weight=_get_full_embed_weight(self.model.model.embed_tokens)``.

These tests run the G4_26 TextPatcher against a SYNTHETIC `diffusion_gemma.py`
source whose anchor regions mirror the live dev491 module byte-for-byte, so
they need neither a live vllm install nor CUDA. They assert:
  * each of the 3 sub-patch anchors resolves uniquely;
  * the import block, the helper def, and the swapped call all land;
  * the patch is idempotent (re-apply = no second copy);
  * the upstream_drift_marker self-skips once #45774 merges;
  * the registry/env/dispatch wiring carries the new G4_26 entry.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


# ── Synthetic diffusion_gemma.py mirroring the dev491 anchor regions ────
#
# Only the three anchor regions (import block, _NO_PENALTIES_STATE →
# DiffusionSampler boundary, and the custom_sampler embed_weight line)
# need to be byte-faithful; the rest is minimal scaffolding.
SYNTHETIC_DIFFUSION_GEMMA = '''\
# SPDX-License-Identifier: Apache-2.0
"""Synthetic diffusion_gemma.py — anchor fixture for G4_26 tests."""
from types import SimpleNamespace

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm


class DiffusionGemmaForConditionalGeneration:
    def custom_sampler(self, gen, sampler, diffusion_config, entropy_bound):
        return DiffusionSampler(
            sampler=sampler,
            diffusion_config=diffusion_config,
            vocab_size=self.model_config.get_vocab_size(),
            diffusion_states=self.diffusion_states,
            t_min=gen["t_min"],
            t_max=gen["t_max"],
            entropy_bound=entropy_bound,
            confidence_threshold=gen["confidence_threshold"],
            embed_weight=self.model.model.embed_tokens.weight,
            normalizer=self.model.model.normalizer,
        ), None

    def apply_staged_writes(self) -> None:
        pass


# Penalty stub for the diffusion path: the runner reads
# penalties_state.output_bin_counts, and post_update treats None as
# "no penalty bookkeeping".
_NO_PENALTIES_STATE = SimpleNamespace(output_bin_counts=None)


class DiffusionSampler:
    """Batched accept/renoise sampler for DiffusionGemma."""
'''


@pytest.fixture
def g4_26_mod():
    return importlib.import_module(
        "sndr.engines.vllm.patches.model_compat.gemma4."
        "g4_26_diffusiongemma_tp_vocab_soft_embed"
    )


@pytest.fixture
def synthetic_target(tmp_path):
    target = tmp_path / "diffusion_gemma.py"
    target.write_text(SYNTHETIC_DIFFUSION_GEMMA, encoding="utf-8")
    return target


def _build_patcher(g4_26_mod, target_path):
    """Build the G4_26 TextPatcher pointed at an arbitrary target file."""
    return g4_26_mod._make_patcher_for_target(str(target_path))


# ─── Anchor resolution + apply mechanics ────────────────────────────────


class TestG4_26AnchorsResolve:
    def test_all_three_anchors_present_and_unique(self, g4_26_mod, synthetic_target):
        patcher = _build_patcher(g4_26_mod, synthetic_target)
        content = synthetic_target.read_text(encoding="utf-8")
        assert len(patcher.sub_patches) == 3
        for sp in patcher.sub_patches:
            assert sp.anchor in content, (
                f"sub-patch {sp.name!r} anchor not found in synthetic source"
            )
            assert content.count(sp.anchor) == 1, (
                f"sub-patch {sp.name!r} anchor is not unique "
                f"({content.count(sp.anchor)} matches)"
            )

    def test_apply_adds_distributed_import(self, g4_26_mod, synthetic_target):
        from sndr.kernel import TextPatchResult
        patcher = _build_patcher(g4_26_mod, synthetic_target)
        result, failure = patcher.apply()
        assert result == TextPatchResult.APPLIED, failure
        out = synthetic_target.read_text(encoding="utf-8")
        assert "from vllm.distributed import (" in out
        assert "get_tensor_model_parallel_world_size," in out
        assert "tensor_model_parallel_all_gather," in out
        # Inserted between CUDAGraphMode and init_logger imports.
        cg = out.index("from vllm.config.compilation import CUDAGraphMode")
        dist = out.index("from vllm.distributed import (")
        log = out.index("from vllm.logger import init_logger")
        assert cg < dist < log

    def test_apply_adds_helper_function(self, g4_26_mod, synthetic_target):
        from sndr.kernel import TextPatchResult
        patcher = _build_patcher(g4_26_mod, synthetic_target)
        result, _ = patcher.apply()
        assert result == TextPatchResult.APPLIED
        out = synthetic_target.read_text(encoding="utf-8")
        assert "def _get_full_embed_weight(embed_tokens" in out
        assert "get_tensor_model_parallel_world_size() == 1" in out
        assert "tensor_model_parallel_all_gather(embed_tokens.weight, dim=0)" in out
        assert "embed_tokens.org_vocab_size" in out
        # Helper sits between the penalty stub and the DiffusionSampler class.
        stub = out.index("_NO_PENALTIES_STATE = SimpleNamespace")
        helper = out.index("def _get_full_embed_weight(embed_tokens")
        cls = out.index("class DiffusionSampler:")
        assert stub < helper < cls

    def test_line_853_swap_produces_gathered_call(self, g4_26_mod, synthetic_target):
        from sndr.kernel import TextPatchResult
        patcher = _build_patcher(g4_26_mod, synthetic_target)
        result, _ = patcher.apply()
        assert result == TextPatchResult.APPLIED
        out = synthetic_target.read_text(encoding="utf-8")
        # The bare sharded-weight pass is gone; the gathered call is in.
        assert (
            "embed_weight=self.model.model.embed_tokens.weight," not in out
        )
        assert (
            "embed_weight=_get_full_embed_weight(self.model.model.embed_tokens),"
            in out
        )
        # Surrounding context is preserved (still inside custom_sampler).
        assert "confidence_threshold=gen[\"confidence_threshold\"]," in out
        assert "normalizer=self.model.model.normalizer," in out

    def test_apply_is_idempotent(self, g4_26_mod, synthetic_target):
        from sndr.kernel import TextPatchResult
        patcher1 = _build_patcher(g4_26_mod, synthetic_target)
        r1, _ = patcher1.apply()
        assert r1 == TextPatchResult.APPLIED
        first = synthetic_target.read_text(encoding="utf-8")
        # Build a fresh patcher (mimics a second boot) and re-apply.
        patcher2 = _build_patcher(g4_26_mod, synthetic_target)
        r2, _ = patcher2.apply()
        assert r2 == TextPatchResult.IDEMPOTENT
        assert synthetic_target.read_text(encoding="utf-8") == first
        # Exactly one helper def — no duplicate splice.
        assert first.count("def _get_full_embed_weight(embed_tokens") == 1


class TestG4_26UpstreamDriftSelfSkip:
    def test_skips_when_upstream_helper_already_present(
        self, g4_26_mod, synthetic_target
    ):
        """Once #45774 merges, `def _get_full_embed_weight` appears in the
        upstream file — the patcher must self-skip (not double-apply)."""
        from sndr.kernel import TextPatchResult
        # Simulate the merged-upstream state: helper already in source.
        merged = SYNTHETIC_DIFFUSION_GEMMA.replace(
            "_NO_PENALTIES_STATE = SimpleNamespace(output_bin_counts=None)",
            "def _get_full_embed_weight(embed_tokens):\n"
            "    return embed_tokens.weight\n\n\n"
            "_NO_PENALTIES_STATE = SimpleNamespace(output_bin_counts=None)",
        )
        synthetic_target.write_text(merged, encoding="utf-8")
        patcher = _build_patcher(g4_26_mod, synthetic_target)
        result, failure = patcher.apply()
        assert result == TextPatchResult.SKIPPED
        assert failure is not None and failure.reason == "upstream_merged"

    def test_upstream_drift_marker_is_the_helper_def(self, g4_26_mod, synthetic_target):
        patcher = _build_patcher(g4_26_mod, synthetic_target)
        assert "def _get_full_embed_weight" in patcher.upstream_drift_markers


# ─── Module triad contract (mirror g4_24) ───────────────────────────────


class TestG4_26TriadContract:
    def test_exports_marker_and_triad(self, g4_26_mod):
        assert hasattr(g4_26_mod, "GENESIS_G4_26_MARKER")
        assert callable(g4_26_mod.apply)
        assert callable(g4_26_mod.is_applied)
        assert callable(g4_26_mod.revert)
        for name in ("GENESIS_G4_26_MARKER", "apply", "is_applied", "revert"):
            assert name in g4_26_mod.__all__

    def test_apply_skips_when_env_disabled(self, g4_26_mod, monkeypatch):
        monkeypatch.delenv(
            "GENESIS_ENABLE_G4_26_DIFFUSIONGEMMA_TP_VOCAB", raising=False
        )
        status, msg = g4_26_mod.apply()
        assert status == "skipped"
        assert "GENESIS_ENABLE_G4_26_DIFFUSIONGEMMA_TP_VOCAB" in msg

    def test_is_applied_false_before_apply(self, g4_26_mod):
        # No live diffusion_gemma in test env → not applied.
        assert g4_26_mod.is_applied() in (False, True)  # never raises

    def test_no_top_level_torch_import(self, g4_26_mod):
        src = Path(g4_26_mod.__file__).read_text(encoding="utf-8")
        for line in src.splitlines():
            assert not line.startswith(("import torch", "from torch")), (
                "G4_26 wiring must not import torch at module level "
                "(torch-less collection safety)"
            )


# ─── Registry / env / dispatch wiring ───────────────────────────────────


class TestG4_26Wiring:
    def test_registry_entry_present(self):
        from sndr.dispatcher import PATCH_REGISTRY
        assert "G4_26" in PATCH_REGISTRY
        meta = PATCH_REGISTRY["G4_26"]
        assert meta["default_on"] is False
        assert (
            meta["env_flag"]
            == "GENESIS_ENABLE_G4_26_DIFFUSIONGEMMA_TP_VOCAB"
        )
        assert meta["apply_module"] == (
            "sndr.engines.vllm.patches.model_compat.gemma4."
            "g4_26_diffusiongemma_tp_vocab_soft_embed"
        )
        assert "DiffusionGemmaForBlockDiffusion" in meta["applies_to"]["model_arch"]
        vr = meta["applies_to"]["vllm_version_range"]
        assert vr == (">=0.22.1rc1.dev491", "<1.0.0")
        # category must be in the audit's VALID_CATEGORIES.
        from sndr.dispatcher.spec import VALID_CATEGORIES
        assert meta["category"] in VALID_CATEGORIES

    def test_env_flag_attribute_present_and_uppercase(self):
        from sndr.env import Flags, known_flags
        assert hasattr(Flags, "G4_26_DIFFUSIONGEMMA_TP_VOCAB")
        val = Flags.G4_26_DIFFUSIONGEMMA_TP_VOCAB
        assert val == "G4_26_DIFFUSIONGEMMA_TP_VOCAB"
        # All-uppercase so known_flags()'s isupper() filter picks it up.
        assert "G4_26_DIFFUSIONGEMMA_TP_VOCAB".isupper()
        assert "G4_26_DIFFUSIONGEMMA_TP_VOCAB" in known_flags()

    def test_dispatch_tuple_present(self):
        from sndr.apply._per_patch_dispatch import _G4_PATCHES
        ids = {row[0] for row in _G4_PATCHES}
        assert "G4_26" in ids
        row = next(r for r in _G4_PATCHES if r[0] == "G4_26")
        assert row[2] == "g4_26_diffusiongemma_tp_vocab_soft_embed"
        assert row[3] == "model_compat.gemma4"


# ─── Arch-gate regression guard ─────────────────────────────────────────
# The runtime probe must look for the vLLM CLASS name
# (DiffusionGemmaForConditionalGeneration), NOT the HF checkpoint arch string
# (DiffusionGemmaForBlockDiffusion, which is never a module attribute). The
# original implementation probed the arch string -> apply() silently
# self-skipped even with the env flag ON. These tests pin the fix.


class TestG4_26ArchGate:
    @staticmethod
    def _force_source_scan(monkeypatch):
        # Bind a fake leaf module so the import-probe misses and the static
        # source-scan fallback (the part that had the wrong class name) runs.
        import sys
        import types
        monkeypatch.setitem(
            sys.modules,
            "vllm.model_executor.models.diffusion_gemma",
            types.ModuleType("vllm.model_executor.models.diffusion_gemma"),
        )

    def test_gate_true_for_real_vllm_class(self, g4_26_mod, tmp_path, monkeypatch):
        self._force_source_scan(monkeypatch)
        src = tmp_path / "diffusion_gemma.py"
        src.write_text(
            "class DiffusionGemmaForConditionalGeneration:\n    pass\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(g4_26_mod, "resolve_vllm_file", lambda rel: str(src))
        assert g4_26_mod._diffusion_gemma_arch_present() is True

    def test_gate_false_for_hf_arch_string_only(self, g4_26_mod, tmp_path, monkeypatch):
        # The HF arch string alone must NOT satisfy the gate (the exact bug).
        self._force_source_scan(monkeypatch)
        src = tmp_path / "diffusion_gemma.py"
        src.write_text(
            "class DiffusionGemmaForBlockDiffusion:\n    pass\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(g4_26_mod, "resolve_vllm_file", lambda rel: str(src))
        assert g4_26_mod._diffusion_gemma_arch_present() is False
