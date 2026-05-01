# SPDX-License-Identifier: Apache-2.0
"""TDD tests for PN32 — GDN chunked-prefill (Cliff 2 fix).

CPU-runnable structural tests. Real GPU validation requires:
  - Single 24GB GPU (1×3090 / 1×4090 / 1×5090)
  - >50K-token single-shot prompt
  - Hybrid GDN model (Qwen3.5/3.6 27B)

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Reporter: noonghunna (CLIFF2_INVESTIGATION_20260430.md).
"""
from __future__ import annotations

import pytest


def test_pn32_wiring_imports():
    from vllm._genesis.wiring.hybrid import patch_N32_gdn_chunked_prefill as mod
    assert hasattr(mod, "apply")
    assert hasattr(mod, "GENESIS_PN32_MARKER")


def test_pn32_dispatcher_registry():
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    assert "PN32" in PATCH_REGISTRY
    e = PATCH_REGISTRY["PN32"]
    assert e["env_flag"] == "GENESIS_ENABLE_PN32_GDN_CHUNKED_PREFILL"
    assert e["default_on"] is False


def test_pn32_skips_when_env_off(monkeypatch):
    monkeypatch.delenv(
        "GENESIS_ENABLE_PN32_GDN_CHUNKED_PREFILL", raising=False
    )
    from vllm._genesis.wiring.hybrid.patch_N32_gdn_chunked_prefill import apply
    status, reason = apply()
    assert status == "skipped"
    assert "opt-in" in reason.lower()


def test_pn32_anchor_matches_gdn_forward_cuda_pattern():
    from vllm._genesis.wiring.hybrid.patch_N32_gdn_chunked_prefill import (
        PN32_ANCHOR, PN32_REPLACEMENT,
    )
    # Anchor must match ORIGINAL upstream pattern (NOT P28-modified).
    # PN32 conflicts with P28 — both text-patch same lines, operator
    # must choose one. See conflicts_with: ["P28"] in dispatcher entry.
    assert "core_attn_out = torch.zeros(" in PN32_ANCHOR
    assert "torch.ops.vllm.gdn_attention_core" in PN32_ANCHOR
    # Anchor includes the post-projection block for full-replacement
    assert "self.out_proj(core_attn_out)" in PN32_ANCHOR


def test_pn32_documents_p28_conflict():
    """PN32 source documents P28 conflict in registry comment + wiring docstring.

    P28 is legacy patch (not in dispatcher PATCH_REGISTRY) so we can't list
    it via conflicts_with. Documented via source comments instead.
    """
    import inspect
    from vllm._genesis.wiring.hybrid import patch_N32_gdn_chunked_prefill as mod
    src = inspect.getsource(mod)
    assert "P28" in src
    # Conflict noted in dispatcher entry too
    from vllm._genesis import dispatcher
    disp_src = inspect.getsource(dispatcher)
    assert "P28" in disp_src and "PN32" in disp_src


def test_pn32_replacement_has_chunked_loop():
    """Replacement implements range-loop chunking with proper slicing."""
    from vllm._genesis.wiring.hybrid.patch_N32_gdn_chunked_prefill import (
        PN32_REPLACEMENT,
    )
    # Conditional gating
    assert "_genesis_pn32_enabled" in PN32_REPLACEMENT
    assert "num_tokens > _genesis_pn32_threshold" in PN32_REPLACEMENT
    # Range loop with chunk size
    assert "for _genesis_pn32_start in range" in PN32_REPLACEMENT
    assert "_genesis_pn32_chunk_size" in PN32_REPLACEMENT
    # Slicing of mixed_qkv, b, a, z, output
    assert "mixed_qkv[_genesis_pn32_start:_genesis_pn32_end]" in PN32_REPLACEMENT
    assert "b[_genesis_pn32_start:_genesis_pn32_end]" in PN32_REPLACEMENT
    assert "a[_genesis_pn32_start:_genesis_pn32_end]" in PN32_REPLACEMENT
    assert "z[_genesis_pn32_start:_genesis_pn32_end]" in PN32_REPLACEMENT
    assert "output[_genesis_pn32_start:_genesis_pn32_end]" in PN32_REPLACEMENT


def test_pn32_replacement_falls_through_when_disabled():
    """When env disabled OR num_tokens ≤ threshold, original path runs."""
    from vllm._genesis.wiring.hybrid.patch_N32_gdn_chunked_prefill import (
        PN32_REPLACEMENT,
    )
    # Replacement has explicit `else:` branch with original upstream logic
    assert "else:" in PN32_REPLACEMENT
    # Original upstream path preserved in else branch (matches anchor —
    # PN32 uses original pattern, NOT P28-modified, due to conflict)
    assert "core_attn_out = torch.zeros(" in PN32_REPLACEMENT
    # gdn_attention_core called twice in replacement (once in chunked path,
    # once in fallback path)
    assert PN32_REPLACEMENT.count("gdn_attention_core(") == 2


def test_pn32_replacement_explicit_del_for_chunk_buffer():
    """Replacement has explicit `del` to help allocator reuse chunk slot."""
    from vllm._genesis.wiring.hybrid.patch_N32_gdn_chunked_prefill import (
        PN32_REPLACEMENT,
    )
    assert "del _genesis_pn32_chunk_attn_out" in PN32_REPLACEMENT


def test_pn32_replacement_documents_state_continuity():
    """Replacement comments explain state continuity assumption."""
    from vllm._genesis.wiring.hybrid.patch_N32_gdn_chunked_prefill import (
        PN32_REPLACEMENT,
    )
    # Critical assumption documented
    assert "state" in PN32_REPLACEMENT.lower()
    assert "layer-name" in PN32_REPLACEMENT or "layer_name" in PN32_REPLACEMENT.lower()


def test_pn32_register_in_apply_all():
    from vllm._genesis.patches.apply_all import (
        PATCH_REGISTRY as APPLY_REGISTRY,
    )
    names = [name for name, _ in APPLY_REGISTRY]
    pn32 = [n for n in names if "PN32" in n and "chunked" in n.lower()]
    assert len(pn32) == 1, f"PN32 chunked-prefill not registered, names: {names[:5]}"


def test_pn32_marker_unique():
    from vllm._genesis.wiring.hybrid.patch_N32_gdn_chunked_prefill import (
        GENESIS_PN32_MARKER,
    )
    assert "PN32" in GENESIS_PN32_MARKER
    assert "Cliff 2" in GENESIS_PN32_MARKER


def test_pn32_threshold_defaults_documented():
    """Source documents threshold=16384 and chunk_size=8192 defaults."""
    import inspect
    from vllm._genesis.wiring.hybrid import patch_N32_gdn_chunked_prefill as mod
    src = inspect.getsource(mod)
    assert "16384" in src
    assert "8192" in src


def test_pn32_documents_cross_rig_validation_requirement():
    """Source explicitly states our PROD doesn't hit Cliff 2 + needs validation."""
    import inspect
    from vllm._genesis.wiring.hybrid import patch_N32_gdn_chunked_prefill as mod
    src = inspect.getsource(mod)
    assert "validation" in src.lower()
    assert "TP=2" in src or "single-GPU" in src or "single-24GB" in src


def test_pn32_documents_peak_memory_math():
    """Source shows the math justifying the chunking approach."""
    import inspect
    from vllm._genesis.wiring.hybrid import patch_N32_gdn_chunked_prefill as mod
    src = inspect.getsource(mod)
    assert "819 MiB" in src or "MiB" in src
    assert "24 GiB" in src or "24GB" in src or "24 GB" in src
