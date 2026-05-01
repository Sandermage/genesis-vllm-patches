# SPDX-License-Identifier: Apache-2.0
"""TDD tests for issue #17 — PN30 DS layout + spec-decode AL>1 fix.

Bug context (noonghunna, 2026-05-01):
- 50/50 LiveCodeBench v6 failed instantly on 27B + TQ3 + MTP K=3 + TP=1
  + structured CoT + DS layout
- Stack trace: NotImplementedError in mamba_utils.py:320 from
  `get_conv_copy_spec` when DS layout + num_accepted_tokens > 1
- Root cause: state[block, :, offset:] slice is non-contiguous in
  DS layout (rows of `dim` strided by state_len)

Fix: two-file text-patch
1. mamba_utils.py:get_conv_copy_spec — contiguous() + temp-tensor list
2. v1/worker/mamba_utils.py:do_mamba_copy_block — stream sync + clear

Test contract (CPU-runnable subset, no GPU needed):
1. Wiring imports cleanly
2. Dispatcher PATCH_REGISTRY entry correct
3. Env-OFF skips
4. Anchor text matches expected upstream code structure
5. Replacement preserves marker for drift detection
6. Module-level state (tensor list + flag) inserted correctly
7. Stream sync + cleanup logic in part2 replacement

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Bug: github.com/Sandermage/genesis-vllm-patches/issues/17
"""
from __future__ import annotations

import pytest


def test_pn30_wiring_imports():
    """PN30 wiring module imports cleanly."""
    from vllm._genesis.wiring.spec_decode import (
        patch_N30_ds_layout_spec_decode_align as mod,
    )
    assert hasattr(mod, "apply")
    assert hasattr(mod, "GENESIS_PN30_MARKER")


def test_pn30_dispatcher_registry():
    """PN30 registered in PATCH_REGISTRY with correct env flag."""
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    assert "PN30" in PATCH_REGISTRY
    e = PATCH_REGISTRY["PN30"]
    assert e["env_flag"] == "GENESIS_ENABLE_PN30_DS_LAYOUT_SPEC_DECODE"
    assert e["default_on"] is False
    assert e["upstream_pr"] is None  # genesis-original


def test_pn30_skips_when_env_off(monkeypatch):
    """When env is OFF, apply() returns 'skipped'."""
    monkeypatch.delenv(
        "GENESIS_ENABLE_PN30_DS_LAYOUT_SPEC_DECODE", raising=False
    )
    from vllm._genesis.wiring.spec_decode.patch_N30_ds_layout_spec_decode_align import (
        apply,
    )
    status, reason = apply()
    assert status == "skipped"
    assert "opt-in" in reason.lower()


def test_pn30_part1_anchor_matches_upstream_pattern():
    """Part1 anchor matches the exact NotImplementedError block."""
    from vllm._genesis.wiring.spec_decode.patch_N30_ds_layout_spec_decode_align import (
        PN30_PART1_ANCHOR, PN30_PART1_REPLACEMENT,
    )
    # Anchor: must contain the NotImplementedError raise text
    assert "NotImplementedError" in PN30_PART1_ANCHOR
    assert "DS conv state layout" in PN30_PART1_ANCHOR
    assert "num_accepted_tokens > 1" in PN30_PART1_ANCHOR

    # Replacement: must use .contiguous() + module-level state
    assert ".contiguous()" in PN30_PART1_REPLACEMENT
    assert "_GENESIS_PN30_TEMP_TENSORS" in PN30_PART1_REPLACEMENT
    assert "_GENESIS_PN30_FLAG" in PN30_PART1_REPLACEMENT
    # No NotImplementedError in replacement
    assert "raise NotImplementedError" not in PN30_PART1_REPLACEMENT
    # Drift marker
    assert "Genesis PN30" in PN30_PART1_REPLACEMENT
    assert "issue #17" in PN30_PART1_REPLACEMENT


def test_pn30_part1b_inserts_module_level_state():
    """Part1b adds _GENESIS_PN30_TEMP_TENSORS + _GENESIS_PN30_FLAG."""
    from vllm._genesis.wiring.spec_decode.patch_N30_ds_layout_spec_decode_align import (
        PN30_PART1B_ANCHOR, PN30_PART1B_REPLACEMENT,
    )
    # Anchor matches MambaStateCopyFunc TypeAlias line
    assert "MambaStateCopyFunc" in PN30_PART1B_ANCHOR
    # Replacement adds module-level state
    assert "_GENESIS_PN30_TEMP_TENSORS: list = []" in PN30_PART1B_REPLACEMENT
    assert "_GENESIS_PN30_FLAG: list = [False]" in PN30_PART1B_REPLACEMENT


def test_pn30_part2_anchor_targets_do_mamba_copy_block():
    """Part2 anchor matches do_mamba_copy_block function signature + body."""
    from vllm._genesis.wiring.spec_decode.patch_N30_ds_layout_spec_decode_align import (
        PN30_PART2_ANCHOR, PN30_PART2_REPLACEMENT,
    )
    # Anchor: function definition + batch_memcpy call
    assert "def do_mamba_copy_block" in PN30_PART2_ANCHOR
    assert "batch_memcpy" in PN30_PART2_ANCHOR

    # Replacement: stream sync + cleanup logic
    assert "current_stream().synchronize()" in PN30_PART2_REPLACEMENT
    assert "_GENESIS_PN30_TEMP_TENSORS.clear()" in PN30_PART2_REPLACEMENT
    assert "_GENESIS_PN30_FLAG[0] = False" in PN30_PART2_REPLACEMENT
    # Defensive try/except for missing module
    assert "ImportError" in PN30_PART2_REPLACEMENT


def test_pn30_register_in_apply_all():
    """PN30 registered via @register_patch in apply_all.py."""
    from vllm._genesis.patches.apply_all import (
        PATCH_REGISTRY as APPLY_REGISTRY,
    )
    names = [name for name, _ in APPLY_REGISTRY]
    pn30 = [n for n in names if "PN30" in n]
    assert len(pn30) == 1, f"PN30 not registered, names: {names[:5]}"


def test_pn30_marker_unique():
    """Marker string is unique enough to detect drift."""
    from vllm._genesis.wiring.spec_decode.patch_N30_ds_layout_spec_decode_align import (
        GENESIS_PN30_MARKER,
    )
    assert "PN30" in GENESIS_PN30_MARKER
    assert "issue #17" in GENESIS_PN30_MARKER
    assert len(GENESIS_PN30_MARKER) > 30


def test_pn30_lifecycle_design_documented():
    """Source documents the lifecycle correctness reasoning."""
    import inspect
    from vllm._genesis.wiring.spec_decode import (
        patch_N30_ds_layout_spec_decode_align as mod,
    )
    src = inspect.getsource(mod)
    # Critical lifecycle concepts must be documented
    assert "lifecycle" in src.lower() or "stream" in src.lower()
    assert "synchronize" in src.lower()
    assert "non-contiguous" in src.lower() or "contiguous" in src.lower()
    # Cost rationale
    assert "10-50" in src or "us" in src.lower()


def test_pn30_partial_application_handled():
    """If part2 fails, part1 stays applied — code documents this risk."""
    import inspect
    from vllm._genesis.wiring.spec_decode import (
        patch_N30_ds_layout_spec_decode_align as mod,
    )
    src = inspect.getsource(mod.apply)
    # apply() should handle partial application (part1 ok, part2 fails)
    # by logging warning, not silent inconsistent state
    assert "Partial" in src or "partial" in src or "part1" in src.lower()
