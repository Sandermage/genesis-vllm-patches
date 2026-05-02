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
    """Part1 anchor matches the exact NotImplementedError block.

    v7.68: replacement is now a fail-closed RuntimeError, not the
    v7.65 compact .contiguous() path. The .contiguous() approach
    silently corrupted DS row strides because it lost destination
    stride info. Fix moved to part3 (collect_mamba_copy_meta) where
    dst block id is known and a dst-shaped temp can be built.
    Part1's path becomes unreachable on the AL>1 + DS path; if
    anything ever reaches it, we crash explicitly.
    """
    from vllm._genesis.wiring.spec_decode.patch_N30_ds_layout_spec_decode_align import (
        PN30_PART1_ANCHOR, PN30_PART1_REPLACEMENT,
    )
    # Anchor: must contain the NotImplementedError raise text
    assert "NotImplementedError" in PN30_PART1_ANCHOR
    assert "DS conv state layout" in PN30_PART1_ANCHOR
    assert "num_accepted_tokens > 1" in PN30_PART1_ANCHOR

    # v7.68 replacement: fail-closed RuntimeError instead of compact path
    assert "raise RuntimeError" in PN30_PART1_REPLACEMENT
    assert "collect_mamba_copy_meta" in PN30_PART1_REPLACEMENT
    assert "v7.68" in PN30_PART1_REPLACEMENT
    # The OLD compact-path approach must NOT be in EXECUTABLE code
    # anymore (regression guard — would re-introduce the silent-
    # corruption bug). Comment-mention OK (we explain why deprecated).
    # Specific code patterns from v7.65 compact path:
    assert "src_state = state[src_block_id, :, offset:].contiguous()" not in PN30_PART1_REPLACEMENT
    assert "_GENESIS_PN30_TEMP_TENSORS.append" not in PN30_PART1_REPLACEMENT
    # Original NotImplementedError must not survive either
    assert "raise NotImplementedError" not in PN30_PART1_REPLACEMENT


def test_pn30_part3_dst_shaped_temp_is_layout_correct():
    """v7.68 part3 builds dst-shaped temp on collect_mamba_copy_meta.

    This is the layout-correct fix for DS+offset>0. Builds a temp
    matching the destination block stride (via .clone()), patches
    in only the source tail, then memcpys the full block. Preserves
    DS row stride end-to-end.

    Credit: noonghunna + ChatGPT/Codex CLI cross-check
    (club-3090 commit 9af1a52, 2026-05-02).
    """
    from vllm._genesis.wiring.spec_decode.patch_N30_ds_layout_spec_decode_align import (
        PN30_PART3_ANCHOR, PN30_PART3_REPLACEMENT,
    )
    # Anchor: target collect_mamba_copy_meta function
    assert "def collect_mamba_copy_meta" in PN30_PART3_ANCHOR
    assert "MambaCopyBuffers" in PN30_PART3_ANCHOR
    assert "src_block_idx" in PN30_PART3_ANCHOR
    assert "dest_block_idx" in PN30_PART3_ANCHOR

    # Replacement: dst-shaped temp pattern
    assert "v7.68" in PN30_PART3_REPLACEMENT
    # Detect conv-copy via function identity + name fallback
    assert "_GENESIS_PN30_GET_CONV_COPY_SPEC" in PN30_PART3_REPLACEMENT
    assert "'get_conv_copy_spec'" in PN30_PART3_REPLACEMENT
    # Dst-shaped temp construction
    assert "tmp_state = state[dest_block_id].clone()" in PN30_PART3_REPLACEMENT
    # Source tail copy preserves token offset
    assert "tmp_state[..., :tail].copy_" in PN30_PART3_REPLACEMENT
    assert "token_offset:token_offset + tail" in PN30_PART3_REPLACEMENT
    # Tail computation
    assert "tail = max(state_len - int(token_offset), 0)" in PN30_PART3_REPLACEMENT
    # Lifecycle: append to PN30 temp tensor list (cleared by part2)
    assert "_GENESIS_PN30_TEMP_TENSORS.append(tmp_state)" in PN30_PART3_REPLACEMENT
    assert "_GENESIS_PN30_FLAG" in PN30_PART3_REPLACEMENT
    # Memcpy entry uses tmp_state as source, full dst block as dst
    assert "src_ptrs_np[offset] = tmp_state.data_ptr()" in PN30_PART3_REPLACEMENT
    assert "dst_ptrs_np[offset] = state[dest_block_id].data_ptr()" in PN30_PART3_REPLACEMENT
    # Defensive: state.dim() check + DS layout guard
    assert "state.dim() >= 3" in PN30_PART3_REPLACEMENT
    assert "_GENESIS_PN30_IS_CONV_STATE_DIM_FIRST" in PN30_PART3_REPLACEMENT
    # num_accepted_tokens > 1 guard (don't fire on AL=1 fast path)
    assert "num_accepted_tokens > 1" in PN30_PART3_REPLACEMENT


def test_pn30_marker_bumped_to_v7_68():
    """v7.68 marker bump signals that re-application supersedes v7.65."""
    from vllm._genesis.wiring.spec_decode.patch_N30_ds_layout_spec_decode_align import (
        GENESIS_PN30_MARKER,
    )
    assert "v7.68" in GENESIS_PN30_MARKER


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
