# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N30 — DS conv state layout + spec-decode AL>1 fix.

================================================================
Issue
================================================================
https://github.com/Sandermage/genesis-vllm-patches/issues/17 (noonghunna)

`get_conv_copy_spec` in `vllm/model_executor/layers/mamba/mamba_utils.py`
raises NotImplementedError when:
- VLLM_SSM_CONV_STATE_LAYOUT=DS (dim-strided layout, +6% TPS on 27B)
- num_accepted_tokens > 1 (every prefill with MTP K=3 + AL>1)
- mamba_cache_mode='align' (default)

50/50 LiveCodeBench v6 problems failed instantly on noonghunna's
27B Lorbus + TQ3 + MTP K=3 + TP=1 + structured-CoT config. Container
exited with status 0 after first batch.

================================================================
ROOT CAUSE
================================================================

DS layout: tensor shape (num_blocks, dim, state_len), strides
(dim*state_len, state_len, 1). Slicing `state[block, :, offset:]`
yields a NON-contiguous view because rows of `dim` are interleaved
with `state_len` chunks in memory; the slice picks `state_len-offset`
elements from each row but rows are strided by `state_len`.

Downstream `do_mamba_copy_block` consumes `MambaCopySpec.start_addr`
as a raw pointer for `batch_memcpy`. Non-contiguous source = invalid
for memcpy. Upstream conservatively raises NotImplementedError rather
than silently corrupt state.

================================================================
FIX
================================================================

Two-file text-patch with module-level temp-tensor list + delayed
cleanup pattern:

1. **`mamba_utils.py:get_conv_copy_spec`** — replace the
   NotImplementedError with `.contiguous()` copy + module-level list
   append. Tensor stays alive until next batch.

2. **`v1/worker/mamba_utils.py:do_mamba_copy_block`** — wrap to clear
   temp-tensor list AFTER batch_memcpy, with stream sync ONLY when
   the DS+offset>0 path was actually exercised (cheap predicate).

================================================================
LIFECYCLE CORRECTNESS
================================================================

`batch_memcpy` enqueues async copy on default CUDA stream. To safely
free contiguous-copy temp tensors, we either need to:
(a) Stream-sync after batch_memcpy and free immediately
(b) Delay free until next batch (FIFO ordering on stream guarantees
    previous async ops completed before new ops execute)

We use approach (a) — explicit `current_stream().synchronize()` is
~10-50us per batch and only fires when DS+offset>0 path was used.
Negligible cost for the workload that triggers this (spec-decode
prefill with structured CoT, where TPS is already dominated by
prefill compute).

================================================================
SAFETY MODEL
================================================================
- Default OFF (opt-in via GENESIS_ENABLE_PN30_DS_LAYOUT_SPEC_DECODE=1)
- Pure text-patch, idempotent via marker
- Drift-aware: anchor includes the exact NotImplementedError raise
  block — if upstream fixes this differently, our anchor won't match
- Anchor missing → SKIPPED, source stays vanilla
- Worst case: extra contiguous() copy + stream sync per batch when
  DS layout active and AL>1; cost amortized across long-form generation

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Reporter: noonghunna (issue #17, 2026-05-01).
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pN30_ds_layout_spec_decode")

GENESIS_PN30_MARKER = (
    "Genesis PN30 DS conv state + spec-decode AL>1 (issue #17) v7.65"
)


# ─── Sub-patch 1: model_executor/layers/mamba/mamba_utils.py ────────
# Replace the NotImplementedError raise with contiguous-copy fix.

PN30_PART1_ANCHOR = (
    "    if is_conv_state_dim_first():\n"
    "        # DS layout: (num_blocks, dim, state_len) — state_len is last.\n"
    "        if offset > 0:\n"
    "            # Slicing along the last dim yields a non-contiguous view\n"
    "            # because features (dim) are strided by state_len.\n"
    "            raise NotImplementedError(\n"
    "                \"DS conv state layout does not yet support speculative \"\n"
    "                \"decoding with mamba_cache_mode='align' \"\n"
    "                \"(num_accepted_tokens > 1).\"\n"
    "            )\n"
    "        src_state = state[src_block_id]\n"
)

PN30_PART1_REPLACEMENT = (
    "    if is_conv_state_dim_first():\n"
    "        # DS layout: (num_blocks, dim, state_len) — state_len is last.\n"
    "        # [Genesis PN30 issue #17 fix] Make non-contiguous slice contiguous\n"
    "        # and retain reference until next batch (cleared by patched\n"
    "        # do_mamba_copy_block after stream sync). Replaces the upstream\n"
    "        # NotImplementedError that blocked all spec-decode AL>1 + DS configs.\n"
    "        if offset > 0:\n"
    "            src_state = state[src_block_id, :, offset:].contiguous()\n"
    "            try:\n"
    "                _GENESIS_PN30_TEMP_TENSORS.append(src_state)\n"
    "                _GENESIS_PN30_FLAG[0] = True\n"
    "            except NameError:\n"
    "                pass  # PN30 not loaded — defensive fallback\n"
    "        else:\n"
    "            src_state = state[src_block_id]\n"
)

# Sub-patch 1b: add module-level state for the temp-tensor list + flag.
# Inserted after `class MambaCopySpec` definition.

PN30_PART1B_ANCHOR = (
    "MambaStateCopyFunc: TypeAlias = Callable[\n"
    "    [torch.Tensor, list[int], int, int], MambaCopySpec\n"
    "]\n"
)

PN30_PART1B_REPLACEMENT = (
    "MambaStateCopyFunc: TypeAlias = Callable[\n"
    "    [torch.Tensor, list[int], int, int], MambaCopySpec\n"
    "]\n"
    "\n"
    "# [Genesis PN30 issue #17] Module-level state for DS layout + spec-decode\n"
    "# AL>1 fix. Temp tensors hold contiguous copies of non-contiguous slices;\n"
    "# cleared by patched do_mamba_copy_block in v1/worker/mamba_utils.py\n"
    "# after batch_memcpy + stream sync. Flag is single-element list (mutable\n"
    "# by reference) to avoid ambiguity with module-level rebinds.\n"
    "_GENESIS_PN30_TEMP_TENSORS: list = []\n"
    "_GENESIS_PN30_FLAG: list = [False]\n"
)


# ─── Sub-patch 2: v1/worker/mamba_utils.py:do_mamba_copy_block ──────
# Wrap to clear PN30 temp-tensor list with stream sync.

PN30_PART2_ANCHOR = (
    "def do_mamba_copy_block(copy_bufs: MambaCopyBuffers):\n"
    "    n = copy_bufs.offset\n"
    "    if n == 0:\n"
    "        return\n"
    "    batch_memcpy(\n"
    "        copy_bufs.src_ptrs.copy_to_gpu(n),\n"
    "        copy_bufs.dst_ptrs.copy_to_gpu(n),\n"
    "        copy_bufs.sizes.copy_to_gpu(n),\n"
    "    )\n"
)

PN30_PART2_REPLACEMENT = (
    "def do_mamba_copy_block(copy_bufs: MambaCopyBuffers):\n"
    "    n = copy_bufs.offset\n"
    "    if n == 0:\n"
    "        # [Genesis PN30 issue #17] Even on n==0, opportunistic clear of\n"
    "        # leftover DS temp tensors (defensive — should be empty).\n"
    "        try:\n"
    "            from vllm.model_executor.layers.mamba.mamba_utils import (\n"
    "                _GENESIS_PN30_TEMP_TENSORS, _GENESIS_PN30_FLAG,\n"
    "            )\n"
    "            _GENESIS_PN30_TEMP_TENSORS.clear()\n"
    "            _GENESIS_PN30_FLAG[0] = False\n"
    "        except (ImportError, AttributeError):\n"
    "            pass\n"
    "        return\n"
    "    batch_memcpy(\n"
    "        copy_bufs.src_ptrs.copy_to_gpu(n),\n"
    "        copy_bufs.dst_ptrs.copy_to_gpu(n),\n"
    "        copy_bufs.sizes.copy_to_gpu(n),\n"
    "    )\n"
    "    # [Genesis PN30 issue #17] If DS layout + offset>0 path was used\n"
    "    # this batch, the contiguous-copy temp tensors are still alive in\n"
    "    # _GENESIS_PN30_TEMP_TENSORS. Sync the stream to ensure batch_memcpy\n"
    "    # consumed them, then clear. Cost: ~10-50us, only fires when DS+\n"
    "    # offset>0 actually triggered (typical AL=1 fast path is no-op).\n"
    "    try:\n"
    "        from vllm.model_executor.layers.mamba.mamba_utils import (\n"
    "            _GENESIS_PN30_TEMP_TENSORS, _GENESIS_PN30_FLAG,\n"
    "        )\n"
    "        if _GENESIS_PN30_FLAG[0]:\n"
    "            import torch as _torch_pn30\n"
    "            _torch_pn30.cuda.current_stream().synchronize()\n"
    "            _GENESIS_PN30_TEMP_TENSORS.clear()\n"
    "            _GENESIS_PN30_FLAG[0] = False\n"
    "    except (ImportError, AttributeError):\n"
    "        pass\n"
)


def _make_patcher_part1() -> TextPatcher | None:
    target = resolve_vllm_file(
        "model_executor/layers/mamba/mamba_utils.py"
    )
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN30 model_executor/layers/mamba/mamba_utils.py — DS layout "
            "spec-decode AL>1 fix (issue #17)"
        ),
        target_file=str(target),
        marker=GENESIS_PN30_MARKER + " part1",
        sub_patches=[
            TextPatch(
                name="pN30_get_conv_copy_spec_contiguous",
                anchor=PN30_PART1_ANCHOR,
                replacement=PN30_PART1_REPLACEMENT,
                required=True,
            ),
            TextPatch(
                name="pN30_module_level_state",
                anchor=PN30_PART1B_ANCHOR,
                replacement=PN30_PART1B_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN30",
            # If upstream removes NotImplementedError or rewrites the path,
            # our anchor won't match → no-op apply.
        ],
    )


def _make_patcher_part2() -> TextPatcher | None:
    target = resolve_vllm_file("v1/worker/mamba_utils.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN30 v1/worker/mamba_utils.py — do_mamba_copy_block stream "
            "sync + temp tensor cleanup (issue #17)"
        ),
        target_file=str(target),
        marker=GENESIS_PN30_MARKER + " part2",
        sub_patches=[
            TextPatch(
                name="pN30_do_mamba_copy_block_cleanup",
                anchor=PN30_PART2_ANCHOR,
                replacement=PN30_PART2_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN30",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN30 — DS layout spec-decode AL>1 fix (two-file text-patch)."""
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN30")
    log_decision("PN30", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    # Both files must patch successfully — partial application would
    # leave the system in inconsistent state (one half of the
    # coordinated fix without the other).
    p1 = _make_patcher_part1()
    p2 = _make_patcher_part2()
    if p1 is None or p2 is None:
        return "skipped", (
            "target file(s) not resolvable — vllm tree may differ "
            "from expected layout"
        )

    r1, f1 = p1.apply()
    if r1 == TextPatchResult.FAILED:
        return "failed", (
            f"PN30 part1 (mamba_utils.py:get_conv_copy_spec) failed: "
            f"{f1.detail if f1 else 'unknown'}"
        )

    r2, f2 = p2.apply()
    if r2 == TextPatchResult.FAILED:
        # Partial patch state — log warning. Part1 stays applied;
        # cleanup will not run but the contiguous() fix itself is
        # correct (just leaks a small list of tensors per batch).
        log.warning(
            "[PN30] part2 (do_mamba_copy_block) failed: %s — part1 "
            "applied but cleanup will not fire. Tensor list will grow "
            "per batch until process restart. Recommend disabling PN30 "
            "until both halves can apply.",
            f2.detail if f2 else "unknown",
        )
        return "failed", "PN30 partial application — see warning"

    # Both halves applied (or skipped if anchors missing — drift-safe)
    return result_to_wiring_status(
        r1 if r1 != TextPatchResult.APPLIED else r2,
        f1 if r1 != TextPatchResult.APPLIED else f2,
        applied_message=(
            "PN30 applied: DS conv state layout + spec-decode AL>1 path "
            "now uses contiguous-copy + delayed cleanup. Two-file patch — "
            "mamba_utils.py:get_conv_copy_spec replaces NotImplementedError "
            "with .contiguous() copy + temp-tensor list; "
            "v1/worker/mamba_utils.py:do_mamba_copy_block adds stream sync "
            "+ list clear after batch_memcpy when DS+offset>0 path used. "
            "Closes issue #17. Cost: ~10-50us per batch when path active."
        ),
        patch_name="PN30 DS layout + spec-decode AL>1",
    )
