# SPDX-License-Identifier: Apache-2.0
"""PN351 — vendor of OPEN PR vllm#43257 (ShuaiShao93) Triton unified_attention tune for head_dim >= 512.

Triton unified attention tile-size + warps + stages tuned for very large head dims
==================================================================================

**Problem**: ``vllm/v1/attention/ops/triton_unified_attention.py``
hardcodes ``num_warps=4`` and ``num_stages=3`` in the kernel launch.
For head sizes < 256 this is the sweet spot — high occupancy, good
pipeline coverage. But for head_dim >= 512 (Gemma 4 31B uses 512;
Gemma 4 26B-A4B uses 256 + extra global heads with 512), the kernel
hits a register cliff: per-thread register pressure forces the
backend to spill or cap occupancy. Author bench on Hopper SM 9.0 + FP8:

  * head_dim=512, prefill, FP8 KV — occupancy at default 4w/3s: 6-13 %
  * head_dim=512, prefill, FP8 KV — with 8w/2s + 64-tile: 25-40 %

The same architectural class applies on Ampere SM 8.6 (A5000): same
register file size, same shared-memory layout class. The numerical
improvement transfers — actual wall-clock varies because A5000 has
6 % less SM count than A100 and 25 % of H100, but the *occupancy
fraction unlock* carries directly. Expected: ``-3-7 % decode_TPOT``
on Gemma 4 31B FP8 prefill on our PROD.

**Three changes in one method + kernel launch**:

  1. ``_get_tile_size`` (helper): add a fast-path for ``head_size >= 512
     and is_prefill and element_size == 1`` → return 64 (vs default
     32). Larger inner-K tile improves L2 reuse on large head dims.
     Restricted to FP8 element size because at BF16 the K+V shared
     memory would exceed the per-SM limit and the kernel would fail
     to launch (verified by author).

  2. ``unified_attention`` (caller): add two kwargs on the kernel
     ``triton_unified_attention[...](...)`` launch::

         num_warps=8 if head_size >= 512 else 4,
         num_stages=2 if head_size >= 512 else 3,

     8 warps spreads the accumulator across more warps relieving
     register pressure; 2-stage pipeline is shallower but uses less
     shared memory (offsetting the larger tile size).

**What our PROD hits**:

  * Gemma 4 31B FP8: head_dim=512 → all three improvements fire on
    prefill. Decode is < 32 query tokens — different tile path.
  * Gemma 4 26B-A4B FP8: standard layer head_dim=256 (no change);
    global-attention heads have head_dim=512 → improvements fire on
    the global-attention layers (every 4th layer per Gemma 4 pattern).
  * Qwen3.6 35B FP8: head_dim=128 — no change (default 32-tile + 4w/3s
    preserved).
  * Qwen3.6 27B INT4: head_dim=128 — no change.

So PN351 is a Gemma-4-specific perf win; no-op on Qwen3.6 (preserves
current well-tuned defaults).

**Why we vendor an OPEN PR**:

  * 14-LOC diff, surgical, gated by ``head_size >= 512`` so the
    Qwen / FA2 / Mamba paths see ZERO change.
  * Author bench shows real occupancy unlock; same numerical class on
    Ampere SM 8.6.
  * No interaction with shmem budget — the FP8 gate keeps the
    K+V shared-memory footprint within ~99 KiB opt-in (the same
    constraint PN345 enforces precisely).

Implementation strategy
=======================

Two atomic sub-patches on a single file
(``v1/attention/ops/triton_unified_attention.py``):

  * Sub-1 (``_get_tile_size``): replace the docstring and add the
    ``head_size >= 512 and is_prefill and element_size == 1`` branch
    BEFORE the default behaviour. ``required=True`` — the
    ``_get_tile_size`` function is untouched by vllm#45151, so this
    anchor never breaks.
  * Sub-2 (``kernel_unified_attention`` launch): add ``num_warps`` and
    ``num_stages`` kwargs on the triton kernel call. MULTI-ANCHOR
    (batch-3 pin-bump protection). THREE launch-site shapes exist in
    the wild — verified 2026-06-13 against the live pin AND upstream
    ``main`` (``gh pr diff 45151`` + raw main fetch):

      * Variant A — our CURRENT pin ``g303916e93`` (0.22.1rc1.dev259):
        the call site ends ``USE_TD_QO=use_td_qo,\n    )`` with NO
        ``**launch_kwargs,`` line. This is the only shape we run TODAY.
      * Variant B — future pin descended from upstream ``main`` but
        BEFORE #45151: ``main`` has refactored the launch to splat a
        ``**launch_kwargs,`` dict (native ``launch_num_warps`` /
        ``launch_num_stages`` machinery), so the tail is now
        ``USE_TD_QO=use_td_qo,\n        **launch_kwargs,\n    )``.
      * Variant C — future pin AFTER #45151: the 7 fused-quant kwargs
        are spliced between ``USE_TD_QO`` and ``**launch_kwargs,`` →
        ``USE_TD_QO=use_td_qo,\n<7 kwargs>\n        **launch_kwargs,\n    )``.

    All three variants are ``required=False`` under the P18B/PN32
    "at-least-one" convention; EXACTLY ONE matches on any given pin and
    the others soft-skip. apply() enforces that at least one fired (a
    tile-only apply is incoherent — Sub-1 sets the tile, Sub-2 sets the
    launch config; either alone is a no-op). In B and C our literal
    ``num_warps=``/``num_stages=`` kwargs are inserted BEFORE the
    ``**launch_kwargs,`` splat; they never collide with main's native
    ``launch_kwargs["num_warps"]`` because that is populated only on the
    ``tuned_large_head`` path (``head_size == 256`` + B200 device-cap-100),
    which is mutually exclusive with PN351's ``head_size >= 512`` gate and
    never fires on our A5000 SM 8.6 PROD. See the anchor-definition block
    below for the full upstream-main + #45151 diff derivation.

Composition + safety
====================

* No overlap with PN29x / PN345 (those target FLA chunk kernels — a
  different file). PN351 touches ``triton_unified_attention.py``
  which is the unified backend for full-attention layers.
* No overlap with PN340/PN341 (MTP decode bubbles — different files).
* Composes with all our existing patches.
* Risk: LOW — gated path; non-Gemma-4 models bypass it entirely.

Author: Sander (Sandermage / Aleksandr Barzov), Odessa, Ukraine.
Vendor target: vllm-project/vllm#43257 (OPEN as of 2026-06-09).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

log = logging.getLogger("genesis.wiring.pn351_triton_unified_attention_large_head")

GENESIS_PN351_MARKER = (
    "Genesis PN351 vendor of vllm#43257 (Triton unified_attention head_dim>=512) v1"
)

_TARGET_REL = "v1/attention/ops/triton_unified_attention.py"


# ── Sub-1: _get_tile_size head_size >= 512 branch ───────────────────────
# Anchor: the function docstring + the Gemma3 branch + the default-behaviour
# comment, ensuring unique match.
PN351_TILE_OLD = (
    "    \"\"\"Select tile size with Gemma3-specific optimization.\"\"\"\n"
    "    if _is_gemma3_attention(head_size, sliding_window):\n"
    "        # Gemma3: use 32 for decode (default is 16)\n"
    "        return 32\n"
    "\n"
    "    # Default behavior\n"
    "    if is_prefill:\n"
)
PN351_TILE_NEW = (
    "    \"\"\"Select tile size with head-size-specific optimization.\n"
    "\n"
    "    [Genesis PN351 vendor of vllm#43257]: for head_dim >= 512 FP8\n"
    "    prefill (Gemma 4 31B / 26B global-attention heads), use a larger\n"
    "    inner-K tile (64 vs default 32) to improve L2 reuse. Restricted\n"
    "    to element_size == 1 (FP8) because with 2-byte dtypes the K+V\n"
    "    shared-memory footprint exceeds the per-SM limit and the kernel\n"
    "    would fail to launch.\"\"\"\n"
    "    if _is_gemma3_attention(head_size, sliding_window):\n"
    "        # Gemma3: use 32 for decode (default is 16)\n"
    "        return 32\n"
    "\n"
    "    # [Genesis PN351] head_dim >= 512 FP8 prefill: larger tile.\n"
    "    if head_size >= 512 and is_prefill and element_size == 1:\n"
    "        return 64\n"
    "\n"
    "    # Default behavior\n"
    "    if is_prefill:\n"
)


# ── Sub-2: unified_attention kernel launch — add num_warps + num_stages ──
#
# MULTI-ANCHOR pin-bump protection (batch-3, 2026-06-13; corrected against
# the REAL upstream-main launch shape after review).
# =====================================================================
# The launch kwargs are appended at the ``kernel_unified_attention[grid](
# ... )`` call site, anchored on the existing CHUNK_SIZE / USE_TD /
# USE_TD_QO sequence and whatever trailing form the call site has.
#
# Verified 2026-06-13 against THREE concrete shapes:
#
# (A) CURRENT pin ``g303916e93`` (0.22.1rc1.dev259, the tree we run). The
#     call site has NO ``**launch_kwargs`` — its tail is literally:
#
#         USE_TD_QO=use_td_qo,
#         )
#
#     Byte-verified: ``launch_kwargs``/``num_warps``/``num_stages`` all
#     count 0 in the pristine pin file.
#
# (B) Upstream ``main`` BEFORE #45151 (raw fetch 2026-06-13). main has
#     refactored the launch to splat a tuned-params dict — the tail is:
#
#         USE_TD_QO=use_td_qo,
#         **launch_kwargs,
#         )
#
#     where ``launch_kwargs`` is built from native ``launch_num_warps`` /
#     ``launch_num_stages`` (populated ONLY on main's own ``tuned_large_head``
#     path: ``head_size == 256`` + ``is_device_capability_family(100)`` /
#     B200 — disjoint from PN351's ``head_size >= 512`` FP8 gate, so on our
#     A5000 path ``launch_kwargs`` is empty and our literal kwargs never
#     collide with a ``**launch_kwargs['num_warps']``).
#
# (C) Upstream ``main`` AFTER #45151 ("Fuse per-group FP8 dynamic quant
#     into Triton attention epilogue", OPEN). ``gh pr diff 45151`` inserts
#     SEVEN kwargs between ``USE_TD_QO`` and the existing ``**launch_kwargs,``
#     context line (the ``**launch_kwargs,`` is a CONTEXT line in the diff,
#     not an added line) — tail:
#
#         USE_TD_QO=use_td_qo,
#         USE_FP8_GROUP=use_fp8_group,           <─┐
#         GROUP_SIZE=group_size,                   │
#         out_group_scale_ptr=output_group_scale,  │ 7 kwargs #45151 adds
#         out_group_scale_stride_0=(...),          │ between USE_TD_QO and
#         out_group_scale_stride_1=(...),          │ **launch_kwargs,
#         NUM_GROUPS_PER_HEAD=num_groups_per_head,  │
#         USE_UE8M0=output_group_ue8m0,          <─┘
#         **launch_kwargs,
#         )
#
# The earlier single-form ``USE_TD_QO=use_td_qo,\n    )`` anchor matches
# ONLY shape (A). Both future shapes (B, C) end with ``**launch_kwargs,``
# before the ``)``, so the fix is a THREE-variant launch sub-patch under
# the P18B/PN32 "required-at-least-one" convention (all ``required=False``;
# apply() enforces at least one fired). On any given pin EXACTLY ONE
# variant's anchor is present (the shapes are mutually exclusive) so the
# others soft-skip. Behavior is byte-identical: every variant emits the
# same warps/stages kwargs. In B and C we insert BEFORE ``**launch_kwargs,``
# so main's splat still applies last.

# Variant A — CURRENT pin g303916e93 (no launch_kwargs). The 3-kwarg
# sequence is unique; the closing `)` follows USE_TD_QO directly.
PN351_LAUNCH_OLD = (
    "        CHUNK_SIZE=chunk_size,\n"
    "        USE_TD=use_td,\n"
    "        USE_TD_QO=use_td_qo,\n"
    "    )\n"
)
PN351_LAUNCH_NEW = (
    "        CHUNK_SIZE=chunk_size,\n"
    "        USE_TD=use_td,\n"
    "        USE_TD_QO=use_td_qo,\n"
    "        # [Genesis PN351 vendor of vllm#43257] head_dim >= 512 hits a\n"
    "        # register cliff under default 4w/3s. 8 warps spreads the\n"
    "        # accumulator across more warps; 2-stage pipeline is shallower\n"
    "        # but uses less shmem (offsetting the larger 64-tile). Gemma 4\n"
    "        # 31B + 26B-A4B global heads benefit; Qwen3.6 head_dim=128\n"
    "        # bypasses entirely.\n"
    "        num_warps=8 if head_size >= 512 else 4,\n"
    "        num_stages=2 if head_size >= 512 else 3,\n"
    "    )\n"
)

# Variant B — upstream main BEFORE #45151. Re-anchored on the same
# USE_TD_QO prefix + the ``**launch_kwargs,`` splat line main introduced.
# We insert our warps/stages kwargs BEFORE ``**launch_kwargs,`` so main's
# (empty-on-our-path) tuned-param splat still applies last and never
# collides with our literals.
PN351_LAUNCH_MAIN_OLD = (
    "        CHUNK_SIZE=chunk_size,\n"
    "        USE_TD=use_td,\n"
    "        USE_TD_QO=use_td_qo,\n"
    "        **launch_kwargs,\n"
    "    )\n"
)
PN351_LAUNCH_MAIN_NEW = (
    "        CHUNK_SIZE=chunk_size,\n"
    "        USE_TD=use_td,\n"
    "        USE_TD_QO=use_td_qo,\n"
    "        # [Genesis PN351 vendor of vllm#43257] head_dim >= 512 hits a\n"
    "        # register cliff under default 4w/3s. 8 warps spreads the\n"
    "        # accumulator across more warps; 2-stage pipeline is shallower\n"
    "        # but uses less shmem (offsetting the larger 64-tile). Gemma 4\n"
    "        # 31B + 26B-A4B global heads benefit; Qwen3.6 head_dim=128\n"
    "        # bypasses entirely. (upstream-main anchor variant: inserted\n"
    "        # BEFORE main's **launch_kwargs splat; on head_dim >= 512 that\n"
    "        # dict is empty — main only fills it on its head_size==256\n"
    "        # B200 tuned_large_head path — so no num_warps kwarg conflict.)\n"
    "        num_warps=8 if head_size >= 512 else 4,\n"
    "        num_stages=2 if head_size >= 512 else 3,\n"
    "        **launch_kwargs,\n"
    "    )\n"
)

# Variant C — upstream main AFTER #45151. Derived from `gh pr diff 45151`
# (2026-06-13): the 7 fused-quant kwargs are spliced between USE_TD_QO and
# the existing ``**launch_kwargs,`` line. We re-anchor on the stable
# USE_TD_QO line + the full inserted 7-kwarg block + ``**launch_kwargs,`` +
# the closing `)`, and append our warps/stages kwargs AFTER #45151's
# USE_UE8M0 but BEFORE ``**launch_kwargs,`` so all three fixes coexist. The
# 7-kwarg block makes the anchor uniquely identify the post-#45151 shape.
PN351_LAUNCH_POST45151_OLD = (
    "        CHUNK_SIZE=chunk_size,\n"
    "        USE_TD=use_td,\n"
    "        USE_TD_QO=use_td_qo,\n"
    "        USE_FP8_GROUP=use_fp8_group,\n"
    "        GROUP_SIZE=group_size,\n"
    "        out_group_scale_ptr=output_group_scale,\n"
    "        out_group_scale_stride_0="
    "(output_group_scale.stride(0) if use_fp8_group else 0),\n"
    "        out_group_scale_stride_1="
    "(output_group_scale.stride(1) if use_fp8_group else 1),\n"
    "        NUM_GROUPS_PER_HEAD=num_groups_per_head,\n"
    "        USE_UE8M0=output_group_ue8m0,\n"
    "        **launch_kwargs,\n"
    "    )\n"
)
PN351_LAUNCH_POST45151_NEW = (
    "        CHUNK_SIZE=chunk_size,\n"
    "        USE_TD=use_td,\n"
    "        USE_TD_QO=use_td_qo,\n"
    "        USE_FP8_GROUP=use_fp8_group,\n"
    "        GROUP_SIZE=group_size,\n"
    "        out_group_scale_ptr=output_group_scale,\n"
    "        out_group_scale_stride_0="
    "(output_group_scale.stride(0) if use_fp8_group else 0),\n"
    "        out_group_scale_stride_1="
    "(output_group_scale.stride(1) if use_fp8_group else 1),\n"
    "        NUM_GROUPS_PER_HEAD=num_groups_per_head,\n"
    "        USE_UE8M0=output_group_ue8m0,\n"
    "        # [Genesis PN351 vendor of vllm#43257] head_dim >= 512 hits a\n"
    "        # register cliff under default 4w/3s. 8 warps spreads the\n"
    "        # accumulator across more warps; 2-stage pipeline is shallower\n"
    "        # but uses less shmem (offsetting the larger 64-tile). Gemma 4\n"
    "        # 31B + 26B-A4B global heads benefit; Qwen3.6 head_dim=128\n"
    "        # bypasses entirely. (post-vllm#45151 anchor variant: our\n"
    "        # warps/stages slot AFTER #45151's fused-quant kwargs and\n"
    "        # BEFORE main's **launch_kwargs splat so all three coexist.)\n"
    "        num_warps=8 if head_size >= 512 else 4,\n"
    "        num_stages=2 if head_size >= 512 else 3,\n"
    "        **launch_kwargs,\n"
    "    )\n"
)

# The names of the three mutually-exclusive launch variants. apply()
# requires that AT LEAST ONE of these fired (a tile-only apply is
# incoherent — the tile sub sets the 64-tile, the launch sub sets the
# 8w/2s config; either alone is a functional no-op regression).
_LAUNCH_VARIANT_NAMES = (
    "pn351_kernel_launch_warps_stages",
    "pn351_kernel_launch_warps_stages_main",
    "pn351_kernel_launch_warps_stages_post45151",
)


def _env_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN351", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _build_patcher(target_file: str) -> TextPatcher:
    """Construct the PN351 patcher for a given target path.

    The tile sub-patch is ``required=True`` (its function ``_get_tile_size``
    is untouched by #45151 AND byte-identical on upstream main, so its
    anchor never breaks). The three launch variants are ``required=False``
    — exactly one matches on any given pin, the others soft-skip
    (TextPatcher continues siblings on a False miss). apply() then asserts
    at least one launch variant actually fired.
    """
    return TextPatcher(
        patch_name="PN351 triton_unified_attention head_dim>=512 (vllm#43257)",
        target_file=target_file,
        marker=GENESIS_PN351_MARKER,
        sub_patches=[
            TextPatch(
                name="pn351_get_tile_size_large_head",
                anchor=PN351_TILE_OLD,
                replacement=PN351_TILE_NEW,
                required=True,
            ),
            # Multi-anchor launch sub-patch (required-at-least-one). All
            # three required=False; the all-miss case is caught explicitly
            # in apply() so a tile-only half-apply fails loudly.
            # Variant A — current pin g303916e93 (no **launch_kwargs).
            TextPatch(
                name="pn351_kernel_launch_warps_stages",
                anchor=PN351_LAUNCH_OLD,
                replacement=PN351_LAUNCH_NEW,
                required=False,
            ),
            # Variant B — upstream main pre-#45151 (has **launch_kwargs).
            TextPatch(
                name="pn351_kernel_launch_warps_stages_main",
                anchor=PN351_LAUNCH_MAIN_OLD,
                replacement=PN351_LAUNCH_MAIN_NEW,
                required=False,
            ),
            # Variant C — upstream main post-#45151 (7 kwargs + launch_kwargs).
            TextPatch(
                name="pn351_kernel_launch_warps_stages_post45151",
                anchor=PN351_LAUNCH_POST45151_OLD,
                replacement=PN351_LAUNCH_POST45151_NEW,
                required=False,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN351",
        ],
    )


def _make_patcher() -> TextPatcher | None:
    """Resolve the target file and build the patcher (None if absent)."""
    target = resolve_vllm_file(_TARGET_REL)
    if target is None:
        return None
    return _build_patcher(str(target))


def _make_patcher_for_source(_source: str, target_file: str) -> TextPatcher:
    """Build a patcher pointed at an explicit path (test/lint helper).

    Used by the drift-marker self-collision lint and unit tests that need a
    patcher without resolving the live vllm tree.
    """
    return _build_patcher(target_file)


def apply() -> tuple[str, str]:
    if _env_disabled():
        return "skipped", "PN351 disabled via GENESIS_DISABLE_PN351=1"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", f"PN351: target file {_TARGET_REL} not found"

    try:
        result, failure = patcher.apply()
    except Exception as e:  # noqa: BLE001
        return "failed", f"PN351 apply raised {e!r}"

    if result == TextPatchResult.FAILED:
        return "failed", f"PN351 FAILED — {failure.reason if failure else 'unknown'}"
    if result == TextPatchResult.SKIPPED:
        return "skipped", f"PN351 skipped — {failure.reason if failure else 'unknown'}"
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "PN351 idempotent (already applied)"

    # At-least-one launch variant must have fired. With all launch variants
    # required=False, a future drift that breaks EVERY shape would let the
    # required tile sub apply alone — an incoherent half-patch (64-tile set
    # but 4w/3s launch config retained). Detect it and FAIL loudly rather
    # than report a misleading "applied".
    applied = set(patcher.applied_sub_patches)
    launch_applied = applied.intersection(_LAUNCH_VARIANT_NAMES)
    if not launch_applied:
        return "failed", (
            "PN351 FAILED — tile sub-patch applied but NONE of the launch "
            "anchor variants matched (current-pin / upstream-main / "
            "post-#45151). The launch config (8w/2s) is the load-bearing "
            "half; a tile-only apply is a no-op regression. Anchor drift past "
            "a NEW pin shape — re-derive the kernel_unified_attention launch "
            "anchor."
        )

    variant = next(iter(launch_applied))
    if variant.endswith("post45151"):
        variant_note = "post-#45151 anchor variant"
    elif variant.endswith("_main"):
        variant_note = "upstream-main anchor variant"
    else:
        variant_note = "current-pin anchor variant"
    n = len(patcher.applied_sub_patches)
    return "applied", (
        f"PN351 applied: {n} sub-patches on triton_unified_attention.py "
        f"({variant_note}) — head_dim >= 512 prefill (Gemma 4 31B + 26B-A4B "
        f"global heads) now uses 64-tile + 8w/2s. Expected -3-7% decode_TPOT "
        f"on Gemma 4 31B FP8. No-op on Qwen3.6 (head_dim=128). Vendor of OPEN "
        f"PR vllm#43257; multi-anchor survives the upstream-main launch_kwargs "
        f"refactor and the vllm#45151 launch-kwarg insertion."
    )


def is_applied() -> bool:
    target = resolve_vllm_file(_TARGET_REL)
    if target is None:
        return False
    try:
        return GENESIS_PN351_MARKER in Path(str(target)).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
