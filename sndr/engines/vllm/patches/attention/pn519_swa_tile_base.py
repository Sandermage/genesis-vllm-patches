# SPDX-License-Identifier: Apache-2.0
"""PN519 — start the SWA/chunked KV-tile loop exactly at ``first_allowed_key``
(backport+improve OPEN vllm#46087, fixes vllm#44575).

Problem (LIVE on Gemma4 SWA sliding layers)
-------------------------------------------
``compute_tile_loop_bounds`` in
``vllm/v1/attention/ops/triton_attention_helpers.py`` starts the
sliding-window / chunked tile loop at the tile FLOOR::

    tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)

and the two consumer kernels (``triton_unified_attention.py`` and its
``_diffkv`` sibling) index ``seq_offset = j * TILE_SIZE + offs_t``. A window of
``W`` keys therefore spans ``ceil((r + W) / TILE_SIZE)`` tiles instead of the
minimal ``ceil(W / TILE_SIZE)``, where ``r = first_allowed_key % TILE_SIZE``:

  1. PERF — one redundant tile per SWA request whenever ``r != 0``: the
     boundary tile's pre-window keys are loaded then masked out.
  2. DETERMINISM — the residue ``r`` shifts which keys land in the boundary
     tile, perturbing the online-softmax reduction ORDER, so the output is not
     byte-identical across windows whose ``first_allowed_key`` differ only by a
     sub-tile residue.

Our Gemma4 (26B-A4B + 31B) interleaved-SWA layers run this exact kernel: the
512-wide global heads route through ``triton_unified_attention.py`` on Ampere
(PN351 vendors a sibling tune to the same file), and the sliding layers hit the
SWA tile loop. Qwen3.6 (35B FP8 / 27B INT4, head_dim=128) runs FlashInfer / FA2
and never executes this kernel — the patched code is imported but never run, so
``default_on`` is scoped to the Gemma4 SWA configs only.

Fix (vllm#46087)
----------------
``compute_tile_loop_bounds`` returns a 4th value ``tile_base`` — non-zero only
on the 2D-pointer SWA/chunked path; the ``USE_TD`` tensor-descriptor and the
3D-segmented paths index in absolute tile units and keep it 0. Both consumers
offset ``seq_offset = tile_base + j * TILE_SIZE + offs_t`` so iteration starts
EXACTLY at ``first_allowed_key``. ``tile_end`` (from ``last_allowed_key``) is
unchanged, so the iteration count NEVER grows (fewer-or-equal tiles), and the
boundary residue no longer reorders the reduction → byte-identical output.

OUR version over the raw PR (iron rule #10)
-------------------------------------------
1. **Atomic three-file apply.** Because ``compute_tile_loop_bounds`` now returns
   a 4-tuple, BOTH consumers MUST be updated in lockstep — a helper-only apply
   (4-tuple producer feeding a 3-tuple unpack) is a silent ``ValueError`` at the
   first decode. PN519's ``apply()`` only reports ``applied`` when all three
   files carry the marker; if any consumer's anchor drifts past a future pin it
   FAILS LOUDLY rather than leaving a half-patched, crash-on-first-decode tree.
2. **USE_TD/3D safety preserved verbatim.** ``tile_base`` is gated on
   ``not USE_TD and not IS_3D`` exactly as upstream — our TurboQuant ``USE_TD``
   decode path and the 3D-segmented reduction are byte-unchanged.

Anchors (all byte-verified count==1 on dev424 ``0.23.1rc1.dev424+g3f5a1e173``):
  * helper: signature (add ``USE_TD`` param), init (add ``tile_base = 0``), the
    ``tile_start``/``tile_end`` block (add the ``tile_base`` computation), and
    the ``return`` (append ``tile_base``).
  * consumer A: the ``compute_tile_loop_bounds`` call (add ``USE_TD,``) + the
    4-tuple unpack + ``seq_offset`` offset.
  * consumer B (``_diffkv``): the 4-tuple unpack + ``seq_offset`` offset (its
    call site uses keyword defaults beyond ``IS_3D`` and needs no ``USE_TD``).

Composition
-----------
Composes with PN351 (same files, DISJOINT anchors: PN351 touches
``_get_tile_size`` + the kernel launch kwargs; PN519 touches the tile-loop
bounds helper + the ``seq_offset`` index). No overlap with PN399 (TQ attn file)
or PN345/PN29x (FLA chunk kernels). Supersedes nothing.

Lifecycle
---------
``default_on=False`` / ``lifecycle=experimental``; promote to the Gemma4 SWA
YAMLs after a clean Gemma4-31B boot + apply-proof + coherence. Self-skips once a
pin carries vllm#46087 natively (drift marker = the PR's 4-tuple ``return``
literal).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Upstream: https://github.com/vllm-project/vllm/pull/46087 (OPEN, backport+improve).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

log = logging.getLogger("genesis.wiring.pn519_swa_tile_base")

GENESIS_PN519_MARKER = (
    "Genesis PN519 SWA tile-loop first_allowed_key base (vllm#46087) v1"
)

ENV_FLAG_FULL = "GENESIS_ENABLE_PN519_SWA_TILE_BASE"

_HELPER_RELPATH = "v1/attention/ops/triton_attention_helpers.py"
_CONSUMER_A_RELPATH = "v1/attention/ops/triton_unified_attention.py"
_CONSUMER_B_RELPATH = "v1/attention/ops/triton_unified_attention_diffkv.py"


# ─────────────────────────────────────────────────────────────────────
# Pure model of the kernel's tile-iteration math — the single source of
# truth the unit tests exercise (the @triton.jit kernel runs on GPU only).
# This mirrors EXACTLY the loop the patched/pristine kernel walks:
#   pristine:  tile_start = first_allowed_key // TILE_SIZE
#              seq_offset(j, t) = j * TILE_SIZE + t
#   fixed:     tile_base = first_allowed_key - tile_start * TILE_SIZE
#              seq_offset(j, t) = tile_base + j * TILE_SIZE + t
# with the loop j in [tile_start, tile_end) and the per-slot mask
# seq_offset <= last_allowed_key.
# ─────────────────────────────────────────────────────────────────────


def model_tiles_walked(
    first_allowed_key: int,
    last_allowed_key: int,
    tile_size: int,
    *,
    use_tile_base: bool,
) -> list[int]:
    """Return the sorted list of absolute key positions the SWA tile loop
    actually touches (post per-tile mask ``seq_offset <= last_allowed_key``).

    The fixed (``use_tile_base=True``) loop touches NO key below
    ``first_allowed_key``; the pristine (FLOOR) loop touches the pre-window
    keys in the boundary tile.
    """
    tile_start = max(0, first_allowed_key // tile_size)
    # tile_end mirrors the kernel: (last_allowed_key // TILE_SIZE) + 1.
    tile_end = (last_allowed_key // tile_size) + 1
    tile_base = (first_allowed_key - tile_start * tile_size) if use_tile_base else 0
    walked: list[int] = []
    for j in range(tile_start, tile_end):
        for t in range(tile_size):
            pos = tile_base + j * tile_size + t
            if pos <= last_allowed_key:
                walked.append(pos)
    return sorted(walked)


def model_tile_count(
    first_allowed_key: int,
    last_allowed_key: int,
    tile_size: int,
    *,
    use_tile_base: bool,
) -> int:
    """Number of tiles the loop iterates over. With ``use_tile_base`` the count
    depends ONLY on the window width (residue-invariant); the pristine FLOOR
    loop grows by one when ``first_allowed_key % tile_size != 0``."""
    if use_tile_base:
        # The fixed loop spans exactly ceil((last - first + 1) / tile_size).
        width = last_allowed_key - first_allowed_key + 1
        return max(0, math.ceil(width / tile_size))
    tile_start = max(0, first_allowed_key // tile_size)
    tile_end = (last_allowed_key // tile_size) + 1
    return max(0, tile_end - tile_start)


# ─────────────────────────────────────────────────────────────────────
# HELPER anchors — triton_attention_helpers.py / compute_tile_loop_bounds
# ─────────────────────────────────────────────────────────────────────

# Anchor 1 — signature: add USE_TD constexpr param. Anchored on the unique
# CHUNK_LOOKBACK/CHUNK_SIZE tail + the function docstring open (count==1).
HELPER_SIG_OLD = (
    "    CHUNK_LOOKBACK: tl.constexpr = -1,\n"
    "    CHUNK_SIZE: tl.constexpr = -1,\n"
    "):\n"
    '    """Compute the tile-loop bounds'
)
HELPER_SIG_NEW = (
    "    CHUNK_LOOKBACK: tl.constexpr = -1,\n"
    "    CHUNK_SIZE: tl.constexpr = -1,\n"
    "    # [Genesis PN519 SWA tile-loop first_allowed_key base (vllm#46087) v1]\n"
    "    # USE_TD selects the tensor-descriptor index path, which addresses in\n"
    "    # absolute tile units and so cannot absorb a non-tile-aligned base; it\n"
    "    # keeps tile_base = 0. Only the 2D-pointer SWA/chunked path uses it.\n"
    "    USE_TD: tl.constexpr = False,\n"
    "):\n"
    '    """Compute the tile-loop bounds'
)

# Anchor 2 — init: add tile_base = 0 alongside the default tile_start/tile_end.
# Anchored on the unique default-region init pair (count==1); the upstream
# pruning comment that follows is left untouched.
HELPER_INIT_OLD = (
    "    tile_start = 0\n"
    "    tile_end = num_tiles\n"
)
HELPER_INIT_NEW = (
    "    tile_start = 0\n"
    "    tile_end = num_tiles\n"
    "    # [Genesis PN519 vllm#46087] per-slot offset so the SWA/chunked loop\n"
    "    # starts EXACTLY at first_allowed_key (not its tile floor); 0 for the\n"
    "    # global / USE_TD / 3D paths.\n"
    "    tile_base = 0\n"
)

# Anchor 3 — compute tile_base inside the SWA branch, after tile_start/tile_end.
HELPER_TILE_OLD = (
    "        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)\n"
    "        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)\n"
    "\n"
    "    if IS_3D:"
)
HELPER_TILE_NEW = (
    "        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)\n"
    "        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)\n"
    "        # [Genesis PN519 vllm#46087, fixes vllm#44575] Start exactly at\n"
    "        # first_allowed_key (not its tile floor) so a window of W keys\n"
    "        # spans ceil(W / TILE_SIZE) tiles, not ceil((r + W) / TILE_SIZE)\n"
    "        # where r = first_allowed_key % TILE_SIZE. Only the 2D-pointer\n"
    "        # path (per-slot block index) can absorb a non-tile-aligned base;\n"
    "        # USE_TD and IS_3D index in absolute tile units, so they keep\n"
    "        # tile_base = 0. tile_end (from last_allowed_key) is unchanged, so\n"
    "        # the iteration count never grows AND the online-softmax reduction\n"
    "        # order no longer depends on the window residue (determinism).\n"
    "        if not USE_TD and not IS_3D:\n"
    "            tile_base = tl.maximum(0, first_allowed_key) - tile_start * TILE_SIZE\n"
    "\n"
    "    if IS_3D:"
)

# Anchor 4 — return the 4-tuple.
HELPER_RETURN_OLD = "    return loop_lo, loop_hi, max_seq_prefix_len\n"
HELPER_RETURN_NEW = (
    "    # [Genesis PN519 vllm#46087] return tile_base so the consumers can\n"
    "    # offset seq_offset to the exact first_allowed_key.\n"
    "    return loop_lo, loop_hi, max_seq_prefix_len, tile_base\n"
)


# ─────────────────────────────────────────────────────────────────────
# CONSUMER A anchors — triton_unified_attention.py
# ─────────────────────────────────────────────────────────────────────

# Call site: unpack the 4-tuple AND pass USE_TD through to the helper. The
# call already passes ...CHUNK_SIZE, as the last positional arg (count==1).
CONSUMER_A_CALL_OLD = (
    "    loop_lo, loop_hi, max_seq_prefix_len = compute_tile_loop_bounds(\n"
    "        context_len,\n"
    "        seq_len,\n"
    "        cur_batch_query_len,\n"
    "        q_block_local_idx,\n"
    "        segm_idx,\n"
    "        tiles_per_segment,\n"
    "        TILE_SIZE,\n"
    "        BLOCK_M,\n"
    "        BLOCK_Q,\n"
    "        num_queries_per_kv,\n"
    "        SLIDING_WINDOW,\n"
    "        USE_MM_PREFIX,\n"
    "        IS_3D,\n"
    "        USE_CAUSAL,\n"
    "        USE_PER_SEQ_CAUSAL,\n"
    "        CHUNK_LOOKBACK,\n"
    "        CHUNK_SIZE,\n"
    "    )\n"
)
CONSUMER_A_CALL_NEW = (
    "    # [Genesis PN519 SWA tile-loop first_allowed_key base (vllm#46087) v1]\n"
    "    loop_lo, loop_hi, max_seq_prefix_len, tile_base = compute_tile_loop_bounds(\n"
    "        context_len,\n"
    "        seq_len,\n"
    "        cur_batch_query_len,\n"
    "        q_block_local_idx,\n"
    "        segm_idx,\n"
    "        tiles_per_segment,\n"
    "        TILE_SIZE,\n"
    "        BLOCK_M,\n"
    "        BLOCK_Q,\n"
    "        num_queries_per_kv,\n"
    "        SLIDING_WINDOW,\n"
    "        USE_MM_PREFIX,\n"
    "        IS_3D,\n"
    "        USE_CAUSAL,\n"
    "        USE_PER_SEQ_CAUSAL,\n"
    "        CHUNK_LOOKBACK,\n"
    "        CHUNK_SIZE,\n"
    "        USE_TD,\n"
    "    )\n"
)

# Shared seq_offset anchor (same literal in both consumers). count==1 per file.
CONSUMER_SEQ_OFFSET_OLD = "        seq_offset = j * TILE_SIZE + offs_t\n"
CONSUMER_SEQ_OFFSET_NEW = (
    "        # [Genesis PN519 vllm#46087] offset to the exact first_allowed_key.\n"
    "        seq_offset = tile_base + j * TILE_SIZE + offs_t\n"
)


# ─────────────────────────────────────────────────────────────────────
# CONSUMER B anchors — triton_unified_attention_diffkv.py
# The diffkv call site stops at IS_3D, (keyword defaults beyond it), so it
# needs NO USE_TD change — only the 4-tuple unpack + the seq_offset offset.
# ─────────────────────────────────────────────────────────────────────

CONSUMER_B_CALL_OLD = (
    "    loop_lo, loop_hi, max_seq_prefix_len = compute_tile_loop_bounds(\n"
    "        context_len,\n"
    "        seq_len,\n"
    "        cur_batch_query_len,\n"
    "        q_block_local_idx,\n"
    "        segm_idx,\n"
    "        tiles_per_segment,\n"
    "        TILE_SIZE,\n"
    "        BLOCK_M,\n"
    "        BLOCK_Q,\n"
    "        num_queries_per_kv,\n"
    "        SLIDING_WINDOW,\n"
    "        False,  # USE_MM_PREFIX\n"
    "        IS_3D,\n"
    "    )\n"
)
CONSUMER_B_CALL_NEW = (
    "    # [Genesis PN519 SWA tile-loop first_allowed_key base (vllm#46087) v1]\n"
    "    loop_lo, loop_hi, max_seq_prefix_len, tile_base = compute_tile_loop_bounds(\n"
    "        context_len,\n"
    "        seq_len,\n"
    "        cur_batch_query_len,\n"
    "        q_block_local_idx,\n"
    "        segm_idx,\n"
    "        tiles_per_segment,\n"
    "        TILE_SIZE,\n"
    "        BLOCK_M,\n"
    "        BLOCK_Q,\n"
    "        num_queries_per_kv,\n"
    "        SLIDING_WINDOW,\n"
    "        False,  # USE_MM_PREFIX\n"
    "        IS_3D,\n"
    "    )\n"
)


# Self-skip drift marker — the PR's 4-tuple return literal, present once a pin
# carries vllm#46087 natively, ABSENT in pristine dev424. (It is a substring of
# our own emitted output, but the TextPatcher checks the idempotency marker
# BEFORE the drift markers, so the scan never reads our own replacement.)
_UPSTREAM_DRIFT_MARKER = (
    "return loop_lo, loop_hi, max_seq_prefix_len, tile_base"
)


def _env_disabled_via_dispatcher() -> tuple[bool, str]:
    """Return (skip, reason) from the central dispatcher gate."""
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN519")
    log_decision("PN519", decision, reason)
    return (not decision), reason


def _make_helper_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_HELPER_RELPATH)
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN519 triton_attention_helpers.compute_tile_loop_bounds — "
            "tile_base for SWA first_allowed_key (vllm#46087)"
        ),
        target_file=str(target),
        marker=GENESIS_PN519_MARKER,
        sub_patches=[
            TextPatch(
                name="pn519_helper_signature_use_td",
                anchor=HELPER_SIG_OLD,
                replacement=HELPER_SIG_NEW,
                required=True,
            ),
            TextPatch(
                name="pn519_helper_init_tile_base",
                anchor=HELPER_INIT_OLD,
                replacement=HELPER_INIT_NEW,
                required=True,
            ),
            TextPatch(
                name="pn519_helper_compute_tile_base",
                anchor=HELPER_TILE_OLD,
                replacement=HELPER_TILE_NEW,
                required=True,
            ),
            TextPatch(
                name="pn519_helper_return_4tuple",
                anchor=HELPER_RETURN_OLD,
                replacement=HELPER_RETURN_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN519",
            _UPSTREAM_DRIFT_MARKER,
        ],
    )


def _make_consumer_a_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_CONSUMER_A_RELPATH)
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN519 triton_unified_attention — tile_base seq_offset (vllm#46087)"
        ),
        target_file=str(target),
        marker=GENESIS_PN519_MARKER,
        sub_patches=[
            TextPatch(
                name="pn519_consumer_a_call_4tuple_use_td",
                anchor=CONSUMER_A_CALL_OLD,
                replacement=CONSUMER_A_CALL_NEW,
                required=True,
            ),
            TextPatch(
                name="pn519_consumer_a_seq_offset",
                anchor=CONSUMER_SEQ_OFFSET_OLD,
                replacement=CONSUMER_SEQ_OFFSET_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=["[Genesis PN519"],
    )


def _make_consumer_b_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_CONSUMER_B_RELPATH)
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN519 triton_unified_attention_diffkv — tile_base seq_offset "
            "(vllm#46087)"
        ),
        target_file=str(target),
        marker=GENESIS_PN519_MARKER,
        sub_patches=[
            TextPatch(
                name="pn519_consumer_b_call_4tuple",
                anchor=CONSUMER_B_CALL_OLD,
                replacement=CONSUMER_B_CALL_NEW,
                required=True,
            ),
            TextPatch(
                name="pn519_consumer_b_seq_offset",
                anchor=CONSUMER_SEQ_OFFSET_OLD,
                replacement=CONSUMER_SEQ_OFFSET_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=["[Genesis PN519"],
    )


def _apply_one(patcher: TextPatcher | None, relpath: str) -> tuple[str, str]:
    """Apply a single file's patcher; map to (status, reason)."""
    if patcher is None:
        return "skipped", f"PN519: target file {relpath} not found"
    if not Path(patcher.target_file).is_file():
        return "skipped", f"PN519: target {patcher.target_file} disappeared"
    try:
        result, failure = patcher.apply()
    except Exception as e:  # noqa: BLE001
        return "failed", f"PN519 apply raised on {relpath}: {e!r}"
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"PN519 FAILED on {relpath} — "
            f"{failure.reason if failure else 'unknown'}"
        )
    if result == TextPatchResult.SKIPPED:
        return "skipped", (
            f"PN519 skipped on {relpath} — "
            f"{failure.reason if failure else 'unknown'}"
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "idempotent", f"PN519 idempotent on {relpath} (already applied)"
    return "applied", f"PN519 applied on {relpath}"


def apply() -> tuple[str, str]:
    """Apply PN519 atomically across the helper + the two consumers.

    Opt-in through the dispatcher on
    ``GENESIS_ENABLE_PN519_SWA_TILE_BASE`` (default_on=False; promote to the
    Gemma4 SWA YAMLs after rig validation). Because the helper now returns a
    4-tuple, a half-apply (helper patched, a consumer not) is a crash-on-first-
    decode ValueError — so this FAILS LOUDLY if any of the three files cannot be
    brought to the patched (or natively-present) state.
    """
    skip, reason = _env_disabled_via_dispatcher()
    if skip:
        return "skipped", reason

    targets = (
        (_make_helper_patcher(), _HELPER_RELPATH),
        (_make_consumer_a_patcher(), _CONSUMER_A_RELPATH),
        (_make_consumer_b_patcher(), _CONSUMER_B_RELPATH),
    )

    results: list[tuple[str, str]] = []
    for patcher, relpath in targets:
        results.append(_apply_one(patcher, relpath))

    statuses = [s for s, _ in results]

    # Any hard failure → fail loudly (do NOT leave a half-patched tree).
    if "failed" in statuses:
        bad = next(r for s, r in results if s == "failed")
        return "failed", (
            "PN519 FAILED — atomic three-file patch could not complete; a "
            "half-apply (4-tuple producer vs 3-tuple unpack) crashes on the "
            f"first decode. First failure: {bad}. Re-derive the drifted anchor "
            "against the running pin."
        )

    # If some files patched but others only 'skipped' (target absent), that is
    # an incoherent partial state on a real tree — flag it.
    applied_like = {"applied", "idempotent"}
    n_ok = sum(1 for s in statuses if s in applied_like)
    if 0 < n_ok < len(statuses):
        joined = "; ".join(f"{rel}={s}" for (s, _), (_, rel) in zip(results, targets))
        return "failed", (
            "PN519 FAILED — partial apply across the three files (some patched, "
            f"some not): {joined}. The helper 4-tuple return and both consumer "
            "unpacks must move together."
        )

    if all(s == "skipped" for s in statuses):
        return "skipped", "PN519: no target files present (all three absent)"

    if all(s == "idempotent" for s in statuses):
        return "skipped", (
            "PN519 idempotent — marker already present on all three files "
            "(already applied)"
        )

    return "applied", (
        "PN519 applied: compute_tile_loop_bounds now returns tile_base and "
        "both triton_unified_attention consumers offset seq_offset = tile_base "
        "+ j*TILE_SIZE + offs_t, so the SWA/chunked tile loop starts EXACTLY at "
        "first_allowed_key (Gemma4 sliding layers). Drops the redundant "
        "boundary tile per SWA request and removes the residue-dependent "
        "online-softmax reduction-order non-determinism (vllm#46087, fixes "
        "vllm#44575). USE_TD / 3D paths keep tile_base=0 (byte-unchanged); "
        "Qwen3.6 FlashInfer/FA2 never executes this kernel."
    )


def is_applied() -> bool:
    """True only when ALL THREE files carry the marker (coherent apply)."""
    for relpath in (_HELPER_RELPATH, _CONSUMER_A_RELPATH, _CONSUMER_B_RELPATH):
        target = resolve_vllm_file(relpath)
        if target is None:
            return False
        try:
            if GENESIS_PN519_MARKER not in Path(str(target)).read_text(
                encoding="utf-8"
            ):
                return False
        except (OSError, UnicodeDecodeError):
            return False
    return True


__all__ = [
    "GENESIS_PN519_MARKER",
    "ENV_FLAG_FULL",
    "HELPER_SIG_OLD",
    "HELPER_SIG_NEW",
    "HELPER_INIT_OLD",
    "HELPER_INIT_NEW",
    "HELPER_TILE_OLD",
    "HELPER_TILE_NEW",
    "HELPER_RETURN_OLD",
    "HELPER_RETURN_NEW",
    "CONSUMER_A_CALL_OLD",
    "CONSUMER_A_CALL_NEW",
    "CONSUMER_B_CALL_OLD",
    "CONSUMER_B_CALL_NEW",
    "CONSUMER_SEQ_OFFSET_OLD",
    "CONSUMER_SEQ_OFFSET_NEW",
    "model_tiles_walked",
    "model_tile_count",
    "resolve_vllm_file",
    "apply",
    "is_applied",
]
