# SPDX-License-Identifier: Apache-2.0
"""PN399 — TurboQuant decode-scratch fixed buffer (fix CUDA IMA in FULL cudagraph).

Genesis backport of OPEN vllm#46067 ([Bugfix][TurboQuant] Fix CUDA
illegal memory access in FULL cudagraph), re-authored against live
dev148 (pin ``0.23.1rc1.dev148+gb4c80ec0f``) and COMPOSED with PN118.

Problem
-------
TQ decode runs inside the FULL cudagraph, so the decode-scratch tensor
addresses are baked into the captured graphs at capture time. The
growable ``WorkspaceManager`` is NOT safe to back that scratch: it
``free``s + reallocs (it calls ``empty_cache()``, unmapping the old
address) whenever it has to grow — across the ``B=1..max`` capture sweep
or on a long continuation-prefill — freeing an address that an earlier
(smaller-batch) captured graph still points at. First replay of that
graph then triggers a CUDA illegal memory access.

Fix
---
Allocate a fixed module-level ``_DECODE_SCRATCH`` ONCE at the largest
CUDA-graph decode batch (``max_cudagraph_capture_size``), reused by every
TQ layer (layers run sequentially) and every captured decode graph,
sliced ``[:B]`` per call (native kernel contract:
``triton_turboquant_decode.py`` slices ``mid_o_buf[:B]`` etc. under
``.shape[0] >= B``). A dedicated buffer sized up front for ``max_batch``
never moves, so the addresses baked into the FULL cudagraphs stay valid.

Consolidated single-owner design (compose + de-dup, NOT retire)
---------------------------------------------------------------
PN399 OWNS the TQ decode-scratch lifecycle. It anchors the LIVE (PN118 +
PN353A patched) output and transforms it — the PN118/PN353A SOURCE files
are NOT edited (only registry composes_with notes). Five sub-patches:

  * A  — module-level ``_DECODE_SCRATCH`` dict + ``_get_decode_scratch``
    + ``reset_tq_decode_scratch`` after the live module constants.
  * B' — ``__init__``: INSERT ``self.max_decode_cudagraph_batch`` AND
    REMOVE the now-dead PN118 ``__init__`` ``_reserve_decode_workspace``
    box + call + method (one consolidated hunk over live 381-426).
  * C  — ``_decode_attention``: insert the CG-path ``_get_decode_scratch``
    branch BEFORE PN118's ``if is_workspace_manager_initialized():`` and
    convert that ``if`` to ``elif``. PN118's ``try_get_simultaneous`` +
    ``torch.empty`` body is left BYTE-UNCHANGED as the eager / over-max
    cold path (the safety net for ``enforce_eager`` / ``B > max_batch``).
  * C2 — ``TurboQuantMetadataBuilder._reserve_workspace``: REMOVE the
    now-dead PN353A decode-scratch ``get_simultaneous`` reservation. The
    PN353A CONTINUATION-PREFILL K/V reservation is KEPT BYTE-INTACT —
    PN399 never touches the prefill path (proven distinct call sites on
    live bytes: separated by the ``reserve_cont`` gate + early return).
  * D  — ``gpu/shutdown.py``: ``reset_tq_decode_scratch`` import + call.

Why removals B'/C2 are safe: the PN118/PN353A decode reservations only
pre-GREW the WorkspaceManager so a later lazy ``get_simultaneous`` would
not hit "locked + undersized". Once ``_DECODE_SCRATCH`` owns the CG-path
fixed buffer (which never grows or locks), that pre-grow is dead on the
hot path. The cold path (eager / ``B > max_batch``) still works: PN118's
``try_get_simultaneous`` returns ``None`` on an undersized workspace ->
the ``torch.empty`` one-shot fires — PN118's designed behavior, WITHOUT
the boot reserve. This single-owner consolidation is BETTER than upstream
#46067, which has neither PN118 nor PN353A to de-duplicate.

With PN399 OFF/unapplied it produces ZERO change: PN118/PN353A keep
owning decode + their reservations (the current crash-free PROD
behavior). This is why PN399 ``requires_patches: [PN118, PN353A]`` and is
placed in the registry AFTER both (so the PN118 ``__init__`` box + decode
head and the PN353A reserve block — which the anchors target — exist as
applied output when PN399 runs). If either is disabled, the IMA defense
is also off, so an operator running PN399 must keep both on; the
dependent sub-patches then SKIP cleanly (required anchor missing).

Lifecycle
---------
``default_on=False``, ``lifecycle=experimental``. On our PROD (35B FP8 +
MTP K=3 + TQ k8v4, capture set [4,8,16], max_batch=16) the live IMA is
ALREADY neutralized by PN118 + PN353A + SNDR_WORKSPACE_001 (5h clean), so
PN399 is a belt-and-suspenders backport pending rig validation at the
next pin-upgrade window — not yet promoted to default_on.

Self-skip
---------
``_get_decode_scratch`` and ``max_decode_cudagraph_batch`` are the
``upstream_drift_markers``: once a pin carries #46067 natively (these
symbols present), PN399 self-skips before touching the file.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Upstream: https://github.com/vllm-project/vllm/pull/46067 (OPEN, backport).
"""
from __future__ import annotations

import logging

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.env import Flags, is_enabled
from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

log = logging.getLogger("genesis.wiring.pn399_tq_decode_scratch_ima")

GENESIS_PN399_MARKER = "Genesis PN399 vllm#46067 v1"

# Full env var name (for tests / operator docs); the canonical bare flag
# lives in sndr.env.Flags.PN399_TQ_DECODE_SCRATCH_IMA.
ENV_FLAG_FULL = "GENESIS_ENABLE_PN399_TQ_DECODE_SCRATCH_IMA"

_TQ_RELPATH = "v1/attention/backends/turboquant_attn.py"
_SHUTDOWN_RELPATH = "v1/worker/gpu/shutdown.py"


# ─────────────────────────────────────────────────────────────────────
# FILE 1 — turboquant_attn.py — 3 anchors (A const defs, B __init__ attr,
# C decode CG-branch wrap of PN118's live output).
# ─────────────────────────────────────────────────────────────────────

# Sub-patch A — module-level _DECODE_SCRATCH / _get_decode_scratch /
# reset_tq_decode_scratch, inserted AFTER the true last module constant.
# Anchored on BOTH live constant lines (live dev148 81-82): the PR keys
# off `_CONTINUATION_DECODE_THRESHOLD = 128` + 2 blank lines, but dev148
# has `= 64` (P101) PLUS a sibling `_CONTINUATION_DECODE_MAX_CACHED_LEN
# = 32768` immediately after (no blank between). Anchoring both lands the
# insert after the real last module constant, not mid-block.
# DEPENDENCY (registry requires_patches=[..., "P101"]): this anchor IS
# P101's APPLIED output — pristine has `= 128` and NO MAX_CACHED_LEN, so
# with P101 off this const sub-patch skips. Same anchors-the-LIVE-applied-
# output edge as the PN118/PN353A requirements below.
TQ_ANCHOR_CONST_OLD = (
    "_CONTINUATION_DECODE_THRESHOLD = 64\n"
    "_CONTINUATION_DECODE_MAX_CACHED_LEN = 32768\n"
)

TQ_ANCHOR_CONST_NEW = (
    "_CONTINUATION_DECODE_THRESHOLD = 64\n"
    "_CONTINUATION_DECODE_MAX_CACHED_LEN = 32768\n"
    "\n"
    "\n"
    "# ════════════════════════════════════════════════════════════════════\n"
    "# [Genesis PN399 — backport of vllm#46067]\n"
    "# Shared, fixed-size TQ decode scratch keyed by\n"
    "# (max_batch, num_heads, num_kv_splits, head_size, dtype, device).\n"
    "# Allocated ONCE at the largest CUDA-graph decode batch and reused by\n"
    "# every TQ layer (layers run sequentially) and every captured decode\n"
    "# graph. Deliberately NOT the growable WorkspaceManager: TQ decode runs\n"
    "# inside the FULL cudagraph, so scratch addresses are baked into the\n"
    "# captured graphs at capture time. WorkspaceManager free+reallocs (it\n"
    "# calls empty_cache(), unmapping the old address) when it grows across\n"
    "# the B=1..max capture sweep / on a long continuation-prefill, freeing\n"
    "# an address an earlier (smaller-batch) graph still points at -> first\n"
    "# replay -> CUDA illegal memory access. A dedicated buffer sized up\n"
    "# front for max_batch never moves, so graph-baked addresses stay valid.\n"
    "# Supersedes PN118/PN353A decode-scratch reservation on the CG path.\n"
    "# ════════════════════════════════════════════════════════════════════\n"
    "_DECODE_SCRATCH: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}\n"
    "\n"
    "\n"
    "def _get_decode_scratch(\n"
    "    max_batch: int,\n"
    "    num_heads: int,\n"
    "    num_kv_splits: int,\n"
    "    head_size: int,\n"
    "    dtype: torch.dtype,\n"
    "    device: torch.device,\n"
    ") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:\n"
    '    """Fixed decode scratch (mid_o, output, lse) sized for the largest batch.\n'
    "\n"
    "    Shared across all TQ layers and all captured decode graphs so the\n"
    "    addresses baked into FULL cudagraphs never become stale. Callers\n"
    "    slice ``[:B]`` (native kernel contract: triton_turboquant_decode.py\n"
    "    slices mid_o_buf[:B], output_buf[:B], lse_buf[:B] under .shape[0]>=B).\n"
    '    """\n'
    "    key = (max_batch, num_heads, num_kv_splits, head_size, dtype, device)\n"
    "    bufs = _DECODE_SCRATCH.get(key)\n"
    "    if bufs is None:\n"
    "        bufs = (\n"
    "            torch.empty(\n"
    "                max_batch,\n"
    "                num_heads,\n"
    "                num_kv_splits,\n"
    "                head_size + 1,\n"
    "                dtype=torch.float32,\n"
    "                device=device,\n"
    "            ),\n"
    "            torch.empty(max_batch, num_heads, head_size, dtype=dtype, device=device),\n"
    "            torch.empty(max_batch, num_heads, dtype=torch.float32, device=device),\n"
    "        )\n"
    "        _DECODE_SCRATCH[key] = bufs\n"
    "    return bufs\n"
    "\n"
    "\n"
    "def reset_tq_decode_scratch() -> None:\n"
    '    """Release the shared decode scratch (called on model-runner teardown)."""\n'
    "    _DECODE_SCRATCH.clear()\n"
)


# Sub-patch B' — CONSOLIDATED __init__ re-author. Two effects in one
# anchor (the kv_splits assignment + the ENTIRE live PN118 __init__
# reserve box + call + _reserve_decode_workspace method, live dev148
# 381-426):
#   (1) INSERT self.max_decode_cudagraph_batch right after the kv_splits
#       assignment (the PR's __init__ attr).
#   (2) REMOVE the now-dead PN118 __init__ decode-workspace reservation:
#       the box comment, the self._reserve_decode_workspace(vllm_config)
#       call, AND the def _reserve_decode_workspace(...) method body. Once
#       _DECODE_SCRATCH owns the CG-path fixed buffer the pre-grow of the
#       WorkspaceManager to decode rows before lock is dead weight (cut to
#       reclaim boot overhead). The cold-path safety net is unaffected:
#       PN118's try_get_simultaneous + torch.empty in _decode_attention
#       (the eager elif from sub-patch C) lands on try_get -> None ->
#       torch.empty when the WorkspaceManager is undersized, exactly
#       PN118's designed behavior, WITHOUT needing the boot-time reserve.
# This single re-authored hunk avoids two sub-patches fighting over the
# overlapping box lines 385-386 (sub-patch B insertion point vs the
# removal region). Byte-confirmed on live dev148 (md5
# e62752610c41d2a691d19c5aa4edda59): box-rule = 60x U+2550, em-dash
# U+2014, anchor count = 1.
TQ_ANCHOR_INIT_OLD = (
    "        self.max_num_kv_splits = (\n"
    "            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph\n"
    "        )\n"
    "\n"
    "        # ════════════════════════════════════════════════════════════\n"
    "        # [Genesis PN118 — backport of vllm#42551]\n"
    "        # Pre-reserve decode scratch buffers so lock_workspace() at\n"
    "        # end of warmup snapshots a workspace large enough for steady-\n"
    "        # state decode. Without this, models whose warmup never lands\n"
    "        # a decode forward through TQ (e.g. dense + hybrid attention,\n"
    "        # 16/64 TQ layers in Lorbus 27B AutoRound) leave the workspace\n"
    "        # locked at 0 MB and the first decode falls back to torch.empty\n"
    "        # every layer/forward.\n"
    "        # ════════════════════════════════════════════════════════════\n"
    "        self._reserve_decode_workspace(vllm_config)\n"
    "\n"
    "    def _reserve_decode_workspace(self, vllm_config) -> None:\n"
    "        if not is_workspace_manager_initialized():\n"
    "            return\n"
    "        manager = current_workspace_manager()\n"
    "        if manager.is_locked():\n"
    "            return\n"
    "        if not hasattr(manager, 'reserve'):\n"
    "            # PN118 workspace-side patch did not apply. Skip silently.\n"
    "            return\n"
    "        scheduler_config = vllm_config.scheduler_config\n"
    "        speculative_config = vllm_config.speculative_config\n"
    "        extra_spec_tokens = (\n"
    "            speculative_config.num_speculative_tokens\n"
    "            if speculative_config is not None else 0\n"
    "        )\n"
    "        max_batch_tokens = scheduler_config.max_num_seqs * (1 + extra_spec_tokens)\n"
    "        query_dtype = vllm_config.model_config.dtype\n"
    "        manager.reserve(\n"
    "            (\n"
    "                (\n"
    "                    max_batch_tokens,\n"
    "                    self.num_heads,\n"
    "                    self.max_num_kv_splits,\n"
    "                    self.head_size + 1,\n"
    "                ),\n"
    "                torch.float32,\n"
    "            ),\n"
    "            ((max_batch_tokens, self.num_heads, self.head_size), query_dtype),\n"
    "            ((max_batch_tokens, self.num_heads), torch.float32),\n"
    "        )\n"
)

TQ_ANCHOR_INIT_NEW = (
    "        self.max_num_kv_splits = (\n"
    "            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph\n"
    "        )\n"
    "\n"
    "        # [Genesis PN399 — backport of vllm#46067] Largest CUDA-graph-\n"
    "        # captured decode batch. _DECODE_SCRATCH is sized to this once and\n"
    "        # reused so addresses baked into FULL cudagraphs never move.\n"
    "        # Resolves to None when cudagraphs are disabled (enforce_eager) —\n"
    "        # the `max_batch is not None and max_batch >= B` guard in\n"
    "        # _decode_attention then falls through to the WorkspaceManager.\n"
    "        self.max_decode_cudagraph_batch = (\n"
    "            vllm_config.compilation_config.max_cudagraph_capture_size\n"
    "        )\n"
    "\n"
    "        # [Genesis PN399 — backport of vllm#46067] PN399 owns the CG\n"
    "        # decode scratch via the module-level _DECODE_SCRATCH fixed\n"
    "        # buffer; the PN118 __init__ _reserve_decode_workspace pre-grow\n"
    "        # (box + call + method) was removed to cut boot overhead — it is\n"
    "        # dead on the CG path and the eager cold path falls back to\n"
    "        # PN118's try_get_simultaneous + torch.empty without it.\n"
)


# Sub-patch C — _decode_attention: insert the CG-path branch BEFORE
# PN118's `if is_workspace_manager_initialized():`, demoting PN118's block
# to the eager `elif`. Re-authored against live (the PR's verbatim OLD
# block does NOT exist — PN118 already rewrote it). Anchors the live PN118
# decode head (live dev148 1455-1462). PN118's try_get_simultaneous +
# torch.empty body (1463-1486) is left UNTOUCHED as the eager arm.
TQ_ANCHOR_DECODE_OLD = (
    "        mid_o_buf = output_buf = lse_buf = None\n"
    "        # ════════════════════════════════════════════════════════════\n"
    "        # [Genesis PN118 — backport of vllm#42551]\n"
    "        # Use try_get_simultaneous: returns None if workspace is locked\n"
    "        # and undersized (instead of raising AssertionError). Caller\n"
    "        # falls back to torch.empty so the engine keeps serving.\n"
    "        # ════════════════════════════════════════════════════════════\n"
    "        if is_workspace_manager_initialized():\n"
)

TQ_ANCHOR_DECODE_NEW = (
    "        mid_o_buf = output_buf = lse_buf = None\n"
    "        # ════════════════════════════════════════════════════════════\n"
    "        # [Genesis PN399 — backport of vllm#46067]\n"
    "        # CG path: fixed buffer sized for the largest captured batch whose\n"
    "        # address never moves, safe to bake into the FULL decode graphs.\n"
    "        # The growable WorkspaceManager is NOT safe here (frees+reallocs\n"
    "        # on grow -> stale graph-baked address -> CUDA IMA). Slices [:B]\n"
    "        # in-kernel. Takes precedence over the PN118 try_get path below,\n"
    "        # which now serves only the eager / over-max-batch cold path.\n"
    "        # ════════════════════════════════════════════════════════════\n"
    "        max_batch = self.max_decode_cudagraph_batch\n"
    "        if max_batch is not None and max_batch >= B:\n"
    "            mid_o_buf, output_buf, lse_buf = _get_decode_scratch(\n"
    "                max_batch, Hq, S, D, query.dtype, query.device\n"
    "            )\n"
    "        # ════════════════════════════════════════════════════════════\n"
    "        # [Genesis PN118 — backport of vllm#42551]\n"
    "        # Use try_get_simultaneous: returns None if workspace is locked\n"
    "        # and undersized (instead of raising AssertionError). Caller\n"
    "        # falls back to torch.empty so the engine keeps serving.\n"
    "        # PN399 demoted this to the eager/cold elif (CG path handled above).\n"
    "        # ════════════════════════════════════════════════════════════\n"
    "        elif is_workspace_manager_initialized():\n"
)


# Sub-patch C2 — REMOVE the now-dead PN353A decode-scratch reservation in
# TurboQuantMetadataBuilder._reserve_workspace. It pre-grows the
# WorkspaceManager to decode shapes (max_num_reqs, num_heads,
# max_num_splits, head_size+1)/f32, (max_num_reqs, num_heads,
# head_size)/model_dtype, (max_num_reqs, num_heads)/f32 — exactly the
# buffers _DECODE_SCRATCH now owns on the CG path, so it is dead weight
# (cut to reclaim boot overhead). Anchored from the `except AttributeError`
# / `return` of the max_num_splits try/except (live dev148 259-260),
# through the decode-scratch comment + get_simultaneous (262-270), the
# bounding blank (271), DOWN TO the first line of the continuation-prefill
# comment (272). The replacement preserves the try/except + a single blank
# and re-emits the continuation-prefill comment line UNCHANGED, so the
# PN353A continuation-prefill K/V reservation (live 272-290) stays
# BYTE-INTACT — only the decode-scratch get_simultaneous is excised.
# DISTINCT call sites proven on live bytes: the decode reserve (262-270)
# and the continuation-prefill reserve (287-290) are separated by the
# reserve_cont gate + early return (275-281); different shapes/dtypes.
# Byte-confirmed on live dev148: em-dash U+2014 in the comment, anchor
# count = 1.
TQ_ANCHOR_PN353A_DECODE_RESERVE_OLD = (
    "        except AttributeError:\n"
    "            return\n"
    "\n"
    "        # Decode scratch — mirrors _decode_attention's get_simultaneous.\n"
    "        current_workspace_manager().get_simultaneous(\n"
    "            (\n"
    "                (max_num_reqs, num_heads, max_num_splits, head_size + 1),\n"
    "                _genesis_pn353a_torch.float32,\n"
    "            ),\n"
    "            ((max_num_reqs, num_heads, head_size), model_config.dtype),\n"
    "            ((max_num_reqs, num_heads), _genesis_pn353a_torch.float32),\n"
    "        )\n"
    "\n"
    "        # Continuation-prefill K/V dequant buffers — only when chunked\n"
)

TQ_ANCHOR_PN353A_DECODE_RESERVE_NEW = (
    "        except AttributeError:\n"
    "            return\n"
    "\n"
    "        # [Genesis PN399 — backport of vllm#46067] PN353A decode-scratch\n"
    "        # get_simultaneous reservation removed — _DECODE_SCRATCH owns the\n"
    "        # CG decode buffers now, so this pre-grow is dead weight (boot\n"
    "        # overhead cut). The continuation-prefill K/V reservation below\n"
    "        # is untouched and stays essential (PN399 never touches prefill).\n"
    "        # Continuation-prefill K/V dequant buffers — only when chunked\n"
)


# ─────────────────────────────────────────────────────────────────────
# FILE 2 — shutdown.py — 1 anchor (D import + call). PRISTINE on dev148
# (live = PR before, byte-for-byte). The import sorts before the existing
# `from vllm.v1.worker.workspace import reset_workspace_manager`
# (attention.backends < worker.workspace) and the call follows
# reset_workspace_manager(). Single anchor spans import+call (live 11-21)
# to preserve ordering in one replacement.
# ─────────────────────────────────────────────────────────────────────

SHUTDOWN_ANCHOR_OLD = (
    "    from vllm.v1.worker.workspace import reset_workspace_manager\n"
    "\n"
    "    cache_config = vllm_config.cache_config\n"
    "    cache_config.num_gpu_blocks = None\n"
    "\n"
    "    compilation_config = vllm_config.compilation_config\n"
    "    compilation_config.static_forward_context.clear()\n"
    "\n"
    "    _ROPE_DICT.clear()\n"
    "    reset_workspace_manager()\n"
)

SHUTDOWN_ANCHOR_NEW = (
    "    from vllm.v1.attention.backends.turboquant_attn import reset_tq_decode_scratch\n"
    "    from vllm.v1.worker.workspace import reset_workspace_manager\n"
    "\n"
    "    cache_config = vllm_config.cache_config\n"
    "    cache_config.num_gpu_blocks = None\n"
    "\n"
    "    compilation_config = vllm_config.compilation_config\n"
    "    compilation_config.static_forward_context.clear()\n"
    "\n"
    "    _ROPE_DICT.clear()\n"
    "    reset_workspace_manager()\n"
    "    reset_tq_decode_scratch()  # [Genesis PN399 — backport of vllm#46067]\n"
)


# Post-fix spellings — present once a pin carries #46067 natively, so
# PN399 self-skips before touching the file. Both are substrings of
# PN399's own emitted replacements (allowlisted in
# tools/lint_drift_markers_allowlist.txt): TextPatcher checks the
# idempotency marker (Layer 2) BEFORE the drift markers (Layer 3), so the
# drift scan never reads PN399's own output.
_UPSTREAM_DRIFT_MARKER_GET = "_get_decode_scratch"
_UPSTREAM_DRIFT_MARKER_ATTR = "max_decode_cudagraph_batch"


def _make_tq_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_TQ_RELPATH)
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN399 TQ decode-scratch fixed buffer (vllm#46067)",
        target_file=target,
        marker=GENESIS_PN399_MARKER,
        sub_patches=[
            TextPatch(
                name="pn399_const_decode_scratch_defs",
                anchor=TQ_ANCHOR_CONST_OLD,
                replacement=TQ_ANCHOR_CONST_NEW,
                required=True,
            ),
            TextPatch(
                name="pn399_init_max_decode_cudagraph_batch",
                anchor=TQ_ANCHOR_INIT_OLD,
                replacement=TQ_ANCHOR_INIT_NEW,
                required=True,
            ),
            TextPatch(
                name="pn399_decode_cg_branch_wrap",
                anchor=TQ_ANCHOR_DECODE_OLD,
                replacement=TQ_ANCHOR_DECODE_NEW,
                required=True,
            ),
            TextPatch(
                name="pn399_pn353a_decode_reserve_remove",
                anchor=TQ_ANCHOR_PN353A_DECODE_RESERVE_OLD,
                replacement=TQ_ANCHOR_PN353A_DECODE_RESERVE_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            _UPSTREAM_DRIFT_MARKER_GET,
            _UPSTREAM_DRIFT_MARKER_ATTR,
        ],
    )


def _make_shutdown_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_SHUTDOWN_RELPATH)
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN399 TQ decode-scratch shutdown reset (vllm#46067)",
        target_file=target,
        marker=GENESIS_PN399_MARKER,
        sub_patches=[
            TextPatch(
                name="pn399_shutdown_reset_call",
                anchor=SHUTDOWN_ANCHOR_OLD,
                replacement=SHUTDOWN_ANCHOR_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[_UPSTREAM_DRIFT_MARKER_GET],
    )


def apply() -> tuple[str, str]:
    """Apply PN399 wiring across both target files. Never raises.

    Both TextPatchers are required=True; the combined status is the worst
    outcome across the two files (FAILED > SKIPPED > APPLIED-or-IDEMPOTENT).
    """
    if not is_enabled(Flags.PN399_TQ_DECODE_SCRATCH_IMA, default=False):
        return "skipped", (
            f"PN399 disabled (set {ENV_FLAG_FULL}=1 to opt in to the "
            "TurboQuant decode-scratch fixed-buffer IMA fix; default OFF, "
            "experimental belt-and-suspenders — vllm#46067)"
        )

    tq_patcher = _make_tq_patcher()
    sd_patcher = _make_shutdown_patcher()
    if tq_patcher is None or sd_patcher is None:
        missing = _TQ_RELPATH if tq_patcher is None else _SHUTDOWN_RELPATH
        return "skipped", f"{missing} not found in vllm install"

    statuses: list[str] = []
    reasons: list[str] = []
    for label, patcher in (
        ("turboquant_attn.py", tq_patcher),
        ("shutdown.py", sd_patcher),
    ):
        result, failure = patcher.apply()
        if result == TextPatchResult.APPLIED:
            statuses.append("applied")
            reasons.append(
                f"{label}: applied "
                f"({', '.join(patcher.applied_sub_patches)})"
            )
        elif result == TextPatchResult.IDEMPOTENT:
            statuses.append("applied")
            reasons.append(f"{label}: already applied (idempotent)")
        elif result == TextPatchResult.SKIPPED:
            statuses.append("skipped")
            msg = failure.reason if failure else "anchor not found"
            detail = failure.detail if failure and failure.detail else ""
            reasons.append(
                f"{label}: skipped — {msg}"
                + (f" ({detail})" if detail else "")
                + " — likely PN118 disabled/drifted or the pin already "
                "carries vllm#46067"
            )
        else:  # FAILED
            statuses.append("failed")
            msg = failure.reason if failure else "unknown failure"
            reasons.append(f"{label}: failed — {msg}")

    joined = "; ".join(reasons)
    if "failed" in statuses:
        return "failed", joined
    if "skipped" in statuses:
        return "skipped", joined
    return "applied", (
        "PN399 applied: TurboQuant decode scratch now uses a fixed "
        "module-level _DECODE_SCRATCH sized for max_cudagraph_capture_size "
        "on the CG hot path (PN118's try_get_simultaneous demoted to the "
        "eager elif), reset on shutdown — fixes the FULL-cudagraph CUDA "
        f"IMA (vllm#46067). [{joined}]"
    )


def is_applied() -> bool:
    tq_patcher = _make_tq_patcher()
    sd_patcher = _make_shutdown_patcher()
    if tq_patcher is None or sd_patcher is None:
        return False
    try:
        for patcher in (tq_patcher, sd_patcher):
            with open(patcher.target_file, encoding="utf-8") as fh:
                if GENESIS_PN399_MARKER not in fh.read():
                    return False
        return True
    except OSError:
        return False


__all__ = [
    "GENESIS_PN399_MARKER",
    "ENV_FLAG_FULL",
    "TQ_ANCHOR_CONST_OLD",
    "TQ_ANCHOR_CONST_NEW",
    "TQ_ANCHOR_INIT_OLD",
    "TQ_ANCHOR_INIT_NEW",
    "TQ_ANCHOR_DECODE_OLD",
    "TQ_ANCHOR_DECODE_NEW",
    "TQ_ANCHOR_PN353A_DECODE_RESERVE_OLD",
    "TQ_ANCHOR_PN353A_DECODE_RESERVE_NEW",
    "SHUTDOWN_ANCHOR_OLD",
    "SHUTDOWN_ANCHOR_NEW",
    "apply",
    "is_applied",
]
