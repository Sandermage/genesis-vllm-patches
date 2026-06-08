# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch PN118 — TurboQuant workspace graceful-fallback fix.

Backport of [vllm#42551](https://github.com/vllm-project/vllm/pull/42551)
by `jasonboukheir` (OPEN at the time of backport), customized so that
it composes with our existing P99 ``get_simultaneous`` memoization
cache (Genesis P99 v7.62.15).

================================================================
WHAT THIS PATCH DOES
================================================================

The upstream PR fixes an AssertionError that fires on the first
TurboQuant decode request when the workspace was locked at 0 MB
during warmup (because warmup never landed a decode forward through
the TQ backend — happens on dense + hybrid models such as our
Lorbus/Qwen3.6-27B-int4-AutoRound where only 16 of 64 layers route
through TQ).

The PR adds two new ``WorkspaceManager`` methods:

  * ``try_get_simultaneous(...)`` — like ``get_simultaneous`` but
    returns ``None`` (instead of raising) when the workspace is
    locked and undersized. Callers fall back to ``torch.empty``.
  * ``reserve(...)`` — pre-allocate workspace large enough for given
    shapes on every ubatch slot at init time, so the lock at end of
    warmup snapshots a workspace that already fits steady-state.

And updates TurboQuant ``__init__`` to call ``reserve(...)`` and
``_decode_attention`` to use ``try_get_simultaneous`` + fallback.

================================================================
COMPOSITION WITH P99 MEMOIZATION
================================================================

Upstream's PR refactors ``get_simultaneous`` to use new helpers
``_compute_layout`` + ``_slice``. Our P99 already lives inside the
body of ``get_simultaneous`` (memoization cache, 5× faster on
hot-loop decode). We **preserve P99 intact** and instead ADD the
two new methods alongside, plus we extend ``_ensure_workspace_size``
signature to accept an optional ``ubatch_id`` parameter (needed by
``reserve`` to size sibling ubatch slots).

This means PN118 and P99 coexist:
  * ``get_simultaneous`` keeps P99 memoization (fast path).
  * ``try_get_simultaneous`` is a NEW method (no memoization;
    locked-fallback semantics).
  * ``reserve`` is a NEW method.

================================================================
SAFETY MODEL
================================================================

- 4 independent string replacements; all-or-nothing apply with
  in-memory validation before write-back.
- Idempotent via Genesis marker at file head.
- Drift retreat: if any anchor isn't found, skip cleanly without
  touching the file.
- TurboQuant side change has a fallback to ``torch.empty`` if
  workspace is locked-undersized — graceful degradation.

================================================================

Author: Genesis backport, original by jasonboukheir.
"""
from __future__ import annotations

import logging
from pathlib import Path

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root

log = logging.getLogger("genesis.wiring.pn118_tq_workspace_fallback")

GENESIS_PN118_MARKER = (
    "Genesis PN118 TQ workspace graceful-fallback (backport: vllm#42551)"
)
GENESIS_PN118_MARKER_LINE = f"# {GENESIS_PN118_MARKER}\n"


# ─────────────────────────────────────────────────────────────────────
# WORKSPACE.PY MODIFICATIONS — 2 anchors
# ─────────────────────────────────────────────────────────────────────

WORKSPACE_ANCHOR_SIG_OLD = (
    "    def _ensure_workspace_size(self, required_bytes: int) -> torch.Tensor:\n"
    '        """Ensure workspace is allocated and large enough, return current workspace.\n'
    "\n"
    "        Args:\n"
    "            required_bytes: The number of bytes required.\n"
    "\n"
    "        Returns:\n"
    "            The current workspace tensor.\n"
    '        """\n'
    "        ubatch_id = dbo_current_ubatch_id()\n"
)

WORKSPACE_ANCHOR_SIG_NEW = (
    "    def _ensure_workspace_size(\n"
    "        self, required_bytes: int, ubatch_id: int | None = None\n"
    "    ) -> torch.Tensor:\n"
    '        """Ensure workspace is allocated and large enough, return that workspace.\n'
    "\n"
    "        Args:\n"
    "            required_bytes: The number of bytes required.\n"
    "            ubatch_id: Which ubatch slot to size. Defaults to current\n"
    "                ubatch (``dbo_current_ubatch_id()``); pass an explicit\n"
    "                id from ``reserve`` to size sibling slots. [PN118]\n"
    "\n"
    "        Returns:\n"
    "            The workspace tensor for the selected ubatch.\n"
    '        """\n'
    "        if ubatch_id is None:\n"
    "            ubatch_id = dbo_current_ubatch_id()\n"
)

# ANCHOR 2: insert new methods BEFORE _ensure_workspace_size definition.
# P99-INDEPENDENT: anchor on the post-sig-change signature line only.
# `get_simultaneous` body above may or may not have P99 memoization
# applied at this point in the boot order — we don't care, we anchor
# at the method boundary which exists in both states.
WORKSPACE_ANCHOR_NEW_METHODS_OLD = (
    "    def _ensure_workspace_size(\n"
    "        self, required_bytes: int, ubatch_id: int | None = None\n"
    "    ) -> torch.Tensor:\n"
)

WORKSPACE_ANCHOR_NEW_METHODS_NEW = (
    "    # ────────────────────────────────────────────────────────────────\n"
    "    # [Genesis PN118 — backport of vllm#42551]\n"
    "    # try_get_simultaneous: like get_simultaneous but returns None\n"
    "    # when the workspace is locked and undersized (instead of raising\n"
    "    # AssertionError). Caller falls back to torch.empty. Does NOT use\n"
    "    # P99 memoization — fallback semantics differ.\n"
    "    # reserve: pre-allocate every ubatch slot at init time so the lock\n"
    "    # at end-of-warmup captures a workspace that already fits steady-\n"
    "    # state usage on every ubatch (not just the one whose forward\n"
    "    # happened to be active during the reservation call).\n"
    "    # ────────────────────────────────────────────────────────────────\n"
    "    def try_get_simultaneous(\n"
    "        self, *shapes_and_dtypes: tuple[tuple[int, ...], torch.dtype]\n"
    "    ) -> list[torch.Tensor] | None:\n"
    '        """Like get_simultaneous but returns None when growth would be\n'
    "        needed on a locked workspace, instead of raising.\n"
    '        """\n'
    "        actual_bytes = [_compute_bytes(s, d) for s, d in shapes_and_dtypes]\n"
    "        aligned_bytes = [round_up(actual, 256) for actual in actual_bytes]\n"
    "        offsets = list(accumulate([0] + aligned_bytes[:-1]))\n"
    "        total_bytes = sum(aligned_bytes)\n"
    "        if self._locked:\n"
    "            ubatch_id = dbo_current_ubatch_id()\n"
    "            current = self._current_workspaces[ubatch_id]\n"
    "            if self._workspace_size_bytes(current) < total_bytes:\n"
    "                return None\n"
    "            workspace = current\n"
    "        else:\n"
    "            workspace = self._ensure_workspace_size(total_bytes)\n"
    "        return [\n"
    "            workspace[offsets[i] : offsets[i] + actual_bytes[i]]\n"
    "            .view(shapes_and_dtypes[i][1])\n"
    "            .reshape(shapes_and_dtypes[i][0])\n"
    "            for i in range(len(shapes_and_dtypes))\n"
    "        ]\n"
    "\n"
    "    def reserve(\n"
    "        self, *shapes_and_dtypes: tuple[tuple[int, ...], torch.dtype]\n"
    "    ) -> None:\n"
    '        """Pre-allocate workspace large enough for these shapes on every\n'
    "        ubatch slot. Call at init time, BEFORE lock(), so the snapshot\n"
    "        captures a workspace that already fits steady-state usage.\n"
    '        """\n'
    "        actual_bytes = [_compute_bytes(s, d) for s, d in shapes_and_dtypes]\n"
    "        aligned_bytes = [round_up(actual, 256) for actual in actual_bytes]\n"
    "        total_bytes = sum(aligned_bytes)\n"
    "        for ubatch_id in range(self._num_ubatches):\n"
    "            self._ensure_workspace_size(total_bytes, ubatch_id=ubatch_id)\n"
    "\n"
    "    def _ensure_workspace_size(\n"
    "        self, required_bytes: int, ubatch_id: int | None = None\n"
    "    ) -> torch.Tensor:\n"
)


# ─────────────────────────────────────────────────────────────────────
# TURBOQUANT_ATTN.PY MODIFICATIONS — 2 anchors
# ─────────────────────────────────────────────────────────────────────

TQ_ANCHOR_INIT_OLD = (
    "        self.max_num_kv_splits = (\n"
    "            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph\n"
    "        )\n"
    "\n"
    "    def _flash_attn_varlen(\n"
)

TQ_ANCHOR_INIT_NEW = (
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
    "\n"
    "    def _flash_attn_varlen(\n"
)


TQ_ANCHOR_DECODE_OLD = (
    "        mid_o_buf = output_buf = lse_buf = None\n"
    "        if is_workspace_manager_initialized():\n"
    "            # output_buf in query dtype — matches the in-kernel fp16 cast in stage2.\n"
    "            mid_o_buf, output_buf, lse_buf = (\n"
    "                current_workspace_manager().get_simultaneous(\n"
    "                    ((B, Hq, S, D + 1), torch.float32),\n"
    "                    ((B, Hq, D), query.dtype),\n"
    "                    ((B, Hq), torch.float32),\n"
    "                )\n"
    "            )\n"
)

TQ_ANCHOR_DECODE_NEW = (
    "        mid_o_buf = output_buf = lse_buf = None\n"
    "        # ════════════════════════════════════════════════════════════\n"
    "        # [Genesis PN118 — backport of vllm#42551]\n"
    "        # Use try_get_simultaneous: returns None if workspace is locked\n"
    "        # and undersized (instead of raising AssertionError). Caller\n"
    "        # falls back to torch.empty so the engine keeps serving.\n"
    "        # ════════════════════════════════════════════════════════════\n"
    "        if is_workspace_manager_initialized():\n"
    "            manager = current_workspace_manager()\n"
    "            if hasattr(manager, 'try_get_simultaneous'):\n"
    "                bufs = manager.try_get_simultaneous(\n"
    "                    ((B, Hq, S, D + 1), torch.float32),\n"
    "                    ((B, Hq, D), query.dtype),\n"
    "                    ((B, Hq), torch.float32),\n"
    "                )\n"
    "                if bufs is not None:\n"
    "                    mid_o_buf, output_buf, lse_buf = bufs\n"
    "            else:\n"
    "                # PN118 workspace-side not applied — use legacy path.\n"
    "                mid_o_buf, output_buf, lse_buf = manager.get_simultaneous(\n"
    "                    ((B, Hq, S, D + 1), torch.float32),\n"
    "                    ((B, Hq, D), query.dtype),\n"
    "                    ((B, Hq), torch.float32),\n"
    "                )\n"
    "        if mid_o_buf is None:\n"
    "            # Fallback path: workspace was locked at an undersized\n"
    "            # snapshot; allocate scratch buffers directly per call.\n"
    "            mid_o_buf = torch.empty(\n"
    "                (B, Hq, S, D + 1), dtype=torch.float32, device=query.device\n"
    "            )\n"
    "            output_buf = torch.empty((B, Hq, D), dtype=query.dtype, device=query.device)\n"
    "            lse_buf = torch.empty((B, Hq), dtype=torch.float32, device=query.device)\n"
)


def _resolve_targets() -> tuple[Path | None, Path | None]:
    w = resolve_vllm_file("v1/worker/workspace.py")
    t = resolve_vllm_file("v1/attention/backends/turboquant_attn.py")
    return (Path(w) if w else None, Path(t) if t else None)


def apply() -> tuple[str, str]:
    """Apply PN118 — TQ workspace graceful-fallback fix."""
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN118")
    log_decision("PN118", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    ws_path, tq_path = _resolve_targets()
    if ws_path is None or tq_path is None:
        return "skipped", "workspace.py or turboquant_attn.py not found"
    if not ws_path.is_file() or not tq_path.is_file():
        return "skipped", "target file missing"

    try:
        ws_content = ws_path.read_text()
        tq_content = tq_path.read_text()
    except OSError as e:
        return "failed", f"read failed: {e}"

    if GENESIS_PN118_MARKER in ws_content and GENESIS_PN118_MARKER in tq_content:
        return "applied", "idempotent (marker present in both files)"

    # ─── workspace.py: apply 2 anchors in sequence ───
    if WORKSPACE_ANCHOR_SIG_OLD not in ws_content:
        return (
            "skipped",
            "workspace.py anchor #1 (_ensure_workspace_size signature) "
            "not found — drift or already partly applied",
        )
    ws_new = ws_content.replace(WORKSPACE_ANCHOR_SIG_OLD, WORKSPACE_ANCHOR_SIG_NEW, 1)
    if WORKSPACE_ANCHOR_NEW_METHODS_OLD not in ws_new:
        return (
            "skipped",
            "workspace.py anchor #2 (insert new methods) not found — "
            "drift or P99 boundary mismatch",
        )
    ws_new = ws_new.replace(
        WORKSPACE_ANCHOR_NEW_METHODS_OLD, WORKSPACE_ANCHOR_NEW_METHODS_NEW, 1
    )

    # ─── turboquant_attn.py: apply 2 anchors in sequence ───
    if TQ_ANCHOR_INIT_OLD not in tq_content:
        return (
            "skipped",
            "turboquant_attn.py anchor #1 (__init__ reserve hook) not found",
        )
    tq_new = tq_content.replace(TQ_ANCHOR_INIT_OLD, TQ_ANCHOR_INIT_NEW, 1)
    if TQ_ANCHOR_DECODE_OLD not in tq_new:
        return (
            "skipped",
            "turboquant_attn.py anchor #2 (_decode_attention) not found",
        )
    tq_new = tq_new.replace(TQ_ANCHOR_DECODE_OLD, TQ_ANCHOR_DECODE_NEW, 1)

    # Inject marker at file head (after SPDX line) — workspace.py
    ws_lines = ws_new.split("\n", 1)
    if ws_lines and ws_lines[0].startswith("#"):
        ws_final = ws_lines[0] + "\n" + GENESIS_PN118_MARKER_LINE + (
            ws_lines[1] if len(ws_lines) > 1 else ""
        )
    else:
        ws_final = GENESIS_PN118_MARKER_LINE + ws_new

    # Inject marker — turboquant_attn.py
    tq_lines = tq_new.split("\n", 1)
    if tq_lines and tq_lines[0].startswith("#"):
        tq_final = tq_lines[0] + "\n" + GENESIS_PN118_MARKER_LINE + (
            tq_lines[1] if len(tq_lines) > 1 else ""
        )
    else:
        tq_final = GENESIS_PN118_MARKER_LINE + tq_new

    # Write both files atomically (sequential, but in-memory validated above).
    try:
        ws_path.write_text(ws_final)
        tq_path.write_text(tq_final)
    except OSError as e:
        return "failed", f"write failed: {e}"

    return (
        "applied",
        "PN118 applied: WorkspaceManager.{try_get_simultaneous,reserve} "
        "added (composes with P99 memoization), TurboQuant __init__ now "
        "reserves decode scratch, _decode_attention uses try_get_simultaneous "
        "with torch.empty fallback. Closes the AssertionError on first "
        "decode request for partial-TQ models (e.g. Lorbus 27B AutoRound)."
    )


def is_applied() -> bool:
    if vllm_install_root() is None:
        return False
    ws_path, tq_path = _resolve_targets()
    if ws_path is None or tq_path is None:
        return False
    try:
        return (
            GENESIS_PN118_MARKER in ws_path.read_text()
            and GENESIS_PN118_MARKER in tq_path.read_text()
        )
    except OSError:
        return False
