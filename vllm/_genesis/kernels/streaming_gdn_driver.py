# SPDX-License-Identifier: Apache-2.0
"""Streaming GDN driver — Variant D Phase 2.

Window-iterative replacement for the `chunk_gated_delta_rule_fwd_h →
chunk_fwd_o` consumer pair in `fla/ops/chunk.py:chunk_gated_delta_rule_fwd`.

Eliminates the `(B, NT, H, V, K)` peak materialization (Cliff 2b OOM
trigger, 805 MiB at T=64K Genesis 27B Lorbus shapes) by processing
WINDOW_NT chunks at a time, reusing a small pooled buffer.

Empirical confirmation (issue #20, 2026-05-05): noonghunna confirmed
"the limitation is the triton kernel for cliff 2; doesn't appear with
llama.cpp" — exactly the materialization pattern this fix removes.

Numerical correctness proof: Phase 1 TDD demonstrates window-iterative
output bit-equivalent to baseline materialize-full at rtol=1e-5
(see `tests/integration/test_streaming_gdn_numerical.py`).

API
---
`streaming_chunk_gated_delta_rule_fwd(q, k, v, g, beta, scale, initial_state,
output_final_state, cu_seqlens, chunk_indices, chunk_offsets) → (g, o, A,
final_state, w_or_none, h_or_none, v_new_or_none)` — drop-in replacement
for `chunk_gated_delta_rule_fwd` in chunk.py.

Eligibility
-----------
Streaming path engages ONLY when ALL of:
  * `GENESIS_ENABLE_PN59_STREAMING_GDN=1` (master env)
  * single-sequence prefill (cu_seqlens is None OR shape == (2,))
  * T > WINDOW_NT * BT * 4 (else overhead exceeds savings)
  * h dtype/device standard (no edge cases)

Otherwise falls through to vanilla `_orig_chunk_gated_delta_rule_fwd`
(passed in from text-patched orchestrator).

Author: Sandermage 2026-05-05, Variant D Phase 2.
"""
from __future__ import annotations

import logging
import os

import torch

from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool

log = logging.getLogger("genesis.kernels.streaming_gdn_driver")


# Hot-path bypass threshold — below this, vanilla path wins on overhead
_BYPASS_T_MULTIPLIER = 4
# FLA chunk size — pinned to upstream constant (`FLA_CHUNK_SIZE`)
_FLA_CHUNK_SIZE = 64


def streaming_chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    *,
    # Injected upstream primitives (from FLA module-level imports)
    chunk_local_cumsum,
    chunk_scaled_dot_kkt_fwd,
    solve_tril,
    recompute_w_u_fwd,
    chunk_gated_delta_rule_fwd_h,
    chunk_fwd_o,
    SUPPRESS_LEVEL: int = 0,
) -> tuple:
    """Streaming variant of `chunk_gated_delta_rule_fwd`.

    Returns the same 7-tuple as upstream:
      (g, o, A, final_state, w_or_None, h_or_None, v_new_or_None)
    """
    # Eligibility — single-seq long prefill only.
    # Audit P2 fix 2026-05-05 (genesis_deep_cross_audit, P2.4): tightened
    # eligibility to also reject calls with non-trivial chunk metadata.
    # The streaming path passes `cu_seqlens=None, chunk_indices=None,
    # chunk_offsets=None` to inner kernels; if the caller supplied
    # significant chunk_offsets/chunk_indices we'd silently drop that
    # metadata and produce divergent results vs vanilla.
    T = q.shape[1]
    is_single_seq = (
        cu_seqlens is None
        or (hasattr(cu_seqlens, "shape") and cu_seqlens.shape == (2,))
    )
    has_no_chunk_metadata = (
        chunk_indices is None and chunk_offsets is None
    )
    window_nt = GdnScratchPool.get_window_nt()
    threshold_T = window_nt * _FLA_CHUNK_SIZE * _BYPASS_T_MULTIPLIER

    if (not GdnScratchPool.is_production_eligible()
            or not is_single_seq
            or not has_no_chunk_metadata
            or T <= threshold_T):
        # Quiet by default; only log at DEBUG so we don't spam normal use.
        # Operator can flip GENESIS_PN59_DEBUG=1 to see why streaming was bypassed.
        if os.environ.get("GENESIS_PN59_DEBUG", "").strip().lower() in (
            "1", "true", "yes", "y", "on",
        ):
            reason = (
                "pool not eligible" if not GdnScratchPool.is_production_eligible()
                else "multi-seq" if not is_single_seq
                else "chunk metadata present" if not has_no_chunk_metadata
                else f"T={T} ≤ threshold={threshold_T}"
            )
            log.info("[PN59] vanilla path (reason: %s)", reason)
        return _vanilla_path(
            q, k, v, g, beta, scale, initial_state, output_final_state,
            cu_seqlens, chunk_indices, chunk_offsets,
            chunk_local_cumsum=chunk_local_cumsum,
            chunk_scaled_dot_kkt_fwd=chunk_scaled_dot_kkt_fwd,
            solve_tril=solve_tril,
            recompute_w_u_fwd=recompute_w_u_fwd,
            chunk_gated_delta_rule_fwd_h=chunk_gated_delta_rule_fwd_h,
            chunk_fwd_o=chunk_fwd_o,
            SUPPRESS_LEVEL=SUPPRESS_LEVEL,
        )

    # Streaming path
    try:
        return _streaming_path(
            q, k, v, g, beta, scale, initial_state, output_final_state,
            cu_seqlens, chunk_indices, chunk_offsets,
            window_nt=window_nt,
            chunk_local_cumsum=chunk_local_cumsum,
            chunk_scaled_dot_kkt_fwd=chunk_scaled_dot_kkt_fwd,
            solve_tril=solve_tril,
            recompute_w_u_fwd=recompute_w_u_fwd,
            chunk_gated_delta_rule_fwd_h=chunk_gated_delta_rule_fwd_h,
            chunk_fwd_o=chunk_fwd_o,
            SUPPRESS_LEVEL=SUPPRESS_LEVEL,
        )
    except Exception as e:
        # Strict no-regression: any failure → vanilla fallback
        log.warning(
            "[PN59] streaming path raised %s — falling back to vanilla. "
            "Disable PN59 if recurrent: GENESIS_ENABLE_PN59_STREAMING_GDN=0",
            type(e).__name__,
        )
        return _vanilla_path(
            q, k, v, g, beta, scale, initial_state, output_final_state,
            cu_seqlens, chunk_indices, chunk_offsets,
            chunk_local_cumsum=chunk_local_cumsum,
            chunk_scaled_dot_kkt_fwd=chunk_scaled_dot_kkt_fwd,
            solve_tril=solve_tril,
            recompute_w_u_fwd=recompute_w_u_fwd,
            chunk_gated_delta_rule_fwd_h=chunk_gated_delta_rule_fwd_h,
            chunk_fwd_o=chunk_fwd_o,
            SUPPRESS_LEVEL=SUPPRESS_LEVEL,
        )


def _vanilla_path(
    q, k, v, g, beta, scale, initial_state, output_final_state,
    cu_seqlens, chunk_indices, chunk_offsets,
    *,
    chunk_local_cumsum, chunk_scaled_dot_kkt_fwd, solve_tril,
    recompute_w_u_fwd, chunk_gated_delta_rule_fwd_h, chunk_fwd_o,
    SUPPRESS_LEVEL: int,
):
    """Identical to upstream `chunk_gated_delta_rule_fwd`. Single allocation
    of full `h` tensor — Cliff 2b OOM trigger, but bit-correct baseline."""
    g = chunk_local_cumsum(
        g, chunk_size=_FLA_CHUNK_SIZE, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g=g,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        output_dtype=torch.float32,
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens,
                   chunk_indices=chunk_indices, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A, g_cumsum=g,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g,
        initial_state=initial_state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    o = chunk_fwd_o(
        q=q, k=k, v=v_new, h=h, g=g, scale=scale,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )
    if SUPPRESS_LEVEL < 3:
        return g, o, A, final_state, None, None, None
    return g, o, A, final_state, w, h, v_new


def _streaming_path(
    q, k, v, g, beta, scale, initial_state, output_final_state,
    cu_seqlens, chunk_indices, chunk_offsets,
    *,
    window_nt: int,
    chunk_local_cumsum, chunk_scaled_dot_kkt_fwd, solve_tril,
    recompute_w_u_fwd, chunk_gated_delta_rule_fwd_h, chunk_fwd_o,
    SUPPRESS_LEVEL: int,
):
    """Window-iterative driver — process WINDOW_NT chunks at a time.

    Same pre-h ops (cumsum, kkt, solve, recompute_w_u) as vanilla.
    Replaces fwd_h+fwd_o tail with windowed loop.

    Key observation (Phase 1 numerical proof): Triton kernel
    `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` is internally
    recurrent in registers (b_h1..b_h4). Calling it with single-window
    inputs + chained `initial_state` produces identical state trajectory
    to a single full-T call. Then `chunk_fwd_o` reads only the current
    window's h slice — per-chunk independent (verified by SGLang
    `chunk_fwd_kernel_o:74` analysis).
    """
    B, T, Hg, K = q.shape
    V = v.shape[-1]
    BT = _FLA_CHUNK_SIZE

    # Phase A: full-input pre-h ops (small allocations, cheap)
    g_full = chunk_local_cumsum(
        g, chunk_size=BT, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g=g_full,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        output_dtype=torch.float32,
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens,
                   chunk_indices=chunk_indices, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A, g_cumsum=g_full,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )

    # Phase B: pre-allocate output o (B, T, H_v, V) — same shape as v
    o_full = torch.empty_like(v)

    # State chained across windows (float32 per kernel signature)
    state = initial_state
    H = u.shape[-2]
    final_state = None

    # Window loop — slice T-dim by window_nt × BT tokens
    window_T = window_nt * BT
    for win_start in range(0, T, window_T):
        win_end = min(win_start + window_T, T)
        cur_T = win_end - win_start
        cur_NT = (cur_T + BT - 1) // BT
        is_last_window = (win_end >= T)

        # Slice T-dim inputs (input_guard wraps will re-contigify if needed)
        k_w = k[:, win_start:win_end]
        w_w = w[:, win_start:win_end]
        u_w = u[:, win_start:win_end]
        g_w = g_full[:, win_start:win_end]
        q_w = q[:, win_start:win_end]

        # Output_final_state ONLY on last window
        out_state = output_final_state and is_last_window

        # Run fwd_h on window — kernel allocates small h locally
        # NOTE: we don't try to inject scratch pool buffer because the
        # kernel takes ownership of the allocation via `k.new_empty(...)`.
        # The savings come from window being small NT, not from bypassing
        # alloc — h_window is automatically GC'd after each iteration
        # because no reference is held outside this scope.
        h_w, v_new_w, state_next = chunk_gated_delta_rule_fwd_h(
            k=k_w, w=w_w, u=u_w, g=g_w,
            initial_state=state,
            output_final_state=out_state,
            cu_seqlens=None,  # window is single-seq slice
            chunk_indices=None,
            chunk_offsets=None,
        )

        # Consume h_w via chunk_fwd_o for this window
        o_w = chunk_fwd_o(
            q=q_w, k=k_w, v=v_new_w, h=h_w, g=g_w, scale=scale,
            cu_seqlens=None, chunk_indices=None,
        )

        # Write window's o into o_full
        o_full[:, win_start:win_end].copy_(o_w)

        # Chain state forward (kernel writes float32 final state)
        if out_state:
            final_state = state_next
        # State for next window: convert kernel's output back if needed
        # The kernel chains internally via b_h1..b_h4 registers; we use
        # last window's intermediate state via h_w[:, -1] as a fallback
        # if state_next not requested for non-final windows.
        if not is_last_window:
            # h_w[:, -1] shape: (B, H, V, K) — the last chunk's state
            # This becomes initial_state for next window.
            # Cast to float32 to match kernel state type expectation.
            state = h_w[:, -1].to(torch.float32)

        # Drop window references to allow GC
        del h_w, v_new_w, o_w

    if SUPPRESS_LEVEL < 3:
        return g_full, o_full, A, final_state, None, None, None
    return g_full, o_full, A, final_state, w, None, None
