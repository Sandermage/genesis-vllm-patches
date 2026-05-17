# SPDX-License-Identifier: Apache-2.0
"""G4_44 — TurboQuant prefill fallback to torch SDPA for head_dim > 256.

================================================================
WHY THIS EXISTS
================================================================

vllm dev371's ``TurboQuantAttentionImpl._prefill_attention`` uses
``flash_attn_varlen_func`` for the fast path (when no prior cache).
FlashAttention has a kernel limit of ``head_size ≤ 256``, so for
Gemma 4 31B AWQ's full-attention layers (head_dim=512) FlashAttn
rejects and TQ backend's prefill path fails.

Decode and continuation-prefill paths are Triton-only and DO support
arbitrary head_dim, but the first-chunk prefill path needs an
alternative SDPA for head_dim > 256.

================================================================
WHAT THIS DOES
================================================================

Monkey-patches ``TurboQuantAttentionImpl._prefill_attention`` to:

  1. Detect ``self.head_size > 256``.
  2. If true: route to a torch-native varlen SDPA implementation
     (``torch.nn.functional.scaled_dot_product_attention`` per-request).
  3. If false: call original (FlashAttn fast path, unchanged).

After SDPA, the **store** still happens via the original TQ store
mechanism (via the separate ``do_kv_cache_update`` op called BEFORE
forward). So the cache is fully populated and decode can read it.

================================================================
TORCH SDPA NOTES
================================================================

torch.nn.functional.scaled_dot_product_attention handles head_dim
arbitrarily (no FA kernel limit). It auto-picks the fastest backend
available (mem-efficient, math-only fallback, or FA if available).
For head_dim=512 it uses mem-efficient attention. Slower than FA but
fully functional.

For BATCH varlen prefill we iterate per-request. Future optimization:
write a Triton-only varlen SDPA kernel for head_dim=512.

================================================================
ENV FLAG
================================================================

``GENESIS_ENABLE_G4_44_TQ_HEAD_DIM_512=1`` — enables the override.
Default OFF; opt-in only after the operator confirms the model
benefits from it (typically Gemma 4 31B AWQ + larger).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.gemma4.g4_44_tq_head_dim_512")

GENESIS_G4_44_MARKER = (
    "Genesis G4_44 TurboQuant prefill torch-SDPA fallback for head_dim > 256 "
    "(enables real KV compression on Gemma 4 full-attention layers)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_44_TQ_HEAD_DIM_512"
_APPLIED = False
_ORIGINAL_PREFILL = None


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _torch_sdpa_varlen_prefill(
    query,       # (N, Hq, D)
    key,         # (N, Hk, D)
    value,       # (N, Hk, D)
    cu_seqlens_q,  # (num_reqs + 1,)
    scale: float,
):
    """Per-request torch SDPA varlen prefill. Causal mask + GQA.

    Slow but correct for head_dim > 256 (where FlashAttn rejects).
    """
    import torch
    import torch.nn.functional as F

    N, Hq, D = query.shape
    _, Hk, _ = key.shape
    out = torch.empty_like(query)

    # cu_seqlens_q on GPU; .item() each entry is a sync point. Cheap because
    # there are <= 256 requests per batch in production configs.
    cu = cu_seqlens_q.tolist() if hasattr(cu_seqlens_q, "tolist") else cu_seqlens_q
    num_reqs = len(cu) - 1
    for r in range(num_reqs):
        s, e = cu[r], cu[r + 1]
        if e <= s:
            continue
        q = query[s:e]  # (seq_len, Hq, D)
        k = key[s:e]    # (seq_len, Hk, D)
        v = value[s:e]  # (seq_len, Hk, D)

        # GQA: repeat K/V to match Q heads
        if Hk < Hq:
            n_rep = Hq // Hk
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Transpose to SDPA expected layout: (Hq, seq, D)
        q_t = q.transpose(0, 1)  # (Hq, seq_q, D)
        k_t = k.transpose(0, 1)
        v_t = v.transpose(0, 1)

        # Add batch dim of 1 for SDPA
        q_t = q_t.unsqueeze(0)  # (1, Hq, seq, D)
        k_t = k_t.unsqueeze(0)
        v_t = v_t.unsqueeze(0)

        attn = F.scaled_dot_product_attention(
            q_t, k_t, v_t,
            is_causal=True,
            scale=scale,
        )  # (1, Hq, seq, D)

        out[s:e] = attn.squeeze(0).transpose(0, 1)  # (seq, Hq, D)

    return out


def apply() -> tuple[str, str]:
    """Install head_dim>256 prefill fallback on TurboQuantAttentionImpl."""
    global _APPLIED, _ORIGINAL_PREFILL

    if not _env_enabled():
        return "skipped", (
            f"G4_44 disabled (set {_ENV_ENABLE}=1 to enable head_dim>256 "
            "prefill via torch SDPA — needed for Gemma 4 full layers)"
        )

    if _APPLIED:
        return "applied", "G4_44 already installed (idempotent)"

    try:
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionImpl,
        )
    except ImportError as e:
        return "skipped", f"TurboQuantAttentionImpl not importable: {e}"

    original = TurboQuantAttentionImpl._prefill_attention
    if getattr(original, "_genesis_g4_44_wrapped", False):
        _APPLIED = True
        return "applied", "G4_44 already wrapped (idempotent)"

    _ORIGINAL_PREFILL = original

    def _wrapped_prefill_attention(
        self,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        Pi,
        centroids,
        PiT=None,
        layer=None,
    ):
        """Route head_dim > 256 to torch SDPA; small head_dim to original.

        For Gemma 4 31B AWQ:
          * Sliding layers: head_dim=256 → original path (FlashAttn fast
            or continuation slow path, both head-256-safe)
          * Full layers: head_dim=512 → torch SDPA fallback
        """
        # Fast path: head_dim <= 256, use upstream
        if self.head_size <= 256:
            return original(
                self, query, key, value, kv_cache, attn_metadata,
                Pi, centroids, PiT, layer,
            )

        # head_dim > 256: torch SDPA. FlashAttn fast-path NOT usable here.
        N, Hq, D = query.shape
        max_query_len = attn_metadata.max_query_len
        max_seq_len = attn_metadata.max_seq_len

        # If max_query_len == max_seq_len, no prior cache → pure varlen SDPA
        if max_query_len == max_seq_len:
            return _torch_sdpa_varlen_prefill(
                query=query,
                key=key,
                value=value,
                cu_seqlens_q=attn_metadata.query_start_loc,
                scale=self.scale,
            )

        # Continuation: prior cache exists. Original's slow path uses
        # _tq_full_dequant_kv (Triton, head-dim-agnostic) followed by
        # per-request attention. That works for head_dim=512 too — let
        # original handle it. The continuation slow path is fully Triton.
        return original(
            self, query, key, value, kv_cache, attn_metadata,
            Pi, centroids, PiT, layer,
        )

    _wrapped_prefill_attention._genesis_g4_44_wrapped = True
    _wrapped_prefill_attention.__wrapped__ = original
    TurboQuantAttentionImpl._prefill_attention = _wrapped_prefill_attention
    _APPLIED = True

    log.info(
        "[G4_44] installed: TurboQuantAttentionImpl._prefill_attention "
        "now routes head_dim > 256 to torch SDPA. FlashAttn fast-path "
        "unchanged for head_dim <= 256."
    )
    return "applied", (
        "G4_44 installed: TurboQuant prefill on head_dim > 256 (e.g., "
        "Gemma 4 full-attention layers @ 512) now uses torch SDPA. "
        "Cache write is unaffected (handled by do_kv_cache_update — "
        "Triton-only, head-dim-agnostic)."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED, _ORIGINAL_PREFILL
    if not _APPLIED or _ORIGINAL_PREFILL is None:
        return False
    try:
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionImpl,
        )
        TurboQuantAttentionImpl._prefill_attention = _ORIGINAL_PREFILL
        _APPLIED = False
        return True
    except ImportError:
        return False


__all__ = ["GENESIS_G4_44_MARKER", "apply", "is_applied", "revert"]
