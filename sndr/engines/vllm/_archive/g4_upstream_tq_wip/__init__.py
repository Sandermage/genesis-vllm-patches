# SPDX-License-Identifier: Apache-2.0
"""Genesis G4_* WIP patches superseded by upstream vllm PR #42637.

These patches were Genesis-original monkey-patches addressing limitations
of the upstream TurboQuant backend ON Gemma 4. PR #42637 (lesj0610,
OPEN as of 2026-05-17) adds proper implementations of the same features
into vllm upstream, making these Genesis patches redundant once #42637
merges or is cherry-picked into our fork.

Modules (all auto-loaded only via env flag, no registry entry):

  - g4_30_upstream_tq_unblock        — monkey-patches
    TurboQuantAttentionBackend.supports_mm_prefix → True. Superseded by
    PR #42637's proper mm_prefix Triton kernel mask (USE_MM_PREFIX
    constexpr, mm_prefix_range_tensor metadata).

  - g4_43_unblock_forced_triton      — reverts Gemma 4's hard-coded
    TRITON_ATTN forcing. Will likely remain useful in upstream until
    PR #42637 lands; review after cherry-pick.

  - g4_44_tq_head_dim_512_prefill    — torch SDPA fallback for
    head_size > 256 during prefill. Superseded by PR #42637's
    `_can_use_flash_attn = head_size <= 256` + `_sdpa_causal_prefill`.

  - g4_45_unify_page_diag            — diagnoses/auto-pads
    unify_kv_cache_spec_page_size for hybrid 3-tier Gemma 4 KV layout.
    Superseded by PR #42637's TQ-aware branch in
    unify_kv_cache_spec_page_size + UniformTypeKVCacheSpecs routing.

  - g4_50_genesis_native_backend     — abandoned attempt at a Genesis-
    original AttentionBackend (`GENESIS_G4_TQ`). Research showed upstream
    `TurboQuantAttentionImpl` + wrapper strategy is strictly better.
    Companion code in `../genesis_tq_abandoned/`.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
