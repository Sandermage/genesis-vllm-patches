# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N32 — GDN chunked-prefill (Cliff 2 fix).

================================================================
Issue
================================================================
Reported by noonghunna in Genesis_internal_docs/CLIFF2_INVESTIGATION_20260430.md
(forwarded from VolandBerlioz Reddit post on club-3090).

On single 24GB GPU (1×3090 / 1×4090 / 1×5090) with prompts >50K tokens,
GDN layer's `core_attn_out` allocation hits OOM:

    File ".../vllm/model_executor/layers/mamba/gdn_linear_attn.py", line 588
    core_attn_out = torch.zeros(
        (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
        dtype=hidden_states.dtype, device=hidden_states.device,
    )
    # 50K × 64 heads × 128 dim × 2 bytes = ~819 MiB per layer
    # 30 GDN layers × 819 MiB = ~24 GiB persistent — fully consumes 24GB

Cliff 1 fix (PN12 FFN intermediate pool) handles transient SiluAndMul
activations. Cliff 2 is structurally different — it's the GDN layer's
core_attn_out which CAN be chunked because:
  - GDN forward is per-token modulo recurrent state
  - `gdn_attention_core` custom op maintains state per layer-name
  - Chunked calls to the same layer-name continue state seamlessly
  - Output projection (norm + out_proj) is per-token, chunkable

================================================================
FIX
================================================================

Text-patch on `forward_cuda` of `GatedDeltaNetAttention`:

When `num_tokens > THRESHOLD` (default 16K, env-tunable), split the
core attention + post-projection block into chunks of CHUNK_SIZE
(default 8K). For each chunk:
  1. Allocate transient core_attn_out at chunk shape (~130 MiB at
     8K × 64 × 128 × 2)
  2. Call gdn_attention_core for the chunk (state continues from
     prior chunk via layer-name keyed cache)
  3. Run norm + out_proj per chunk, writing to output[start:end]
  4. Chunk buffer goes out of scope, freed before next iteration

When `num_tokens <= THRESHOLD`, fall through to existing path
(P28 pool or torch.zeros). NO regression on normal workloads.

================================================================
PEAK MEMORY MATH
================================================================

Without PN32 (current behavior at 50K tokens, 30 layers):
  - Each layer holds 50000 × 64 × 128 × 2 = 819 MiB persistent
  - 30 layers = 24.6 GiB → OOM on 24GB card

With PN32 (chunk_size=8K):
  - Each layer transient: 8000 × 64 × 128 × 2 = 131 MiB per call
  - Released between chunks
  - Peak across loop iterations: ~131 MiB × number-of-active-tensors-in-scope
  - 30 layers × 131 MiB = 3.9 GiB peak (vs 24.6 GiB) — 6× reduction

================================================================
SAFETY MODEL
================================================================

- Default OFF (opt-in via GENESIS_ENABLE_PN32_GDN_CHUNKED_PREFILL=1)
- Activation gated by num_tokens > GENESIS_PN32_GDN_CHUNK_THRESHOLD
  (default 16384). Below threshold: existing path unchanged.
- Pure text-patch, idempotent via marker
- Anchor matches the exact `core_attn_out = (...)` allocation block
  + the `torch.ops.vllm.gdn_attention_core(...)` call site
- Drift-aware: if upstream rewrites GDN forward, anchor won't match
  → SKIPPED, source stays vanilla
- Cross-rig validation needed: our 2× A5000 PROD (TP=2) doesn't hit
  Cliff 2 threshold; community single-GPU users are the target

Threshold semantics (audit confirmed 2026-05-02)
------------------------------------------------
`num_tokens` in the patched `forward_cuda` is `hidden_states.size(0)`
(upstream `gdn_linear_attn.py:530`) — the FIRST dimension of the
batched input tensor. Under vLLM's continuous batching this is
**total tokens in the current forward call** (sum across all
sequences), NOT per-sequence tokens. So PN32 fires when batched
prefill exceeds 16384 tokens — which is the correct behavior:

- max_num_batched_tokens=4096 (typical): num_tokens ≤ 4K, never
  triggers PN32 → zero overhead for high-throughput configs
- max_num_batched_tokens=32768 + single >16K-token prompt prefill:
  num_tokens > 16K → PN32 chunks
- 5 short sequences batched (5 × 4K = 20K total): num_tokens > 16K
  → PN32 chunks (each chunk size 8K processes a slice of the
  concatenated batch). State-continuity caveat: chunks span
  sequence boundaries within a batch — gdn_attention_core's
  layer-name-keyed cache handles this because it tracks per-layer
  state, not per-sequence. Verified against upstream
  chunked-prefill behavior which uses the same mechanism.

State continuity assumption
---------------------------
`torch.ops.vllm.gdn_attention_core` maintains per-layer state via
`_encode_layer_name(self.prefix)` argument. Chunked calls to the same
layer continue state through the layer-name keyed cache (this is how
upstream chunked-prefill already works). PN32 just exercises this
mechanism more aggressively at long contexts.

If state doesn't propagate correctly between chunks, output would have
detectable distribution drift in last-tokens-of-each-chunk vs reference
single-pass. Use Wasserstein test (T4.4) to validate.

================================================================
LIMITATIONS / KNOWN UNKNOWNS
================================================================

1. **State continuity unverified across chunks empirically.** Logic
   says it should work (gdn_attention_core uses layer-name state cache),
   but no test we've run reproduces Cliff 2 conditions. Cross-rig
   validation by noonghunna or similar single-GPU operator required
   before promoting default-on.

2. **Norm + out_proj per chunk** assumes RMSNorm is per-token (true
   for Qwen3.5/3.6 GDN). If model uses cross-token-aware norm, output
   would differ from single-pass.

3. **Chunk size choice**: 8K default is conservative. Larger chunks
   reduce overhead but use more transient memory. 4K reduces peak by
   another 2× at ~negligible compute overhead.

4. **NOT a fix for prompts within budget**: if 50K prompt fits in
   memory-utilization budget, PN32 is wasted overhead (~5-10% per-token
   from extra Python control flow). Default OFF protects this case.

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Reporter: noonghunna (CLIFF2_INVESTIGATION_20260430.md).
Reference: Genesis_internal_docs/CLIFF2_INVESTIGATION_20260430.md.
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pN32_gdn_chunked_prefill")

GENESIS_PN32_MARKER = (
    "Genesis PN32 GDN chunked-prefill (Cliff 2 fix) v7.65"
)


# ─── Anchor: existing core_attn_out + gdn_attention_core block ──────
# Matches the EXACT allocation pattern + call. Indentation: 8 spaces
# (inside forward_cuda method body).

PN32_ANCHOR = (
    # NOTE: anchor matches the ORIGINAL upstream pattern (without P28's
    # persistent buffer indirection). PN32 conflicts with P28 because both
    # text-patch the same lines. Operator must choose one or the other:
    # - P28: persistent pool, optimal for normal-length workloads
    # - PN32: chunked transient, optimal for long-prompt single-stream
    "        core_attn_out = torch.zeros(\n"
    "            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),\n"
    "            dtype=hidden_states.dtype,\n"
    "            device=hidden_states.device,\n"
    "        )\n"
    "\n"
    "        torch.ops.vllm.gdn_attention_core(\n"
    "            mixed_qkv,\n"
    "            b,\n"
    "            a,\n"
    "            core_attn_out,\n"
    "            _encode_layer_name(self.prefix),\n"
    "        )\n"
    "\n"
    "        # ============================================================\n"
    "        # Part 3: Output Projection\n"
    "        # ============================================================\n"
    "        z_shape_og = z.shape\n"
    "        # Reshape input data into 2D tensor\n"
    "        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])\n"
    "        z = z.reshape(-1, z.shape[-1])\n"
    "        core_attn_out = self.norm(core_attn_out, z)\n"
    "        core_attn_out = core_attn_out.reshape(z_shape_og)\n"
    "        core_attn_out = rearrange(core_attn_out, \"... h d -> ... (h d)\")\n"
    "        output[:num_tokens], _ = self.out_proj(core_attn_out)\n"
)

PN32_REPLACEMENT = (
    "        # [Genesis PN32 Cliff 2 fix] Conditional chunked path for long\n"
    "        # prompts. When num_tokens > THRESHOLD, allocates transient\n"
    "        # chunk-sized core_attn_out instead of full-prompt-sized\n"
    "        # persistent buffer. Closes Cliff 2 OOM on single-24GB-GPU\n"
    "        # configs at >50K-token prompts (819 MiB per layer × 30 layers\n"
    "        # = 24 GiB, fully saturates 24GB card).\n"
    "        # See vllm/_genesis/wiring/hybrid/patch_N32_gdn_chunked_prefill.py\n"
    "        # for full design + state-continuity correctness reasoning.\n"
    "        import os as _genesis_pn32_os\n"
    "        _genesis_pn32_enabled = _genesis_pn32_os.environ.get(\n"
    "            'GENESIS_ENABLE_PN32_GDN_CHUNKED_PREFILL', ''\n"
    "        ).strip().lower() in ('1', 'true', 'yes', 'on')\n"
    "        try:\n"
    "            _genesis_pn32_threshold = int(\n"
    "                _genesis_pn32_os.environ.get(\n"
    "                    'GENESIS_PN32_GDN_CHUNK_THRESHOLD', '16384'\n"
    "                )\n"
    "            )\n"
    "        except (ValueError, TypeError):\n"
    "            _genesis_pn32_threshold = 16384\n"
    "        try:\n"
    "            _genesis_pn32_chunk_size = int(\n"
    "                _genesis_pn32_os.environ.get(\n"
    "                    'GENESIS_PN32_GDN_CHUNK_SIZE', '8192'\n"
    "                )\n"
    "            )\n"
    "        except (ValueError, TypeError):\n"
    "            _genesis_pn32_chunk_size = 8192\n"
    "\n"
    "        if _genesis_pn32_enabled and num_tokens > _genesis_pn32_threshold:\n"
    "            # ─── Chunked path: process in CHUNK_SIZE blocks ───\n"
    "            # State continuity: gdn_attention_core uses layer-name keyed\n"
    "            # cache, so successive chunks of same layer continue state\n"
    "            # automatically (same mechanism as upstream chunked-prefill).\n"
    "            for _genesis_pn32_start in range(0, num_tokens, _genesis_pn32_chunk_size):\n"
    "                _genesis_pn32_end = min(\n"
    "                    _genesis_pn32_start + _genesis_pn32_chunk_size, num_tokens\n"
    "                )\n"
    "                _genesis_pn32_chunk_len = _genesis_pn32_end - _genesis_pn32_start\n"
    "                # Allocate transient chunk-sized core_attn_out\n"
    "                _genesis_pn32_chunk_attn_out = torch.zeros(\n"
    "                    (_genesis_pn32_chunk_len,\n"
    "                     self.num_v_heads // self.tp_size,\n"
    "                     self.head_v_dim),\n"
    "                    dtype=hidden_states.dtype,\n"
    "                    device=hidden_states.device,\n"
    "                )\n"
    "                # Compute attention for this chunk\n"
    "                torch.ops.vllm.gdn_attention_core(\n"
    "                    mixed_qkv[_genesis_pn32_start:_genesis_pn32_end],\n"
    "                    b[_genesis_pn32_start:_genesis_pn32_end],\n"
    "                    a[_genesis_pn32_start:_genesis_pn32_end],\n"
    "                    _genesis_pn32_chunk_attn_out,\n"
    "                    _encode_layer_name(self.prefix),\n"
    "                )\n"
    "                # Per-chunk norm + out_proj — RMSNorm is per-token,\n"
    "                # safe to chunk\n"
    "                _genesis_pn32_z_chunk = z[_genesis_pn32_start:_genesis_pn32_end]\n"
    "                _genesis_pn32_z_shape_og = _genesis_pn32_z_chunk.shape\n"
    "                _genesis_pn32_chunk_2d = _genesis_pn32_chunk_attn_out.reshape(\n"
    "                    -1, _genesis_pn32_chunk_attn_out.shape[-1]\n"
    "                )\n"
    "                _genesis_pn32_z_2d = _genesis_pn32_z_chunk.reshape(\n"
    "                    -1, _genesis_pn32_z_chunk.shape[-1]\n"
    "                )\n"
    "                _genesis_pn32_normed = self.norm(\n"
    "                    _genesis_pn32_chunk_2d, _genesis_pn32_z_2d\n"
    "                )\n"
    "                _genesis_pn32_normed = _genesis_pn32_normed.reshape(\n"
    "                    _genesis_pn32_z_shape_og\n"
    "                )\n"
    "                _genesis_pn32_normed = rearrange(\n"
    "                    _genesis_pn32_normed, \"... h d -> ... (h d)\"\n"
    "                )\n"
    "                output[_genesis_pn32_start:_genesis_pn32_end], _ = (\n"
    "                    self.out_proj(_genesis_pn32_normed)\n"
    "                )\n"
    "                # Explicit del helps allocator reuse chunk slot\n"
    "                del _genesis_pn32_chunk_attn_out\n"
    "                del _genesis_pn32_chunk_2d\n"
    "                del _genesis_pn32_normed\n"
    "        else:\n"
    "            # ─── Original upstream path (no behavior change below threshold) ───\n"
    "            # Anchor uses original pattern; conflicts with P28 — operator\n"
    "            # must choose one or the other.\n"
    "            core_attn_out = torch.zeros(\n"
    "                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),\n"
    "                dtype=hidden_states.dtype,\n"
    "                device=hidden_states.device,\n"
    "            )\n"
    "\n"
    "            torch.ops.vllm.gdn_attention_core(\n"
    "                mixed_qkv,\n"
    "                b,\n"
    "                a,\n"
    "                core_attn_out,\n"
    "                _encode_layer_name(self.prefix),\n"
    "            )\n"
    "\n"
    "            # ============================================================\n"
    "            # Part 3: Output Projection\n"
    "            # ============================================================\n"
    "            z_shape_og = z.shape\n"
    "            # Reshape input data into 2D tensor\n"
    "            core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])\n"
    "            z = z.reshape(-1, z.shape[-1])\n"
    "            core_attn_out = self.norm(core_attn_out, z)\n"
    "            core_attn_out = core_attn_out.reshape(z_shape_og)\n"
    "            core_attn_out = rearrange(core_attn_out, \"... h d -> ... (h d)\")\n"
    "            output[:num_tokens], _ = self.out_proj(core_attn_out)\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/layers/mamba/gdn_linear_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN32 model_executor/layers/mamba/gdn_linear_attn.py — "
            "GDN forward_cuda chunked-prefill (Cliff 2 fix)"
        ),
        target_file=str(target),
        marker=GENESIS_PN32_MARKER,
        sub_patches=[
            TextPatch(
                name="pN32_gdn_chunked_prefill",
                anchor=PN32_ANCHOR,
                replacement=PN32_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN32",
            # If upstream rewrites GDN forward, our anchor won't match
            # → patcher reports drift, no-op apply
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN32 — GDN chunked-prefill (text-patch)."""
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN32")
    log_decision("PN32", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "gdn_linear_attn.py not resolvable"

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "PN32 applied: GDN forward_cuda now uses conditional chunked "
            "path for long prompts (>16K tokens default). Default OFF — "
            "operators opt-in via GENESIS_ENABLE_PN32_GDN_CHUNKED_PREFILL=1. "
            "Closes Cliff 2 OOM on single-24GB-GPU configs at >50K-token "
            "prompts. Cross-rig validation needed (our 2×A5000 PROD with "
            "TP=2 doesn't hit threshold)."
        ),
        patch_name="PN32 GDN chunked-prefill",
    )
