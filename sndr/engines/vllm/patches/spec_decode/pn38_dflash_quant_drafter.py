"""Wiring for PN38 — DFlash drafter quantization support (vllm PR #40425 backport).

Backports vllm-project/vllm#40425 (infatoshi, OPEN 2026-04-XX) which
enables quantized DFlash drafter checkpoints (FP8 W8A8, NVFP4, AWQ, etc.).

Per the upstream PR description: this is a CORRECTNESS/COMPATIBILITY fix,
NOT a throughput improvement claim. Without it, loading a quantized
DFlash drafter checkpoint either fails (KeyError on `qkv_proj.weight`)
or silently uses dense BF16 weights, defeating the quantization purpose.

Real-world value:
  - Today: NO-OP — Genesis stack uses configured BF16 DFlash drafters
    (`Qwen3.6-{27B,35B-A3B}-DFlash` variants are BF16 in current profiles)
  - Tomorrow: enables drop-in FP8/NVFP4 drafter checkpoints when they
    become available (e.g. AEON-7/Qwen3.6-NVFP4-DFlash, llm-compressor
    self-quantized variants per vllm/blog/2025/12/13/speculators-v030)
  - Memory savings: BF16 drafter ~2.4 GB → FP8 drafter ~1.2 GB
    = ~1.2 GB freed per worker (TP=2 → 2.4 GB total) for KV-cache
    headroom

Composition (verified no conflicts):
  - PN21 (DFlash SWA) — different file (dflash.py)
  - PN23 (combine_hidden_states cast) — different method, same file
  - PN24 (aux layer +1) — different file
  - PN40-A (fused per-layer K-norm) — different anchor: PN38 modifies
    `_build_fused_kv_buffers` and adds fallback BEFORE the per-layer
    K-norm loop; PN40-A modifies the K-norm loop itself. Composable.

This patch has 3 sub-patches landing 3 distinct anchors in
`vllm/model_executor/models/qwen3_dflash.py`. All are required (the
changes form one coherent feature; partial application would leave
the model in a broken-quant state).

Site B history (2026-06-11 retirement): the original 4th sub-patch
passed `quant_config=self.quant_config` to DFlashQwen3DecoderLayer.
Upstream made that plumbing native in pin 0.22.1rc1.dev259+g303916e93
(PR #40425 itself is still OPEN — the quant_config wiring landed
separately): pristine qwen3_dflash.py line 228 reads
`self.quant_config = get_draft_quant_config(vllm_config)` and the
constructor kwargs at lines 248-254 already pass
`quant_config=self.quant_config,`. Re-applying Site B would inject a
DUPLICATE `quant_config=` keyword argument → SyntaxError on model
import. The sub-patch was removed; `apply()` now carries an
upstream-presence guard requiring BOTH native lines (loud skip when
absent — A/C/D depend on `self.quant_config` existing).

Anchors (validated against vllm pin 0.22.1rc1.dev259+g303916e93,
originally derived on 0.20.2rc1.dev9+g01d4d1ad3):
  Site A (line 134): `qkv = F.linear(...)` → `qkv, _ = self.qkv_proj(...)`
  Site B — removed 2026-06-11, upstream-native since dev259 (see above)
  Site C (lines 297-313): `_build_fused_kv_buffers` becomes conditional
  Site D (lines 380-385): `precompute_and_store_context_kv` adds quantized fallback

Default OFF (`GENESIS_ENABLE_PN38_DFLASH_QUANT_DRAFTER=1`) until a
quantized DFlash drafter checkpoint exists in the deployment. Strict
no-regression: when `quant_config is None` (BF16 drafter), the new
fallback path is gated and original dense fast-path runs unchanged.

Author: Sandermage (Sander) Barzov Aleksandr — backport of upstream
PR #40425 by infatoshi.
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pN38_dflash_quant_drafter")

GENESIS_PN38_MARKER = (
    "Genesis PN38 DFlash drafter quantization (PR #40425 backport)"
)


# ─── Site A: F.linear → quant-aware module call ────────────────────────────
PN38_A_ANCHOR = (
    "        qkv = F.linear(hidden_states, self.qkv_proj.weight, self.qkv_proj.bias)\n"
)

PN38_A_REPLACEMENT = (
    "        # [Genesis PN38 Site A] PR #40425: quant-aware QKV projection\n"
    "        qkv, _ = self.qkv_proj(hidden_states)\n"
)


# ─── Site B (RETIRED 2026-06-11): pass quant_config to decoder layer ───────
# Upstream-native since pin 0.22.1rc1.dev259+g303916e93. Pristine
# qwen3_dflash.py line 228:
#     self.quant_config = get_draft_quant_config(vllm_config)
# and the DFlashQwen3DecoderLayer kwargs at pristine lines 248-254:
#     DFlashQwen3DecoderLayer(
#         current_vllm_config,
#         config=self.config,
#         cache_config=current_vllm_config.cache_config,
#         quant_config=self.quant_config,
#         prefix=maybe_prefix(prefix, f"layers.{layer_idx + start_layer_id}"),
#     )
# The old anchor below is dead on dev259 (kwarg order changed: prefix now
# comes AFTER config/cache_config/quant_config) AND the functionality is
# native — re-anchoring/re-applying would inject a duplicate
# `quant_config=` keyword argument → SyntaxError on model import. The
# anchor and replacement strings are kept in this file for git-history
# reference but are not registered in sub_patches (P78 Site A convention).
PN38_B_ANCHOR = (
    "                    current_vllm_config,\n"
    "                    prefix=maybe_prefix(prefix, f\"layers.{layer_idx + start_layer_id}\"),\n"
    "                    config=self.config,\n"
    "                )\n"
    "                for layer_idx in range(self.config.num_hidden_layers)\n"
)

PN38_B_REPLACEMENT = (
    "                    current_vllm_config,\n"
    "                    prefix=maybe_prefix(prefix, f\"layers.{layer_idx + start_layer_id}\"),\n"
    "                    config=self.config,\n"
    "                    quant_config=self.quant_config,  # [Genesis PN38 Site B] PR #40425\n"
    "                )\n"
    "                for layer_idx in range(self.config.num_hidden_layers)\n"
)


# ─── Upstream-presence guard for native Site B ─────────────────────────────
# A/C/D depend on `self.quant_config` existing on the model (Site C reads
# it; Site D's fallback assumes quant-aware layer modules from Site B's
# plumbing). On pins where the plumbing is NOT native yet, applying A/C/D
# without B would leave the model in exactly the broken-quant state the
# patch exists to prevent — so apply() requires BOTH native lines and
# skips loudly otherwise. Both strings verified count==1 on pristine
# 0.22.1rc1.dev259+g303916e93 (lines 228 and 251-252 respectively).
PN38_NATIVE_QUANT_INIT = (
    "        self.quant_config = get_draft_quant_config(vllm_config)\n"
)

# Two-line form pins the check to the DFlashQwen3DecoderLayer constructor
# (pristine lines 248-254); a bare `quant_config=self.quant_config,` also
# matches the ReplicatedLinear fc at pristine line 273 and would weaken
# the guard.
PN38_NATIVE_QUANT_KWARG = (
    "                    cache_config=current_vllm_config.cache_config,\n"
    "                    quant_config=self.quant_config,\n"
)


def _native_site_b_status(content: str) -> tuple[bool, str]:
    """Check that BOTH upstream-native Site B lines are present.

    Returns (ok, reason). When ok is False the reason names every
    missing line so the operator sees exactly why PN38 refused to
    apply (loud skip, never partial A/C/D application).
    """
    missing: list[str] = []
    if PN38_NATIVE_QUANT_INIT not in content:
        missing.append(
            "`self.quant_config = get_draft_quant_config(vllm_config)`"
            " (pristine dev259 line 228)"
        )
    if PN38_NATIVE_QUANT_KWARG not in content:
        missing.append(
            "`quant_config=self.quant_config,` decoder-layer kwarg"
            " (pristine dev259 lines 248-254)"
        )
    if missing:
        return False, (
            "upstream-presence guard: native Site B line(s) missing from "
            "qwen3_dflash.py: " + "; ".join(missing) + ". Site B was "
            "retired 2026-06-11 (upstream-native since 0.22.1rc1.dev259); "
            "on a pin without the native plumbing the retired sub-patch "
            "would be required again — refusing partial A/C/D application."
        )
    return True, "native Site B plumbing present (init + decoder-layer kwarg)"


# ─── Site C: _build_fused_kv_buffers becomes conditional ──────────────────
PN38_C_ANCHOR = (
    "        layers_attn = [layer.self_attn for layer in self.layers]\n"
    "        attn0 = layers_attn[0]\n"
    "        has_bias = attn0.qkv_proj.bias is not None\n"
    "\n"
    "        self._hidden_norm_weight = self.hidden_norm.weight.data\n"
    "\n"
    "        # KV projection weights: [num_layers * 2 * kv_size, hidden_size]\n"
    "        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]\n"
    "        self._fused_kv_weight = torch.cat(kv_weights, dim=0)\n"
    "        if has_bias:\n"
    "            kv_biases = [a.qkv_proj.bias[a.q_size :] for a in layers_attn]\n"
    "            self._fused_kv_bias: torch.Tensor | None = torch.cat(kv_biases, dim=0)\n"
    "        else:\n"
    "            self._fused_kv_bias = None\n"
    "\n"
    "        # K-norm weights: list of [head_dim] tensors, one per layer.\n"
    "        self._k_norm_weights = [a.k_norm.weight.data for a in layers_attn]\n"
)

PN38_C_REPLACEMENT = (
    "        # [Genesis PN38 Site C] PR #40425: conditional fused-KV buffers.\n"
    "        # Quantized drafters skip dense fast-path; use per-layer fallback.\n"
    "        layers_attn = [layer.self_attn for layer in self.layers]\n"
    "        attn0 = layers_attn[0]\n"
    "        self._hidden_norm_weight = self.hidden_norm.weight.data\n"
    "        self._use_quantized_kv_fallback = self.quant_config is not None\n"
    "\n"
    "        # K-norm weights: list of [head_dim] tensors, one per layer.\n"
    "        self._k_norm_weights = [a.k_norm.weight.data for a in layers_attn]\n"
    "\n"
    "        if not self._use_quantized_kv_fallback:\n"
    "            has_bias = attn0.qkv_proj.bias is not None\n"
    "\n"
    "            # KV projection weights: [num_layers * 2 * kv_size, hidden_size]\n"
    "            kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]\n"
    "            self._fused_kv_weight = torch.cat(kv_weights, dim=0)\n"
    "            if has_bias:\n"
    "                kv_biases = [a.qkv_proj.bias[a.q_size :] for a in layers_attn]\n"
    "                self._fused_kv_bias: torch.Tensor | None = torch.cat(kv_biases, dim=0)\n"
    "            else:\n"
    "                self._fused_kv_bias = None\n"
)


# ─── Site D: precompute_and_store_context_kv adds quantized fallback ──────
# Anchor: the line just BEFORE `all_kv_flat = F.linear(...)` after eps norm.
# We insert the quantized-fallback block here; if not active, original
# dense path follows unchanged.
PN38_D_ANCHOR = (
    "            self._hidden_norm_weight,\n"
    "            self._rms_norm_eps,\n"
    "        )\n"
    "        all_kv_flat = F.linear(\n"
    "            normed_context_states, self._fused_kv_weight, self._fused_kv_bias\n"
    "        )\n"
)

PN38_D_REPLACEMENT = (
    "            self._hidden_norm_weight,\n"
    "            self._rms_norm_eps,\n"
    "        )\n"
    "        # [Genesis PN38 Site D] PR #40425: quantized DFlash drafter fallback\n"
    "        if self._use_quantized_kv_fallback:\n"
    "            for layer in self.layers:\n"
    "                attn_layer = layer.self_attn\n"
    "                qkv, _ = attn_layer.qkv_proj(normed_context_states)\n"
    "                _, k, v = qkv.split(\n"
    "                    [attn_layer.q_size, attn_layer.kv_size, attn_layer.kv_size], dim=-1,\n"
    "                )\n"
    "                k = attn_layer.k_norm(\n"
    "                    k.view(num_ctx, attn_layer.num_kv_heads, attn_layer.head_dim)\n"
    "                )\n"
    "                v = v.view(num_ctx, attn_layer.num_kv_heads, attn_layer.head_dim)\n"
    "                k_flat = k.contiguous().view(num_ctx, attn_layer.kv_size)\n"
    "                cos_sin_cache = attn_layer.rotary_emb.cos_sin_cache\n"
    "                if cos_sin_cache.dtype != k_flat.dtype:\n"
    "                    cos_sin_cache = cos_sin_cache.to(dtype=k_flat.dtype)\n"
    "                ops.rotary_embedding(\n"
    "                    context_positions,\n"
    "                    k_flat,\n"
    "                    None,\n"
    "                    attn_layer.rotary_emb.head_size,\n"
    "                    cos_sin_cache,\n"
    "                    attn_layer.rotary_emb.is_neox_style,\n"
    "                )\n"
    "                if context_slot_mapping is None:\n"
    "                    continue\n"
    "                attn = attn_layer.attn\n"
    "                kv_cache = attn.kv_cache\n"
    "                attn.impl.do_kv_cache_update(\n"
    "                    attn,\n"
    "                    k_flat.view(\n"
    "                        num_ctx, attn_layer.num_kv_heads, attn_layer.head_dim,\n"
    "                    ),\n"
    "                    v,\n"
    "                    kv_cache,\n"
    "                    context_slot_mapping,\n"
    "                )\n"
    "            return\n"
    "        all_kv_flat = F.linear(\n"
    "            normed_context_states, self._fused_kv_weight, self._fused_kv_bias\n"
    "        )\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/models/qwen3_dflash.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN38 qwen3_dflash.py — quantized drafter support (PR #40425)",
        target_file=str(target),
        marker=GENESIS_PN38_MARKER,
        sub_patches=[
            TextPatch(
                name="pN38_a_qkv_proj_call",
                anchor=PN38_A_ANCHOR,
                replacement=PN38_A_REPLACEMENT,
                required=True,
            ),
            # pN38_b_pass_quant_config removed 2026-06-11 — Site B
            # upstream-native since dev259 (see Site B history block);
            # _native_site_b_status() guards the native lines instead.
            TextPatch(
                name="pN38_c_conditional_fused_kv",
                anchor=PN38_C_ANCHOR,
                replacement=PN38_C_REPLACEMENT,
                required=True,
            ),
            TextPatch(
                name="pN38_d_quant_fallback",
                anchor=PN38_D_ANCHOR,
                replacement=PN38_D_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN38",
            # Self-collision lint (triage plan §6 2026-06-11): former entry
            # "_use_quantized_kv_fallback" is the #40425 helper baked
            # verbatim by our own backport replacement — it cannot
            # distinguish a real upstream merge from our residue (false
            # "upstream_merged" skip, PN369 class). Real-merge detection
            # via required-anchor mismatch (Layer 5) + preflight deep-diff.
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN38 — quantized DFlash drafter support."""
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN38")
    log_decision("PN38", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "qwen3_dflash.py not resolvable"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"

    # Upstream-presence guard (Site B retired 2026-06-11): require BOTH
    # native quant_config lines before landing A/C/D. Loud skip — the
    # reason names every missing line.
    try:
        with open(patcher.target_file, encoding="utf-8") as fh:
            content = fh.read()
    except OSError as e:
        return "skipped", f"target unreadable: {patcher.target_file} ({e})"
    native_ok, guard_reason = _native_site_b_status(content)
    if not native_ok:
        log.warning("[Genesis PN38] %s", guard_reason)
        return "skipped", guard_reason

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "PN38 applied: 3 sub-patches (Sites A/C/D) into "
            "qwen3_dflash.py — quantized DFlash drafter support "
            "(PR #40425 backport). Site B (quant_config plumbing) "
            "upstream-native since 0.22.1rc1.dev259, presence-guarded. "
            "Today no-op for BF16 drafters; ready for FP8/NVFP4 drafter "
            "checkpoints when available. Composes with PN40-A (different "
            "anchor surfaces)."
        ),
        patch_name=patcher.patch_name,
    )
