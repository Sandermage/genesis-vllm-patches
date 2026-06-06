# SPDX-License-Identifier: Apache-2.0
"""G4_67 — backport of upstream PR #40914 (TQ K+1 spec-verify routing).

================================================================
PROBLEM
================================================================

When speculative decoding is active (MTP ``num_speculative_tokens=K``),
the verify pass produces uniform-query batches with ``max_query_len =
K+1`` (e.g. K=3 → q_len=4 per request) where ``max_seq_len > max_query
_len`` (each request has prior cached KV).

Upstream ``TurboQuantAttentionImpl._prefill_attention`` reads
``query_start_loc.tolist()`` which:

  1. Forces GPU→CPU sync incompatible with active CUDA stream capture.
  2. Was the root cause for issue #40880 (degenerate token cascades
     on Qwen3.6-MoE under MTP=3 + FULL_AND_PIECEWISE cudagraph).

Without this fix, operators must use ``--compilation-config
'{"cudagraph_mode":"NONE"}'`` to avoid degenerate output, which causes
a **4.9× TPS slowdown** (validated 2026-05-17 on A5000:
118 TPS production → 24 TPS overlay+NONE).

================================================================
FIX (PR #40914 cherry-pick, Gemma 4-specific)
================================================================

This is the **Genesis Gemma 4 backport** of my upstream PR #40914
(https://github.com/vllm-project/vllm/pull/40914). It monkey-patches
``TurboQuantAttentionImpl.forward`` to inject a spec-verify dispatch
block BEFORE the default ``_prefill_attention`` branch.

When the dispatch predicate matches::

    is_prefill && num_decodes==0
    && 1 < max_query_len <= 16
    && max_seq_len > max_query_len
    && N % max_query_len == 0
    && query_start_loc.shape[0] == B+1

The patch:

  1. Builds ``synth_seq_lens`` + ``synth_block_table`` entirely on-GPU
     (no ``.tolist()`` or ``.item()`` sync) via:

       synth_seq_lens[req*K1+i] = base_seq_lens[req] - K1 + 1 + i
       synth_block_table[req*K1+i] = block_table[req]

  2. Calls ``triton_turboquant_decode_attention`` with synth args. The
     decode kernel handles compressed K+V cache lookup natively and is
     cudagraph-safe.

  3. Returns early — bypassing the default continuation prefill branch
     that has the GPU→CPU sync.

================================================================
ALTERNATIVES & RELATIONSHIPS
================================================================

  * **P67 / P67b** — Genesis-original alternative (new multi-query
    Triton kernel). Better performance (+32% TPS empirically) but
    requires custom kernel maintenance. Pin-gated to dev16-dev93 in
    registry — NOT applicable on dev371.

  * **PR #40914** (our upstream submission) — equivalent fix via synth_seq_lens
    routing through EXISTING decode kernel. OPEN, awaits maintainer
    review. This G4_67 is the Genesis backport of it for Gemma 4
    while we wait for merge.

  * **cudagraph_mode=NONE** — operator workaround. Works but 4.9×
    slower per Genesis bench 2026-05-17.

================================================================
DEPENDENCIES
================================================================

  * Requires **G4_60b** (turboquant_attn.py overlay) OR upstream dev371
    ``TurboQuantAttentionImpl`` to be present. G4_67 monkey-patches its
    ``forward`` method.

  * Composable with **G4_61** (shared workspace) — buffers acquired via
    WorkspaceManager OR via cached ``layer._tq_*_buf`` (G4_67 prefers
    pre-existing cached buffers per gemini-code-assist review on
    PR #40914 to avoid per-call ``torch.empty`` allocations breaking
    CUDA graph replay).

  * Composable with **G4_62** (warmup) — both improve cudagraph reliability.

================================================================
SCOPE
================================================================

Active only when ``GENESIS_ENABLE_G4_67_TQ_SPEC_VERIFY_ROUTE=1``.

For non-speculative workloads (no MTP / Eagle3 / DFlash), the dispatch
predicate never matches → no-op. For pure decode batches (``num_decodes
> 0``), predicate doesn't match → no-op.

The patch is **additive** — it adds an early-return path before the
default ``_prefill_attention`` continuation. If predicate doesn't
match, original behaviour preserved.

================================================================
EXPECTED IMPACT (per upstream PR + Genesis bench)
================================================================

For configurations using:
  * vllm TURBOQUANT backend (overlay G4_60b)
  * MTP num_speculative_tokens >= 2
  * FULL_AND_PIECEWISE cudagraph (not NONE)

Expected throughput recovery vs cudagraph=NONE workaround:
  * Removes 4.9× slowdown observed in Genesis bench 2026-05-17
  * Matches or exceeds production G4_19 wrapper-stack TPS
  * Enables full PR #42637 overlay path benefits to materialize

For non-TQ-backend workloads (G4_19 wrapper path): no effect (decode
kernel not in flow).

================================================================
RISK
================================================================

  * **Buffer caching invariant**: G4_67 uses ``getattr(layer,
    "_tq_*_buf", None)`` to access pre-allocated decode buffers. If
    G4_61 (shared workspace) is active and uses WorkspaceManager
    instead of per-layer buffers, those will be None and the launcher
    falls through to either WorkspaceManager.get_simultaneous() (with
    G4_60c overlay) or per-call torch.empty (cudagraph-unsafe). Plan
    section "Future overlay" tests this combination.

  * **Predicate tightness**: ``1 < max_query_len <= 16`` upper bound
    matches typical MTP K values. If operator uses K > 15
    (rare — diminishing returns), predicate misses and falls through.

  * **Hardware-specific validation**: PR #40914 body says "tested ONLY
    on NVIDIA Ampere SM 8.6 (RTX A5000)". Hopper/Blackwell validation
    requested but not done. Same scope applies to G4_67.

================================================================
REFERENCES
================================================================

  * Upstream PR (ours): https://github.com/vllm-project/vllm/pull/40914
  * Original issue: https://github.com/vllm-project/vllm/issues/40880
  * Genesis P67 / P67b (alternative fix): see
    ``integrations/attention/turboquant/p67_turboquant_multi_query_kernel.py``
  * Genesis bench 2026-05-17: documents 4.9× slowdown from
    cudagraph=NONE workaround that this patch removes.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.turboquant.g4_67_tq_spec_verify_route")

GENESIS_G4_67_MARKER = (
    "Genesis G4_67 TurboQuantAttentionImpl spec-verify K+1 routing "
    "through triton_turboquant_decode_attention (PR #40914 backport)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_67_TQ_SPEC_VERIFY_ROUTE"
_APPLIED = False
_ORIGINAL_FORWARD = None


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Monkey-patch TurboQuantAttentionImpl.forward with spec-verify dispatch."""
    global _APPLIED, _ORIGINAL_FORWARD

    if not _env_enabled():
        return "skipped", (
            f"G4_67 disabled (set {_ENV_ENABLE}=1 to enable TQ K+1 "
            "spec-verify routing — PR #40914 backport)"
        )

    if _APPLIED:
        return "applied", "G4_67 already installed (idempotent)"

    try:
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionImpl,
        )
    except ImportError as e:
        return "skipped", (
            f"vllm.v1.attention.backends.turboquant_attn not importable: {e}"
        )

    original = TurboQuantAttentionImpl.forward
    if getattr(original, "_genesis_g4_67_wrapped", False):
        _APPLIED = True
        return "applied", "G4_67 already wrapped (idempotent)"
    _ORIGINAL_FORWARD = original

    def _wrapped_forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
        positions=None,
        **extra_kwargs,
    ):
        """Inject spec-verify K+1 routing before original forward.

        Verbatim port of PR #40914 dispatch block.
        """
        # Late imports to keep cold-import surface minimal.
        import torch

        # Only meaningful when attn_metadata exists and points to a
        # prefill-shape batch. Pure decode (num_decodes > 0) skipped.
        if attn_metadata is None:
            return original(
                self, layer, query, key, value, kv_cache, attn_metadata,
                output=output, output_scale=output_scale, output_block_scale=output_block_scale, **extra_kwargs,
            )

        num_decodes = getattr(attn_metadata, "num_decodes", 0)
        is_prefill = getattr(attn_metadata, "is_prefill", False)
        max_q = getattr(attn_metadata, "max_query_len", 0) or 0
        max_s = getattr(attn_metadata, "max_seq_len", 0) or 0
        qsl = getattr(attn_metadata, "query_start_loc", None)
        seq_lens_t = getattr(attn_metadata, "seq_lens", None)
        block_table_t = getattr(attn_metadata, "block_table", None)

        # Predicate from PR #40914 (lines 228-237 of forward).
        N = query.shape[0] if query is not None and query.ndim >= 1 else 0
        _spec_verify_eligible = (
            is_prefill
            and num_decodes == 0
            and 1 < max_q <= 16
            and max_s > max_q
            and N > 0
            and (N % max_q) == 0
            and qsl is not None
            and seq_lens_t is not None
            and block_table_t is not None
        )

        if not _spec_verify_eligible:
            return original(
                self, layer, query, key, value, kv_cache, attn_metadata,
                output=output, output_scale=output_scale, output_block_scale=output_block_scale, **extra_kwargs,
            )

        K_PLUS_1 = int(max_q)
        B = N // K_PLUS_1
        if qsl.shape[0] != B + 1:
            return original(
                self, layer, query, key, value, kv_cache, attn_metadata,
                output=output, output_scale=output_scale, output_block_scale=output_block_scale, **extra_kwargs,
            )

        # === Spec-verify K+1 path (PR #40914) ===
        try:
            from vllm.v1.attention.ops.triton_turboquant_decode import (
                triton_turboquant_decode_attention,
            )

            # Reshape query to (N, num_heads, head_size).
            num_heads = getattr(self, "num_heads", None)
            head_size = getattr(self, "head_size", None)
            if num_heads is None or head_size is None:
                return original(
                    self, layer, query, key, value, kv_cache, attn_metadata,
                    output=output, output_scale=output_scale, output_block_scale=output_block_scale, **extra_kwargs,
                )
            _q_flat = query[:N].view(N, num_heads, head_size)

            # synth_seq_lens — purely on-GPU
            _offs = torch.arange(
                K_PLUS_1,
                device=query.device,
                dtype=seq_lens_t.dtype,
            )
            _synth_seq_lens = (
                seq_lens_t[:B, None] - K_PLUS_1 + 1 + _offs[None, :]
            ).reshape(-1)
            _synth_block_table = block_table_t[:B].repeat_interleave(
                K_PLUS_1, dim=0,
            )

            # Reuse cached buffers (gemini review on PR #40914).
            _mid_o_buf = getattr(layer, "_tq_mid_o_buf", None)
            _output_buf = getattr(layer, "_tq_output_buf", None)
            _lse_buf = getattr(layer, "_tq_lse_buf", None)

            # Pi / centroids / PiT — set on layer by _ensure_on_device
            Pi = getattr(layer, "_tq_Pi", None)
            centroids = getattr(layer, "_tq_centroids", None)
            PiT = getattr(layer, "_tq_PiT", None)
            if Pi is None or centroids is None:
                # Layer not yet warmed; fall through.
                return original(
                    self, layer, query, key, value, kv_cache, attn_metadata,
                    output=output, output_scale=output_scale, output_block_scale=output_block_scale, **extra_kwargs,
                )

            kwargs = dict(
                query=_q_flat,
                kv_cache=kv_cache,
                block_table=_synth_block_table,
                seq_lens=_synth_seq_lens,
                Pi=Pi,
                centroids=centroids,
                scale=self.scale,
                mse_bits=self.tq_config.key_mse_bits,
                key_packed_size=self.tq_config.key_packed_size,
                value_quant_bits=self.tq_config.effective_value_quant_bits,
                key_fp8=self.tq_config.key_fp8,
                norm_correction=self.tq_config.norm_correction,
                PiT=PiT,
                mid_o_buf=_mid_o_buf,
                output_buf=_output_buf,
                lse_buf=_lse_buf,
                max_num_kv_splits=self.max_num_kv_splits,
            )
            # Pass buf_holder only if pre-PR #40798 launcher (some signatures
            # already removed this; G4_61 wraps the launcher and pops it).
            try:
                attn_out = triton_turboquant_decode_attention(
                    buf_holder=layer, **kwargs,
                )
            except TypeError:
                # PR #40798 removed buf_holder kwarg — retry without it.
                attn_out = triton_turboquant_decode_attention(**kwargs)
            return attn_out
        except Exception as e:  # noqa: BLE001
            log.warning(
                "[G4_67] spec-verify routing failed (%r); falling back to "
                "default _prefill_attention path",
                e,
            )
            return original(
                self, layer, query, key, value, kv_cache, attn_metadata,
                output=output, output_scale=output_scale, output_block_scale=output_block_scale, **extra_kwargs,
            )

    _wrapped_forward._genesis_g4_67_wrapped = True  # type: ignore[attr-defined]
    TurboQuantAttentionImpl.forward = _wrapped_forward  # type: ignore[method-assign]

    _APPLIED = True
    log.info(
        "[G4_67] TurboQuantAttentionImpl.forward wrapped with spec-verify "
        "K+1 routing through triton_turboquant_decode_attention "
        "(PR #40914 backport)."
    )
    return "applied", (
        "G4_67 installed: MTP K+1 verify batches route through decode "
        "kernel (cudagraph-safe, no GPU↔CPU sync)."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    """Best-effort revert (test isolation only)."""
    global _APPLIED, _ORIGINAL_FORWARD
    if not _APPLIED or _ORIGINAL_FORWARD is None:
        return False
    try:
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionImpl,
        )

        TurboQuantAttentionImpl.forward = _ORIGINAL_FORWARD  # type: ignore[method-assign]
    except Exception:  # noqa: BLE001
        return False
    _APPLIED = False
    _ORIGINAL_FORWARD = None
    return True


__all__ = [
    "GENESIS_G4_67_MARKER",
    "apply",
    "is_applied",
    "revert",
]
