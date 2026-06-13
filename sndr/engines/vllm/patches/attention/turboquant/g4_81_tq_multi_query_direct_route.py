# SPDX-License-Identifier: Apache-2.0
"""G4_81 — TQ multi-query DIRECT decode routing (vllm#45144 blueprint).

================================================================
PROBLEM
================================================================

MTP K=3 x TurboQuant on Gemma-4-31B dense is blocked: spec-verify
batches (uniform ``max_query_len = K+1`` with prior cached KV) reach
``TurboQuantAttentionImpl.forward`` and fall into the per-request
``_prefill_attention`` continuation path, which

  1. reads ``query_start_loc.tolist()`` / ``seq_lens.tolist()``
     (GPU->CPU sync, cudagraph-hostile — the issue #40880 class), and
  2. routes through ``_decode_prefill_from_cache`` whose cache-read
     assumption broke under capture (PN255/PN256 cycle 2026-05-18),
     forcing the P65 v2 cudagraph downgrade + PN256 raw-KV detour —
     a 4.9x-slowdown-class workaround (Genesis bench 2026-05-17).

The result: the 31B TQ profile cannot run MTP at competitive decode
TPS; G4_80 (fp8e5m2 weight-only KV) landed as the non-TQ fallback.

================================================================
BLUEPRINT — upstream vllm#45144 (studied via gh pr view/diff 2026-06-11)
================================================================

#45144 (ROCm: MTP + fp8 KV + AITER Shuffle-KV) hits the IDENTICAL
problem class: their multi-token decode branch ASSERTED out for the
SKV layout because the generic dispatch path silently truncated
multi-query batches (HIP kernel hardcodes ``mtp=1`` — "silently
processes only the first query token"). Their fix:

  1. DIRECT routing in the decode forward path — bypass
     ``paged_attention_common`` and call ``aiter.pa_fwd_asm`` directly
     with ``max_qlen=decode_max_query_len`` and
     ``qo_indptr=query_start_loc[:num_decodes+1]``.
  2. ``_maybe_allocate_shuffle_scale()`` applied to BOTH ``build()``
     AND ``build_for_drafting()`` — their drafter SIGSEGV'd because
     only ``build()`` had the fix.

This is the second independent upstream validation of the Genesis
P67/P67b technique class (PROD-active on Qwen3.6-35B, +32% TPS):
route uniform multi-query verify batches through the single-token
decode kernel machinery instead of the prefill path.

ADAPTATION (iron rule #10 — adapt, don't blind-copy): we do not vendor
any ROCm code. Our TQ split-KV decode kernel has no native
``qo_indptr``/``max_qlen`` multi-query mode, so DIRECT routing takes
the established Genesis form: synthetic per-token expansion (P67b
upstream path / G4_67 / our upstream PR #40914). Each of the B*(K+1)
query rows becomes a virtual single-token decode:

    synth_seq_lens[i*K1 + j]   = seq_lens[i] - K1 + 1 + j
    synth_block_table[i*K1+j]  = block_table[i]

================================================================
KERNEL CONTRACT VERIFICATION (iron rule #11 — verified 2026-06-11)
================================================================

Verified against BOTH launcher generations:

  * Overlay (repo): ``sndr/engines/vllm/patches/attention/turboquant/
    overlays/pr42637/triton_turboquant_decode.py:718`` —
    ``triton_turboquant_decode_attention(query[B,Hq,D], kv_cache,
    block_table[B,max_blocks], seq_lens[B], ..., buf_holder,
    max_num_kv_splits, sliding_window, mm_prefix_range)``.
  * Pristine pin 0.22.1rc1.dev259+g303916e93:
    ``v1/attention/ops/triton_turboquant_decode.py:486`` — same
    contract MINUS ``sliding_window`` / ``mm_prefix_range``.

Contract facts that make the synthetic expansion exact:

  * q-layout: ``forward`` receives token-major rows, per-request
    consecutive (``query[:N].view(N, Hq, D)``); the launcher calls
    ``.contiguous()`` (fp8-key path) or rotates via GEMM into a fresh
    contiguous tensor (MSE path), then passes ``stride(0)/stride(1)``
    — row-major (Hq*D, D, 1) regardless of B = number of virtual rows.
  * Stage-1 kernel (``_tq_decode_stage1``) loads per-row
    ``seq_len = Seq_lens_ptr[bid]`` and masks ``kv_offs < split_end <=
    seq_len`` — each virtual row attends to exactly its causal prefix.
  * ``query_pos = seq_len - 1`` (sliding-window + mm-prefix masking,
    overlay kernel lines 270-293): the virtual row's query IS the last
    token of its synthetic sequence, so per-row sliding-window and
    mm-prefix semantics are exact, not approximate.
  * ``block_table.repeat_interleave(K1, dim=0)`` is contiguous;
    ``stride(0)`` is passed; ``mm_prefix_range`` is indexed packed
    (``bid * MAX_MM_RANGES * 2``) so the expanded copy must be (and
    is) contiguous as well.
  * Grids ``(B*K1, Hq, NUM_KV_SPLITS)`` and ``(B*K1, Hq)`` are far
    below CUDA grid limits at our operating point (max_num_seqs<=8,
    K1<=16); mid_o is (B*K1, Hq, S, D+1) fp32 — a few MiB.

KV completeness precondition: the decode kernel reads the current K+1
tokens FROM CACHE. ``do_kv_cache_update`` (separate custom op, runs
BEFORE attention forward) writes ``N = slot_mapping.shape[0]`` rows,
and slot_mapping is sliced to ``num_actual_tokens`` — which equals
B*(K+1) for BOTH classifications this patch routes (prefill-classified
verify batches today; decode-classified ones if spec-as-decode flips).
KV-sharing layers (Gemma-4 MTP drafter) skip the write (PN242) and
read the target layer's cache, which the target already populated —
cache-only reads are exactly correct for them too.

================================================================
WHAT THIS PATCH DOES
================================================================

Monkey-patches ``TurboQuantAttentionImpl.forward`` (runtime hook, no
TextPatcher) with a dispatch wrapper that intercepts BOTH spec-verify
shapes BEFORE the original prefill/mixed dispatch:

  (a) prefill-classified (today's builder,
      ``supports_spec_as_decode=False``): ``is_prefill`` and
      ``num_decodes == 0`` and uniform K+1;
  (b) decode-classified (the vllm#45144 shape — guards the future
      ``supports_spec_as_decode=True`` builder generation):
      ``num_decodes > 0`` and ``num_decode_tokens ==
      num_actual_tokens`` and uniform K+1.

Uniformity is proven arithmetically without GPU sync: sum of B
query lens each <= max_query_len equals B * max_query_len => every
len == max_query_len (CPU ints from metadata only).

Routed batches go DIRECTLY to the module-level
``triton_turboquant_decode_attention`` binding (fetched at dispatch
time, so the P40 grouped-kernel wrapper is honored when applied) with
synthetic args. Anything else — and ANY routing failure — falls
through to the original forward unchanged (safety contract:
half-routed batches never happen; worst case is today's behavior).

Genesis extras over the G4_67 predecessor (PR #40914 backport):

  * sliding-window layers routed correctly (Gemma-4 interleaved
    layout) — ``sliding_window`` forwarded when the live launcher
    supports it; route REFUSED (clean fallback) when it does not;
  * mm-prefix ranges expanded per virtual row and forwarded under the
    same capability gate (G4_79 / G4_60L lineage);
  * engine output-buffer contract respected: result is written into
    ``output`` (2D or 3D) and ``output`` is returned — not a raw 3D
    tensor (the G4_67 contract gap from the 2026-05-18 diagnosis);
  * per-K1 SimpleNamespace buffer holders on the impl — never the
    layer's decode buffers (P67b illegal-address lesson: layer buffers
    are sized for max_num_seqs rows; B*K1 exceeds that);
  * launcher capability inspection cached per binding (P40-compatible).

================================================================
PATCH CHECKLIST (verify on every pin bump / builder change)
================================================================

  1. Launcher signature: ``triton_turboquant_decode_attention``
     keyword names unchanged (query/kv_cache/block_table/seq_lens/...).
     Capability probe handles optional-arg drift automatically.
  2. ``TurboQuantMetadata`` field names: num_actual_tokens,
     max_query_len, max_seq_len, num_decodes, num_decode_tokens,
     is_prefill, query_start_loc, seq_lens, block_table.
  3. build() coverage: ``TurboQuantMetadataBuilder.build`` populates
     the split fields from ``split_decodes_and_prefills`` — shape (a)
     predicates depend on it.
  4. build_for_drafting() coverage (vllm#45144 lesson — their drafter
     crashed because build_for_drafting() missed the fix build() had):
     on this pin the TQ builder does NOT override build_for_drafting;
     the base ``AttentionMetadataBuilder.build_for_drafting``
     (v1/attention/backend.py:636) delegates to ``self.build(...,
     fast_build=True)``, so drafter-built metadata flows through the
     same classified fields and stays covered by this wrapper. IF a
     future pin adds a TQ-specific build_for_drafting override, verify
     it populates the same split fields before trusting shape (b).
  5. ``supports_spec_as_decode`` remains False in the TQ builder
     (pristine turboquant_attn.py:203 == overlay :370). If a pin flips
     it True (PN243 retry / upstream), shape (b) takes over —
     re-validate the mixed-batch demotion story BEFORE enabling,
     because the original mixed branch slices block_table by
     num_decodes while q rows are per-token (silent corruption) and
     crashes on the empty prefill-portion ``.max()`` (the exact PN243
     "empty-tensor max()" failure).
  6. Kernel masking: ``query_pos = seq_len - 1`` per row (overlay
     triton_turboquant_decode.py:270/287) — synthetic expansion
     correctness depends on it.

================================================================
RELATIONSHIPS
================================================================

  * **G4_67** (PR #40914 backport) — predecessor, same technique on
    shape (a) only; lacks sliding-window/mm-prefix forwarding and
    returns a raw 3D tensor instead of writing ``output``. G4_81 is
    its surgical successor: enable ONE of the two (G4_81 wraps outer
    and intercepts first when both are on; benign but redundant).
  * **P67/P67b** — Qwen-family equivalent (PROD +32% TPS on 35B);
    G4_81 reuses its uniformity arithmetic and buffer-holder pattern.
  * **P40** — grouped GQA stage-1 kernel dispatcher; honored because
    the launcher binding is fetched from the ops module per dispatch.
  * **PN240/PN255/PN256/P65 v2** — diagnosis/workaround generation
    this patch supersedes on the verify path (keep them; they cover
    the padded-decode and non-uniform shapes G4_81 refuses).
  * **G4_79/G4_31/G4_80** — 31B boot-gate companions (mm validity gate
    + dtype preserve + non-TQ fallback). The TQ profile that consumes
    G4_81 needs G4_79's mm-prefix unblock first.

================================================================
EXPECTED IMPACT / RISK
================================================================

Expected: +20-40% decode TPS on the Gemma-4-31B dense TQ profile
(roadmap 2026-06-11 chunk-2 Theme A) by replacing the per-request
Python continuation loop with one fused kernel launch per layer and
removing the GPU->CPU syncs from the verify hot path. #45144's
acceptance data (K=2: ~0.75, K=3: ~0.50) makes a K=2-vs-K=3 bench a
free follow-up experiment.

Risk: opt-in (default OFF), per-batch predicate is conservative,
every failure falls through to the original path. Validate with the
#45100 min-len/short-output distribution scoring (spec-decode
corruption detector) before any default-on discussion. First bring-up
under PIECEWISE/eager (P65 v2 ON), as the 31B profile runs today.

Opt-in: ``GENESIS_ENABLE_G4_81_TQ_MQ_DIRECT_ROUTE=1`` (default OFF).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import inspect
import logging
import os

log = logging.getLogger("genesis.turboquant.g4_81_tq_mq_direct_route")

GENESIS_G4_81_MARKER = (
    "Genesis G4_81 TurboQuantAttentionImpl multi-query DIRECT decode "
    "routing (vllm#45144 blueprint, synthetic per-token expansion)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_81_TQ_MQ_DIRECT_ROUTE"
_WRAP_ATTR = "_genesis_g4_81_wrapped"

_MAX_K1 = 16  # MTP K+1 upper bound (K > 15 has diminishing returns)

_APPLIED = False
_ORIGINAL_FORWARD = None


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


# ─── Pure helpers (torch-less, unit-tested) ───────────────────────────


def _classify_spec_verify_batch(
    num_actual_tokens: int,
    max_query_len: int,
    max_seq_len: int,
    num_decodes: int,
    num_decode_tokens: int,
    is_prefill: bool,
    query_start_loc_len: int,
) -> tuple[int, int] | None:
    """Classify a batch as a routable uniform multi-query spec-verify
    batch. Returns ``(B, K1)`` or ``None``.

    Uses ONLY CPU-resident ints from the metadata — no GPU sync. The
    uniformity proof is arithmetic: B request query-lens, each
    <= max_query_len, summing to B * max_query_len => all equal.

    Shape (a) — prefill-classified (supports_spec_as_decode=False,
    today's builder): the whole batch is B uniform K+1 requests with
    prior cache (max_seq_len > max_query_len excludes first-chunk
    prompts, which the flash prefill path serves better).

    Shape (b) — decode-classified (the vllm#45144 shape; future
    supports_spec_as_decode=True builder generation): the decode split
    covers the WHOLE batch (num_decode_tokens == num_actual_tokens) and
    is uniform. Mixed batches are refused — the synthetic expansion
    must never half-route a batch.
    """
    if num_actual_tokens <= 0:
        return None
    if not (1 < max_query_len <= _MAX_K1):
        return None
    if num_actual_tokens % max_query_len != 0:
        return None

    if num_decodes > 0:
        # Shape (b): pure decode-classified multi-query batch.
        if num_decode_tokens != num_actual_tokens:
            return None  # mixed decode+prefill — refuse
        if num_actual_tokens != num_decodes * max_query_len:
            return None  # non-uniform decode portion — refuse
        return num_decodes, max_query_len

    # Shape (a): prefill-classified uniform K+1 verify batch.
    if not is_prefill:
        return None
    if max_seq_len <= max_query_len:
        return None  # first-chunk prefill — flash path serves it
    b = num_actual_tokens // max_query_len
    if query_start_loc_len != b + 1:
        return None  # uniformity proof needs exactly B requests
    return b, max_query_len


def _launcher_params(fn) -> frozenset[str]:
    """Parameter names accepted by the live decode launcher binding.

    The binding drifts across generations (pristine pin lacks
    sliding_window/mm_prefix_range; the P40 wrapper has its own
    signature), so capabilities are probed, never assumed.
    """
    try:
        return frozenset(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return frozenset()


def _route_refusal(
    caps: frozenset[str],
    sliding_window: int | None,
    has_mm_prefix: bool,
) -> str | None:
    """Return a refusal reason when routing would LOSE masking semantics
    on the live launcher, else None.

    Correctness over speed: a sliding-window layer routed through a
    launcher that cannot mask the window would attend outside it.
    """
    if sliding_window is not None and "sliding_window" not in caps:
        return (
            "sliding-window layer but live launcher has no sliding_window "
            "parameter (pristine-pin or P40 binding)"
        )
    if has_mm_prefix and "mm_prefix_range" not in caps:
        return (
            "mm_prefix ranges present but live launcher has no "
            "mm_prefix_range parameter"
        )
    return None


def _synth_seq_len_ref(seq_len: int, k1: int, j: int) -> int:
    """Reference formula: virtual row j of a K1-token verify request
    attends to its causal prefix — its query is the LAST token of the
    synthetic sequence (kernel sets query_pos = seq_len - 1)."""
    return seq_len - k1 + 1 + j


def _build_synth_args(seq_lens, block_table, b: int, k1: int):
    """Build (synth_seq_lens, synth_block_table) on-GPU, no sync.

    synth_seq_lens[i*K1+j] = seq_lens[i] - K1 + 1 + j  (see
    _synth_seq_len_ref); synth_block_table replicates each request row
    K1 times (contiguous — the kernel indexes mm/block tables packed).
    """
    import torch

    offs = torch.arange(k1, device=seq_lens.device, dtype=seq_lens.dtype)
    synth_sl = (seq_lens[:b, None] - k1 + 1 + offs[None, :]).reshape(-1)
    synth_bt = block_table[:b].repeat_interleave(k1, dim=0)
    return synth_sl, synth_bt


# ─── Routing core ─────────────────────────────────────────────────────


def _try_route(self, layer, query, kv_cache, attn_metadata, output):
    """Attempt the multi-query direct route. Returns the engine output
    buffer on success, None when the batch is not routable (caller
    falls through to the original forward). Raises on unexpected
    internal errors (caller catches and falls through)."""
    if attn_metadata is None or query is None:
        return None

    qsl = getattr(attn_metadata, "query_start_loc", None)
    route = _classify_spec_verify_batch(
        num_actual_tokens=getattr(attn_metadata, "num_actual_tokens", 0) or 0,
        max_query_len=getattr(attn_metadata, "max_query_len", 0) or 0,
        max_seq_len=getattr(attn_metadata, "max_seq_len", 0) or 0,
        num_decodes=getattr(attn_metadata, "num_decodes", 0) or 0,
        num_decode_tokens=getattr(attn_metadata, "num_decode_tokens", 0) or 0,
        is_prefill=bool(getattr(attn_metadata, "is_prefill", False)),
        query_start_loc_len=(qsl.shape[0] if qsl is not None else 0),
    )
    if route is None:
        return None
    b, k1 = route
    n = b * k1

    seq_lens_t = getattr(attn_metadata, "seq_lens", None)
    block_table_t = getattr(attn_metadata, "block_table", None)
    if seq_lens_t is None or block_table_t is None:
        return None

    # Live launcher binding — fetched per dispatch so the P40 grouped
    # wrapper (module-attr rebind) is honored. Capability set cached
    # per binding identity.
    import importlib

    ops_mod = importlib.import_module(
        "vllm.v1.attention.ops.triton_turboquant_decode"
    )
    launcher = getattr(ops_mod, "triton_turboquant_decode_attention", None)
    if launcher is None:
        return None
    cached = getattr(self, "_genesis_g4_81_sigcache", None)
    if cached is None or cached[0] is not launcher:
        cached = (launcher, _launcher_params(launcher))
        self._genesis_g4_81_sigcache = cached
    caps = cached[1]

    sliding_window = getattr(self, "sliding_window", None)
    mm_prefix_t = getattr(attn_metadata, "mm_prefix_range_tensor", None)
    refusal = _route_refusal(caps, sliding_window, mm_prefix_t is not None)
    if refusal is not None:
        if not getattr(self, "_genesis_g4_81_refusal_logged", False):
            self._genesis_g4_81_refusal_logged = True
            log.warning(
                "[G4_81] route refused (%s) — falling through to original "
                "forward (logged once per impl)", refusal,
            )
        return None

    # TQ codebook tensors must be device-resident before kernel launch
    # (idempotent one-time migration, same call the original makes).
    self._ensure_on_device(layer, query.device)
    pi = getattr(layer, "_tq_Pi", None)
    pi_t = getattr(layer, "_tq_PiT", None)
    centroids = getattr(layer, "_tq_centroids", None)
    if pi is None or centroids is None:
        return None  # layer not warmed — original path handles it

    q = query[:n].view(n, self.num_heads, self.head_size)
    synth_sl, synth_bt = _build_synth_args(seq_lens_t, block_table_t, b, k1)

    # Per-K1 buffer holder on the impl — NEVER the layer's decode
    # buffers (P67b illegal-address lesson: those are sized for
    # max_num_seqs rows; n = B*K1 exceeds that). The launcher
    # allocates on first call and caches on the holder; the >= n
    # shape check lets buffers grow monotonically.
    holders = getattr(self, "_genesis_g4_81_holders", None)
    if holders is None:
        holders = {}
        self._genesis_g4_81_holders = holders
    holder = holders.get(k1)
    if holder is None:
        import types as _types

        holder = _types.SimpleNamespace()
        holders[k1] = holder

    kwargs = dict(
        query=q,
        kv_cache=kv_cache,
        block_table=synth_bt,
        seq_lens=synth_sl,
        Pi=pi,
        centroids=centroids,
        scale=self.scale,
        mse_bits=self.tq_config.key_mse_bits,
        key_packed_size=self.tq_config.key_packed_size,
        value_quant_bits=self.tq_config.effective_value_quant_bits,
        key_fp8=self.tq_config.key_fp8,
        norm_correction=self.tq_config.norm_correction,
        PiT=pi_t,
        mid_o_buf=getattr(holder, "_tq_mid_o_buf", None),
        output_buf=getattr(holder, "_tq_output_buf", None),
        lse_buf=getattr(holder, "_tq_lse_buf", None),
        max_num_kv_splits=self.max_num_kv_splits,
    )
    if "buf_holder" in caps:
        kwargs["buf_holder"] = holder
    if sliding_window is not None:
        kwargs["sliding_window"] = sliding_window
    if mm_prefix_t is not None:
        kwargs["mm_prefix_range"] = mm_prefix_t[:b].repeat_interleave(
            k1, dim=0
        )

    attn_out = launcher(**kwargs)

    # Engine output-buffer contract (the G4_67 gap): write into
    # `output` (2D or 3D) and return it — never a raw 3D tensor.
    import torch

    if output is None:
        output = torch.zeros(
            query.shape[0],
            self.num_heads * self.head_size,
            dtype=query.dtype,
            device=query.device,
        )
    attn_flat = attn_out.view(n, self.num_heads, self.head_size)
    if output.ndim == 3:
        output[:n] = attn_flat.to(output.dtype)
    else:
        output[:n] = attn_flat.reshape(n, -1).to(output.dtype)
    return output


# ─── apply / revert ───────────────────────────────────────────────────


def apply() -> tuple[str, str]:
    """Wrap TurboQuantAttentionImpl.forward with multi-query routing."""
    global _APPLIED, _ORIGINAL_FORWARD

    if not _env_enabled():
        return "skipped", (
            f"G4_81 disabled (set {_ENV_ENABLE}=1 to enable TQ multi-query "
            "DIRECT decode routing — vllm#45144 blueprint)"
        )

    if _APPLIED:
        return "applied", "G4_81 already installed (idempotent)"

    try:
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionImpl,
        )
    except ImportError as e:
        return "skipped", (
            f"vllm.v1.attention.backends.turboquant_attn not importable: {e}"
        )

    original = TurboQuantAttentionImpl.forward
    if getattr(original, _WRAP_ATTR, False):
        _APPLIED = True
        return "applied", "G4_81 already wrapped (idempotent)"
    _ORIGINAL_FORWARD = original

    g4_67_present = getattr(original, "_genesis_g4_67_wrapped", False)

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
        **extra_kwargs,
    ):
        """G4_81 multi-query direct dispatch BEFORE original forward.

        Safety contract: ANY non-routable shape or routing failure
        falls through to the original forward unchanged.
        """
        try:
            routed = _try_route(
                self, layer, query, kv_cache, attn_metadata, output
            )
            if routed is not None:
                return routed
        except Exception as e:  # noqa: BLE001 — fall through, never break decode
            if not getattr(self, "_genesis_g4_81_error_logged", False):
                self._genesis_g4_81_error_logged = True
                log.warning(
                    "[G4_81] multi-query direct route failed (%r); falling "
                    "through to original forward (full trace once per impl)",
                    e,
                    exc_info=True,
                )
            else:
                log.debug("[G4_81] route failed again: %r", e)
        return original(
            self, layer, query, key, value, kv_cache, attn_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
            **extra_kwargs,
        )

    setattr(_wrapped_forward, _WRAP_ATTR, True)
    TurboQuantAttentionImpl.forward = _wrapped_forward  # type: ignore[method-assign]

    _APPLIED = True
    suffix = (
        " (NOTE: G4_67 wrapper detected underneath — G4_81 intercepts "
        "first; disable G4_67 to drop the redundant predicate)"
        if g4_67_present
        else ""
    )
    log.info(
        "[G4_81] TurboQuantAttentionImpl.forward wrapped — uniform K+1 "
        "spec-verify batches now route DIRECTLY through the TQ decode "
        "kernel (vllm#45144 blueprint).%s", suffix,
    )
    return "applied", (
        "G4_81 installed: MTP multi-query verify batches bypass the "
        "prefill-attention continuation path via direct decode-kernel "
        f"routing.{suffix}"
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
    "GENESIS_G4_81_MARKER",
    "apply",
    "is_applied",
    "revert",
]
