# SPDX-License-Identifier: Apache-2.0
"""PN77 — FP8 lm_head EmbeddingMethod subclass (Phase E.5 architectural redesign).

Replaces the broken Phase E.2-3 design (load_weights post-hook + raw
`nn.Parameter(...)` swap that orphans `weight_loader` callback) with the
canonical vllm extension point:

  1. Duck-typed protocol matching `UnquantizedEmbeddingMethod` interface →
     `Genesis_FP8_LMHead_EmbeddingMethod` (no `class X(UnquantizedEmbeddingMethod):`
     literal inheritance — see line 187+ explanation. Audit A-13 honesty fix
     2026-05-06: docstring corrected from "Subclass" since current vllm pin
     has zero `isinstance(quant_method, UnquantizedEmbeddingMethod)` checks
     in the active code path; protocol matching is sufficient and avoids
     parent's `__init__` semantics on running PROD.)
  2. Override `process_weights_after_loading(layer)` — vllm calls this hook
     AFTER all weight loading, after `tie_weights`, with `device_loading_context`
     already active. Use `replace_parameter()` (vllm's canonical primitive) to
     preserve `weight_loader` attribute through Parameter swap.
  3. Override `apply(layer, x, bias)` — hardware-tier dispatch:
       - Ampere (sm86):  weight-only FP8 via `apply_fp8_marlin_linear` (Marlin)
       - Ada/Hopper/Blackwell (sm89+): native FP8 GEMM via `torch._scaled_mm`
       - Fallback: cast-back to bf16, original GEMM (covers CPU/ROCm/old GPUs)

WHY THIS ARCHITECTURE
======================

Boot failure of Phase E.2-3 (env=1):
   `lm_head.weight = nn.Parameter(weight_fp8)` orphans the `weight_loader`
   callback that `set_weight_attrs(weight, {"weight_loader": ...})` registered
   in `create_weights`. On any post-hook re-touch of that Parameter (e.g.
   the SECOND iteration of `lm_head.weight` shard load through multimodal
   wrapper recursion `Qwen3_5MoeForConditionalGeneration → Qwen3_5ForCausalLMBase`),
   `default_weight_loader` is used instead → no TP-sharding → assertion fail
   `(248320, 5120) → (124160, 5120)`.

`vllm.model_executor.utils.replace_parameter` solves this by COPYING the
old Parameter's `weight_loader` onto the new one. Reference impl:
`Fp8LinearMethod.process_weights_after_loading` (`fp8.py:530`).

HARDWARE TIER DISPATCH
======================

Decision is made ONCE at `process_weights_after_loading` time, cached on
the layer as `_genesis_pn77_path = "marlin" | "scaled_mm" | "cast_back"`.
Per-forward dispatch is a single attribute read.

  Tier A (Ampere sm86 — A5000/3090): `apply_fp8_marlin_linear` requires a
    one-time `prepare_fp8_layer_for_marlin` repack at hook time. After
    repack, `layer.weight` is int32-packed (NOT FP8 dtype). Per-forward
    Marlin kernel dequant'ts back inside the GEMM.

  Tier B (Ada/Hopper sm89+): keep raw FP8 e4m3fn weight + per-tensor scale.
    Per-forward `torch._scaled_mm(x_fp8, weight_fp8, scale_a=1, scale_b=scalar)`
    → bf16/fp16 output. Native FP8 GEMM, ~1.3-2× FLOPs over BF16.

  Fallback: cast weight FP8→BF16 per call, original GEMM. ~3 ms per call on
    248K vocab × 5120 hidden. Acceptable for sampling step (one matmul/token).

DRIFT-RESISTANCE
=================

When upstream lands PR #41000 (config-driven `lm_head_quantized: true` in
`Fp8Config`), the wiring text-patch detects `lm_head_quantized` marker in
`fp8.py` source and self-retires. Genesis takes a back seat to upstream.

Author: Sandermage (Sander) Barzov Aleksandr — Ukraine, Odessa.
References:
  - vllm PR #35696 (lucaspirola, OPEN) — naive load_weights hook (mirrored in
    Phase E.2 design, broken due to weight_loader orphan)
  - vllm PR #41000 (webcodes-cz, OPEN) — config-driven Fp8Config dispatch
    (the architecture upstream is converging on)
  - `Fp8LinearMethod.process_weights_after_loading` (fp8.py:530) — REFERENCE
    implementation of the `replace_parameter` + Marlin tier dispatch pattern
"""
from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("genesis.kernels.lm_head_fp8_method")

# Marker attribute: set on layer once compression is done.
PN77_APPLIED_MARKER = "_already_called_process_weights_after_loading"
PN77_PATH_ATTR = "_genesis_pn77_path"  # "marlin" | "scaled_mm" | "cast_back"

ENV_FLAG = "GENESIS_ENABLE_PN77_FP8_LM_HEAD"


def _is_enabled() -> bool:
    """Read env once; opt-in default OFF."""
    return os.environ.get(ENV_FLAG, "").strip().lower() in (
        "1", "true", "yes", "y", "on",
    )


def _detect_hardware_tier() -> str:
    """Return tier identifier: 'marlin' | 'scaled_mm' | 'cast_back'.

    Decision matrix:
      sm86 (Ampere consumer/A5000) → marlin (weight-only FP8)
      sm80 (Ampere datacenter A100) → marlin (same — sm80+ supported)
      sm89/90 (Ada/Hopper) → scaled_mm (native FP8 GEMM via torch._scaled_mm)
      sm100+ (Blackwell) → scaled_mm (native; FP4-accumulator may come later)
      else → cast_back fallback
    """
    if not torch.cuda.is_available():
        return "cast_back"
    cap = torch.cuda.get_device_capability()
    if cap is None:
        return "cast_back"
    major, minor = cap[0], cap[1]
    sm = major * 10 + minor
    if sm >= 89:
        return "scaled_mm"  # Ada/Hopper/Blackwell — native FP8
    if sm >= 80:
        return "marlin"  # Ampere — weight-only FP8 via Marlin
    return "cast_back"


def _is_lm_head(layer) -> bool:
    """Detect ParallelLMHead vs (regular) VocabParallelEmbedding.

    ParallelLMHead inherits VocabParallelEmbedding; the ONLY runtime
    difference is the .apply() call site (LogitsProcessor) vs .embedding().
    Class-name match is the cheapest reliable signal.
    """
    cls_name = type(layer).__name__
    return cls_name == "ParallelLMHead" or cls_name.endswith("LMHead")


def maybe_swap_pn77_quant_method(layer, current_method):
    """Hook invoked from text-patched `process_weights_after_loading` walker.

    Swaps `layer.quant_method` to `Genesis_FP8_LMHead_EmbeddingMethod` if:
      - env GENESIS_ENABLE_PN77_FP8_LM_HEAD=1
      - `layer` is a ParallelLMHead (not regular embed_tokens)
      - hardware supports a useful FP8 path (not 'cast_back' fallback only)
      - current method is `UnquantizedEmbeddingMethod` (don't override
        if a real quant config already chose another method)

    Returns the method that should actually be used (swapped or original).
    NEVER raises — fallback to original on any failure.
    """
    try:
        if not _is_enabled():
            return current_method
        if not _is_lm_head(layer):
            return current_method
        # Only swap pristine UnquantizedEmbeddingMethod (don't override real quant)
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            UnquantizedEmbeddingMethod,
        )
        if not isinstance(current_method, UnquantizedEmbeddingMethod):
            return current_method
        # Idempotency
        if isinstance(current_method, Genesis_FP8_LMHead_EmbeddingMethod):
            return current_method
        # Skip if weight already FP8 (native checkpoint)
        weight = getattr(layer, "weight", None)
        if weight is not None and weight.dtype == torch.float8_e4m3fn:
            return current_method
        # Skip on cast_back-only hardware (no real win beyond fallback path)
        # Actually — cast-back still SAVES VRAM, so include it. Just slower per call.
        new_method = Genesis_FP8_LMHead_EmbeddingMethod()
        layer.quant_method = new_method
        log.info(
            "[PN77] swapped lm_head.quant_method UnquantizedEmbeddingMethod → "
            "Genesis_FP8_LMHead_EmbeddingMethod (tier=%s, weight=%s)",
            _detect_hardware_tier(),
            tuple(weight.shape) if weight is not None else "?",
        )
        return new_method
    except Exception as e:
        log.warning(
            "[PN77] swap helper failed (%s) — keeping original method",
            type(e).__name__,
        )
        return current_method


class Genesis_FP8_LMHead_EmbeddingMethod:
    """FP8 lm_head method — drop-in replacement for `UnquantizedEmbeddingMethod`.

    Pattern mirrors `Fp8LinearMethod.process_weights_after_loading` from
    `vllm/model_executor/layers/quantization/fp8.py:530`. The important
    invariant: use `replace_parameter()` (vllm's canonical helper) which
    PRESERVES the `weight_loader` attribute on the new Parameter, so
    subsequent re-loads (multimodal wrapper recursion, MTP head sync, etc.)
    continue to TP-shard correctly.

    INHERITANCE NOTE: we do NOT inherit from `UnquantizedEmbeddingMethod`
    directly to avoid CPU-path coupling in `process_weights_after_loading`.
    Instead we duplicate the small `create_weights` and `embedding` methods
    (these are the surface vllm reads). The quant_method protocol is duck-typed.
    """

    # ─── Init / shape setup ────────────────────────────────────────────

    def create_weights(
        self,
        layer,
        input_size_per_partition: int,
        output_partition_sizes: list,
        input_size: int,
        output_size: int,
        params_dtype,
        **extra_weight_attrs,
    ):
        """Identical to UnquantizedEmbeddingMethod.create_weights — load BF16
        weights first; we only convert to FP8 in process_weights_after_loading."""
        from vllm.model_executor.utils import set_weight_attrs

        weight = torch.nn.Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    # ─── Post-load mutation: BF16 → FP8 conversion ────────────────────

    def process_weights_after_loading(self, layer) -> None:
        """vllm canonical hook. Fires AFTER all weight load + tie_weights done.

        Uses `replace_parameter()` to swap weight Parameter while preserving
        the `weight_loader` callback (proven pattern from Fp8LinearMethod).

        Sets `output_size_per_partition`/`input_size_per_partition`/`orig_dtype`
        attrs that `prepare_fp8_layer_for_marlin` requires (these come from
        ColumnParallelLinear naturally but ParallelLMHead uses different names
        like `num_embeddings_per_partition` / `embedding_dim`).
        """
        if getattr(layer, PN77_APPLIED_MARKER, False):
            return  # Idempotent

        try:
            from sndr.engines.vllm.kernels_legacy.lm_head_fp8_compressor import compress
            from vllm.model_executor.utils import replace_parameter
        except Exception as e:
            log.warning(
                "[PN77] import failed (%s) — keeping BF16 lm_head", type(e).__name__,
            )
            return

        weight = layer.weight
        if weight.dtype == torch.float8_e4m3fn:
            log.info("[PN77] lm_head already FP8 — skipping compression")
            setattr(layer, PN77_APPLIED_MARKER, True)
            return

        # Save ORIGINAL dtype before compress — Marlin's prep needs `layer.orig_dtype`
        # to cast scales back. ParallelLMHead doesn't have it natively (set by us).
        orig_dtype = weight.dtype

        # Compress: BF16/FP16 → FP8 e4m3fn + per-channel scale
        try:
            weight_fp8, scale = compress(weight.data)
        except Exception as e:
            log.warning(
                "[PN77] compress() failed (%s) — keeping BF16 lm_head",
                type(e).__name__,
            )
            return

        # Tier dispatch — decide path ONCE, cache on layer
        tier = _detect_hardware_tier()
        setattr(layer, PN77_PATH_ATTR, tier)

        # Replace weight Parameter — preserves weight_loader via replace_parameter
        try:
            replace_parameter(layer, "weight", weight_fp8)
            # Also register scale as Parameter (for symmetric reload behavior).
            # Use replace_parameter even though `weight_scale` doesn't pre-exist —
            # it handles the missing-old-param case (just creates new).
            scale_param = torch.nn.Parameter(scale, requires_grad=False)
            layer.register_parameter("weight_scale", scale_param)
        except Exception as e:
            log.error(
                "[PN77] Parameter replacement failed (%s) — model state may be "
                "inconsistent; recommend container restart with env=0",
                type(e).__name__,
            )
            return

        # Set Marlin-required attrs (ParallelLMHead lacks ColumnParallelLinear's
        # naming convention). These attrs are what prepare_fp8_layer_for_marlin
        # reads:
        #   - output_size_per_partition: vocab_per_rank (N dim of GEMM)
        #   - input_size_per_partition: hidden_size (K dim)
        #   - orig_dtype: BF16/FP16 dtype to cast scales back into
        # ParallelLMHead.weight has shape (n, k) = (vocab_per_rank, hidden) —
        # this is `size_k_first=False` layout for Marlin prep.
        if not hasattr(layer, "output_size_per_partition"):
            layer.output_size_per_partition = getattr(
                layer, "num_embeddings_per_partition", weight_fp8.shape[0]
            )
        if not hasattr(layer, "input_size_per_partition"):
            layer.input_size_per_partition = getattr(
                layer, "embedding_dim", weight_fp8.shape[1]
            )
        if not hasattr(layer, "orig_dtype"):
            layer.orig_dtype = orig_dtype

        # Tier-specific post-processing
        if tier == "marlin":
            try:
                self._prepare_marlin(layer)
            except Exception as e:
                # Capture FULL exception detail for diagnosis (was just type name)
                import traceback
                log.warning(
                    "[PN77] Marlin prepare failed (%s: %s) — falling back to "
                    "cast_back tier. Full trace:\n%s",
                    type(e).__name__, str(e)[:200],
                    "".join(traceback.format_exception(type(e), e, e.__traceback__))[:1500],
                )
                setattr(layer, PN77_PATH_ATTR, "cast_back")

        setattr(layer, PN77_APPLIED_MARKER, True)
        log.info(
            "[PN77] lm_head compressed BF16→FP8: shape=%s, tier=%s, "
            "saved ~%.0f MiB/rank",
            tuple(weight_fp8.shape),
            getattr(layer, PN77_PATH_ATTR, "cast_back"),
            weight_fp8.numel() / (1024 * 1024),  # FP8=1byte
        )

        # VRAM-aware cleanup (2026-05-07): on 27B (large vocab×hidden) the
        # compress + Marlin repack flow creates large intermediate tensors
        # (BF16 → FP8 → packed int32 → marlin format) that the PyTorch caching
        # allocator may keep in non-split blocks until next empty_cache. vllm's
        # own empty_cache happens later in capture_model, but in our 27B+TQ
        # k8v4 measurement that wasn't enough — VRAM stayed +2 GB above
        # baseline. Explicit cleanup HERE forces the freed BF16/FP32/FP8/int32
        # intermediates back to OS BEFORE the rest of vllm load proceeds.
        # Cost: ~10-100 ms one-shot (per-layer hook call). Win: real VRAM save.
        del weight_fp8, scale, weight
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass  # cleanup is best-effort, never fail the patch

    def _prepare_marlin(self, layer):
        """One-time Marlin repack for Ampere weight-only FP8 path.

        Use `size_k_first=False` because ParallelLMHead.weight has the natural
        nn.Linear layout `(out, in) = (n, k) = (vocab_per_rank, hidden)`,
        opposite to FP8LinearMethod's intermediate which gets transposed first.
        """
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            prepare_fp8_layer_for_marlin,
        )
        prepare_fp8_layer_for_marlin(layer, size_k_first=False)

    # ─── Forward dispatch ──────────────────────────────────────────────

    def apply(self, layer, x, bias=None):
        """Forward pass — dispatch by tier flag set in process_weights_after_loading."""
        # If process_weights_after_loading hasn't run (e.g. env disabled mid-flight),
        # treat as plain UnquantizedEmbeddingMethod.
        if not getattr(layer, PN77_APPLIED_MARKER, False):
            return self._unquant_apply(layer, x, bias)

        tier = getattr(layer, PN77_PATH_ATTR, "cast_back")
        if tier == "marlin":
            return self._apply_marlin(layer, x, bias)
        if tier == "scaled_mm":
            return self._apply_scaled_mm(layer, x, bias)
        return self._apply_cast_back(layer, x, bias)

    # ─── Tier-specific apply implementations ──────────────────────────

    def _unquant_apply(self, layer, x, bias):
        """Bypass — used when marker not set (env disabled or pre-PN77 state)."""
        import vllm.envs as envs
        from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
        from vllm.platforms import current_platform

        if envs.VLLM_BATCH_INVARIANT and current_platform.is_cuda_alike():
            from vllm.model_executor.layers.batch_invariant import (
                linear_batch_invariant,
            )
            return linear_batch_invariant(x, layer.weight, bias)
        return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)

    def _apply_marlin(self, layer, x, bias):
        """Ampere weight-only FP8 via Marlin.

        Use persistent `output_size_per_partition`/`input_size_per_partition`
        attrs we set in `process_weights_after_loading` — `layer.weight.shape`
        is no longer valid after Marlin repacked it to int32-packed format.
        """
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            apply_fp8_marlin_linear,
        )
        return apply_fp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )

    def _apply_scaled_mm(self, layer, x, bias):
        """Ada/Hopper/Blackwell native FP8 GEMM."""
        # Per-tensor scale derived from per-channel max (lossy but acceptable
        # for sampling-step matmul; quality tested in unit + integration A/B).
        scale_per_tensor = layer.weight_scale.amax().to(torch.float32)
        # Cast x to FP8 with its own per-tensor scale
        x_amax = x.abs().amax().clamp(min=1e-12)
        x_scale = (x_amax / 448.0).to(torch.float32)
        x_fp8 = (x / x_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        # [Genesis PN77 0-D scale-rank fix, vendor of vllm#44912] Both scales
        # above are produced by an unkeyed reduction (`amax()` with no `dim`),
        # which collapses to a 0-D (scalar) tensor. torch._scaled_mm under
        # torch.compile / future Inductor lowering asserts
        # `len(scale_a.size()) == len(scale_b.size())` and rejects 0-D scales
        # outright (guaranteed InductorError on sm89+; latent on our Marlin-tier
        # Ampere today since scaled_mm tier never fires there, but a real
        # correctness/compile bug). Normalise both to 1-D before the call so the
        # ranks match and Inductor can lower the op. `.view(1)` is a zero-copy
        # reshape on a single-element tensor. Guarded by `dim() == 0` so this is
        # a no-op if a future path already feeds 1-D scales.
        if scale_per_tensor.dim() == 0:
            scale_per_tensor = scale_per_tensor.view(1)
        if x_scale.dim() == 0:
            x_scale = x_scale.view(1)
        # Native FP8 GEMM
        out = torch._scaled_mm(
            x_fp8,
            layer.weight.t(),
            scale_a=x_scale,
            scale_b=scale_per_tensor,
            bias=bias,
            out_dtype=x.dtype,
        )
        return out

    def _apply_cast_back(self, layer, x, bias):
        """Fallback — decompress to x.dtype, normal GEMM."""
        from sndr.engines.vllm.kernels_legacy.lm_head_fp8_compressor import decompress
        weight = decompress(layer.weight, layer.weight_scale, output_dtype=x.dtype)
        import vllm.envs as envs
        from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
        from vllm.platforms import current_platform

        if envs.VLLM_BATCH_INVARIANT and current_platform.is_cuda_alike():
            from vllm.model_executor.layers.batch_invariant import (
                linear_batch_invariant,
            )
            return linear_batch_invariant(x, weight, bias)
        # Build temporary "layer" facade with decompressed weight for dispatch
        # (dispatch_unquantized_gemm may read other layer attrs).
        return dispatch_unquantized_gemm()(layer, x, weight, bias)

    # ─── Embedding path (unchanged from Unquant) ──────────────────────

    def embedding(self, layer, input_):
        """Embedding lookup — for embed_tokens path only.

        ParallelLMHead generally doesn't go through this (embed_tokens does),
        but we provide it for protocol completeness. Decompress if FP8.
        """
        import torch.nn.functional as F
        if layer.weight.dtype == torch.float8_e4m3fn:
            from sndr.engines.vllm.kernels_legacy.lm_head_fp8_compressor import decompress
            weight = decompress(
                layer.weight, layer.weight_scale, output_dtype=torch.bfloat16,
            )
            return F.embedding(input_, weight)
        return F.embedding(input_, layer.weight)
