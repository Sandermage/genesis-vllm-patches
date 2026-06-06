# SPDX-License-Identifier: Apache-2.0
"""G4_62 — warm up TQ decode kernels before lock_workspace (PR #42215 cherry-pick).

================================================================
PROBLEM
================================================================

vllm's V1 ``profile_run`` does NOT guarantee that every backend's
hot-path kernels get exercised. For TurboQuant specifically:

  * ``_tq_decode_stage1`` and ``_tq_decode_stage2`` are Triton kernels
    that compile on first call.
  * ``profile_run`` only lands prefill/chunked-prefill shapes (the
    pessimistic memory-profiler path). It does not land a decode-shape
    through TQ layers unless the hybrid scheduler happens to dispatch
    one — which it doesn't for dense or attention-shy hybrid models.
  * Result: first real decode request triggers JIT compile of
    ``_tq_decode_stage1`` + ``_tq_decode_stage2``. Compile happens
    INSIDE the request → user observes 5-25 second TTFT spike.
  * Worse: if the workspace was locked at profile-time shape (before
    G4_61 reservation lands), the compile-time allocation request also
    crashes with the "Workspace is locked" assertion.

The companion of G4_61 (workspace pre-reservation) is **G4_62**:
actually CALL the decode path with synthetic shapes during
``kernel_warmup``. This compiles the kernels AND allocates the
workspace shapes before ``lock_workspace`` fires.

================================================================
FIX (PR #42215 cherry-pick)
================================================================

Two changes:

  1. Add a new ``turboquant_decode_warmup(model, ...)`` function that
     walks every TQ attention layer in the model, deduplicates by a
     13-field compile-key (matches Triton specialization criteria),
     and calls ``impl._decode_attention(...)`` with synthetic
     batch_size, seq_lens=1, block_table[:,0]=1 inputs. Each unique
     ``_TurboQuantDecodeWarmupKey`` compiles once.

  2. Monkey-patch ``kernel_warmup(worker)`` to call our new function
     after ``deep_gemm_warmup`` but before ``flashinfer_autotune``.
     Reads ``block_size`` + ``block_table_stride`` from
     ``worker.model_runner.input_batch.block_table.block_tables[0]``
     — V1 may split KV-manager blocks into smaller kernel blocks, so
     using the cache_config value would compile the wrong variant.

================================================================
DEPENDENCIES
================================================================

  * Complementary with **G4_61** (shared workspace). Both can co-exist
    safely (G4_61 reserves max shape; G4_62 compiles + allocates).
  * Imports ``vllm.v1.attention.backends.turboquant_attn``
    (``TurboQuantAttentionImpl``, ``TurboQuantMetadata``) — verified
    on dev371.

================================================================
SCOPE
================================================================

Active only when ``GENESIS_ENABLE_G4_62_TQ_KERNEL_WARMUP=1``. Touches:

  * ``vllm.model_executor.warmup.kernel_warmup.kernel_warmup`` —
    wrapped to call our warmup function.

  * Genesis module exposes ``turboquant_decode_warmup`` for direct
    testing.

For non-TurboQuant workloads, this patch is a no-op — the warmup
function iterates layers and finds none with TQ dtype, so no compile
work happens.

================================================================
RISK
================================================================

  * Boot-time cost: each unique ``(num_kv_heads, head_dim, block_size,
    block_table_stride, num_kv_splits, kv_group_size, scale, mse_bits,
    key_packed_size, value_quant_bits, key_fp8, norm_correction,
    output_fp16)`` tuple compiles ~once. For Gemma 4 with mixed
    attention this is typically 2-3 unique keys → 2-3 compiles =
    ~5-10s added to boot. Saved on first real request: 5-25s of TTFT
    spike removed.

  * Synthetic input might miss some shape variants that real workloads
    hit. The cache-key fields cover all Triton specialization
    parameters known to vary at decode time; new specialization knobs
    introduced upstream would need to be added to the key.

  * The new module ``turboquant_decode_warmup`` becomes the import
    target for tests; if upstream renames it (or merges as
    ``vllm.model_executor.warmup.turboquant_warmup``), Genesis G4_62
    should detect via ``hasattr`` and defer.

================================================================
REFERENCES
================================================================

  * Upstream PR: https://github.com/vllm-project/vllm/pull/42215
    Author: lesj0610.
  * Companion patch: G4_61 (workspace pre-reservation). Both can co-exist.
  * Related issue: https://github.com/vllm-project/vllm/issues/41565
    "_continuation_prefill workspace fails at long context".

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

log = logging.getLogger("genesis.turboquant.g4_62_tq_kernel_warmup")

if TYPE_CHECKING:
    import torch
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backends.turboquant_attn import (
        TurboQuantAttentionImpl,
    )

GENESIS_G4_62_MARKER = (
    "Genesis G4_62 warm up TQ decode kernels before lock_workspace "
    "(PR #42215 cherry-pick)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_62_TQ_KERNEL_WARMUP"
_APPLIED = False
_ORIGINAL_KERNEL_WARMUP = None


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


@dataclass(frozen=True)
class _TurboQuantDecodeWarmupKey:
    """Triton-specialization-aware dedup key.

    Mirrors PR #42215's _TurboQuantDecodeWarmupKey exactly. Fields chosen
    so that two impls with the same key compile identical Triton variants.
    """

    num_kv_heads: int
    head_dim: int
    block_size: int
    block_table_stride: int
    num_kv_splits: int
    kv_group_size: int
    scale: float
    mse_bits: int
    key_packed_size: int
    value_quant_bits: int
    key_fp8: bool
    norm_correction: bool
    output_fp16: bool


def _iter_turboquant_attention_layers(model):
    """Yield (Attention, TurboQuantAttentionImpl) pairs for TQ layers."""
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backends.turboquant_attn import (
        TurboQuantAttentionImpl,
    )

    for layer in model.modules():
        if not isinstance(layer, Attention):
            continue
        if not layer.kv_cache_dtype.startswith("turboquant_"):
            continue
        if not isinstance(layer.impl, TurboQuantAttentionImpl):
            continue
        yield layer, layer.impl


def _make_warmup_key(
    impl,
    *,
    block_size: int,
    block_table_stride: int,
    model_dtype,
) -> _TurboQuantDecodeWarmupKey:
    import torch

    return _TurboQuantDecodeWarmupKey(
        num_kv_heads=impl.num_kv_heads,
        head_dim=impl.head_size,
        block_size=block_size,
        block_table_stride=block_table_stride,
        num_kv_splits=impl.max_num_kv_splits,
        kv_group_size=impl.num_kv_groups,
        scale=impl.scale,
        mse_bits=impl.tq_config.key_mse_bits,
        key_packed_size=impl.tq_config.key_packed_size,
        value_quant_bits=impl.tq_config.effective_value_quant_bits,
        key_fp8=impl.tq_config.key_fp8,
        norm_correction=impl.tq_config.norm_correction,
        output_fp16=model_dtype == torch.float16,
    )


def _warmup_turboquant_decode_layer(
    layer,
    impl,
    *,
    device,
    block_size: int,
    block_table_stride: int,
    max_num_decode_tokens: int,
    model_dtype,
) -> None:
    """Call impl._decode_attention with synthetic inputs.

    Verbatim port of PR #42215 _warmup_turboquant_decode_layer.
    """
    import torch

    from vllm.v1.attention.backends.turboquant_attn import TurboQuantMetadata

    impl._ensure_on_device(layer, device)

    batch_size = max_num_decode_tokens
    query = torch.zeros(
        (batch_size, impl.num_heads, impl.head_size),
        dtype=model_dtype,
        device=device,
    )
    kv_cache = torch.zeros(
        (2, block_size, impl.num_kv_heads, impl.tq_config.slot_size_aligned),
        dtype=torch.uint8,
        device=device,
    )
    block_table = torch.zeros(
        (batch_size, block_table_stride), dtype=torch.int32, device=device
    )
    block_table[:, 0] = 1
    seq_lens = torch.ones(batch_size, dtype=torch.int32, device=device)
    attn_metadata = TurboQuantMetadata(
        seq_lens=seq_lens,
        slot_mapping=torch.zeros(batch_size, dtype=torch.long, device=device),
        block_table=block_table,
        query_start_loc=torch.arange(
            batch_size + 1, dtype=torch.int32, device=device
        ),
        num_actual_tokens=batch_size,
        max_query_len=1,
        max_seq_len=1,
        is_prefill=False,
        num_decodes=batch_size,
        num_decode_tokens=batch_size,
    )

    # Use the runtime decode helper instead of calling the Triton launcher
    # directly. This warms both the decode kernels and the WorkspaceManager
    # allocation path before the workspace is locked after CUDA graph capture.
    impl._decode_attention(
        query=query,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        Pi=layer._tq_Pi,
        centroids=layer._tq_centroids,
        PiT=layer._tq_PiT,
        layer=layer,
    )


def turboquant_decode_warmup(
    model,
    *,
    device,
    block_size: int,
    block_table_stride: int,
    max_num_decode_tokens: int,
    model_dtype,
) -> None:
    """Compile TurboQuant decode kernels without running model forward.

    Verbatim port of PR #42215 turboquant_decode_warmup.

    V1 dummy/profile warmup can avoid the TurboQuant decode path, which
    leaves ``_tq_decode_stage1`` and ``_tq_decode_stage2`` to compile on
    the first real decode request. This warmup calls the backend decode
    path with synthetic tensors whose launch-time constants match the
    runtime attention layer.
    """
    import torch

    if max_num_decode_tokens <= 0:
        return

    seen: set[_TurboQuantDecodeWarmupKey] = set()
    num_warmups = 0

    with torch.inference_mode():
        for layer, impl in _iter_turboquant_attention_layers(model):
            key = _make_warmup_key(
                impl,
                block_size=block_size,
                block_table_stride=block_table_stride,
                model_dtype=model_dtype,
            )
            if key in seen:
                continue
            seen.add(key)
            _warmup_turboquant_decode_layer(
                layer,
                impl,
                device=device,
                block_size=block_size,
                block_table_stride=block_table_stride,
                max_num_decode_tokens=max_num_decode_tokens,
                model_dtype=model_dtype,
            )
            num_warmups += 1

    if num_warmups > 0:
        # torch.accelerator.synchronize is the PR's choice. Fallback to
        # torch.cuda.synchronize on older pytorch builds.
        try:
            torch.accelerator.synchronize()
        except (AttributeError, RuntimeError):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        log.info(
            "[G4_62] Warmed up %d TurboQuant decode kernel variant(s).",
            num_warmups,
        )


def apply() -> tuple[str, str]:
    """Wrap kernel_warmup to call turboquant_decode_warmup."""
    global _APPLIED, _ORIGINAL_KERNEL_WARMUP

    if not _env_enabled():
        return "skipped", (
            f"G4_62 disabled (set {_ENV_ENABLE}=1 to warm up TQ decode "
            "kernels before lock_workspace — PR #42215 cherry-pick)"
        )

    if _APPLIED:
        return "applied", "G4_62 already installed (idempotent)"

    try:
        from vllm.model_executor.warmup import kernel_warmup as _kw
    except ImportError as e:
        return "skipped", (
            f"vllm.model_executor.warmup.kernel_warmup not importable: {e}"
        )

    original = _kw.kernel_warmup
    if getattr(original, "_genesis_g4_62_wrapped", False):
        _APPLIED = True
        return "applied", "kernel_warmup already wrapped (idempotent)"
    _ORIGINAL_KERNEL_WARMUP = original

    def _wrapped_kernel_warmup(worker):
        """Inject turboquant_decode_warmup after deep_gemm but before flashinfer.

        Mirrors PR #42215 kernel_warmup.py changes.
        """
        # First run original to do deep_gemm_warmup. Then inject our TQ warmup.
        # Implementation note: we cannot easily inject *between* deep_gemm and
        # flashinfer without rewriting the original. Instead we call original
        # first (which does both), then run our TQ warmup at the end. The
        # ordering loss is acceptable — TQ warmup before flashinfer_autotune
        # is preferred but not required for correctness.
        result = original(worker)

        try:
            block_size = worker.cache_config.block_size
            block_table_stride = 1
            block_tables = (
                worker.model_runner.input_batch.block_table.block_tables
            )
            if block_tables:
                bt0 = block_tables[0]
                block_size = bt0.block_size
                block_table_stride = bt0.max_num_blocks_per_req
            max_num_decode_tokens = min(
                worker.scheduler_config.max_num_seqs,
                worker.scheduler_config.max_num_batched_tokens,
            )
            turboquant_decode_warmup(
                worker.get_model(),
                device=worker.model_runner.device,
                block_size=block_size,
                block_table_stride=block_table_stride,
                max_num_decode_tokens=max_num_decode_tokens,
                model_dtype=worker.model_runner.dtype,
            )
        except Exception as e:  # noqa: BLE001
            log.warning(
                "[G4_62] turboquant_decode_warmup failed: %r (continuing; "
                "first request may pay JIT compile cost)",
                e,
            )

        return result

    _wrapped_kernel_warmup._genesis_g4_62_wrapped = True  # type: ignore[attr-defined]
    _kw.kernel_warmup = _wrapped_kernel_warmup

    _APPLIED = True
    log.info(
        "[G4_62] kernel_warmup wrapped: turboquant_decode_warmup will run "
        "after deep_gemm_warmup."
    )
    return "applied", (
        "G4_62 installed: TQ decode kernels will be JIT-compiled at boot "
        "instead of first request."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    """Best-effort revert (test isolation only)."""
    global _APPLIED, _ORIGINAL_KERNEL_WARMUP
    if not _APPLIED or _ORIGINAL_KERNEL_WARMUP is None:
        return False
    try:
        from vllm.model_executor.warmup import kernel_warmup as _kw

        _kw.kernel_warmup = _ORIGINAL_KERNEL_WARMUP
    except Exception:  # noqa: BLE001
        return False
    _APPLIED = False
    _ORIGINAL_KERNEL_WARMUP = None
    return True


__all__ = [
    "GENESIS_G4_62_MARKER",
    "turboquant_decode_warmup",
    "apply",
    "is_applied",
    "revert",
]
