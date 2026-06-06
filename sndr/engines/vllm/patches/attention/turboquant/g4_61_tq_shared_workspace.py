# SPDX-License-Identifier: Apache-2.0
"""G4_61 — share decode workspace across TQ layers (PR #40798 cherry-pick).

================================================================
PROBLEM
================================================================

In vllm pin ``0.20.2rc1.dev371+gbf610c2f5``, each TurboQuant attention
layer's ``Attention`` object holds ``_tq_mid_o_buf``, ``_tq_output_buf``,
and ``_tq_lse_buf`` as **per-layer registered buffers**. For a model
with N attention layers, that allocates N copies of the decode scratch
— but layers execute sequentially per request, so only ONE set is ever
in use at any moment.

Concrete impact (PR #40798 author's bench, Llama-3.1-70B TP=2):

| Version | Loading mem | Available KV | KV tokens   | Concurrency @ 64K |
|---------|-------------|--------------|-------------|-------------------|
| Before  | 105.23 GiB  | 14.61 GiB    | 400,128     | 6.11x             |
| After   | 65.74 GiB   | 53.97 GiB    | 1,478,384   | 22.56x            |

Memory loss for 70-layer model: **~40 GiB** wasted on per-layer scratch
that's never accessed concurrently.

Secondary problem: when ``capture_model`` locks the WorkspaceManager
**after** CUDA graph capture, the first decode that allocates a
workspace shape larger than profile-time crashes with::

    AssertionError: Workspace is locked but allocation from
    'turboquant_attn.py:879:_decode_attention' requires X.XX MB,
    current size is 0.00 MB. Workspace growth is not allowed after
    locking.

This fires because ``profile_run`` doesn't always land a decode-shape
through the TQ layers (especially in hybrid models with mixed
attention types — see jasonboukheir's analysis in issue #42544).

================================================================
FIX (PR #40798 cherry-pick)
================================================================

Two coordinated changes:

  1. ``triton_turboquant_decode_attention`` launcher gains a workspace-
     acquisition path: when caller passes ``mid_o_buf=None``, it asks
     ``current_workspace_manager().get_simultaneous(...)`` for the
     three buffer shapes. No more per-layer ``buf_holder`` attribute
     pollution. Layers share one workspace pool.

  2. ``GPUModelRunner.capture_model()`` calls a new
     ``_reserve_turboquant_decode_workspace()`` method **before** the
     early-return for ``CUDAGraphMode.NONE``. The method walks ALL
     ``attn_groups`` (not just ``[0]`` — gemini-code-assist review
     fix), and for each TQ group reserves the max-shape workspace plus
     the continuation prefill cache buffer (when chunked prefill
     enabled).

Net effect: workspace allocations happen **before** ``lock_workspace()``
fires, so the locked size already covers any future decode call. No
more "locked at 0.00 MB" assertions.

================================================================
DEPENDENCIES
================================================================

  * Compatible with **G4_60a..k** stack (orthogonal — different code paths).
  * Compatible with Genesis **PN118** (workspace graceful fallback —
    backport of vllm#42551). PN118 catches assertion at runtime; G4_61
    prevents it at boot. Belt-and-suspenders.

================================================================
SCOPE
================================================================

Active only when ``GENESIS_ENABLE_G4_61_TQ_SHARED_WORKSPACE=1``.
Touches 2 functions on 2 modules:

  * ``vllm.v1.attention.ops.triton_turboquant_decode.
    triton_turboquant_decode_attention`` — wraps launcher to acquire
    workspace.

  * ``vllm.v1.worker.gpu_model_runner.GPUModelRunner.capture_model`` —
    wraps to call ``_reserve_turboquant_decode_workspace`` before
    early return.

Also injects ``GPUModelRunner._reserve_turboquant_decode_workspace``
method (new).

================================================================
RISK
================================================================

  * **Workspace manager API drift**: assumes
    ``current_workspace_manager().get_simultaneous(*shapes_and_dtypes)``
    returns tuple of tensors. Stable since #40941 merge (2026-04-27).
    Verified on dev371 via ``grep -rn "def get_simultaneous"``.

  * **attn_groups schema**: PR #40798 v2 (after gemini review) iterates
    nested ``list[list[AttentionGroup]]``. Verify on your pin: the
    outer list is "kv_cache_group index", inner list is "subgroup
    within that kv_cache_group". This patch matches the v2 layout.

  * **kv_cache_group_id attribute**: not always present (some pre-PR
    layouts). G4_61 uses ``getattr(group, "kv_cache_group_id", 0)``
    with default fallback.

================================================================
REFERENCES
================================================================

  * Upstream PR: https://github.com/vllm-project/vllm/pull/40798
    Author: Bot1822 (Guipeng Zhang). OPEN, blocked on maintainer review.
  * Related issue: https://github.com/vllm-project/vllm/issues/42544
    jasonboukheir's crystal-clear root-cause analysis.
  * Related issue: https://github.com/vllm-project/vllm/issues/41565
    MidasMining's bisect identifying #40941 as regression source.
  * Companion patch: G4_62 (warmup) — complementary; both can co-exist.
  * Companion patch: PN118 (workspace graceful fallback) — runtime
    catch vs boot-time prevent.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.turboquant.g4_61_tq_shared_workspace")

GENESIS_G4_61_MARKER = (
    "Genesis G4_61 share TQ decode workspace across layers via "
    "WorkspaceManager (PR #40798 cherry-pick)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_61_TQ_SHARED_WORKSPACE"
_APPLIED = False
_ORIGINAL_DECODE_LAUNCHER = None
_ORIGINAL_CAPTURE_MODEL = None

# Match PR #40798's constant (gpu_model_runner.py).
_TURBOQUANT_CONTINUATION_DECODE_THRESHOLD = 128


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Patch decode launcher + capture_model for shared workspace."""
    global _APPLIED, _ORIGINAL_DECODE_LAUNCHER, _ORIGINAL_CAPTURE_MODEL

    if not _env_enabled():
        return "skipped", (
            f"G4_61 disabled (set {_ENV_ENABLE}=1 to share TQ decode "
            "workspace across layers — PR #40798 cherry-pick)"
        )

    if _APPLIED:
        return "applied", "G4_61 already installed (idempotent)"

    try:
        from vllm.v1.attention.ops import triton_turboquant_decode as _decode
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError as e:
        return "skipped", f"required modules not importable: {e}"

    try:
        from vllm.v1.worker.workspace import (  # noqa: F401
            current_workspace_manager,
            is_workspace_manager_initialized,
        )
    except ImportError as e:
        return "skipped", (
            f"vllm.v1.worker.workspace API missing: {e}; PR #40941 "
            "WorkspaceManager not yet integrated in this pin"
        )

    # === Patch 1: wrap triton_turboquant_decode_attention launcher ===
    original_launcher = _decode.triton_turboquant_decode_attention
    if getattr(original_launcher, "_genesis_g4_61_wrapped", False):
        _APPLIED = True
        return "applied", "G4_61 already wrapped (idempotent)"
    _ORIGINAL_DECODE_LAUNCHER = original_launcher

    def _wrapped_decode_launcher(
        query,
        kv_cache,
        block_table,
        seq_lens,
        Pi,
        centroids,
        scale,
        mse_bits,
        key_packed_size,
        value_quant_bits,
        PiT=None,
        norm_correction=False,
        kv_group_size=None,
        out=None,
        mid_o_buf=None,
        output_buf=None,
        lse_buf=None,
        max_num_kv_splits=32,
        **extra_kwargs,
    ):
        """Acquire workspace when buffers are None (PR #40798 lines 530-542)."""
        if mid_o_buf is None or output_buf is None or lse_buf is None:
            from vllm.v1.worker.workspace import (
                current_workspace_manager as _cwm,
                is_workspace_manager_initialized as _iwmi,
            )

            if _iwmi():
                import torch

                B = query.shape[0]
                Hq = query.shape[1] if query.ndim >= 2 else 1
                D = query.shape[-1]
                NUM_KV_SPLITS = max_num_kv_splits

                mid_o_buf, output_buf, lse_buf = _cwm().get_simultaneous(
                    ((B, Hq, NUM_KV_SPLITS, D + 1), torch.float32),
                    ((B, Hq, D), query.dtype),
                    ((B, Hq), torch.float32),
                )

        # Filter buf_holder out if caller passed it (PR removes the param)
        extra_kwargs.pop("buf_holder", None)

        # PR #42637 launcher signature: NO kv_group_size, NO out kwargs.
        # Computed internally as Hq // Hk. Filter these for forward-compat
        # with overlay path. Also pop any caller-supplied kv_group_size
        # from extra_kwargs.
        extra_kwargs.pop("kv_group_size", None)
        extra_kwargs.pop("out", None)
        return original_launcher(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            Pi=Pi,
            centroids=centroids,
            scale=scale,
            mse_bits=mse_bits,
            key_packed_size=key_packed_size,
            value_quant_bits=value_quant_bits,
            PiT=PiT,
            norm_correction=norm_correction,
            mid_o_buf=mid_o_buf,
            output_buf=output_buf,
            lse_buf=lse_buf,
            max_num_kv_splits=max_num_kv_splits,
            **extra_kwargs,
        )

    _wrapped_decode_launcher._genesis_g4_61_wrapped = True  # type: ignore[attr-defined]
    _decode.triton_turboquant_decode_attention = _wrapped_decode_launcher

    # === Patch 2: inject _reserve_turboquant_decode_workspace + wrap capture_model ===
    def _reserve_turboquant_decode_workspace(self) -> None:
        """Pre-allocate TQ decode workspace before lock_workspace fires.

        Verbatim port of PR #40798 gpu_model_runner.py:6198.
        """
        import torch

        from vllm.utils.math_utils import round_up
        from vllm.v1.worker.workspace import current_workspace_manager as _cwm

        if not self.cache_config.cache_dtype.startswith("turboquant_"):
            return
        attn_groups = getattr(self, "attn_groups", None)
        if not attn_groups:
            return

        max_num_reqs = self.scheduler_config.max_num_seqs
        max_num_tokens = self.scheduler_config.max_num_batched_tokens
        max_model_len = self.model_config.max_model_len
        num_heads = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        head_size = self.model_config.get_head_size()
        max_num_splits = (
            self.vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        )

        for groups in attn_groups:
            # Each `groups` is a list of AttentionGroup objects sharing a
            # KV cache group. PR #40798 v2 iterates all of them per outer
            # element, finding the first TQ group.
            for group in groups:
                if group.backend.get_name() != "TURBOQUANT":
                    continue

                _cwm().get_simultaneous(
                    (
                        (max_num_reqs, num_heads, max_num_splits, head_size + 1),
                        torch.float32,
                    ),
                    ((max_num_reqs, num_heads, head_size), self.dtype),
                    ((max_num_reqs, num_heads), torch.float32),
                )
                reserve_continuation_prefill = (
                    self.scheduler_config.enable_chunked_prefill
                    and max_num_tokens
                    > _TURBOQUANT_CONTINUATION_DECODE_THRESHOLD
                )
                if reserve_continuation_prefill:
                    kernel_block_sizes = getattr(
                        self, "_kernel_block_sizes", None
                    )
                    group_id = getattr(group, "kv_cache_group_id", 0)
                    if (
                        kernel_block_sizes is not None
                        and group_id < len(kernel_block_sizes)
                    ):
                        block_size = kernel_block_sizes[group_id]
                    else:
                        block_size = self.cache_config.block_size
                    if block_size is not None:
                        max_cached_len = max(0, max_model_len - 1)
                        alloc_len = round_up(max_cached_len, block_size)
                        cache_buf_shape = (1, num_kv_heads, alloc_len, head_size)
                        _cwm().get_simultaneous(
                            (cache_buf_shape, torch.float16),
                            (cache_buf_shape, torch.float16),
                        )
                return

    GPUModelRunner._reserve_turboquant_decode_workspace = (  # type: ignore[attr-defined]
        _reserve_turboquant_decode_workspace
    )

    original_capture_model = GPUModelRunner.capture_model
    _ORIGINAL_CAPTURE_MODEL = original_capture_model

    def _wrapped_capture_model(self, *args, **kwargs):
        """Reserve TQ workspace before capture_model's early return."""
        try:
            self._reserve_turboquant_decode_workspace()
        except Exception as e:  # noqa: BLE001
            log.warning(
                "[G4_61] _reserve_turboquant_decode_workspace failed: %r "
                "(continuing; PN118 fallback may catch runtime issues)",
                e,
            )
        return original_capture_model(self, *args, **kwargs)

    _wrapped_capture_model._genesis_g4_61_wrapped = True  # type: ignore[attr-defined]
    GPUModelRunner.capture_model = _wrapped_capture_model  # type: ignore[method-assign]

    _APPLIED = True
    log.info(
        "[G4_61] TQ decode workspace sharing installed: launcher acquires "
        "via WorkspaceManager; capture_model pre-reserves max-shape."
    )
    return "applied", (
        "G4_61 installed: TQ decode workspace shared across layers; "
        "max-shape pre-reserved before lock."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    """Best-effort revert (test isolation only)."""
    global _APPLIED, _ORIGINAL_DECODE_LAUNCHER, _ORIGINAL_CAPTURE_MODEL
    if not _APPLIED:
        return False
    try:
        from vllm.v1.attention.ops import triton_turboquant_decode as _decode
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        if _ORIGINAL_DECODE_LAUNCHER is not None:
            _decode.triton_turboquant_decode_attention = (
                _ORIGINAL_DECODE_LAUNCHER
            )
        if _ORIGINAL_CAPTURE_MODEL is not None:
            GPUModelRunner.capture_model = _ORIGINAL_CAPTURE_MODEL  # type: ignore[method-assign]
        if hasattr(GPUModelRunner, "_reserve_turboquant_decode_workspace"):
            try:
                delattr(
                    GPUModelRunner, "_reserve_turboquant_decode_workspace"
                )
            except AttributeError:
                pass
    except Exception:  # noqa: BLE001
        return False
    _APPLIED = False
    _ORIGINAL_DECODE_LAUNCHER = None
    _ORIGINAL_CAPTURE_MODEL = None
    return True


__all__ = [
    "GENESIS_G4_61_MARKER",
    "apply",
    "is_applied",
    "revert",
]
