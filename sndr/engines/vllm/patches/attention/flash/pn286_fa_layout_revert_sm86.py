# SPDX-License-Identifier: Apache-2.0
"""PN286 — FlashAttention KV cache layout revert for Ampere SM 8.6 (Genesis-original).

================================================================
PROBLEM
================================================================

Upstream PR #42095 (merged 2026-05-27, commit 7e33081ce) flipped the
FlashAttention KV cache layout from

    (2, num_blocks, block_size, num_kv_heads, head_size)        # pre-#42095

to

    (num_blocks, 2, block_size, num_kv_heads, head_size)        # post-#42095

and changed the K/V split via ``kv_cache.unbind(0)`` → ``unbind(1)``.

On Hopper (SM 9.0+) and Blackwell with TMA, the new layout is
performance-neutral or slightly faster — K and V of the same block
sit adjacent in memory, fitting one prefetch window.

On Ampere SM 8.6 (NVIDIA RTX A5000 / A6000), the L2 is only 6 MB, no
TMA, and the prefetcher streams sequential blocks during paged
decode. The new ``unbind(1)`` produces a strided view whose outer
stride is ``2 * block_size * num_kv_heads * head_size`` — DOUBLE the
pre-#42095 stride — so each block index lookup jumps over the V data
of the previous block to reach the next K data.

Under MTP K=3 speculative decode each accepted token requires ~4 KV
cache passes (1 target verify + 3 draft steps). Empirically on
Qwen3.6-27B INT4 + TQ k8v4 + MTP K=3 on 2× A5000 this manifested as
a **−9.27% wall TPS regression** between vllm pins ``bf610c2f5``
(dev371, May 14) and ``626fa9bb`` (May 28).

Math derivation (per K.1.R.R.5 diagnostic):

  Base TPS (no MTP) on 626fa9bb: 57.75
  MTP K=1 multiplier: 1.45×  →  per-K cost ratio T_K ≈ 0.24
  MTP K=3 multiplier: 2.07×  →  T_K ≈ 0.17
  dev371 K=3 implied: 2.28×  →  T_K ≈ 0.155

  Draft step is ~31% slower on 626fa9bb. K=3 amplifies into 9% wall
  TPS regression. Matches empirical exactly.

35B FP8 dense MoE on the same pin is **+3.67% TPS** because it has
no MTP (smaller per-token cost amplification) and FP8 cache fits L2
regardless of layout.

================================================================
FIX
================================================================

Restore pre-#42095 KV cache layout for FlashAttention WHEN AND ONLY
WHEN the current device capability is exactly SM 8.6 (Ampere consumer
+ workstation: A5000, A6000, A40, RTX 3090, RTX 4000 Ada, etc.).

Components:

  1. **Backend shape override** — monkey-patch
     ``FlashAttentionBackend.get_kv_cache_shape`` to return
     ``(2, num_blocks, ...)`` on SM 8.6.

  2. **Backend stride order override** — monkey-patch
     ``FlashAttentionBackend.get_kv_cache_stride_order`` to return
     pre-#42095 stride tuples.

  3. **Hybrid layout twist neutralization** — monkey-patch
     ``GPUModelRunner._update_hybrid_attention_mamba_layout`` to skip
     the FA stride twist for groups whose backend is our patched
     FlashAttention (which now returns block_dim=1, so upstream would
     try to apply ``as_strided_`` that creates the new interleaved
     pattern we are trying to avoid).

  4. **Forward unbind axis revert** — TextPatch
     ``FlashAttentionImpl.forward`` and
     ``FlashAttentionImpl.do_kv_cache_update`` to use ``unbind(0)``
     instead of ``unbind(1)`` on SM 8.6.

The combination produces:

  * Physical KV memory layout: ``(2, N, B, H, D)`` — K bank then V bank
  * Logical tensor shape: same
  * Forward path: ``unbind(0)`` → 2 contiguous (N, B, H, D) views
  * No stride twist, no L2 prefetch miss amplification

================================================================
WHY NOT JUST `.contiguous()`?
================================================================

Calling ``.contiguous()`` on each forward step would double the
peak KV memory transiently and defeat the purpose. The fix has to
be at the allocator + dispatcher boundary, not in the hot path.

================================================================
SM GATING RATIONALE
================================================================

Strict gate ``current_platform.is_device_capability(86)``. NOT
``is_device_capability_family(80)`` because:
- Ampere data-center (A100, SM 8.0) has 40 MB L2 → new layout neutral
- SM 8.6 is the exact consumer/workstation Ampere variant we target
- SM 8.9 (Ada) has different L2 patterns; not validated here
- SM 9.0+ (Hopper, Blackwell) genuinely benefit from the new layout

If you have an A100 (SM 8.0) or H100 (SM 9.0), this patch is a no-op.
Only A5000/A6000/RTX 30/40-series-workstation users see the revert.

================================================================
ENV FLAG
================================================================

  GENESIS_ENABLE_PN286_FA_LAYOUT_REVERT_SM86=1   (opt-in)
  GENESIS_PN286_FORCE                            (force-apply on any SM)

Default OFF until pin bump is operator-driven; on rig validation
confirmed +TPS recovery on 27B PROD, will flip to default_on=True.

================================================================
COMPOSITION
================================================================

  * Composes cleanly with K.1.R.R.4 fallback resolvers (different file).
  * Composes with LayoutIntrospect (K.1.R.R.3) — PN286 ensures our
    fast-path uses the layout the rest of the stack expects on SM 8.6.
  * NOT for SM 9.0+ (Hopper). Self-skips automatically.
  * Auto-retires on a future upstream PR that adds per-SM layout
    selection natively (watch upstream RFC #42082).

================================================================
ACCEPTANCE GATE
================================================================

  1. Boot smoke — 27B + MTP K=3 + TQ k8v4 → coherent first response
  2. Bench — 27B wall_TPS ≥ 130 (within 2% of dev371 baseline 131.84)
  3. 35B sanity — 35B FP8 wall_TPS unchanged or improved (not regressed)
  4. Apply matrix — no other Genesis patches broken
  5. Memory probe — no peak VRAM regression (layout is same shape)

Author: Sander, Odessa. K.1.R.R.5 (2026-05-29).
"""
from __future__ import annotations

import logging
import os

from sndr.kernel import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)
from sndr.engines.vllm.detection.guards import resolve_vllm_file

log = logging.getLogger("genesis.attention.flash.pn286_fa_layout_revert_sm86")

GENESIS_PN286_MARKER = (
    "Genesis PN286 FA KV cache layout revert for Ampere SM 8.6 "
    "(closes 9% MTP K=3 regression from upstream #42095)"
)

_ENV_ENABLE = "GENESIS_ENABLE_PN286_FA_LAYOUT_REVERT_SM86"
_ENV_FORCE = "GENESIS_PN286_FORCE"
_APPLIED = False
_ORIGINAL_SHAPE = None
_ORIGINAL_STRIDE = None
_ORIGINAL_HYBRID_LAYOUT = None


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _env_forced() -> bool:
    return os.environ.get(_ENV_FORCE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _is_sm86() -> tuple[bool, str]:
    """Detect Ampere SM 8.6 (A5000/A6000/A40/RTX 3090 etc).

    Returns ``(is_sm86, description)`` for logging.
    """
    try:
        from vllm.platforms import current_platform

        if hasattr(current_platform, "is_device_capability"):
            is_86 = bool(current_platform.is_device_capability(86))
            if is_86:
                return True, "SM 8.6 detected (Ampere consumer/workstation)"
            return False, f"capability != 8.6 (current platform={type(current_platform).__name__})"
        return False, "no is_device_capability method"
    except Exception as e:
        log.warning("[PN286] platform detection failed: %s", e)
        return False, f"detection error: {e!r}"


def _make_unbind_revert_patcher() -> TextPatcher | None:
    """Two TextPatches for FA forward / do_kv_cache_update unbind axis."""
    target = resolve_vllm_file("v1/attention/backends/flash_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN286 FA unbind axis revert (#42095 backout for SM 8.6)"
        ),
        target_file=str(target),
        marker=GENESIS_PN286_MARKER,
        sub_patches=[
            # Forward decode site
            TextPatch(
                name="PN286.fwd_unbind",
                anchor=(
                    "        # For decoder and cross-attention, use KV cache as before\n"
                    "        key_cache, value_cache = kv_cache.unbind(1)"
                ),
                replacement=(
                    "        # For decoder and cross-attention, use KV cache as before\n"
                    "        # [Genesis PN286] SM 8.6 layout revert for #42095 perf regression\n"
                    "        from vllm.platforms import current_platform as _g_pn286_platform\n"
                    "        _g_pn286_unbind = 0 if _g_pn286_platform.is_device_capability(86) else 1\n"
                    "        key_cache, value_cache = kv_cache.unbind(_g_pn286_unbind)"
                ),
            ),
            # do_kv_cache_update site
            # NOTE: upstream simplified the else-block in dev354+ (vllm@626fa9bba),
            # removing the `else:` precursor. Anchor now matches the new flat form;
            # required=True ensures Genesis surfaces failure rather than silent half-apply.
            TextPatch(
                name="PN286.update_unbind",
                anchor=(
                    "        # Scatter write into the KV cache using slot_mapping indices.\n"
                    "        # No TMA kernel is invoked here, so stride canonicalization is not needed.\n"
                    "        key_cache, value_cache = kv_cache.unbind(1)"
                ),
                replacement=(
                    "        # Scatter write into the KV cache using slot_mapping indices.\n"
                    "        # No TMA kernel is invoked here, so stride canonicalization is not needed.\n"
                    "        # [Genesis PN286] SM 8.6 layout revert paired with shape patch\n"
                    "        from vllm.platforms import current_platform as _g_pn286_platform\n"
                    "        _g_pn286_unbind = 0 if _g_pn286_platform.is_device_capability(86) else 1\n"
                    "        key_cache, value_cache = kv_cache.unbind(_g_pn286_unbind)"
                ),
                required=True,
            ),
        ],
    )


def _install_shape_override() -> bool:
    """Monkey-patch FlashAttentionBackend.get_kv_cache_shape on SM 8.6."""
    global _ORIGINAL_SHAPE
    try:
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionBackend,
        )
    except ImportError as e:
        log.warning("[PN286] FlashAttentionBackend not importable: %s", e)
        return False

    if getattr(FlashAttentionBackend.get_kv_cache_shape, "_pn286_wrapped", False):
        return True

    _ORIGINAL_SHAPE = FlashAttentionBackend.get_kv_cache_shape

    @staticmethod
    def _pn286_get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        # Revert pre-#42095 shape: K/V split at axis 0, num_blocks at axis 1.
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    _pn286_get_kv_cache_shape._pn286_wrapped = True
    FlashAttentionBackend.get_kv_cache_shape = _pn286_get_kv_cache_shape
    log.warning(
        "[PN286] FlashAttentionBackend.get_kv_cache_shape replaced "
        "with pre-#42095 (2, N, B, H, D) form on SM 8.6"
    )
    return True


def _install_stride_override() -> bool:
    """Monkey-patch FlashAttentionBackend.get_kv_cache_stride_order on SM 8.6."""
    global _ORIGINAL_STRIDE
    try:
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionBackend,
        )
        from vllm.v1.attention.backends.utils import get_kv_cache_layout
    except ImportError as e:
        log.warning("[PN286] stride override import failed: %s", e)
        return False

    if getattr(
        FlashAttentionBackend.get_kv_cache_stride_order, "_pn286_wrapped", False
    ):
        return True

    _ORIGINAL_STRIDE = FlashAttentionBackend.get_kv_cache_stride_order

    @staticmethod
    def _pn286_get_kv_cache_stride_order(include_num_layers_dimension=False):
        # Pre-#42095 stride orders for SM 8.6 (the layout was 2-first).
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_layers, 2, num_blocks, block_size, num_kv_heads, head_size)
            return (2, 0, 1, 3, 4, 5)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            return (2, 4, 0, 1, 3, 5)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    _pn286_get_kv_cache_stride_order._pn286_wrapped = True
    FlashAttentionBackend.get_kv_cache_stride_order = (
        _pn286_get_kv_cache_stride_order
    )
    log.warning(
        "[PN286] FlashAttentionBackend.get_kv_cache_stride_order replaced "
        "with pre-#42095 stride tuples on SM 8.6"
    )
    return True


def _install_hybrid_layout_skip() -> bool:
    """Monkey-patch GPUModelRunner._update_hybrid_attention_mamba_layout.

    With shape reverted to (2, N, ...), the FA backend's default
    ``get_kv_cache_block_dim`` (via base-class sentinel trick) will
    return 1 — telling upstream the cache is K/V-first. Upstream would
    then apply a stride twist meant to convert legacy backends into
    blocks-first physical layout. PN286 prevents this twist for FA
    backends so the cache stays in our reverted layout.

    We do this by replacing the method with one that skips when the
    backend's full class name is FlashAttention.
    """
    global _ORIGINAL_HYBRID_LAYOUT
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError as e:
        log.warning("[PN286] GPUModelRunner not importable: %s", e)
        return False

    if not hasattr(GPUModelRunner, "_update_hybrid_attention_mamba_layout"):
        log.warning(
            "[PN286] _update_hybrid_attention_mamba_layout absent — "
            "likely pre-#42095 pin (no-op)"
        )
        return True

    original = GPUModelRunner._update_hybrid_attention_mamba_layout
    if getattr(original, "_pn286_wrapped", False):
        return True

    _ORIGINAL_HYBRID_LAYOUT = original

    def _pn286_update_hybrid_attention_mamba_layout(
        self, kv_caches, kernel_block_sizes
    ):
        """Skip FA backends when on SM 8.6 with PN286 active."""
        try:
            from vllm.v1.kv_cache_interface import AttentionSpec
        except ImportError:
            return original(self, kv_caches, kernel_block_sizes)

        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            if not isinstance(kv_cache_spec, AttentionSpec):
                continue
            backend_name = getattr(group.backend, "__name__", "").lower()
            # Skip FA backends — PN286 keeps them in pre-#42095 layout
            if "flashattention" in backend_name or "flash_attn" in backend_name:
                continue
            # For non-FA backends fall through to original behavior
            try:
                block_dim = group.backend.get_kv_cache_block_dim(
                    kernel_block_sizes[group.kv_cache_group_id],
                    kv_cache_spec.num_kv_heads,
                    kv_cache_spec.head_size,
                    cache_dtype_str=self.cache_config.cache_dtype,
                )
            except Exception:
                continue
            if block_dim == 0:
                continue
            if block_dim == 1:
                for layer_name in group.layer_names:
                    kv_cache = kv_caches[layer_name]
                    hidden_size = kv_cache.shape[2:].numel()
                    kv_cache.as_strided_(
                        size=kv_cache.shape,
                        stride=(
                            hidden_size,
                            2 * hidden_size,
                            *kv_cache.stride()[2:],
                        ),
                    )

    _pn286_update_hybrid_attention_mamba_layout._pn286_wrapped = True
    GPUModelRunner._update_hybrid_attention_mamba_layout = (
        _pn286_update_hybrid_attention_mamba_layout
    )
    log.warning(
        "[PN286] GPUModelRunner._update_hybrid_attention_mamba_layout "
        "wrapped to skip FA backends (preserves pre-#42095 layout)"
    )
    return True


def apply() -> tuple[str, str]:
    """Install PN286 if SM 8.6 detected and env enabled."""
    global _APPLIED

    if not _env_enabled():
        return "skipped", (
            f"PN286 disabled (set {_ENV_ENABLE}=1 to revert FA KV layout "
            "to pre-#42095 on Ampere SM 8.6 for MTP K=3 perf recovery)"
        )

    if _APPLIED:
        return "applied", "PN286 already installed (idempotent)"

    is_86, desc = _is_sm86()
    if not is_86 and not _env_forced():
        return "skipped", (
            f"PN286 self-skip: not SM 8.6 ({desc}). The patch only helps "
            "Ampere consumer/workstation. Set GENESIS_PN286_FORCE=1 to "
            "override (not recommended)."
        )

    log.warning("[PN286] apply() entered — %s", desc)

    # Step 1: shape override
    if not _install_shape_override():
        return "skipped", "PN286 shape override failed"

    # Step 2: stride order override
    if not _install_stride_override():
        return "skipped", "PN286 stride override failed"

    # Step 3: hybrid layout twist neutralization
    if not _install_hybrid_layout_skip():
        return "skipped", "PN286 hybrid layout skip failed"

    # Step 4: unbind axis TextPatch
    patcher = _make_unbind_revert_patcher()
    if patcher is None:
        return "skipped", "PN286 unbind TextPatcher build failed"

    result, failure = patcher.apply()
    if result == TextPatchResult.FAILED:
        return "failed", failure.reason if failure else "unknown TextPatch failure"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown TextPatch skip"
    # APPLIED or IDEMPOTENT — proceed
    applied = patcher.applied_sub_patches or [sp.name for sp in patcher.sub_patches]
    detail = f"TextPatches applied: {', '.join(applied)}"
    _APPLIED = True
    log.warning(
        "[PN286] INSTALLED: FA KV cache layout reverted to pre-#42095 "
        "for SM 8.6 (4 components: shape, stride, hybrid-skip, unbind-axis)"
    )
    return "applied", (
        f"PN286 installed: FA layout revert for SM 8.6 (Ampere). "
        f"Expected: +9% TPS recovery on MTP K=3 hybrid models. "
        f"Detail: {detail}"
    )


def is_applied() -> bool:
    return _APPLIED


__all__ = [
    "GENESIS_PN286_MARKER",
    "apply",
    "is_applied",
]
