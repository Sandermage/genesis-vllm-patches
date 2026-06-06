# SPDX-License-Identifier: Apache-2.0
"""G4_19b — TurboQuant KV spec integration with vLLM v1 memory accounting.

================================================================
WHY THIS PATCH EXISTS
================================================================

G4_19 attaches G4TurboQuantConfig to vllm_config, but vLLM v1's
`_check_enough_kv_cache_memory` at engine boot reads the **native**
KV cache spec (fp16 size) — it doesn't know about our compression.
Result: 256K context fails the check because vLLM thinks it needs
~22 GB of KV cache while only ~10 GB is available.

This patch hooks the check to:

  1. Read `vllm_config._g4_19_turboquant_config` (set by G4_19).
  2. Multiply the available KV cache memory by the effective
     compression ratio (16/bits_global × overhead_factor).
  3. Let the check proceed with the corrected available memory.

It is a **temporary monkey-patch** until upstream vLLM ships per-cache
compression-aware memory accounting (planned in vllm#38171).

================================================================
WHAT IS HOOKED
================================================================

* ``vllm.v1.core.kv_cache_utils._check_enough_kv_cache_memory``
  — preflight check that raises if 1 max_seq_len request doesn't fit.

* ``vllm.v1.core.kv_cache_utils.get_kv_cache_configs``
  — computes block size and num_blocks; we adjust block budget.

This is **monkey-patching upstream vLLM internals** so it's brittle
to pin bumps. We pin to dev371+bf610c2f5 explicitly; if upstream
moves these functions, the patch becomes no-op (logs warning, falls
back to G4_19-only behavior).

================================================================
SAFETY MODEL
================================================================

* default_on: False (only fires when G4_19 is also on AND config attached)
* env_flag: GENESIS_ENABLE_G4_19B_GEMMA4_TQ_KV_SPEC
* applies_to: same as G4_19 (gemma4 arch)
* conflicts_with: none — composes with G4_19
* superseded_by: vllm#38171 when merged

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

from ...model_compat.gemma4._gemma4_detect import env_truthy

log = logging.getLogger("genesis.turboquant.g4_19b_tq_kv_spec")

GENESIS_G4_19B_MARKER = (
    "Genesis G4_19b gemma4 TurboQuant KV spec integration v1 "
    "(hooks vLLM v1 _check_enough_kv_cache_memory to account for TQ compression)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_19B_GEMMA4_TQ_KV_SPEC"

_APPLIED = False
_ORIGINAL_CHECK = None


def _env_enabled() -> bool:
    return env_truthy(_ENV_ENABLE)


def _compute_compression_factor(tq_config) -> float:
    """Compute effective compression ratio based on G4-TQ config.

    Returns the ratio fp16_bytes / tq_bytes for KV cache.

    For mixed-layer config (sliding=N bits, global=M bits), the per-token
    storage per head is N+M bytes_per_coord weighted by layer count.

    Args:
        tq_config: G4TurboQuantConfig instance.

    Returns:
        float — compression ratio (>1.0 means compressed)
    """
    # bytes per coord at each bit width: bits/8 (indices) + per-token-head
    # scale of 4 bytes amortized over head_dim coords
    # Effective bytes-per-coord:
    #   indices: bits/8 (but we store as uint8 = 1 byte until packing — TODO)
    #   scale_amortized: 4 / head_dim ≈ 0.016 byte/coord for head_dim=256

    head_dim = tq_config.head_dim
    bits_s = tq_config.bits_sliding
    bits_g = tq_config.bits_global

    # We currently store indices as uint8 (1 byte per coord) — this gives
    # the same compression as 4-bit packed; 3-bit packing would need more
    # work in the Triton kernels. So effective bits = max(bits, 8).
    # TODO: pack indices to actual bit-width for tighter compression.
    eff_bits_s = max(bits_s, 8)
    eff_bits_g = max(bits_g, 8)

    # Bytes per coord including scale overhead
    bytes_per_coord_s = eff_bits_s / 8 + 4 / head_dim
    bytes_per_coord_g = eff_bits_g / 8 + 4 / head_dim

    # If per-layer-types provided, weighted average; else use global-only
    layer_types = tq_config.per_layer_types
    if layer_types:
        n_sliding = sum(1 for t in layer_types if t == "sliding_attention")
        n_global = len(layer_types) - n_sliding
        n_total = n_sliding + n_global
        avg_bytes = (
            n_sliding * bytes_per_coord_s + n_global * bytes_per_coord_g
        ) / max(n_total, 1)
    else:
        avg_bytes = bytes_per_coord_g

    fp16_bytes_per_coord = 2.0
    return fp16_bytes_per_coord / avg_bytes


def _make_patched_check(original, get_compression_factor):
    """Wrap _check_enough_kv_cache_memory with compression awareness.

    Real vLLM v1 signature (dev371):
      _check_enough_kv_cache_memory(
          available_memory: int,
          get_needed_memory: Callable[[], int],
          max_model_len: int,
          estimate_max_model_len: Callable[[int], int],
      )

    G4_19b approach: multiply ``available_memory`` by the G4-TurboQuant
    compression factor before calling the original. vLLM then sees more
    KV cache memory than is physically available, but the actual cache
    is logically smaller post-compression so the math works out.

    Note: G4_19 attaches ``_g4_19_turboquant_config`` to vllm_config in
    parent process. In EngineCore subprocess, the attribute survives
    pickle/unpickle. We read from sys.modules to find any vllm_config
    instance — but simpler: only apply when G4_19 env is on (operator
    explicit opt-in).
    """
    import os

    # Cache decision once at wrap time — operator must explicitly enable
    g4_19_on = os.environ.get(
        "GENESIS_ENABLE_G4_19_GEMMA4_TURBOQUANT_KV", ""
    ).strip().lower() in ("1", "true", "yes")

    def _patched_check(
        available_memory,
        get_needed_memory,
        max_model_len,
        estimate_max_model_len,
    ):
        if not g4_19_on:
            return original(
                available_memory, get_needed_memory,
                max_model_len, estimate_max_model_len,
            )

        # Compute factor from operator env (no need for vllm_config attr —
        # bits come from env directly).
        bits_g = int(os.environ.get("GENESIS_G4_TQ_BITS_GLOBAL", "3"))
        bits_s = int(os.environ.get("GENESIS_G4_TQ_BITS_SLIDING", "4"))
        pack_mode = os.environ.get(
            "GENESIS_G4_TQ_PACK_MODE", "uint32"
        ).strip().lower()

        # Packing layout determines effective bytes/coord:
        #   uint32 (default): 8 coords per uint32 = 4 bytes per 8 coords
        #                     → 0.5 byte/coord regardless of 3 or 4 bits
        #                     → factor 16/8 / 0.5 = 4.0×
        #   tight (3-bit):    8 coords per 3 bytes = 0.375 byte/coord
        #                     → factor 16/8 / 0.375 = 5.33×
        #   uint8 (legacy):   1 byte per coord → factor 16/8 = 2.0×
        if pack_mode == "tight" and min(bits_g, bits_s) == 3:
            factor = 5.33  # tight 3-bit packing
        elif pack_mode == "uint32":
            factor = 4.0  # uint32 pack of either 3-bit or 4-bit
        else:
            factor = 2.0  # uint8 (legacy unpacked)

        effective_available = int(available_memory * factor)
        log.warning(
            "[G4_19b] KV cache memory check: available %.2f GB × compression "
            "%.2fx (pack=%s, bits sliding=%d/global=%d) = effective %.2f GB",
            available_memory / (1024 ** 3), factor, pack_mode, bits_s, bits_g,
            effective_available / (1024 ** 3),
        )
        return original(
            effective_available, get_needed_memory,
            max_model_len, estimate_max_model_len,
        )

    _patched_check._genesis_g4_19b_wrapped = True
    _patched_check.__wrapped__ = original
    return _patched_check


def apply() -> tuple[str, str]:
    """Install KV cache memory check wrapper for G4-TQ compression accounting."""
    global _APPLIED, _ORIGINAL_CHECK

    if not _env_enabled():
        return "skipped", (
            f"G4_19b disabled (set {_ENV_ENABLE}=1 to enable compression-aware "
            "KV cache memory check for G4_19 TurboQuant)"
        )

    if _APPLIED:
        return "applied", "G4_19b already installed (idempotent)"

    try:
        from vllm.v1.core import kv_cache_utils
    except ImportError as e:
        return "skipped", f"vllm.v1.core.kv_cache_utils not importable: {e}"

    original = getattr(kv_cache_utils, "_check_enough_kv_cache_memory", None)
    if original is None:
        return "skipped", (
            "vllm.v1.core.kv_cache_utils._check_enough_kv_cache_memory not "
            "found in this pin; G4_19b is no-op"
        )

    if getattr(original, "_genesis_g4_19b_wrapped", False):
        _APPLIED = True
        return "applied", "G4_19b already wrapped (idempotent)"

    _ORIGINAL_CHECK = original
    kv_cache_utils._check_enough_kv_cache_memory = _make_patched_check(
        original, _compute_compression_factor
    )
    _APPLIED = True
    log.info(
        "[G4_19b] installed: vLLM v1 KV cache memory check now multiplies "
        "available memory by G4-TQ compression factor when G4_19 active."
    )
    return "applied", (
        "G4_19b installed: vLLM v1 KV cache memory check is now "
        "compression-aware for G4_19 TurboQuant. Enables 256K context "
        "boot on Gemma 4 31B + 2× A5000."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED, _ORIGINAL_CHECK
    if not _APPLIED or _ORIGINAL_CHECK is None:
        return False
    try:
        from vllm.v1.core import kv_cache_utils
        kv_cache_utils._check_enough_kv_cache_memory = _ORIGINAL_CHECK
        _APPLIED = False
        return True
    except ImportError:
        return False


__all__ = ["GENESIS_G4_19B_MARKER", "apply", "is_applied", "revert"]
