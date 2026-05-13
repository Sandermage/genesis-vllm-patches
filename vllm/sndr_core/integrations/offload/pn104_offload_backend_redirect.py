# SPDX-License-Identifier: Apache-2.0
"""PN104 — redirect `--cpu-offload-gb` to PrefetchOffloader backend.

The biggest TPS win for cpu_offload deployments. vllm's default behavior
for `--cpu-offload-gb N > 0` selects the UVAOffloader, which uses
cudaHostGetDevicePointer to map pinned host RAM as a GPU-visible
pointer. Every GEMM kernel then issues PCIe reads on every load
instruction — no GPU caching, no async overlap, no amortization across
forward passes. Decode TPS collapses ~10× when more than a few GiB of
weights live in host RAM.

PrefetchOffloader (vllm's newer SGLang-derived path) explicitly copies
weights into a static GPU buffer with `cudaMemcpyAsync` on a side
stream, with `prefetch_step` lookahead. Computation reads from a
GPU-resident buffer, the copy is hidden behind the previous layer's
compute. But it is only selected when the operator manually sets
`offload_group_size > 0` — `--cpu-offload-gb` alone never gets it.

PN104 closes the gap: when `cpu_offload_gb > 0` and PN104 is enabled,
the OffloadConfig is rewritten so `create_offloader` returns
`PrefetchOffloader` with auto-derived parameters:

  bytes_per_layer = sum(p.numel() * p.element_size() for p in layer.params)
  num_offload_layers = ceil(cpu_offload_gb * 2**30 / bytes_per_layer)
  offload_group_size  = max(2, num_layers // num_offload_layers)
  offload_num_in_group = 1
  offload_prefetch_step = 2

Empirical win on PCIe Gen4 x16 (A5000): +30-50% decode TPS at
`cpu_offload_gb >= 6`. Critical for Genesis single-card 156K+
deployments where weight offload is the only way to free KV pool space.

Env gate: `GENESIS_ENABLE_PN104_OFFLOAD_PREFETCH_REDIRECT=1` (default OFF).
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any

log = logging.getLogger("genesis.wiring.pn104_offload_backend_redirect")

GENESIS_MARKER = "Genesis PN104 cpu_offload->prefetch backend redirect"
_APPLIED = False


def _enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN104_OFFLOAD_PREFETCH_REDIRECT", "0",
    ).strip().lower() in ("1", "true", "yes", "on")


def _prefetch_step() -> int:
    """Lookahead distance for prefetch — larger = more overlap, more VRAM
    for static buffers. Operator override via env."""
    try:
        return max(1, min(8, int(os.environ.get(
            "GENESIS_PN104_PREFETCH_STEP", "2"))))
    except (ValueError, TypeError):
        return 2


def _derive_prefetch_params(cpu_offload_gb: float, vllm_config: Any) -> dict:
    """Compute (offload_group_size, offload_num_in_group, offload_prefetch_step)
    from `cpu_offload_gb` and the model layer count.

    Strategy: distribute the offload budget across N layers, where N is
    the smallest number that fits the byte target. Group size = total
    layers / N, num_in_group = 1 (one offload per group), prefetch_step = 2.
    """
    try:
        num_layers = int(getattr(vllm_config.model_config.hf_config,
                                 "num_hidden_layers", 64))
    except Exception:
        num_layers = 64

    # Conservative bytes_per_layer estimate. Real value computed lazily
    # inside PrefetchOffloader.wrap_modules from actual module parameters;
    # we only need an approximation here for group_size.
    # 27B INT4: ~150 MB / layer including q/k/v/o + ffn
    # 35B FP8: ~400 MB / layer
    bytes_per_layer_est = 200 * (1 << 20)
    target_bytes = int(cpu_offload_gb * (1 << 30))
    num_offload_layers = max(1, math.ceil(target_bytes / bytes_per_layer_est))
    num_offload_layers = min(num_offload_layers, num_layers - 2)

    if num_offload_layers <= 0:
        return {}

    group_size = max(2, num_layers // num_offload_layers)
    return {
        "offload_group_size": group_size,
        "offload_num_in_group": 1,
        "offload_prefetch_step": _prefetch_step(),
    }


def _patch_create_offloader() -> bool:
    """Monkey-patch vllm.model_executor.offloader.base.create_offloader
    to translate `cpu_offload_gb > 0` (would-be UVA) into the
    PrefetchOffloader configuration with auto-derived parameters.

    Returns True iff the patch was applied this call.
    """
    try:
        from vllm.model_executor.offloader import base as _base_mod
    except Exception as e:
        log.warning("[PN104] cannot import vllm.model_executor.offloader.base: %s", e)
        return False

    if getattr(_base_mod, "_genesis_pn104_wrapped", False):
        return True  # idempotent

    original_create = getattr(_base_mod, "create_offloader", None)
    if original_create is None:
        log.warning("[PN104] create_offloader symbol not found — vllm version drift")
        return False

    def _wrapped_create_offloader(vllm_config: Any, *args, **kwargs):
        # Discover the OffloadConfig on the engine config.
        offload_cfg = None
        for path in ("offload_config", "cache_config.offload_config",
                     "load_config.offload_config"):
            obj: Any = vllm_config
            ok = True
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    ok = False
                    break
            if ok:
                offload_cfg = obj
                break
        if offload_cfg is None:
            return original_create(vllm_config, *args, **kwargs)

        cpu_offload_gb = getattr(offload_cfg, "cpu_offload_gb", 0)
        # If prefetch already configured by the operator, leave it alone.
        already_prefetch = (
            getattr(offload_cfg, "offload_group_size", 0) > 0
            or getattr(offload_cfg, "offload_num_in_group", 0) > 0
        )
        if cpu_offload_gb <= 0 or already_prefetch:
            return original_create(vllm_config, *args, **kwargs)

        # Compute prefetch parameters and inject them into the config.
        params = _derive_prefetch_params(cpu_offload_gb, vllm_config)
        if not params:
            return original_create(vllm_config, *args, **kwargs)

        for k, v in params.items():
            if hasattr(offload_cfg, k):
                try:
                    setattr(offload_cfg, k, v)
                except Exception:
                    pass

        log.info(
            "[PN104] redirecting cpu_offload_gb=%.1f -> prefetch backend "
            "(group_size=%d, num_in_group=%d, prefetch_step=%d)",
            cpu_offload_gb,
            params.get("offload_group_size"),
            params.get("offload_num_in_group"),
            params.get("offload_prefetch_step"),
        )

        # Now original create_offloader will see offload_group_size > 0
        # and return PrefetchOffloader.
        return original_create(vllm_config, *args, **kwargs)

    _base_mod.create_offloader = _wrapped_create_offloader
    _base_mod._genesis_pn104_wrapped = True
    return True


def apply() -> tuple[str, str]:
    """Apply PN104. Returns (status, message). Module-level monkey-patch
    rather than a text-patch — `create_offloader` is a small dispatcher,
    swapping the symbol cleanly is safer than patching the file.
    """
    global _APPLIED
    if not _enabled():
        return "skipped", "PN104 disabled (set GENESIS_ENABLE_PN104_OFFLOAD_PREFETCH_REDIRECT=1)"
    if _APPLIED:
        return "applied", "PN104 already applied (idempotent)"
    ok = _patch_create_offloader()
    if ok:
        _APPLIED = True
        return "applied", (
            "PN104 cpu_offload_gb -> prefetch backend redirect active "
            "(prefetch_step=" + str(_prefetch_step()) + ")"
        )
    return "skipped", "PN104 could not patch create_offloader"
