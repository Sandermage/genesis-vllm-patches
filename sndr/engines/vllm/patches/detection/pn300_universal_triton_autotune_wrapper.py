# SPDX-License-Identifier: Apache-2.0
"""Patch PN300 — Universal Triton Autotune Arch-Aware Wrapper.

Genesis-original 2026-06-05 — replaces per-file PN298/PN299 patches with
a SINGLE wrapper around `triton.autotune` decorator that filters configs
based on the current GPU's architecture profile.

================================================================
WHY UNIVERSAL
================================================================

PN298 patched 1 file (chunk_o.py). PN299 patched 3 more (kkt, wy_fast,
l2norm). Each new FLA op file would need its own patch. The pattern is
identical — read `gpu_arch_profile`, drop num_warps>max_safe, drop
num_stages>max_safe.

Enterprise-grade approach: monkey-patch `triton.autotune` at import time
to wrap the configs list. ALL @triton.autotune decorators in vllm get
automatic arch-aware filtering — past, present, and future kernels.

Coverage with this single patch:
  - All `vllm/model_executor/layers/fla/ops/*.py` (10+ autotune sites)
  - All `vllm/model_executor/layers/mamba/ops/*.py` (5+ sites)
  - All `vllm/v1/attention/ops/*.py`
  - Any third-party Triton kernels imported into vllm

================================================================
MECHANISM
================================================================

1. Import `triton.runtime.autotuner.Autotuner` class.
2. Replace its `__init__` (or class itself) with a Genesis-wrapped
   version that, before calling super().__init__, filters the
   `configs` list using `prune_triton_autotune_configs(configs)`.
3. The wrap is idempotent (skip if already wrapped).
4. Per-file Triton autotune decorators continue to work — they pass
   through Genesis filter before reaching real Triton.

================================================================
SAFETY MODEL
================================================================

- Pure config filtering — never modifies kernel logic.
- Triton autotune still picks the BEST surviving config via real
  benchmark (we just shrink the search space).
- If `prune_*` returns EMPTY list, restore original (avoid catastrophic
  fallback that breaks all kernels).
- Operator escape hatch: `GENESIS_PN300_DISABLE=1` → no wrapping.
- Logs each unique kernel where pruning happened (first time only).

================================================================
PROFILE READING
================================================================

Reads `get_gpu_arch_profile()` once at install. If profile shows
`max_safe_num_warps=8` (A100, Hopper, Blackwell), wrapping is a no-op
(all configs survive). Only fires on SM 8.x consumer hardware where
filtering actually matters.

Author: Sandermage (Sander) Barzov Aleksandr, 2026-06-05.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.wiring.pn300_universal_triton_autotune_wrapper")


_APPLIED = False
_PRUNED_KERNELS: set[str] = set()


def _make_genesis_config_filter(profile):
    """Build the actual filter function from arch profile."""
    max_warps = profile.max_safe_num_warps
    max_stages = profile.max_safe_num_stages

    def filter_configs(configs):
        """Drop configs with num_warps>max or num_stages>max.

        Returns ORIGINAL configs list if filter would empty it (safety).
        """
        if not configs:
            return configs
        pruned = []
        for cfg in configs:
            nw = getattr(cfg, "num_warps", None)
            ns = getattr(cfg, "num_stages", None)
            if nw is not None and nw > max_warps:
                continue
            if ns is not None and ns > max_stages:
                continue
            pruned.append(cfg)
        if not pruned:
            # Empty filter result — original was already minimal.
            # Don't break autotune: keep original.
            return configs
        return pruned

    return filter_configs


def _wrap_autotuner(profile) -> tuple[bool, str]:
    """Monkey-patch triton.runtime.autotuner.Autotuner.__init__.

    Returns (success, detail_msg).
    """
    try:
        import triton
        from triton.runtime.autotuner import Autotuner
    except Exception as e:
        return False, f"triton import failed: {e}"

    # Idempotent: skip if already wrapped
    if getattr(Autotuner, "_genesis_pn300_wrapped", False):
        return True, "Autotuner already wrapped (idempotent)"

    config_filter = _make_genesis_config_filter(profile)
    original_init = Autotuner.__init__

    def wrapped_init(self, fn, arg_names, configs, *args, **kwargs):
        """Wrapped Autotuner.__init__ — filters configs before delegation."""
        try:
            original_len = len(configs) if configs else 0
            filtered = config_filter(configs)
            new_len = len(filtered)
            if new_len < original_len:
                # Log first time per kernel
                kernel_name = getattr(fn, "__name__", "<anon>")
                if kernel_name not in _PRUNED_KERNELS:
                    _PRUNED_KERNELS.add(kernel_name)
                    log.warning(
                        "[PN300] Triton autotune pruned for %s: %d → %d configs "
                        "(max_warps=%d, max_stages=%d)",
                        kernel_name, original_len, new_len,
                        profile.max_safe_num_warps,
                        profile.max_safe_num_stages,
                    )
            configs = filtered
        except Exception as e:
            # Filter failed — fall back to original configs (safety).
            log.warning(
                "[PN300] config filter raised %s for kernel %s — using original",
                type(e).__name__, getattr(fn, "__name__", "<anon>"),
            )
        return original_init(self, fn, arg_names, configs, *args, **kwargs)

    # Install wrapper
    Autotuner.__init__ = wrapped_init
    Autotuner._genesis_pn300_wrapped = True
    return True, "Autotuner.__init__ wrapped"


def apply() -> tuple[str, str]:
    """Apply PN300 — universal Triton autotune wrapping."""
    global _APPLIED

    if os.environ.get(
        "GENESIS_ENABLE_PN300_UNIVERSAL_TRITON_AUTOTUNE_WRAPPER", ""
    ).lower() not in ("1", "true", "yes", "on"):
        return "skipped", (
            "PN300 default OFF — set "
            "GENESIS_ENABLE_PN300_UNIVERSAL_TRITON_AUTOTUNE_WRAPPER=1. "
            "Monkey-patches triton.runtime.autotuner.Autotuner to filter "
            "configs by current arch's max_safe_num_warps/stages. Single "
            "wrapper replaces per-file PN298/PN299 patches."
        )

    if os.environ.get("GENESIS_PN300_DISABLE", "").lower() in ("1", "true", "yes"):
        return "skipped", "GENESIS_PN300_DISABLE=1 forces escape"

    try:
        from sndr.detection.gpu_arch_profile import (
            get_gpu_arch_profile,
        )
    except Exception as e:
        return "failed", f"gpu_arch_profile import failed: {e}"

    profile = get_gpu_arch_profile()
    if profile is None:
        return "skipped", "GPU not detected (non-CUDA or detection failed)"

    # Optimization: if max_warps>=8 (A100, Hopper, Blackwell), wrapping
    # is a no-op for any reasonable autotune list. Skip to avoid the
    # tiny per-decorator overhead.
    if profile.max_safe_num_warps >= 8 and profile.max_safe_num_stages >= 4:
        return "skipped", (
            f"PN300 not needed on this arch — max_warps={profile.max_safe_num_warps}, "
            f"max_stages={profile.max_safe_num_stages} (no configs to prune)"
        )

    success, detail = _wrap_autotuner(profile)
    if not success:
        return "failed", detail

    _APPLIED = True
    return "applied", (
        f"PN300 installed: Triton autotuner wrapped — configs filtered by "
        f"max_warps={profile.max_safe_num_warps}, max_stages={profile.max_safe_num_stages}. "
        f"Coverage: ALL @triton.autotune decorators across vllm. {detail}"
    )


def is_applied() -> bool:
    return _APPLIED
