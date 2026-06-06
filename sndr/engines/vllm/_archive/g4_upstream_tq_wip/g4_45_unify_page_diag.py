# SPDX-License-Identifier: Apache-2.0
"""G4_45 — diagnostic + auto-fix for KVCache page-size unification.

When kv_cache_dtype is ``turboquant_*`` on Gemma 4 (mixed head_dim:
256 sliding, 512 full), vllm's ``unify_kv_cache_spec_page_size`` may
fail with::

    NotImplementedError: The page size of the layer is not divisible
    by the maximum page size. Cannot unify by adjusting block_size.

This patch:
  1. Logs each layer's ``page_size_bytes`` BEFORE unification for
     diagnostics.
  2. If unification fails (max %% min != 0), it pads the smaller
     layer's ``page_size_bytes`` (via a wrapped spec) to align with
     the max — accepting a small memory overhead on smaller layers
     in exchange for boot success.

================================================================
ENV FLAG
================================================================

``GENESIS_ENABLE_G4_45_UNIFY_DIAG=1`` — diagnostic mode (logs only).
``GENESIS_ENABLE_G4_45_UNIFY_FIX=1`` — auto-pad to unify (boot rescue).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.gemma4.g4_45_unify")

GENESIS_G4_45_MARKER = (
    "Genesis G4_45 KVCache page-size unification diagnostic + auto-fix"
)

_ENV_DIAG = "GENESIS_ENABLE_G4_45_UNIFY_DIAG"
_ENV_FIX = "GENESIS_ENABLE_G4_45_UNIFY_FIX"
_APPLIED = False
_ORIGINAL = None


def _env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    global _APPLIED, _ORIGINAL

    if not (_env(_ENV_DIAG) or _env(_ENV_FIX)):
        return "skipped", (
            f"G4_45 disabled (set {_ENV_DIAG}=1 for log-only diag, or "
            f"{_ENV_FIX}=1 for auto-pad fix)"
        )

    if _APPLIED:
        return "applied", "G4_45 already installed (idempotent)"

    try:
        from vllm.v1.core import kv_cache_utils
    except ImportError as e:
        return "skipped", f"vllm.v1.core.kv_cache_utils not importable: {e}"

    original = kv_cache_utils.unify_kv_cache_spec_page_size
    if getattr(original, "_genesis_g4_45_wrapped", False):
        _APPLIED = True
        return "applied", "G4_45 already wrapped (idempotent)"

    _ORIGINAL = original
    fix_mode = _env(_ENV_FIX)

    def _wrapped_unify(kv_cache_spec):
        # Log each spec's page_size for diagnostic
        page_sizes = {}
        for layer_name, spec in kv_cache_spec.items():
            ps = spec.page_size_bytes
            page_sizes[layer_name] = ps
        unique_sizes = sorted(set(page_sizes.values()))
        log.warning(
            "[G4_45 DIAG] unify_kv_cache_spec_page_size: %d layers, "
            "%d unique page sizes: %s",
            len(kv_cache_spec), len(unique_sizes), unique_sizes,
        )
        # Group by page_size and log count of each
        from collections import Counter
        size_counts = Counter(page_sizes.values())
        for sz, cnt in sorted(size_counts.items()):
            log.warning(
                "[G4_45 DIAG]   page_size=%d bytes (%.1f KB) × %d layers",
                sz, sz / 1024, cnt,
            )

        try:
            return original(kv_cache_spec)
        except NotImplementedError as e:
            if not fix_mode:
                log.error(
                    "[G4_45] unify failed: %s. Re-run with "
                    "GENESIS_ENABLE_G4_45_UNIFY_FIX=1 to auto-pad.",
                    e,
                )
                raise

            # Auto-pad: pad smaller layers' page_size to lcm/max so they
            # divide. Simple strategy: set ALL layer specs to max(page_size).
            max_ps = max(page_sizes.values())
            log.warning(
                "[G4_45 FIX] auto-padding all layers to page_size=%d (max)",
                max_ps,
            )
            from dataclasses import replace
            new_spec = {}
            for layer_name, spec in kv_cache_spec.items():
                current_ps = spec.page_size_bytes
                if current_ps == max_ps:
                    new_spec[layer_name] = spec
                    continue
                # Increase block_size proportionally if it divides
                ratio_f = max_ps / current_ps
                if ratio_f.is_integer():
                    ratio = int(ratio_f)
                    new_block = spec.block_size * ratio
                    new_spec[layer_name] = replace(spec, block_size=new_block)
                    log.warning(
                        "[G4_45 FIX]   %s: block_size %d→%d (ratio %d)",
                        layer_name, spec.block_size, new_block, ratio,
                    )
                else:
                    # Non-integer ratio — can't pad cleanly. Try
                    # setting page_size_padded if spec supports it.
                    if hasattr(spec, "page_size_padded"):
                        new_spec[layer_name] = replace(
                            spec, page_size_padded=max_ps,
                        )
                        log.warning(
                            "[G4_45 FIX]   %s: page_size_padded=%d",
                            layer_name, max_ps,
                        )
                    else:
                        log.error(
                            "[G4_45 FIX]   %s: cannot unify (ratio=%.3f)",
                            layer_name, ratio_f,
                        )
                        raise
            return new_spec

    _wrapped_unify._genesis_g4_45_wrapped = True
    _wrapped_unify.__wrapped__ = original
    kv_cache_utils.unify_kv_cache_spec_page_size = _wrapped_unify
    _APPLIED = True

    mode = "FIX" if fix_mode else "DIAG"
    log.info("[G4_45] installed (mode=%s)", mode)
    return "applied", (
        f"G4_45 installed in {mode} mode: unify_kv_cache_spec_page_size "
        "now logs per-layer page sizes" +
        (" + auto-pads on divisibility failure" if fix_mode else "")
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED, _ORIGINAL
    if not _APPLIED or _ORIGINAL is None:
        return False
    try:
        from vllm.v1.core import kv_cache_utils
        kv_cache_utils.unify_kv_cache_spec_page_size = _ORIGINAL
        _APPLIED = False
        return True
    except ImportError:
        return False


__all__ = ["GENESIS_G4_45_MARKER", "apply", "is_applied", "revert"]
