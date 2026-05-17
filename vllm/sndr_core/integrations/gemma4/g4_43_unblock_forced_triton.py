# SPDX-License-Identifier: Apache-2.0
"""G4_43 — bypass vllm's forced TRITON_ATTN on Gemma 4 heterogeneous head dims.

================================================================
WHY THIS EXISTS
================================================================

``vllm/model_executor/models/config.py::Gemma4Config.verify_and_update_config``
forces ``attention_config.backend = TRITON_ATTN`` when:

  * ``head_dim != global_head_dim`` (Gemma 4 31B: 256 vs 512)
  * ``max(head_dim, global_head_dim) > 256``
  * No backend explicitly chosen

The intent is to "prevent mixed-backend numerical divergence" — keep
all layers on one backend. But this means:

  * Sliding layers (head=256, fits FA) → forced to TRITON_ATTN
  * Full layers (head=512, needs Triton) → forced to TRITON_ATTN

The forced TRITON_ATTN doesn't support ``turboquant_*`` kv_cache_dtype.
So forcing it blocks ALL real KV compression on Gemma 4.

================================================================
WHAT THIS DOES
================================================================

Hooks ``Gemma4Config.verify_and_update_config`` to skip the forced
backend assignment. This re-enables per-layer auto-pick: each Attention
layer picks its OWN backend based on its kv_cache_dtype:

  * Sliding (kv_cache_dtype="auto" via --kv-cache-dtype-skip-layers
    sliding_window) → vllm picks TRITON_ATTN (head=256, FA OK; auto OK)
  * Full (kv_cache_dtype="turboquant_*") → vllm picks TURBOQUANT
    (which handles head=512 via Triton, no FlashAttn limit)

Mixed-backend execution is documented by vllm as a numerical-divergence
risk — but TurboQuant's API + the TRITON_ATTN-equivalent kernel path
inside TQ produce numerically equivalent attention math for the cache
portion. The "divergence" warning is conservative; we accept the risk
in exchange for REAL memory savings on full-attention layers.

================================================================
ENV FLAG
================================================================

``GENESIS_ENABLE_G4_43_UNBLOCK_TRITON_FORCE=1`` enables the hook.
Default OFF.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.gemma4.g4_43_unblock_triton_force")

GENESIS_G4_43_MARKER = (
    "Genesis G4_43 unblock vllm's forced TRITON_ATTN on Gemma 4 "
    "heterogeneous head dims (allows per-layer auto-pick for TQ on full)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_43_UNBLOCK_TRITON_FORCE"
_APPLIED = False
_ORIGINAL_VERIFY = None


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Install hook on Gemma4Config.verify_and_update_config to skip the
    forced TRITON_ATTN assignment."""
    global _APPLIED, _ORIGINAL_VERIFY

    if not _env_enabled():
        return "skipped", (
            f"G4_43 disabled (set {_ENV_ENABLE}=1 to unblock per-layer "
            "backend auto-pick on Gemma 4 heterogeneous head dims)"
        )

    if _APPLIED:
        return "applied", "G4_43 already installed (idempotent)"

    try:
        from vllm.model_executor.models.config import Gemma4Config
    except ImportError as e:
        return "skipped", (
            f"vllm.model_executor.models.config.Gemma4Config not importable: {e}"
        )

    original = Gemma4Config.verify_and_update_config
    if getattr(original, "_genesis_g4_43_wrapped", False):
        _APPLIED = True
        return "applied", "G4_43 already wrapped (idempotent)"

    _ORIGINAL_VERIFY = original

    def _genesis_skip_triton_force_inner(vllm_config):
        """Call original verify_and_update_config BUT scrub any
        attention_config.backend assignment it made."""
        prev_backend = vllm_config.attention_config.backend
        original(vllm_config)
        new_backend = vllm_config.attention_config.backend
        if new_backend is not None and prev_backend is None:
            # The original assigned a backend; we revert it.
            from vllm.v1.attention.backends.registry import (
                AttentionBackendEnum,
            )
            if new_backend == AttentionBackendEnum.TRITON_ATTN:
                vllm_config.attention_config.backend = None
                log.warning(
                    "[G4_43] reverted vllm's forced TRITON_ATTN backend "
                    "assignment — vllm will auto-pick per Attention layer "
                    "based on its kv_cache_dtype (TQ for full, TRITON for "
                    "sliding)."
                )

    _genesis_skip_triton_force_inner._genesis_g4_43_wrapped = True
    _genesis_skip_triton_force_inner.__wrapped__ = (
        original.__func__ if hasattr(original, "__func__") else original
    )

    Gemma4Config.verify_and_update_config = staticmethod(
        _genesis_skip_triton_force_inner
    )
    _APPLIED = True

    log.info(
        "[G4_43] installed: Gemma4Config.verify_and_update_config will "
        "no longer force TRITON_ATTN; per-layer backend auto-pick enabled."
    )
    return "applied", (
        "G4_43 installed: vllm's forced TRITON_ATTN on Gemma 4 "
        "heterogeneous head dims is now reverted. Each Attention layer "
        "picks its own backend by its kv_cache_dtype."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED, _ORIGINAL_VERIFY
    if not _APPLIED or _ORIGINAL_VERIFY is None:
        return False
    try:
        from vllm.model_executor.models.config import Gemma4Config
        Gemma4Config.verify_and_update_config = _ORIGINAL_VERIFY
        _APPLIED = False
        return True
    except ImportError:
        return False


__all__ = ["GENESIS_G4_43_MARKER", "apply", "is_applied", "revert"]
