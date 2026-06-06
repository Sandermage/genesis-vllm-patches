# SPDX-License-Identifier: Apache-2.0
"""G4_31 — preserve ``turboquant_*`` kv_cache_dtype against quant-config overrides.

================================================================
PROBLEM
================================================================

When booting Gemma 4 AWQ-4bit with ``--kv-cache-dtype turboquant_4bit_nc
--attention-backend TURBOQUANT`` + G4_30 (multimodal unblock), the
``TurboQuantAttentionBackend`` validation rejects with::

    ValueError: Selected backend AttentionBackendEnum.TURBOQUANT is not
    valid for this configuration. Reason: ['kv_cache_dtype not supported']

The TQ backend's ``supports_kv_cache_dtype()`` accepts any ``turboquant_*``
string and rejects everything else. So at validation time, the dtype
must have already been overridden away from ``turboquant_4bit_nc``.

Tracing ``vllm/model_executor/layers/attention/attention.py::Attention.__init__``
shows the override path::

    kv_cache_dtype = cache_config.cache_dtype       # "turboquant_4bit_nc"
    kv_cache_scheme = getattr(quant_config, "kv_cache_scheme", None)
    if kv_cache_scheme is not None:                 # <-- AWQ sets this
        kv_cache_dtype = "fp8"                      # <-- override!
        ...

The llm-compressor / AWQ quant config carries a ``kv_cache_scheme``
hint intended for FP8 KV caches. When set, vllm hard-overrides
``kv_cache_dtype`` to "fp8" — silently discarding our CLI flag and the
operator's TurboQuant intent.

================================================================
FIX
================================================================

Hook ``Attention.__init__`` to detect the situation
``cache_config.cache_dtype.startswith("turboquant_")`` AND
``quant_config.kv_cache_scheme is not None`` — and **suppress** the
``kv_cache_scheme``-driven override by temporarily clearing
``kv_cache_scheme`` on the quant_config for the duration of this
``__init__`` call. The override path then sees ``None`` and the
CLI-supplied ``turboquant_4bit_nc`` propagates through.

We restore the original ``kv_cache_scheme`` after ``__init__`` returns
so any downstream code that reads it still sees the model's true
intent.

================================================================
SCOPE
================================================================

This only changes behavior when **all three** conditions hold:
  1. ``cache_config.cache_dtype`` starts with ``turboquant_``
  2. ``quant_config.kv_cache_scheme`` is set (AWQ / llm-compressor)
  3. The operator opted in via ``GENESIS_ENABLE_G4_31_TQ_DTYPE_PRESERVE=1``

For non-TurboQuant operators, this patch is a no-op even when enabled.

================================================================
RISK
================================================================

The model's ``kv_cache_scheme`` typically controls FP8 weight
calibration metadata. By suppressing it during ``Attention.__init__``,
we tell vllm "do NOT use the model's FP8 scheme for KV cache" — but
the model itself still uses its own FP8 weights elsewhere. The
TurboQuant attention backend manages its own quantization for the KV
cache, so this is the correct behavior.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.turboquant.g4_31_tq_dtype_preserve")

GENESIS_G4_31_MARKER = (
    "Genesis G4_31 preserve turboquant_* kv_cache_dtype against AWQ "
    "kv_cache_scheme override (Attention.__init__ wrap)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_31_TQ_DTYPE_PRESERVE"
_APPLIED = False
_ORIGINAL_INIT = None
_DEBUG_HITS: list = []


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Install Attention.__init__ wrap that preserves turboquant_* dtype."""
    global _APPLIED, _ORIGINAL_INIT

    if not _env_enabled():
        return "skipped", (
            f"G4_31 disabled (set {_ENV_ENABLE}=1 to preserve "
            "turboquant_* kv_cache_dtype against AWQ kv_cache_scheme override)"
        )

    if _APPLIED:
        return "applied", "G4_31 already installed (idempotent)"

    try:
        from vllm.model_executor.layers.attention.attention import Attention
    except ImportError as e:
        return "skipped", f"vllm.model_executor.layers.attention not importable: {e}"

    original = Attention.__init__
    if getattr(original, "_genesis_g4_31_wrapped", False):
        _APPLIED = True
        return "applied", "Attention.__init__ already wrapped (idempotent)"

    _ORIGINAL_INIT = original

    def _wrapped_init(self, *args, **kwargs):
        cache_config = kwargs.get("cache_config")
        quant_config = kwargs.get("quant_config")
        cache_dtype = getattr(cache_config, "cache_dtype", None) if cache_config else None
        scheme = getattr(quant_config, "kv_cache_scheme", None) if quant_config else None

        # Unconditional diagnostic log for first 3 calls (will be removed
        # once we understand the actual override path)
        if len(_DEBUG_HITS) < 3:
            _DEBUG_HITS.append((kwargs.get("prefix", "?"), cache_dtype, scheme))
            log.warning(
                "[G4_31 DIAG] Attention.__init__ prefix=%s cache_dtype=%r "
                "quant_cls=%s kv_cache_scheme=%r",
                kwargs.get("prefix", "?"), cache_dtype,
                type(quant_config).__name__ if quant_config else None,
                scheme,
            )

        should_suppress = (
            isinstance(cache_dtype, str)
            and cache_dtype.startswith("turboquant_")
            and scheme is not None
        )

        if should_suppress:
            # Log first 3 hits for diagnostics
            if len(_DEBUG_HITS) < 3:
                _DEBUG_HITS.append(kwargs.get("prefix", "?"))
                log.warning(
                    "[G4_31] suppressing kv_cache_scheme override at %s: "
                    "cache_dtype=%s, prior scheme=%r",
                    kwargs.get("prefix", "?"), cache_dtype, scheme,
                )
            try:
                quant_config.kv_cache_scheme = None
                return original(self, *args, **kwargs)
            finally:
                # Restore — keep the model's true intent visible to other code
                try:
                    quant_config.kv_cache_scheme = scheme
                except Exception:  # noqa: BLE001
                    pass
        else:
            return original(self, *args, **kwargs)

    _wrapped_init._genesis_g4_31_wrapped = True
    _wrapped_init.__wrapped__ = original
    Attention.__init__ = _wrapped_init

    # Additional diagnostic: hook supports_kv_cache_dtype on TQ backend
    # to capture what dtype actually reaches the validator.
    try:
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionBackend,
        )
        _orig_supports = TurboQuantAttentionBackend.supports_kv_cache_dtype
        _supports_hits = []

        def _diag_supports_kv_cache_dtype(cls, kv_cache_dtype):
            result = _orig_supports.__func__(cls, kv_cache_dtype)
            if len(_supports_hits) < 5:
                _supports_hits.append((kv_cache_dtype, result))
                log.warning(
                    "[G4_31 SUPPORTS] supports_kv_cache_dtype(%r) -> %r",
                    kv_cache_dtype, result,
                )
            return result

        TurboQuantAttentionBackend.supports_kv_cache_dtype = classmethod(
            _diag_supports_kv_cache_dtype
        )
        log.info("[G4_31] diagnostic supports_kv_cache_dtype hook installed")
    except Exception as e:  # noqa: BLE001
        log.warning("[G4_31] could not hook supports_kv_cache_dtype: %r", e)

    _APPLIED = True

    log.info(
        "[G4_31] installed: Attention.__init__ now suppresses "
        "kv_cache_scheme override when cache_dtype is turboquant_*."
    )
    return "applied", (
        "G4_31 installed: AWQ kv_cache_scheme override is suppressed when "
        "the operator has requested turboquant_* kv-cache-dtype, allowing "
        "the TURBOQUANT attention backend to validate successfully."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED, _ORIGINAL_INIT
    if not _APPLIED or _ORIGINAL_INIT is None:
        return False
    try:
        from vllm.model_executor.layers.attention.attention import Attention
        Attention.__init__ = _ORIGINAL_INIT
        _APPLIED = False
        return True
    except ImportError:
        return False


__all__ = ["GENESIS_G4_31_MARKER", "apply", "is_applied", "revert"]
