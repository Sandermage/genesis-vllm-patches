# SPDX-License-Identifier: Apache-2.0
"""G4_50 — install Genesis-native TurboQuant attention backend.

This is the single apply entry-point that wires the entire Genesis-
native TQ stack for Gemma 4:

  1. Register ``GENESIS_G4_TQ`` in vllm's ``AttentionBackendEnum``.
  2. Mark our backend in the vllm config so it's selectable.
  3. (Caller) launches vllm with:
       --kv-cache-dtype genesis_tq_3bit_full
       --attention-backend GENESIS_G4_TQ
     OR sets these as defaults via env-vars.

The full Genesis TQ package is in
``vllm/sndr_core/integrations/gemma4/genesis_tq/``.

================================================================
STATUS — INITIAL SCAFFOLD
================================================================

This patch installs the BACKEND CLASS and registers it with vllm's
selector. The forward path implementation (impl.py) is currently
a FUNCTIONAL SCAFFOLD that:

  * Compresses K/V on write (correct via existing Genesis kernels)
  * Runs torch SDPA on the current-chunk K/V (correct for prefill)
  * **TODO**: scatter packed K/V into cache buffer (write stub)
  * **TODO**: gather + decompress from cache for decode/continuation
  * **TODO**: GQA-aware variable-seq-len attention with cache

Mode of operation today:
  * Sliding layers: behave like fresh-each-step attention (no cache reuse)
  * Decode beyond 1 token: cache reads return zeros (stub)

Sufficient for boot-validation + smoke test. Production-grade
correctness requires the scatter/gather kernel work in impl.py
(see TODO markers).

================================================================
ENV FLAG
================================================================

``GENESIS_ENABLE_G4_50_NATIVE_TQ=1`` enables registration.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.gemma4.g4_50_native_tq")

GENESIS_G4_50_MARKER = (
    "Genesis G4_50 native TurboQuant attention backend — installs "
    "per-tier compressed KV cache for Gemma 4's 3-tier architecture"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_50_NATIVE_TQ"
_APPLIED = False


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Register Genesis G4 TQ backend with vllm."""
    global _APPLIED

    if not _env_enabled():
        return "skipped", (
            f"G4_50 disabled (set {_ENV_ENABLE}=1 to register the "
            "Genesis-native TurboQuant backend with vllm)"
        )

    if _APPLIED:
        return "applied", "G4_50 already installed (idempotent)"

    try:
        from .genesis_tq.register import register_backend
    except ImportError as e:
        return "skipped", f"genesis_tq package not importable: {e}"

    ok = register_backend()
    if not ok:
        return "skipped", "G4_50 register_backend() returned False"

    _APPLIED = True
    log.info("[G4_50] Genesis G4 TQ backend registered with vllm")
    return "applied", (
        "G4_50 installed: GENESIS_G4_TQ backend registered. To activate, "
        "launch vllm with `--kv-cache-dtype genesis_tq_3bit_full "
        "--attention-backend GENESIS_G4_TQ`."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    """Backend registration is one-way at process level; revert is no-op."""
    return False


__all__ = ["GENESIS_G4_50_MARKER", "apply", "is_applied", "revert"]
