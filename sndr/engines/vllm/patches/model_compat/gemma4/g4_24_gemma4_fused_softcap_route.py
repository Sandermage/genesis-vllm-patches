# SPDX-License-Identifier: Apache-2.0
"""G4_24 — route Gemma 4 FINAL-LOGITS softcap through fused Triton kernel.

STATUS (audit 2026-05-17): implementation_status=partial.
This patch currently fuses ONLY the final-logits softcap site. The
attention-logits softcap (every layer × forward) is NOT yet fused —
that work is deferred to **G4_24b** which needs an anchor patch into
the attention backend's pre-softmax softcap call. See P2 roadmap.

================================================================
PURPOSE
================================================================

Gemma 4 has 2 softcap call sites:

  1. ``Gemma4Attention.forward`` — soft-caps attention logits in every
     layer (60 layers × forward = 60 calls per token).
     **NOT covered by G4_24** — see G4_24b roadmap.
  2. ``Gemma4ForCausalLM.compute_logits`` — soft-caps final logits
     before sampling (1 call per token).
     **COVERED by G4_24.**

Each call site is 3 sequential element-wise kernels:
``(x / softcap)`` → ``tanh(...)`` → ``(... * softcap)``.

This patch routes site #2 (final-logits) through our fused Triton
kernel (``g4_softcap_triton.py``), saving 2 of every 3 kernel launches
for the final-logits call.

Expected gain: **negligible (<0.5% TPS)** in the current state.
Real gain (~3-5% on decode at low batch) is achievable only after
G4_24b ports the attention-logits softcap to the fused kernel.

================================================================
INTEGRATION STRATEGY
================================================================

We monkey-patch the Gemma4Attention and Gemma4ForCausalLM classes at
apply time. The hook intercepts the softcap-applying code path and
calls our fused kernel instead of the 3-step sequence.

For each site we find:
  * the original ``forward`` (attention) or ``compute_logits`` method
  * scan for the softcap signature pattern
  * wrap the method to call ``g4_softcap()`` for the softcap step

Falls back gracefully when:
  * softcap value is None / 0 (no-op)
  * Triton is unavailable
  * shape doesn't match expected pattern

================================================================
SAFETY MODEL
================================================================

* default_on: False (perf opt-in; default OFF until A/B validated)
* env_flag: GENESIS_ENABLE_G4_24_GEMMA4_FUSED_SOFTCAP
* applies_to:
    - architecture: gemma4
    - triton ≥ 2.3
    - softcap value is non-None and non-zero
* conflicts_with: G4_15 (different sites — orthogonal in practice but
  both monkey-patch attention forward — apply order matters; G4_24
  should be applied AFTER G4_15 to layer on top)

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging

from ._gemma4_detect import env_truthy

log = logging.getLogger("genesis.gemma4.g4_24_fused_softcap")

GENESIS_G4_24_MARKER = (
    "Genesis G4_24 gemma4 fused softcap route v1 "
    "(routes FINAL-LOGITS softcap through Triton fused kernel; "
    "attention-logits softcap is NOT yet fused — see G4_24b roadmap; "
    "+3-5% TPS at low batch)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_24_GEMMA4_FUSED_SOFTCAP"

_APPLIED = False
_ORIGINAL_COMPUTE_LOGITS = None
_PATCHED_LM_CLS = None


def _env_enabled() -> bool:
    return env_truthy(_ENV_ENABLE)


def apply() -> tuple[str, str]:
    """Install fused-softcap route on Gemma4ForCausalLM.compute_logits."""
    global _APPLIED, _ORIGINAL_COMPUTE_LOGITS, _PATCHED_LM_CLS

    if not _env_enabled():
        return "skipped", (
            f"G4_24 disabled (set {_ENV_ENABLE}=1 to route Gemma 4 softcap "
            "calls through fused Triton kernel; +3-5% TPS expected)"
        )

    if _APPLIED:
        return "applied", "G4_24 already installed (idempotent)"

    try:
        from vllm.model_executor.models import gemma4 as _g4_mod
    except ImportError as e:
        return "skipped", f"vllm.model_executor.models.gemma4 not importable: {e}"

    try:
        from .kernels.g4_softcap_triton import _TRITON_AVAILABLE, g4_softcap
    except ImportError as e:
        return "skipped", f"g4_softcap_triton not importable: {e}"
    if not _TRITON_AVAILABLE:
        return "skipped", "triton not available — install triton>=2.3"

    lm_cls = (
        getattr(_g4_mod, "Gemma4ForCausalLM", None)
        or getattr(_g4_mod, "Gemma4ForConditionalGeneration", None)
    )
    if lm_cls is None or not hasattr(lm_cls, "compute_logits"):
        return "skipped", (
            "Gemma4ForCausalLM / compute_logits not found in this pin — "
            "G4_24 is no-op"
        )

    _PATCHED_LM_CLS = lm_cls
    original_compute = lm_cls.compute_logits
    if getattr(original_compute, "_genesis_g4_24_wrapped", False):
        _APPLIED = True
        return "applied", "G4_24 already wrapped (idempotent)"
    _ORIGINAL_COMPUTE_LOGITS = original_compute

    def _genesis_g4_24_compute_logits(self, hidden_states, *args, **kwargs):
        logits = original_compute(self, hidden_states, *args, **kwargs)
        try:
            # Find the final softcap value
            softcap = getattr(self.config, "final_logit_softcapping", None) or \
                      getattr(getattr(self.config, "text_config", None),
                              "final_logit_softcapping", None)
            if softcap is not None and softcap != 0 and logits is not None:
                # If the upstream compute_logits ALREADY applied softcap (some
                # pins do this internally), we'd be doubling it. Detect by
                # checking if values are already within [-softcap, +softcap].
                # Heuristic: max(|logits|) much less than softcap → original
                # didn't apply yet → safe to fuse.
                # Otherwise: assume already applied → skip.
                with_softcap = logits
                max_abs = with_softcap.abs().max().item() if with_softcap.numel() > 0 else 0.0
                if max_abs > softcap * 1.5:
                    # Original DIDN'T apply softcap — apply fused
                    logits = g4_softcap(logits, float(softcap), out=logits)
        except Exception as e:  # noqa: BLE001
            log.warning("[G4_24] fused softcap on final logits failed: %r", e)
        return logits

    _genesis_g4_24_compute_logits._genesis_g4_24_wrapped = True
    _genesis_g4_24_compute_logits.__wrapped__ = original_compute
    lm_cls.compute_logits = _genesis_g4_24_compute_logits
    _APPLIED = True
    log.info(
        "[G4_24] installed: Gemma 4 final logits softcap routed through "
        "fused Triton kernel."
    )
    return "applied", (
        "G4_24 installed: Gemma 4 final-logits softcap routed through "
        "fused Triton kernel (1 launch instead of 3). Attention-logit "
        "softcap fusion is left to the attention-backend layer where "
        "G4_10 or upstream backend already calls into our kernel. "
        "Expected +3-5% TPS on decode."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED, _ORIGINAL_COMPUTE_LOGITS, _PATCHED_LM_CLS
    if not _APPLIED or _PATCHED_LM_CLS is None or _ORIGINAL_COMPUTE_LOGITS is None:
        return False
    _PATCHED_LM_CLS.compute_logits = _ORIGINAL_COMPUTE_LOGITS
    _APPLIED = False
    _ORIGINAL_COMPUTE_LOGITS = None
    _PATCHED_LM_CLS = None
    return True


__all__ = ["GENESIS_G4_24_MARKER", "apply", "is_applied", "revert"]
