# SPDX-License-Identifier: Apache-2.0
"""G4_01 — refuse FP8_BLOCK Gemma 4 checkpoints on Ampere SM 8.6.

================================================================
WHAT BREAKS WITHOUT THIS PATCH
================================================================

When an operator launches Gemma 4 with an FP8_BLOCK checkpoint
(e.g. ``RedHatAI/gemma-4-31B-it-FP8-block``) on consumer Ampere
(RTX 3090 / 4090 / A5000 / A6000, SM 8.6), the boot succeeds and
generation produces **silent garbage** — a single token repeated
indefinitely (``" a a a a"`` style output).

Root cause (vllm-project/vllm#39407, OPEN as of 2026-05-17, 19
comments, owner @maralbahari):

  llm-compressor's FP8_BLOCK format stores **pre-absorbed activation
  scales** into the weight tensor at quantization time. vLLM's
  ``compressed_tensors_w8a8_fp8::process_weights_after_loading`` still
  applies dynamic per-token activation quantization at inference,
  so activations are **scaled twice**. Hidden state norms explode
  across layers until every logit saturates at the softcap wall
  (``30·tanh(x/30) ≈ 23.625`` in BF16). With every logit at the
  same value, argmax collapses to a single token.

Operator-visible symptom: 30 minute cold boot (model loads, JIT
warms up, cudagraphs capture) producing only ``" a a a"``-style
output. No error, no warning — just broken.

================================================================
THE FIX (this patch — short term)
================================================================

Refuse to load the broken combination at the earliest point we can
reliably detect it (``process_weights_after_loading``), with a clear
error message pointing at:

  * upstream bug #39407
  * the recommended working stack (cyankiwi AWQ-4bit + MTP assistant)
  * an explicit override env (``GENESIS_DISABLE_G4_01_GUARD=1``) so
    developers testing the deeper G4_07 fix can bypass this.

Saves operators a 30-minute cold-boot-to-garbage debug cycle.

================================================================
DEEP FIX
================================================================

Reach out to G4_07 (``g4_07_gemma4_fp8_block_double_scale_fix.py``)
which **actually closes #39407** by registering a custom quantization
config that detects pre-absorbed scales and skips the second
activation quant. Once G4_07 ships and validates, G4_01's guard
becomes redundant (G4_07 makes the combo work, not just refuse).
At that point G4_01 should be marked ``superseded_by: [G4_07]`` and
default_on flipped to False.

================================================================
SAFETY MODEL
================================================================

* default_on: True (cheap; fires only on the known-broken combo)
* env_flag: GENESIS_ENABLE_G4_01_GEMMA4_FP8_BLOCK_GUARD
* override: GENESIS_DISABLE_G4_01_GUARD=1 (bypass for G4_07 testing)
* applies_to:
    - architecture: gemma4
    - quantization: FP8_BLOCK (compressed-tensors / float-quantized / block)
    - hardware: Ampere SM 8.6
* conflicts_with: []
* superseded_by: [G4_07] (when G4_07 ships + validates)

================================================================
COMPOSITION
================================================================

* Safe alongside G4_02 (different quant family — INT4 vs FP8) and
  G4_03 (drafter-side, doesn't touch weight loading).
* Idempotent: marker on the wrapped method's ``__wrapped__`` attr.
* Never raises out of ``apply()`` itself — guard is installed via
  monkey-patching ``process_weights_after_loading`` so failures
  happen at model-load time only, when the bad combo is loading.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
References:
  * https://github.com/vllm-project/vllm/issues/39407 (root-cause analysis)
  * vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py
"""
from __future__ import annotations

import logging

from ._gemma4_detect import (
    detect_fp8_block_format,
    env_disable,
    env_truthy,
    is_ampere_sm86,
    is_gemma4_arch,
)

log = logging.getLogger("genesis.gemma4.g4_01_fp8_block_guard")

GENESIS_G4_01_MARKER = (
    "Genesis G4_01 gemma4 ampere FP8_BLOCK guard v1 "
    "(closes operator confusion from vllm#39407 silent garbage output)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_01_GEMMA4_FP8_BLOCK_GUARD"
_ENV_DISABLE = "GENESIS_DISABLE_G4_01_GUARD"

_APPLIED = False
_ORIGINAL_PWAL: object = None  # process_weights_after_loading stash


def _env_enabled() -> bool:
    if env_disable(_ENV_DISABLE):
        return False
    return env_truthy(_ENV_ENABLE)


def _resolve_target_vllm_config(layer) -> object | None:
    """Walk a few well-known back-references from layer → vllm_config.

    vLLM doesn't expose a clean ``layer.vllm_config`` everywhere, so we
    probe several attribute names that newer pins use.
    """
    for attr in ("_vllm_config", "vllm_config", "_model_config", "model_config"):
        val = getattr(layer, attr, None)
        if val is not None:
            return val
    # Last resort — vLLM v1 stashes via prefix attribute on weights
    return None


def _is_guarded_combo(layer) -> tuple[bool, str]:
    """Return (refuse, reason) when layer matches the broken combo."""
    if not is_ampere_sm86():
        return False, "not Ampere SM 8.6"
    cfg = _resolve_target_vllm_config(layer)
    if cfg is None:
        return False, "no vllm_config reachable from layer"
    mc = getattr(cfg, "model_config", None) or cfg
    hf = getattr(mc, "hf_config", None) or mc
    if not is_gemma4_arch(hf):
        return False, "not Gemma 4 architecture"
    quant = getattr(hf, "quantization_config", None) or getattr(mc, "quant_config", None)
    if quant is None:
        return False, "no quantization config visible"
    if not detect_fp8_block_format(quant):
        return False, "not FP8_BLOCK format"
    return True, "Gemma 4 + Ampere SM 8.6 + FP8_BLOCK — vllm#39407 broken combo"


def apply() -> tuple[str, str]:
    """Install the ``process_weights_after_loading`` guard. Never raises."""
    global _APPLIED, _ORIGINAL_PWAL

    if not _env_enabled():
        return "skipped", (
            f"G4_01 disabled (set {_ENV_ENABLE}=1 to refuse FP8_BLOCK Gemma 4 "
            f"on Ampere — see vllm#39407 silent garbage output bug)"
        )

    if _APPLIED:
        return "applied", "G4_01 already installed (idempotent)"

    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
            compressed_tensors_w8a8_fp8 as mod,
        )
    except ImportError as e:
        return "skipped", (
            "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
            f"compressed_tensors_w8a8_fp8 not importable: {e}; pin may lack this path"
        )

    target_cls = getattr(mod, "CompressedTensorsW8A8Fp8", None)
    if target_cls is None:
        return "skipped", (
            "CompressedTensorsW8A8Fp8 not found in vllm compressed-tensors schemes — "
            "pin may have renamed the class; G4_01 is no-op"
        )

    original = target_cls.process_weights_after_loading
    if getattr(original, "_genesis_g4_01_wrapped", False):
        _APPLIED = True
        return "applied", "G4_01 already wrapped on target_cls (idempotent)"

    _ORIGINAL_PWAL = original

    def _genesis_g4_01_guarded_pwal(self, layer):
        """Refuse FP8_BLOCK Gemma 4 on Ampere; pass-through otherwise."""
        try:
            should_refuse, reason = _is_guarded_combo(layer)
        except Exception as e:  # noqa: BLE001
            log.warning(
                "[G4_01] detection raised %r; falling through to upstream pwal "
                "to avoid blocking unaffected loads", e,
            )
            should_refuse, reason = False, "detection error"

        if should_refuse:
            raise RuntimeError(
                "[Genesis G4_01] Refusing to load this FP8_BLOCK Gemma 4 checkpoint "
                f"on Ampere SM 8.6: {reason}.\n"
                "\n"
                "This combination is known-broken (vllm-project/vllm#39407): "
                "activation scales pre-absorbed in the checkpoint cause the engine "
                "to apply them a second time at inference, producing silent garbage "
                "output (single-token repetition loops at the softcap wall).\n"
                "\n"
                "RECOMMENDED FIX — switch to an Ampere-compatible variant:\n"
                "  * cyankiwi/gemma-4-31B-it-AWQ-4bit  + google/gemma-4-31B-it-assistant (MTP)\n"
                "  * Intel/gemma-4-31B-it-int4-AutoRound + same MTP assistant\n"
                "\n"
                "OVERRIDE — to test Genesis G4_07 (deep FP8_BLOCK fix), set\n"
                f"  {_ENV_DISABLE}=1\n"
                "and Genesis will let the broken combo through.\n"
            )

        return _ORIGINAL_PWAL(self, layer)

    _genesis_g4_01_guarded_pwal._genesis_g4_01_wrapped = True
    _genesis_g4_01_guarded_pwal._genesis_g4_01_original = original
    _genesis_g4_01_guarded_pwal.__wrapped__ = original

    target_cls.process_weights_after_loading = _genesis_g4_01_guarded_pwal
    _APPLIED = True

    log.info(
        "[G4_01] installed: FP8_BLOCK + Gemma 4 + Ampere SM 8.6 combo will "
        "now raise a clear error at model-load time (vllm#39407)."
    )
    return "applied", (
        "G4_01 installed: Ampere SM 8.6 + Gemma 4 + FP8_BLOCK combo will refuse "
        "at process_weights_after_loading with a pointer to the working stack. "
        f"Override via {_ENV_DISABLE}=1 for G4_07 testing."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    """Restore the original process_weights_after_loading. Returns True on success."""
    global _APPLIED, _ORIGINAL_PWAL
    if not _APPLIED or _ORIGINAL_PWAL is None:
        return False
    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
            compressed_tensors_w8a8_fp8 as mod,
        )
    except ImportError:
        return False
    target_cls = getattr(mod, "CompressedTensorsW8A8Fp8", None)
    if target_cls is None:
        return False
    target_cls.process_weights_after_loading = _ORIGINAL_PWAL  # type: ignore[assignment]
    _APPLIED = False
    _ORIGINAL_PWAL = None
    return True


__all__ = [
    "GENESIS_G4_01_MARKER",
    "apply",
    "is_applied",
    "revert",
]
