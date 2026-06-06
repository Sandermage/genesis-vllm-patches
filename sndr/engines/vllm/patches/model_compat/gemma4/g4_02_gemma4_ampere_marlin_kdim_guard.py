# SPDX-License-Identifier: Apache-2.0
"""G4_02 — refuse Gemma 4 MoE with K%128≠0 Marlin shape on Ampere.

================================================================
WHAT BREAKS WITHOUT THIS PATCH
================================================================

Gemma 4 26B-A4B-it has ``moe_intermediate_size = 704``. When loaded
at TP=2 the per-partition intermediate shrinks to ``704 / 2 = 352``,
which becomes the K dimension of the ``down_proj`` Marlin GEMM:

  MKN = [M_batch, 352, hidden=2816], num_bits = 8 (FP8 path)

vLLM's Marlin tile-finder (csrc/moe/marlin_moe_wna16/ops.cu:230):

  if (prob_k % th_config.thread_k != 0) return false;

Available ``thread_k`` values: ``64, 128``. Available C++ ``min_thread_k``:
``64``. Python-side stricter ``GPTQ_MARLIN_MIN_THREAD_K = 128``.

  * 352 % 64 = 32  → no tile config valid
  * 352 % 128 = 96 → fails stricter check too

No tile configuration satisfies the divisibility constraint, so the
kernel raises an opaque "Invalid thread config" error at the first
dummy-batch forward, **after** model load + JIT + cudagraph capture
have all completed (~3-5 minute cold boot wasted).

Affects:
  * Gemma 4 26B-A4B + TP=2 + FP8 (compressed-tensors block)
  * Gemma 4 26B-A4B + TP=2 + AWQ (W4A16 Marlin)
  * Any future Gemma-like MoE with moe_intermediate not divisible by 128

Does NOT affect:
  * Gemma 4 31B dense (intermediate=21504 → divisible by 128)
  * Gemma 4 26B-A4B + TP=1 (intermediate=704 → divisible by 64 only —
    still works via C++ fallback)
  * Gemma 4 26B-A4B with K-pad fallback (G4_08 — see below)

================================================================
THE FIX (this patch — short term)
================================================================

Wrap ``CompressedTensorsMoEWNA16MarlinMethod.apply_weights`` (and
the WNA16 sibling for AWQ) with a shape-precheck. If we infer
``prob_k % 128 ≠ 0``, raise a clear ``RuntimeError`` pointing at:

  * the Marlin divisibility constraint
  * the recommended workaround (TP=1 with AWQ)
  * the deep fix (G4_08 K-pad Triton MoE fallback)

================================================================
DEEP FIX
================================================================

Reach out to G4_08 (``g4_08_gemma4_marlin_kdim_pad_fallback.py``)
which **implements a Triton MoE GEMM kernel** that pads K to the
next multiple of 64 and masks out the padding zeros at load time.
That is the canonical solution and will be the upstream PR back to
``vllm-project/vllm``.

Once G4_08 ships and validates, G4_02's guard auto-disables when
G4_08 is enabled (G4_08 detects K%64≠0 and routes to its own kernel).

================================================================
SAFETY MODEL
================================================================

* default_on: True (cheap; fires only on broken K dim)
* env_flag: GENESIS_ENABLE_G4_02_GEMMA4_MARLIN_KDIM_GUARD
* override: GENESIS_DISABLE_G4_02_GUARD=1 (bypass for G4_08 testing)
* superseded_by: [G4_08] (when G4_08 fallback ships)

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
References:
  * https://github.com/vllm-project/vllm/issues/40354 (Ampere W4A16 TP=2 Marlin)
  * https://github.com/vllm-project/vllm/issues/41403 (TQ + Gemma 4 5-gate tracker)
  * csrc/moe/marlin_moe_wna16/ops.cu:220-260 (is_valid_config + tile-finder)
  * vllm/model_executor/layers/quantization/utils/marlin_utils.py:27-28
"""
from __future__ import annotations

import logging

from ._gemma4_detect import (
    env_disable,
    env_truthy,
    infer_marlin_kdim_for_moe,
    is_ampere_sm86,
    is_gemma4_arch,
    marlin_kdim_supported,
)

log = logging.getLogger("genesis.gemma4.g4_02_marlin_kdim_guard")

GENESIS_G4_02_MARKER = (
    "Genesis G4_02 gemma4 ampere Marlin K-dim guard v1 "
    "(closes operator confusion from vllm#40354 obscure tile-config error)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_02_GEMMA4_MARLIN_KDIM_GUARD"
_ENV_DISABLE = "GENESIS_DISABLE_G4_02_GUARD"

_APPLIED = False
_ORIGINAL_METHODS: dict[str, object] = {}


def _env_enabled() -> bool:
    if env_disable(_ENV_DISABLE):
        return False
    return env_truthy(_ENV_ENABLE)


def _layer_is_gemma4(layer) -> bool:
    """Best-effort: probe a FusedMoE layer for its parent model architecture."""
    for attr in ("_vllm_config", "vllm_config", "model_config", "_model_config"):
        cfg = getattr(layer, attr, None)
        if cfg is not None:
            mc = getattr(cfg, "model_config", None) or cfg
            hf = getattr(mc, "hf_config", None) or mc
            if is_gemma4_arch(hf):
                return True
    return False


def _make_guarded_apply(original):
    """Wrap a FusedMoE method's apply() with a K-dim precheck."""

    def _genesis_g4_02_guarded_apply(self, layer, *args, **kwargs):
        try:
            if is_ampere_sm86() and _layer_is_gemma4(layer):
                prob_k = infer_marlin_kdim_for_moe(layer)
                if prob_k is not None and not marlin_kdim_supported(prob_k, strict_python_check=True):
                    # Check G4_08 active via env (operator opted in to deep fix)
                    import os
                    if os.environ.get("GENESIS_ENABLE_G4_08_MARLIN_KDIM_PAD", "").strip() in ("1", "true", "yes", "on"):
                        # Caller has the K-pad fallback; let the call go through
                        # (G4_08's method override will route correctly).
                        log.info(
                            "[G4_02] prob_k=%d not Marlin-aligned but G4_08 enabled — "
                            "deferring to G4_08 fallback", prob_k,
                        )
                    else:
                        raise RuntimeError(
                            "[Genesis G4_02] Refusing Marlin MoE GEMM with "
                            f"prob_k={prob_k} on Ampere SM 8.6.\n"
                            "\n"
                            "Marlin requires prob_k divisible by min_thread_k=64 (C++) "
                            "and by 128 (Python verifier). prob_k=" + str(prob_k) +
                            f" gives {prob_k}%64={prob_k%64}, {prob_k}%128={prob_k%128}. "
                            "No tile config can dispatch this GEMM, kernel will "
                            "raise an obscure 'Invalid thread config' error.\n"
                            "\n"
                            "This affects Gemma 4 26B-A4B at TP=2: moe_intermediate=704 / "
                            "TP=2 = 352, and 352 fails the divisibility constraint.\n"
                            "\n"
                            "WORKAROUND — run at TP=1 (AWQ weights fit on one A5000):\n"
                            "  sndr launch prod-gemma4-26b-a4b   # TP=1 preset\n"
                            "\n"
                            "DEEP FIX — enable Genesis G4_08 Triton K-pad fallback:\n"
                            "  GENESIS_ENABLE_G4_08_MARLIN_KDIM_PAD=1\n"
                            "(requires G4_08 implemented; check ``sndr patches list "
                            "--family gemma4``)\n"
                            "\n"
                            "OVERRIDE — bypass to get the raw kernel error:\n"
                            f"  {_ENV_DISABLE}=1\n"
                        )
        except Exception as e:  # noqa: BLE001
            if not isinstance(e, RuntimeError) or "[Genesis G4_02]" not in str(e):
                log.warning(
                    "[G4_02] precheck raised %r; falling through to upstream "
                    "to avoid blocking unaffected loads", e,
                )
            else:
                raise

        return original(self, layer, *args, **kwargs)

    _genesis_g4_02_guarded_apply._genesis_g4_02_wrapped = True
    _genesis_g4_02_guarded_apply._genesis_g4_02_original = original
    _genesis_g4_02_guarded_apply.__wrapped__ = original
    return _genesis_g4_02_guarded_apply


def apply() -> tuple[str, str]:
    """Install guards on the two MoE Marlin method classes. Never raises."""
    global _APPLIED, _ORIGINAL_METHODS

    if not _env_enabled():
        return "skipped", (
            f"G4_02 disabled (set {_ENV_ENABLE}=1 to refuse non-aligned Marlin K "
            "before kernel raises an obscure error — see vllm#40354)"
        )

    if _APPLIED:
        return "applied", "G4_02 already installed (idempotent)"

    wrapped_count = 0

    # 1) compressed-tensors WNA16 Marlin (FP8 + AWQ paths share this in some builds)
    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
            compressed_tensors_moe_wna16_marlin as mod,
        )
        target_cls = getattr(mod, "CompressedTensorsMoEWNA16MarlinMethod", None)
        if target_cls is not None and not getattr(
            target_cls.apply_weights, "_genesis_g4_02_wrapped", False
        ):
            _ORIGINAL_METHODS["wna16_marlin"] = target_cls.apply_weights
            target_cls.apply_weights = _make_guarded_apply(target_cls.apply_weights)
            wrapped_count += 1
    except ImportError as e:
        log.debug("compressed_tensors_moe_wna16_marlin not importable: %s", e)

    # 2) AWQ Marlin MoE — separate class in some pin layouts
    try:
        from vllm.model_executor.layers.quantization import awq_marlin as awq_mod
        for cls_name in ("AWQMarlinMoEMethod", "AwqMarlinMoEMethod"):
            cls = getattr(awq_mod, cls_name, None)
            if cls is not None and not getattr(cls.apply, "_genesis_g4_02_wrapped", False):
                _ORIGINAL_METHODS[cls_name] = cls.apply
                cls.apply = _make_guarded_apply(cls.apply)
                wrapped_count += 1
                break
    except ImportError as e:
        log.debug("awq_marlin not importable: %s", e)

    if wrapped_count == 0:
        return "skipped", (
            "Could not locate any Marlin MoE method class to guard; pin may have "
            "renamed the path. G4_02 is no-op."
        )

    _APPLIED = True
    log.info(
        "[G4_02] installed: wrapped %d Marlin MoE method class(es) with K-dim "
        "precheck. Non-aligned K will now raise a clear error pre-kernel.",
        wrapped_count,
    )
    return "applied", (
        f"G4_02 installed: {wrapped_count} Marlin MoE method class(es) wrapped. "
        "Non-aligned K (e.g. Gemma 4 26B-A4B at TP=2 with K=352) will raise a "
        f"clear error. Override via {_ENV_DISABLE}=1."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED, _ORIGINAL_METHODS
    if not _APPLIED:
        return False
    reverted = False
    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
            compressed_tensors_moe_wna16_marlin as mod,
        )
        cls = getattr(mod, "CompressedTensorsMoEWNA16MarlinMethod", None)
        if cls is not None and "wna16_marlin" in _ORIGINAL_METHODS:
            cls.apply_weights = _ORIGINAL_METHODS["wna16_marlin"]  # type: ignore[assignment]
            reverted = True
    except ImportError:
        pass
    try:
        from vllm.model_executor.layers.quantization import awq_marlin as awq_mod
        for cls_name in ("AWQMarlinMoEMethod", "AwqMarlinMoEMethod"):
            cls = getattr(awq_mod, cls_name, None)
            if cls is not None and cls_name in _ORIGINAL_METHODS:
                cls.apply = _ORIGINAL_METHODS[cls_name]  # type: ignore[assignment]
                reverted = True
    except ImportError:
        pass
    if reverted:
        _APPLIED = False
        _ORIGINAL_METHODS.clear()
    return reverted


__all__ = ["GENESIS_G4_02_MARKER", "apply", "is_applied", "revert"]
