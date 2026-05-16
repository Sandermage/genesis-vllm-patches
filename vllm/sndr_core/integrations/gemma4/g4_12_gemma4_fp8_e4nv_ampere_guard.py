# SPDX-License-Identifier: Apache-2.0
"""G4_12 — refuse FP8 e4nv (no-sat) Gemma 4 checkpoints on Ampere SM 8.6.

================================================================
WHAT BREAKS WITHOUT THIS PATCH
================================================================

vllm-project/vllm#41014 (OPEN as of 2026-05-17, 7 comments): a subset
of FP8 Gemma 4 checkpoints are quantized with the **e4nv** (no-saturate)
format — distinct from the more common e4m3fn / e4m3fnuz. Ampere SM 8.6
hardware does **not** have native FP8 e4nv tensor-core instructions —
only e4m3 (with sat) and e5m2. vLLM's FP8 GEMM kernels assume e4m3fn
on Ampere and fail at the first forward pass with:

    RuntimeError: cublasLtMatmul failed with status CUBLAS_STATUS_NOT_SUPPORTED
                  (typeA=FP8_E4NV, typeB=FP8_E4NV not supported on this GPU)

The cuBLASLt error string is opaque — operators see it after a 30+
minute cold-boot + warmup and have no idea the *format* was wrong (not
the model, not the kernel choice). This patch fails fast at weight-load
time with a precise pointer to:

  * which checkpoint is at fault
  * what hardware would work (Hopper / Blackwell have native e4nv)
  * an alternative AWQ-4bit checkpoint that works on Ampere

================================================================
DETECTION
================================================================

We probe ``hf_quant_config.weight_quant.type`` (and equivalents in
the compressed-tensors schema) for the strings:
  * ``"float8_e4nv"``
  * ``"e4nv"``
  * ``"fp8_e4nv"``
  * ``"f8e4nv"``

If found AND model is Gemma 4 AND we're on Ampere SM 8.6 — refuse with
a structured error.

================================================================
SAFETY MODEL
================================================================

* default_on: True (cheap; fires only on the broken combo)
* env_flag: GENESIS_ENABLE_G4_12_GEMMA4_FP8_E4NV_GUARD
* override: GENESIS_DISABLE_G4_12_GUARD=1
* applies_to:
    - architecture: gemma4
    - quantization: FP8 e4nv
    - hardware: Ampere SM 8.6 (and earlier — Volta/Turing also lack
      native FP8, but those are off the supported matrix already)
* conflicts_with: []
* superseded_by: when upstream cuBLAS adds e4nv fallback on Ampere
  (effectively never — this is a HW limitation, not a SW bug)

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
References:
  * https://github.com/vllm-project/vllm/issues/41014 (OPEN, 7 comments)
  * NVIDIA cuBLAS docs: cublasLtMatmul FP8 type matrix
"""
from __future__ import annotations

import logging

from ._gemma4_detect import env_disable, env_truthy, is_ampere_sm86, is_gemma4_arch

log = logging.getLogger("genesis.gemma4.g4_12_fp8_e4nv_guard")

GENESIS_G4_12_MARKER = (
    "Genesis G4_12 gemma4 FP8 e4nv Ampere guard v1 "
    "(refuses unsupported FP8 type on consumer Ampere; closes vllm#41014)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_12_GEMMA4_FP8_E4NV_GUARD"
_ENV_DISABLE = "GENESIS_DISABLE_G4_12_GUARD"

_E4NV_MARKERS: tuple[str, ...] = (
    "float8_e4nv",
    "e4nv",
    "fp8_e4nv",
    "f8e4nv",
    "float8e4nv",
)

_APPLIED = False
_ORIGINAL_VERIFY = None


def _env_enabled() -> bool:
    return env_truthy(_ENV_ENABLE)


def _walk_quant_type(obj: object, depth: int = 0) -> str | None:
    """Walk a quant config object/dict tree looking for an e4nv marker."""
    if obj is None or depth > 6:
        return None
    for attr in ("type", "dtype", "quant_dtype", "weight_dtype", "format"):
        v = getattr(obj, attr, None)
        if isinstance(v, str):
            low = v.lower().replace("-", "_")
            for marker in _E4NV_MARKERS:
                if marker in low:
                    return v
    if isinstance(obj, dict):
        for k in ("type", "dtype", "quant_dtype", "weight_dtype", "format"):
            v = obj.get(k)
            if isinstance(v, str):
                low = v.lower().replace("-", "_")
                for marker in _E4NV_MARKERS:
                    if marker in low:
                        return v
        for v in obj.values():
            r = _walk_quant_type(v, depth + 1)
            if r is not None:
                return r
    if isinstance(obj, (list, tuple)):
        for item in obj:
            r = _walk_quant_type(item, depth + 1)
            if r is not None:
                return r
    # Also walk known config-object attrs
    for attr in ("weight_quant", "input_quant", "config_groups", "quant_config",
                 "compression_config", "quantization_config"):
        sub = getattr(obj, attr, None)
        if sub is not None and sub is not obj:
            r = _walk_quant_type(sub, depth + 1)
            if r is not None:
                return r
    return None


def _detect_e4nv(model_config) -> str | None:
    """Return the offending e4nv type string when present in model_config."""
    if model_config is None:
        return None
    hf_config = getattr(model_config, "hf_config", None)
    for src in (model_config, hf_config):
        if src is None:
            continue
        for attr in ("quantization_config", "compression_config", "quant_config"):
            qc = getattr(src, attr, None)
            if qc is None and isinstance(src, dict):
                qc = src.get(attr)
            if qc is None:
                continue
            r = _walk_quant_type(qc)
            if r is not None:
                return r
    # vLLM's quant_config on model_config (parsed form)
    qc = getattr(model_config, "quantization_config", None)
    if qc is not None:
        r = _walk_quant_type(qc)
        if r is not None:
            return r
    return None


_REFUSAL_TEMPLATE = (
    "[Genesis G4_12 REFUSAL] Gemma 4 FP8 e4nv checkpoint detected — "
    "consumer Ampere (RTX 3090/A5000 etc., SM 8.6) tensor cores do "
    "**not** support FP8 e4nv. Symptom upstream: CUBLAS_STATUS_NOT_SUPPORTED "
    "after a 30+ minute cold boot. Offending quant type: {qtype!r}.\n"
    "Working alternatives on Ampere:\n"
    "  * AWQ-4bit (cyankiwi/gemma-4-31B-it-AWQ-4bit) — validated on 2× A5000\n"
    "  * FP8 e4m3 (RedHatAI/gemma-4-31B-it-FP8-static) — natively supported\n"
    "  * GPTQ-INT4 — validated on club-3090 dual 3090 stack\n"
    "FP8 e4nv requires Hopper (H100/H200) or Blackwell (B100/RTX 5090) "
    "for native tensor-core support. Upstream vllm#41014 tracks the "
    "issue but cuBLAS adding Ampere e4nv fallback is unlikely.\n"
    "Override (NOT recommended): GENESIS_DISABLE_G4_12_GUARD=1 — will "
    "produce CUBLAS_STATUS_NOT_SUPPORTED on first forward."
)


def apply() -> tuple[str, str]:
    """Install guard via wrapping Gemma4Config.verify_and_update_config."""
    global _APPLIED, _ORIGINAL_VERIFY

    if not _env_enabled():
        return "skipped", (
            f"G4_12 disabled (set {_ENV_ENABLE}=1 to refuse FP8 e4nv "
            "Gemma 4 checkpoints on Ampere — closes vllm#41014)"
        )
    if env_disable(_ENV_DISABLE):
        return "skipped", (
            f"G4_12 explicitly disabled via {_ENV_DISABLE}=1 — operator "
            "bypassed the guard; FP8 e4nv WILL fail on Ampere with cuBLAS error"
        )

    if _APPLIED:
        return "applied", "G4_12 already installed (idempotent)"

    if not is_ampere_sm86():
        return "skipped", (
            "G4_12 not applicable: hardware is not Ampere SM 8.6 "
            "(FP8 e4nv natively supported on Hopper / Blackwell)"
        )

    try:
        from vllm.model_executor.models import gemma4 as _g4_mod
    except ImportError as e:
        return "skipped", f"vllm.model_executor.models.gemma4 not importable: {e}"

    target_cls = None
    for cls_name in ("Gemma4Config", "Gemma4TextConfig", "Gemma4ForConditionalGenerationConfig"):
        cls = getattr(_g4_mod, cls_name, None)
        if cls is not None and hasattr(cls, "verify_and_update_config"):
            target_cls = cls
            break
    if target_cls is None:
        return "skipped", (
            "No Gemma4Config-like class with verify_and_update_config found; "
            "G4_12 is no-op on this pin"
        )

    original = target_cls.verify_and_update_config
    if getattr(original, "_genesis_g4_12_wrapped", False):
        _APPLIED = True
        return "applied", "G4_12 already wrapped (idempotent)"
    _ORIGINAL_VERIFY = original

    def _genesis_g4_12_wrapped_verify(vllm_config):
        result = original(vllm_config)
        try:
            mc = getattr(vllm_config, "model_config", None)
            if mc is not None and is_gemma4_arch(mc):
                qtype = _detect_e4nv(mc)
                if qtype is not None:
                    raise RuntimeError(_REFUSAL_TEMPLATE.format(qtype=qtype))
        except RuntimeError:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("[G4_12] e4nv probe failed: %r; allowing boot to proceed", e)
        return result

    _genesis_g4_12_wrapped_verify._genesis_g4_12_wrapped = True
    _genesis_g4_12_wrapped_verify.__wrapped__ = original

    def _classmethod_shim(cls, vllm_config):
        return _genesis_g4_12_wrapped_verify(vllm_config)
    _classmethod_shim._genesis_g4_12_wrapped = True
    target_cls.verify_and_update_config = classmethod(_classmethod_shim)
    _APPLIED = True
    log.info(
        "[G4_12] installed: Gemma 4 FP8 e4nv checkpoints will be refused at "
        "config-verify time on Ampere SM 8.6."
    )
    return "applied", (
        "G4_12 installed: Gemma 4 + FP8 e4nv combination will be refused at "
        "config-verify with a clear pointer to working alternatives "
        "(AWQ-4bit, FP8 e4m3, GPTQ-INT4). Closes vllm#41014 user pain at "
        "config-time instead of 30 min into boot."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED, _ORIGINAL_VERIFY
    if not _APPLIED or _ORIGINAL_VERIFY is None:
        return False
    try:
        from vllm.model_executor.models import gemma4 as _g4_mod
        for cls_name in ("Gemma4Config", "Gemma4TextConfig", "Gemma4ForConditionalGenerationConfig"):
            cls = getattr(_g4_mod, cls_name, None)
            if cls is not None and getattr(
                cls.verify_and_update_config, "_genesis_g4_12_wrapped", False
            ):
                cls.verify_and_update_config = _ORIGINAL_VERIFY  # type: ignore[assignment]
                _APPLIED = False
                return True
    except ImportError:
        pass
    return False


__all__ = ["GENESIS_G4_12_MARKER", "apply", "is_applied", "revert"]
