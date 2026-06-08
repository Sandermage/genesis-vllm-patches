# SPDX-License-Identifier: Apache-2.0
"""Patch PN302 — Genesis Model Profile boot-time initializer.

Companion to PN296 (GPU arch profile init). Detects MODEL architecture,
quantization, topology, spec-decode method, and emits a structured log
line that operators can grep.

Auto-sets follow-on env vars based on model characteristics:
  - GENESIS_MODEL_FAMILY (qwen3 / gemma / llama / mamba)
  - GENESIS_MODEL_HOT_KERNELS (comma-separated kernel families)
  - GENESIS_MODEL_USES_GDN, _USES_MARLIN, _USES_TQ
  - GENESIS_MODEL_HAS_MTP

These env stamps let downstream patches make model-aware decisions
WITHOUT having to import sndr.engines.vllm.detection.model_profile (which
requires VllmConfig instance).

Composes with PN296 (arch profile) for unified decision matrix:
  Decision = f(arch_profile, model_profile, workload_signal)

Author: Sandermage (Sander) Barzov Aleksandr, 2026-06-05.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.wiring.pn302_model_profile_init")


_APPLIED = False
_LAZY_HOOK_INSTALLED = False


def _install_lazy_retry_hook() -> bool:
    """Monkey-patch vllm engine init points so PN302 retries when
    vllm_config becomes available. Covers v0 LLMEngine and v1 EngineCore.
    Returns True if at least one hook was successfully installed.
    """
    global _LAZY_HOOK_INSTALLED
    if _LAZY_HOOK_INSTALLED:
        return True

    installed_any = False

    def _wrap_method(cls, attr_name: str) -> bool:
        """Wrap a method that runs AFTER __init__ (e.g. step, execute_model).
        On first invocation, retries PN302.apply() once, then becomes a no-op
        passthrough on subsequent calls. Cheaper than per-call guard.
        """
        try:
            original = getattr(cls, attr_name)
            if getattr(original, "_genesis_pn302_wrapped", False):
                return True

            def wrapped(self, *args, **kwargs):
                if not wrapped._fired:
                    wrapped._fired = True
                    try:
                        cfg = getattr(self, "vllm_config", None)
                        if cfg is not None and not _APPLIED:
                            status, detail = apply(vllm_config=cfg)
                            log.warning(
                                "[PN302 lazy-retry] status=%s detail=%s",
                                status, detail,
                            )
                    except Exception as e:
                        log.warning(
                            "[PN302 lazy-retry] exception: %s", e
                        )
                return original(self, *args, **kwargs)

            wrapped._fired = False
            wrapped._genesis_pn302_wrapped = True
            setattr(cls, attr_name, wrapped)
            return True
        except Exception as e:
            log.debug("[PN302] hook install on %s failed: %s", cls, e)
            return False

    # v1 path: hook step() — runs AFTER apply_all bootstrap completes
    # in child process. self.vllm_config is available; one-shot guard.
    try:
        from vllm.v1.engine import core as _v1_core_mod
        for cls_name in (
            "EngineCore", "EngineCoreProc", "DPEngineCoreProc",
            "DPMoEEngineCoreActor", "EngineCoreActor",
        ):
            cls = getattr(_v1_core_mod, cls_name, None)
            if cls is None:
                continue
            # Wrap multiple candidate methods. One-shot guard ensures
            # PN302.apply() runs exactly once across them. _process_engine_step
            # is the actual per-iteration callable in v1 busy loop;
            # _handle_client_request runs per incoming request.
            for method_name in (
                "_process_engine_step", "_handle_client_request",
                "step", "execute_model", "run_busy_loop", "add_request",
            ):
                if hasattr(cls, method_name):
                    if _wrap_method(cls, method_name):
                        installed_any = True

    except Exception:
        pass

    # v0 path (legacy)
    try:
        from vllm.engine.llm_engine import LLMEngine as _V0Eng
        for method_name in ("step", "execute_model"):
            if hasattr(_V0Eng, method_name):
                if _wrap_method(_V0Eng, method_name):
                    installed_any = True
                break
    except Exception:
        pass

    _LAZY_HOOK_INSTALLED = installed_any
    return installed_any


def _set_env_stamps(profile) -> int:
    """Write diagnostic env stamps. Returns count of stamps written."""
    stamps = {
        "GENESIS_MODEL_ARCHITECTURE": profile.architecture,
        "GENESIS_MODEL_FAMILY": profile.family,
        "GENESIS_MODEL_NAME": profile.model_name,
        "GENESIS_MODEL_HOT_KERNELS": ",".join(profile.hot_kernels),
        "GENESIS_MODEL_USES_GDN": "1" if profile.uses_gdn else "0",
        "GENESIS_MODEL_USES_MARLIN": "1" if profile.uses_marlin else "0",
        "GENESIS_MODEL_USES_TQ": "1" if profile.uses_tq else "0",
        "GENESIS_MODEL_HAS_MTP": "1" if profile.has_mtp else "0",
        "GENESIS_MODEL_IS_FP8": "1" if profile.is_fp8_quant else "0",
        "GENESIS_MODEL_IS_MOE": "1" if profile.is_moe else "0",
        "GENESIS_MODEL_QUANT_METHOD": profile.quant_method,
        "GENESIS_MODEL_WEIGHT_DTYPE": profile.weight_dtype,
        "GENESIS_MODEL_KV_CACHE_DTYPE": profile.kv_cache_dtype,
        "GENESIS_MODEL_SPEC_METHOD": str(profile.spec_method or "none"),
        "GENESIS_MODEL_SPEC_K": str(profile.spec_K),
        "GENESIS_MODEL_TP_SIZE": str(profile.tensor_parallel_size),
    }
    for k, v in stamps.items():
        os.environ[k] = str(v)
    return len(stamps)


def apply(vllm_config=None) -> tuple[str, str]:
    """Apply PN302 — model profile detection + env stamps.

    Args:
        vllm_config: Optional VllmConfig. If None, looks up via vllm's
                     get_current_vllm_config() helper.
    """
    global _APPLIED

    if os.environ.get(
        "GENESIS_ENABLE_PN302_MODEL_PROFILE_INIT", ""
    ).lower() not in ("1", "true", "yes", "on"):
        return "skipped", (
            "PN302 default OFF — set GENESIS_ENABLE_PN302_MODEL_PROFILE_INIT=1. "
            "Detects model architecture / quantization / topology and emits "
            "GENESIS_MODEL_* env stamps for downstream patches."
        )

    try:
        from sndr.engines.vllm.detection.model_profile import get_model_profile
    except Exception as e:
        return "failed", f"model_profile module import failed: {e}"

    # Try to obtain vllm_config from current context
    if vllm_config is None:
        try:
            from vllm.config import get_current_vllm_config
            vllm_config = get_current_vllm_config()
        except Exception:
            vllm_config = None

        # Boot-time fallback: install lazy retry hook on engine init,
        # so detection runs the moment vllm_config exists.
        if vllm_config is None:
            installed = _install_lazy_retry_hook()
            return "skipped", (
                f"vllm_config not available at apply time — "
                f"lazy retry hook installed={installed} on engine init"
            )

    profile = get_model_profile(vllm_config)
    if profile is None:
        return "skipped", "model profile detection returned None"

    n_stamps = _set_env_stamps(profile)

    # Combined info line with arch (if available) for unified visibility
    try:
        from sndr.detection.gpu_arch_profile import (
            get_gpu_arch_profile,
        )
        arch = get_gpu_arch_profile()
        if arch is not None:
            log.warning(
                "[Genesis UNIFIED] arch=%s SM=%s | model=%s family=%s | "
                "quant=%s kv=%s | topology: hybrid=%s moe=%s | "
                "spec=%s K=%d | hot_kernels=%s",
                arch.device_name, arch.sm_string,
                profile.architecture, profile.family,
                profile.quant_method, profile.kv_cache_dtype,
                profile.is_hybrid, profile.is_moe,
                profile.spec_method, profile.spec_K,
                list(profile.hot_kernels),
            )
    except Exception:
        pass

    _APPLIED = True
    return "applied", (
        f"PN302 installed: model={profile.architecture} family={profile.family} "
        f"hot_kernels={list(profile.hot_kernels)}. {n_stamps} env stamps written."
    )


def is_applied() -> bool:
    return _APPLIED
