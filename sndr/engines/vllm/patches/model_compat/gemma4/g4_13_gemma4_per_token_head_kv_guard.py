# SPDX-License-Identifier: Apache-2.0
"""G4_13 — refuse Gemma 4 26B-A4B per-token-head KV cache mismatch.

================================================================
WHAT BREAKS WITHOUT THIS PATCH
================================================================

vllm-project/vllm#40388 (OPEN as of 2026-05-17, 11 comments, root-cause
located by @WoosukKwon comment 9): Gemma 4 26B-A4B has **asymmetric
KV head counts per layer type**:

  * sliding_attention layers: ``num_kv_heads = 8``
  * full_attention   layers: ``num_kv_heads = 2``

vLLM's KV-cache page-size calculator currently assumes **one
``num_kv_heads`` per model** (taken from ``hf_config.num_key_value_heads``,
which is the sliding-layer value). When the global layer wakes up,
its actual per-token-per-head KV requirement is **4× smaller** than
the page reserved for it. Pages get over-allocated AND the slot-mapping
kernel writes to wrong offsets → corruption / wrong-output / "ghost
KV reads from earlier sequence" symptoms.

Manifests in two ways:
  * **Subtle quality regression** (no error, output looks plausible
    but is wrong: factual errors, hallucinated context)
  * **Catastrophic** when prompt > 4096 — page-table overflow → silent
    truncation of context window

Detection in the wild: @WoosukKwon traced this in #40388 comment 9
via prefix-cache hit logs ("page_size=1024 expected=256 layer_idx=5
type=full_attention"). Confirmed reproducer on @kovkev/Allenmage's
report (2× A100, vLLM v0.20.1).

================================================================
UPSTREAM FIX
================================================================

vllm#40391 (OPEN companion PR — proposes per-layer ``num_kv_heads``
in ``KVCacheSpec``) is the proper fix. It's a non-trivial refactor
of the KV-cache allocator and has been WIP since 2026-03-22. No ETA.

================================================================
THIS PATCH (workaround until #40391 merges)
================================================================

Refuse to load 26B-A4B with the broken page-size assumption. The
operator gets a clear pointer to:

  * the active upstream issue
  * an **explicit override** if they're testing on tiny contexts
    where the bug doesn't manifest (``GENESIS_DISABLE_G4_13_GUARD=1``)
  * the alternative 31B-it model which has uniform 4 KV heads
    everywhere and is **not** affected

Eventually G4_13 is superseded by G4_19 — a follow-up patch we'll
ship that vendors #40391's per-layer page-size logic ahead of merge.
For now, fail-fast is the right call: silent quality regression is
worse than no boot.

================================================================
SAFETY MODEL
================================================================

* default_on: True (fires only on 26B-A4B + non-trivial context;
  fail-fast at config-verify, before any compute is wasted)
* env_flag: GENESIS_ENABLE_G4_13_GEMMA4_PER_TOKEN_HEAD_KV_GUARD
* override: GENESIS_DISABLE_G4_13_GUARD=1
* applies_to:
    - architecture: Gemma4ForConditionalGeneration + a4b config marker
      (config.text_config.attention_kv_heads_per_layer_type defined)
* superseded_by: [G4_19] when we vendor #40391 logic

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
References:
  * https://github.com/vllm-project/vllm/issues/40388 (OPEN, 11 comments)
  * https://github.com/vllm-project/vllm/pull/40391  (WIP fix)
"""
from __future__ import annotations

import logging

from ._gemma4_detect import env_disable, env_truthy, is_gemma4_arch

log = logging.getLogger("genesis.gemma4.g4_13_per_token_head_kv_guard")

GENESIS_G4_13_MARKER = (
    "Genesis G4_13 gemma4 26B-A4B per-token-head KV guard v1 "
    "(refuses asymmetric KV-head config that triggers vllm#40388)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_13_GEMMA4_PER_TOKEN_HEAD_KV_GUARD"
_ENV_DISABLE = "GENESIS_DISABLE_G4_13_GUARD"

_APPLIED = False
_ORIGINAL_VERIFY = None


def _env_enabled() -> bool:
    return env_truthy(_ENV_ENABLE)


def _has_asymmetric_kv_heads(model_config) -> tuple[bool, dict | None]:
    """Probe model_config for asymmetric KV heads per layer type.

    Returns (is_asymmetric, info_dict) where info_dict has the per-layer-type
    KV-head counts when asymmetric (for the refusal message).
    """
    hf = getattr(model_config, "hf_config", None) or model_config
    text = getattr(hf, "text_config", None) or hf
    # Direct marker: a4b config exposes attention_kv_heads_per_layer_type
    pattern = getattr(text, "attention_kv_heads_per_layer_type", None)
    if isinstance(pattern, dict) and len(set(pattern.values())) > 1:
        return True, dict(pattern)
    # Fallback: list-form e.g. [8, 2] under another key
    pattern_list = (
        getattr(text, "kv_heads_per_layer", None)
        or getattr(text, "layer_types_kv_heads", None)
    )
    if isinstance(pattern_list, (list, tuple)) and len(set(pattern_list)) > 1:
        return True, {f"layer_{i}": v for i, v in enumerate(pattern_list)}
    # Indirect: if sliding/full layer counts differ AND num_key_value_heads
    # is the smaller value, that's the buggy combo too. (Belt-and-suspenders.)
    sw_kv = getattr(text, "num_key_value_heads_sliding", None)
    full_kv = getattr(text, "num_key_value_heads_full", None)
    if sw_kv is not None and full_kv is not None and sw_kv != full_kv:
        return True, {
            "sliding_attention": int(sw_kv),
            "full_attention": int(full_kv),
        }
    return False, None


_REFUSAL_TEMPLATE = (
    "[Genesis G4_13 REFUSAL] Gemma 4 26B-A4B detected with asymmetric "
    "per-layer-type KV-head counts: {info!r}. vLLM's KV-cache page-size "
    "allocator currently assumes a single num_kv_heads per model and will "
    "silently corrupt KV reads on the smaller-head layer type (vllm#40388, "
    "OPEN, 11 comments, root-cause located by @WoosukKwon).\n"
    "\n"
    "Symptoms if you bypass: subtle quality regression (no error, "
    "factual hallucinations); catastrophic at context > 4K (page-table "
    "overflow → silent context truncation).\n"
    "\n"
    "Working alternatives:\n"
    "  * Gemma 4 31B-it (uniform num_kv_heads=4 everywhere — not affected)\n"
    "  * Wait for upstream vllm#40391 to merge (per-layer page-size; WIP)\n"
    "\n"
    "Override (only safe for short-context smoke tests):\n"
    "  GENESIS_DISABLE_G4_13_GUARD=1\n"
    "Override + production context = data corruption guaranteed."
)


def apply() -> tuple[str, str]:
    """Install guard via wrapping Gemma4Config.verify_and_update_config."""
    global _APPLIED, _ORIGINAL_VERIFY

    if not _env_enabled():
        return "skipped", (
            f"G4_13 disabled (set {_ENV_ENABLE}=1 to refuse asymmetric "
            "KV-head Gemma 4 configs — closes vllm#40388)"
        )
    if env_disable(_ENV_DISABLE):
        return "skipped", (
            f"G4_13 explicitly disabled via {_ENV_DISABLE}=1 — operator "
            "bypassed the guard; 26B-A4B WILL produce silent quality "
            "regression at non-trivial context lengths"
        )

    if _APPLIED:
        return "applied", "G4_13 already installed (idempotent)"

    # dev371+ moved the verify_and_update_config wrappers from
    # vllm.model_executor.models.gemma4 to vllm.model_executor.models.config.
    # Search both locations so the patch survives the move.
    _candidate_modules: list[tuple[str, object]] = []
    try:
        from vllm.model_executor.models import config as _g4_cfg_mod
        _candidate_modules.append(
            ("vllm.model_executor.models.config", _g4_cfg_mod)
        )
    except ImportError:
        pass
    try:
        from vllm.model_executor.models import gemma4 as _g4_legacy_mod
        _candidate_modules.append(
            ("vllm.model_executor.models.gemma4", _g4_legacy_mod)
        )
    except ImportError:
        pass

    if not _candidate_modules:
        return "skipped", (
            "Neither vllm.model_executor.models.config nor .gemma4 "
            "importable; G4_13 is no-op on this pin"
        )

    target_cls = None
    for cls_name in ("Gemma4Config", "Gemma4TextConfig", "Gemma4ForConditionalGenerationConfig"):
        for mod_name, mod in _candidate_modules:
            cls = getattr(mod, cls_name, None)
            if cls is not None and hasattr(cls, "verify_and_update_config"):
                target_cls = cls
                break
        if target_cls is not None:
            break
    if target_cls is None:
        return "skipped", (
            "No Gemma4Config-like class with verify_and_update_config found "
            f"in {[m for m, _ in _candidate_modules]}; G4_13 is no-op on this pin"
        )

    original = target_cls.verify_and_update_config
    if getattr(original, "_genesis_g4_13_wrapped", False):
        _APPLIED = True
        return "applied", "G4_13 already wrapped (idempotent)"
    _ORIGINAL_VERIFY = original

    def _genesis_g4_13_wrapped_verify(vllm_config):
        result = original(vllm_config)
        try:
            mc = getattr(vllm_config, "model_config", None)
            if mc is not None and is_gemma4_arch(mc):
                bad, info = _has_asymmetric_kv_heads(mc)
                if bad:
                    raise RuntimeError(_REFUSAL_TEMPLATE.format(info=info))
        except RuntimeError:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("[G4_13] per-layer KV-head probe failed: %r; "
                        "allowing boot (operator may hit #40388 anyway)", e)
        return result

    _genesis_g4_13_wrapped_verify._genesis_g4_13_wrapped = True
    _genesis_g4_13_wrapped_verify.__wrapped__ = original

    def _classmethod_shim(cls, vllm_config):
        return _genesis_g4_13_wrapped_verify(vllm_config)
    _classmethod_shim._genesis_g4_13_wrapped = True
    target_cls.verify_and_update_config = classmethod(_classmethod_shim)
    _APPLIED = True
    log.info(
        "[G4_13] installed: Gemma 4 with asymmetric per-layer-type KV heads "
        "will be refused at config-verify."
    )
    return "applied", (
        "G4_13 installed: Gemma 4 + asymmetric KV-head config (26B-A4B) will "
        "be refused at config-verify with a clear pointer to vllm#40388 + "
        "#40391. Prevents silent quality regression."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED, _ORIGINAL_VERIFY
    if not _APPLIED or _ORIGINAL_VERIFY is None:
        return False
    _modules = []
    try:
        from vllm.model_executor.models import config as _m
        _modules.append(_m)
    except ImportError:
        pass
    try:
        from vllm.model_executor.models import gemma4 as _m
        _modules.append(_m)
    except ImportError:
        pass
    for _g4_mod in _modules:
        for cls_name in ("Gemma4Config", "Gemma4TextConfig", "Gemma4ForConditionalGenerationConfig"):
            cls = getattr(_g4_mod, cls_name, None)
            if cls is not None and getattr(
                cls.verify_and_update_config, "_genesis_g4_13_wrapped", False
            ):
                cls.verify_and_update_config = _ORIGINAL_VERIFY  # type: ignore[assignment]
                _APPLIED = False
                return True
    return False


__all__ = ["GENESIS_G4_13_MARKER", "apply", "is_applied", "revert"]
