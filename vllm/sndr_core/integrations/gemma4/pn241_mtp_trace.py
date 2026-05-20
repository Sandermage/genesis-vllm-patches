# SPDX-License-Identifier: Apache-2.0
"""PN241 — Finite/norm trace at SpecDecodeBaseProposer.propose boundary.

Hooks the Python orchestration layer above the torch.compile fullgraph
boundary. Instrumenting inside Gemma4MTPAttention.forward is unsafe:
vLLM v1 compiles MTP forward with fullgraph=True; any disabled
function inside triggers gb0098/gb0099. Hooking propose() side-steps
the compile boundary.

Purpose
-------
Bracket each call to `SpecDecodeBaseProposer.propose()` with
finite/norm checks on:
  * target_hidden_states (input to drafter)
  * returned draft_token_ids
Logged to /tmp/genesis_pn241_mtp_trace.log inside container.

Hypothesis under test:
  If target_hidden_states are finite on entry but draft_token_ids show
  signs of degenerate sampling (all same token, or token IDs at end
  of vocab), the drafter's attention is corrupting state — confirming
  the kv_dummy/KV-sharing bug.

Activation
----------
GENESIS_ENABLE_PN241_MTP_TRACE=1

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("genesis.gemma4.pn241_mtp_trace")

_ENV = "GENESIS_ENABLE_PN241_MTP_TRACE"
_LOG_PATH = "/tmp/genesis_pn241_mtp_trace.log"
_CALL_IDX = [0]
_APPLIED = False
_ORIGINAL_PROPOSE = None


def _on() -> bool:
    return os.environ.get(_ENV, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _tensor_stats(name: str, t):
    """Return (name, finite_str, norm_str). Forces GPU sync; only call
    OUTSIDE cudagraph capture (this runs in Python orchestration so
    that's fine here)."""
    if t is None:
        return f"{name}=None"
    try:
        if not isinstance(t, torch.Tensor):
            return f"{name}.type={type(t).__name__}"
        if t.numel() == 0:
            return f"{name}.empty=True shape={tuple(t.shape)}"
        finite = bool(torch.isfinite(t).all().item())
        norm = float(t.detach().float().norm().item())
        extra = ""
        if t.dtype in (torch.int32, torch.int64):
            try:
                tmin = int(t.min().item())
                tmax = int(t.max().item())
                extra = f" min={tmin} max={tmax}"
            except Exception:
                pass
        return f"{name}.shape={tuple(t.shape)} finite={finite} norm={norm:.4e}{extra}"
    except Exception as e:
        return f"{name}.err={e!r}"


def apply() -> tuple[str, str]:
    """Wrap SpecDecodeBaseProposer.propose with finite/norm tracing."""
    global _APPLIED, _ORIGINAL_PROPOSE

    if _APPLIED:
        return "applied", "PN241 already installed (idempotent)"
    if not _on():
        return "skipped", (
            f"PN241 disabled (set {_ENV}=1 to enable proposer-boundary trace)"
        )

    try:
        from vllm.v1.spec_decode.llm_base_proposer import (
            SpecDecodeBaseProposer,
        )
    except ImportError as e:
        return "skipped", f"llm_base_proposer not importable: {e}"

    original = SpecDecodeBaseProposer.propose
    if getattr(original, "_genesis_pn241_wrapped", False):
        _APPLIED = True
        return "applied", "PN241 already wrapped (idempotent)"
    _ORIGINAL_PROPOSE = original

    def wrapped(self, *args, **kwargs):
        _CALL_IDX[0] += 1
        call_idx = _CALL_IDX[0]
        # propose() signature has target_token_ids, target_positions,
        # target_hidden_states, next_token_ids as first positional args.
        target_token_ids = args[0] if len(args) > 0 else kwargs.get("target_token_ids")
        target_positions = args[1] if len(args) > 1 else kwargs.get("target_positions")
        target_hidden = args[2] if len(args) > 2 else kwargs.get("target_hidden_states")
        next_token_ids = args[3] if len(args) > 3 else kwargs.get("next_token_ids")

        # PN246: one-shot mapping diagnostic on first propose() call.
        # Captures actual kv_sharing_target_layer_name wiring (real source
        # of truth for Gemma 4 MTP, vs. hf_text_config which is YOCO-style
        # metadata that G4_60h relies on but Gemma 4 MTP doesn't populate).
        if call_idx == 1:
            try:
                with open(_LOG_PATH, "a") as f:
                    f.write("\n[PN246 mapping trace — first propose call]\n")
                    # Try multiple paths to find target model
                    target_model = None
                    runner = getattr(self, "runner", None)
                    if runner is not None:
                        target_model = getattr(runner, "model", None)
                    if target_model is None:
                        target_model = getattr(self, "target_model", None)
                    # Drafter model (this proposer's self.model)
                    drafter_model = getattr(self, "model", None)

                    # Hf text config (for G4_60h source of truth check)
                    vllm_config = getattr(self, "vllm_config", None)
                    if vllm_config is not None:
                        mc = getattr(vllm_config, "model_config", None)
                        if mc is not None:
                            htc = getattr(mc, "hf_text_config", None)
                            f.write(
                                f"[PN246] hf_text_config.num_hidden_layers={getattr(htc, 'num_hidden_layers', '?')!r}\n"
                                f"[PN246] hf_text_config.num_kv_shared_layers={getattr(htc, 'num_kv_shared_layers', '?')!r}\n"
                                f"[PN246] hf_text_config.layer_types={getattr(htc, 'layer_types', None)!r}\n"
                            )
                        cc = getattr(vllm_config, "cache_config", None)
                        if cc is not None:
                            f.write(
                                f"[PN246] cache_config.cache_dtype={getattr(cc, 'cache_dtype', '?')!r}\n"
                                f"[PN246] cache_config.kv_cache_dtype_skip_layers={getattr(cc, 'kv_cache_dtype_skip_layers', '?')!r}\n"
                            )
                        sc = getattr(vllm_config, "speculative_config", None)
                        if sc is not None:
                            f.write(
                                f"[PN246] speculative_config.method={getattr(sc, 'method', '?')!r}\n"
                                f"[PN246] speculative_config.num_speculative_tokens={getattr(sc, 'num_speculative_tokens', '?')!r}\n"
                            )

                    # Walk drafter modules — find kv_sharing_target_layer_name
                    if drafter_model is not None and hasattr(drafter_model, "named_modules"):
                        f.write(f"[PN246] drafter_model class={type(drafter_model).__name__}\n")
                        count = 0
                        for name, module in drafter_model.named_modules():
                            tgt = getattr(module, "kv_sharing_target_layer_name", None)
                            if tgt:
                                f.write(
                                    f"[PN246] drafter kv_sharing: module={name} "
                                    f"target_layer_name={tgt!r} "
                                    f"class={type(module).__name__}\n"
                                )
                                count += 1
                        f.write(f"[PN246] drafter kv_sharing modules found: {count}\n")

                    # Walk target modules — find their layer_names + backends
                    if target_model is not None and hasattr(target_model, "named_modules"):
                        f.write(f"[PN246] target_model class={type(target_model).__name__}\n")
                        attn_count = 0
                        for name, module in target_model.named_modules():
                            if module.__class__.__name__ == "Attention":
                                layer_name = getattr(module, "layer_name", None)
                                impl = getattr(module, "impl", None)
                                impl_class = type(impl).__name__ if impl is not None else "?"
                                kv_dtype = getattr(module, "kv_cache_dtype", "?")
                                attn_count += 1
                                if attn_count <= 50:
                                    f.write(
                                        f"[PN246] target Attention: name={name} "
                                        f"layer_name={layer_name!r} impl={impl_class} "
                                        f"kv_cache_dtype={kv_dtype!r}\n"
                                    )
                        f.write(f"[PN246] target Attention modules found: {attn_count}\n")
                    f.write("[PN246] mapping trace end\n\n")
            except Exception as e:
                try:
                    with open(_LOG_PATH, "a") as f:
                        f.write(f"[PN246 ERROR] {type(e).__name__}: {e}\n")
                except Exception:
                    pass

        try:
            with open(_LOG_PATH, "a") as f:
                f.write(
                    f"[propose ENTER call={call_idx}] "
                    f"{_tensor_stats('target_tok', target_token_ids)} "
                    f"{_tensor_stats('target_pos', target_positions)} "
                    f"{_tensor_stats('target_hidden', target_hidden)} "
                    f"{_tensor_stats('next_tok', next_token_ids)}\n"
                )
        except Exception:
            pass

        result = original(self, *args, **kwargs)

        try:
            with open(_LOG_PATH, "a") as f:
                f.write(
                    f"[propose EXIT  call={call_idx}] "
                    f"{_tensor_stats('draft_tok', result)}\n"
                )
        except Exception:
            pass
        return result

    wrapped._genesis_pn241_wrapped = True  # type: ignore[attr-defined]
    SpecDecodeBaseProposer.propose = wrapped  # type: ignore[method-assign]
    _APPLIED = True
    log.info(
        "[PN241] SpecDecodeBaseProposer.propose wrapped — finite/norm "
        "trace at proposer boundary to %s (compile-safe; runs at Python "
        "orchestration layer above compile)",
        _LOG_PATH,
    )
    return "applied", "PN241 proposer trace installed"


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED, _ORIGINAL_PROPOSE
    if not _APPLIED or _ORIGINAL_PROPOSE is None:
        return False
    try:
        from vllm.v1.spec_decode.llm_base_proposer import (
            SpecDecodeBaseProposer,
        )
        SpecDecodeBaseProposer.propose = _ORIGINAL_PROPOSE  # type: ignore[method-assign]
    except Exception:
        return False
    _APPLIED = False
    _ORIGINAL_PROPOSE = None
    return True
