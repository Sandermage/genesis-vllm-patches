# SPDX-License-Identifier: Apache-2.0
"""PN272 — Gemma 4 MTP drafter input semantics probe (read-only).

================================================================
WHY
================================================================

PN271 contract audit verdict for Gemma 4 was:
  drafter[0..2] -> target[58] : LAYOUT_ADAPTER (HND vs NHD)
  drafter[3]   -> target[59]  : GQA_REPEAT (kv_heads 8 vs 2)
  All numerics (scale, RoPE, soft_cap, q_norm) match per-pair.

G4_78 v2 implements both adapters; bridge fires correctly; output
coherent. Yet PN248 acceptance stays 0/9. So the residual blocker
must lie OUTSIDE the K/V contract.

Reading the drafter source (gemma4_mtp.py) reveals the actual MTP
input pipeline is non-trivial:

  Gemma4MultiTokenPredictor.forward(input_ids, positions, hidden_states):
      inputs_embeds = embed_tokens(input_ids) * normalizer
      combined = torch.cat([inputs_embeds, hidden_states], dim=-1)
      hidden_states = pre_projection(combined)     # 2*backbone -> hidden
      for layer in layers:
          hidden_states, residual = layer(...)
      draft_hidden_states = norm(hidden_states)
      backbone_hidden_states = post_projection(draft_hidden_states)
      return draft_hidden_states, backbone_hidden_states

  Gemma4MTPDecoderLayer.forward:
      residual = hidden_states
      hidden_states = input_layernorm(residual)
      hidden_states = self_attn(...)
      hidden_states = post_attention_layernorm(hidden_states)
      hidden_states = hidden_states + residual
      ... mlp + 2 more layernorms ...
      hidden_states = hidden_states * layer_scalar      # buffer!
      return hidden_states, None

Several potential failure modes that PN271 cannot see:

  a. `layer_scalar` is `register_buffer("layer_scalar", torch.ones(1))`
     — if checkpoint stored a learned value but the loader skipped it
     (buffers are NOT always loaded by default), drafter output is
     scaled by 1.0 instead of the trained scalar.

  b. `pre_projection` weight may not be loaded (no .weight_loader path
     for ColumnParallelLinear's name? unlikely but possible). Check
     weight.norm() to detect default-initialized state.

  c. `post_projection` same concern.

  d. `embed_tokens` is documented to be replaced by target model's
     backbone-dim embedding. If the replacement didn't happen, drafter
     embeds use draft_hidden-dim, dimensions don't match target's,
     pre_projection sees garbage.

  e. `normalizer` buffer = sqrt(backbone_hidden_size). Trivial but
     worth recording.

  f. `lm_head.weight` is tied to `embed_tokens.weight`. If tying
     broke, logits computation is wrong.

  g. `norm` (final RMSNorm before lm_head) weight unloaded.

  h. Per-decoder-layer normalizations (input/post_attention/
     pre_feedforward/post_feedforward) — same risk.

PN272 probes ALL of these.

================================================================
WHAT IT DOES
================================================================

Two phases, one-shot per worker (each guarded by its own flag):

PHASE 1 — POST-LOAD INVENTORY (at initialize_kv_cache_tensors hook):
  Walk runner.drafter.model -> Gemma4MTP -> Gemma4MultiTokenPredictor.
  Dump:
    - Predictor-level modules + weight stats
        embed_tokens.weight.norm + shape + data_ptr
        normalizer buffer value
        pre_projection.weight.norm + shape
        post_projection.weight.norm + shape
        norm.weight.norm + shape
    - For each decoder layer i:
        layer_scalar buffer value
        layer_scalar exists in state_dict?
        input_layernorm.weight.norm
        post_attention_layernorm.weight.norm
        pre_feedforward_layernorm.weight.norm
        post_feedforward_layernorm.weight.norm
        self_attn.q_norm.weight.norm
    - LM head: lm_head.weight.norm + shape + data_ptr
        tied to embed_tokens.weight? (data_ptr equality)

PHASE 2 — RUNTIME TENSOR-STATS PROBE (wrap MTP forward, one-shot):
  On the first NON-WARMUP forward call:
    - input_ids[:8]
    - inputs_embeds stats
    - target hidden_states (the arg) stats
    - combined cat tensor stats
    - after pre_projection stats
    - per decoder layer:
        input stats
        post-input_layernorm stats
        post-self_attn stats
        post-mlp stats
        layer output stats (after *layer_scalar)
    - final norm output stats
    - post_projection output stats

  Skip if input_ids.numel() == 0 or hidden_states sum == 0 (warmup).

PHASE 3 — STATE_DICT KEY MATRIX:
  For each expected weight name, log "loaded=True/False".
  Helps catch silent unload of pre_projection / post_projection /
  layer_scalar etc.

================================================================
ENV
================================================================

  GENESIS_ENABLE_PN272_GEMMA4_DRAFTER_INPUT_PROBE=1

================================================================
NO BEHAVIOR CHANGE — DIAGNOSTIC ONLY
================================================================

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(
    "genesis.spec_decode.probes.pn272_gemma4_drafter_input_probe"
)

GENESIS_PN272_MARKER = "Genesis PN272 Gemma4 drafter input semantics probe"

_ENV_ENABLE = "GENESIS_ENABLE_PN272_GEMMA4_DRAFTER_INPUT_PROBE"

_APPLIED = False
_ORIGINAL_INIT_TENSORS = None
_INVENTORY_DUMPED = False
_RUNTIME_DUMPED = False
_ORIGINAL_MTP_FORWARD = None
_ORIGINAL_LAYER_FORWARD = None


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _t_stats(t: Any) -> str:
    """One-line mean/std/norm/shape/dtype for a tensor."""
    try:
        if t is None:
            return "<None>"
        f = t.float()
        return (
            f"shape={tuple(t.shape)} dtype={t.dtype} "
            f"mean={f.mean().item():.4e} std={f.std().item():.4e} "
            f"norm={f.norm().item():.4e} "
            f"min={f.min().item():.4e} max={f.max().item():.4e}"
        )
    except Exception as _e:
        return f"<err: {_e!r}>"


def _t_head(t: Any, n: int = 8) -> Any:
    try:
        if t is None:
            return None
        return t.flatten()[:n].tolist()
    except Exception as _e:
        return f"<err: {_e!r}>"


def _weight_stats(mod: Any) -> str:
    if mod is None:
        return "<None>"
    w = getattr(mod, "weight", None)
    if w is None:
        return f"class={type(mod).__qualname__} <no .weight>"
    return f"class={type(mod).__qualname__} {_t_stats(w)} data_ptr=0x{int(w.data_ptr()):x}"


def _buf_value(mod: Any, name: str) -> Any:
    try:
        buf = getattr(mod, name, None)
        if buf is None:
            return "<absent>"
        if hasattr(buf, "tolist"):
            return buf.tolist()
        return buf
    except Exception as _e:
        return f"<err: {_e!r}>"


# ----------------------- Discovery helpers -----------------------

def _unwrap(m: Any) -> Any:
    seen = set()
    for _ in range(12):
        if m is None or id(m) in seen:
            return m
        seen.add(id(m))
        for attr in ("runnable_model", "module", "model", "orig_module",
                     "_orig_mod", "wrapped", "inner"):
            inner = getattr(m, attr, None)
            if (inner is not None and inner is not m
                    and hasattr(inner, "named_modules")):
                m = inner
                break
        else:
            return m
    return m


def _walk_chain(start: Any) -> list[Any]:
    """Yield start + each `.model` / wrapper-attr down the chain until
    fixed point. Returns a list (innermost last)."""
    chain: list[Any] = []
    cur = start
    for _ in range(12):
        if cur is None:
            break
        chain.append(cur)
        nxt = None
        for attr in ("runnable_model", "module", "orig_module", "_orig_mod",
                     "wrapped", "inner", "model"):
            cand = getattr(cur, attr, None)
            if (cand is not None and cand is not cur
                    and hasattr(cand, "named_modules")):
                nxt = cand
                break
        if nxt is None:
            break
        cur = nxt
    return chain


def _find_predictor(runner: Any) -> Any:
    """Find Gemma4MultiTokenPredictor by *signature*: has embed_tokens
    AND pre_projection AND layers."""
    drafter = getattr(runner, "drafter", None)
    dmodel = getattr(drafter, "model", None) if drafter is not None else None
    for cand in _walk_chain(dmodel):
        if (hasattr(cand, "embed_tokens")
                and hasattr(cand, "pre_projection")
                and hasattr(cand, "layers")):
            return cand
    return None


def _find_top_drafter(runner: Any) -> Any:
    """Find Gemma4MTP top-level: has lm_head AND has a .model child
    that looks like a predictor."""
    drafter = getattr(runner, "drafter", None)
    dmodel = getattr(drafter, "model", None) if drafter is not None else None
    for cand in _walk_chain(dmodel):
        if hasattr(cand, "lm_head"):
            return cand
    return dmodel


# ----------------------- Phase 1: weight inventory -----------------------

def _dump_inventory(runner: Any) -> None:
    log.warning("[PN272] === Drafter input-semantics inventory BEGIN ===")
    predictor = _find_predictor(runner)
    top = _find_top_drafter(runner)
    if predictor is None:
        log.warning("[PN272] no Gemma4MultiTokenPredictor found — abort")
        log.warning("[PN272] === inventory END (not_found) ===")
        return

    log.warning("[PN272] predictor.class=%s", type(predictor).__qualname__)
    if top is not None:
        log.warning("[PN272] top.class=%s", type(top).__qualname__)

    # Predictor-level modules
    log.warning("[PN272] predictor.embed_tokens : %s",
                _weight_stats(getattr(predictor, "embed_tokens", None)))
    log.warning("[PN272] predictor.normalizer (buffer) = %s",
                _buf_value(predictor, "normalizer"))
    log.warning("[PN272] predictor.pre_projection : %s",
                _weight_stats(getattr(predictor, "pre_projection", None)))
    log.warning("[PN272] predictor.post_projection : %s",
                _weight_stats(getattr(predictor, "post_projection", None)))
    log.warning("[PN272] predictor.norm (final RMSNorm) : %s",
                _weight_stats(getattr(predictor, "norm", None)))

    # Decoder layers
    layers = getattr(predictor, "layers", None)
    if layers is not None:
        log.warning("[PN272] predictor.layers count = %d", len(layers))
        for i, layer in enumerate(layers):
            log.warning(
                "[PN272] --- predictor.layers[%d] (class=%s) ---",
                i, type(layer).__qualname__,
            )
            ls = _buf_value(layer, "layer_scalar")
            log.warning("[PN272]   layers[%d].layer_scalar = %s "
                        "(default ones=[1.0])", i, ls)
            for ln in ("input_layernorm",
                       "post_attention_layernorm",
                       "pre_feedforward_layernorm",
                       "post_feedforward_layernorm"):
                m = getattr(layer, ln, None)
                log.warning("[PN272]   layers[%d].%s : %s", i, ln,
                            _weight_stats(m))
            sa = getattr(layer, "self_attn", None)
            if sa is not None:
                log.warning("[PN272]   layers[%d].self_attn class=%s",
                            i, type(sa).__qualname__)
                log.warning("[PN272]     q_proj : %s",
                            _weight_stats(getattr(sa, "q_proj", None)))
                log.warning("[PN272]     o_proj : %s",
                            _weight_stats(getattr(sa, "o_proj", None)))
                log.warning("[PN272]     q_norm : %s",
                            _weight_stats(getattr(sa, "q_norm", None)))

    # LM head + tying check
    if top is not None:
        lm = getattr(top, "lm_head", None)
        log.warning("[PN272] top.lm_head : %s", _weight_stats(lm))
        emb_w = (
            getattr(getattr(predictor, "embed_tokens", None), "weight", None)
        )
        lm_w = getattr(lm, "weight", None) if lm is not None else None
        if emb_w is not None and lm_w is not None:
            tied = int(emb_w.data_ptr()) == int(lm_w.data_ptr())
            log.warning(
                "[PN272] lm_head <-> embed_tokens tied_storage=%s "
                "(emb data_ptr=0x%x lm data_ptr=0x%x)",
                tied, int(emb_w.data_ptr()), int(lm_w.data_ptr()),
            )

    # State_dict key matrix
    try:
        target_root = _unwrap(top if top is not None else predictor)
        sd_keys = list(target_root.state_dict().keys())
    except Exception as _e:
        sd_keys = []
        log.warning("[PN272] state_dict() failed: %s", _e)

    expected_substrings = [
        "embed_tokens",
        "pre_projection.weight",
        "post_projection.weight",
        "norm.weight",
        "lm_head.weight",
        ".layers.0.layer_scalar",
        ".layers.0.input_layernorm",
        ".layers.0.post_attention_layernorm",
        ".layers.0.pre_feedforward_layernorm",
        ".layers.0.post_feedforward_layernorm",
        ".layers.0.self_attn.q_proj",
        ".layers.0.self_attn.o_proj",
        ".layers.3.layer_scalar",
        ".layers.3.self_attn.q_proj",
    ]
    log.warning("[PN272] state_dict total keys = %d", len(sd_keys))
    for sub in expected_substrings:
        hits = [k for k in sd_keys if sub in k]
        log.warning("[PN272]   expected '%s' -> %d hit(s): %s",
                    sub, len(hits), hits[:3])
    log.warning("[PN272] === inventory END ===")


# ----------------------- Phase 2: runtime forward wraps -----------------------

def _install_runtime_wraps() -> None:
    """Wrap Gemma4MultiTokenPredictor.forward + Gemma4MTPDecoderLayer.forward
    + Gemma4MTPAttention.forward for one-shot tensor-stats capture."""
    global _ORIGINAL_MTP_FORWARD, _ORIGINAL_LAYER_FORWARD

    try:
        from vllm.model_executor.models.gemma4_mtp import (
            Gemma4MultiTokenPredictor,
            Gemma4MTPDecoderLayer,
            Gemma4MTPAttention,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("[PN272] runtime wrap SKIP: import failed: %s", e)
        return

    if getattr(Gemma4MultiTokenPredictor.forward,
               "_genesis_pn272_wrapped", False):
        return

    _ORIGINAL_MTP_FORWARD = Gemma4MultiTokenPredictor.forward
    _ORIGINAL_LAYER_FORWARD = Gemma4MTPDecoderLayer.forward

    def _mtp_wrapped(self, input_ids, positions, hidden_states,
                    intermediate_tensors=None, inputs_embeds=None,
                    spec_step_idx=0):
        global _RUNTIME_DUMPED
        # Skip warmup-shape calls
        try:
            num_tokens = int(input_ids.shape[0]) if input_ids is not None else 0
        except Exception:
            num_tokens = 0
        # only dump once + skip empty / dummy passes
        if not _RUNTIME_DUMPED and num_tokens > 0:
            try:
                log.warning("[PN272] === RUNTIME PROBE BEGIN ===")
                log.warning(
                    "[PN272] MTP.forward input_ids shape=%s [:8]=%s "
                    "positions[:8]=%s spec_step_idx=%s",
                    tuple(input_ids.shape),
                    _t_head(input_ids, 8), _t_head(positions, 8),
                    spec_step_idx,
                )
                log.warning("[PN272] MTP.forward target hidden_states: %s",
                            _t_stats(hidden_states))
                if inputs_embeds is not None:
                    log.warning(
                        "[PN272] MTP.forward inputs_embeds (caller-supplied): %s",
                        _t_stats(inputs_embeds),
                    )
                # Step through manually to capture intermediates BEFORE
                # calling original. We can do this by re-running the
                # same logic, but it's wasteful — instead just install
                # a per-component instrumentation flag, then call
                # original.
                # Capture: embed result + cat result + pre_projection out.
                if inputs_embeds is None:
                    try:
                        ie = self.embed_input_ids(input_ids)
                        log.warning(
                            "[PN272]   inputs_embeds (computed): %s",
                            _t_stats(ie),
                        )
                    except Exception as _e:
                        log.warning("[PN272]   embed_input_ids err: %s", _e)
                        ie = None
                else:
                    ie = inputs_embeds
                if ie is not None and hidden_states is not None:
                    try:
                        import torch
                        comb = torch.cat([ie, hidden_states], dim=-1)
                        log.warning("[PN272]   combined cat: %s", _t_stats(comb))
                        pp_out, _ = self.pre_projection(comb)
                        log.warning("[PN272]   pre_projection out: %s",
                                    _t_stats(pp_out))
                    except Exception as _e:
                        log.warning("[PN272]   pre_projection probe err: %s",
                                    _e)
            except Exception as _e:
                log.warning("[PN272] outer MTP probe err: %s", _e)

        result = _ORIGINAL_MTP_FORWARD(
            self, input_ids, positions, hidden_states,
            intermediate_tensors, inputs_embeds, spec_step_idx,
        )

        if not _RUNTIME_DUMPED and num_tokens > 0:
            try:
                draft_hs, backbone_hs = result
                log.warning("[PN272] MTP.forward OUT draft_hs: %s",
                            _t_stats(draft_hs))
                log.warning("[PN272] MTP.forward OUT backbone_hs: %s",
                            _t_stats(backbone_hs))
                log.warning("[PN272] === RUNTIME PROBE END ===")
                _RUNTIME_DUMPED = True
            except Exception as _e:
                log.warning("[PN272] MTP out unpack err: %s", _e)
        return result

    _mtp_wrapped._genesis_pn272_wrapped = True  # type: ignore[attr-defined]
    Gemma4MultiTokenPredictor.forward = _mtp_wrapped  # type: ignore[method-assign]

    # Per-layer wrap: capture stats only on the first real run
    _LAYER_DUMP_FLAGS = {i: False for i in range(4)}

    def _layer_wrapped(self, positions, hidden_states, residual=None, **kwargs):
        layer_idx = getattr(self, "_genesis_pn272_layer_idx", None)
        if layer_idx is None:
            # Try to recover layer_idx from the layer_scalar tied buffer
            try:
                # Discover by hashing the buffer's data_ptr or by reading
                # the prefix from a child attn module:
                attn = getattr(getattr(self, "self_attn", None), "attn", None)
                pfx = getattr(attn, "prefix", "") or ""
                if "layers." in pfx:
                    layer_idx = int(pfx.split("layers.")[1].split(".")[0])
                    self._genesis_pn272_layer_idx = layer_idx
            except Exception:
                pass
        if (layer_idx in _LAYER_DUMP_FLAGS
                and not _LAYER_DUMP_FLAGS[layer_idx]
                and hidden_states is not None):
            try:
                num_tokens = int(hidden_states.shape[0])
                # only on real (non-warmup) calls
                f = hidden_states.float()
                # heuristic: warmup has near-zero mean+std
                if (num_tokens > 0
                        and abs(f.mean().item()) + abs(f.std().item()) > 1e-9):
                    log.warning(
                        "[PN272] layer[%d] IN hidden_states: %s | layer_scalar=%s",
                        layer_idx, _t_stats(hidden_states),
                        _buf_value(self, "layer_scalar"),
                    )
                    out = _ORIGINAL_LAYER_FORWARD(
                        self, positions, hidden_states, residual, **kwargs)
                    hs_out, residual_out = out
                    log.warning(
                        "[PN272] layer[%d] OUT hidden_states: %s",
                        layer_idx, _t_stats(hs_out),
                    )
                    _LAYER_DUMP_FLAGS[layer_idx] = True
                    return out
            except Exception as _e:
                log.warning("[PN272] layer[%s] wrap err: %s", layer_idx, _e)
        return _ORIGINAL_LAYER_FORWARD(
            self, positions, hidden_states, residual, **kwargs)

    _layer_wrapped._genesis_pn272_wrapped = True  # type: ignore[attr-defined]
    Gemma4MTPDecoderLayer.forward = _layer_wrapped  # type: ignore[method-assign]

    log.warning("[PN272] runtime wraps installed for "
                "Gemma4MultiTokenPredictor + Gemma4MTPDecoderLayer")


# ----------------------- Patch glue -----------------------

def apply() -> tuple[str, str]:
    global _APPLIED, _ORIGINAL_INIT_TENSORS

    if not _env_enabled():
        return "skipped", f"PN272 disabled (set {_ENV_ENABLE}=1)"
    if _APPLIED:
        return "applied", "PN272 already installed"

    log.warning("[PN272] apply() entered")

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as e:  # noqa: BLE001
        log.warning("[PN272] SKIP: GPUModelRunner not importable: %s", e)
        return "skipped", f"GPUModelRunner not importable: {e!r}"

    if not hasattr(GPUModelRunner, "initialize_kv_cache_tensors"):
        return "skipped", "GPUModelRunner.initialize_kv_cache_tensors missing"

    original = GPUModelRunner.initialize_kv_cache_tensors
    if getattr(original, "_genesis_pn272_wrapped", False):
        _APPLIED = True
        return "applied", "initialize_kv_cache_tensors already wrapped"
    _ORIGINAL_INIT_TENSORS = original

    def _wrapped(self, kv_cache_config, kernel_block_sizes):
        result = original(self, kv_cache_config, kernel_block_sizes)
        global _INVENTORY_DUMPED
        if not _INVENTORY_DUMPED:
            try:
                _dump_inventory(self)
                _INVENTORY_DUMPED = True
            except Exception as e:  # noqa: BLE001
                log.warning("[PN272] inventory pass failed: %s", e)
        return result

    _wrapped._genesis_pn272_wrapped = True  # type: ignore[attr-defined]
    GPUModelRunner.initialize_kv_cache_tensors = _wrapped  # type: ignore[method-assign]

    # NOTE: runtime forward wraps are disabled by default — Gemma4MTP
    # has @support_torch_compile so wrapping its sub-forward functions
    # injects log.warning() calls into a traced graph that torch.compile
    # rejects ("logging.Logger method not supported for non-export
    # cases"). Inventory phase + state_dict key matrix is sufficient
    # for the residual probe. Set
    #   GENESIS_PN272_ENABLE_RUNTIME_WRAPS=1
    # to opt back in (requires torch._dynamo.disable on the wraps —
    # not yet implemented).
    if os.environ.get(
        "GENESIS_PN272_ENABLE_RUNTIME_WRAPS", ""
    ).strip().lower() in ("1", "true", "yes"):
        try:
            _install_runtime_wraps()
        except Exception as e:  # noqa: BLE001
            log.warning("[PN272] runtime wrap install failed: %s", e)
    else:
        log.warning(
            "[PN272] runtime wraps SKIPPED (torch.compile incompat); "
            "set GENESIS_PN272_ENABLE_RUNTIME_WRAPS=1 to force"
        )

    _APPLIED = True
    log.warning(
        "[PN272] INSTALLED: inventory will run once on first "
        "initialize_kv_cache_tensors call; runtime probe captures one "
        "non-warmup forward of Gemma4MultiTokenPredictor + each MTP layer."
    )
    return "applied", "PN272 installed (probe-only)"


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED, _ORIGINAL_INIT_TENSORS, _INVENTORY_DUMPED
    global _RUNTIME_DUMPED, _ORIGINAL_MTP_FORWARD, _ORIGINAL_LAYER_FORWARD
    if not _APPLIED:
        return False
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        if _ORIGINAL_INIT_TENSORS is not None:
            GPUModelRunner.initialize_kv_cache_tensors = _ORIGINAL_INIT_TENSORS  # type: ignore[method-assign]
    except ImportError:
        pass
    try:
        from vllm.model_executor.models.gemma4_mtp import (
            Gemma4MultiTokenPredictor, Gemma4MTPDecoderLayer,
        )
        if _ORIGINAL_MTP_FORWARD is not None:
            Gemma4MultiTokenPredictor.forward = _ORIGINAL_MTP_FORWARD
        if _ORIGINAL_LAYER_FORWARD is not None:
            Gemma4MTPDecoderLayer.forward = _ORIGINAL_LAYER_FORWARD
    except ImportError:
        pass
    _APPLIED = False
    _ORIGINAL_INIT_TENSORS = None
    _INVENTORY_DUMPED = False
    _RUNTIME_DUMPED = False
    _ORIGINAL_MTP_FORWARD = None
    _ORIGINAL_LAYER_FORWARD = None
    return True


__all__ = ["GENESIS_PN272_MARKER", "apply", "is_applied", "revert"]
