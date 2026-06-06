# SPDX-License-Identifier: Apache-2.0
"""mapping/gemma4 — Gemma 4 MTP drafter→target mapping provider.

Re-runs the logic of vLLM's
``Gemma4Proposer._setup_gemma4_kv_sharing`` in read-only mode:

  for each drafter layer:
      candidates = [i for i,t in target_layer_types[:non_kv_shared_cutoff]
                    if t == drafter_layer_types[i]]
      target_idx = candidates[-1]    # last layer of same type

The result is a list of LayerMapping with the live nn.Module handles
for both sides. Errors are non-fatal: provider returns [] when it
can't reach into the model tree.

Provenance: extracted from
``integrations/spec_decode/pn271_kv_contract_audit.py``
2026-05-20 per architectural directive (PN273). Relocated from
``integrations/gemma4/`` 2026-05-21 (Phase 3 bucket 1).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from .base import LayerMapping, MappingProvider

log = logging.getLogger("genesis.spec_decode.mapping.gemma4")


def _walk_chain(start: Any) -> list[Any]:
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


def _find_drafter_predictor(runner: Any) -> Any:
    """Find Gemma4MultiTokenPredictor by signature."""
    drafter = getattr(runner, "drafter", None)
    dmodel = getattr(drafter, "model", None) if drafter is not None else None
    for cand in _walk_chain(dmodel):
        if (hasattr(cand, "layers")
                and hasattr(cand, "embed_tokens")
                and hasattr(cand, "pre_projection")):
            return cand
    return None


def _find_target_root(runner: Any) -> Any:
    rm = getattr(runner, "model", None)
    if rm is None:
        return None
    # Unwrap CUDAGraphWrapper / torch.compile / etc.
    for cand in _walk_chain(rm):
        # Pick the deepest module that has named_modules
        pass
    chain = _walk_chain(rm)
    return chain[-1] if chain else rm


def _find_target_attn_module(target_root: Any, target_prefix: str) -> Any:
    """target_prefix like 'language_model.model.layers.58.self_attn.attn'.
    We want the parent '.self_attn' module.
    """
    if target_root is None or not target_prefix:
        return None
    parent_prefix = target_prefix.rsplit(".attn", 1)[0]
    try:
        for name, mod in target_root.named_modules():
            if name.endswith(parent_prefix):
                return mod
    except Exception as _e:
        log.warning("[mapping.gemma4] named_modules walk failed: %s", _e)
    return None


def _is_gemma4_config(hf_config: Any) -> bool:
    if hf_config is None:
        return False
    cls_name = type(hf_config).__qualname__.lower()
    model_type = str(getattr(hf_config, "model_type", "")).lower()
    return ("gemma" in cls_name) or ("gemma" in model_type)


def _is_mtp_method(spec_cfg: Any) -> bool:
    if spec_cfg is None:
        return False
    method = getattr(spec_cfg, "method", None)
    if method is None:
        return False
    return str(method).strip().lower() == "mtp"


def _is_quantized_kv(vllm_config: Any) -> bool:
    """Walk both model_config and cache_config — vLLM stores
    kv_cache_dtype on cache_config in v1, but legacy paths sometimes
    stash it on model_config. Return True if any source declares a
    quantized form (turboquant_*, fp8_*, etc.)."""
    if vllm_config is None:
        return False
    candidates: list[str] = []
    for parent_name, parent in (
        ("model_config", getattr(vllm_config, "model_config", None)),
        ("cache_config", getattr(vllm_config, "cache_config", None)),
    ):
        if parent is None:
            continue
        for attr in ("kv_cache_dtype", "kv_cache_dtype_str", "cache_dtype"):
            v = getattr(parent, attr, None)
            if v is not None:
                candidates.append(str(v).lower())
    for s in candidates:
        if "turboquant" in s:
            return True
        if "fp8" in s and "auto" not in s:
            return True
        if "quant" in s and s not in ("auto", "default", "none"):
            return True
    return False


class Gemma4MappingProvider(MappingProvider):
    name = "Gemma4"

    def supports(self, runner: Any) -> bool:
        try:
            mc = getattr(runner, "model_config", None)
            hf = getattr(mc, "hf_config", None) if mc is not None else None
            return _is_gemma4_config(hf)
        except Exception:
            return False

    def supports_config(self, vllm_config: Any) -> bool:
        """Match Gemma 4 ANY-form MTP at config time.

        We engage the guard for Gemma 4 + MTP. The actual verdict in
        ``evaluate_from_config`` differentiates between native bf16
        (EXACT_COPY -> allow) and quantized target KV
        (FUNCTIONAL_UNVERIFIED -> deny by default).
        """
        try:
            mc = getattr(vllm_config, "model_config", None)
            spec_cfg = getattr(vllm_config, "speculative_config", None)
            if mc is None or spec_cfg is None:
                return False
            hf = getattr(mc, "hf_config", None)
            return _is_gemma4_config(hf) and _is_mtp_method(spec_cfg)
        except Exception:
            return False

    def evaluate_from_config(self, vllm_config: Any) -> tuple[Any, str]:
        """Return (Verdict, reason) for a matched Gemma 4 MTP config.

        Decision rule (config-time, before drafter loads):

          (a) Global ``--attention-backend`` is TURBOQUANT (or similar
              quantized impl) AND target[58/59] are forced native via
              skip-list -> the drafter inherits TQ impl but its
              KV-sharing source is bf16 -> KERNEL_STORAGE_DTYPE_MISMATCH
              (β empirical finding 2026-05-20; non-overridable).

          (b) Otherwise, if target KV is quantized
              (turboquant_* / fp8) ->
              ADAPTER_STRUCTURAL_OK_FUNCTIONAL_UNVERIFIED.

          (c) Target KV native -> EXACT_COPY.
        """
        from ..kv_contract import Verdict
        # (a) Kernel-vs-storage mismatch heuristic.
        # If the engine writes a quantized cache (drafter inherits
        # this kernel) BUT the kv-sharing source target layer is
        # forced native via skip-list, the drafter kernel will
        # interpret native bf16 bytes as quantized. β empirical:
        # acceptance=0.
        # Using cache_config.cache_dtype as the engine-kernel proxy
        # is more robust than walking vllm_config attribute paths
        # for attention_backend (vLLM v1 places that field on a
        # version-dependent path).
        cache_is_tq = _is_quantized_kv(vllm_config)
        engine_backend_hint = self._engine_backend(vllm_config)
        skip_layers = self._tq_skip_layers(vllm_config)
        kv_share_targets = self._kv_share_target_indices(vllm_config)
        # Genesis-side fixes that resolve the kernel-vs-storage
        # mismatch for drafter layers. If the operator enabled these,
        # the drafter no longer inherits the TQ kernel: G4_71b routes
        # sliding drafter layers (head=256) to native Triton, and
        # G4_75 routes the full drafter layer (head=512) to native
        # Triton. With both ON, the drafter side matches the
        # native-skip-listed target side.
        # is_enabled() resolves SNDR_ENABLE_* / GENESIS_ENABLE_* aliases.
        from ....env import is_enabled
        g71b_on = is_enabled("G4_71B_DRAFTER_SLIDING_TRITON")
        g75_on = is_enabled("G4_75_DRAFTER_HEAD512_TRITON")
        if cache_is_tq:
            mismatched = [t for t in kv_share_targets if t in skip_layers]
            if mismatched and not (g71b_on and g75_on):
                missing_fixes = []
                if not g71b_on:
                    missing_fixes.append(
                        "SNDR_ENABLE_G4_71B_DRAFTER_SLIDING_TRITON=1"
                    )
                if not g75_on:
                    missing_fixes.append(
                        "SNDR_ENABLE_G4_75_DRAFTER_HEAD512_TRITON=1"
                    )
                return (
                    Verdict.KERNEL_STORAGE_DTYPE_MISMATCH,
                    f"cache_dtype is quantized (engine uses TQ-style "
                    f"kernel) but kv_sharing source layer(s) {mismatched} "
                    f"are forced native via skip-list. Drafter would read "
                    f"native bf16 bytes as TQ-packed. β empirical: "
                    f"acceptance=0 with this contract. Genesis-side fixes "
                    f"to align drafter with native source: {missing_fixes}. "
                    f"attention_backend_hint={engine_backend_hint!r}",
                )
        # (b) Plain quantized-target case.
        if _is_quantized_kv(vllm_config):
            dt = "<unknown>"
            for parent_name, parent in (
                ("model_config", getattr(vllm_config, "model_config", None)),
                ("cache_config", getattr(vllm_config, "cache_config", None)),
            ):
                if parent is None:
                    continue
                for attr in ("kv_cache_dtype", "kv_cache_dtype_str",
                             "cache_dtype"):
                    v = getattr(parent, attr, None)
                    if v is not None:
                        dt = f"{parent_name}.{attr}={v!r}"
                        break
                if dt != "<unknown>":
                    break
            return (
                Verdict.ADAPTER_STRUCTURAL_OK_FUNCTIONAL_UNVERIFIED,
                f"Gemma4 MTP with quantized target KV ({dt}) breaks "
                f"physical kv_sharing — bridge required; runtime "
                f"acceptance not validated for this configuration.",
            )
        # (c) Native fallthrough.
        return (
            Verdict.EXACT_COPY,
            "Gemma4 MTP with native KV — physical kv_sharing works as "
            "designed.",
        )

    @staticmethod
    def _engine_backend(vllm_config: Any) -> str | None:
        """Resolve the engine-wide attention backend.

        vLLM places it on different attribute paths depending on the
        version: VllmConfig.attention_backend (most recent), or
        model_config.attention_backend (older). Also falls back to
        the env var VLLM_ATTENTION_BACKEND if neither is set.
        """
        import os
        for owner_name, owner in (
            ("vllm_config", vllm_config),
            ("model_config", getattr(vllm_config, "model_config", None)),
            ("parallel_config", getattr(vllm_config, "parallel_config", None)),
            ("scheduler_config", getattr(vllm_config, "scheduler_config", None)),
        ):
            if owner is None:
                continue
            for attr in ("attention_backend", "attn_backend"):
                v = getattr(owner, attr, None)
                if v is not None:
                    s = str(v).strip()
                    if s and s.lower() not in ("none", "auto"):
                        return s.upper().replace("_ATTN", "")
        env_v = os.environ.get("VLLM_ATTENTION_BACKEND", "").strip()
        if env_v:
            return env_v.upper().replace("_ATTN", "")
        return None

    @staticmethod
    def _tq_skip_layers(vllm_config: Any) -> set[int]:
        """Read skip-list set from env (operator-set), since the
        skip-list is delivered to the runtime via env var.

        Resolves SNDR_G4_TQ_FORCE_SKIP_LAYERS first, then
        GENESIS_G4_TQ_FORCE_SKIP_LAYERS with a deprecation warning.
        """
        from ....env import get_sndr_env
        raw = get_sndr_env("G4_TQ_FORCE_SKIP_LAYERS") or ""
        out: set[int] = set()
        for piece in raw.split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                out.add(int(piece))
            except ValueError:
                continue
        return out

    def artifact_lookup_keys(self, vllm_config: Any
                              ) -> tuple[str, str, str] | None:
        """Return (model_id, profile, config_hash) for artifact lookup.

        For the Gemma4 β′-A path (TQ + skip-list 58,59 + G4_71b + G4_75
        + kv_sharing + no bridge), the profile is
        ``gemma4-31b-tq-mtp-structured-k4`` and the config_hash is computed
        over the live KV plan + MTP K + drafter backend choice.
        """
        try:
            mc = getattr(vllm_config, "model_config", None)
            raw_model = getattr(mc, "model", None) or "<unknown>"
            # Normalize: artifacts store the model basename (last path
            # component) so a launcher that passes a local path matches
            # an artifact generated against the canonical HF id.
            model_id = (raw_model.rstrip("/").rsplit("/", 1)[-1]
                        if isinstance(raw_model, str) else raw_model)

            spec_cfg = getattr(vllm_config, "speculative_config", None)
            mtp_k = (getattr(spec_cfg, "num_speculative_tokens", None)
                     if spec_cfg is not None else None)

            skip_layers = sorted(self._tq_skip_layers(vllm_config))
            cache_is_tq = _is_quantized_kv(vllm_config)
            kv_share_targets = sorted(
                self._kv_share_target_indices(vllm_config))

            from ....env import is_enabled, get_sndr_env
            g71b_on = is_enabled("G4_71B_DRAFTER_SLIDING_TRITON")
            g75_on = is_enabled("G4_75_DRAFTER_HEAD512_TRITON")
            # Preserve the pre-P1 default: when the operator does
            # not set the G4_76 disable env at all, disable_kv_sharing
            # is treated as ON (kv_sharing_on = False). Operator
            # explicitly sets the env to '0' to allow native sharing.
            raw76 = get_sndr_env(
                "ENABLE_G4_76_DISABLE_DRAFTER_KV_SHARING", default="1")
            kv_sharing_on = (
                str(raw76).strip().lower() in ("0", "false", "no", "off", "")
            )
            bridge_on = is_enabled("G4_78_DRAFTER_TARGET_KV_BRIDGE")

            drafter_backend = "TRITON_ATTN" if (g71b_on and g75_on) else None

            # Decide profile name from live config shape.
            if (cache_is_tq and skip_layers and kv_share_targets
                    and set(skip_layers) >= set(kv_share_targets)
                    and g71b_on and g75_on and kv_sharing_on
                    and not bridge_on and mtp_k == 4):
                profile = "gemma4-31b-tq-mtp-structured-k4"
            else:
                # No profile we have an artifact for.
                return None

            kv_plan = {
                "target_native_layers": skip_layers,
                "target_tq_layers_range": "0..57",
                "drafter_layers_backend": drafter_backend,
                "drafter_layers": [0, 1, 2, 3],
                "physical_kv_sharing": kv_sharing_on,
                "bridge_enabled": bridge_on,
                "skip_list": ",".join(str(i) for i in skip_layers),
                "drafter_kv_cache_dtype_reset_to": "auto",
                "genesis_patches": [
                    "G4_69 skip-list",
                    "G4_71b drafter head=256 -> Triton",
                    "G4_75 drafter head=512 -> Triton",
                    "G4_76 disable_drafter_kv_sharing=OFF (allow native "
                    "sharing)",
                ],
            }

            vllm_pin = getattr(vllm_config, "vllm_version", "") or ""
            if not vllm_pin:
                # Best-effort from version module
                try:
                    import vllm.version as _vv
                    vllm_pin = getattr(_vv, "__version__", "") or ""
                except Exception:
                    pass

            from ..functional_artifact import compute_config_hash
            config_hash = compute_config_hash(
                model_id=model_id, vllm_pin=vllm_pin, kv_plan=kv_plan,
                mtp_k=mtp_k, drafter_backend=drafter_backend,
            )
            return (model_id, profile, config_hash)
        except Exception as _e:  # noqa: BLE001
            log.warning("[mapping.gemma4] artifact_lookup_keys failed: %s",
                        _e)
            return None

    @staticmethod
    def _kv_share_target_indices(vllm_config: Any) -> list[int]:
        """Indices of (last sliding, last full) target layers — the
        only ones Gemma4Proposer's mapping points drafter to."""
        try:
            target_hf = vllm_config.model_config.hf_config
            target_text = (target_hf.get_text_config()
                           if hasattr(target_hf, "get_text_config")
                           else target_hf)
            layer_types = list(getattr(target_text, "layer_types", []))
            num_kv_shared = int(getattr(
                target_text, "num_kv_shared_layers", 0) or 0)
            non_shared = layer_types[:len(layer_types) - num_kv_shared]
            out: list[int] = []
            # last sliding
            last_sliding = next(
                (i for i, t in enumerate(reversed(non_shared))
                 if t == "sliding_attention"), None,
            )
            if last_sliding is not None:
                out.append(len(non_shared) - 1 - last_sliding)
            # last full
            last_full = next(
                (i for i, t in enumerate(reversed(non_shared))
                 if t == "full_attention"), None,
            )
            if last_full is not None:
                out.append(len(non_shared) - 1 - last_full)
            return out
        except Exception:
            return []

    def get_mapping(self, runner: Any) -> list[LayerMapping]:
        predictor = _find_drafter_predictor(runner)
        if predictor is None:
            log.warning("[mapping.gemma4] no Gemma4MultiTokenPredictor — empty")
            return []

        # Build drafter layer index -> self_attn module list
        drafter_layers: list[tuple[int, Any]] = []
        try:
            layers = list(predictor.layers)
            for idx, layer in enumerate(layers):
                sa = getattr(layer, "self_attn", None)
                if sa is not None:
                    drafter_layers.append((idx, sa))
        except Exception as _e:
            log.warning("[mapping.gemma4] drafter layer iter failed: %s", _e)
            return []

        # Resolve target layer indices via vLLM's own rule
        try:
            vllm_cfg = (getattr(runner, "vllm_config", None)
                        or getattr(getattr(runner, "drafter", None),
                                   "vllm_config", None))
            if vllm_cfg is None:
                log.warning("[mapping.gemma4] vllm_config not found — empty")
                return []
            target_hf = vllm_cfg.model_config.hf_config
            target_text = (target_hf.get_text_config()
                           if hasattr(target_hf, "get_text_config")
                           else target_hf)
            target_layer_types = list(getattr(target_text, "layer_types", []))
            target_num_kv_shared = int(getattr(
                target_text, "num_kv_shared_layers", 0) or 0)
            num_non_shared = len(target_layer_types) - target_num_kv_shared

            type_to_target_indices: dict[str, list[int]] = defaultdict(list)
            for idx, lt in enumerate(target_layer_types[:num_non_shared]):
                type_to_target_indices[lt].append(idx)

            drafter_hf = vllm_cfg.speculative_config.draft_model_config.hf_config
            drafter_text = (drafter_hf.get_text_config()
                            if hasattr(drafter_hf, "get_text_config")
                            else drafter_hf)
            drafter_layer_types = list(getattr(drafter_text, "layer_types", []))
        except Exception as _e:
            log.warning("[mapping.gemma4] config read failed: %s", _e)
            return []

        target_root = _find_target_root(runner)

        # Discover target prefix by inspecting the live target module
        # tree for an existing self_attn.attn matching the layer regex.
        target_prefix_base = None
        try:
            if target_root is not None:
                for name, _mod in target_root.named_modules():
                    if (name.endswith(".self_attn.attn")
                            and ".layers." in name
                            and "draft" not in name):
                        target_prefix_base = name.rsplit(
                            ".layers.", 1)[0] + ".layers"
                        break
        except Exception:
            pass
        if target_prefix_base is None:
            target_prefix_base = "language_model.model.layers"

        results: list[LayerMapping] = []
        for draft_idx, draft_sa in drafter_layers:
            draft_lt = (drafter_layer_types[draft_idx]
                        if draft_idx < len(drafter_layer_types)
                        else "full_attention")
            candidates = type_to_target_indices.get(draft_lt, [])
            if not candidates:
                log.warning(
                    "[mapping.gemma4] no target candidate of type %r for "
                    "drafter[%d]", draft_lt, draft_idx,
                )
                continue
            target_idx = candidates[-1]
            target_prefix = (
                f"{target_prefix_base}.{target_idx}.self_attn.attn"
            )
            target_sa = _find_target_attn_module(target_root, target_prefix)
            results.append(LayerMapping(
                drafter_idx=draft_idx,
                target_full_prefix=target_prefix,
                drafter_self_attn=draft_sa,
                target_self_attn=target_sa,
            ))
        return results


__all__ = ["Gemma4MappingProvider"]
