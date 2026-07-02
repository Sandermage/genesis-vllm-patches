# SPDX-License-Identifier: Apache-2.0
"""PN520 — restore the imperative Qwen3.5/3.6 GDN weight loader (revert vllm#47058).

ROOT CAUSE
==========
vLLM PR #47058 ("Remove more unnecessary ``load_weights`` methods", merged
2026-06-30 = 0.23.1rc1.dev630) replaced ``Qwen3_5Model.load_weights`` — an
imperative loader with an explicit ``stacked_params_mapping`` that fused the
split GDN projection shards ``in_proj_b``/``in_proj_a`` → ``in_proj_ba`` and
``in_proj_qkv``/``in_proj_z`` → ``in_proj_qkvz`` — with a declarative
``WeightsMapper(orig_to_new_stacked=...)`` + ``AutoWeightsLoader`` path.

For the Lorbus **Qwen3.6-27B-int4-AutoRound** checkpoint the GDN
``linear_attn.in_proj_a`` / ``in_proj_b`` shards are kept **unquantized BF16**
(``extra_config`` bits:16, data_type:fp) while everything else is INT4. The new
declarative loader fails to route those split BF16 shards into the fused
``in_proj_ba`` param, so the Gated-DeltaNet layers are left effectively
uninitialised. The model BOOTS clean (weights "load", ``apply failed=0``,
health 200) but the linear-attention path computes garbage — the classic
degenerate collapse (``"\n\n\n"`` / ``"is is is is"`` / empty, never hits EOS,
``finish_reason=length``).

This is a pure weight-load correctness bug, upstream of MTP / KV-dtype / chat
template / sampling / the whole Genesis patch stack — which is why the 27B
degenerates in EVERY runtime configuration and is identical on dev672 and
dev714 (both post-#47058). v0.24.0 (branched 2026-06-23, pre-#47058) is clean;
club-3090 serves this exact model on v0.24.0 with the old imperative loader.

Upstream state: the revert (#47233) and the follow-up MoE-zero-weights fix
(#47221) were both CLOSED WITHOUT MERGING, so ``main`` (dev630..HEAD) is still
affected. Genesis carries the fix until it lands upstream.

FIX
===
Rebind ``Qwen3_5Model.load_weights`` to the pre-#47058 imperative loader
(ported from v0.24.0 ``qwen3_5.py``), which walks the explicit
``stacked_params_mapping`` and routes each shard through the destination
param's own ``weight_loader(param, w, shard_id)`` — the BF16 ``in_proj_ba``
shards land correctly. Class-rebind (setattr), so it is robust to line drift.
The MoE-expert branch is guarded (no-op for the dense 27B) so a hypothetical
Qwen3.5-MoE checkpoint still loads its dense params correctly.

Author: Sandermage (Sander) Barzov Aleksandr — Genesis backport of the
pre-#47058 upstream loader (v0.24.0).
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.wiring.pn520_qwen3_5_load_weights")

_ENV_ENABLE = "GENESIS_ENABLE_PN520_QWEN3_5_LOAD_WEIGHTS"
_ENV_DISABLE = "GENESIS_DISABLE_PN520_QWEN3_5_LOAD_WEIGHTS"
_SENTINEL = "_genesis_pn520_imperative_load_weights"
_APPLIED = False


def _env_enabled() -> bool:
    if os.environ.get(_ENV_DISABLE, "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in ("1", "true", "yes", "on")


def _build_load_weights():
    """Build the imperative load_weights closure with its upstream deps captured
    from the live vllm install (post-#47058 qwen3_5.py no longer imports them)."""
    import torch  # noqa: F401  (referenced by the checkpoint iterable typing)
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.model_executor.models.utils import is_pp_missing_parameter
    # maybe_remap_kv_scale_name moved between models/utils.py and
    # model_loader/weight_utils.py across pins — resolve from either.
    try:
        from vllm.model_executor.models.utils import maybe_remap_kv_scale_name
    except ImportError:
        from vllm.model_executor.model_loader.weight_utils import (
            maybe_remap_kv_scale_name,
        )

    def load_weights(self, weights):
        # (param_name, shard_name, shard_id) — the pre-#47058 explicit mapping.
        stacked_params_mapping = [
            # GDN split projections (the shards #47058's declarative mapper drops)
            ("in_proj_qkvz", "in_proj_qkv", (0, 1, 2)),
            ("in_proj_qkvz", "in_proj_z", 3),
            ("in_proj_ba", "in_proj_b", 0),
            ("in_proj_ba", "in_proj_a", 1),
            # self attention
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # mlp
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        # Dense (27B) has no experts → empty mapping, expert branch is skipped.
        try:
            expert_params_mapping = self.get_expert_mapping()
        except Exception:  # noqa: BLE001 — dense model / no MoE helper
            expert_params_mapping = []

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if name.startswith("mtp."):
                continue
            if name.endswith("scale"):
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                # Skip extra bias for GPTQ/AutoRound models.
                if mapped.endswith(".bias") and mapped not in params_dict:
                    continue
                if is_pp_missing_parameter(mapped, self):
                    continue
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                name = mapped
                break
            else:
                # Expert branch (guarded; no-op for the dense 27B).
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    if (
                        name_mapped.endswith(".bias") or name_mapped.endswith("_bias")
                    ) and name_mapped not in params_dict:
                        continue
                    if name_mapped not in params_dict:
                        continue
                    param = params_dict[name_mapped]
                    success = param.weight_loader(
                        param, loaded_weight, name_mapped,
                        shard_id=shard_id, expert_id=expert_id, return_success=True,
                    )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        continue
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        log.warning("[PN520] param %s not in params_dict, skip loading", name)
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    return load_weights


def apply() -> tuple[str, str]:
    """Rebind Qwen3_5Model.load_weights to the pre-#47058 imperative loader."""
    global _APPLIED
    if not _env_enabled():
        return "skipped", (
            f"PN520 disabled (set {_ENV_ENABLE}=1 to restore the pre-#47058 "
            "Qwen3.5/3.6 GDN weight loader that fixes the 27B AutoRound garbage)"
        )
    if _APPLIED:
        return "applied", "PN520 already installed (idempotent)"
    try:
        from vllm.model_executor.models import qwen3_5 as mod
    except Exception as e:  # noqa: BLE001
        return "skipped", f"PN520: vllm qwen3_5 model module not present ({e!r})"
    cls = getattr(mod, "Qwen3_5Model", None)
    if cls is None:
        return "skipped", "PN520: Qwen3_5Model not found (pin may pre-date qwen3_5)"
    if getattr(cls, _SENTINEL, False):
        _APPLIED = True
        return "applied", "PN520 already installed (class sentinel present)"
    try:
        new_load_weights = _build_load_weights()
        cls.load_weights = new_load_weights
        setattr(cls, _SENTINEL, True)
    except Exception as e:  # noqa: BLE001
        return "failed", f"PN520 rebind raised {e!r}"
    _APPLIED = True
    log.info(
        "[PN520] rebound Qwen3_5Model.load_weights to the pre-#47058 imperative "
        "loader (fixes GDN in_proj_ba/in_proj_qkvz BF16 shard routing for the "
        "27B AutoRound checkpoint).",
    )
    return "applied", (
        "PN520 installed on Qwen3_5Model.load_weights (imperative GDN loader; "
        f"reverts vllm#47058). Override via {_ENV_DISABLE}=1."
    )


# Opt-in binding contract for the drift checker (class-rebind, no text anchor):
# the symbols this patch rebinds against upstream.
def _upstream_bindings():
    return [
        ("vllm.model_executor.models.qwen3_5", "Qwen3_5Model"),
        ("vllm.model_executor.models.utils", "is_pp_missing_parameter"),
        ("vllm.model_executor.model_loader.weight_utils", "default_weight_loader"),
        # maybe_remap_kv_scale_name resolves from weight_utils OR models.utils
        # (moved across pins) — the patch tries both, so no hard binding here.
    ]
