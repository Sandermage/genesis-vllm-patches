# SPDX-License-Identifier: Apache-2.0
"""G4_60h — augment ``TurboQuantConfig`` with KV-sharing-aware skip helpers.

================================================================
PROBLEM
================================================================

vllm pin ``0.20.2rc1.dev371+gbf610c2f5`` ships a partial
``TurboQuantConfig`` (``vllm/model_executor/layers/quantization/
turboquant/config.py``, 254 lines). Present:

  * ``slot_size_aligned`` property                 ✅
  * ``get_boundary_skip_layers`` static method     ✅
  * ``from_cache_dtype`` static method             ✅
  * ``_get_full_attention_layer_indices`` helper   ✅

Missing (vs upstream PR #42637 lines 220-396):

  * ``align_kv_sharing_skip_layers`` static method    ❌
  * ``get_kv_sharing_target_skip_layers`` static method ❌
  * ``_sort_skip_layers`` helper                      ❌
  * ``_get_kv_sharing_target_fanout`` helper          ❌

Without these, **G4_60k** (engine-config skip-layer union) silently
falls through the ``hasattr`` checks and leaves ``kv_cache_dtype_skip_
layers`` containing only the boundary layers — missing the high-fanout
KV-sharing target protection.

For Gemma 4 with KV-sharing (full-attention "target" layers reused by
all subsequent sliding layers), this means the target layers get
TQ-compressed and corrupt every downstream consumer's reads. Critical
quality blocker that PR #42637 closes.

================================================================
FIX
================================================================

Inject 4 missing symbols into ``vllm.model_executor.layers.quantization.
turboquant.config`` module:

  1. ``TurboQuantConfig.align_kv_sharing_skip_layers`` (static method)
     — given an initial skip-list and the model's KV-sharing topology
     (``num_kv_shared_layers`` + ``layer_types``), propagate skip flags
     from target layers to their downstream consumers, and remove
     orphan shared-only entries that have unprotected targets. Logs a
     warning when shared layers are removed.

  2. ``TurboQuantConfig.get_kv_sharing_target_skip_layers`` (static
     method) — return only target layers with fanout ≥ half of all
     shared consumers. Low-fanout targets remain compressed to avoid
     over-padding for marginal quality benefit.

  3. ``_sort_skip_layers(list[str]) -> list[str]`` — stable sort that
     keeps numeric layer indices ordered numerically and non-numeric
     entries lexicographically after them.

  4. ``_get_kv_sharing_target_fanout(model_config, target_attention_
     type=None) -> dict[int, int]`` — counts how many later layers
     reuse each earlier layer's KV cache, filtered optionally by
     attention type.

All four are verbatim cherry-picks from PR #42637 (HEAD
``fdeb14981``), file ``vllm/model_executor/layers/quantization/
turboquant/config.py`` lines 220-396.

================================================================
DEPENDENCIES
================================================================

  * **G4_60k** consumes ``align_kv_sharing_skip_layers`` and
    ``get_kv_sharing_target_skip_layers`` via ``hasattr``-gated calls.
    Order: apply G4_60h *before* G4_60k.

================================================================
SCOPE
================================================================

Active only when ``GENESIS_ENABLE_G4_60H_TQ_CONFIG_AUGMENT=1``. Adds
4 symbols on a single module. Idempotent — checks ``hasattr`` before
injection.

For non-KV-sharing models (``num_kv_shared_layers == 0``), both new
methods short-circuit to empty/identity (matching upstream).

================================================================
RISK
================================================================

The added helpers read several optional ``hf_text_config`` attributes
(``num_hidden_layers``, ``num_kv_shared_layers``, ``layer_types``).
Missing attributes degrade to empty results — no exceptions raised,
no behaviour change for models without those fields.

================================================================
REFERENCES
================================================================

  * Upstream PR: https://github.com/vllm-project/vllm/pull/42637
  * Upstream source (PR #42637 HEAD ``fdeb14981``):
    ``vllm/model_executor/layers/quantization/turboquant/config.py``
      - ``align_kv_sharing_skip_layers``       lines 220-282
      - ``get_kv_sharing_target_skip_layers``  lines 284-304
      - ``_get_kv_sharing_target_fanout``      lines 355-386
      - ``_sort_skip_layers``                  lines 389-396

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.gemma4.g4_60h_tq_config_augment")

GENESIS_G4_60H_MARKER = (
    "Genesis G4_60h TurboQuantConfig augment: align_kv_sharing_skip_layers "
    "+ get_kv_sharing_target_skip_layers + sort/fanout helpers "
    "(PR #42637 cherry-pick)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_60H_TQ_CONFIG_AUGMENT"
_APPLIED = False


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _sort_skip_layers(skip_layers: list[str]) -> list[str]:
    """Stable sort numeric-then-lexicographic.

    Verbatim from PR #42637 lines 389-396.
    """
    def sort_key(layer: str):
        try:
            return (0, int(layer))
        except ValueError:
            return (1, layer)

    return sorted(set(skip_layers), key=sort_key)


def _get_kv_sharing_target_fanout(
    model_config,
    target_attention_type=None,
):
    """Count downstream consumers per KV-sharing target layer.

    Verbatim from PR #42637 lines 355-386.
    """
    hf_text_config = model_config.hf_text_config
    num_layers = getattr(hf_text_config, "num_hidden_layers", 0)
    num_kv_shared_layers = getattr(hf_text_config, "num_kv_shared_layers", 0)
    layer_types = getattr(hf_text_config, "layer_types", None)
    if not num_layers or not num_kv_shared_layers or not layer_types:
        return {}

    first_shared_layer = num_layers - num_kv_shared_layers
    if first_shared_layer <= 0:
        return {}

    target_fanout: dict[int, int] = {}
    prev_layer_types = layer_types[:first_shared_layer]
    for shared_idx in range(
        first_shared_layer, min(num_layers, len(layer_types))
    ):
        current_type = layer_types[shared_idx]
        try:
            target_idx = (
                len(prev_layer_types)
                - 1
                - prev_layer_types[::-1].index(current_type)
            )
        except ValueError:
            continue
        if (
            target_attention_type is not None
            and layer_types[target_idx] != target_attention_type
        ):
            continue
        target_fanout[target_idx] = target_fanout.get(target_idx, 0) + 1
    return target_fanout


def apply() -> tuple[str, str]:
    """Inject missing skip-layer helpers into TurboQuantConfig."""
    global _APPLIED

    if not _env_enabled():
        return "skipped", (
            f"G4_60h disabled (set {_ENV_ENABLE}=1 to inject "
            "TurboQuantConfig.align_kv_sharing_skip_layers + "
            "get_kv_sharing_target_skip_layers — PR #42637 cherry-pick)"
        )

    if _APPLIED:
        return "applied", "G4_60h already installed (idempotent)"

    try:
        from vllm.model_executor.layers.quantization.turboquant import (
            config as _tq_cfg,
        )
    except ImportError as e:
        return "skipped", (
            f"vllm.model_executor.layers.quantization.turboquant.config "
            f"not importable: {e}"
        )

    if (
        hasattr(_tq_cfg.TurboQuantConfig, "align_kv_sharing_skip_layers")
        and hasattr(
            _tq_cfg.TurboQuantConfig, "get_kv_sharing_target_skip_layers"
        )
    ):
        _APPLIED = True
        return "applied", (
            "TurboQuantConfig already has KV-sharing helpers (pin may "
            "have merged PR #42637 natively)"
        )

    # === Inject module-level helpers ===
    if not hasattr(_tq_cfg, "_sort_skip_layers"):
        _tq_cfg._sort_skip_layers = _sort_skip_layers
    if not hasattr(_tq_cfg, "_get_kv_sharing_target_fanout"):
        _tq_cfg._get_kv_sharing_target_fanout = _get_kv_sharing_target_fanout

    # === Inject static methods on TurboQuantConfig ===
    @staticmethod
    def align_kv_sharing_skip_layers(model_config, skip_layers):
        """Align skip layers with YOCO-style KV-sharing targets.

        Verbatim from PR #42637 lines 220-282.
        """
        hf_text_config = model_config.hf_text_config
        num_layers = getattr(hf_text_config, "num_hidden_layers", 0)
        num_kv_shared_layers = getattr(
            hf_text_config, "num_kv_shared_layers", 0
        )
        layer_types = getattr(hf_text_config, "layer_types", None)
        if not num_layers or not num_kv_shared_layers or not layer_types:
            return _sort_skip_layers(skip_layers)

        first_shared_layer = num_layers - num_kv_shared_layers
        if first_shared_layer <= 0:
            return _sort_skip_layers(skip_layers)

        skip_indices: set[int] = set()
        non_index_layers: set[str] = set()
        for layer in skip_layers:
            try:
                skip_indices.add(int(layer))
            except ValueError:
                non_index_layers.add(layer)

        removed_shared_layers: list[tuple[int, int]] = []
        prev_layer_types = layer_types[:first_shared_layer]
        for shared_idx in range(
            first_shared_layer, min(num_layers, len(layer_types))
        ):
            current_type = layer_types[shared_idx]
            try:
                target_idx = (
                    len(prev_layer_types)
                    - 1
                    - prev_layer_types[::-1].index(current_type)
                )
            except ValueError:
                continue

            if target_idx in skip_indices:
                skip_indices.add(shared_idx)
            else:
                if shared_idx in skip_indices:
                    removed_shared_layers.append((shared_idx, target_idx))
                    skip_indices.discard(shared_idx)

        if removed_shared_layers:
            log.warning(
                "Removed %d shared layer(s) from TurboQuant skip set "
                "because their KV-sharing target layers are not skipped: "
                "%s",
                len(removed_shared_layers),
                ", ".join(
                    f"{shared_idx}->target {target_idx}"
                    for shared_idx, target_idx in removed_shared_layers
                ),
            )

        aligned = [str(idx) for idx in skip_indices] + list(non_index_layers)
        return _sort_skip_layers(aligned)

    @staticmethod
    def get_kv_sharing_target_skip_layers(model_config):
        """High-fanout target layers whose KV cache is reused later.

        Verbatim from PR #42637 lines 284-304.
        """
        target_fanout = _get_kv_sharing_target_fanout(model_config)
        total_shared_consumers = sum(target_fanout.values())
        if not total_shared_consumers:
            return []
        targets = {
            target
            for target, fanout in target_fanout.items()
            if fanout * 2 >= total_shared_consumers
        }
        return _sort_skip_layers([str(idx) for idx in targets])

    _tq_cfg.TurboQuantConfig.align_kv_sharing_skip_layers = (
        align_kv_sharing_skip_layers
    )
    _tq_cfg.TurboQuantConfig.get_kv_sharing_target_skip_layers = (
        get_kv_sharing_target_skip_layers
    )

    _APPLIED = True
    log.info(
        "[G4_60h] TurboQuantConfig augmented: align_kv_sharing_skip_layers "
        "+ get_kv_sharing_target_skip_layers + helpers active."
    )
    return "applied", (
        "G4_60h installed: TurboQuantConfig now has "
        "align_kv_sharing_skip_layers + get_kv_sharing_target_skip_layers "
        "+ _sort_skip_layers + _get_kv_sharing_target_fanout."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    """Best-effort revert (test isolation only)."""
    global _APPLIED
    if not _APPLIED:
        return False
    try:
        from vllm.model_executor.layers.quantization.turboquant import (
            config as _tq_cfg,
        )

        for attr in (
            "align_kv_sharing_skip_layers",
            "get_kv_sharing_target_skip_layers",
        ):
            if hasattr(_tq_cfg.TurboQuantConfig, attr):
                try:
                    delattr(_tq_cfg.TurboQuantConfig, attr)
                except AttributeError:
                    pass
        for attr in ("_sort_skip_layers", "_get_kv_sharing_target_fanout"):
            if hasattr(_tq_cfg, attr):
                try:
                    delattr(_tq_cfg, attr)
                except AttributeError:
                    pass
    except Exception:  # noqa: BLE001
        return False
    _APPLIED = False
    return True


__all__ = [
    "GENESIS_G4_60H_MARKER",
    "apply",
    "is_applied",
    "revert",
]
