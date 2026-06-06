# SPDX-License-Identifier: Apache-2.0
"""mapping/base — abstract drafter↔target layer mapping provider.

A MappingProvider returns, for one model architecture, the mapping
{drafter_layer_idx -> target_full_attention_prefix}. It does NOT
build adapters — that's kv_bridge's job. It does NOT enforce policy —
that's safety_guard's job.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, NamedTuple


class LayerMapping(NamedTuple):
    """One (drafter, target) pair, with the live nn.Module handles."""
    drafter_idx: int
    target_full_prefix: str  # ends with ".self_attn.attn"
    drafter_self_attn: Any  # nn.Module (e.g., Gemma4MTPAttention)
    target_self_attn: Any | None  # nn.Module (e.g., Gemma4Attention); may be None


class MappingProvider(ABC):
    """Resolves drafter layers to their target K/V source layers."""

    #: Short identifier (used in logs and provider registry).
    name: str = "abstract"

    @abstractmethod
    def supports(self, runner: Any) -> bool:
        """Return True iff this provider can handle the given runner.

        Typical implementation inspects ``runner.model_config.hf_config``
        / its ``model_type`` / class name.
        """
        raise NotImplementedError

    def supports_config(self, vllm_config: Any) -> bool:
        """Config-only variant of supports(). Called from
        ``safety_guard.evaluate_from_config`` BEFORE workers spawn,
        when only the VllmConfig is available (no live runner / no
        drafter loaded yet).

        Default implementation returns False (provider doesn't engage
        at config time). Override to participate in the boot-time
        safety guard.
        """
        return False

    def evaluate_from_config(self, vllm_config: Any) -> tuple[Any, str]:
        """Return (Verdict, reason_str) decision based on config alone.

        Default returns (Verdict.EXACT_COPY, '<not implemented>') —
        callers must check ``supports_config`` first and only invoke
        this on matched providers.

        ``Verdict`` here is the same enum as in ``kv_contract.py``
        but passed back as Any to keep this base module
        torch/import-free.
        """
        from ..kv_contract import Verdict
        return Verdict.EXACT_COPY, "default: provider does not opine"

    @abstractmethod
    def get_mapping(self, runner: Any) -> list[LayerMapping]:
        """Return a list of (drafter, target) layer pairs for this runner.

        Implementations should:
          - Walk ``runner.drafter`` to locate the drafter sub-model
            (unwrap CUDAGraphWrapper etc.).
          - Walk ``runner.model`` to locate the target sub-model.
          - Build LayerMapping entries.
          - Return [] if the provider does not apply.

        The mapping is allowed to span N drafter layers -> 1 target
        layer (drafter[0..2] -> target[58] is fine).
        """
        raise NotImplementedError


__all__ = ["LayerMapping", "MappingProvider"]
