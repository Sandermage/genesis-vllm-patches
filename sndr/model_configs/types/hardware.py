# SPDX-License-Identifier: Apache-2.0
"""HardwareSpec — GPU + system requirements for a ModelConfig.

Relocated from ``model_configs/schema.py`` in M.5.1. The class body is
byte-identical to the pre-refactor version; the only delta is the
import path for :class:`SchemaError`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ._base import SchemaError


@dataclass
class HardwareSpec:
    """GPU + system requirements for the config to apply cleanly."""
    gpu_match_keys: list[str]   # ['rtx a5000', 'a100']
    n_gpus: int
    min_vram_per_gpu_mib: int
    cuda_capability_min: Optional[tuple[int, int]] = None  # (8, 6) for Ampere

    def validate(self) -> None:
        if not self.gpu_match_keys:
            raise SchemaError("HardwareSpec.gpu_match_keys must be non-empty")
        if self.n_gpus < 1:
            raise SchemaError(
                f"HardwareSpec.n_gpus must be >= 1 (got {self.n_gpus})"
            )
        if self.min_vram_per_gpu_mib < 1:
            raise SchemaError(
                "HardwareSpec.min_vram_per_gpu_mib must be > 0"
            )
