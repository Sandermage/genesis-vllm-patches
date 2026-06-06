# SPDX-License-Identifier: Apache-2.0
"""GPU name → preset class hint (S-05 install.sh refactor, 2026-05-08).

The shell installer used to map ~32 nvidia-smi product names to
`GPU_CLASS_HINT` strings consumed by `compat.cli preset match`. That
mapping was bash case statements with brittle ordering rules ("specific
before general" on RTX 40-series).

This module is the canonical Python version. Order-matters semantics
preserved by walking the patterns in declared order and stopping on the
first substring match.

Usage:
    from vllm.sndr_core.detection.gpu_class_map import classify_gpu
    hint = classify_gpu("NVIDIA GeForce RTX 4070 Ti SUPER")
    # → "rtx 4070 ti super"
"""
from __future__ import annotations

# Order matters: specific patterns BEFORE general (e.g. "rtx 4060 ti"
# must be checked before "rtx 4060"). Same ordering as the original
# install.sh case statement.
_GPU_PATTERNS: tuple[tuple[str, str], ...] = (
    # Ampere consumer / pro
    ("rtx 3060", "rtx 3060"),
    ("rtx 3070", "rtx 3070"),
    ("rtx 3080", "rtx 3080"),
    ("rtx 3090", "rtx 3090"),
    ("rtx a4000", "rtx a4000"),
    ("rtx a5000", "rtx a5000"),
    ("rtx a6000", "rtx a6000"),
    ("a100", "a100"),
    # Ada Lovelace consumer (RTX 40-series) — specific BEFORE general
    ("rtx 4060 ti", "rtx 4060 ti"),
    ("rtx 4060", "rtx 4060"),
    ("rtx 4070 ti super", "rtx 4070 ti super"),
    ("rtx 4070 ti", "rtx 4070 ti"),
    ("rtx 4070 super", "rtx 4070 super"),
    ("rtx 4070", "rtx 4070"),
    ("rtx 4080 super", "rtx 4080 super"),
    ("rtx 4080", "rtx 4080"),
    ("rtx 4090", "rtx 4090"),
    # Ada Lovelace pro / DC
    ("l40", "l40"),
    ("rtx 6000 ada", "rtx 6000 ada"),
    # Hopper
    ("h100", "h100"),
    ("h200", "h200"),
    ("h20", "h20"),
    # Blackwell consumer (RTX 50-series, sm_120)
    ("rtx 5060 ti", "rtx 5060 ti"),
    ("rtx 5060", "rtx 5060"),
    ("rtx 5070 ti", "rtx 5070 ti"),
    ("rtx 5070", "rtx 5070"),
    ("rtx 5080", "rtx 5080"),
    ("rtx 5090", "rtx 5090"),
    # Blackwell pro (RTX PRO Blackwell line) — specific Max-Q BEFORE base
    ("rtx pro 6000 blackwell max-q", "rtx pro 6000 blackwell max-q"),
    ("rtx pro 6000 blackwell", "rtx pro 6000 blackwell"),
    ("rtx pro 4000 blackwell", "rtx pro 4000 blackwell"),
    ("rtx pro 4500 blackwell", "rtx pro 4500 blackwell"),
    ("rtx pro 5000 blackwell", "rtx pro 5000 blackwell"),
    # Blackwell DC
    ("b200", "b200"),
)


def classify_gpu(name: str) -> str:
    """Return a Genesis preset-class hint for a `nvidia-smi --query-gpu=name`
    string, or empty string if no pattern matches.

    Args:
        name: Raw GPU name (any case).

    Returns:
        Lowercase hint matching keys in `compat.gpu_profile.GPU_SPECS`,
        or "" when the GPU is not in the preset matrix.
    """
    if not name:
        return ""
    n = name.lower()
    for pattern, hint in _GPU_PATTERNS:
        if pattern in n:
            return hint
    return ""


__all__ = ["classify_gpu"]
