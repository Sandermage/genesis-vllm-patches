# SPDX-License-Identifier: Apache-2.0
"""GPU architecture advisor — turns a GPU name / compute capability into the
arch-aware engine flags that actually matter on that silicon.

The shipped Genesis composes target 2× A5000 (Ampere, SM 8.6) and assume
Ampere-safe defaults: FP8 KV is *storage-only* (saves VRAM, no native-compute
speedup), no native FP8 weights, FlashAttention-2. On Ada (4090, SM 8.9) and
Blackwell (5090, SM 12.x) the same flags are conservative — FP8 KV becomes
native-compute, FP8/FP4 weights and FlashAttention-3 unlock real speedups.

This module classifies the silicon and surfaces *recommendations*, so the GUI
(host discovery + config editor) can tell the operator what to enable for the
hardware actually present rather than the 3090-tuned defaults.
"""
from __future__ import annotations

import re
from typing import Any, Optional

# Curated SM → arch facts (public NVIDIA capability data).
_ARCHES = [
    # (sm_min, sm_max_exclusive, arch, fp8_kv_native, fp8_w_native, fp4, fa_version)
    (8.0, 8.6, "Ampere (A100)", False, False, False, 2),
    (8.6, 8.9, "Ampere", False, False, False, 2),
    (8.9, 9.0, "Ada Lovelace", True, True, False, 2),
    (9.0, 10.0, "Hopper", True, True, False, 3),
    (10.0, 13.0, "Blackwell", True, True, True, 3),
]

# Best-effort GPU-name → compute capability (when the engine doesn't report it).
_NAME_CAP = [
    (r"a100", 8.0), (r"a6000|a5000|a4000|a40\b|3090|3080", 8.6),
    (r"4090|4080|4070|l40|l4\b|ada", 8.9), (r"h100|h200|hopper", 9.0),
    (r"5090|5080|b100|b200|blackwell", 12.0),
]


def _cap_from_name(name: str) -> Optional[float]:
    low = (name or "").lower()
    for pat, cap in _NAME_CAP:
        if re.search(pat, low):
            return cap
    return None


def classify(*, name: Optional[str] = None, compute_cap: Optional[Any] = None) -> dict[str, Any]:
    """Classify a GPU and return arch facts + arch-aware flag recommendations."""
    cap: Optional[float] = None
    if compute_cap is not None:
        try:
            cap = float(str(compute_cap))
        except (TypeError, ValueError):
            cap = None
    if cap is None and name:
        cap = _cap_from_name(name)

    arch = "unknown"
    fp8_kv_native = fp8_w_native = fp4 = False
    fa = 2
    if cap is not None:
        for lo, hi, a, kvn, wn, f4, fav in _ARCHES:
            if lo <= cap < hi:
                arch, fp8_kv_native, fp8_w_native, fp4, fa = a, kvn, wn, f4, fav
                break

    recs: list[dict[str, str]] = []
    if cap is None:
        recs.append({"level": "info", "text": "Could not determine compute capability — flags left at safe defaults."})
    elif cap < 8.9:  # Ampere
        recs.append({"level": "ok", "text": "FP8 KV (fp8_e5m2) is storage-only here — still ~halves KV VRAM. Keep it on for long context."})
        recs.append({"level": "info", "text": "No native FP8/FP4 weight compute — keep INT4/AWQ + Marlin weights."})
        recs.append({"level": "info", "text": "FlashAttention-2 is the ceiling on Ampere (no FA3)."})
    elif cap < 9.0:  # Ada
        recs.append({"level": "ok", "text": "Ada: FP8 KV is native-compute — enable fp8_e4m3 KV for speed + VRAM."})
        recs.append({"level": "ok", "text": "Native FP8 weights available — FP8 checkpoints run at full speed."})
    else:  # Hopper / Blackwell
        recs.append({"level": "ok", "text": "Hopper/Blackwell: enable FP8 KV + FP8 weights and FlashAttention-3."})
        if fp4:
            recs.append({"level": "ok", "text": "Blackwell: FP4 (nvfp4) weights unlock the largest models on the least VRAM."})

    return {
        "name": name,
        "compute_cap": cap,
        "arch": arch,
        "fp8_kv_native": fp8_kv_native,
        "fp8_weights_native": fp8_w_native,
        "fp4_weights": fp4,
        "flash_attention": fa,
        "recommendations": recs,
    }


__all__ = ["classify"]
