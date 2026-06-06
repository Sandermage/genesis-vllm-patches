# SPDX-License-Identifier: Apache-2.0
"""G4_60c — verify ``triton_turboquant_decode.py`` overlay is active.

Companion to G4_60b. Verifies bind-mount of upstream PR #42637's Triton
decode kernel file with SLIDING_WINDOW + USE_MM_PREFIX constexpr branches.

================================================================
WHAT THIS PATCH DOES
================================================================

  1. Imports ``vllm.v1.attention.ops.triton_turboquant_decode``.
  2. Inspects ``triton_turboquant_decode_attention`` launcher signature
     for PR #42637 keyword arguments (``sliding_window``,
     ``mm_prefix_range``).
  3. Logs result.

================================================================
BIND-MOUNT
================================================================

```bash
-v ${OVL}/triton_turboquant_decode.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_turboquant_decode.py:ro
```

================================================================
REFERENCES
================================================================

  * Upstream PR: https://github.com/vllm-project/vllm/pull/42637
  * Source: ``overlays/pr42637/triton_turboquant_decode.py``.
  * Companions: G4_60b (attn overlay), G4_60d (store overlay).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import inspect
import logging
import os

log = logging.getLogger("genesis.turboquant.g4_60c_tq_decode_overlay")

GENESIS_G4_60C_MARKER = (
    "Genesis G4_60c verify triton_turboquant_decode.py PR #42637 "
    "overlay is active"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_60C_TQ_DECODE_OVERLAY"
_APPLIED = False

# Expected kwargs in PR #42637's launcher signature.
_PR42637_LAUNCHER_KWARGS = ("sliding_window", "mm_prefix_range")


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Verify triton_turboquant_decode.py overlay is bind-mounted."""
    global _APPLIED

    if not _env_enabled():
        return "skipped", (
            f"G4_60c disabled (set {_ENV_ENABLE}=1 to verify overlay)"
        )

    if _APPLIED:
        return "applied", "G4_60c already verified (idempotent)"

    try:
        from vllm.v1.attention.ops import triton_turboquant_decode as _td
    except ImportError as e:
        return "error", (
            f"vllm.v1.attention.ops.triton_turboquant_decode not "
            f"importable: {e}"
        )

    launcher = getattr(_td, "triton_turboquant_decode_attention", None)
    if launcher is None:
        return "error", (
            "triton_turboquant_decode_attention launcher missing — "
            "overlay structurally broken"
        )

    try:
        sig = inspect.signature(launcher)
        param_names = set(sig.parameters.keys())
    except (TypeError, ValueError) as e:
        return "error", f"cannot inspect launcher signature: {e}"

    missing = [k for k in _PR42637_LAUNCHER_KWARGS if k not in param_names]
    if missing:
        return "error", (
            f"Overlay NOT active: launcher missing PR #42637 kwargs "
            f"{missing}. Bind-mount the PR overlay file. Existing "
            f"params: {sorted(param_names)}"
        )

    _APPLIED = True
    log.info(
        "[G4_60c] triton_turboquant_decode.py PR #42637 overlay "
        "verified: launcher accepts %s.",
        ", ".join(_PR42637_LAUNCHER_KWARGS),
    )
    return "applied", (
        f"G4_60c overlay verified: launcher accepts {_PR42637_LAUNCHER_KWARGS}."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    return False


__all__ = ["GENESIS_G4_60C_MARKER", "apply", "is_applied", "revert"]
