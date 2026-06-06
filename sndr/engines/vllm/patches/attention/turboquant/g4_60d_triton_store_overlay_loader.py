# SPDX-License-Identifier: Apache-2.0
"""G4_60d — verify ``triton_turboquant_store.py`` overlay is active.

Companion to G4_60b/c. PR #42637's changes to the store kernel file are
minor (+4/-4) — mostly signature alignment with the new launcher. This
loader just verifies the module imports successfully after bind-mount.

================================================================
BIND-MOUNT
================================================================

```bash
-v ${OVL}/triton_turboquant_store.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_turboquant_store.py:ro
```

================================================================
REFERENCES
================================================================

  * Upstream PR: https://github.com/vllm-project/vllm/pull/42637
  * Source: ``overlays/pr42637/triton_turboquant_store.py``.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.turboquant.g4_60d_tq_store_overlay")

GENESIS_G4_60D_MARKER = (
    "Genesis G4_60d verify triton_turboquant_store.py PR #42637 "
    "overlay is active"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_60D_TQ_STORE_OVERLAY"
_APPLIED = False


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Verify triton_turboquant_store.py overlay is bind-mounted."""
    global _APPLIED

    if not _env_enabled():
        return "skipped", (
            f"G4_60d disabled (set {_ENV_ENABLE}=1 to verify overlay)"
        )

    if _APPLIED:
        return "applied", "G4_60d already verified (idempotent)"

    try:
        from vllm.v1.attention.ops import triton_turboquant_store as _ts
    except ImportError as e:
        return "error", (
            f"vllm.v1.attention.ops.triton_turboquant_store not "
            f"importable: {e}"
        )

    # PR #42637 changes to store are minor — module must import and
    # expose the expected launcher.
    if not hasattr(_ts, "_tq_fused_store_fp8") and not hasattr(
        _ts, "_tq_fused_store_mse"
    ):
        return "error", (
            "store module missing _tq_fused_store_fp8/_tq_fused_store_mse "
            "kernels — overlay structurally broken"
        )

    _APPLIED = True
    log.info(
        "[G4_60d] triton_turboquant_store.py overlay verified imports."
    )
    return "applied", "G4_60d overlay verified imports OK."


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    return False


__all__ = ["GENESIS_G4_60D_MARKER", "apply", "is_applied", "revert"]
