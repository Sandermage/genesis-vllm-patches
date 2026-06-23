# SPDX-License-Identifier: Apache-2.0
"""G4_60b — verify ``turboquant_attn.py`` overlay is active (PR #42637 overlay loader).

================================================================
PROBLEM
================================================================

The Genesis G4_60a/e/g/h/k monkey-patches install Python-side hooks
for PR #42637's spec dispatch + skip-layer logic, but the full
``TurboQuantAttentionImpl`` in upstream
``vllm/v1/attention/backends/turboquant_attn.py`` is 1308 LOC of new
forward-path code (mm_prefix handling, sliding-window mask integration,
shared-K=V cache attention, continuation prefill restructuring). That
file CANNOT be monkey-patched piecewise — its internal methods call
each other through closures and shared state.

The only realistic way to land PR #42637 on a Genesis fork without waiting
for upstream merge is **file-level bind-mount overlay**: replace the
``site-packages/vllm/v1/attention/backends/turboquant_attn.py`` file
inside the container at runtime via ``docker run -v <overlay>:<target>:ro``.

================================================================
WHAT THIS PATCH DOES
================================================================

G4_60b is NOT a monkey-patch. It's a **loader verifier** that:

  1. Imports ``vllm.v1.attention.backends.turboquant_attn`` (which
     triggers Python to read whatever file is at the bind-mount target).

  2. Inspects the loaded module for PR #42637 signature methods
     (``_decode_prefill_from_cache``, ``_continuation_prefill``,
     ``reserve_turboquant_decode_workspace``, etc.).

  3. Logs whether overlay is active OR returns ``error`` if signature
     methods missing (overlay was not bind-mounted).

  4. Also imports the 2 Triton kernel files
     (``triton_turboquant_decode`` + ``triton_turboquant_store``) and
     inspects them for PR #42637 signatures (``SLIDING_WINDOW``,
     ``USE_MM_PREFIX`` constexprs).

================================================================
HOW TO BIND-MOUNT
================================================================

In launch script:

```bash
OVL=${GENESIS_REPO}/sndr/engines/vllm/patches/attention/turboquant/overlays/pr42637
docker run \
  -v ${OVL}/turboquant_attn.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/turboquant_attn.py:ro \
  -v ${OVL}/triton_turboquant_decode.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_turboquant_decode.py:ro \
  -v ${OVL}/triton_turboquant_store.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_turboquant_store.py:ro \
  -v ${OVL}/turboquant_config.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/turboquant/config.py:ro \
  -v ${OVL}/kv_cache_interface.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/kv_cache_interface.py:ro \
  -v ${OVL}/kv_cache_utils.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py:ro \
  -e GENESIS_ENABLE_G4_60B_TQ_ATTN_OVERLAY=1 \
  -e GENESIS_ENABLE_G4_60C_TQ_DECODE_OVERLAY=1 \
  -e GENESIS_ENABLE_G4_60D_TQ_STORE_OVERLAY=1 \
  ...
```

Genesis launcher ``start_g4_60_full_overlay.sh`` does this automatically.

================================================================
DEPENDENCIES
================================================================

  * Bind-mount must be in place at container start (this patch only
    verifies; it doesn't perform the mount itself).
  * Compatible with G4_60a/e/g/h/k (monkey-patches on top of overlay).
    When overlay is active, G4_60a/h injections become no-ops because
    the overlay file already defines the symbols natively.
  * G4_61 + G4_62 (workspace + warmup) also remain useful — they patch
    different surfaces.

================================================================
SCOPE
================================================================

Active only when ``GENESIS_ENABLE_G4_60B_TQ_ATTN_OVERLAY=1``. Pure
diagnostic — no monkey-patching here. Failure mode: returns ``error``
with explanation if overlay missing.

================================================================
REFERENCES
================================================================

  * Upstream PR: https://github.com/vllm-project/vllm/pull/42637
  * Overlay sources: ``overlays/pr42637/turboquant_attn.py`` etc.
  * Companion loaders: G4_60c (decode kernel), G4_60d (store kernel).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.turboquant.g4_60b_tq_attn_overlay")

GENESIS_G4_60B_MARKER = (
    "Genesis G4_60b verify turboquant_attn.py PR #42637 overlay is active"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_60B_TQ_ATTN_OVERLAY"
_APPLIED = False

# PR #42637 signature symbols expected on overlay
_PR42637_SIGNATURE = (
    "_decode_prefill_from_cache",
    "_continuation_prefill",
    "_cache_prefill_attention",
)


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Verify turboquant_attn.py PR #42637 overlay is bind-mounted."""
    global _APPLIED

    if not _env_enabled():
        return "skipped", (
            f"G4_60b disabled (set {_ENV_ENABLE}=1 to verify overlay)"
        )

    if _APPLIED:
        return "applied", "G4_60b already verified (idempotent)"

    try:
        from vllm.v1.attention.backends import turboquant_attn as _tqa
    except ImportError as e:
        return "error", (
            f"vllm.v1.attention.backends.turboquant_attn not importable: {e}"
        )

    # Check TurboQuantAttentionImpl class has PR #42637 methods.
    impl_cls = getattr(_tqa, "TurboQuantAttentionImpl", None)
    if impl_cls is None:
        return "error", (
            "TurboQuantAttentionImpl class missing — overlay file is "
            "structurally broken"
        )

    missing = [
        name for name in _PR42637_SIGNATURE if not hasattr(impl_cls, name)
    ]
    if missing:
        return "error", (
            f"Overlay NOT active: TurboQuantAttentionImpl missing PR "
            f"#42637 methods {missing}. Bind-mount may not be in place. "
            f"Re-launch container with -v overlay flag (see "
            f"overlays/pr42637/README.md)."
        )

    _APPLIED = True
    log.info(
        "[G4_60b] turboquant_attn.py PR #42637 overlay verified active: "
        "TurboQuantAttentionImpl has %s.",
        ", ".join(_PR42637_SIGNATURE),
    )
    return "applied", (
        f"G4_60b overlay verified: TurboQuantAttentionImpl has all "
        f"{len(_PR42637_SIGNATURE)} PR #42637 signature methods."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    """No-op: overlay is file-level bind-mount, controlled outside Python."""
    return False


__all__ = ["GENESIS_G4_60B_MARKER", "apply", "is_applied", "revert"]
