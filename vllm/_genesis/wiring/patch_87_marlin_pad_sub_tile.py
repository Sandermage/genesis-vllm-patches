# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 87 — Marlin W4A16/W8A16 sub-tile output dim pad-on-load.

Backport of [vllm#40361](https://github.com/vllm-project/vllm/pull/40361)
("[Kernel][Bugfix] Marlin W4A16: pad sub-tile output dims on load").

================================================================
ROOT CAUSE
================================================================

The Marlin GPTQ/AutoRound kernel requires each per-rank output dim to be
a multiple of `GPTQ_MARLIN_MIN_THREAD_N = 64`. When a packed layer's
natural output dim shards below 64 under TP (e.g. Qwen3.5
`GatedDeltaNet.in_proj_ba` with `num_v_heads=64` at TP>=2, or
`Intel/Qwen3.6-35B-A3B-int4-AutoRound` whose packing yields an n=32 shard
at TP=2), `MarlinLinearKernel.can_implement` returns False and load fails
with `size_n=... not divisible by tile_n_size=64`. On Ampere there is no
stock fallback (Machete / CutlassW4A8 are sm_90+, AllSpark requires
`group_size=-1`, etc.), so TP=2 is effectively unusable for these quants
without this patch — the model falls back to a much slower path (or
straight-up refuses to load).

Closes #35924 generically; complements #36329 (Qwen3.5-GDN-specific
`MergedColumnParallelLinear` -> `ReplicatedLinear` swap).

================================================================
FIX (class-rebind, faithful port of PR diff)
================================================================

We wrap three methods on `MarlinLinearKernel`:

  1. `can_implement` — validates shape against `round_up(n, 64)` instead
     of raw `n`, so layers with sub-tile output dims report supported.
  2. `process_weights_after_loading` — calls a new `_maybe_pad_n` helper
     BEFORE the original processing. `_maybe_pad_n` zero-pads qweight,
     scales, qzeros and bias along the output dim to the next tile
     multiple, swaps `self.config.partition_weight_shape[1]` to the
     padded value (so downstream repack/permute/zero-point transforms
     see the padded size), and stores the original `n` on the layer for
     later slicing.
  3. `apply_weights` — pads the bias if caller supplied one sized for
     the un-padded output, calls the original wrapped method (which now
     sees padded_n via `c.partition_weight_shape[1]`), and slices the
     extra columns off the output.

The padded weight columns decode to zero, so `marlin_gemm` produces zero
contribution for them, and the slice discards both before they reach the
caller. Runtime cost is zero — padding happens once at load time. VRAM
cost is a few KB per affected layer (zero-filled padding along output
dim). When the shard is already tile-aligned, `_maybe_pad_n` returns
early with `padded_n == orig_n` and the path is a no-op.

================================================================
RELEVANCE TO OUR DEPLOYMENTS
================================================================

- **Minachist/Qwen3.6-27B-INT8-AutoRound** (W8A16 hybrid, our current
  INT8 PROD-candidate): the GDN `in_proj_ba` shard at TP=2 may or may
  not be tile-aligned depending on `num_v_heads` of the specific
  checkpoint. If aligned → no-op, no harm. If not → unblocks Marlin
  path that was silently falling back to slower kernel.

- **Lorbus/Qwen3.6-27B-int4-AutoRound** (W4A16, same hybrid shape):
  same story for the INT4 path.

- **Intel/Qwen3.6-35B-A3B-int4-AutoRound** (W4A16 dense MoE, n=32
  shard at TP=2, the PR's primary repro): blocked entirely without this
  patch on TP=2, would benefit ~+24% per the PR's bench (137 → 170 t/s
  on 2× RTX 3090 SM 8.6 — same hardware family as our 2× A5000).

- **Qwen3.6-35B-A3B-FP8** (PROD): N/A — does not go through Marlin
  dispatch (FP8 has its own kernel path). No effect on prod.

================================================================
SAFETY MODEL
================================================================

- `can_implement` returns the same answer for shapes that are already
  tile-aligned (the `padded_n == orig_n` no-op fast-path is preserved).
- Original `process_weights_after_loading` still runs with full original
  semantics — `_maybe_pad_n` only grows the tensor shape; the rest of
  PWA reads `self.config.partition_weight_shape[1]` (now padded) and
  produces a consistent Marlin layout.
- `apply_weights` end-to-end behavior is identical: caller provides any
  bias size (orig_n or padded_n), the wrapper normalizes to padded_n
  before invoking the original, and slices output back to orig_n.
- Default OFF (env-gated). When OFF, the kernel runs as upstream nightly.
- Idempotent: second `apply()` call no-ops (looks for `_genesis_p87_applied`
  marker on the class).

================================================================
DRIFT DETECTION
================================================================

If upstream PR #40361 (or an equivalent) merges, the live class will
already have `_maybe_pad_n` defined. We detect that via
`hasattr(MarlinLinearKernel, '_maybe_pad_n')` and skip with a
diagnostic message — no double-wrap risk.

Author backport: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Original PR: vllm#40361.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("genesis.wiring.p87_marlin_pad_sub_tile")


# ─── Sentinel — set on MarlinLinearKernel class once apply() succeeds. ────
_GENESIS_P87_MARKER = "_genesis_p87_applied_v7_62"


# ─── Helper: pad weight/scales/qzeros/bias along output dim. ──────────────


def _genesis_p87_maybe_pad_n(self: Any, layer: Any) -> None:
    """Pad qweight/scales/qzeros/bias along the output dim to the next
    multiple of GPTQ_MARLIN_MIN_THREAD_N. Sets `layer._marlin_orig_n`
    for later output slicing. Replaces `self.config` with a new
    dataclass instance whose `partition_weight_shape` reports the padded
    out-dim so downstream transforms see the padded size.

    No-op when already tile-aligned (sets `_marlin_orig_n` and returns).
    """
    import dataclasses

    import torch.nn.functional as F  # noqa: N812

    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        GPTQ_MARLIN_MIN_THREAD_N,
    )
    from vllm.utils.math_utils import round_up

    c = self.config
    orig_n = c.partition_weight_shape[1]
    padded_n = round_up(orig_n, GPTQ_MARLIN_MIN_THREAD_N)
    layer._marlin_orig_n = orig_n
    if padded_n == orig_n:
        return

    pad = padded_n - orig_n
    pack_factor = 32 // c.weight_type.size_bits

    # qweight: [k/pack, n] int32, output_dim=1 unpacked. Pad with zeros →
    # those output columns decode to weight 0.
    q = getattr(layer, self.w_q_name)
    q.data = F.pad(q.data, (0, pad), value=0)

    # scales: [num_groups, n], output_dim=1. Pad with zeros (values don't
    # matter; padded weight columns are already zero).
    s = getattr(layer, self.w_s_name)
    s.data = F.pad(s.data, (0, pad), value=0)

    # qzeros: [num_groups, n/pack] int32, packed_dim=1. Pad by pad/pack
    # extra packed columns. Pad value 0 is safe (used only for padded
    # weight columns, which are themselves zero).
    if c.zero_points and self.w_zp_name is not None:
        zp = getattr(layer, self.w_zp_name, None)
        if zp is not None:
            zp_pad_cols = pad // pack_factor
            if zp_pad_cols > 0:
                zp.data = F.pad(zp.data, (0, zp_pad_cols), value=0)

    # bias: [n] -> [padded_n]
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.data = F.pad(layer.bias.data, (0, pad), value=0)

    # Swap config so all downstream transforms use padded n.
    self.config = dataclasses.replace(
        c,
        partition_weight_shape=(c.partition_weight_shape[0], padded_n),
    )
    log.info(
        "[Genesis P87] padded output dim %d -> %d (tile=%d)",
        orig_n,
        padded_n,
        GPTQ_MARLIN_MIN_THREAD_N,
    )


# ─── Wrapped methods. Originals captured at apply() time. ─────────────────

_ORIGINAL_CAN_IMPLEMENT: Any = None
_ORIGINAL_PROCESS_WEIGHTS_AFTER_LOADING: Any = None
_ORIGINAL_APPLY_WEIGHTS: Any = None


def _genesis_p87_can_implement(cls, c: Any) -> tuple[bool, str | None]:
    """Replacement for MarlinLinearKernel.can_implement that allows
    shapes whose per-rank out-dim is not divisible by the Marlin tile
    (GPTQ_MARLIN_MIN_THREAD_N=64). Validation happens against
    `round_up(n, 64)`; the actual padding is performed at
    `process_weights_after_loading` time by `_maybe_pad_n`.
    """
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        GPTQ_MARLIN_MIN_THREAD_N,
        MARLIN_SUPPORTED_GROUP_SIZES,
        check_marlin_supports_shape,
        query_marlin_supported_quant_types,
    )
    from vllm.platforms import current_platform
    from vllm.utils.math_utils import round_up

    if not current_platform.is_cuda():
        return False, "Marlin only supported on CUDA"

    quant_types = query_marlin_supported_quant_types(c.zero_points)
    if c.weight_type not in quant_types:
        return (
            False,
            f"Quant type ({c.weight_type}) not supported by"
            f"  Marlin, supported types are: {quant_types}",
        )

    if c.group_size not in MARLIN_SUPPORTED_GROUP_SIZES:
        return (
            False,
            f"Group size ({c.group_size}) not supported by "
            "Marlin, supported group sizes are: "
            f"{MARLIN_SUPPORTED_GROUP_SIZES}",
        )

    padded_n = round_up(c.partition_weight_shape[1], GPTQ_MARLIN_MIN_THREAD_N)
    return check_marlin_supports_shape(
        padded_n,
        c.partition_weight_shape[0],
        c.full_weight_shape[0],
        c.group_size,
    )


def _genesis_p87_process_weights_after_loading(self: Any, layer: Any) -> None:
    """Pad out-dim BEFORE the original process_weights_after_loading runs,
    so downstream repack / permute / marlin_zero_points transforms see
    the padded shape and produce a consistent Marlin layout.
    """
    _genesis_p87_maybe_pad_n(self, layer)
    _ORIGINAL_PROCESS_WEIGHTS_AFTER_LOADING(self, layer)


def _genesis_p87_apply_weights(
    self: Any,
    layer: Any,
    x: Any,
    bias: Any | None = None,
) -> Any:
    """Pad caller-supplied bias to padded_n (if necessary), call the
    original apply_weights (which now sees padded out-dim via
    `c.partition_weight_shape[1]`), and slice the extra padded columns
    off the output.
    """
    import torch.nn.functional as F  # noqa: N812

    c = self.config
    padded_n = c.partition_weight_shape[1]
    orig_n = getattr(layer, "_marlin_orig_n", padded_n)

    if bias is not None and bias.shape[-1] != padded_n:
        bias = F.pad(bias, (0, padded_n - bias.shape[-1]), value=0)

    out = _ORIGINAL_APPLY_WEIGHTS(self, layer, x, bias)

    if orig_n != padded_n:
        out = out[..., :orig_n].contiguous()
    return out


# ─── apply / is_applied / revert ──────────────────────────────────────────


def apply() -> tuple[str, str]:
    """Wrap MarlinLinearKernel methods so sub-tile output dims are
    handled via load-time padding + apply-time slicing.

    Returns (status, reason). status in {"applied", "skipped", "failed"}.
    """
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("P87")
    log_decision("P87", decision, reason)
    if not decision:
        return "skipped", reason

    try:
        from vllm.model_executor.kernels.linear.mixed_precision.marlin import (
            MarlinLinearKernel,
        )
    except Exception as e:
        return "skipped", f"MarlinLinearKernel import failed: {e}"

    # Idempotency: already wrapped
    if getattr(MarlinLinearKernel, _GENESIS_P87_MARKER, False):
        return "applied", "idempotent (already wrapped)"

    # Drift: upstream may have shipped its own _maybe_pad_n
    if hasattr(MarlinLinearKernel, "_maybe_pad_n"):
        return (
            "skipped",
            "MarlinLinearKernel._maybe_pad_n already present — upstream "
            "PR #40361 (or equivalent) appears merged; auto-skip P87",
        )

    # Sanity: required methods exist
    for required in (
        "can_implement",
        "process_weights_after_loading",
        "apply_weights",
        "config",
        "w_q_name",
        "w_s_name",
        "w_zp_name",
    ):
        if not hasattr(MarlinLinearKernel, required) and not hasattr(
            MarlinLinearKernel, "__annotations__"
        ):
            return (
                "skipped",
                f"MarlinLinearKernel missing expected attribute {required!r} "
                "— upstream layout drift, refusing to wrap",
            )

    # Verify the imports we need are still available before wrapping
    try:
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            GPTQ_MARLIN_MIN_THREAD_N,  # noqa: F401
        )
        from vllm.utils.math_utils import round_up as _round_up  # noqa: F401
    except Exception as e:
        return "skipped", f"required helper import failed: {e}"

    global _ORIGINAL_CAN_IMPLEMENT
    global _ORIGINAL_PROCESS_WEIGHTS_AFTER_LOADING
    global _ORIGINAL_APPLY_WEIGHTS

    _ORIGINAL_CAN_IMPLEMENT = MarlinLinearKernel.can_implement
    _ORIGINAL_PROCESS_WEIGHTS_AFTER_LOADING = (
        MarlinLinearKernel.process_weights_after_loading
    )
    _ORIGINAL_APPLY_WEIGHTS = MarlinLinearKernel.apply_weights

    MarlinLinearKernel.can_implement = classmethod(_genesis_p87_can_implement)
    MarlinLinearKernel.process_weights_after_loading = (
        _genesis_p87_process_weights_after_loading
    )
    MarlinLinearKernel.apply_weights = _genesis_p87_apply_weights
    setattr(MarlinLinearKernel, _GENESIS_P87_MARKER, True)

    return (
        "applied",
        "MarlinLinearKernel.{can_implement, process_weights_after_loading, "
        "apply_weights} wrapped — sub-tile out-dim now zero-padded at "
        "load + sliced at apply",
    )


def is_applied() -> bool:
    """Return True iff our wrappers are installed on MarlinLinearKernel."""
    try:
        from vllm.model_executor.kernels.linear.mixed_precision.marlin import (
            MarlinLinearKernel,
        )
    except Exception:
        return False
    return bool(getattr(MarlinLinearKernel, _GENESIS_P87_MARKER, False))


def revert() -> tuple[str, str]:
    """Restore the original MarlinLinearKernel methods. Tests-only."""
    try:
        from vllm.model_executor.kernels.linear.mixed_precision.marlin import (
            MarlinLinearKernel,
        )
    except Exception as e:
        return "failed", f"MarlinLinearKernel import failed: {e}"

    if not getattr(MarlinLinearKernel, _GENESIS_P87_MARKER, False):
        return "skipped", "P87 not currently applied"

    if _ORIGINAL_CAN_IMPLEMENT is not None:
        MarlinLinearKernel.can_implement = _ORIGINAL_CAN_IMPLEMENT
    if _ORIGINAL_PROCESS_WEIGHTS_AFTER_LOADING is not None:
        MarlinLinearKernel.process_weights_after_loading = (
            _ORIGINAL_PROCESS_WEIGHTS_AFTER_LOADING
        )
    if _ORIGINAL_APPLY_WEIGHTS is not None:
        MarlinLinearKernel.apply_weights = _ORIGINAL_APPLY_WEIGHTS
    delattr(MarlinLinearKernel, _GENESIS_P87_MARKER)
    return "applied", "MarlinLinearKernel methods restored"
