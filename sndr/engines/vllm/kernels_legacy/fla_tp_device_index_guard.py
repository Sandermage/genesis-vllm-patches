# SPDX-License-Identifier: Apache-2.0
"""FLA TP device-index int32→int64 overflow guard — T2.2 / vllm#40265.

Why this exists
───────────────
Some FLA / GDN tensor layouts compute flat device indices as
``tp_rank × num_heads × head_dim × max_seq_len``. On large TP groups
(tp_size=8, num_heads=64, head_dim=128) at long context (320K), the
product crosses 2^31 and silently wraps when the kernel uses int32
indexing. The kernel emits garbage outputs without crashing — the
worst kind of silent corruption.

vllm#40265 was closed unmerged but the bug class is real. We ship a
preflight guard that runtime patches call to verify the index space
fits in int64 and to detect the int32 boundary so kernels can opt
into int64 indexing explicitly.

Public API
──────────
- ``index_space_bytes(tp_size, num_heads, head_dim, seq_len, dtype_bytes)``
  → returns the raw flat-index magnitude as an int.
- ``check_index_overflow(...)`` → returns IndexOverflowReport with
  ``fits_int32 / fits_int64 / margin_pct`` so callers can decide
  between switching dtype, warning, or refusing to launch.
- ``raise_if_int64_overflow(...)`` — hard-fail when the index space
  exceeds int64 (theoretically impossible on current hardware but
  guards against future TP=128 / multi-petabyte KV caches).

Author: Sandermage(Sander)-Barzov Aleksandr.
"""
from __future__ import annotations

from dataclasses import dataclass

# Two's-complement bounds. We compare against magnitude (positive only)
# because index expressions in FLA/GDN never go negative.
_INT32_MAX = (1 << 31) - 1
_INT64_MAX = (1 << 63) - 1


@dataclass(frozen=True)
class IndexOverflowReport:
    """Outcome of a flat-index overflow probe.

    `magnitude` is the computed flat-index size (NOT bytes — pure
    element count). `margin_pct` is how much of int32's range is
    used; values >100% mean the int32 path silently corrupts.
    """
    magnitude: int
    fits_int32: bool
    fits_int64: bool
    margin_int32_pct: float
    margin_int64_pct: float


def index_space_bytes(
    *,
    tp_size: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    dtype_bytes: int = 1,
) -> int:
    """Compute the flat-index magnitude (in elements, multiplied by
    dtype_bytes when callers want byte-pressure).

    Negative or zero inputs raise ValueError so we don't silently
    return 0 and lull the caller into thinking everything fits.
    """
    for name, val in (("tp_size", tp_size), ("num_heads", num_heads),
                      ("head_dim", head_dim), ("seq_len", seq_len),
                      ("dtype_bytes", dtype_bytes)):
        if not isinstance(val, int):
            raise TypeError(f"{name} must be int (got {type(val).__name__})")
        if val <= 0:
            raise ValueError(f"{name} must be > 0 (got {val})")
    return tp_size * num_heads * head_dim * seq_len * dtype_bytes


def check_index_overflow(
    *,
    tp_size: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    dtype_bytes: int = 1,
) -> IndexOverflowReport:
    """Probe whether a planned index space fits int32 / int64.

    Returns a structured report so the caller can decide between:
      - keep int32 (margin <85%)
      - switch to int64 (margin 85-100%)
      - refuse to launch (margin >100% — already corrupting)

    The 85% threshold is the audit's recommendation: leaves room for
    paddings and CUDA-graph capture quirks that compute slightly
    larger indices than the operator's max_seq_len.
    """
    mag = index_space_bytes(
        tp_size=tp_size, num_heads=num_heads, head_dim=head_dim,
        seq_len=seq_len, dtype_bytes=dtype_bytes,
    )
    return IndexOverflowReport(
        magnitude=mag,
        fits_int32=mag <= _INT32_MAX,
        fits_int64=mag <= _INT64_MAX,
        margin_int32_pct=mag / _INT32_MAX * 100,
        margin_int64_pct=mag / _INT64_MAX * 100,
    )


def raise_if_int64_overflow(
    *,
    tp_size: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    dtype_bytes: int = 1,
) -> None:
    """Hard-fail when even int64 is insufficient.

    On current hardware this is unreachable (would require
    >9.2 × 10^18 elements). Kept as a defense for hypothetical
    future TP scales / multi-petabyte caches; cheap to call.
    """
    report = check_index_overflow(
        tp_size=tp_size, num_heads=num_heads, head_dim=head_dim,
        seq_len=seq_len, dtype_bytes=dtype_bytes,
    )
    if not report.fits_int64:
        raise OverflowError(
            f"FLA TP device-index space ({report.magnitude:,} elements) "
            "exceeds int64 — kernel cannot index this layout. Reduce "
            "tp_size, num_heads, head_dim, or seq_len."
        )


def warn_if_int32_overflow_imminent(
    *,
    tp_size: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    dtype_bytes: int = 1,
    threshold_pct: float = 85.0,
) -> tuple[bool, str]:
    """Soft warning when int32 utilization passes `threshold_pct` (default 85%).

    Returns (warn, message). When warn is True, the caller should
    upcast device indices to int64 before kernel launch.

    This is the function bound into our P7 dual-stream wiring per
    audit §16.7 — every GDN forward should call this with the live
    (tp_size, num_heads, head_dim, seq_len) tuple before computing
    flat indices, so silent int32 wrap is replaced by a startup-time
    upcast.
    """
    report = check_index_overflow(
        tp_size=tp_size, num_heads=num_heads, head_dim=head_dim,
        seq_len=seq_len, dtype_bytes=dtype_bytes,
    )
    if report.margin_int32_pct >= threshold_pct:
        return True, (
            f"FLA TP device-index utilization at "
            f"{report.margin_int32_pct:.1f}% of int32 "
            f"(magnitude={report.magnitude:,}). "
            "Upcast device indices to int64 before kernel launch to "
            "avoid silent overflow on long-context boundary cases."
        )
    return False, ""


__all__ = [
    "IndexOverflowReport",
    "index_space_bytes",
    "check_index_overflow",
    "raise_if_int64_overflow",
    "warn_if_int32_overflow_imminent",
]
