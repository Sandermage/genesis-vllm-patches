# SPDX-License-Identifier: Apache-2.0
"""Pure-API layer for ``sndr patches bench-attach`` — M.6.2.

Thin wrapper over :func:`sndr.proof.bench_attach.attach_bench`
that returns a typed result with the post-write ``bench_delta`` dict
loaded from the artefact. The underlying ``attach_bench`` raises
``BenchAttachError`` on operator-visible failure; this module re-exports
the exception so callers can catch a single name without crossing
package boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


# Re-export so callers don't have to reach into ``proof.bench_attach``.
# (Imported lazily inside ``attach_bench`` so module load stays cheap
# on hosts where the underlying module pulls in heavy deps.)
class _BenchAttachErrorProxy:
    """Lazy attribute proxy for :class:`BenchAttachError`."""


def _bench_attach_error_cls():
    from sndr.proof.bench_attach import BenchAttachError as _Err

    return _Err


@dataclass(frozen=True)
class BenchAttachResult:
    """Outcome of attaching a bench-suite JSON to a patch proof artefact."""

    patch_id: str
    artefact_path: Path
    bench_delta: dict[str, Any]


def attach_bench(
    patch_id: str,
    bench_path: Path,
    *,
    baseline_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> BenchAttachResult:
    """Attach ``bench_path`` (and optional ``baseline_path``) to the
    proof artefact for ``patch_id``.

    Raises :class:`sndr.proof.bench_attach.BenchAttachError`
    on operator-visible failure (missing file, unrecognised shape, etc.).
    """
    from sndr.proof import DEFAULT_PROOF_DIR, load_proof_artefact
    from sndr.proof.bench_attach import attach_bench as _attach_bench

    target_dir = Path(out_dir) if out_dir is not None else DEFAULT_PROOF_DIR
    target = _attach_bench(
        patch_id,
        Path(bench_path),
        baseline_path=Path(baseline_path) if baseline_path is not None else None,
        out_dir=target_dir,
    )
    data = load_proof_artefact(target)
    bench_delta = data.get("bench_delta", {}) or {}
    return BenchAttachResult(
        patch_id=patch_id,
        artefact_path=target,
        bench_delta=bench_delta,
    )
