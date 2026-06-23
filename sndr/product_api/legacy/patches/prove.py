# SPDX-License-Identifier: Apache-2.0
"""Pure-API layer for ``sndr patches prove`` — M.6.2.

Wraps the artefact-writer in :mod:`sndr.proof` so non-CLI
callers (tests, SDK) can run the static-check sweep with explicit
parametrisation. Write paths are guarded by ``no_write`` — when ``True``
the API performs the checks only, never touches the filesystem.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class ProveOneResult:
    """Outcome of ``prove_one``.

    ``artefact_path`` is the on-disk evidence path; ``None`` when
    ``no_write=True`` was requested OR when the P-1 (registry-presence)
    check failed (we never persist "patch not found" as evidence).
    """

    proof: Any  # sndr.proof.PatchProof
    artefact_path: Optional[Path]
    static_passed: bool


@dataclass(frozen=True)
class ProveAllResult:
    """Outcome of ``prove_all`` sweep across PATCH_REGISTRY."""

    total: int
    passed: int
    failed: int
    coverage_pct: float
    results: tuple[dict[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DeadDetectResult:
    """Outcome of ``dead_detect``: list of patches with no proof artefact."""

    total_patches: int
    proven: int
    dead_count: int
    coverage_pct: float
    dead_patches: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def prove_one(
    patch_id: str,
    *,
    out_dir: Optional[Path] = None,
    no_write: bool = False,
) -> ProveOneResult:
    """Verify one patch and (optionally) write the artefact.

    Writes happen only when ``no_write=False`` *and* the first static
    check (P-1 patch-in-registry) passed — otherwise we'd persist a
    "patch not found" failure as evidence, which is misleading.
    """
    from sndr.proof import (
        DEFAULT_PROOF_DIR,
        build_proof_for_patch,
        write_proof_artefact,
    )

    target_dir = Path(out_dir) if out_dir is not None else DEFAULT_PROOF_DIR
    proof = build_proof_for_patch(patch_id)
    artefact_path: Optional[Path] = None
    if not no_write and proof.static_checks and proof.static_checks[0].passed:
        artefact_path = write_proof_artefact(proof, target_dir)
    return ProveOneResult(
        proof=proof,
        artefact_path=artefact_path,
        static_passed=bool(proof.static_passed),
    )


def prove_all(
    *,
    out_dir: Optional[Path] = None,
    no_write: bool = False,
) -> ProveAllResult:
    """Sweep every PATCH_REGISTRY entry and report coverage."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    from sndr.proof import (
        DEFAULT_PROOF_DIR,
        build_proof_for_patch,
        write_proof_artefact,
    )

    target_dir = Path(out_dir) if out_dir is not None else DEFAULT_PROOF_DIR
    results: list[dict[str, Any]] = []
    passed = 0
    for pid in PATCH_REGISTRY:
        proof = build_proof_for_patch(pid)
        if not no_write and proof.static_checks[0].passed:
            write_proof_artefact(proof, target_dir)
        ok = bool(proof.static_passed)
        if ok:
            passed += 1
        results.append({
            "patch_id": pid,
            "passed": ok,
            "errors": [
                {"rule": c.rule, "message": c.message}
                for c in proof.static_errors
            ],
        })
    total = len(results)
    coverage = (passed / total) if total else 1.0
    return ProveAllResult(
        total=total,
        passed=passed,
        failed=total - passed,
        coverage_pct=round(coverage * 100, 1),
        results=tuple(results),
    )


def dead_detect(*, out_dir: Optional[Path] = None) -> DeadDetectResult:
    """List patches with no passing proof artefact."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    from sndr.proof import DEFAULT_PROOF_DIR, list_dead_patches

    target_dir = Path(out_dir) if out_dir is not None else DEFAULT_PROOF_DIR
    dead = list(list_dead_patches(out_dir=target_dir))
    total = len(PATCH_REGISTRY)
    proven = total - len(dead)
    coverage = (proven / total) if total else 1.0
    return DeadDetectResult(
        total_patches=total,
        proven=proven,
        dead_count=len(dead),
        coverage_pct=round(coverage * 100, 1),
        dead_patches=tuple(dead),
    )
