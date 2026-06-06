# SPDX-License-Identifier: Apache-2.0
"""Phase 5 community SDK — release-tier validator.

`PatchManifest.validate()` (in schema_v2) covers shape-level invariants:
schema_version, kind, semver, default_on requires env_flag, etc.

This module adds RELEASE-TIER validation that the lighter dataclass
validator can't enforce because it needs filesystem access AND knowledge
of the rest of the community registry:

  R-1  anchor md5 matches the actual file in `pristine_fixture` (text patches)
  R-2  `requires_patches` references resolve to manifests we discovered
  R-3  `conflicts_with` references resolve too (a phantom conflict masks
       a typo and would silently disable a patch matrix slot)
  R-4  entry_points.apply is importable when `type == "runtime_hook"`
  R-5  tests_required globs match at least one file under tests/
  R-6  patch.id is unique within `(namespace, id)` across all manifests
  R-7  default_on=True patches must be implementation_status=stable AND
       publish_state=published (extends schema rule with publish_state)

Issues are returned as `ValidationIssue` rows, NEVER raised. CLI layer
formats them and decides exit code.
"""
from __future__ import annotations

import hashlib
import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from vllm.sndr_core.model_configs.schema import SchemaError
from vllm.sndr_core.model_configs.schema_v2 import PatchManifest

from .manifest import REPO_ROOT, list_manifest_paths, load_manifest


__all__ = [
    "ValidationIssue",
    "ValidationResult",
    "validate_manifest",
    "validate_directory",
    "Severity",
]


log = logging.getLogger("genesis.community.validator")

Severity = str  # "error" | "warning" | "info"


@dataclass(frozen=True)
class ValidationIssue:
    """One row of validator output."""
    rule: str                            # e.g. "R-1", "R-7", "schema"
    severity: Severity                   # "error" | "warning" | "info"
    message: str
    path: Optional[str] = None           # manifest path the issue applies to


@dataclass
class ValidationResult:
    """Aggregate result of validating one or more manifests."""
    manifests: list[PatchManifest] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def passed(self) -> bool:
        """Validator-level pass = no errors. Warnings allowed."""
        return not self.errors

    def add(self, *issues: ValidationIssue) -> None:
        for i in issues:
            self.issues.append(i)


# ─── Individual rule helpers ──────────────────────────────────────────


def _file_md5(path: Path) -> Optional[str]:
    """Return md5 hex digest of a file's contents, or None on read error."""
    try:
        h = hashlib.md5()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _check_anchor_md5(
    manifest: PatchManifest,
    manifest_path: Optional[Path],
) -> list[ValidationIssue]:
    """R-1: every PatchTargetFile with `context_md5` set must match a
    pristine fixture's actual md5."""
    out: list[ValidationIssue] = []
    if manifest.type != "text_patch":
        return out
    for target in manifest.target_files:
        if not target.context_md5 or not target.pristine_fixture:
            continue
        # Resolve fixture path. Prefer manifest-adjacent if relative, else REPO_ROOT.
        fp = Path(target.pristine_fixture)
        if not fp.is_absolute():
            base = manifest_path.parent if manifest_path else REPO_ROOT
            fp = (base / fp).resolve()
        if not fp.is_file():
            out.append(ValidationIssue(
                rule="R-1",
                severity="error",
                message=(
                    f"target_files[{target.path!r}].pristine_fixture "
                    f"{target.pristine_fixture!r} does not exist (resolved to {fp})"
                ),
                path=str(manifest_path) if manifest_path else None,
            ))
            continue
        actual = _file_md5(fp)
        if actual is None:
            out.append(ValidationIssue(
                rule="R-1",
                severity="error",
                message=f"could not read pristine_fixture {fp} for md5 check",
                path=str(manifest_path) if manifest_path else None,
            ))
            continue
        if actual != target.context_md5:
            out.append(ValidationIssue(
                rule="R-1",
                severity="error",
                message=(
                    f"target_files[{target.path!r}] context_md5 mismatch — "
                    f"manifest declares {target.context_md5!r}, fixture "
                    f"({fp.name}) is {actual!r}. Re-anchor or rebase patch."
                ),
                path=str(manifest_path) if manifest_path else None,
            ))
    return out


def _check_cross_references(
    manifest: PatchManifest,
    known_ids: set[tuple[str, str]],
    manifest_path: Optional[Path],
) -> list[ValidationIssue]:
    """R-2/R-3: requires_patches + conflicts_with must reference known
    patches (resolves typo-and-silently-pass class of bugs)."""
    out: list[ValidationIssue] = []
    known_id_only = {pid for (_ns, pid) in known_ids}
    for req in manifest.requires_patches:
        if req not in known_id_only:
            out.append(ValidationIssue(
                rule="R-2",
                severity="error",
                message=(
                    f"requires_patches references {req!r} but no manifest "
                    f"with that id is discoverable"
                ),
                path=str(manifest_path) if manifest_path else None,
            ))
    for conf in manifest.conflicts_with:
        if conf not in known_id_only:
            out.append(ValidationIssue(
                rule="R-3",
                severity="warning",
                message=(
                    f"conflicts_with references {conf!r} but no manifest "
                    f"with that id is discoverable (likely typo — would "
                    f"silently never fire)"
                ),
                path=str(manifest_path) if manifest_path else None,
            ))
    return out


def _check_runtime_hook_apply_importable(
    manifest: PatchManifest,
    manifest_path: Optional[Path],
) -> list[ValidationIssue]:
    """R-4: runtime_hook patches must have an importable apply hook."""
    out: list[ValidationIssue] = []
    if manifest.type != "runtime_hook":
        return out
    apply_ref = manifest.entry_points.get("apply")
    if not apply_ref:
        # PatchManifest.validate() already rejects this; double-guard.
        return out
    # entry_points.apply is `module.path:callable_name`.
    if ":" not in apply_ref:
        out.append(ValidationIssue(
            rule="R-4",
            severity="error",
            message=(
                f"entry_points.apply={apply_ref!r} must be "
                f"`module.path:callable_name`"
            ),
            path=str(manifest_path) if manifest_path else None,
        ))
        return out
    mod_path, attr = apply_ref.rsplit(":", 1)
    try:
        mod = importlib.import_module(mod_path)
    except ImportError as e:
        out.append(ValidationIssue(
            rule="R-4",
            severity="error",
            message=f"cannot import {mod_path!r}: {e}",
            path=str(manifest_path) if manifest_path else None,
        ))
        return out
    if not hasattr(mod, attr):
        out.append(ValidationIssue(
            rule="R-4",
            severity="error",
            message=f"module {mod_path!r} has no attribute {attr!r}",
            path=str(manifest_path) if manifest_path else None,
        ))
        return out
    if not callable(getattr(mod, attr)):
        out.append(ValidationIssue(
            rule="R-4",
            severity="error",
            message=f"{apply_ref!r} resolves but is not callable",
            path=str(manifest_path) if manifest_path else None,
        ))
    return out


def _check_tests_required_present(
    manifest: PatchManifest,
    manifest_path: Optional[Path],
) -> list[ValidationIssue]:
    """R-5: every `tests_required` glob must resolve to ≥1 file."""
    out: list[ValidationIssue] = []
    if not manifest_path:
        return out
    base = manifest_path.parent
    for glob in manifest.tests_required:
        # Allow both relative-to-manifest and repo-root globs.
        matches = list(base.glob(glob)) or list(REPO_ROOT.glob(glob))
        if not matches:
            out.append(ValidationIssue(
                rule="R-5",
                severity="error",
                message=(
                    f"tests_required[{glob!r}] matches no files (looked under "
                    f"{base} and {REPO_ROOT})"
                ),
                path=str(manifest_path),
            ))
    return out


def _check_publish_state_for_default_on(
    manifest: PatchManifest,
    manifest_path: Optional[Path],
) -> list[ValidationIssue]:
    """R-7: default_on patches must be stable AND published (matches the
    release-tier patch-proof threshold §6.8)."""
    out: list[ValidationIssue] = []
    if not manifest.default_on:
        return out
    if manifest.implementation_status != "stable":
        out.append(ValidationIssue(
            rule="R-7",
            severity="error",
            message=(
                f"default_on=True with implementation_status="
                f"{manifest.implementation_status!r}: auto-enabled patches "
                f"must be `stable`"
            ),
            path=str(manifest_path) if manifest_path else None,
        ))
    if manifest.publish_state != "published":
        out.append(ValidationIssue(
            rule="R-7",
            severity="error",
            message=(
                f"default_on=True with publish_state="
                f"{manifest.publish_state!r}: must be `published`"
            ),
            path=str(manifest_path) if manifest_path else None,
        ))
    return out


# ─── Aggregate API ────────────────────────────────────────────────────


def validate_manifest(
    manifest: PatchManifest,
    *,
    manifest_path: Optional[Path] = None,
    known_ids: Optional[set[tuple[str, str]]] = None,
) -> list[ValidationIssue]:
    """Run all release-tier rules against ONE manifest.

    `known_ids` is the set of `(namespace, id)` pairs across the full
    registry — used for cross-reference rules R-2/R-3. Pass an empty set
    if you only have the one manifest in hand.
    """
    issues: list[ValidationIssue] = []
    # Schema-level validation in case caller skipped `.validate()` itself.
    try:
        manifest.validate()
    except SchemaError as e:
        issues.append(ValidationIssue(
            rule="schema",
            severity="error",
            message=str(e),
            path=str(manifest_path) if manifest_path else None,
        ))
        return issues  # No point running the rest — shape is bad.

    issues.extend(_check_anchor_md5(manifest, manifest_path))
    issues.extend(_check_cross_references(
        manifest,
        known_ids or set(),
        manifest_path,
    ))
    issues.extend(_check_runtime_hook_apply_importable(manifest, manifest_path))
    issues.extend(_check_tests_required_present(manifest, manifest_path))
    issues.extend(_check_publish_state_for_default_on(manifest, manifest_path))
    return issues


def validate_directory(
    root: Path,
) -> ValidationResult:
    """Walk `root` (typically `plugins/community/`), validate every
    manifest, and return aggregated `ValidationResult`.

    Cross-reference rules (R-2/R-3/R-6) need the full set of manifests
    visible; this function provides that — `validate_manifest` alone
    can't know what else is in the registry.
    """
    result = ValidationResult()

    manifest_paths = list_manifest_paths(root)
    loaded: list[tuple[Path, PatchManifest]] = []
    for path in manifest_paths:
        try:
            m = load_manifest(path)
            loaded.append((path, m))
            result.manifests.append(m)
        except SchemaError as e:
            result.add(ValidationIssue(
                rule="schema",
                severity="error",
                message=str(e),
                path=str(path),
            ))

    # R-6: uniqueness of (namespace, id).
    seen: dict[tuple[str, str], Path] = {}
    for path, m in loaded:
        key = (m.namespace, m.id)
        if key in seen:
            result.add(ValidationIssue(
                rule="R-6",
                severity="error",
                message=(
                    f"duplicate manifest: {key} already declared by "
                    f"{seen[key]} — community ids must be unique"
                ),
                path=str(path),
            ))
        else:
            seen[key] = path

    known_ids = set(seen.keys())

    # Per-manifest rules R-1, R-2, R-3, R-4, R-5, R-7.
    for path, m in loaded:
        result.add(*validate_manifest(
            m, manifest_path=path, known_ids=known_ids,
        ))

    return result
