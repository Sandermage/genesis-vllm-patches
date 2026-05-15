# SPDX-License-Identifier: Apache-2.0
"""Site Map — anchor offset manifest schema, builder, loader.

Part of P2.1 of the patcher evolution plan (2026-05-07). The manifest
records, for each registered TextPatcher, the byte-offset and content
hash of every sub_patch.anchor in pristine vllm source. At runtime
(Phase 3, separate session), TextPatcherV2 will use the manifest to
skip the O(N×M) `anchor in content` scan and jump directly to the
known offset, verifying via md5 on a 64-byte slice instead of a full
file linear scan.

This module is the FOUNDATION (Узел 1 of the design doc). It does NOT
itself integrate into TextPatcher.apply() — that's Phase 3. MVP scope:

  - Define manifest JSON schema (v1)
  - Compute anchor metadata (byte_offset + length + md5)
  - Assemble per-patcher manifest fragment
  - Validate manifest schema
  - Verify manifest against live source files (md5 sanity)
  - Atomic write + safe load with corruption tolerance

Design principles (from research, 2026-05-07):

  1. Position-stable identifiers — pair byte_offset with anchor_md5.
     A bytes-only offset is anti-pattern (rust-analyzer lesson, Triton
     issue #2597): 1 insertion at file head invalidates every offset
     below. The md5 pairing gives O(64-byte) sanity check, not full
     file rescan, when offset suspect.

  2. Atomic write — temp file in same directory, fsync, os.replace,
     fsync parent dir. POSIX-correct pattern to survive power-fail
     (we're Linux-only, so no need for macOS F_FULLFSYNC quirk).

  3. Graceful degrade — every loader return path includes an
     "absent / corrupted / outdated" branch that returns None. Caller
     falls back to legacy O(N×M) scan. NEVER raise to caller.

  4. Schema versioning — `manifest_version` integer field. Bump on
     incompatible change. Old loaders see new version → return None
     (graceful degrade).

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

log = logging.getLogger("genesis.wiring.anchor_manifest")


# Schema version — bump on incompatible changes. Loaders MUST refuse
# to load a manifest with version different from MANIFEST_SCHEMA_VERSION.
MANIFEST_SCHEMA_VERSION = 1


# ─────────────────────────────────────────────────────────────────────────
# Hashing helpers
# ─────────────────────────────────────────────────────────────────────────


def _md5_str(text: str) -> str:
    """MD5 hex of a UTF-8 string. Matches `md5sum` of equivalent bytes."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _md5_bytes(data: bytes) -> str:
    """MD5 hex of raw bytes."""
    return hashlib.md5(data).hexdigest()


# ─────────────────────────────────────────────────────────────────────────
# Anchor metadata extraction
# ─────────────────────────────────────────────────────────────────────────


def compute_anchor_meta(pristine_src: str, anchor: str,
                        replacement: Optional[str] = None) -> Optional[dict]:
    """For given pristine source + anchor, compute manifest entry.

    Returns dict with keys:
      byte_offset:    0-based byte position of first occurrence in source
      byte_length:    byte length of the anchor
      anchor_md5:     md5 of the anchor bytes (sanity check at runtime)
      replacement_md5: optional md5 of replacement bytes (post-apply check)

    Returns None if anchor not found OR not unique (count != 1) — caller
    decides whether that's a fatal manifest defect or skip-this-anchor.

    Bytes vs characters note: Python `str.find` returns CHARACTER offset,
    but our runtime path will read file in binary mode for MD5 stability.
    For ASCII-clean Python source files (which all vllm targets are),
    char offset == byte offset. We assert this on entry.
    """
    if not isinstance(pristine_src, str) or not isinstance(anchor, str):
        return None

    # Find first occurrence (char offset)
    char_offset = pristine_src.find(anchor)
    if char_offset == -1:
        return None

    # Uniqueness check — manifest is meaningless if anchor is ambiguous
    if pristine_src.count(anchor) != 1:
        log.debug(
            "anchor not unique (count=%d): %r...",
            pristine_src.count(anchor), anchor[:40],
        )
        return None

    # Verify char offset == byte offset for ASCII-clean files. If
    # surrounding bytes are non-ASCII, the manifest entry is unreliable
    # and we return None — caller stays on legacy path.
    src_bytes = pristine_src.encode("utf-8")
    anchor_bytes = anchor.encode("utf-8")
    byte_offset = src_bytes.find(anchor_bytes)
    if byte_offset == -1:
        return None  # sanity — should never happen if str find succeeded
    if src_bytes.count(anchor_bytes) != 1:
        return None  # bytes-level non-uniqueness (multi-byte char edge case)

    entry: dict[str, Any] = {
        "byte_offset": byte_offset,
        "byte_length": len(anchor_bytes),
        "anchor_md5": _md5_bytes(anchor_bytes),
    }
    if replacement is not None:
        entry["replacement_md5"] = _md5_str(replacement)
    return entry


def _file_meta(src_text: str) -> dict:
    """Compute file-level metadata: pristine md5 + byte size."""
    src_bytes = src_text.encode("utf-8")
    return {
        "md5_pristine": _md5_bytes(src_bytes),
        "size_bytes": len(src_bytes),
    }


# ─────────────────────────────────────────────────────────────────────────
# Manifest assembly
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class PatcherManifestInput:
    """Adapter for builder — describes a single patcher's contribution
    to the manifest.

    Why not pass TextPatcher directly: TextPatcher requires
    `target_file` to be an absolute filesystem path, which makes it
    awkward to use against pristine fixtures (different path) or
    relative-path lookup. PatcherManifestInput decouples the patch
    metadata from the source location.
    """
    patch_id: str  # e.g. "PN79"
    rel_path: str  # path RELATIVE to vllm root, posix-style
                   # e.g. "model_executor/layers/fla/ops/chunk.py"
    # Each tuple: (sub_patch_name, anchor_str, replacement_str)
    sub_patches: list[tuple[str, str, str]]


def build_file_entry(pristine_src: str,
                     patch_inputs: list[PatcherManifestInput]
                     ) -> Optional[dict]:
    """Build manifest fragment for one file from N patches that target it.

    Returns dict matching `files.<rel_path>` schema. None if file is empty
    or no patcher contributed any anchors.

    Note: cross-patch overlap is NOT validated here — this is the builder's
    job is to record. P1.1 invariant test catches overlaps separately.
    """
    if not pristine_src:
        return None

    file_meta = _file_meta(pristine_src)
    patches: dict[str, dict] = {}

    for inp in patch_inputs:
        anchors: dict[str, dict] = {}
        for sp_name, anchor, replacement in inp.sub_patches:
            entry = compute_anchor_meta(pristine_src, anchor, replacement)
            if entry is None:
                log.warning(
                    "[%s.%s] anchor missing or non-unique in pristine — "
                    "manifest entry skipped",
                    inp.patch_id, sp_name,
                )
                continue
            anchors[sp_name] = entry
        if anchors:
            patches[inp.patch_id] = {"anchors": anchors}

    if not patches:
        return None

    return {
        "md5_pristine": file_meta["md5_pristine"],
        "size_bytes": file_meta["size_bytes"],
        "patches": patches,
    }


def assemble_manifest(
    *,
    vllm_pin: str,
    genesis_pin: str,
    file_to_inputs: dict[str, tuple[str, list[PatcherManifestInput]]],
) -> dict:
    """Assemble full manifest from per-file inputs.

    Args:
      vllm_pin: vllm version string (`vllm.__version__` value)
      genesis_pin: Genesis version (`vllm.sndr_core.version.SNDR_CORE_VERSION`)
      file_to_inputs: dict mapping relative_path -> (pristine_src, [patch_inputs])

    Returns: manifest dict ready for serialization. Files with no anchors
    are silently omitted.
    """
    files: dict[str, dict] = {}
    for rel_path, (pristine_src, patch_inputs) in file_to_inputs.items():
        entry = build_file_entry(pristine_src, patch_inputs)
        if entry is not None:
            files[rel_path] = entry

    return {
        "manifest_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "vllm.sndr_core.wiring.anchor_manifest.assemble_manifest",
        "pins": {
            "vllm": str(vllm_pin),
            "genesis": str(genesis_pin),
        },
        "files": files,
    }


# ─────────────────────────────────────────────────────────────────────────
# Schema validation
# ─────────────────────────────────────────────────────────────────────────


def validate_manifest_schema(manifest: Any) -> list[str]:
    """Return list of validation errors. Empty list = valid.

    Cheap structural validation — does NOT verify md5 against live source
    (that's `verify_manifest_against_files`).
    """
    errors: list[str] = []

    if not isinstance(manifest, dict):
        return [f"manifest must be dict, got {type(manifest).__name__}"]

    # Top-level required fields
    for key in ("manifest_version", "generated_at", "pins", "files"):
        if key not in manifest:
            errors.append(f"missing top-level key: {key!r}")

    if "manifest_version" in manifest:
        v = manifest["manifest_version"]
        if not isinstance(v, int):
            errors.append(
                f"manifest_version must be int, got {type(v).__name__}"
            )
        elif v != MANIFEST_SCHEMA_VERSION:
            errors.append(
                f"manifest_version {v} != expected {MANIFEST_SCHEMA_VERSION}"
            )

    if "pins" in manifest:
        pins = manifest["pins"]
        if not isinstance(pins, dict):
            errors.append(f"pins must be dict, got {type(pins).__name__}")
        else:
            for k in ("vllm", "genesis"):
                if k not in pins:
                    errors.append(f"pins.{k} missing")
                elif not isinstance(pins[k], str):
                    errors.append(f"pins.{k} must be str")

    if "files" in manifest:
        files = manifest["files"]
        if not isinstance(files, dict):
            errors.append(f"files must be dict, got {type(files).__name__}")
        else:
            for rel_path, entry in files.items():
                errors.extend(_validate_file_entry(rel_path, entry))

    return errors


def _validate_file_entry(rel_path: str, entry: Any) -> list[str]:
    """Sub-validator for files.<rel_path>.{md5_pristine,size_bytes,patches}."""
    errors: list[str] = []
    prefix = f"files[{rel_path!r}]"

    if not isinstance(entry, dict):
        return [f"{prefix} must be dict"]

    if "md5_pristine" not in entry:
        errors.append(f"{prefix}.md5_pristine missing")
    elif not isinstance(entry["md5_pristine"], str):
        errors.append(f"{prefix}.md5_pristine must be str")
    elif len(entry["md5_pristine"]) != 32:
        errors.append(
            f"{prefix}.md5_pristine wrong length (expected 32 hex chars, "
            f"got {len(entry['md5_pristine'])})"
        )

    if "size_bytes" not in entry:
        errors.append(f"{prefix}.size_bytes missing")
    elif not isinstance(entry["size_bytes"], int):
        errors.append(f"{prefix}.size_bytes must be int")
    elif entry["size_bytes"] < 0:
        errors.append(f"{prefix}.size_bytes must be >= 0")

    patches = entry.get("patches")
    if patches is None:
        errors.append(f"{prefix}.patches missing")
    elif not isinstance(patches, dict):
        errors.append(f"{prefix}.patches must be dict")
    else:
        for patch_id, patch_entry in patches.items():
            errors.extend(_validate_patch_entry(prefix, patch_id, patch_entry))

    return errors


def _validate_patch_entry(prefix: str, patch_id: str,
                          patch_entry: Any) -> list[str]:
    """Sub-validator for files.<rel_path>.patches.<patch_id>.anchors.{...}."""
    errors: list[str] = []
    pp = f"{prefix}.patches[{patch_id!r}]"

    if not isinstance(patch_entry, dict):
        return [f"{pp} must be dict"]

    anchors = patch_entry.get("anchors")
    if anchors is None:
        errors.append(f"{pp}.anchors missing")
        return errors
    if not isinstance(anchors, dict):
        return [f"{pp}.anchors must be dict"]

    for anchor_name, anchor_entry in anchors.items():
        ap = f"{pp}.anchors[{anchor_name!r}]"
        if not isinstance(anchor_entry, dict):
            errors.append(f"{ap} must be dict")
            continue
        for k, expected_type, type_name in (
            ("byte_offset", int, "int"),
            ("byte_length", int, "int"),
            ("anchor_md5", str, "str"),
        ):
            if k not in anchor_entry:
                errors.append(f"{ap}.{k} missing")
            elif not isinstance(anchor_entry[k], expected_type):
                errors.append(f"{ap}.{k} must be {type_name}")
        if anchor_entry.get("byte_offset", 0) < 0:
            errors.append(f"{ap}.byte_offset must be >= 0")
        if anchor_entry.get("byte_length", 1) <= 0:
            errors.append(f"{ap}.byte_length must be > 0")
        amd5 = anchor_entry.get("anchor_md5")
        if isinstance(amd5, str) and len(amd5) != 32:
            errors.append(f"{ap}.anchor_md5 wrong length")
        # replacement_md5 optional
        rmd5 = anchor_entry.get("replacement_md5")
        if rmd5 is not None:
            if not isinstance(rmd5, str):
                errors.append(f"{ap}.replacement_md5 must be str")
            elif len(rmd5) != 32:
                errors.append(f"{ap}.replacement_md5 wrong length")

    return errors


# ─────────────────────────────────────────────────────────────────────────
# Runtime verify (manifest vs live source)
# ─────────────────────────────────────────────────────────────────────────


def verify_manifest_against_source(
    manifest: dict,
    source_loader,  # callable(rel_path: str) -> Optional[str]
) -> list[str]:
    """For each file in manifest, check that the source loader returns
    content with matching md5_pristine. Validates anchors at recorded
    byte_offset.

    Returns list of mismatch descriptions. Empty = manifest matches reality.

    Used in CI / `genesis sitemap verify`. NOT called on every apply
    (per HF cache verify pattern — separate opt-in command).
    """
    errors: list[str] = []
    files = manifest.get("files", {})
    for rel_path, entry in files.items():
        src = source_loader(rel_path)
        if src is None:
            errors.append(f"{rel_path}: source not loadable")
            continue
        src_bytes = src.encode("utf-8") if isinstance(src, str) else src
        actual_md5 = _md5_bytes(src_bytes)
        if actual_md5 != entry["md5_pristine"]:
            errors.append(
                f"{rel_path}: md5 mismatch "
                f"(manifest {entry['md5_pristine']}, actual {actual_md5})"
            )
            continue
        # Verify each anchor at recorded offset
        for patch_id, patch_entry in entry.get("patches", {}).items():
            for sp_name, a in patch_entry.get("anchors", {}).items():
                start = a["byte_offset"]
                end = start + a["byte_length"]
                if end > len(src_bytes):
                    errors.append(
                        f"{rel_path}: {patch_id}.{sp_name} offset+length "
                        f"{end} exceeds file size {len(src_bytes)}"
                    )
                    continue
                slice_md5 = _md5_bytes(src_bytes[start:end])
                if slice_md5 != a["anchor_md5"]:
                    errors.append(
                        f"{rel_path}: {patch_id}.{sp_name} anchor_md5 "
                        f"mismatch at offset {start}"
                    )
    return errors


# ─────────────────────────────────────────────────────────────────────────
# Persistence — atomic write + graceful load
# ─────────────────────────────────────────────────────────────────────────


def write_manifest_atomic(path: Union[str, Path], manifest: dict) -> None:
    """Atomic JSON write: temp file in same dir → fsync → os.replace →
    fsync parent dir entry.

    Raises OSError on filesystem failure. Caller catches OR lets propagate
    (build script case).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")

    # Pretty JSON for human review (manifest is committed to repo)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"

    # Write + fsync the file content
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())

    # Atomic rename — overwrites target if exists
    os.replace(tmp, p)

    # Fsync the directory entry — POSIX requirement to survive power-fail
    # after rename. macOS would need F_FULLFSYNC; we're Linux-only.
    try:
        dir_fd = os.open(str(p.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError as e:
        # Some filesystems (NFS, FUSE) don't support directory fsync.
        # The rename itself is atomic — log warn but don't fail.
        log.warning(
            "directory fsync failed for %s: %s — rename was atomic, "
            "but power-fail recovery is filesystem-dependent",
            p.parent, e,
        )


def load_manifest(path: Union[str, Path]) -> Optional[dict]:
    """Graceful manifest load. Returns None on:
      - file missing
      - JSON corrupted
      - schema version mismatch
      - schema validation errors

    NEVER raises to caller. Always logs reason at INFO/WARN level.
    """
    p = Path(path)
    if not p.is_file():
        log.debug("manifest absent at %s", p)
        return None

    try:
        with open(p, "r", encoding="utf-8") as f:
            payload = f.read()
    except (OSError, PermissionError) as e:
        log.warning("manifest read failed for %s: %s", p, e)
        return None

    try:
        manifest = json.loads(payload)
    except json.JSONDecodeError as e:
        log.warning(
            "manifest JSON corrupted at %s: %s — fall back to legacy path",
            p, e,
        )
        return None

    # Cheap structural validation
    errors = validate_manifest_schema(manifest)
    if errors:
        log.warning(
            "manifest schema invalid (%d errors) at %s — first: %s",
            len(errors), p, errors[0],
        )
        return None

    return manifest


def load_manifest_for_pins(
    path: Union[str, Path],
    *,
    vllm_pin: Optional[str] = None,
    genesis_pin: Optional[str] = None,
) -> Optional[dict]:
    """Like load_manifest but also enforces pin match. Returns None if
    either pin is given AND mismatches the manifest's recorded pin.

    Use case: TextPatcherV2 runtime path — only trust manifest if it
    was generated for the current vllm + genesis combo.
    """
    manifest = load_manifest(path)
    if manifest is None:
        return None
    pins = manifest.get("pins", {})
    if vllm_pin is not None and pins.get("vllm") != vllm_pin:
        log.info(
            "manifest pin mismatch: vllm manifest=%r runtime=%r — invalidate",
            pins.get("vllm"), vllm_pin,
        )
        return None
    if genesis_pin is not None and pins.get("genesis") != genesis_pin:
        log.info(
            "manifest pin mismatch: genesis manifest=%r runtime=%r — invalidate",
            pins.get("genesis"), genesis_pin,
        )
        return None
    return manifest


# ─────────────────────────────────────────────────────────────────────────
# Manifest path helpers
# ─────────────────────────────────────────────────────────────────────────


def default_manifest_path() -> Path:
    """Canonical location of the committed manifest in this repo.

    `vllm/_genesis/manifests/anchor_manifest.json`

    Phase 3 may add per-pin storage in `~/.cache/genesis/sitemap/` but
    MVP uses single committed file (one current pin at a time).
    """
    here = Path(__file__).resolve().parent
    # `wiring/anchor_manifest.py` -> `vllm/_genesis/manifests/anchor_manifest.json`
    return here.parent / "manifests" / "anchor_manifest.json"
