"""Ф2 — per-pin anchor manifest generator + true-drift classifier (R2).

Given the discovery (anchor_discovery.iter_anchor_targets) and a PRISTINE vLLM
source tree for a target pin, classify every anchor against the REAL source
(no heuristics — real string search via the proven compute_anchor_meta) and
build ``pins/<pin>/anchors.json`` for the ``ok`` set plus a ``.rej`` list of
the genuinely-drifted anchors that need a human re-anchor.

R2: drift is decided from the actual pristine source text, never assumed. A
patch is ``ok`` iff its anchor is present EXACTLY ONCE (compute_anchor_meta
returns a meta); ``anchor_drift`` iff absent; ``ambiguous`` iff >1 or
non-ASCII surroundings (manifest can't address it safely).

Source reading is injected (``read_source(target_rel) -> str | None``) so the
classifier is unit-testable on synthetic fixtures locally and runs against the
live pristine tree on the rig.

See docs/superpowers/specs/2026-06-21-anchor-sot-design.md (Ф2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from sndr.engines.vllm.anchor_discovery import AnchorTarget, iter_anchor_targets
from sndr.engines.vllm.wiring.anchor_manifest import compute_anchor_meta

# status constants (a subset of check_upstream_drift's, manifest-relevant)
STATUS_OK = "ok"
STATUS_ANCHOR_DRIFT = "anchor_drift"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_UPSTREAM_MERGED = "upstream_merged"
STATUS_TARGET_MISSING = "target_missing"


def classify_anchor(
    pristine_src: str,
    anchor: str,
    replacement: Optional[str] = None,
) -> tuple[str, Optional[dict]]:
    """Classify one anchor against the pristine source (R2 — real text search).

    Returns ``(status, meta)``. ``meta`` is the compute_anchor_meta dict
    (byte_offset + md5 + replacement_md5) only when status is ``ok``.
    """
    count = pristine_src.count(anchor)
    if count == 0:
        return STATUS_ANCHOR_DRIFT, None
    if count > 1:
        return STATUS_AMBIGUOUS, None
    meta = compute_anchor_meta(pristine_src, anchor, replacement)
    if meta is None:
        # count == 1 but compute bailed (non-ASCII surroundings) — not safely
        # addressable by byte-offset; the manifest must not carry it.
        return STATUS_AMBIGUOUS, None
    return STATUS_OK, meta


def apply_via_meta(pristine_src: str, meta: dict, replacement: str) -> str:
    """Splice ``replacement`` at the manifest byte_offset — the Layer-4.5 op,
    in bytes (md5-stable). The inverse the runtime engine performs."""
    bo = meta["byte_offset"]
    bl = meta["byte_length"]
    b = pristine_src.encode("utf-8")
    return (b[:bo] + replacement.encode("utf-8") + b[bo + bl:]).decode("utf-8")


def verify_roundtrip(pristine_src: str, anchor: str, replacement: str) -> bool:
    """R3 core: the manifest meta must splice BYTE-IDENTICALLY to the inline
    anchor replace. Returns True iff apply-via-manifest == apply-via-inline.

    This is what guarantees switching the runtime from inline-anchor to the
    per-pin manifest cannot change a single byte of any patched file.
    """
    status, meta = classify_anchor(pristine_src, anchor, replacement)
    if status != STATUS_OK or meta is None:
        return False
    via_manifest = apply_via_meta(pristine_src, meta, replacement)
    via_inline = pristine_src.replace(anchor, replacement, 1)
    return via_manifest == via_inline


@dataclass
class GenResult:
    ok: dict[str, dict] = field(default_factory=dict)        # "patch_id::sub" -> meta(+target_rel)
    rej: list[dict] = field(default_factory=list)            # drifted/ambiguous/merged entries
    counts: dict[str, int] = field(default_factory=dict)     # status -> n


def build_pin_manifest(
    read_source: Callable[[str], Optional[str]],
    targets: Optional[list[AnchorTarget]] = None,
    *,
    is_upstream_merged: Optional[Callable[[AnchorTarget, str], bool]] = None,
) -> GenResult:
    """Classify every anchor target against the pristine tree (R1 × R2).

    - ``read_source(target_rel)`` returns the pristine file content (or None if
      the target file is absent at this pin).
    - ``targets`` defaults to the full discovery (iter_anchor_targets) — R1.
    - ``is_upstream_merged(target, content)`` lets the caller mark anchors whose
      content upstream merged (so the per-pin file carries only NOT-merged
      patches, per the operator requirement).
    """
    if targets is None:
        targets = list(iter_anchor_targets())
    result = GenResult()
    _src_cache: dict[str, Optional[str]] = {}

    for t in targets:
        key = f"{t.patch_id}::{t.sub}"
        if t.target_rel not in _src_cache:
            _src_cache[t.target_rel] = read_source(t.target_rel)
        src = _src_cache[t.target_rel]

        if src is None:
            result.rej.append({"key": key, "target_rel": t.target_rel,
                               "status": STATUS_TARGET_MISSING})
            result.counts[STATUS_TARGET_MISSING] = result.counts.get(STATUS_TARGET_MISSING, 0) + 1
            continue

        if is_upstream_merged is not None and is_upstream_merged(t, src):
            result.rej.append({"key": key, "target_rel": t.target_rel,
                               "status": STATUS_UPSTREAM_MERGED})
            result.counts[STATUS_UPSTREAM_MERGED] = result.counts.get(STATUS_UPSTREAM_MERGED, 0) + 1
            continue

        status, meta = classify_anchor(src, t.anchor, t.replacement)
        if status == STATUS_OK:
            entry = dict(meta)
            entry["target_rel"] = t.target_rel
            result.ok[key] = entry
        else:
            result.rej.append({"key": key, "target_rel": t.target_rel,
                               "status": status, "anchor_head": t.anchor[:60]})
        result.counts[status] = result.counts.get(status, 0) + 1

    return result
