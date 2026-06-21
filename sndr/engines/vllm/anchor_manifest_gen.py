"""Phase 2 — per-pin anchor manifest generator + true-drift classifier (R2).

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

See docs/superpowers/specs/2026-06-21-anchor-sot-design.md (Phase 2).
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
STATUS_VERSION_GATED = "version_gated"
STATUS_OPTIONAL_ABSENT = "optional_absent"  # absent but required=False -> not drift
STATUS_TARGET_MISSING = "target_missing"


def version_excludes_pin(vrange, pin: Optional[str]) -> bool:
    """True iff the patch's vllm_version_range EXCLUDES ``pin`` — then an absent
    anchor is EXPECTED (the patch isn't for this pin), not genuine drift.

    Mirrors check_upstream_drift._version_gated_out using the engine's version
    checker. No range or no pin → not gated (conservative).
    """
    if not vrange or not pin:
        return False
    try:
        from sndr.compat.version_check import (
            VersionProfile,
            check_version_constraints,
        )
    except Exception:  # noqa: BLE001 — checker unavailable: don't gate
        return False
    vr = list(vrange) if isinstance(vrange, tuple) else vrange
    profile = VersionProfile(vllm=pin)
    ok, _results = check_version_constraints(
        {"vllm_version_range": vr}, profile=profile
    )
    return not ok  # not ok → pin not in range → gated out


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


def to_engine_manifest(
    res: "GenResult",
    pristine: Callable[[str], Optional[str]],
    *,
    vllm_pin: str,
    genesis_pin: str,
) -> dict:
    """Convert the classified ``ok`` set into the EXISTING engine manifest
    schema (files -> rel -> {md5_pristine, size_bytes, patches -> pid ->
    {anchors -> sub -> meta}}), so the runtime Layer-4.5 loads it unchanged.
    ``pristine(rel)`` returns the pristine source for the per-file md5/size.
    """
    import hashlib
    import time

    from sndr.engines.vllm.wiring.anchor_manifest import MANIFEST_SCHEMA_VERSION

    files: dict[str, dict] = {}
    _META_KEYS = ("anchor_md5", "byte_length", "byte_offset", "replacement_md5")
    for key, e in res.ok.items():
        rel = e["target_rel"]
        pid, _, sub = key.partition("::")
        if rel not in files:
            src = pristine(rel) or ""
            sb = src.encode("utf-8")
            files[rel] = {
                "md5_pristine": hashlib.md5(sb).hexdigest(),
                "size_bytes": len(sb),
                "patches": {},
            }
        files[rel]["patches"].setdefault(pid, {"anchors": {}})
        files[rel]["patches"][pid]["anchors"][sub] = {
            k: e[k] for k in _META_KEYS if k in e
        }
    return {
        "manifest_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "sndr.engines.vllm.anchor_manifest_gen.to_engine_manifest",
        "pins": {"vllm": str(vllm_pin), "genesis": str(genesis_pin)},
        "files": files,
    }


def build_pin_manifest(
    read_source: Callable[[str], Optional[str]],
    targets: Optional[list[AnchorTarget]] = None,
    *,
    pin: Optional[str] = None,
    is_upstream_merged: Optional[Callable[[AnchorTarget, str], bool]] = None,
) -> GenResult:
    """Classify every anchor target against the pristine tree (R1 × R2).

    Order (so an absent anchor is split into its TRUE cause, not lumped as
    "drift"): target_missing → version_gated (range excludes ``pin``) →
    upstream_merged (a merge marker is present) → ok/anchor_drift/ambiguous.

    - ``read_source(target_rel)`` returns pristine file content (None if absent).
    - ``targets`` defaults to the full discovery (iter_anchor_targets) — R1.
    - ``pin`` is the target full version (e.g. "0.23.1rc1.dev148+gb4c80ec0f");
      enables the version-gate split.
    - ``is_upstream_merged(target, content)`` is an extra caller hook on top of
      each target's own ``upstream_merged_markers``.
    """
    if targets is None:
        targets = list(iter_anchor_targets())
    result = GenResult()
    _src_cache: dict[str, Optional[str]] = {}

    def _rej(key, t, status, **extra):
        result.rej.append({"key": key, "target_rel": t.target_rel,
                           "status": status, **extra})
        result.counts[status] = result.counts.get(status, 0) + 1

    for t in targets:
        key = f"{t.patch_id}::{t.sub}"
        if t.target_rel not in _src_cache:
            _src_cache[t.target_rel] = read_source(t.target_rel)
        src = _src_cache[t.target_rel]

        if src is None:
            _rej(key, t, STATUS_TARGET_MISSING)
            continue

        if version_excludes_pin(t.vllm_version_range, pin):
            _rej(key, t, STATUS_VERSION_GATED, vrange=list(t.vllm_version_range or ()))
            continue

        merged = any(m in src for m in (t.upstream_merged_markers or ())) or (
            is_upstream_merged is not None and is_upstream_merged(t, src)
        )
        if merged:
            _rej(key, t, STATUS_UPSTREAM_MERGED)
            continue

        status, meta = classify_anchor(src, t.anchor, t.replacement)
        if status == STATUS_OK:
            entry = dict(meta)
            entry["target_rel"] = t.target_rel
            result.ok[key] = entry
            result.counts[STATUS_OK] = result.counts.get(STATUS_OK, 0) + 1
        elif status == STATUS_ANCHOR_DRIFT and not t.required:
            # an absent OPTIONAL sub-patch is a soft-skip at apply time, not
            # drift — mirrors check_upstream_drift (only required anchors drift).
            _rej(key, t, STATUS_OPTIONAL_ABSENT, anchor_head=t.anchor[:60])
        else:
            _rej(key, t, status, anchor_head=t.anchor[:60], required=t.required)

    return result
