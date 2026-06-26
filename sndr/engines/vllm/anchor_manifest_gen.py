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

Implements Phase 2 of the per-pin anchor source-of-truth design.
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
# A retired (superseded / upstream-absorbed) patch. Its anchor legitimately no
# longer matches the dev source — it must NEVER be re-anchored and NEVER land in
# the applied `ok` manifest. Routed here REGARDLESS of whether its anchor still
# matches, so a retired patch is never counted as genuine anchor_drift (false
# re-anchor backlog). Kept VISIBLE in the reject set so the operator can SEE
# "N retired patches' anchors are gone, as expected".
STATUS_RETIRED = "retired"

# Per-PATCH upstream-merge tri-state recorded IN the manifest (the operator's
# explicit ask). Distinct from the per-sub STATUS_UPSTREAM_MERGED above: this
# aggregates a patch's sub-patches so a pin-switch SEES whether the whole patch
# was upstreamed (skip it) or only partly (apply the remaining anchors).
MERGE_NOT_MERGED = "not_merged"          # no sub's markers fired -> still ours
MERGE_FULLY_MERGED = "fully_merged"      # ALL subs upstreamed -> patch is moot
MERGE_PARTIALLY_MERGED = "partially_merged"  # SOME subs upstreamed
MERGE_STATUSES = (MERGE_NOT_MERGED, MERGE_FULLY_MERGED, MERGE_PARTIALLY_MERGED)


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
    # Per-PATCH upstream-merge tri-state (TASK 1). patch_id -> {
    #   "merge_status": MERGE_*,
    #   "target_rel": rel,                # the file the patch targets
    #   "merged_subs": [sub, ...],        # subs whose markers fired (sorted)
    # }. Recorded for EVERY patch seen so a pin-switch can SEE a patch that
    # became fully_merged even when no anchors remain to splice.
    merge: dict[str, dict] = field(default_factory=dict)


def aggregate_merge_status(
    total_subs: int, merged_subs: set[str]
) -> str:
    """Aggregate a patch's per-sub upstream-merge signal into the tri-state.

    ``total_subs`` is the count of the patch's anchor-bearing sub-patches seen
    on this pin (present target file, not version-gated); ``merged_subs`` is the
    subset whose ``upstream_merged_markers`` fired in the pristine source.

    - none merged              -> MERGE_NOT_MERGED
    - all merged (and at least one) -> MERGE_FULLY_MERGED
    - some but not all merged  -> MERGE_PARTIALLY_MERGED
    """
    n_merged = len(merged_subs)
    if n_merged == 0:
        return MERGE_NOT_MERGED
    if total_subs > 0 and n_merged >= total_subs:
        return MERGE_FULLY_MERGED
    return MERGE_PARTIALLY_MERGED


def to_engine_manifest(
    res: "GenResult",
    pristine: Callable[[str], Optional[str]],
    *,
    vllm_pin: str,
    genesis_pin: str,
) -> dict:
    """Convert the classified ``ok`` set into the EXISTING engine manifest
    schema (files -> rel -> {md5_pristine, size_bytes, patches -> pid ->
    {merge_status, anchors -> sub -> meta}}), so the runtime Layer-4.5 loads it
    unchanged (it reads ``patches.<pid>.anchors`` and ignores the sibling
    ``merge_status``). ``pristine(rel)`` returns the pristine source for the
    per-file md5/size.

    TASK 1: every patch carries a ``merge_status`` tri-state (not_merged /
    fully_merged / partially_merged). ``partially_merged`` also carries
    ``merged_subs`` (the sub-patches the apply/operator should SKIP). A
    ``fully_merged`` patch is recorded even with zero anchors left to splice, so
    a pin-switch SEES it became upstreamed and skips it instead of breaking.
    """
    import hashlib
    import time

    from sndr.engines.vllm.wiring.anchor_manifest import MANIFEST_SCHEMA_VERSION

    files: dict[str, dict] = {}
    _META_KEYS = ("anchor_md5", "byte_length", "byte_offset", "replacement_md5")

    def _ensure_file(rel: str) -> dict:
        if rel not in files:
            src = pristine(rel) or ""
            sb = src.encode("utf-8")
            files[rel] = {
                "md5_pristine": hashlib.md5(sb).hexdigest(),
                "size_bytes": len(sb),
                "patches": {},
            }
        return files[rel]

    for key, e in res.ok.items():
        rel = e["target_rel"]
        pid, _, sub = key.partition("::")
        fe = _ensure_file(rel)
        fe["patches"].setdefault(pid, {"merge_status": MERGE_NOT_MERGED,
                                       "anchors": {}})
        fe["patches"][pid]["anchors"][sub] = {
            k: e[k] for k in _META_KEYS if k in e
        }

    # Stamp the per-PATCH merge tri-state. For fully_merged patches with no ok
    # anchors, materialize the file + patch entry (empty anchors) so the patch
    # stays VISIBLE in the manifest rather than vanishing into the rej set.
    for pid, m in res.merge.items():
        rel = m["target_rel"]
        fe = _ensure_file(rel)
        pe = fe["patches"].setdefault(pid, {"merge_status": MERGE_NOT_MERGED,
                                            "anchors": {}})
        pe["merge_status"] = m["merge_status"]
        if m["merge_status"] == MERGE_PARTIALLY_MERGED:
            pe["merged_subs"] = list(m.get("merged_subs", []))
        else:
            pe.pop("merged_subs", None)

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
    "drift"): target_missing → retired (lifecycle=retired) → version_gated
    (range excludes ``pin``) → upstream_merged (a merge marker is present) →
    ok/anchor_drift/ambiguous.

    ``retired`` is checked BEFORE the anchor is even classified: a retired
    patch's anchor legitimately drifted (its code was superseded / absorbed
    upstream), so it is routed to the reject set under STATUS_RETIRED regardless
    of whether the anchor still matches — never the applied ``ok`` manifest,
    never counted as anchor_drift. This keeps genuine_anchor_drift (the
    re-anchor backlog) free of retired patches while keeping them VISIBLE.

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

    # Per-PATCH merge aggregation (TASK 1). For each patch_id present+applicable
    # on this pin: which sub-patches we considered, and which had an upstream
    # merge marker fire. target_rel kept so a fully_merged patch (zero anchors
    # left) still records WHERE it lived.
    _seen_subs: dict[str, set[str]] = {}
    _merged_subs: dict[str, set[str]] = {}
    _merge_target_rel: dict[str, str] = {}

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

        # Retired patch: its anchor legitimately drifted (superseded / absorbed
        # upstream). Route to STATUS_RETIRED REGARDLESS of anchor match — never
        # the applied `ok` set, never anchor_drift. Recorded with anchor_head so
        # the operator still SEES it. Deliberately does NOT feed the merge
        # aggregation below (a retired patch is out of the active set entirely).
        if (t.lifecycle or "").lower() == "retired":
            _rej(key, t, STATUS_RETIRED, anchor_head=t.anchor[:60],
                 required=t.required)
            continue

        if version_excludes_pin(t.vllm_version_range, pin):
            _rej(key, t, STATUS_VERSION_GATED, vrange=list(t.vllm_version_range or ()))
            continue

        # This sub is present + applicable on this pin -> it counts toward the
        # patch's merge aggregation (denominator). version_gated/target_missing
        # subs deliberately do NOT count (they aren't "ours on this pin").
        _seen_subs.setdefault(t.patch_id, set()).add(t.sub)
        _merge_target_rel.setdefault(t.patch_id, t.target_rel)

        merged = any(m in src for m in (t.upstream_merged_markers or ())) or (
            is_upstream_merged is not None and is_upstream_merged(t, src)
        )
        if merged:
            _merged_subs.setdefault(t.patch_id, set()).add(t.sub)
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

    # Roll up the per-patch merge tri-state. Recorded for EVERY patch that had
    # at least one present+applicable sub (so partially/fully-merged patches are
    # VISIBLE in the manifest instead of being silently dropped to the rej set).
    for pid, subs in _seen_subs.items():
        merged = _merged_subs.get(pid, set())
        status = aggregate_merge_status(len(subs), merged)
        entry = {
            "merge_status": status,
            "target_rel": _merge_target_rel[pid],
        }
        if status == MERGE_PARTIALLY_MERGED:
            entry["merged_subs"] = sorted(merged)
        result.merge[pid] = entry

    return result
