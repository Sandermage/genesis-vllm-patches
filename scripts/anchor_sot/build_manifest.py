#!/usr/bin/env python3
"""Phase 4 step 3 — classify + write the per-pin anchor manifest (host-runnable).

Reads the discovery targets (from the running container) + the pristine source
(from a bare image), classifies every anchor against the REAL pristine source
(R2), round-trip-verifies the ok set (R3), and writes
``sndr/engines/vllm/pins/<pin>/anchors.json`` (engine schema, validated) +
``drift.rej.json``. Needs sndr on the path; does NOT need vLLM (targets are
passed in), so it runs on the host where the manifest is committed.

Exit codes: 0 ok · 2 round-trip failure · 3 schema invalid.

Usage: build_manifest.py <targets.json> <pristine.json> <repo_root> <vllm_pin> [genesis_pin]
"""
import json
import os
import sys

from sndr.engines.vllm.anchor_discovery import AnchorTarget
from sndr.engines.vllm.anchor_manifest_gen import (
    build_pin_manifest,
    to_engine_manifest,
    verify_roundtrip,
)
from sndr.engines.vllm.wiring.anchor_manifest import (
    normalize_pin,
    validate_manifest_schema,
)


def _mk(d):
    d = dict(d)
    vr = d.get("vllm_version_range")
    d["vllm_version_range"] = tuple(vr) if vr else None
    d["upstream_merged_markers"] = tuple(d.get("upstream_merged_markers") or ())
    return AnchorTarget(**d)


def _dependency_breakage(rej):
    """Run the retire-impact detector against the live registry, feeding the
    per-pin version-gated-OUT set derived from this pin's reject entries.

    Returns the report dict (high_count / medium_count / edges), or an empty
    report when the dispatcher registry is not importable on the host (the
    section is still present so consumers can rely on the key existing).
    """
    try:
        from sndr.engines.vllm.retire_impact import detect_on_live_registry
    except Exception:  # noqa: BLE001 — registry unavailable: empty section
        return {"high_count": 0, "medium_count": 0, "edges": []}
    # patch ids whose anchors are absent on THIS pin because the version range
    # gates them out — treated as "retired on this pin" for breakage analysis
    # (their dependents will skip exactly as if the patch were retired).
    gated = {
        e["key"].split("::", 1)[0]
        for e in rej if e.get("status") == "version_gated"
    }
    return detect_on_live_registry(gated_out=gated).to_dict()


def main():
    targets_path, pristine_path, repo, pin = sys.argv[1:5]
    gpin = sys.argv[5] if len(sys.argv) > 5 else "genesis"

    targets = [_mk(d) for d in json.load(open(targets_path))["targets"]]
    pristine = json.load(open(pristine_path))["files"]
    read = lambda rel: pristine.get(rel)

    res = build_pin_manifest(read, targets, pin=pin)

    rt_fail = []
    for t in targets:
        key = "%s::%s" % (t.patch_id, t.sub)
        if key in res.ok and t.replacement is not None:
            src = pristine.get(t.target_rel)
            if not (src and verify_roundtrip(src, t.anchor, t.replacement)):
                rt_fail.append(key)
    if rt_fail:
        print("FATAL: round-trip failed for %s" % rt_fail[:10], file=sys.stderr)
        sys.exit(2)

    manifest = to_engine_manifest(res, read, vllm_pin=pin, genesis_pin=gpin)
    errors = validate_manifest_schema(manifest)
    if errors:
        print("FATAL: schema invalid: %s" % errors[:5], file=sys.stderr)
        sys.exit(3)

    # Coverage assertion (TASK 4): every discovered target must land EXACTLY
    # once in ok or rej — no silent loss. A drift.rej.json that doesn't account
    # for every dropped anchor hides which patches were dropped.
    discovered = len(targets)
    accounted = len(res.ok) + len(res.rej)
    if accounted != discovered:
        print("FATAL: coverage mismatch — discovered=%d but ok=%d + rejected=%d = %d"
              % (discovered, len(res.ok), len(res.rej), accounted), file=sys.stderr)
        sys.exit(4)

    norm = normalize_pin(pin) or pin.replace("+", "_")
    pindir = os.path.join(repo, "sndr/engines/vllm/pins", norm)
    os.makedirs(pindir, exist_ok=True)
    json.dump(manifest, open(os.path.join(pindir, "anchors.json"), "w"),
              indent=1, sort_keys=True)
    # genuine_anchor_drift is the human re-anchor backlog. It is EXACTLY the
    # `anchor_drift` status — retired patches (status `retired`) are absent here
    # by construction: a retired patch's anchor legitimately drifted and must
    # never be re-anchored, so it is never a false re-anchor candidate.
    genuine = [e for e in res.rej if e.get("status") == "anchor_drift"]
    retired = [e for e in res.rej if e.get("status") == "retired"]
    # Retire-impact / dependency-breakage section: for every patch RETIRED or
    # version-gated-OUT on this pin, which OTHER (live) patch silently breaks
    # because it depends on it (requires_patches / composes_with / anchor name).
    # This is the class of bug that slipped through on dev148->dev301: PN353A
    # retired -> PN399 SKIPPED as a benign skip (NOT genuine anchor_drift) ->
    # its perf optimization no-op'd -> a real TPS regression no gate caught. A
    # perf-tier dependent that breaks is HIGH severity.
    breakage = _dependency_breakage(res.rej)
    json.dump({
        "pin": pin,
        "genesis_pin": gpin,
        "coverage": {"discovered": discovered, "ok": len(res.ok),
                     "rejected": len(res.rej)},
        "counts": dict(res.counts),
        "merge_status": res.merge,
        "rejected": res.rej,
        "genuine_anchor_drift": genuine,
        "dependency_breakage": breakage,
    }, open(os.path.join(pindir, "drift.rej.json"), "w"), indent=1, sort_keys=True)

    print("OK pin=%s -> %s/anchors.json (%d anchors, %d files)" % (
        pin, pindir, len(res.ok), len(manifest["files"])))
    print("coverage: discovered=%d == ok=%d + rejected=%d" % (
        discovered, len(res.ok), len(res.rej)))
    print("counts=%s  roundtrip_fail=0  genuine_drift=%d %s" % (
        dict(res.counts), len(genuine), [e["key"] for e in genuine[:8]]))
    print("retired=%d (anchors gone, as expected) %s" % (
        len(retired), [e["key"] for e in retired[:8]]))
    hi = breakage.get("high_count", 0)
    md = breakage.get("medium_count", 0)
    # APPLY-STATE-AWARE: a mitigated HIGH (dependent has a working fallback anchor
    # independent of the retired id — the PN399 native-form sibling) is handled,
    # so it is surfaced but does NOT trigger the WARN-run-A/B block.
    unmit = [e for e in breakage.get("edges", [])
             if e.get("severity") == "HIGH" and not e.get("mitigated")]
    mit = [e for e in breakage.get("edges", [])
           if e.get("severity") == "HIGH" and e.get("mitigated")]
    if hi or md:
        print("dependency_breakage: HIGH=%d [mitigated=%d unmitigated=%d] "
              "MEDIUM=%d %s" % (
                  hi, len(mit), len(unmit), md,
                  ["%s->%s" % (e["retired"], e["dependent"])
                   for e in breakage["edges"][:8]]))
        if mit:
            print("  NOTE retire-broken PERF dependents already MITIGATED "
                  "(working fallback anchor; surfaced for visibility):")
            for e in mit:
                print("    * %s" % e["detail"])
        if unmit:
            print("  WARN UNMITIGATED retire-broken PERF dependents (run a "
                  "canonical A/B before promoting — iron-rule #9):")
            for e in unmit:
                print("    * %s" % e["detail"])


if __name__ == "__main__":
    main()
