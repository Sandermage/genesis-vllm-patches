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

    norm = normalize_pin(pin) or pin.replace("+", "_")
    pindir = os.path.join(repo, "sndr/engines/vllm/pins", norm)
    os.makedirs(pindir, exist_ok=True)
    json.dump(manifest, open(os.path.join(pindir, "anchors.json"), "w"),
              indent=1, sort_keys=True)
    genuine = [e for e in res.rej if e.get("status") == "anchor_drift"]
    json.dump({"pin": pin, "counts": dict(res.counts), "genuine_anchor_drift": genuine},
              open(os.path.join(pindir, "drift.rej.json"), "w"), indent=1, sort_keys=True)

    print("OK pin=%s -> %s/anchors.json (%d anchors, %d files)" % (
        pin, pindir, len(res.ok), len(manifest["files"])))
    print("counts=%s  roundtrip_fail=0  genuine_drift=%d %s" % (
        dict(res.counts), len(genuine), [e["key"] for e in genuine[:8]]))


if __name__ == "__main__":
    main()
