#!/usr/bin/env python3
"""Phase 4 — compare two anchor manifests ignoring volatile metadata.

The substantive content (files / patches / anchors / pins) is what must be stable;
``generated_at`` and ``generated_by`` change every run and are excluded. Exit 0 if
the manifests are substantively identical, 1 if they drift (and print what moved).

Usage: compare_manifest.py <committed.json> <fresh.json>
"""
import json
import sys

VOLATILE = ("generated_at", "generated_by")


def _load(path):
    m = json.load(open(path))
    return {k: v for k, v in m.items() if k not in VOLATILE}


def main():
    a_path, b_path = sys.argv[1:3]
    a, b = _load(a_path), _load(b_path)
    if json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True):
        print("MATCH: substantively identical (ignoring %s)" % (VOLATILE,))
        return 0

    print("DRIFT: %s != %s" % (a_path, b_path))
    fa, fb = a.get("files", {}), b.get("files", {})
    only_a = sorted(set(fa) - set(fb))
    only_b = sorted(set(fb) - set(fa))
    if only_a:
        print("  only in committed: %s" % only_a[:10])
    if only_b:
        print("  only in fresh:     %s" % only_b[:10])
    for f in sorted(set(fa) & set(fb)):
        if json.dumps(fa[f], sort_keys=True) != json.dumps(fb[f], sort_keys=True):
            print("  changed: %s" % f)
    if a.get("pins") != b.get("pins"):
        print("  pins: committed=%s fresh=%s" % (a.get("pins"), b.get("pins")))
    return 1


if __name__ == "__main__":
    sys.exit(main())
