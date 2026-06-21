#!/usr/bin/env python3
"""Phase 4 step 2 — dump pristine source (run IN a BARE pinned image, no Genesis apply).

The running container's vLLM files are patched in-place at boot, so the only place
to read the un-patched source for the SAME pin is a fresh bare container (Genesis
mounted but never registered → apply never runs). Reads the discovery targets and
emits ``{pin, files:{rel: source}}`` for every unique target file.

Usage: pristine_dump.py <targets.json> <out.json>
"""
import json
import os
import sys

import vllm


def main():
    targets_path, out = sys.argv[1:3]
    root = os.path.dirname(vllm.__file__)
    targets = json.load(open(targets_path))["targets"]
    rels = sorted({t["target_rel"] for t in targets})

    files = {}
    missing = []
    for rel in rels:
        p = os.path.join(root, rel)
        try:
            files[rel] = open(p, encoding="utf-8").read()
        except Exception:
            files[rel] = None
            missing.append(rel)

    with open(out, "w", encoding="utf-8") as f:
        json.dump({"pin": vllm.__version__, "files": files}, f)
    print("pristine: pin=%s files=%d missing=%d -> %s" % (
        vllm.__version__, len(files) - len(missing), len(missing), out))
    if missing:
        print("  target_missing (expected for version-gated): %s" % missing[:8])


if __name__ == "__main__":
    main()
