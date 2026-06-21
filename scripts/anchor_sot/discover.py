#!/usr/bin/env python3
"""Phase 4 step 1 — discover anchor targets (run IN the running pinned container).

The running container builds the full set of patchers with the real runtime env
(a bare container builds fewer — 8 patchers gate on detected hardware/config), so
discovery MUST run here to get the canonical anchor set. Emits a JSON envelope
``{pin, genesis_pin, targets:[AnchorTarget...]}`` consumed by build_manifest.py.

Usage: discover.py <out.json>
"""
import dataclasses
import json
import sys

import vllm

from sndr.engines.vllm.anchor_discovery import iter_anchor_targets


def _genesis_pin():
    for getter in (
        lambda: __import__("sndr").__version__,
        lambda: __import__("sndr.version", fromlist=["__version__"]).__version__,
        lambda: __import__("importlib.metadata", fromlist=["version"]).version("sndr"),
    ):
        try:
            v = getter()
            if v:
                return v
        except Exception:
            continue
    return "genesis"


def main():
    out = sys.argv[1]
    targets = [dataclasses.asdict(t) for t in iter_anchor_targets()]
    envelope = {
        "pin": vllm.__version__,
        "genesis_pin": _genesis_pin(),
        "targets": targets,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(envelope, f)
    print("discover: pin=%s genesis=%s targets=%d -> %s" % (
        envelope["pin"], envelope["genesis_pin"], len(targets), out))


if __name__ == "__main__":
    main()
