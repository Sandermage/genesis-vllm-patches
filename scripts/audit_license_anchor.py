#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Release-tier gate: the live `_TRUST_ANCHOR_PUBKEY_B64URL` must NOT
match any of the development-only fingerprints listed in
`_DEV_ANCHOR_FINGERPRINT_FORBIDDEN`.

The development anchor that ships in the public repo has its private
key on the homelab; signing with it does not establish customer-tier
trust. Before flipping a build to public release the maintainer MUST
run the offline trust-anchor ceremony which rotates the constant in
`vllm/sndr_core/license.py` to a production anchor. This audit fails
the build until that rotation happens, so a stray release does not
ship a dev key.

Modes:
  python3 scripts/audit_license_anchor.py            # dev-tier (warn-only)
  python3 scripts/audit_license_anchor.py --release  # release-tier (fail)

Exit codes:
  0 — anchor is OK (or dev-tier with a warning)
  1 — dev anchor detected in --release mode
  2 — internal error (import failure / module shape changed)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load() -> tuple[str, frozenset[str]]:
    try:
        from sndr.license import (
            _TRUST_ANCHOR_PUBKEY_B64URL as anchor,
            _DEV_ANCHOR_FINGERPRINT_FORBIDDEN as forbidden,
        )
    except (ImportError, AttributeError) as e:
        print(f"audit-license-anchor: cannot read license module — {e}", file=sys.stderr)
        sys.exit(2)
    return anchor, forbidden


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--release", action="store_true",
        help="Strict release-tier check — fail when the anchor still "
             "matches the dev fingerprint. Without --release this is a "
             "warning-only check so dev builds stay unblocked.",
    )
    args = ap.parse_args(argv)
    anchor, forbidden = _load()

    if anchor in forbidden:
        msg = (
            f"audit-license-anchor: development-only trust anchor in use "
            f"({anchor!r}). Rotate via the offline trust-anchor "
            f"ceremony before public release."
        )
        if args.release:
            print(msg, file=sys.stderr)
            return 1
        print(f"WARNING: {msg}")
        return 0

    print(f"audit-license-anchor: production anchor active ({anchor[:8]}…) — OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
