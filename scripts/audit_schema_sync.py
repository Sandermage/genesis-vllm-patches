#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""audit_schema_sync.py — assert the two patch-entry schemas stay byte-identical.

The patch-entry JSON schema lives in two places:

  1. `sndr/schemas/patch_entry.schema.json` — canonical package copy
     (v12 tree; was `vllm/sndr_core/schemas/`), shipped in the wheel.
     `compat/schema_validator.py` loads this first via
     `importlib.resources.files()`.
  2. `schemas/patch_entry.schema.json` — repo-root duplicate, retained for
     v10.x callers and pre-package-data tooling that walks the source tree.

Audit `CURRENT_PROJECT_RECHECK_CLEANUP_ERRORS_2026-05-14_eaa44975_RU.md`
P0-3 documented the two files silently diverging (root had three newer
fields, package had four other newer fields). This gate runs `cmp` over
the two paths and fails when they differ.

Synchronization policy: edit the **package** file; run
`cp sndr/schemas/patch_entry.schema.json schemas/patch_entry.schema.json`
to mirror it. This gate enforces that step before commit.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL = REPO_ROOT / "sndr" / "schemas" / "patch_entry.schema.json"
MIRROR = REPO_ROOT / "schemas" / "patch_entry.schema.json"


def main() -> int:
    print("=== audit_schema_sync.py ===")
    if not CANONICAL.exists():
        print(f"✗ canonical schema missing: {CANONICAL.relative_to(REPO_ROOT)}")
        return 1
    if not MIRROR.exists():
        print(f"✗ root mirror schema missing: {MIRROR.relative_to(REPO_ROOT)}")
        return 1

    canonical_bytes = CANONICAL.read_bytes()
    mirror_bytes = MIRROR.read_bytes()

    if canonical_bytes == mirror_bytes:
        print(
            f"✓ schemas in sync ({len(canonical_bytes)} bytes): "
            f"{CANONICAL.relative_to(REPO_ROOT)} == {MIRROR.relative_to(REPO_ROOT)}"
        )
        return 0

    print("✗ schemas differ:")
    print(f"    canonical: {CANONICAL.relative_to(REPO_ROOT)} ({len(canonical_bytes)} bytes)")
    print(f"    mirror:    {MIRROR.relative_to(REPO_ROOT)} ({len(mirror_bytes)} bytes)")
    print()
    print("Fix:")
    print("  Edit the canonical (package) file, then mirror it:")
    print(f"    cp {CANONICAL.relative_to(REPO_ROOT)} {MIRROR.relative_to(REPO_ROOT)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
