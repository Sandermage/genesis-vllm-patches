#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Genesis V2 dirty-state gate — three-tier check (dev / audit / release).

Reads policy from `tools/policies/dirty_state_allowlist.yaml` and walks
`git status --porcelain` against it. Exit code:

    0 — worktree state matches the tier's allowlist (gate passes)
    1 — disallowed untracked or modified file found (gate fails)
    2 — policy file missing or malformed

Usage:
    scripts/check_dirty_state.py --tier dev
    scripts/check_dirty_state.py --tier audit
    scripts/check_dirty_state.py --tier release [--host local|server]

Per PROJECT_ROADMAP_V2 §6.3 gate #6 + LOCAL_SERVER_ALLOWED_DIRTY_STATE.
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.stderr.write("PyYAML required. `pip install pyyaml`.\n")
    sys.exit(2)


REPO_ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = REPO_ROOT / "tools" / "policies" / "dirty_state_allowlist.yaml"


def _load_policy() -> dict:
    if not POLICY_PATH.exists():
        sys.stderr.write(f"policy file not found: {POLICY_PATH}\n")
        sys.exit(2)
    with POLICY_PATH.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        sys.stderr.write(f"policy schema_version mismatch in {POLICY_PATH}\n")
        sys.exit(2)
    return data


def _git_status() -> list[tuple[str, str]]:
    """Returns list of (status_code, path) from `git status --porcelain`.

    status_code is the 2-char XY prefix per git porcelain v1. Examples:
      ' M file.py'   → ('M', 'file.py')          modified, not staged
      'M  file.py'   → ('M', 'file.py')          modified, staged
      'A  file.py'   → ('A', 'file.py')          added, staged
      '?? file.py'   → ('??', 'file.py')         untracked
      'D  file.py'   → ('D', 'file.py')          deleted, staged
    """
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    entries = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        # git status --porcelain prefix is exactly 2 chars + space.
        code = line[:2].strip() or line[:2]
        path = line[3:]
        # rename has "old -> new" form; we treat the new path as the entry.
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        entries.append((code, path))
    return entries


def _matches_any(path: str, patterns: list[str]) -> bool:
    """Match path against fnmatch patterns. Trailing `/**` makes the
    pattern recursive into a directory."""
    for pat in patterns:
        # fnmatch doesn't handle ** specially; convert to a regex.
        regex = re.escape(pat).replace(r"\*\*", ".*").replace(r"\*", "[^/]*")
        if re.fullmatch(regex, path):
            return True
        # Also accept the pattern as a directory prefix (when path is below it).
        if pat.endswith("/**") and path.startswith(pat[:-3] + "/"):
            return True
        if fnmatch.fnmatch(path, pat):
            return True
    return False


def _check_entry(
    code: str,
    path: str,
    tier_policy: dict,
) -> tuple[bool, str]:
    """Returns (allowed, reason). reason describes either why it was
    allowed or why it was rejected — for transparent reporting."""
    is_untracked = code.startswith("?")
    is_modified_tracked = not is_untracked
    forbidden_unt = tier_policy.get("forbidden_untracked", [])
    allow_unt = tier_policy.get("allow_untracked", [])
    forbidden_mod = tier_policy.get("forbidden_tracked_modified", [])

    if is_untracked:
        if _matches_any(path, forbidden_unt):
            return False, f"forbidden untracked pattern: {path}"
        if _matches_any(path, allow_unt):
            return True, f"allowed untracked: {path}"
        return False, f"untracked file not in tier allowlist: {path}"

    if is_modified_tracked:
        if forbidden_mod and _matches_any(path, forbidden_mod):
            return False, f"modified tracked file not allowed on this tier: {path}"
        return True, f"modified tracked allowed on this tier: {path}"

    return True, f"unknown status code {code!r} for {path} (defaulting to allow)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tier",
        choices=["dev", "audit", "release"],
        required=True,
        help="Which tier to enforce.",
    )
    ap.add_argument(
        "--host",
        choices=["local", "server"],
        default="local",
        help="Informational label for evidence ledger.",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON summary instead of human text.",
    )
    args = ap.parse_args()

    policy = _load_policy()
    tier_key = args.tier
    if tier_key not in policy:
        sys.stderr.write(f"tier {tier_key!r} missing from {POLICY_PATH}\n")
        return 2
    tier_policy = policy[tier_key]

    entries = _git_status()

    rejections: list[str] = []
    acceptances: list[str] = []
    for code, path in entries:
        ok, reason = _check_entry(code, path, tier_policy)
        if ok:
            acceptances.append(reason)
        else:
            rejections.append(reason)

    if args.json:
        import json

        print(
            json.dumps(
                {
                    "tier": tier_key,
                    "host": args.host,
                    "total_entries": len(entries),
                    "accepted": len(acceptances),
                    "rejected": len(rejections),
                    "rejections": rejections,
                    "policy_source": str(POLICY_PATH.relative_to(REPO_ROOT)),
                },
                indent=2,
            )
        )
    else:
        print(
            f"check_dirty_state: tier={tier_key} host={args.host} "
            f"entries={len(entries)} accepted={len(acceptances)} "
            f"rejected={len(rejections)}"
        )
        for r in rejections:
            print(f"  REJECT: {r}")
        if not rejections:
            print("  OK — worktree matches tier policy")

    return 0 if not rejections else 1


if __name__ == "__main__":
    sys.exit(main())
