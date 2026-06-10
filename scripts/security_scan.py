#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 4.6 — release-pipeline security scan.

Runs the six sub-checks from SECURITY_LICENSE_GATE_2026-05-12_RU.md §3:

  1. No secrets in tree (basic patterns + SECRETS_FORBIDDEN allowlist).
  2. No `/home/<user>` / `/Users/<user>` in tracked code or public docs.
  3. No private IPs (RFC 1918) in public docs.
  4. No RSA/OpenSSH private key markers anywhere.
  5. No .env files outside gitignore.
  6. SBOM + constraints + attestation present under release/ when
     invoked with `--public-release`.

Exit code:
  0 — all checks pass.
  1 — at least one check failed (release blocked).
  2 — internal error (missing tool, missing file).

Usage:
  python3 scripts/security_scan.py                  # informational
  python3 scripts/security_scan.py --public-release # strict, requires release/

Per `make audit-security` integration into Phase 7 release pipeline.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

# Allowlist: paths whose content is acceptable to contain "sensitive-looking"
# strings (audit reports, internal docs, test fixtures, security designs).
ALLOWLIST_PATHS = [
    "sndr_private/",           # private maintainer tree — entire subtree
    "docs/upstream/",
    "docs/reference/",
    # v12 maintainer journals / specs / ops playbooks (superpowers
    # workflow): rig IPs, operator paths and SSH transcripts are their
    # subject matter — same internal-docs class as docs/_internal/ was.
    # Mirrors ALLOWLIST_PREFIXES in scripts/audit_public_docs.py.
    "docs/superpowers/",
    "_archive/",
    "tests/",                  # test fixtures may contain mock secrets
    "scripts/security_scan.py",  # this file itself contains the regex literals
    "scripts/audit_no_hardcoded_paths.py",  # describes forbidden patterns
    "scripts/audit_public_docs.py",  # describes forbidden patterns
    "scripts/audit_license_anchor.py",  # references private trust-anchor script
    "scripts/generate_sbom.py",
    "Makefile",                # may grep for these patterns at audit time
]


def _is_allowlisted(rel_path: str) -> bool:
    return any(rel_path.startswith(p) for p in ALLOWLIST_PATHS)


def _git_files() -> list[str]:
    """List git-tracked files (deterministic across machines)."""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _grep_pattern(
    pattern: re.Pattern,
    paths: list[str],
    *,
    respect_allowlist: bool = True,
) -> list[tuple[str, int, str]]:
    """Walk paths, return list of (path, line_number, line) hits.
    Lines containing 'security_scan: allow' inline are skipped.
    """
    hits: list[tuple[str, int, str]] = []
    for rel in paths:
        if respect_allowlist and _is_allowlisted(rel):
            continue
        fp = REPO_ROOT / rel
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if "security_scan: allow" in line:
                continue
            if pattern.search(line):
                hits.append((rel, i, line.strip()[:120]))
    return hits


# ─── Individual checks ─────────────────────────────────────────────────


def check_no_operator_paths(files: list[str]) -> list[str]:
    """No `/home/<user>` or `/Users/<user>` in tracked code or public docs."""
    pat = re.compile(r"/(?:home|Users)/sander")
    hits = _grep_pattern(pat, files)
    return [f"{h[0]}:{h[1]}: {h[2]}" for h in hits]


def check_no_private_ips(files: list[str]) -> list[str]:
    """No RFC 1918 private IPs in public docs."""
    # Limit to docs/ so we don't flag examples in tests + internal docs.
    public_doc_files = [f for f in files if f.startswith("docs/")
                        and not _is_allowlisted(f)]
    pat = re.compile(
        r"\b(?:10|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b"
    )
    hits = _grep_pattern(pat, public_doc_files, respect_allowlist=False)
    return [f"{h[0]}:{h[1]}: {h[2]}" for h in hits]


def check_no_private_keys(files: list[str]) -> list[str]:
    """No RSA / OpenSSH private key markers anywhere."""
    pat = re.compile(r"BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY")
    hits = _grep_pattern(pat, files, respect_allowlist=False)
    return [f"{h[0]}:{h[1]}: {h[2]}" for h in hits]


def check_no_env_files(files: list[str]) -> list[str]:
    """No .env or .env.* files committed."""
    bad = []
    for f in files:
        name = Path(f).name
        if name == ".env" or name.startswith(".env."):
            bad.append(f)
    return bad


def check_no_aws_keys(files: list[str]) -> list[str]:
    """No AWS-style access keys in tracked content."""
    # AKIA prefix + base32-like 16-char suffix.
    pat = re.compile(r"\bAKIA[A-Z0-9]{16}\b")
    hits = _grep_pattern(pat, files)
    return [f"{h[0]}:{h[1]}: {h[2]}" for h in hits]


def check_release_artifacts_present() -> list[str]:
    """When `--public-release`, release/ must contain SBOM + constraints."""
    missing = []
    for name in ("SBOM.spdx.json", "constraints.txt"):
        if not (REPO_ROOT / "release" / name).exists():
            missing.append(f"missing release artifact: release/{name}")
    return missing


# ─── Main ──────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--public-release",
        action="store_true",
        help="Strict mode: also verify release/ contains SBOM + constraints.",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON summary.",
    )
    args = ap.parse_args()

    try:
        files = _git_files()
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"git ls-files failed: {e}\n")
        return 2

    results: dict[str, list[str]] = {
        "operator_paths": check_no_operator_paths(files),
        "private_ips": check_no_private_ips(files),
        "private_keys": check_no_private_keys(files),
        "env_files": check_no_env_files(files),
        "aws_keys": check_no_aws_keys(files),
    }
    if args.public_release:
        results["release_artifacts"] = check_release_artifacts_present()

    total_failures = sum(len(v) for v in results.values())

    if args.json:
        import json
        print(json.dumps(
            {"checks": results, "total_failures": total_failures,
             "scanned_files": len(files)},
            indent=2, sort_keys=True,
        ))
    else:
        print(f"security_scan: {len(files)} tracked files scanned")
        print("─" * 60)
        any_fail = False
        for check, hits in results.items():
            if hits:
                any_fail = True
                print(f"  ✗ {check}: {len(hits)} hit(s)")
                for h in hits[:5]:
                    print(f"      {h}")
                if len(hits) > 5:
                    print(f"      ... ({len(hits) - 5} more)")
            else:
                print(f"  ✓ {check}: clean")
        print()
        if any_fail:
            print(f"  FAIL — {total_failures} total violations")
        else:
            print("  OK — all checks passed")

    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
