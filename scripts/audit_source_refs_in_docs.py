#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""audit_source_refs_in_docs.py — docs ↔ source path integrity (forward
direction). Phase 10.5 E-extension 2026-06-01.

Catches stale ``vllm/sndr_core/<path>.{py,yaml}`` references that live
in tracked markdown docs but point at a source file that does not
exist in the tracked tree.

Why this gate exists
--------------------

The companion ``audit_docs_refs_in_source.py`` gate walks Python
source for ``docs/<name>.md`` references and verifies the docs exist.
The companion ``audit_links.py`` gate walks markdown for inline
``[text](target)`` links. Neither covers the third class of drift:
**bare ``vllm/sndr_core/<path>`` mentions in markdown prose / code
spans**, which is exactly how five refs across CONTRIBUTING.md /
GLOSSARY.md / MODEL_CONFIG_LAUNCHER.md / RELEASE_POLICY.md silently
drifted after the v11.0.0 module reorganisation:

  - ``vllm/sndr_core/core.py`` — should be ``core/text_patch.py``
  - ``vllm/sndr_core/cli/doctor.py`` — should be the legacy bridge
    at ``compat/cli.py`` (the basic ``sndr doctor`` lives there)
  - ``vllm/sndr_core/dispatcher.py`` — should be the package dir
    ``dispatcher/`` + boot loop at ``apply/orchestrator.py``
  - ``vllm/sndr_core/guards.py`` — should be ``detection/guards.py``
  - ``vllm/sndr_core/audit/release_check.py`` — should be
    ``proof/release_check.py``

How the gate works
------------------

  1. Walks every ``.md`` file tracked under ``docs/`` and ``README.md``.
  2. Scans for token pattern matching ``vllm/sndr_core/[\\w./-]+\\.
     (py|yaml)`` (any depth, any module name terminated by .py/.yaml).
  3. For each unique ref, asserts the path resolves to a tracked file.
  4. Honors an inline ``<!-- audit-source-refs: allow -->`` marker
     for historical references intentionally pointing at non-existent
     paths — e.g. ``git checkout <historical_sha> -- <pre-sunset V1
     YAML>`` recipes in rollback playbooks. The marker must appear on
     the SAME line as the offending ref.

Exit codes:
  0 — every ref resolves (or is explicitly allowlisted)
  1 — at least one unresolved ref
  2 — internal error / git not available

Modes:

  python3 scripts/audit_source_refs_in_docs.py            # gating
  python3 scripts/audit_source_refs_in_docs.py --json     # machine-readable
  python3 scripts/audit_source_refs_in_docs.py --warn     # informational only
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_REF_RE = re.compile(r"vllm/sndr_core/[A-Za-z0-9_./-]+\.(py|yaml)\b")
_ALLOW_MARKER = "<!-- audit-source-refs: allow -->"


def _tracked_docs() -> list[Path]:
    """List markdown files we should scan (tracked docs + README).

    ``docs/superpowers/`` is excluded: v12 maintainer journals / specs
    / ops playbooks are historical session logs whose subject matter
    includes pre-v12 paths (e.g. the v12 move mapping tables) — not
    operator-facing docs this gate polices. Mirrors ALLOWLIST_PREFIXES
    in scripts/audit_public_docs.py.
    """
    out = subprocess.run(
        ["git", "ls-files", "docs/", "README.md"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        raise SystemExit(f"git ls-files failed: {out.stderr.strip()}")
    return [
        REPO_ROOT / line
        for line in out.stdout.split()
        if line.endswith(".md")
        and not line.startswith("docs/superpowers/")
    ]


def _strip_fenced_blocks(text: str) -> str:
    """Replace fenced code blocks with newline-preserving blanks.

    Fenced blocks (```…``` and ~~~…~~~) hold teaching examples,
    git/shell recipes, and historical context that often intentionally
    quote stale or hypothetical paths (e.g. the rollback recipe
    ``git checkout <historical_sha> -- <pre-sunset V1 YAML>`` in
    TROUBLESHOOTING.md R-001). Inline backtick spans (`…`) on prose
    lines are NOT stripped — they're the canonical citation form for
    a live path and the primary surface this gate exists to police.

    Line numbers are preserved by mapping each stripped char to "\n"
    only when it was already a "\n", and to " " otherwise.
    """
    pat = re.compile(r"(?ms)^(```|~~~).*?^\1\s*$")
    out: list[str] = []
    last = 0
    for m in pat.finditer(text):
        out.append(text[last:m.start()])
        # Preserve newlines so line numbers stay aligned
        out.append("".join("\n" if c == "\n" else " " for c in m.group(0)))
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def _scan(doc: Path) -> list[dict]:
    """Return one record per source ref found in `doc`."""
    raw = doc.read_text(encoding="utf-8", errors="replace")
    text = _strip_fenced_blocks(raw)
    records: list[dict] = []
    seen_on_doc: set[str] = set()
    for m in _REF_RE.finditer(text):
        ref = m.group(0)
        # Each distinct ref on each doc reported once (multiple
        # occurrences of the same broken ref on the same doc would
        # produce noise).
        if ref in seen_on_doc:
            continue
        seen_on_doc.add(ref)
        line_no = text.count("\n", 0, m.start()) + 1
        # Re-read the raw text at the same line for the allow marker
        # check (the stripping pass preserves line numbers).
        raw_lines = raw.split("\n")
        line = raw_lines[line_no - 1] if line_no <= len(raw_lines) else ""
        allowed = _ALLOW_MARKER in line
        exists = (REPO_ROOT / ref).is_file()
        records.append({
            "doc": doc.relative_to(REPO_ROOT).as_posix(),
            "line": line_no,
            "ref": ref,
            "exists": exists,
            "allowed": allowed,
        })
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON report")
    ap.add_argument("--warn", action="store_true",
                    help="informational only — never exit non-zero on broken")
    args = ap.parse_args()

    all_records: list[dict] = []
    for doc in _tracked_docs():
        all_records.extend(_scan(doc))

    broken = [r for r in all_records if not r["exists"] and not r["allowed"]]
    allowlisted = [r for r in all_records if not r["exists"] and r["allowed"]]
    total_refs = len(all_records)

    if args.json:
        result = {
            "total_refs": total_refs,
            "broken": broken,
            "allowlisted": allowlisted,
            "status": "FAIL" if broken else "OK",
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"audit_source_refs_in_docs: {total_refs} unique refs scanned")
        if not broken:
            print(f"✓ all refs resolve (or are explicitly allowlisted: "
                  f"{len(allowlisted)})")
        else:
            print(f"✗ {len(broken)} broken ref(s):")
            for r in broken:
                print(f"  {r['doc']}:{r['line']}  -> {r['ref']}")
            if allowlisted:
                print(f"\n  ({len(allowlisted)} explicitly allowlisted via "
                      f"`{_ALLOW_MARKER}` — skipped)")

    if broken and not args.warn:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
