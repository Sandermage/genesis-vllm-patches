#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""generate_patches_md.py — auto-generate full patch table from PATCH_REGISTRY.

Produces machine-derived companion to docs/PATCHES.md, eliminating manual sync drift.
PATCHES.md remains the curated narrative; this script generates the exhaustive
table linked from there.

Background: 2026-05-11 audit found PATCHES.md detailed only 21 of 134 entries.
Manual maintenance impractical at 134 entries × 9 fields. This script reads
registry.py source (regex parse, no torch import) and emits:
  - Statistics block (counts by tier/lifecycle/family/default_on)
  - Full patch table grouped by family, sorted naturally (P1...P107, PN8...PN95)
  - Auto-gen timestamp + verification command

Usage:
    # Generate to docs/PATCHES_AUTO.md (creates / overwrites)
    python3 scripts/generate_patches_md.py

    # Print to stdout (for diff in CI)
    python3 scripts/generate_patches_md.py --stdout

    # Verify committed file is in sync (CI gate)
    python3 scripts/generate_patches_md.py --check    # exit 1 if divergence

Exit codes:
    0 — generated successfully (or --check passed)
    1 — --check failed (committed file out of sync)
    2 — registry.py not parseable

CI integration:
    Add to .github/workflows: `python3 scripts/generate_patches_md.py --check`
    Pre-commit hook: same command
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# v12.x moved the registry to sndr/dispatcher/registry.py; vllm/sndr_core/...
# is now a re-export shim with no literal to parse. Prefer the canonical
# source, fall back to the shim path for older layouts.
REGISTRY_PATH = REPO_ROOT / "sndr" / "dispatcher" / "registry.py"
if not REGISTRY_PATH.is_file():
    REGISTRY_PATH = REPO_ROOT / "sndr" / "dispatcher" / "registry.py"
OUTPUT_PATH = REPO_ROOT / "docs" / "PATCHES_AUTO.md"

# Fields we extract per patch entry
FIELDS = ["title", "tier", "family", "env_flag", "default_on", "lifecycle", "upstream_pr"]


def natural_sort_key(patch_id: str) -> tuple:
    """Natural sort: P1 < P2 < ... < P107 < PN8 < PN9 < ... < PN95."""
    m = re.match(r"^(P|PN|SPRINT)(\d+)([A-Z]*)$", patch_id)
    if not m:
        return (3, patch_id)  # Unknown format last
    prefix, num, suffix = m.group(1), int(m.group(2)), m.group(3)
    prefix_order = {"P": 0, "PN": 1, "SPRINT": 2}.get(prefix, 3)
    return (prefix_order, num, suffix)


# ─── upstream_pr renderer ──────────────────────────────────────────────────


# Matches a GitHub PR/issue URL and captures the (pull|issues) segment plus
# the trailing numeric id. Anchored — extra path/query components reject.
_GITHUB_PR_OR_ISSUE_URL_RE = re.compile(
    r"^https?://github\.com/[\w.-]+/[\w.-]+/(?P<kind>pull|issues)/(?P<num>\d+)/?$"
)


def render_upstream_pr(pr) -> str:
    """Render the registry's ``upstream_pr`` field as a markdown cell.

    Registry has 3 distinct value shapes that this helper dispatches:

      * ``None``               → ``"—"`` (em dash, no reference)
      * ``int`` (numeric PR)   → ``[#N](https://github.com/vllm-project/vllm/pull/N)``
      * ``str`` (full URL)     → if it parses as a GitHub PR or issue URL,
                                 ``[#N](URL)`` with the trailing number lifted
                                 into the link text and the URL preserved
                                 verbatim (pull vs issues distinction kept).
                                 Unknown URL shape falls back to backticks
                                 so the malformed-link bug class
                                 (``[#https://...](https://.../pull/https://...)``)
                                 cannot recur.

    Defensive defaults: any other type (bool, list, dict, etc.) renders
    as ``"—"`` — matches the ``None`` case rather than producing a
    malformed cell. The audit (``audit_generated_links.py``) catches
    that as a follow-up safety net.
    """
    if pr is None:
        return "—"
    if isinstance(pr, bool):
        # bool is a subclass of int — guard separately.
        return "—"
    if isinstance(pr, int):
        return f"[#{pr}](https://github.com/vllm-project/vllm/pull/{pr})"
    if isinstance(pr, str):
        m = _GITHUB_PR_OR_ISSUE_URL_RE.match(pr.strip())
        if m:
            return f"[#{m.group('num')}]({pr.strip()})"
        # Unknown URL shape — render raw URL inside backticks so it is
        # operator-visible without being a malformed markdown link.
        safe = pr.strip().replace("|", "\\|").replace("`", "")
        return f"`{safe}`"
    # Defensive — unknown type.
    return "—"


def parse_registry(registry_path: Path) -> dict[str, dict]:
    """Parse PATCH_REGISTRY entries from registry.py source via regex.

    No torch import — registry.py imports torch indirectly, so we can't exec it.
    We rely on the canonical formatting: each entry is `    "PATCH_ID": {` at
    exactly 4-space indent (verified by check_doc_sync.py).
    """
    if not registry_path.is_file():
        raise FileNotFoundError(f"Registry not found: {registry_path}")
    text = registry_path.read_text()
    entries: dict[str, dict] = {}

    # Find each `    "PATCH_ID": {` line and the matching closing brace `    },`.
    # Keys may contain hyphens (e.g. `PN40-classifier`), so include `-` in the
    # character class. The opening brace can be followed by an inline comment
    # (`# rename note`, `# PN122 rename 2026-05-14`, etc.) so we tolerate
    # trailing whitespace + `#` up to end-of-line. Must mirror
    # scripts/check_doc_sync.py regex.
    entry_pattern = re.compile(
        r"^    \"([A-Za-z0-9_\-]+)\":\s*\{[ \t]*(?:#[^\n]*)?$",
        re.M,
    )
    for m in entry_pattern.finditer(text):
        patch_id = m.group(1)
        start = m.end()
        # Find matching closing `    },` at same indent
        # Simple state machine: count braces
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        body = text[start:i]
        entries[patch_id] = parse_body(body)
    return entries


def parse_body(body: str) -> dict:
    """Extract field values from a dict-body string.

    Handles `"field": <value>,  # optional comment` line format.
    Strips trailing comma + inline comment before ast.literal_eval (else
    `"foo",` would parse as 1-tuple `("foo",)`).
    """
    result: dict = {}
    for field in FIELDS:
        # Match `"field": <value>` up to end-of-line (not greedy)
        # Capture group 1 is the value portion
        m = re.search(rf'"{field}":\s*(.+?)$', body, flags=re.M)
        if not m:
            result[field] = None
            continue
        val = m.group(1).strip()
        # Strip trailing inline comment ` # ...`
        val = re.sub(r'\s+#.*$', '', val).strip()
        # Strip trailing comma
        val = val.rstrip(",").strip()
        # Try to evaluate as Python literal
        try:
            result[field] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            result[field] = val  # Keep raw if can't eval
    return result


def render_markdown(entries: dict[str, dict]) -> str:
    """Render patch table grouped by family."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines: list[str] = []

    lines.append("# Genesis vLLM Patches — Auto-generated Full Table")
    lines.append("")
    lines.append("> ⚠ **Auto-generated by `scripts/generate_patches_md.py` — DO NOT EDIT MANUALLY.**")
    lines.append("> Source of truth: `sndr/dispatcher/registry.py`.")
    lines.append("> Companion to curated [PATCHES.md](PATCHES.md) (which has narrative + tombstones + engine boundary discussion).")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append(f"Total entries: **{len(entries)}**")
    lines.append("")

    # Statistics
    tier_counts: dict = {}
    lifecycle_counts: dict = {}
    family_counts: dict = {}
    default_on_count = 0
    for e in entries.values():
        tier_counts[e.get("tier", "?")] = tier_counts.get(e.get("tier", "?"), 0) + 1
        lifecycle_counts[e.get("lifecycle", "?")] = lifecycle_counts.get(e.get("lifecycle", "?"), 0) + 1
        family_counts[e.get("family", "?")] = family_counts.get(e.get("family", "?"), 0) + 1
        if e.get("default_on") is True:
            default_on_count += 1

    lines.append("## Statistics")
    lines.append("")
    lines.append("### By tier")
    for tier, count in sorted(tier_counts.items(), key=lambda x: str(x[0])):
        lines.append(f"- `tier={tier}`: **{count}**")
    lines.append("")
    lines.append("### By lifecycle")
    for lc, count in sorted(lifecycle_counts.items(), key=lambda x: str(x[0])):
        lines.append(f"- `lifecycle={lc}`: **{count}**")
    lines.append("")
    lines.append(f"### Default-on at boot: **{default_on_count}** / {len(entries)}")
    lines.append("")
    lines.append("### By family")
    for fam, count in sorted(family_counts.items(), key=lambda x: str(x[0])):
        lines.append(f"- `{fam}`: {count}")
    lines.append("")

    # Group by family
    by_family: dict[str, list[tuple[str, dict]]] = {}
    for pid, entry in entries.items():
        fam = entry.get("family", "uncategorized")
        by_family.setdefault(fam, []).append((pid, entry))

    lines.append("## Patches by family")
    lines.append("")
    for family in sorted(by_family.keys(), key=str):
        items = sorted(by_family[family], key=lambda x: natural_sort_key(x[0]))
        lines.append(f"### `{family}` ({len(items)})")
        lines.append("")
        lines.append("| ID | Tier | Lifecycle | Default | Env flag | Upstream PR | Title |")
        lines.append("|---|---|---|:---:|---|:---:|---|")
        for pid, entry in items:
            tier = entry.get("tier", "?")
            lc = entry.get("lifecycle", "?")
            default_on = "✓" if entry.get("default_on") is True else "·"
            env = entry.get("env_flag") or "—"
            pr = entry.get("upstream_pr")
            pr_md = render_upstream_pr(pr)
            title = (entry.get("title") or "").replace("|", "\\|").replace("\n", " ").strip()
            # Truncate long titles
            if len(title) > 80:
                title = title[:77] + "..."
            lines.append(f"| **{pid}** | `{tier}` | `{lc}` | {default_on} | `{env}` | {pr_md} | {title} |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Regenerate / verify")
    lines.append("")
    lines.append("```bash")
    lines.append("# Regenerate (after registry.py changes)")
    lines.append("python3 scripts/generate_patches_md.py")
    lines.append("")
    lines.append("# Verify committed file is in sync with registry (CI gate)")
    lines.append("python3 scripts/generate_patches_md.py --check")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stdout", action="store_true", help="print to stdout instead of writing file")
    ap.add_argument("--check", action="store_true", help="verify committed file matches generated, exit 1 on divergence")
    args = ap.parse_args()

    try:
        entries = parse_registry(REGISTRY_PATH)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if not entries:
        print(f"ERROR: No entries parsed from {REGISTRY_PATH}", file=sys.stderr)
        return 2

    content = render_markdown(entries)

    if args.stdout:
        print(content)
        return 0

    if args.check:
        if not OUTPUT_PATH.is_file():
            print(f"FAIL: {OUTPUT_PATH.relative_to(REPO_ROOT)} doesn't exist — run without --check to generate", file=sys.stderr)
            return 1
        committed = OUTPUT_PATH.read_text()
        # Normalize timestamp comparison (timestamp will always differ)
        gen_normalized = re.sub(r"Generated: \S+", "Generated: <TIMESTAMP>", content)
        com_normalized = re.sub(r"Generated: \S+", "Generated: <TIMESTAMP>", committed)
        if gen_normalized == com_normalized:
            print(f"✓ {OUTPUT_PATH.relative_to(REPO_ROOT)} in sync with registry ({len(entries)} entries)")
            return 0
        else:
            print(f"✗ {OUTPUT_PATH.relative_to(REPO_ROOT)} OUT OF SYNC with registry", file=sys.stderr)
            print("  Run: python3 scripts/generate_patches_md.py", file=sys.stderr)
            return 1

    # Write file
    OUTPUT_PATH.write_text(content)
    print(f"✓ Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)} ({len(entries)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
