#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sync hardcoded counters in README.md to the live registry.

Roadmap §8 open item "Patch counts auto-sync": README.md hardcodes
patch / model / hardware / profile counts in several places. Each
review cycle drifts at least one (manual review noticed `134 patches`
text claim while the badge said `136`). This script:

  • computes authoritative counts from PATCH_REGISTRY + V2 builtin tree
  • rewrites only the well-known counter lines/badges in README.md
  • idempotent — no diff on a clean tree

Modes:

  python3 scripts/sync_readme_counters.py            # rewrite README.md
  python3 scripts/sync_readme_counters.py --check    # exit 1 if drift exists
  python3 scripts/sync_readme_counters.py --json     # machine-readable

Each rewrite rule is a (pattern, replacement_template) pair. Patterns
are conservative — they target one specific sentence/badge shape so
we don't accidentally touch unrelated numbers in the same file.

Exit codes:
  0 — README matches authoritative counts (or rewrite succeeded)
  1 — drift detected (--check mode) OR rewrite required
  2 — internal error
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
README_PATH = REPO_ROOT / "README.md"


# Ensure repo importable for the registry walk.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ─── Authoritative counts ──────────────────────────────────────────────


@dataclass(frozen=True)
class Counts:
    patches: int
    families: int
    v2_models: int
    v2_hardware: int
    v2_profiles: int
    v2_aliases: int


def collect_counts() -> Counts:
    from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
    families = Counter(
        v.get("family", "?") for v in PATCH_REGISTRY.values()
    )

    presets_dir = (
        REPO_ROOT / "vllm" / "sndr_core" / "model_configs" / "builtin" / "presets"
    )
    aliases = (
        sum(1 for _ in presets_dir.glob("*.yaml")) if presets_dir.is_dir() else 0
    )

    try:
        from vllm.sndr_core.model_configs.registry_v2 import (
            list_hardware, list_models, list_profiles,
        )
        n_models = len(list_models())
        n_hardware = len(list_hardware())
        n_profiles = len(list_profiles())
    except Exception:
        # Best-effort fallback for installs without V2 registry.
        n_models = n_hardware = n_profiles = 0

    return Counts(
        patches=len(PATCH_REGISTRY),
        families=len(families),
        v2_models=n_models,
        v2_hardware=n_hardware,
        v2_profiles=n_profiles,
        v2_aliases=aliases,
    )


# ─── Rewrite rules ────────────────────────────────────────────────────


@dataclass(frozen=True)
class Rule:
    """One counter we know how to find and rewrite in README.md.

    `pattern` is a regex with a single `\\d+` capture group representing
    the number; `template` reuses the rest of the matched text and slots
    in the authoritative value.
    """
    rule_id: str
    description: str
    pattern: re.Pattern
    replacement: str
    expected_count_attr: str   # attribute of `Counts`


# Each replacement template uses `{n}` for the authoritative count.
RULES: list[Rule] = [
    Rule(
        rule_id="R-patch-badge",
        description="shields.io badge: `patches-N-green.svg`",
        pattern=re.compile(r"patches-(\d+)-green\.svg"),
        replacement="patches-{n}-green.svg",
        expected_count_attr="patches",
    ),
    Rule(
        rule_id="R-text-N-community-patches",
        description='inline "**N community patches**" text',
        pattern=re.compile(r"\*\*(\d+) community patches\*\*"),
        replacement="**{n} community patches**",
        expected_count_attr="patches",
    ),
    Rule(
        rule_id="R-coverage-line",
        description="### Patch coverage — N patches across M categories",
        pattern=re.compile(
            r"### Patch coverage — (\d+) patches across (\d+) categories",
        ),
        # Two capture groups in this rule; we handle it separately in apply()
        # because it has TWO authoritative values.
        replacement="### Patch coverage — {n} patches across {m} categories",
        expected_count_attr="patches",   # primary; categories handled in apply()
    ),
    Rule(
        rule_id="R-by-category-heading",
        description='"## 📦 N patches by category"',
        pattern=re.compile(r"## 📦 (\d+) patches by category"),
        replacement="## 📦 {n} patches by category",
        expected_count_attr="patches",
    ),
    Rule(
        rule_id="R-all-N-patches-table",
        description='link text "All N patches table"',
        pattern=re.compile(r"All (\d+) patches table"),
        replacement="All {n} patches table",
        expected_count_attr="patches",
    ),
]


# ─── Apply ────────────────────────────────────────────────────────────


@dataclass
class RuleHit:
    rule: Rule
    line_no: int
    old: str
    new: str

    @property
    def changed(self) -> bool:
        return self.old != self.new


def apply_rules(text: str, counts: Counts) -> tuple[str, list[RuleHit]]:
    """Apply every rule to `text`. Returns (new_text, hits)."""
    lines = text.splitlines(keepends=True)
    hits: list[RuleHit] = []
    for rule in RULES:
        n_value = getattr(counts, rule.expected_count_attr)
        for i, line in enumerate(lines):
            m = rule.pattern.search(line)
            if not m:
                continue
            # Special-case: R-coverage-line has 2 capture groups.
            if rule.rule_id == "R-coverage-line":
                new_match = rule.replacement.format(
                    n=counts.patches, m=counts.families,
                )
            else:
                new_match = rule.replacement.format(n=n_value)
            new_line = line[:m.start()] + new_match + line[m.end():]
            if new_line != line:
                hits.append(RuleHit(
                    rule=rule, line_no=i + 1, old=line.rstrip("\n"),
                    new=new_line.rstrip("\n"),
                ))
                lines[i] = new_line
    return "".join(lines), hits


# ─── Main ─────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check", action="store_true",
        help="Exit 1 if README needs rewrite; don't modify the file.",
    )
    ap.add_argument(
        "--json", action="store_true",
        help="Emit JSON summary instead of human-readable text.",
    )
    ap.add_argument(
        "--file", default=str(README_PATH),
        help="README path (default: ./README.md).",
    )
    args = ap.parse_args()

    target = Path(args.file)
    if not target.is_file():
        sys.stderr.write(f"sync_readme_counters: file not found: {target}\n")
        return 2

    try:
        counts = collect_counts()
    except Exception as e:
        sys.stderr.write(f"sync_readme_counters: count error: "
                         f"{type(e).__name__}: {e}\n")
        return 2

    original = target.read_text(encoding="utf-8")
    new_text, hits = apply_rules(original, counts)
    changed_hits = [h for h in hits if h.changed]

    # JSON mode: emit summary regardless of --check. When --check is set,
    # we never write; when not set, we do write (same as text mode).
    if args.json:
        wrote = False
        if changed_hits and not args.check:
            target.write_text(new_text, encoding="utf-8")
            wrote = True
        print(json.dumps({
            "file": str(target),
            "counts": counts.__dict__,
            "rules": [
                {
                    "rule_id": h.rule.rule_id,
                    "description": h.rule.description,
                    "line": h.line_no,
                    "old": h.old,
                    "new": h.new,
                    "changed": h.changed,
                }
                for h in changed_hits
            ],
            "drift_count": len(changed_hits),
            "passed": not changed_hits,
            "wrote_file": wrote,
        }, indent=2, sort_keys=True))
        # --check semantics: exit 1 iff drift exists.
        if args.check:
            return 0 if not changed_hits else 1
        # Rewrite semantics: exit 0 (success — either no drift, or we fixed it).
        return 0

    print(f"sync_readme_counters: {target}")
    print("  Authoritative counts:")
    print(f"    patches:      {counts.patches}")
    print(f"    families:     {counts.families}")
    print(f"    v2_models:    {counts.v2_models}")
    print(f"    v2_hardware:  {counts.v2_hardware}")
    print(f"    v2_profiles:  {counts.v2_profiles}")
    print(f"    v2_aliases:   {counts.v2_aliases}")
    print("─" * 60)
    if not changed_hits:
        print("  ✓ README already matches authoritative counts")
        return 0

    for h in changed_hits:
        print(f"  ✗ [{h.rule.rule_id}] line {h.line_no} — {h.rule.description}")
        print(f"        old: {h.old.strip()}")
        print(f"        new: {h.new.strip()}")

    if args.check:
        print()
        print(f"  drift: {len(changed_hits)} line(s) need sync — "
              f"run without --check to rewrite")
        return 1

    target.write_text(new_text, encoding="utf-8")
    print()
    print(f"  ✓ wrote {len(changed_hits)} update(s) to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
