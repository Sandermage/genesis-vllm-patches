#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GUI i18n coverage audit.

Every ``tr("English")`` call in the GUI looks the source string up in i18n.ts's
``RU_BY_EN`` map; a missing entry silently renders English to a Russian operator.
This audit extracts every ``tr()`` source string and every ``RU_BY_EN`` key and
reports:

* **untranslated** — ``tr()`` in code with no Russian entry (the bug we gate on);
* **unused** — a Russian entry with no ``tr()`` call (dead weight, informational).

Ratchet gate: fails when untranslated exceeds the baseline (default 0). Some
strings are legitimately language-neutral (code ids, HTTP verbs, symbols) and are
allow-listed below. Skips cleanly if the GUI source isn't present.

Usage: ``python3 scripts/audit_i18n.py [--baseline N] [--list]``
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "gui" / "web" / "src"
I18N = SRC / "i18n.ts"

# Language-neutral strings that legitimately need no Russian (technical ids,
# HTTP verbs, symbols, units rendered verbatim). Extend with care.
ALLOW_UNTRANSLATED: set[str] = set()

_DQ = re.compile(r'\btr\(\s*"((?:[^"\\]|\\.)*)"\s*[),]')
_SQ = re.compile(r"\btr\(\s*'((?:[^'\\]|\\.)*)'\s*[),]")
_KEY = re.compile(r'^\s*"((?:[^"\\]|\\.)*)"\s*:', re.M)
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r"}


def _unescape(s: str) -> str:
    """Decode JS string escapes so a single-quoted tr() with a literal `"` and a
    double-quoted map key with `\\"` compare equal (escaping is just delimiter
    noise, not part of the string)."""
    return re.sub(r"\\(.)", lambda m: _ESCAPES.get(m.group(1), m.group(1)), s)


def ru_keys() -> set[str]:
    text = I18N.read_text(encoding="utf-8")
    m = re.search(r"RU_BY_EN[^=]*=\s*\{(.*?)\n\};", text, re.S)
    block = m.group(1) if m else ""
    return {_unescape(km.group(1)) for km in _KEY.finditer(block)}


def tr_strings() -> set[str]:
    out: set[str] = set()
    for f in SRC.rglob("*.ts*"):
        if f.name == "i18n.ts" or ".test." in f.name:
            continue
        text = f.read_text(encoding="utf-8")
        out.update(_unescape(m.group(1)) for m in _DQ.finditer(text))
        out.update(_unescape(m.group(1)) for m in _SQ.finditer(text))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=int, default=0)
    ap.add_argument("--list", action="store_true", help="print every untranslated string")
    args = ap.parse_args()

    if not I18N.is_file():
        print("audit_i18n: gui/web/src not present — skipping")
        return 0

    keys = ru_keys()
    trs = tr_strings()
    untranslated = sorted(trs - keys - ALLOW_UNTRANSLATED)
    unused = sorted(keys - trs)

    print(f"tr() source strings: {len(trs)} | RU_BY_EN entries: {len(keys)}")
    print(f"untranslated (tr in code, no RU): {len(untranslated)}")
    print(f"unused RU entries (no tr call):   {len(unused)} (informational)")
    shown = untranslated if args.list else untranslated[:40]
    for u in shown:
        print("  MISSING-RU:", repr(u))
    if len(untranslated) > len(shown):
        print(f"  … +{len(untranslated) - len(shown)} more (use --list)")

    if len(untranslated) > args.baseline:
        print(f"\nFAIL: {len(untranslated)} untranslated tr() string(s) > baseline {args.baseline}")
        return 1
    print("\nOK: i18n coverage within baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
