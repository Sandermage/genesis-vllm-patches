#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit english-only rule for code files.

Per CLAUDE.md user rule: all in-code text — comments, docstrings, error
messages, log messages, printed strings, exception messages, argparse
help — must be English only. Russian (or other Cyrillic) text in code
breaks grep, accumulates dialects between authors, and confuses future
maintainers / reviewers / AI assistants.

Markdown docs are exempt (Russian planning docs are explicitly allowed
per the same rule). This audit only covers code files.

Ratchet-down model:
  - Baseline file lists files with currently-tolerated Russian char counts.
  - --check mode fails CI ONLY if total exceeds baseline (no regression).
  - --strict mode fails on any > 0 (long-term goal once all files cleaned).
  - --update-baseline regenerates baseline after a translation pass.

Exit codes:
  0 — within baseline (or strict-clean)
  1 — regression: count exceeds baseline OR new file with Russian
  2 — internal error / invalid baseline JSON

Modes:
  python3 scripts/audit_english_only.py                # human report
  python3 scripts/audit_english_only.py --check        # CI gate (vs baseline)
  python3 scripts/audit_english_only.py --strict       # CI gate (zero tolerance)
  python3 scripts/audit_english_only.py --json         # machine-readable
  python3 scripts/audit_english_only.py --update-baseline  # operator opt-in
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "scripts" / "audit_english_only.baseline.json"

CODE_GLOBS = [
    "vllm/sndr_core/**/*.py",
    "scripts/*.py",
    "tools/**/*.py",
    "tools/**/*.sh",
    "tests/**/*.py",
]

EXCLUDE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    "_retired",
    "dead_patches",
    "snapshots",
    ".git",
}

# Per-file waivers — Cyrillic in these files is INTENTIONAL and audited.
# Each entry maps a repo-relative path to the rationale. Waived files are
# omitted from --strict counts entirely; they still show up in default
# report mode under [WAIVED].
WAIVERS: dict[str, str] = {
    "scripts/audit_english_only.py": (
        "The Cyrillic regex pattern itself ([Ѐ-ӿԀ-ԯ]) is functional code, "
        "not a comment or string. Required for the audit to detect "
        "Cyrillic characters at all."
    ),
    "tests/probes/streaming_thinking_probe.py": (
        "Multilingual streaming x thinking test fixtures. Russian Bitcoin "
        "analyst prompts simulate real Sander production usage (multi-"
        "paragraph RAG + tool-call streaming). Translating would alter "
        "the test semantic; the model must work correctly with the "
        "Russian input."
    ),
    "tests/unit/scripts/test_audit_english_only.py": (
        "Unit tests for the Cyrillic detector itself. The test inputs MUST "
        "contain Cyrillic by design (regex-positive cases, sample fixtures "
        "for count_cyrillic). Translating defeats the test purpose."
    ),
    "tools/tokenizer_fingerprint.py": (
        "Multilingual tokenizer-fingerprint corpus (wave-2 tool, 2026-06-13): "
        "the 'multilingual_cyrillic' probe string is intentional non-Latin "
        "test DATA that exercises how the tokenizer segments Cyrillic after "
        "a pin bump (sibling probes cover CJK / emoji / unicode punctuation). "
        "Translating it to Latin would defeat the fingerprint's purpose — it "
        "must contain Cyrillic to detect Cyrillic-tokenization drift. Same "
        "fixture-data class as tests/probes/streaming_thinking_probe.py."
    ),
    "tests/unit/product_api/test_market_tools.py": (
        "Intentional Russian test INPUT: the crypto/market symbol extractor "
        "must work on real Russian user queries (Sander is a Russian-speaking "
        "operator), so the test asserts _extract_crypto_symbols('дай анализ "
        "актива wld на сегодня') == ['WLD']. Translating the query to English "
        "would defeat the test's purpose (it verifies Cyrillic-query symbol "
        "extraction). Same fixture-data class as streaming_thinking_probe.py."
    ),
}

CYRILLIC_RE = re.compile(r"[Ѐ-ӿԀ-ԯ]")


def iter_targets() -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for pattern in CODE_GLOBS:
        for p in REPO_ROOT.glob(pattern):
            if any(part in EXCLUDE_DIRS for part in p.parts):
                continue
            if not p.is_file():
                continue
            rel = p.relative_to(REPO_ROOT)
            if rel in seen:
                continue
            seen.add(rel)
            out.append(p)
    return sorted(out)


def count_cyrillic(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return 0
    return len(CYRILLIC_RE.findall(text))


def scan_all(include_waivers: bool = True) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in iter_targets():
        n = count_cyrillic(path)
        if n > 0:
            rel = str(path.relative_to(REPO_ROOT))
            if not include_waivers and rel in WAIVERS:
                continue
            counts[rel] = n
    return counts


def load_baseline() -> dict[str, int]:
    if not BASELINE_PATH.exists():
        return {}
    try:
        with BASELINE_PATH.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as e:
        print(f"ERROR: cannot parse baseline {BASELINE_PATH}: {e}",
              file=sys.stderr)
        sys.exit(2)
    return {k: int(v) for k, v in data.get("files", {}).items()}


def write_baseline(counts: dict[str, int]) -> None:
    total = sum(counts.values())
    payload = {
        "rule": "English only in code (CLAUDE.md user rule). Baseline "
                "captures currently-tolerated counts; new violations or "
                "regressions vs this baseline fail audit. Translate + "
                "regenerate baseline as you clean files.",
        "totals": {"files": len(counts), "chars": total},
        "files": dict(sorted(counts.items())),
    }
    BASELINE_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"baseline written: {BASELINE_PATH} "
          f"({len(counts)} files, {total} chars)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if regression vs baseline")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any file has > 0 (zero tolerance)")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON")
    ap.add_argument("--update-baseline", action="store_true",
                    help="regenerate baseline file from current state")
    args = ap.parse_args()

    current = scan_all()

    if args.update_baseline:
        # Baseline tracks NON-waivered files only; waivered entries
        # are documented in the WAIVERS dict above instead.
        write_baseline({k: v for k, v in current.items() if k not in WAIVERS})
        return 0

    baseline = load_baseline()
    regressions: list[tuple[str, int, int]] = []
    new_files: list[tuple[str, int]] = []
    improvements: list[tuple[str, int, int]] = []

    for rel, n in sorted(current.items()):
        if rel in WAIVERS:
            continue
        b = baseline.get(rel)
        if b is None:
            new_files.append((rel, n))
        elif n > b:
            regressions.append((rel, b, n))
        elif n < b:
            improvements.append((rel, b, n))

    cleaned = sorted(set(baseline) - set(current))

    total_current = sum(current.values())
    total_baseline = sum(baseline.values())

    if args.json:
        out = {
            "totals": {"current": total_current, "baseline": total_baseline},
            "files_current": len(current),
            "files_baseline": len(baseline),
            "new_files": [{"path": p, "count": n} for p, n in new_files],
            "regressions": [
                {"path": p, "baseline": b, "current": n}
                for p, b, n in regressions
            ],
            "improvements": [
                {"path": p, "baseline": b, "current": n}
                for p, b, n in improvements
            ],
            "cleaned": cleaned,
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(f"english-only audit: {len(current)} files contain "
              f"Cyrillic ({total_current} chars total)")
        print(f"baseline:           {len(baseline)} files "
              f"({total_baseline} chars)")
        if new_files:
            print(f"\n[NEW VIOLATIONS] {len(new_files)} files not in baseline:")
            for p, n in new_files:
                print(f"  + {p}: {n} chars")
        if regressions:
            print(f"\n[REGRESSIONS] {len(regressions)} files exceed baseline:")
            for p, b, n in regressions:
                print(f"  ! {p}: {b} → {n} (+{n - b})")
        if improvements:
            print(f"\n[IMPROVEMENTS] {len(improvements)} files below baseline:")
            for p, b, n in improvements[:10]:
                print(f"  - {p}: {b} → {n} (-{b - n})")
            if len(improvements) > 10:
                print(f"    ... and {len(improvements) - 10} more")
        if cleaned:
            print(f"\n[CLEANED] {len(cleaned)} files now Cyrillic-free:")
            for p in cleaned[:10]:
                print(f"  ✓ {p}")
            if len(cleaned) > 10:
                print(f"    ... and {len(cleaned) - 10} more")
        waived_present = sorted(p for p in current if p in WAIVERS)
        if waived_present:
            print(f"\n[WAIVED] {len(waived_present)} intentional exceptions:")
            for p in waived_present:
                print(f"  ~ {p}: {current[p]} chars — {WAIVERS[p][:60]}…")
        if improvements or cleaned:
            print("\nrun `--update-baseline` to lock in improvements")

    if args.strict:
        # Strict mode: anything outside WAIVERS is a violation.
        non_waived = scan_all(include_waivers=False)
        return 1 if non_waived else 0
    if args.check:
        return 1 if (new_files or regressions) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
