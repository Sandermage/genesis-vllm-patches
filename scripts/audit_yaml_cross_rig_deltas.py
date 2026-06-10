#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit script — extract cross-rig delta annotations from PATCH_REGISTRY
credit fields and suggest enhancements for builtin model YAML comments.

Per CLUB3090 cross-reference plan §3.4 (cross-rig delta inline documentation):
their `.env.example` annotates every `GENESIS_ENABLE_*` with measured
delta (`+15-30% on L2≥24MB only`, `+10.5% on 3090 INT4`). We had this
only partially (~10 of ~80 enabled patches).

This script bridges the gap WITHOUT manual lookup — it parses
PATCH_REGISTRY credit fields for delta patterns and outputs:

  * For each YAML's `GENESIS_ENABLE_X: '1'` line:
    - current inline comment
    - extracted deltas from registry credit
    - suggested enhanced comment (auto-stub)

Operator workflow
-----------------
  1. Phase A (this script): run it, get suggested annotations
  2. Phase B: manually review top 30 high-impact patches
  3. Phase C: defer remaining as `# no isolated bench data` placeholders
  4. Phase D: optionally auto-apply suggestions via --apply flag (DANGEROUS)

Default mode is REPORT — no file modification. The script is read-only
unless --apply is explicitly passed.

Usage
-----
  # Report mode (default)
  python3 scripts/audit_yaml_cross_rig_deltas.py

  # Filter to specific YAML
  python3 scripts/audit_yaml_cross_rig_deltas.py --yaml qwen3.6-35b-a3b-fp8

  # Show only patches missing deltas
  python3 scripts/audit_yaml_cross_rig_deltas.py --only-missing

  # JSON output (machine-readable)
  python3 scripts/audit_yaml_cross_rig_deltas.py --json

Pattern detection
-----------------
The script greps the credit field for:
  - `+N%`, `-N%`, `±N%`     — percentage delta
  - `+N.M%`, `-N.M%`         — decimal percentage
  - `+N MB`, `+N MiB`        — VRAM delta
  - `+N tok/s`, `+N TPS`     — throughput delta
  - `neutral`                — explicit neutral
  - `validated`, `verified`  — empirical confirmation
  - `bench`, `measured`      — bench evidence
  - `on A5000`, `on 3090`    — rig-specific delta

Conservative — only suggests, never auto-writes. Author retains judgment.

Sander 2026-05-30 — per UNIFIED_DEVELOPMENT_PLAN v1.1 §2.9.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILTIN_MODEL_DIR = (
    REPO_ROOT / "sndr" / "model_configs" / "builtin" / "model"
)
REGISTRY_PATH = REPO_ROOT / "sndr" / "dispatcher" / "registry.py"

# Pattern detectors — order matters (more specific first).
_PCT_RE = re.compile(
    r"[−–\-]?(\d+(?:\.\d+)?(?:[\-–](?:\d+(?:\.\d+)?))?)\s*(?:%|pp)",
    re.IGNORECASE,
)
_VRAM_RE = re.compile(
    r"[+\-−–]?\s*(\d+(?:\.\d+)?(?:[\-–]\d+(?:\.\d+)?)?)\s*(MB|MiB|GiB|GB)",
    re.IGNORECASE,
)
_TOK_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(tok/s|TPS|tps|t/s)",
    re.IGNORECASE,
)
_RIG_RE = re.compile(
    r"on\s+(A5000|3090|4090|5090|H100|H200|A100|L40S?|RTX\s+\d+)",
    re.IGNORECASE,
)
_NEUTRAL_RE = re.compile(
    r"\b(?:neutral|no\s+measurable|within\s+CV|n\.?s\.?)\b",
    re.IGNORECASE,
)
_BENCH_RE = re.compile(
    r"\b(?:bench-validated|empirically|measured|validated|verified|"
    r"confirmed|bench(?:marked)?)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class PatchInfo:
    patch_id: str
    title: str
    credit: str
    deltas: list[str] = field(default_factory=list)
    rigs: list[str] = field(default_factory=list)
    has_neutral: bool = False
    has_bench_evidence: bool = False


@dataclass(slots=True)
class YamlPatchUsage:
    patch_id: str
    env_var: str
    enabled: bool
    line_no: int
    current_comment: str


def _load_registry_credits() -> dict[str, PatchInfo]:
    """Parse registry.py for patch_id → credit/title mapping."""
    text = REGISTRY_PATH.read_text(encoding="utf-8")
    out: dict[str, PatchInfo] = {}
    # Match `"PATCH_ID": {` then capture until `},` at same indent.
    # Then within block, look for `"title"`, `"credit"`.
    patch_block_re = re.compile(
        r'^    "([A-Z][A-Z0-9_a-z]+)":\s*\{$',
        re.MULTILINE,
    )
    title_re = re.compile(r'"title":\s*"([^"]*)"', re.MULTILINE)
    credit_re = re.compile(
        r'"credit":\s*\((.*?)\),\s*\n\s*"upstream_pr"',
        re.DOTALL,
    )
    credit_str_re = re.compile(
        r'"credit":\s*"([^"]*)"',
        re.DOTALL,
    )

    # Split by patch block start; reassemble.
    matches = list(patch_block_re.finditer(text))
    for i, m in enumerate(matches):
        pid = m.group(1)
        start = m.end()
        # Block ends at next patch or registry close brace.
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]
        title_m = title_re.search(block)
        title = title_m.group(1) if title_m else ""
        credit_m = credit_re.search(block)
        if credit_m:
            # Credit was a Python tuple of strings; strip quotes + concat.
            raw = credit_m.group(1)
            credit = " ".join(
                re.findall(r'"((?:[^"\\]|\\.)*)"', raw)
            ).replace('\\"', '"')
        else:
            cs_m = credit_str_re.search(block)
            credit = cs_m.group(1) if cs_m else ""
        out[pid] = PatchInfo(patch_id=pid, title=title, credit=credit)
    return out


def _extract_deltas(info: PatchInfo) -> None:
    """Populate deltas/rigs/has_neutral/has_bench fields by parsing credit."""
    credit = info.credit
    if not credit:
        return

    # Percentage deltas
    for m in _PCT_RE.finditer(credit):
        token = m.group(0).strip()
        if token not in info.deltas:
            info.deltas.append(token)

    # VRAM
    for m in _VRAM_RE.finditer(credit):
        token = m.group(0).strip()
        if token not in info.deltas:
            info.deltas.append(token)

    # Token rate
    for m in _TOK_RE.finditer(credit):
        token = m.group(0).strip()
        if token not in info.deltas:
            info.deltas.append(token)

    # Rig identification
    for m in _RIG_RE.finditer(credit):
        rig = m.group(1).strip()
        if rig not in info.rigs:
            info.rigs.append(rig)

    info.has_neutral = bool(_NEUTRAL_RE.search(credit))
    info.has_bench_evidence = bool(_BENCH_RE.search(credit))


# Match `  GENESIS_ENABLE_X: '0|1'  # comment text`
_YAML_PATCH_LINE = re.compile(
    r"^(\s+)GENESIS_(ENABLE|P67_USE_UPSTREAM|TQ_MAX_MODEL_LEN|LEGACY_P7|"
    r"PN16_(?:TOOL_THINK_BUDGET|CLASSIFIER_MAX_TOKENS)|P67_NUM_KV_SPLITS)"
    r"_?([A-Za-z0-9_]*):\s*'?([01]|\d+)'?\s*(#.*)?$"
)


def _scan_yaml(yaml_path: Path) -> list[YamlPatchUsage]:
    """Find all GENESIS_ENABLE_X lines in a YAML file."""
    text = yaml_path.read_text(encoding="utf-8")
    out: list[YamlPatchUsage] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if "GENESIS_ENABLE_" not in line:
            continue
        # Extract: indent, env_var, value, optional comment
        # Match `  GENESIS_ENABLE_FOO: '1'   # comment`
        m = re.match(
            r"^(\s+)(GENESIS_ENABLE_[A-Z0-9_]+):\s*'?([01])'?\s*(#.*)?$",
            line,
        )
        if not m:
            continue
        env_var = m.group(2)
        value = m.group(3)
        comment = (m.group(4) or "").strip()
        # Reverse-derive patch_id from env_var. Convention:
        #   GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL → "P67"
        #   GENESIS_ENABLE_PN286_FA_LAYOUT_REVERT_SM86 → "PN286"
        #   GENESIS_ENABLE_G4_61_TQ_SHARED_WORKSPACE → "G4_61"
        body = env_var.removeprefix("GENESIS_ENABLE_")
        m_id = re.match(
            r"^(P\d+[a-zC]*|PN\d+|G4_\d+[A-Z]*|SPRINT26)_?", body
        )
        if not m_id:
            continue
        patch_id = m_id.group(1).upper()
        # Normalize: G4_61 stays G4_61, but Pn286 → PN286, P67c → P67c
        if patch_id.startswith("G4"):
            pass  # keep as-is
        elif patch_id.startswith("PN"):
            pass
        else:
            # P67c / P98 / etc. — preserve trailing lowercase
            patch_id = re.match(r"^P\d+[a-zA-Z]*", body).group(0)
        out.append(YamlPatchUsage(
            patch_id=patch_id,
            env_var=env_var,
            enabled=value == "1",
            line_no=line_no,
            current_comment=comment,
        ))
    return out


def _suggest_annotation(
    patch: PatchInfo, usage: YamlPatchUsage
) -> str:
    """Build suggested inline comment from extracted deltas."""
    if not patch:
        return f"# {usage.env_var} — no registry entry"
    parts = []
    if patch.deltas:
        # Compact: top 3 deltas joined
        deltas_str = ", ".join(patch.deltas[:3])
        parts.append(deltas_str)
    if patch.rigs:
        rigs_str = " / ".join(patch.rigs[:3])
        parts.append(f"on {rigs_str}")
    if patch.has_neutral:
        parts.append("neutral / within-CV on at least one rig")
    if patch.has_bench_evidence:
        parts.append("bench-validated")
    if not parts:
        return "# no isolated bench data in registry credit"
    return "# " + "; ".join(parts) + " (auto-extracted; verify before commit)"


def _has_deltas_in_comment(comment: str) -> bool:
    """Heuristic — does the current YAML comment already mention deltas?"""
    if not comment:
        return False
    if _PCT_RE.search(comment) or _VRAM_RE.search(comment):
        return True
    if _NEUTRAL_RE.search(comment):
        return True
    return False


def _format_human(
    rows: list[tuple[Path, YamlPatchUsage, PatchInfo, str]],
    only_missing: bool,
) -> str:
    by_yaml: dict[Path, list] = {}
    for yaml_path, usage, info, suggestion in rows:
        if only_missing and _has_deltas_in_comment(usage.current_comment):
            continue
        by_yaml.setdefault(yaml_path, []).append(
            (usage, info, suggestion)
        )

    lines = []
    for yaml_path in sorted(by_yaml):
        lines.append(f"\n=== {yaml_path.name} ===")
        for usage, info, suggestion in by_yaml[yaml_path]:
            on_off = "✓" if usage.enabled else "✗"
            lines.append(
                f"  [{on_off}] line {usage.line_no:>3} "
                f"{usage.env_var}"
            )
            curr = (usage.current_comment or "(no comment)")[:100]
            lines.append(f"     current:   {curr}")
            lines.append(f"     suggested: {suggestion}")
            if not info or not (info.deltas or info.rigs
                                or info.has_neutral
                                or info.has_bench_evidence):
                lines.append("     evidence:  none found in registry credit")
            else:
                ev = []
                if info.deltas:
                    ev.append(f"deltas={info.deltas[:3]}")
                if info.rigs:
                    ev.append(f"rigs={info.rigs[:3]}")
                if info.has_neutral:
                    ev.append("neutral")
                if info.has_bench_evidence:
                    ev.append("bench-validated")
                lines.append(f"     evidence:  {' '.join(ev)}")
    if not by_yaml:
        lines.append("(no rows to report — all annotations have deltas)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--yaml", type=str, default=None,
                    help="filter to specific YAML basename (no .yaml suffix)")
    ap.add_argument("--only-missing", action="store_true",
                    help="show only patches whose current comment lacks deltas")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable JSON output")
    args = ap.parse_args(argv)

    if not BUILTIN_MODEL_DIR.is_dir():
        sys.stderr.write(f"ERROR: builtin model dir not found: {BUILTIN_MODEL_DIR}\n")
        return 2

    registry = _load_registry_credits()
    for pid, info in registry.items():
        _extract_deltas(info)

    yamls = sorted(BUILTIN_MODEL_DIR.glob("*.yaml"))
    if args.yaml:
        yamls = [p for p in yamls if p.stem == args.yaml]
        if not yamls:
            sys.stderr.write(f"ERROR: no YAML named {args.yaml}.yaml\n")
            return 2

    rows: list[tuple[Path, YamlPatchUsage, PatchInfo, str]] = []
    for yp in yamls:
        usages = _scan_yaml(yp)
        for u in usages:
            info = registry.get(u.patch_id)
            suggestion = _suggest_annotation(info, u)
            rows.append((yp, u, info, suggestion))

    if args.json:
        out = []
        for yaml_path, usage, info, suggestion in rows:
            if args.only_missing and _has_deltas_in_comment(
                usage.current_comment
            ):
                continue
            out.append({
                "yaml": yaml_path.name,
                "line": usage.line_no,
                "patch_id": usage.patch_id,
                "env_var": usage.env_var,
                "enabled": usage.enabled,
                "current_comment": usage.current_comment,
                "suggested_comment": suggestion,
                "extracted_deltas": info.deltas if info else [],
                "extracted_rigs": info.rigs if info else [],
                "has_neutral": info.has_neutral if info else False,
                "has_bench_evidence": (
                    info.has_bench_evidence if info else False
                ),
            })
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(_format_human(rows, args.only_missing))
        # Summary
        total = len(rows)
        missing = sum(
            1 for _, u, _, _ in rows
            if not _has_deltas_in_comment(u.current_comment)
        )
        with_evidence = sum(
            1 for _, _, info, _ in rows
            if info and (info.deltas or info.rigs
                         or info.has_neutral or info.has_bench_evidence)
        )
        print(f"\n--- Summary ---")
        print(f"  Total enable lines:        {total}")
        print(f"  Without inline delta:      {missing}")
        print(f"  Registry has evidence for: {with_evidence}")
        print(f"  Could enhance (delta missing × evidence present): "
              f"{sum(1 for _, u, info, _ in rows if not _has_deltas_in_comment(u.current_comment) and info and (info.deltas or info.rigs))}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
