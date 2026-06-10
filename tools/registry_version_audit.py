#!/usr/bin/env python3
"""tools/registry_version_audit.py — patch-vs-pin version-range audit.

Iterate ``PATCH_REGISTRY``; for each entry with ``vllm_version_range``,
evaluate against a target pin (live or operator-supplied), and report
``WOULD_SKIP / APPLIES / UNCONSTRAINED``. Red-flag ``default_on=True``
entries that would skip on the target pin — those are silent feature
regressions on a pin bump.

Motivation (from 2026-06-09 internal audit report A1)
=====================================================

The audit found ~30 patches gated to ``vllm_version_range="<0.22.0"``
while our pin is 0.22.1rc1.dev259+g303916e93. The dispatcher does NOT
hard-enforce the range at runtime (live boot traces show those patches
still apply via env flag), but:

  * ``patches doctor`` and other declarative consumers treat the range
    as authoritative.
  * Operators reading the registry assume the range is binding.
  * On the next pin bump (0.23.0), the range will silently flip to
    "out of range" for entries with ``<0.22.0`` and the doctor will
    start reporting them as expected-skip.

This tool surfaces drift so we can fix the ceilings before they bite.

Usage
=====

  # Audit against the running engine's pin (detected via the container)
  python3 tools/registry_version_audit.py

  # Audit against an explicit pin
  python3 tools/registry_version_audit.py --pin 0.23.1rc0+g123456

  # JSON output for CI / scripts
  python3 tools/registry_version_audit.py --json

  # Exit non-zero if any default_on=True patch would skip on the pin
  python3 tools/registry_version_audit.py --ci-strict

Exit codes
==========

  * 0 — no actionable drift (clean for this pin)
  * 1 — at least one default_on=True patch would skip (CI-strict mode)
  * 2 — invalid usage / argument errors
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any

# Try to import the registry — if not in package mode, walk up the tree.
try:
    from sndr.dispatcher.registry import PATCH_REGISTRY
except ImportError:
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from sndr.dispatcher.registry import PATCH_REGISTRY  # type: ignore


@dataclass
class PatchVerdict:
    """One row of the audit output."""
    patch_id: str
    title: str
    default_on: bool
    env_flag: str | None
    vllm_version_range: tuple[str, str] | None
    verdict: str  # APPLIES / WOULD_SKIP / UNCONSTRAINED / RANGE_INVALID
    reason: str
    severity: str  # info / warn / critical


def _detect_live_pin() -> str | None:
    """Detect the live pin from a running engine container.

    Looks for ``vllm-qwen3.6-*`` first (PROD), falls back to any
    ``vllm-*`` container. Reads ``__version__`` from inside the
    container via ``docker exec`` on the rig named by the ``SSH_HOST``
    env var (``<user>@<host>``; no default — live probe is skipped
    when unset).

    Returns the version string (e.g. ``0.22.1rc1.dev259+g303916e93``)
    or None if no live container is found.
    """
    import os
    import subprocess
    ssh_host = os.environ.get("SSH_HOST", "").strip()
    if not ssh_host:
        return None
    try:
        r = subprocess.run(
            ["ssh", ssh_host,
             "docker ps --format '{{.Names}}' | grep -E '^vllm-' | head -1"],
            capture_output=True, text=True, timeout=10,
        )
        container = r.stdout.strip()
        if not container:
            return None
        r = subprocess.run(
            ["ssh", ssh_host,
             f"docker exec {container} python3 -c 'import vllm; print(vllm.__version__)'"],
            capture_output=True, text=True, timeout=15,
        )
        version = r.stdout.strip()
        return version if version else None
    except (subprocess.SubprocessError, OSError):
        return None


def _version_matches(version: str, vrange: tuple[str, str]) -> str:
    """Evaluate ``version`` against PEP-440 ``vrange``.

    Returns "APPLIES", "WOULD_SKIP", or "RANGE_INVALID".
    """
    try:
        from packaging.specifiers import SpecifierSet
        spec_str = ",".join(vrange)
        sset = SpecifierSet(spec_str)
        sset.prereleases = True
        # vllm uses local-version segments (+gSHA); strip them for matching.
        clean_version = version.partition("+")[0]
        return "APPLIES" if clean_version in sset else "WOULD_SKIP"
    except ImportError:
        return "RANGE_INVALID"
    except Exception:  # noqa: BLE001
        return "RANGE_INVALID"


def audit(pin: str) -> list[PatchVerdict]:
    """Audit every patch in ``PATCH_REGISTRY`` against ``pin``."""
    verdicts: list[PatchVerdict] = []
    for patch_id, entry in sorted(PATCH_REGISTRY.items()):
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title", "(no title)"))
        default_on = bool(entry.get("default_on", False))
        env_flag = entry.get("env_flag")
        applies_to = entry.get("applies_to") or {}
        vrange_raw = applies_to.get("vllm_version_range")

        if vrange_raw is None:
            verdicts.append(PatchVerdict(
                patch_id=patch_id, title=title, default_on=default_on,
                env_flag=env_flag, vllm_version_range=None,
                verdict="UNCONSTRAINED",
                reason="No vllm_version_range — applies on any pin",
                severity="info",
            ))
            continue

        if isinstance(vrange_raw, (list, tuple)) and len(vrange_raw) == 2:
            vrange: tuple[str, str] = (str(vrange_raw[0]), str(vrange_raw[1]))
        else:
            verdicts.append(PatchVerdict(
                patch_id=patch_id, title=title, default_on=default_on,
                env_flag=env_flag, vllm_version_range=None,
                verdict="RANGE_INVALID",
                reason=f"Malformed range: {vrange_raw!r}",
                severity="warn",
            ))
            continue

        result = _version_matches(pin, vrange)
        if result == "APPLIES":
            verdicts.append(PatchVerdict(
                patch_id=patch_id, title=title, default_on=default_on,
                env_flag=env_flag, vllm_version_range=vrange,
                verdict="APPLIES",
                reason=f"Pin {pin} satisfies {vrange[0]},{vrange[1]}",
                severity="info",
            ))
        elif result == "WOULD_SKIP":
            severity = "critical" if default_on else "warn"
            verdicts.append(PatchVerdict(
                patch_id=patch_id, title=title, default_on=default_on,
                env_flag=env_flag, vllm_version_range=vrange,
                verdict="WOULD_SKIP",
                reason=f"Pin {pin} is OUTSIDE {vrange[0]},{vrange[1]} — "
                       f"declarative consumers (doctor) will report skip; "
                       f"runtime dispatch may still apply if env flag is set",
                severity=severity,
            ))
        else:  # RANGE_INVALID
            verdicts.append(PatchVerdict(
                patch_id=patch_id, title=title, default_on=default_on,
                env_flag=env_flag, vllm_version_range=vrange,
                verdict="RANGE_INVALID",
                reason=f"Cannot parse range {vrange[0]},{vrange[1]} as PEP-440",
                severity="warn",
            ))
    return verdicts


def render_table(verdicts: list[PatchVerdict], pin: str) -> str:
    """Render verdicts as a fixed-width table."""
    # Group: critical first, then warn, then info
    order = {"critical": 0, "warn": 1, "info": 2}
    verdicts_sorted = sorted(
        verdicts,
        key=lambda v: (order.get(v.severity, 9), v.verdict, v.patch_id),
    )

    counts: dict[str, int] = {"APPLIES": 0, "WOULD_SKIP": 0,
                              "UNCONSTRAINED": 0, "RANGE_INVALID": 0}
    crit = 0
    for v in verdicts_sorted:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1
        if v.severity == "critical":
            crit += 1

    header = f"Genesis Patch Registry Version Audit — pin: {pin}\n"
    header += "=" * 72 + "\n"
    header += f"  total: {len(verdicts_sorted)}    "
    header += "    ".join(f"{k}: {v}" for k, v in counts.items() if v) + "\n"
    if crit:
        header += f"\n  ⚠ CRITICAL: {crit} default_on=True patches would skip on this pin\n"
    header += "\n"

    rows: list[str] = []
    rows.append(
        f"{'ID':<11} {'OnByDefault':<11} {'Verdict':<14} "
        f"{'Range':<26} Reason"
    )
    rows.append("-" * 72)
    for v in verdicts_sorted:
        if v.severity == "info" and v.verdict in ("APPLIES", "UNCONSTRAINED"):
            continue  # by default hide the all-green rows
        on = "YES" if v.default_on else "no"
        rng = (
            f"{v.vllm_version_range[0]},{v.vllm_version_range[1]}"
            if v.vllm_version_range else "—"
        )
        mark = "⚠" if v.severity == "critical" else (
            "·" if v.severity == "warn" else " ")
        rows.append(
            f"{mark} {v.patch_id:<10} {on:<11} {v.verdict:<14} "
            f"{rng:<26} {v.reason[:80]}"
        )
    if not any(v.severity in ("critical", "warn") for v in verdicts_sorted):
        rows.append("  (no warnings — clean for this pin)")
    return header + "\n".join(rows) + "\n"


def render_json(verdicts: list[PatchVerdict], pin: str) -> str:
    out = {
        "pin": pin,
        "total": len(verdicts),
        "verdicts": [
            {
                "patch_id": v.patch_id,
                "title": v.title,
                "default_on": v.default_on,
                "env_flag": v.env_flag,
                "vllm_version_range": list(v.vllm_version_range) if v.vllm_version_range else None,
                "verdict": v.verdict,
                "reason": v.reason,
                "severity": v.severity,
            } for v in verdicts
        ],
    }
    return json.dumps(out, indent=2, ensure_ascii=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pin", help="vllm version to audit against "
                    "(default: detect from live container)")
    ap.add_argument("--json", action="store_true",
                    help="Emit JSON instead of a table")
    ap.add_argument("--ci-strict", action="store_true",
                    help="Exit 1 if any default_on=True patch would skip "
                         "on the target pin")
    args = ap.parse_args()

    pin = args.pin or _detect_live_pin()
    if not pin:
        print("error: could not detect live pin; pass --pin <version>",
              file=sys.stderr)
        return 2

    verdicts = audit(pin)

    if args.json:
        print(render_json(verdicts, pin))
    else:
        print(render_table(verdicts, pin))

    if args.ci_strict:
        critical = [v for v in verdicts if v.severity == "critical"]
        if critical:
            print(f"\nCI-strict: {len(critical)} critical drift(s) — "
                  f"exit 1", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
