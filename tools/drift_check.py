#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare a live engine install against a committed manifest; report drift.

Usage::

    # Check current vllm install against the manifest for a specific pin
    python3 tools/drift_check.py --engine vllm --pin 0.22.1_da1daf40b

    # Output JSON for CI consumption
    python3 tools/drift_check.py --engine vllm --pin auto --output drift.json

Exit codes:
    0  No drift detected
    1  Drift detected (one or more files have changed md5)
    2  Invocation error (manifest missing, install missing, etc.)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def compute_file_md5(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324
    h.update(path.read_bytes())
    return h.hexdigest()


def check_drift(
    manifest_path: Path,
    install_root: Path,
) -> dict:
    """Compare manifest against live files.

    Returns:
        Report dict with per-file status.
    """
    manifest = yaml.safe_load(manifest_path.read_text())
    results = {}
    drift_count = 0
    missing_count = 0
    ok_count = 0

    for rel, file_data in manifest.get("files", {}).items():
        abs_path = install_root / rel
        if not abs_path.is_file():
            results[rel] = {"severity": "blocked", "reason": "file missing in live install"}
            missing_count += 1
            continue

        live_md5 = compute_file_md5(abs_path)
        expected_md5 = file_data.get("md5")

        if live_md5 != expected_md5:
            results[rel] = {
                "severity": "drift",
                "expected_md5": expected_md5,
                "actual_md5": live_md5,
            }
            drift_count += 1
        else:
            results[rel] = {"severity": "ok"}
            ok_count += 1

    return {
        "engine": manifest.get("engine"),
        "pin": manifest.get("pin"),
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": results,
        "summary": {
            "ok": ok_count,
            "drift": drift_count,
            "blocked": missing_count,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--engine", default="vllm")
    p.add_argument("--pin", required=True, help="Pin identifier OR 'auto' to use the current detected pin")
    p.add_argument("--install-root", help="Path to engine install (default: import the package)")
    p.add_argument("--output", type=Path, help="Write JSON report to this path")
    args = p.parse_args()

    repo_root = Path(__file__).parent.parent

    # Resolve install root
    if args.install_root:
        install_root = Path(args.install_root)
    else:
        if args.engine == "vllm":
            try:
                import vllm  # type: ignore
                install_root = Path(vllm.__file__).parent
            except ImportError:
                print("ERROR: vllm not installed; use --install-root", file=sys.stderr)
                return 2
        else:
            print(f"ERROR: --install-root required for engine '{args.engine}'", file=sys.stderr)
            return 2

    # Resolve pin
    pin = args.pin
    if pin == "auto":
        # Try to detect from sndr adapter
        try:
            from sndr.config import SndrConfig
            from sndr.engines import get_engine
            EngineCls = get_engine(args.engine)
            config = SndrConfig.from_env()
            engine = EngineCls(config=config)
            pin = engine._normalize_pin(engine.detect_version())  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: auto-detection failed: {e}", file=sys.stderr)
            return 2

    manifest_path = (
        repo_root / "sndr" / "engines" / args.engine / "pins" / pin / "manifest.yaml"
    )
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found at {manifest_path}", file=sys.stderr)
        print("       Run: python3 tools/manifest_gen.py --engine X --pin Y", file=sys.stderr)
        return 2

    report = check_drift(manifest_path, install_root)

    if args.output:
        args.output.write_text(json.dumps(report, indent=2))

    summary = report["summary"]
    if summary["drift"] > 0 or summary["blocked"] > 0:
        print(f"DRIFT DETECTED: {summary['drift']} files drifted, {summary['blocked']} missing")
        if not args.output:
            print(json.dumps(report, indent=2))
        return 1

    print(f"OK: {summary['ok']} files match manifest; no drift detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
