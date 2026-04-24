# SPDX-License-Identifier: Apache-2.0
"""Run every Part 11.1 harness sequentially and aggregate results.

Order (fast → slow):
  1. offline_api_parity     — ~10 seconds
  2. quality_harness        — ~1 min
  3. gsm8k_regression       — ~5 min (200 problems)
  4. tgs_decode             — ~3 min (warmup + 3 timed at 160k)
  5. long_context_oom       — ~5 min (one 256k request)
  6. cuda_graph_recapture   — ~5 min (250 small requests)

Exit code: 0 if all P0 gates pass; 1 if any P0 fails; 2 on setup error.

Usage:
  python -m benchmarks.harness.run_all --endpoint http://192.168.1.10:8000/v1

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from datetime import datetime
from typing import Any

SEQUENCE = [
    ("offline_api_parity", "P1"),
    ("quality_harness", "P0"),
    ("gsm8k_regression", "P0"),
    ("tgs_decode", "P1"),
    ("long_context_oom", "P0"),
    ("cuda_graph_recapture", "P0"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        default=os.environ.get(
            "GENESIS_BENCH_ENDPOINT",
            "http://192.168.1.10:8000/v1",
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GENESIS_BENCH_API_KEY", "genesis-local"),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "GENESIS_BENCH_MODEL",
            "qwen3.6-35b-a3b-integration",
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for per-harness JSON + aggregate. Default: "
             "benchmarks/results/<ISO>_run_all/",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated harness names; runs only listed ones.",
    )
    args = parser.parse_args()

    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "results"),
    )
    os.makedirs(root, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or os.path.join(root, f"{ts}_run_all")
    os.makedirs(out_dir, exist_ok=True)

    only = (
        set(x.strip() for x in args.only.split(",") if x.strip())
        if args.only else None
    )

    summary: dict[str, Any] = {
        "started_at": datetime.utcnow().isoformat() + "Z",
        "endpoint": args.endpoint,
        "model": args.model,
        "results": [],
    }
    p0_fails: list[str] = []
    for name, tier in SEQUENCE:
        if only and name not in only:
            continue
        print(f"=== [{tier}] {name} ===", flush=True)
        # Import the harness's main and pass sys.argv override
        mod = importlib.import_module(f"benchmarks.harness.{name}")
        out_path = os.path.join(out_dir, f"{name}.json")
        saved_argv = sys.argv
        sys.argv = [
            name,
            "--endpoint", args.endpoint,
            "--api-key", args.api_key,
            "--model", args.model,
            "--out", out_path,
            "--quiet",
        ]
        t0 = time.perf_counter()
        try:
            rc = mod.main()
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 2
        finally:
            sys.argv = saved_argv
        dur = time.perf_counter() - t0

        # Parse the harness-emitted JSON
        summary_entry = {
            "name": name, "tier": tier, "exit": rc, "duration_sec": dur,
        }
        if os.path.isfile(out_path):
            with open(out_path) as f:
                summary_entry["result"] = json.load(f)
        else:
            summary_entry["result"] = {"error": "no report produced"}
        summary["results"].append(summary_entry)

        if tier == "P0" and rc != 0:
            p0_fails.append(name)
            print(
                f"    ❌ P0 gate FAILED: {name} (exit={rc})", flush=True,
            )
        elif rc == 0:
            print(f"    ✓ PASS ({dur:.1f}s)", flush=True)
        else:
            print(
                f"    ⚠ non-P0 tier={tier} exit={rc}", flush=True,
            )

    summary["finished_at"] = datetime.utcnow().isoformat() + "Z"
    summary["p0_fails"] = p0_fails

    agg_path = os.path.join(out_dir, "summary.json")
    with open(agg_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("", flush=True)
    print(f"Aggregated summary → {agg_path}", flush=True)
    if p0_fails:
        print(f"❌ P0 gates failed: {p0_fails}", flush=True)
        return 1
    print("✅ All P0 gates passed.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
