# SPDX-License-Identifier: Apache-2.0
"""Offline vs API parity — master plan Part 11.1 gate P1.

Threshold: 100% identical output when temperature=0 + same seed over a
fixed set of prompts issued twice (round 1 vs round 2).

Design:
  - Send each prompt twice with seed=42, temperature=0.
  - Expect byte-identical output for deterministic gates.
  - Differences → FAIL.

We can't practically run vLLM offline (Python in-process) from this
harness script without pulling in the whole engine; instead we treat
"API-twice-same-seed" as the parity proxy. This is sufficient to catch
non-determinism introduced by patches (P31 router fp32 upcast being the
main one).

Usage:
  python -m benchmarks.harness.offline_api_parity

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import sys as _sys

from benchmarks.harness._common import (
    GateResult, HarnessReport, make_arg_parser, post_chat, probe_health,
    default_out_path, write_report,
)

PROMPTS = [
    "Write a haiku about debugging.",
    "List three facts about the Ampere GPU architecture.",
    "Explain attention mechanism in one sentence.",
    "Translate 'hello world' to Russian.",
    "What is 17 * 23?",
]


def main() -> int:
    parser = make_arg_parser("offline_api_parity")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    out_path = args.out or default_out_path("offline_api_parity")

    report = HarnessReport(
        name="offline_api_parity", endpoint=args.endpoint, model=args.model,
    )
    try:
        if not probe_health(args.endpoint):
            report.error = f"/health check failed for {args.endpoint}"
            write_report(report, out_path, quiet=args.quiet)
            return 2

        mismatches: list[dict] = []
        for prompt in PROMPTS:
            r1 = post_chat(
                endpoint=args.endpoint, api_key=args.api_key,
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_tokens, temperature=0.0, seed=42,
            )
            c1 = r1["choices"][0]["message"].get("content") or ""
            r2 = post_chat(
                endpoint=args.endpoint, api_key=args.api_key,
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_tokens, temperature=0.0, seed=42,
            )
            c2 = r2["choices"][0]["message"].get("content") or ""
            if c1 != c2:
                mismatches.append({
                    "prompt": prompt,
                    "run_1_tail": c1[-120:],
                    "run_2_tail": c2[-120:],
                })

        total = len(PROMPTS)
        matched = total - len(mismatches)
        report.metrics = {
            "total_prompts": total,
            "matched": matched,
            "mismatches": len(mismatches),
        }
        report.gates = [
            GateResult(
                name="deterministic_api_twice",
                value=matched,
                threshold=f"=={total} matched",
                passed=matched == total,
            ),
        ]
        report.raw = {"mismatches": mismatches}
    except Exception as e:
        report.error = f"{type(e).__name__}: {e}"
        write_report(report, out_path, quiet=args.quiet)
        return 2

    write_report(report, out_path, quiet=args.quiet)
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    _sys.exit(main())
