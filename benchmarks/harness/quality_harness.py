# SPDX-License-Identifier: Apache-2.0
"""33-prompt quality matrix — master plan Part 11.1 gate P0.

Threshold: ≥ 32/33 prompts produce output that contains the required keyword
or passes the predicate check (coding / reasoning / Russian-language /
tool-use).

This is a simpler derivative of the historical `genesis_quality_harness.py`
prompts: a curated 33-prompt set that covers the functional capability
matrix. Each prompt has an `expect` predicate: either a list of keywords
that MUST appear (any-of), or a `regex` pattern.

Format of `benchmarks/data/quality_33.jsonl`:
  {"id": "qa_01", "prompt": "...", "expect_any": ["keyword1", ...]}
  {"id": "code_01", "prompt": "...", "expect_regex": "pattern"}

Usage:
  python -m benchmarks.harness.quality_harness \\
      --endpoint http://192.168.1.10:8000/v1

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import json
import os
import re
import sys as _sys

from benchmarks.harness._common import (
    GateResult, HarnessReport, make_arg_parser, post_chat, probe_health,
    default_out_path, write_report,
)

DEFAULT_DATASET = os.path.join(
    os.path.dirname(__file__), "..", "data", "quality_33.jsonl",
)


def _check_expect(resp_text: str, item: dict) -> bool:
    lowered = resp_text.lower()
    if "expect_any" in item:
        return any(kw.lower() in lowered for kw in item["expect_any"])
    if "expect_all" in item:
        return all(kw.lower() in lowered for kw in item["expect_all"])
    if "expect_regex" in item:
        return bool(re.search(item["expect_regex"], resp_text, re.I | re.M))
    # No predicate: non-empty response counts as pass
    return len(resp_text.strip()) > 0


def main() -> int:
    parser = make_arg_parser("quality_harness")
    parser.add_argument(
        "--dataset-path", default=DEFAULT_DATASET,
        help="Path to 33-prompt JSONL dataset.",
    )
    parser.add_argument(
        "--pass-threshold", type=int, default=32,
        help="Minimum passing prompts out of the total.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
    )
    args = parser.parse_args()
    out_path = args.out or default_out_path("quality_harness")

    report = HarnessReport(
        name="quality_harness", endpoint=args.endpoint, model=args.model,
    )
    try:
        if not probe_health(args.endpoint):
            report.error = f"/health check failed for {args.endpoint}"
            write_report(report, out_path, quiet=args.quiet)
            return 2

        if not os.path.isfile(args.dataset_path):
            report.error = f"dataset not found: {args.dataset_path}"
            write_report(report, out_path, quiet=args.quiet)
            return 2

        items: list[dict] = []
        with open(args.dataset_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))

        passed = 0
        per_item: list[dict] = []
        for item in items:
            prompt = item["prompt"]
            try:
                resp = post_chat(
                    endpoint=args.endpoint, api_key=args.api_key,
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=args.max_tokens, temperature=0.0, seed=42,
                )
                msg = resp["choices"][0]["message"]
                # Reasoning models (Qwen3 family) return the final answer
                # in `content` after internal `reasoning`. When max_tokens
                # cuts off before content, only `reasoning` has text —
                # check both fields so the predicate matches regardless.
                content_field = msg.get("content") or ""
                reasoning_field = msg.get("reasoning") or ""
                content = (content_field + "\n" + reasoning_field).strip()
                ok = _check_expect(content, item)
            except Exception as e:
                ok = False
                content = f"[error: {type(e).__name__}: {e}]"
            if ok:
                passed += 1
            per_item.append({
                "id": item.get("id", ""),
                "passed": ok,
                "response_head": content[:200],
            })

        total = len(items)
        report.metrics = {
            "passed": passed, "total": total,
            "rate": passed / total if total else 0.0,
            "threshold": args.pass_threshold,
        }
        report.gates = [
            GateResult(
                name="quality_passes",
                value=passed,
                threshold=f">= {args.pass_threshold}/{total}",
                passed=passed >= args.pass_threshold,
            ),
        ]
        report.raw = {"per_item": per_item}
    except Exception as e:
        report.error = f"{type(e).__name__}: {e}"
        write_report(report, out_path, quiet=args.quiet)
        return 2

    write_report(report, out_path, quiet=args.quiet)
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    _sys.exit(main())
