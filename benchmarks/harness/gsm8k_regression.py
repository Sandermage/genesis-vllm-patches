# SPDX-License-Identifier: Apache-2.0
"""GSM8K regression harness — master plan Part 11.1 gate P0.

Threshold: ≥ baseline −0.5 percentage points.

Design:
  - Load the GSM8K test split (200 default problems, 5-shot).
  - Send each problem as an OpenAI-compatible chat request.
  - Parse the numeric answer from each response (strip commas, take the
    last number in the generated text).
  - Accuracy = correct / total.
  - Compare against `--baseline-accuracy` (default 0.70 — matches
    Qwen3.6-35B-A3B prod).

Usage:
  python -m benchmarks.harness.gsm8k_regression \\
      --endpoint http://192.168.1.10:8000/v1 \\
      --baseline-accuracy 0.70 --num-problems 200

Datasets:
  Bundled tiny GSM8K sample in `benchmarks/data/gsm8k_200.jsonl`. For full
  runs, provide `--dataset-path <file.jsonl>` pointing to the HF GSM8K
  test split in jsonl format.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from benchmarks.harness._common import (
    GateResult, HarnessReport, make_arg_parser, post_chat, probe_health,
    default_out_path, write_report, run_harness,
)

DEFAULT_BASELINE = 0.70  # Qwen3.6-35B-A3B v5.14.1 measured

GSM8K_5SHOT_PREFIX = (
    "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May?\n"
    "Answer: Natalia sold 48/2 = 24 clips in May. Total = 48 + 24 = 72. The answer is 72.\n\n"
    "Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\n"
    "Answer: Per minute she earns 12/60 = 0.2. 50 × 0.2 = 10. The answer is 10.\n\n"
    "Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. "
    "Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. "
    "How much more money does Betty need to buy the wallet?\n"
    "Answer: Betty has 100/2 = 50. Grandparents gave 15×2 = 30. Total = 50+15+30 = 95. She needs 100-95 = 5. The answer is 5.\n\n"
    "Question: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. "
    "If she wants to read half of the remaining pages tomorrow, how many pages should she read?\n"
    "Answer: Today = 12×2 = 24. Read = 12+24 = 36. Remaining = 120-36 = 84. Half = 84/2 = 42. The answer is 42.\n\n"
    "Question: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?\n"
    "Answer: Per week = 3×2×2 = 12. Per year = 12×52 = 624. The answer is 624.\n\n"
)


def _extract_number(text: str) -> float | None:
    """Extract last number from the model's response (strip commas)."""
    cleaned = text.replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _load_gsm8k(path: str, limit: int) -> list[dict]:
    items: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if len(items) >= limit:
                break
    return items


def _answer_number(full_answer: str) -> float | None:
    """Parse ground-truth answer: GSM8K format is '... #### <number>'."""
    if "####" in full_answer:
        tail = full_answer.split("####", 1)[1].strip()
        return _extract_number(tail)
    return _extract_number(full_answer)


def main() -> int:
    parser = make_arg_parser("gsm8k_regression")
    parser.add_argument(
        "--baseline-accuracy", type=float, default=DEFAULT_BASELINE,
        help=f"Baseline accuracy to compare against (default {DEFAULT_BASELINE}).",
    )
    parser.add_argument(
        "--dataset-path",
        default=os.path.join(
            os.path.dirname(__file__), "..", "data", "gsm8k_200.jsonl",
        ),
        help="Path to GSM8K JSONL file (question / answer fields).",
    )
    parser.add_argument(
        "--num-problems", type=int, default=200,
        help="How many problems to sample.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Max tokens per answer.",
    )
    parser.add_argument(
        "--drop-tolerance", type=float, default=0.005,
        help="Allowed drop below baseline (default 0.5 percentage points).",
    )
    args = parser.parse_args()
    out_path = args.out or default_out_path("gsm8k_regression")

    report = HarnessReport(
        name="gsm8k_regression",
        endpoint=args.endpoint,
        model=args.model,
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

        items = _load_gsm8k(args.dataset_path, args.num_problems)
        correct = 0
        wrong_samples: list[dict] = []
        for i, item in enumerate(items):
            q = item.get("question") or item["input"]
            a_full = item.get("answer") or item.get("target") or ""
            gt = _answer_number(a_full)
            if gt is None:
                continue
            prompt = GSM8K_5SHOT_PREFIX + f"Question: {q}\nAnswer:"
            resp = post_chat(
                endpoint=args.endpoint, api_key=args.api_key,
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_tokens,
                temperature=0.0,
                seed=42,
            )
            # Reasoning models (Qwen3) place the final answer in
            # `content` after internal `reasoning`; when max_tokens is
            # tight the answer lands in `reasoning` instead. Concatenate
            # so the "last number in output" heuristic picks up either.
            msg = resp["choices"][0]["message"]
            content = (
                (msg.get("content") or "")
                + "\n"
                + (msg.get("reasoning") or "")
            )
            pred = _extract_number(content)
            if pred is not None and abs(pred - gt) < 1e-3:
                correct += 1
            else:
                if len(wrong_samples) < 10:
                    wrong_samples.append({
                        "question": q[:200],
                        "ground_truth": gt,
                        "prediction": pred,
                        "response_tail": content[-200:],
                    })

        total = len(items)
        accuracy = correct / total if total else 0.0
        report.metrics = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "baseline_accuracy": args.baseline_accuracy,
            "drop_tolerance": args.drop_tolerance,
            "drop_actual": args.baseline_accuracy - accuracy,
        }
        report.gates = [
            GateResult(
                name="gsm8k_accuracy",
                value=accuracy,
                threshold=(
                    f">= {args.baseline_accuracy - args.drop_tolerance:.3f} "
                    f"(baseline {args.baseline_accuracy} − {args.drop_tolerance})"
                ),
                passed=accuracy >= args.baseline_accuracy - args.drop_tolerance,
            ),
        ]
        report.raw = {"wrong_samples": wrong_samples}
    except Exception as e:
        report.error = f"{type(e).__name__}: {e}"
        write_report(report, out_path, quiet=args.quiet)
        return 2

    write_report(report, out_path, quiet=args.quiet)
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
