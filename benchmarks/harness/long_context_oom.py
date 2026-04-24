# SPDX-License-Identifier: Apache-2.0
"""256k context stress test — master plan Part 11.1 gate P0.

Threshold: No OOM at max-model-len 262144 with a single active request.

Design:
  - Send one request sized near max-model-len (default 256k tokens).
  - Confirm HTTP 200 and at least some generated output.
  - Any HTTP 500 / CUDA OOM / timeout = FAIL.

This is a smoke test; the real production validation depends on real
production traffic patterns, but this verifies the cliff identified in
#40420 doesn't regress.

Usage:
  python -m benchmarks.harness.long_context_oom \\
      --endpoint http://192.168.1.10:8000/v1 \\
      --context-tokens 262000 --gen-tokens 128

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import sys as _sys

from benchmarks.harness._common import (
    GateResult, HarnessReport, make_arg_parser, post_chat, probe_health,
    default_out_path, write_report, make_tokenizer_calibrated_filler,
)
from benchmarks.harness.tgs_decode import _make_filler


def main() -> int:
    parser = make_arg_parser("long_context_oom")
    parser.add_argument(
        "--context-tokens", type=int, default=262000,
        help="Approximate prompt size (tokens). Default 262000 (~256k).",
    )
    parser.add_argument(
        "--gen-tokens", type=int, default=128,
        help="Max output tokens. Small to keep test fast.",
    )
    args = parser.parse_args()
    out_path = args.out or default_out_path("long_context_oom")

    report = HarnessReport(
        name="long_context_oom", endpoint=args.endpoint, model=args.model,
    )
    try:
        if not probe_health(args.endpoint):
            report.error = f"/health check failed for {args.endpoint}"
            write_report(report, out_path, quiet=args.quiet)
            return 2

        # Tokenizer-calibrated filler — asks the server's tokenizer so
        # we don't overshoot/undershoot the target token count.
        # Falls back to the heuristic filler if /tokenize is unavailable.
        filler, measured = make_tokenizer_calibrated_filler(
            endpoint=args.endpoint, api_key=args.api_key, model=args.model,
            target_tokens=args.context_tokens,
        )
        report.metrics["filler_measured_tokens"] = measured

        resp = post_chat(
            endpoint=args.endpoint, api_key=args.api_key,
            model=args.model,
            messages=[{"role": "user", "content": filler}],
            max_tokens=args.gen_tokens,
            temperature=0.0,
            timeout=900.0,
        )
        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning") or ""
        usage = resp.get("usage") or {}
        completion_tokens = usage.get("completion_tokens") or 0
        # Qwen3 reasoning_parser emits content into `reasoning` — count either.
        ok = (len(content) + len(reasoning) > 0) or completion_tokens > 0
        report.metrics = {
            "output_chars": len(content),
            "reasoning_chars": len(reasoning),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": completion_tokens,
            "context_tokens_target": args.context_tokens,
        }
        report.gates = [
            GateResult(
                name="completes_without_oom_at_256k",
                value=ok,
                threshold="completion returned non-empty content or reasoning",
                passed=ok,
            ),
        ]
        report.raw = {
            "output_sample": content[:400],
            "reasoning_sample": reasoning[:400],
        }
    except Exception as e:
        report.error = f"{type(e).__name__}: {e}"
        # An HTTP 500/OOM surfaces as RuntimeError here; we treat that
        # as a test failure (gate=False) rather than a setup error.
        report.gates = [
            GateResult(
                name="completes_without_oom_at_256k",
                value=False,
                threshold="completion returned non-empty content",
                passed=False,
            ),
        ]
        write_report(report, out_path, quiet=args.quiet)
        return 1

    write_report(report, out_path, quiet=args.quiet)
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    _sys.exit(main())
