# SPDX-License-Identifier: Apache-2.0
"""Decode TGS (tokens/sec) harness — master plan Part 11.1 gate P1.

Threshold: ≥ 49 tokens/sec decode at 160k context (Qwen3.6 prod baseline).

Design:
  - Prime a large context (default 160k tokens) of repetitive filler text
    that is semantically trivial but exercises long-context decode memory
    pressure.
  - Issue a streaming completion with small max_tokens (default 256).
  - Measure TTFT vs total time; decode TGS = (max_tokens - 1) / (total - ttft).

Usage:
  python -m benchmarks.harness.tgs_decode \\
      --endpoint http://192.168.1.10:8000/v1 \\
      --context-tokens 160000 --gen-tokens 256 --threshold 49

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import random
import string

from benchmarks.harness._common import (
    GateResult, HarnessReport, make_arg_parser, post_completion_stream,
    probe_health, default_out_path, write_report,
)


def _make_filler(approx_tokens: int) -> str:
    """Build a filler prompt of approximately `approx_tokens` tokens.

    vLLM tokenizers typically produce ~0.75 tokens per word of English-like
    filler. We emit paragraphs of nonsense words until we cross the token
    budget at ~1.33 words per token heuristic.
    """
    random.seed(42)
    words_needed = int(approx_tokens * 1.33) + 32
    vocab = [
        "".join(random.choices(string.ascii_lowercase, k=5))
        for _ in range(1000)
    ]
    out = []
    i = 0
    while len(out) < words_needed:
        out.append(vocab[i % len(vocab)])
        i += 1
    return " ".join(out) + "\n\nPlease continue writing: "


def main() -> int:
    parser = make_arg_parser("tgs_decode")
    parser.add_argument(
        "--context-tokens", type=int, default=160000,
        help="Approximate context length in tokens (prompt size).",
    )
    parser.add_argument(
        "--gen-tokens", type=int, default=256,
        help="How many output tokens to generate for timing.",
    )
    parser.add_argument(
        "--threshold", type=float, default=49.0,
        help="Minimum decode tokens/sec to pass the gate.",
    )
    parser.add_argument(
        "--warmup-runs", type=int, default=1,
        help="Warmup runs before timing (discarded).",
    )
    parser.add_argument(
        "--timed-runs", type=int, default=3,
        help="Timed runs — best run used (dampens noise).",
    )
    args = parser.parse_args()
    out_path = args.out or default_out_path("tgs_decode")

    report = HarnessReport(
        name="tgs_decode", endpoint=args.endpoint, model=args.model,
    )
    try:
        if not probe_health(args.endpoint):
            report.error = f"/health check failed for {args.endpoint}"
            write_report(report, out_path, quiet=args.quiet)
            return 2

        filler = _make_filler(args.context_tokens)

        # Warmup
        for _ in range(args.warmup_runs):
            post_completion_stream(
                args.endpoint, args.api_key, args.model, filler,
                max_tokens=args.gen_tokens, temperature=0.0,
            )

        # Timed runs
        samples: list[dict] = []
        best_tgs = 0.0
        for _ in range(args.timed_runs):
            ttft, total_tokens, total = post_completion_stream(
                args.endpoint, args.api_key, args.model, filler,
                max_tokens=args.gen_tokens, temperature=0.0,
            )
            decode_sec = max(total - ttft, 1e-9)
            # total_tokens includes first one (arriving at TTFT); subtract 1
            # to treat decode as post-first-token generation only.
            decode_toks = max(total_tokens - 1, 0)
            tgs = decode_toks / decode_sec if decode_sec > 0 else 0.0
            samples.append({
                "ttft_sec": ttft,
                "total_sec": total,
                "decode_sec": decode_sec,
                "output_tokens": total_tokens,
                "tgs": tgs,
            })
            if tgs > best_tgs:
                best_tgs = tgs

        report.metrics = {
            "best_tgs": best_tgs,
            "threshold": args.threshold,
            "samples": samples,
            "context_tokens": args.context_tokens,
            "gen_tokens": args.gen_tokens,
        }
        report.gates = [
            GateResult(
                name="decode_tgs_at_long_context",
                value=best_tgs,
                threshold=f">= {args.threshold} t/s",
                passed=best_tgs >= args.threshold,
            ),
        ]
    except Exception as e:
        report.error = f"{type(e).__name__}: {e}"
        write_report(report, out_path, quiet=args.quiet)
        return 2

    write_report(report, out_path, quiet=args.quiet)
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
