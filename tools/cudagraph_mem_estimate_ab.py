#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""cudagraph_mem_estimate_ab.py — vllm#45197 measure-first A/B harness.

Upstream vllm#45197 (OPEN, CHANGES_REQUESTED as of 2026-06-11) wants to
flip the VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS default to 0 because
the estimator's linear extrapolation
(``per_graph_estimate[mode] = per_graph * (len(descs) - 1)``,
gpu_model_runner.py) ignores CUDA mempool block reuse and OVERESTIMATES
— on B200/Minimax the reporter lost 3.1x KV capacity. Upstream review
pushed back: the estimate is the post-warmup OOM protection added in
v0.21.0. Roadmap verdict (2026-06-11 pr-sweep, chunk 1): do NOT flip
configs blindly — MEASURE =1 vs =0 on 35B/27B first; only if a
>~200 MiB overestimate is confirmed, write the root-cause patch
(capture-all-and-measure for descs <= 16). Our short capture-size lists
(P66 filter) make both the bug small and the exact-measure fix cheap.

This tool boots NOTHING itself. It has two modes:

  emit-env     Print (or write) the TWO launcher env permutations:
                 arm estimate-on:  VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
                 arm estimate-off: VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
               Both arms force VLLM_LOGGING_LEVEL=DEBUG: the free-VRAM
               lines ("Initial free memory", "Free memory after
               profiling") and the per-mode estimate breakdown are
               logger.debug, and PROD launchers run WARNING which would
               suppress every line diff-report parses. The rig stage
               applies one arm per boot (same model YAML, same
               --gpu-memory-utilization, all else identical), captures
               `docker logs` after warmup, then feeds both logs back.

  diff-report  Parse the two boot logs for the estimate lines +
               free-VRAM lines (exact pin 0.22.1rc1.dev259+g303916e93
               format strings, byte-verified):
                 - "Profiling CUDA graph memory: FULL=N (largest=M)..."
                 - "Estimated <MODE> CUDA graph memory: X MiB
                    first-capture + (N-1) x Y MiB per-graph"   [DEBUG]
                 - "Estimated CUDA graph memory: X GiB total"
                 - "Initial free memory: X GiB; Requested memory:
                    U (util), Y GiB"                            [DEBUG]
                 - "Free memory after profiling: X GiB (total),
                    Y GiB (within requested)"                   [DEBUG]
                 - "Available KV cache memory: X GiB"
                 - "GPU KV cache size: N tokens"
                 - "Maximum concurrency for N tokens per request: X.XXx"
                 - "Graph capturing finished in N secs, took X GiB"
                 - the enabled-INFO / disabled-WARNING advisory lines
                   (also used to verify the operator didn't swap arms)
               and compute:
                 - kv_gib_recovered / kv_tokens_recovered (=0 vs =1):
                   what disabling the estimate frees for TQ k8v4 KV
                 - overestimate_gib = estimate_total - actual capture
                   cost ("took X GiB"): THE #45197 verdict number
                 - verdict vs --threshold-mib (default 200):
                   OVERESTIMATE_CONFIRMED -> root-cause patch
                   candidate per the roadmap; WITHIN_TOLERANCE ->
                   keep =1 (OOM protection is worth its price);
                   INSUFFICIENT_DATA -> re-run with DEBUG logging.

NOTE — the pin runs profile_cudagraph_memory() in BOTH arms (the flag
only gates whether the estimate is SUBTRACTED from the KV budget,
gpu_worker.py "cudagraph_memory_estimate_applied"), so the estimate
lines appear in both logs; only the applied KV budget differs.

INCONSISTENCY FINDING (documented per the roadmap, measure-first — DO
NOT flip either value before the A/B):
  - sndr/compat/presets.py, preset "3090-1x-long-context", system_env:
      "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "0"
    (community 1x 3090 rig: estimate disabled — max KV on 24 GB at the
    cost of post-warmup OOM protection).
  - sndr/model_configs/builtin/hardware/
    a5000-2x-24gbvram-16cpu-128gbram.yaml (PROD, both Qwen3.6 models)
    and a5000-1x-24gbvram-16cpu-128gbram.yaml:
      VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: '1'
    (estimate enabled — protective sizing).
  The two profiles answer the same 24 GB trade-off in opposite
  directions with no recorded measurement behind either. This harness
  exists to settle it with numbers; the winning value gets recorded in
  reference_metrics evidence before any YAML/preset edit.

Usage:
    python3 tools/cudagraph_mem_estimate_ab.py emit-env \
        [--format env|shell|docker] [--out-dir DIR]
    python3 tools/cudagraph_mem_estimate_ab.py diff-report \
        --log-on boot_estimate_on.log --log-off boot_estimate_off.log \
        [--threshold-mib 200] [--json]

Exit codes:
    0 — success (any verdict; a confirmed overestimate is a finding,
        not a failure)
    2 — usage error: missing/unreadable log, or the logs' advisory
        lines contradict the --log-on/--log-off assignment (swapped
        arms would sign-flip the report)

Fully offline — no docker, no SSH, no GPU; never boots an engine.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

FLAG = "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"

GIB_PER_MIB = 1.0 / 1024.0


# ─── emit-env ─────────────────────────────────────────────────────────


def launcher_env_arms() -> list[dict[str, Any]]:
    """The two launcher env permutations for the A/B boots."""
    common = {
        # DEBUG is required: the free-VRAM lines and the per-mode
        # estimate breakdown are logger.debug in the pin.
        "VLLM_LOGGING_LEVEL": "DEBUG",
    }
    return [
        {
            "name": "estimate-on",
            "description": (
                "estimate applied to KV budget (a5000 YAML status quo)"
            ),
            "env": {**common, FLAG: "1"},
        },
        {
            "name": "estimate-off",
            "description": (
                "estimate NOT applied (#45197 proposed default / "
                "presets.py 3090 status quo)"
            ),
            "env": {**common, FLAG: "0"},
        },
    ]


def _render_arm(arm: dict[str, Any], fmt: str) -> str:
    lines = [f"# arm: {arm['name']} — {arm['description']}"]
    for key, value in arm["env"].items():
        if fmt == "docker":
            lines.append(f"  -e {key}={value} \\")
        elif fmt == "shell":
            lines.append(f"export {key}={value}")
        else:  # env
            lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


def render_env_arms(fmt: str) -> str:
    return "\n".join(_render_arm(arm, fmt) for arm in launcher_env_arms())


# ─── boot-log parsing ─────────────────────────────────────────────────

# Pin 0.22.1rc1.dev259+g303916e93 format strings (byte-verified).
_RE_ESTIMATE_TOTAL = re.compile(
    r"Estimated CUDA graph memory: ([\d.]+) GiB total"
)
_RE_PER_MODE = re.compile(
    r"Estimated (\w+) CUDA graph memory: ([\d.]+) MiB first-capture "
    r"\+ \((\d+)-1\) [x×] ([\d.]+) MiB per-graph"
)
_RE_PROFILING_GROUPS = re.compile(r"Profiling CUDA graph memory: (.+)$")
_RE_AVAILABLE_KV = re.compile(r"Available KV cache memory: ([\d.]+) GiB")
_RE_KV_TOKENS = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
_RE_MAX_CONC = re.compile(
    r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x"
)
_RE_ACTUAL_CAPTURE = re.compile(
    r"Graph capturing finished in \d+ secs, took ([\d.]+) GiB"
)
_RE_INITIAL_FREE = re.compile(
    r"Initial free memory: ([\d.]+) GiB; Requested memory: "
    r"([\d.]+) \(util\), ([\d.]+) GiB"
)
_RE_FREE_AFTER = re.compile(
    r"Free memory after profiling: ([\d.]+) GiB \(total\), "
    r"(-?[\d.]+) GiB \(within requested\)"
)
_RE_FLAG_ON = re.compile(r"CUDA graph memory profiling is enabled")
_RE_FLAG_OFF = re.compile(
    r"CUDA graph memory profiling is disabled\s*"
    r"\(VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0\)"
)
# Both advisory lines end in the suggested-util number.
_RE_SUGGESTED_UTIL_ON = re.compile(
    r"increase\s+--gpu-memory-utilization to ([\d.]+)\."
)
_RE_SUGGESTED_UTIL_OFF = re.compile(
    r"increasing --gpu-memory-utilization from [\d.]+ to ([\d.]+)\."
)


def _first_float(pattern: re.Pattern, text: str) -> Optional[float]:
    m = pattern.search(text)
    return float(m.group(1)) if m else None


def parse_boot_log(text: str) -> dict[str, Any]:
    """Extract the estimate + free-VRAM lines from one boot log.

    Capacity values that may repeat per TP rank (info_once still
    surfaces once per worker process under multiproc) are reduced with
    min() — the conservative bound; the actual capture cost uses max().
    """
    per_mode = [
        {
            "mode": m.group(1),
            "first_capture_mib": float(m.group(2)),
            "num_graphs": int(m.group(3)),
            "per_graph_mib": float(m.group(4)),
        }
        for m in _RE_PER_MODE.finditer(text)
    ]

    kv_values = [float(v) for v in _RE_AVAILABLE_KV.findall(text)]
    token_values = [
        int(v.replace(",", "")) for v in _RE_KV_TOKENS.findall(text)
    ]
    capture_values = [float(v) for v in _RE_ACTUAL_CAPTURE.findall(text)]

    flag_state: Optional[str] = None
    if _RE_FLAG_ON.search(text):
        flag_state = "on"
    elif _RE_FLAG_OFF.search(text):
        flag_state = "off"

    suggested_util = _first_float(_RE_SUGGESTED_UTIL_ON, text)
    if suggested_util is None:
        suggested_util = _first_float(_RE_SUGGESTED_UTIL_OFF, text)

    initial = _RE_INITIAL_FREE.search(text)
    profiling = _RE_PROFILING_GROUPS.search(text)

    return {
        "estimate_total_gib": _first_float(_RE_ESTIMATE_TOTAL, text),
        "per_mode": per_mode,
        "profiling_groups": profiling.group(1).strip() if profiling else None,
        "available_kv_gib": min(kv_values) if kv_values else None,
        "kv_tokens": min(token_values) if token_values else None,
        "max_concurrency": _first_float(_RE_MAX_CONC, text),
        "actual_capture_gib": max(capture_values) if capture_values else None,
        "initial_free_gib": (
            float(initial.group(1)) if initial else None
        ),
        "requested_util": float(initial.group(2)) if initial else None,
        "requested_gib": float(initial.group(3)) if initial else None,
        "free_after_profiling_gib": _first_float(_RE_FREE_AFTER, text),
        "flag_state": flag_state,
        "suggested_util": suggested_util,
    }


# ─── diff report ──────────────────────────────────────────────────────


def build_report(
    parsed_on: dict[str, Any],
    parsed_off: dict[str, Any],
    threshold_mib: float = 200.0,
) -> dict[str, Any]:
    """Deltas + verdict from the two parsed boot logs (pure, no I/O)."""
    kv_on = parsed_on.get("available_kv_gib")
    kv_off = parsed_off.get("available_kv_gib")
    kv_gib_recovered = (
        kv_off - kv_on if kv_on is not None and kv_off is not None else None
    )

    tokens_on = parsed_on.get("kv_tokens")
    tokens_off = parsed_off.get("kv_tokens")
    kv_tokens_recovered = (
        tokens_off - tokens_on
        if tokens_on is not None and tokens_off is not None
        else None
    )

    estimate_total = parsed_on.get("estimate_total_gib")
    if estimate_total is None:
        estimate_total = parsed_off.get("estimate_total_gib")
    actual_capture = parsed_on.get("actual_capture_gib")
    if actual_capture is None:
        actual_capture = parsed_off.get("actual_capture_gib")

    overestimate_gib = (
        estimate_total - actual_capture
        if estimate_total is not None and actual_capture is not None
        else None
    )

    if overestimate_gib is None or kv_gib_recovered is None:
        verdict = "INSUFFICIENT_DATA"
    elif overestimate_gib > threshold_mib * GIB_PER_MIB:
        verdict = "OVERESTIMATE_CONFIRMED"
    else:
        verdict = "WITHIN_TOLERANCE"

    return {
        "tool": "cudagraph_mem_estimate_ab v1",
        "upstream_pr": 45197,
        "threshold_mib": threshold_mib,
        "kv_gib_recovered": kv_gib_recovered,
        "kv_tokens_recovered": kv_tokens_recovered,
        "estimate_total_gib": estimate_total,
        "actual_capture_gib": actual_capture,
        "overestimate_gib": overestimate_gib,
        "verdict": verdict,
        "verdict_meaning": {
            "OVERESTIMATE_CONFIRMED": (
                "estimate exceeds measured capture cost beyond threshold "
                "— write the root-cause capture-all-and-measure patch "
                "(descs <= 16) per the #45197 roadmap; do NOT just flip "
                "the flag"
            ),
            "WITHIN_TOLERANCE": (
                "estimate is honest on this stack — keep =1 (post-warmup "
                "OOM protection); reconcile presets.py to match"
            ),
            "INSUFFICIENT_DATA": (
                "missing estimate/KV lines — re-run boots with "
                "VLLM_LOGGING_LEVEL=DEBUG (see emit-env) and full logs"
            ),
        }[verdict],
        "arms": {"on": parsed_on, "off": parsed_off},
    }


def _print_human(report: dict[str, Any]) -> None:
    err = sys.stderr

    def fmt(value, suffix=""):
        return "n/a" if value is None else f"{value:.3f}{suffix}"

    print("=" * 72, file=err)
    print("vllm#45197 cudagraph memory-estimate A/B report", file=err)
    print("-" * 72, file=err)
    on = report["arms"]["on"]
    off = report["arms"]["off"]
    print(f"{'metric':38} {'=1 (on)':>14} {'=0 (off)':>14}", file=err)
    for label, key, suffix in (
        ("Available KV cache memory (GiB)", "available_kv_gib", ""),
        ("GPU KV cache size (tokens)", "kv_tokens", ""),
        ("Max concurrency (x)", "max_concurrency", ""),
        ("Estimated CG memory total (GiB)", "estimate_total_gib", ""),
        ("Actual capture cost (GiB)", "actual_capture_gib", ""),
        ("Free after profiling (GiB)", "free_after_profiling_gib", ""),
    ):
        left = on.get(key)
        right = off.get(key)
        ls = "n/a" if left is None else f"{left}"
        rs = "n/a" if right is None else f"{right}"
        print(f"{label:38} {ls:>14} {rs:>14}", file=err)
    print("-" * 72, file=err)
    print(
        f"KV recovered by =0: {fmt(report['kv_gib_recovered'], ' GiB')} "
        f"({report['kv_tokens_recovered']} tokens)",
        file=err,
    )
    print(
        f"Overestimate (estimate - actual): "
        f"{fmt(report['overestimate_gib'], ' GiB')} "
        f"(threshold {report['threshold_mib']:.0f} MiB)",
        file=err,
    )
    print(f"VERDICT: {report['verdict']} — {report['verdict_meaning']}",
          file=err)
    print("=" * 72, file=err)


# ─── CLI ──────────────────────────────────────────────────────────────


def _cmd_emit_env(args: argparse.Namespace) -> int:
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for arm in launcher_env_arms():
            path = out_dir / f"cudagraph_ab_{arm['name']}.env"
            path.write_text(_render_arm(arm, args.format))
            print(str(path))
        return 0
    print(render_env_arms(args.format))
    return 0


def _cmd_diff_report(args: argparse.Namespace) -> int:
    parsed: dict[str, dict[str, Any]] = {}
    for arm_name, raw_path in (("on", args.log_on), ("off", args.log_off)):
        path = Path(raw_path)
        if not path.is_file():
            print(
                f"cudagraph_mem_estimate_ab: --log-{arm_name} "
                f"{path} is not a readable file",
                file=sys.stderr,
            )
            return 2
        parsed[arm_name] = parse_boot_log(
            path.read_text(encoding="utf-8", errors="replace")
        )

    # Arm-assignment sanity: the advisory lines in each log state which
    # flag value the boot actually ran. Swapped files would sign-flip
    # every delta in the report.
    for arm_name, expected in (("on", "on"), ("off", "off")):
        actual = parsed[arm_name]["flag_state"]
        if actual is not None and actual != expected:
            print(
                f"cudagraph_mem_estimate_ab: --log-{arm_name} advisory "
                f"line says the boot ran with the estimate {actual!r} — "
                "the arm files appear swapped; fix the assignment.",
                file=sys.stderr,
            )
            return 2
        if actual is None:
            print(
                f"cudagraph_mem_estimate_ab: WARNING: --log-{arm_name} "
                "carries no flag advisory line (estimate may be 0 or "
                "log level too high); arm assignment unverified.",
                file=sys.stderr,
            )

    report = build_report(
        parsed["on"], parsed["off"], threshold_mib=args.threshold_mib
    )
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print_human(report)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "vllm#45197 measure-first A/B harness for "
            "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS — boots nothing; "
            "emits launcher env permutations and diffs two boot logs."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_env = sub.add_parser(
        "emit-env", help="print/write the two launcher env permutations"
    )
    p_env.add_argument(
        "--format", choices=("env", "shell", "docker"), default="env"
    )
    p_env.add_argument(
        "--out-dir", default=None,
        help="write one .env file per arm instead of stdout",
    )

    p_diff = sub.add_parser(
        "diff-report", help="diff two boot logs (=1 arm vs =0 arm)"
    )
    p_diff.add_argument("--log-on", required=True,
                        help="boot log of the =1 (estimate applied) arm")
    p_diff.add_argument("--log-off", required=True,
                        help="boot log of the =0 arm")
    p_diff.add_argument("--threshold-mib", type=float, default=200.0,
                        help="overestimate threshold for the verdict "
                             "(roadmap: ~200 MiB)")
    p_diff.add_argument("--json", action="store_true",
                        help="print the JSON report to stdout")
    p_diff.add_argument("--json-out", default=None,
                        help="also write the JSON report to this path")

    try:
        args = parser.parse_args(argv)
    except SystemExit:
        return 2

    if args.command == "emit-env":
        return _cmd_emit_env(args)
    return _cmd_diff_report(args)


if __name__ == "__main__":
    sys.exit(main())
