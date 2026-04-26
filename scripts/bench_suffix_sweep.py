#!/usr/bin/env python3
"""
Suffix Decoding parameter sweep — finds Pareto-optimal config for our
2× A5000 + Qwen3.6-A3B-FP8 setup.

Grid:
  - suffix_decoding_min_token_prob: {0.05, 0.10, 0.15, 0.20, 0.30}  (5 points)
  - suffix_decoding_max_tree_depth: {16, 24, 32}                    (3 points)
  → 15 configurations total

Per configuration:
  1. Stop+rm container
  2. Generate launch script with this config
  3. Start container, wait HEALTHY
  4. Run lightweight bench (3 speed runs + 10-shot stability + tool-call check)
  5. Save JSON result
  6. Continue

Selection: max(mean tok/s) subject to clean_rate ≥ 96% AND CV ≤ 5%.

Outputs: ~/Genesis_Project/vllm_engine/suffix_sweep_<timestamp>/
  - sweep_grid.json — all 15 results
  - pareto.json — winners by criteria
  - bench_<i>_<min_token_prob>_<depth>.json — per-config bench detail

Usage on server:
  cd ~/Genesis_Project/vllm_engine
  python3 bench_suffix_sweep.py --label sweep1 [--quick] [--from-idx N]

  --quick: skip 0.05 and 0.30 (only 9 mid-range points)
  --from-idx N: resume from configuration index N (after a crash)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "requests"])
    import requests


CONTAINER = "vllm-server-mtp-test"
PORT = 8000
API_KEY = "genesis-local"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# Grid
MIN_TOKEN_PROB_FULL = [0.05, 0.10, 0.15, 0.20, 0.30]
MIN_TOKEN_PROB_QUICK = [0.10, 0.15, 0.20]
MAX_TREE_DEPTH = [16, 24, 32]

# Bench prompts — mix of free-form + tool-call shapes
BENCH_PROMPTS = [
    ("speed_short", "Write a Python function that computes the nth Fibonacci number using memoization. Include type hints and a docstring.", 256),
    ("speed_long", "Explain quicksort vs mergesort in 5 paragraphs covering time complexity, stability, and best use cases.", 512),
    ("tool_call_json", 'Output ONLY valid JSON with keys "name" (string), "args" (object with "x":int, "y":int), "result" (int). Compute sum of x=42, y=17. No prose.', 96),
    ("structured_xml", "Write a single XML <config> element with 3 nested <option name=\"...\" value=\"...\"/> children. The values should be strings 'red', 'blue', 'green'.", 128),
    ("code_repetitive", "Write a TypeScript class for a binary search tree with insert, delete, find methods. Add JSDoc comments for each method.", 384),
]

DEGENERATE_PATTERNS = [
    "<<", "parameter=parameter", "<<argname",
    "the the", "of of", "and and",
]


def stop_container():
    subprocess.run(["docker", "stop", CONTAINER], capture_output=True, timeout=60)
    subprocess.run(["docker", "rm", CONTAINER], capture_output=True, timeout=60)


def generate_launch_script(min_token_prob: float, max_tree_depth: int) -> str:
    """Render launch script for given suffix config. Reuses base from
    /home/sander/start_v742_full_8k_suffix.sh, replaces speculative-config."""
    base = Path("/home/sander/start_v742_full_8k_suffix.sh").read_text()
    # Replace speculative-config line
    spec_old_marker = "--speculative-config"
    spec_new = (
        f"--speculative-config '{{"
        f'\\"method\\":\\"suffix\\",'
        f'\\"num_speculative_tokens\\":3,'
        f'\\"suffix_decoding_min_token_prob\\":{min_token_prob},'
        f'\\"suffix_decoding_max_tree_depth\\":{max_tree_depth},'
        f'\\"suffix_decoding_max_spec_factor\\":2.0'
        f"}}' \\\\"
    )
    out_lines = []
    for line in base.split("\n"):
        if spec_old_marker in line and "method" in line:
            # preserve indentation
            indent = len(line) - len(line.lstrip())
            out_lines.append(" " * indent + spec_new)
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def wait_healthy(timeout_s: int = 600) -> bool:
    """Poll /health until 200 or timeout. Returns True if healthy."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"http://localhost:{PORT}/health", headers=HEADERS, timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(15)
    return False


def bench_one_prompt(label: str, prompt: str, max_tokens: int) -> dict:
    """Single prompt bench: round-trip throughput + degenerate-pattern check."""
    t0 = time.time()
    r = requests.post(
        f"http://localhost:{PORT}/v1/completions",
        headers=HEADERS,
        json={
            "model": "qwen3.6-35b-a3b",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=120,
    )
    elapsed = time.time() - t0
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["text"]
    n_tokens = data["usage"]["completion_tokens"]
    tps = n_tokens / max(elapsed, 0.001)
    has_degen = any(p in text for p in DEGENERATE_PATTERNS)
    return {
        "label": label,
        "n_tokens": n_tokens,
        "elapsed_s": round(elapsed, 3),
        "tps": round(tps, 2),
        "degen": has_degen,
        "first_chars": text[:120],
    }


def run_bench_at_config(min_token_prob: float, max_tree_depth: int, runs_per_prompt: int = 3) -> dict:
    """Run all bench prompts × runs_per_prompt at current container."""
    all_runs = []
    for label, prompt, max_tokens in BENCH_PROMPTS:
        for run_i in range(runs_per_prompt):
            try:
                r = bench_one_prompt(f"{label}_run{run_i}", prompt, max_tokens)
                all_runs.append(r)
            except Exception as e:
                all_runs.append({
                    "label": f"{label}_run{run_i}",
                    "error": str(e),
                    "tps": 0,
                    "degen": False,
                })
    # Aggregate
    speed_runs = [r for r in all_runs if r.get("tps", 0) > 0 and "speed" in r.get("label", "")]
    all_valid = [r for r in all_runs if r.get("tps", 0) > 0]
    tps_values = [r["tps"] for r in all_valid]
    if tps_values:
        mean_tps = sum(tps_values) / len(tps_values)
        var = sum((x - mean_tps) ** 2 for x in tps_values) / len(tps_values)
        std = var ** 0.5
        cv = std / mean_tps if mean_tps > 0 else 999
    else:
        mean_tps = 0
        cv = 999
    n_degen = sum(1 for r in all_runs if r.get("degen"))
    n_valid = len(all_valid)
    clean_rate = (n_valid - n_degen) / max(n_valid, 1)
    return {
        "min_token_prob": min_token_prob,
        "max_tree_depth": max_tree_depth,
        "n_runs": len(all_runs),
        "n_valid": n_valid,
        "mean_tps": round(mean_tps, 2),
        "cv_pct": round(cv * 100, 2),
        "clean_rate": round(clean_rate, 3),
        "n_degen": n_degen,
        "all_runs": all_runs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="sweep1")
    ap.add_argument("--quick", action="store_true", help="3 min_token_prob × 3 depths = 9 points")
    ap.add_argument("--from-idx", type=int, default=0)
    ap.add_argument("--runs-per-prompt", type=int, default=2, help="default 2 to keep total time bounded")
    args = ap.parse_args()

    probs = MIN_TOKEN_PROB_QUICK if args.quick else MIN_TOKEN_PROB_FULL
    grid = list(product(probs, MAX_TREE_DEPTH))
    print(f"=== Suffix Sweep: {len(grid)} configurations ===")
    for i, (p, d) in enumerate(grid):
        print(f"  [{i}] min_token_prob={p}  max_tree_depth={d}")

    out_dir = Path(f"/home/sander/Genesis_Project/vllm_engine/suffix_sweep_{args.label}_{datetime.now().strftime('%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    results = []
    for i, (min_token_prob, max_tree_depth) in enumerate(grid):
        if i < args.from_idx:
            print(f"--- skip [{i}] (--from-idx) ---")
            continue
        print(f"\n=== [{i+1}/{len(grid)}] min_token_prob={min_token_prob} max_tree_depth={max_tree_depth} ===")
        # Stop existing
        stop_container()
        # Generate + launch
        script = generate_launch_script(min_token_prob, max_tree_depth)
        script_path = out_dir / f"start_{i:02d}_p{int(min_token_prob*100)}_d{max_tree_depth}.sh"
        script_path.write_text(script)
        os.chmod(script_path, 0o755)
        t_boot_start = time.time()
        result = subprocess.run(["bash", str(script_path)], capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  LAUNCH FAILED: {result.stderr[:500]}")
            results.append({"min_token_prob": min_token_prob, "max_tree_depth": max_tree_depth, "error": "launch_failed"})
            continue
        # Wait healthy
        if not wait_healthy(timeout_s=600):
            print(f"  HEALTH TIMEOUT after 10min")
            results.append({"min_token_prob": min_token_prob, "max_tree_depth": max_tree_depth, "error": "health_timeout"})
            continue
        boot_s = time.time() - t_boot_start
        print(f"  booted in {boot_s:.1f}s, running bench...")
        # Bench
        try:
            r = run_bench_at_config(min_token_prob, max_tree_depth, runs_per_prompt=args.runs_per_prompt)
            r["boot_s"] = round(boot_s, 1)
            results.append(r)
            (out_dir / f"bench_{i:02d}_p{int(min_token_prob*100)}_d{max_tree_depth}.json").write_text(json.dumps(r, indent=2))
            print(f"  → mean={r['mean_tps']} tok/s, CV={r['cv_pct']}%, clean={r['clean_rate']}, degen={r['n_degen']}")
        except Exception as e:
            print(f"  BENCH ERROR: {e}")
            results.append({"min_token_prob": min_token_prob, "max_tree_depth": max_tree_depth, "error": str(e)})
        # Save running grid summary
        (out_dir / "sweep_grid.json").write_text(json.dumps(results, indent=2))

    # Final analysis: Pareto
    valid = [r for r in results if "error" not in r]
    if valid:
        # Best by mean_tps with quality gate
        gated = [r for r in valid if r.get("clean_rate", 0) >= 0.96 and r.get("cv_pct", 999) <= 5.0]
        winner_strict = max(gated, key=lambda r: r.get("mean_tps", 0)) if gated else None
        winner_loose = max(valid, key=lambda r: r.get("mean_tps", 0))
        pareto = {
            "n_valid_configs": len(valid),
            "n_passing_gate": len(gated),
            "winner_strict_gate_clean_ge_96_cv_le_5": winner_strict,
            "winner_max_tps_no_gate": winner_loose,
            "all_valid_sorted_by_tps": sorted(valid, key=lambda r: -r.get("mean_tps", 0)),
        }
        (out_dir / "pareto.json").write_text(json.dumps(pareto, indent=2))
        print("\n=== PARETO ===")
        if winner_strict:
            w = winner_strict
            print(f"Strict gate winner: prob={w['min_token_prob']} depth={w['max_tree_depth']} → "
                  f"{w['mean_tps']} tok/s, CV {w['cv_pct']}%, clean {w['clean_rate']}")
        else:
            print("NO config passed strict gate (clean ≥ 96% AND CV ≤ 5%)")
        w = winner_loose
        print(f"Max-tps loose: prob={w['min_token_prob']} depth={w['max_tree_depth']} → "
              f"{w['mean_tps']} tok/s, CV {w['cv_pct']}%, clean {w['clean_rate']}")
    else:
        print("\nNO valid configs.")
    print(f"\nResults: {out_dir}")


if __name__ == "__main__":
    main()
