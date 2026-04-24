# Genesis v7.0 Benchmark Harness

Implements the Part 11.1 pre-deploy validation checklist from the master plan.

## Quick start

```bash
# Point at the integration container (VM 100)
export GENESIS_BENCH_ENDPOINT="http://192.168.1.10:8000/v1"
export GENESIS_BENCH_API_KEY="genesis-local"
export GENESIS_BENCH_MODEL="qwen3.6-35b-a3b-integration"

# Run everything sequentially (P0 + P1 gates)
python -m benchmarks.harness.run_all

# Or a single harness
python -m benchmarks.harness.gsm8k_regression --num-problems 50
python -m benchmarks.harness.tgs_decode --context-tokens 160000 --threshold 49
python -m benchmarks.harness.long_context_oom --context-tokens 262000
python -m benchmarks.harness.quality_harness
python -m benchmarks.harness.offline_api_parity
python -m benchmarks.harness.cuda_graph_recapture
```

## Output

Each harness writes a JSON report to
`benchmarks/results/<ISO_UTC>_<name>.json`:

```json
{
  "name": "gsm8k_regression",
  "endpoint": "http://192.168.1.10:8000/v1",
  "model": "qwen3.6-35b-a3b-integration",
  "started_at": "2026-04-24T12:00:00Z",
  "finished_at": "2026-04-24T12:05:00Z",
  "metrics": { "accuracy": 0.725, "correct": 145, "total": 200, ... },
  "gates": [
    {
      "name": "gsm8k_accuracy",
      "value": 0.725,
      "threshold": ">= 0.695 (baseline 0.70 − 0.005)",
      "passed": true
    }
  ],
  "raw": { "wrong_samples": [...] }
}
```

`run_all` additionally writes `summary.json` aggregating every harness.

## Go / No-Go thresholds (master plan Part 11.2)

| Harness               | Gate                              | Tier |
|----------------------|-----------------------------------|------|
| gsm8k_regression     | accuracy ≥ baseline − 0.005       | P0   |
| long_context_oom     | 256k req returns non-empty        | P0   |
| quality_harness      | ≥ 32 / 33 prompts pass            | P0   |
| cuda_graph_recapture | 0 recaptures after warmup         | P0   |
| tgs_decode           | decode t/s ≥ 49 at 160k           | P1   |
| offline_api_parity   | two runs byte-identical           | P1   |

Exit codes:
- `0` — all P0 gates passed (run_all)
- `1` — at least one P0 gate failed
- `2` — setup error (endpoint unreachable, dataset missing)

## Datasets

- `benchmarks/data/quality_33.jsonl` — 33 quality-matrix prompts
  (QA/code/reasoning/Russian/tool/writing/math/safety).
- `benchmarks/data/gsm8k_200.jsonl` — starter 10-problem GSM8K sample;
  replace with the real HuggingFace `gsm8k/test` split for a full run.

## Extending

To add a new harness, follow `_common.py` conventions:
1. Import `GateResult`, `HarnessReport`, `make_arg_parser`,
   `default_out_path`, `write_report`.
2. Populate `report.metrics` + `report.gates` inside a try/except.
3. Return `0 if report.all_passed else 1`.
4. Register in `SEQUENCE` in `run_all.py`.
