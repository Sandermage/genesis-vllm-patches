# SPDX-License-Identifier: Apache-2.0
"""Genesis v7.0 benchmark harness.

A collection of scripts that implement the Part 11.1 pre-deploy validation
checklist from the master plan. Run each script against a live vLLM
endpoint (default: http://192.168.1.10:8000 / integration container) and
write results to `benchmarks/results/<timestamp>/` in JSON for later diff
against a baseline.

Contract for a harness script:
  1. `python -m benchmarks.harness.<name> --endpoint <url> [--out <path>]`
  2. Emits a JSON report with keys:
        "name":       harness identifier
        "endpoint":   URL under test
        "model":      resolved served-model-name
        "started_at": ISO timestamp
        "finished_at": ISO timestamp
        "metrics":    dict of measured metrics
        "go_no_go":   dict mapping gate name → bool + threshold line
        "raw":        per-sample data (optional)
  3. Exit code 0 if all gates pass, 1 if any critical gate fails, 2 on
     setup error (bad endpoint, auth failure, etc.).

Scripts:
  - gsm8k_regression.py       — GSM8K accuracy regression gate (P0)
  - tgs_decode.py             — decode tokens/sec at 160k context (P1)
  - long_context_oom.py       — 256k context stress test (P0)
  - quality_harness.py        — 33-prompt quality matrix (P0 ≥32/33)
  - offline_api_parity.py     — offline vs API deterministic match (P1)
  - cuda_graph_recapture.py   — recapture count stays 0 after warmup (P0)

Run everything sequentially via `make benchmarks` or CI job.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
