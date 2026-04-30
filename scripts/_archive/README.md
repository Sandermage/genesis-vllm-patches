# `scripts/_archive/` — Superseded scripts

Scripts in this directory are **superseded** by newer canonical implementations
elsewhere in the repo. Kept for git history / reference. Do not invoke.

## Contents

| Old script | Superseded by | Why archived |
|---|---|---|
| `genesis_bench_v3.py` | `tools/genesis_bench_suite.py` | bytewise-identical to v4; both predate the canonical suite. Audit 2026-04-30 flagged as dead. |
| `genesis_bench_v4.py` | `tools/genesis_bench_suite.py` | bytewise-identical to v3; same reason. |

## How to use the canonical bench

```bash
# Via unified CLI (recommended):
python3 -m vllm._genesis.compat.cli bench --quick

# Or direct:
python3 tools/genesis_bench_suite.py --quick
```

See [`docs/BENCHMARK_GUIDE.md`](../../docs/BENCHMARK_GUIDE.md) for the full
operator guide.
