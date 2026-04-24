# Genesis v7.10 validation — qwen3_next_fp8

**Date**: 2026-04-24 20:19:19 UTC
**Container**: vllm-integration-v7
**Model**: qwen3.6-35b-a3b-integration
**Max ctx**: 262144

## Results at a glance

| Check | Result |
|---|---|
| Boot (Genesis applied) | [INFO:genesis.apply_all] Genesis Results: 28 applied, 4 skipped, 0 failed |
| Smoke 10/10 | 10/10 |
| Context sweep | see `context_sweep_full.jsonl` |
| Stress (Probe M) | see `stress_probe_m.jsonl` |
| Speed bench | see `speed_100k.jsonl` |
| Memory delta | before=24003 MiB, after=24003 MiB |
| P51 fires | 0
0 (expected: false) |
| P52 skips | 0
0 |
| P53 skips | 0
0 |

## Expected profile

| Attr | Expected |
|---|---|
| moe | true |
| hybrid | true |
| turboquant | true |

## Raw files

- [boot.log](./boot.log)
- [apply_all.log](./apply_all.log)
- [dispatch_profile.json](./dispatch_profile.json)
- [smoke.jsonl](./smoke.jsonl)
- [context_sweep_full.jsonl](./context_sweep_full.jsonl)
- [stress_probe_m.jsonl](./stress_probe_m.jsonl)
- [speed_100k.jsonl](./speed_100k.jsonl)
- [memory_profile.json](./memory_profile.json)
- [run.log](./run.log)
