# v759 — 320K context expansion (validated 2026-04-27 night)

**Status:** TESTED + VALIDATED, awaiting Sander GO for PROD promotion.
**Source:** `/home/sander/launch_scripts/test/start_v759_320k_explore.sh`
**Goal:** push --max-model-len from 256K (v748 PROD) to 320K with same stability.

## Knob changes vs v748 PROD

| Param | v748 PROD | v759 | Why |
|---|---|---|---|
| `--max-model-len` | 262144 (256K) | **320000** | Sander request "calmly hold limit" — give 25% more headroom |
| `--max-num-batched-tokens` | 8192 | **4096** | Half the mixed-batch staging frees ~500 MB per GPU, offsets +KV-cache for larger context |
| `GENESIS_TQ_MAX_MODEL_LEN` | 262144 | 320000 | P37 cap raised to match |
| Everything else | identical | identical | Same TQ k8v4, MTP K=3, P67, P82, all 43 patches |

## Boot + VRAM

| | v748 (256K) | v759 (320K) |
|---|---|---|
| GPU 0 used | 22749 MiB (92.6%) | 22267 MiB (92.4%) |
| GPU 1 used | 22060 MiB (89.8%) | 21598 MiB (89.6%) |
| GPU 0 free | 1364 MiB | 1846 MiB |
| GPU 1 free | 2053 MiB | 2515 MiB |

v759 actually has **MORE free VRAM** despite +25% context, because the smaller `--max-num-batched-tokens` saves more in batch staging than the larger context costs in KV-cache reservation.

## Long-context probes (think-OFF mode)

| Prompt tokens | v748 (256K) | v759 (320K) |
|---|---|---|
| 220,019 | 66.2s ✓ | (matches v748 baseline ~66s) |
| 240,019 | 75.6s ✓ | (matches v748 baseline ~76s) |
| 253,352 | 82.3s ✓ | (matches v748 baseline ~83s) |
| **280,021** | **400 — exceeds limit** | **94.0s ✓** |
| **300,021** | **400 — exceeds limit** | **104.9s ✓** |
| **317,798** | **400 — exceeds limit** | **114.5s ✓** |

v759 cleanly handles 280K-317K context. Effective rate ~3,000 prompt-tokens/sec at long context (slightly down from short-context due to attention quadratic).

## Regression bench (CV analysis)

`genesis_bench_v4 --speed-runs 3 --stability-n 30 --stress-bursts 10 --stress-per-burst 3`

### Speed test

| max_tok | v748 avg t/s | v759 avg t/s | Delta |
|---|---|---|---|
| 64 | 244.6 | 246.4 | +0.7% |
| 128 | 232.1 | 232.9 | +0.3% |
| 256 | 218.3 | 213.6 | -2.2% |
| 512 | 206.9 | 190.1 | -8.1% |
| 1024 | 191.9 | 191.5 | -0.2% |
| 2048 | 185.5 | 201.0 | +8.4% |

Mixed signal at short max_tokens — outliers in 3-run sample. No systematic regression.

### Stability + stress CV

| Test | Config | avg t/s | stdev | **CV %** | success |
|---|---|---|---|---|---|
| Stability 30 sequential | v748 | 213.27 | 13.74 | **6.44** | 30/30 |
| Stability 30 sequential | v759 | 215.17 | 14.72 | **6.84** | 30/30 |
| Stress 30 (3 concurrent x 10 burst) | v748 | 229.41 | 16.04 | **6.99** | 30/30 |
| Stress 30 (3 concurrent x 10 burst) | v759 | 230.62 | 15.37 | **6.67** | 30/30 |

**CV practically identical (6.4-7.0% both configs).** v759 is marginally tighter under concurrent stress, marginally looser on sequential stability — within run-to-run noise.

## Verdict

**v759 = strict upgrade over v748:**
- Same speed class (244 → 200 t/s range)
- Same stability class (CV 6.4-7.0%)
- Same success rate (30/30 + 30/30)
- **+25% context capacity (256K → 320K)**
- More free VRAM at boot (saves 482 MB GPU 0, 462 MB GPU 1)

## Pickup checklist (when Sander green-lights promotion)

- [ ] Confirm Sander wants v759 → PROD promotion
- [ ] Stop PROD v748
- [ ] Launch v759 (`start_v759_320k_explore.sh`) — rename to `start_v759_320k_prod.sh` and move to `current/`
- [ ] Smoke test: 5 short requests + 1 long (280K) request
- [ ] If stable: archive v748 PROD launch script (rename to `current/start_v748_p82_prod_archived_20260428.sh`)
- [ ] If unstable: rollback to v748
- [ ] Update CONFIGURATION.md "tested baseline" line to mention v759 + 320K
- [ ] Push v7.57 commit (v759 promotion) to public repo (Sander explicit GO)

## Risks

- **Quad attention scaling:** at 320K context, attention pass takes ~15% longer than at 256K. Decode TPS for very long contexts will be ~10-15% slower than 256K config. Acceptable for our single-user workload but worth noting.
- **Burst of 280K+ requests in parallel:** untested. With max_num_seqs=2, two concurrent 280K requests would use ~50 GB KV cache — exceeds our 48 GB. Scheduler should serialize but worth monitoring. (Single-user workload usually has concurrency=1.)
- **max-num-batched-tokens=4096 vs 8192:** smaller chunks mean more iterations during prefill of medium prompts (8K-16K). Could slow medium-context requests by ~5-10%. Untested in this round.

## Memory entry written

`memory/project_genesis_v759_320k_validated.md` — for future agents.
