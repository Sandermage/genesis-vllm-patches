# v758 — P75 Suffix Decoding deploy variant

**Status:** READY ON SERVER, NOT YET TESTED IN SUSTAINED BENCH
**Source:** `/home/sander/launch_scripts/test/start_v758_p75_suffix.sh`
**Goal:** opt-in alternative to v748 PROD (MTP K=3) — Suffix Decoding via
P75 auto-swap.

---

## Why try Suffix Decoding

Per [arXiv 2411.04975](https://arxiv.org/abs/2411.04975) (Arctic
Inference) and [vllm#25784](https://github.com/vllm-project/vllm/pull/25784)
(merged in our pin):

- **Tool-call / heavy-repeat workloads:** +40-60% TPS over plain ngram
  (suffix tree captures longer repeats than fixed prompt_lookup_min)
- **Free-form text:** +15-25% over plain ngram with prompt_lookup_min=8
- Dynamic K speculation per-prompt (vs ngram's fixed
  num_speculative_tokens)

For Genesis prod (Qwen3.6-A3B-FP8, single-user agentic workloads),
expected: faster than v748 on tool-call traffic, comparable on
free-form. Worth a real-workload A/B vs MTP K=3 baseline.

## What v758 changes vs v748 PROD

| Knob | v748 PROD | v758 P75 |
|---|---|---|
| Speculative method | `mtp` K=3 | `ngram` → P75 swaps to `suffix` |
| Spec K | 3 (fixed) | dynamic (tree-based) |
| Acceptance | P82 SGLang OR-clause @ t=0.3 | inherits P82 if applicable |
| Prefix cache | OFF | OFF (same — avoids the v756 race) |
| KV cache dtype | turboquant_k8v4 | turboquant_k8v4 (same) |
| Genesis patch stack | 43 applied | 43 applied + P75 active |
| Extra dependency | — | `arctic-inference` (pip install at boot) |

Everything else identical: TQ k8v4, max-num-seqs=2, max-model-len 262144,
chunked-prefill, P67 multi-query kernel, P82 acceptance.

## Deployment

```bash
# Stop PROD
docker stop vllm-server-mtp-test
docker rm vllm-server-mtp-test

# Launch v758
bash /home/sander/launch_scripts/test/start_v758_p75_suffix.sh

# Wait ~2 min for boot (`Application startup complete`)
# Verify P75 swapped: docker logs | grep "P75" — should see
#   "[Genesis P75] swapped method: ngram -> suffix"

# Smoke test
curl -s http://localhost:8000/health
curl -s -H "Authorization: Bearer genesis-local" \
     -H "Content-Type: application/json" \
     -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}' \
     http://localhost:8000/v1/chat/completions
```

## Validation gates (before promoting to PROD)

1. **Boot health:** `Application startup complete` reached within 3 min,
   `arctic-inference` import succeeds (not silent fall-back to ngram).
2. **Speed bench:** `genesis_bench_v4.py --speed-runs 5 --skip-context
   --skip-sweep --skip-stability --skip-stress`. Compare avg tok/s vs
   v748 baseline.
3. **Quality:** `genesis_quality_harness.py` ≥ 30/31 (regression gate).
4. **Real-workload sample:** 10 tool-call prompts from agent traffic
   capture, verify clean tool_call XML output rate.
5. **Sustained stability:** stability test 30 sequential. (Skip
   stress-burst — that's the v756-class race we already isolated to
   TQ + cache + chunked + burst; v758 keeps cache OFF so should not hit
   it.)

## Rollback

```bash
docker stop vllm-server-mtp-test
docker rm vllm-server-mtp-test
bash /home/sander/launch_scripts/current/start_v748_p82_prod.sh
```

Rollback is exactly the PROD launch script — no state to clean up.

## Risks

- **`arctic-inference` install fails at boot** → P75 silently keeps
  method=ngram (per wiring docstring). Net effect: identical to running
  v748 without MTP, slower than PROD. Recoverable, no crash.
- **Suffix tree memory:** bounded by
  `suffix_decoding_max_cached_requests=10000` (default). Per-prompt
  trees, evicted LRU. Worst-case ~few hundred MB extra. Fits in our
  budget.
- **Acceptance heuristic interaction with P82:** P82 OR-clause threshold
  is for any spec method. With suffix's dynamic K, threshold may behave
  differently than with MTP fixed K=3. May want to A/B `t=0.3` vs `t=0.0`
  (disable P82) to see which works better with suffix.

## Decision criteria for promotion

Promote v758 → PROD only if ALL of:
- Speed bench shows ≥ +20% TPS over v748 on tool-call workload
- Free-form bench shows ≤ -5% TPS regression vs v748
- Quality ≥ 30/31
- Stability 30/30
- Real-workload sample looks clean

Otherwise keep v748 PROD, leave v758 as alternative endpoint for
power-users who explicitly want it.

## Pickup checklist (when ready to test)

- [ ] Confirm Sander OK with another ~10 min PROD downtime + bench window
- [ ] Stop PROD, launch v758
- [ ] Verify P75 swapped (boot logs)
- [ ] Run 5-bench gates above
- [ ] If any gate fails → write up reason, restart PROD
- [ ] If all gates pass → ask Sander whether to promote (involves
      `start_v748_p82_prod.sh` rename / new "current/" entry)
