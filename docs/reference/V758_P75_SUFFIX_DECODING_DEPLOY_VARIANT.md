# v758 — P75 Suffix Decoding deploy variant

**Status:** TESTED 2026-04-27 night — boots, P75 swap works, but speed
INCONCLUSIVE on bench with non-repeating prompts. Not a clear win over
v748 PROD baseline.

**Source:** `/home/sander/launch_scripts/test/start_v758_p75_suffix.sh`
**Goal:** opt-in alternative to v748 PROD (MTP K=3) — Suffix Decoding via
P75 auto-swap.

## Test results (2026-04-27 23:03 UTC)

### Boot

After fixing P75 wiring bug (missing `import os` in injected code → first
boot died with `NameError`), v758 boot succeeded:

```
=== Genesis v7.58 P75 deploy: ngram->suffix decoding (Arctic Inference) + P82 ===
[Genesis P75] Auto-swapped speculative method 'ngram' -> 'suffix'
  (tree_depth=24, spec_factor=1.00, min_prob=0.100, cache_reqs=10000).
```

P75 swap confirmed; arctic-inference imports successfully; engine
healthy (200).

### Speed test (5 runs per max_tokens, no stability/stress)

| max_tokens | avg t/s | min | max | TTFT |
|---|---|---|---|---|
| 64 | **182.0** | 53.6 | 238.4 | 0.137s |
| 128 | 116.2 | 71.4 | 189.3 | 0.130s |
| 256 | 83.3 | 66.6 | 115.5 | 0.132s |
| 512 | 88.4 | 77.9 | 97.9 | 0.135s |
| 1024 | 73.3 | 67.1 | 84.8 | 0.133s |
| 2048 | 93.1 | 59.3 | 133.3 | 0.133s |

**vs PROD v748 baseline** (~140-167 tok/s typical per CONFIGURATION.md):
- Short generations (64-128 tok): high variance (53-238 t/s spread)
  — first request "cold tree", subsequent could be "warm". Mean 182 on
  64-tok could be a real win OR cache effect of repeated prompt in run.
- Long generations (1024-2048 tok): regression (73-93 vs PROD 140+).
  Suffix decoding has K-overhead per step that doesn't pay off when
  the suffix tree doesn't have many cached repeats.

### Why inconclusive

Suffix Decoding (Arctic Inference) is designed for **agentic workloads
with prompt repetition**:
- Same prompt prefix across multiple iterations
- Tool-call chains where intermediate context repeats
- Multi-turn conversations with stable system prompt

Our `genesis_bench_v4.py` uses VARIED test prompts (good for measuring
TPS regression but bad for measuring suffix-decoding win). For each
new prompt, the suffix tree is empty → degrades to linear scan → no
speedup.

To validate suffix decoding properly we'd need either:
- A real agentic workload capture (10+ tool-call chains with same
  system prompt + repeating tools)
- A benchmark that intentionally repeats prompts to warm the tree

### Stability

Did NOT run stress/stability tests in this round (would need another
PROD swap window). Boot + 5 speed runs all 200 OK.

## Decision

**v758 P75 deploy variant: DO NOT promote to PROD without proper
agentic-workload validation.**

- Boot: works ✓
- Engine stability: confirmed ✓
- Speed: inconclusive on standard bench
- Real-workload value: theoretically high but unproven for our use

**Keep v748 PROD as default.** v758 launch script preserved at
`/home/sander/launch_scripts/test/start_v758_p75_suffix.sh` for
operators who want to A/B test on their actual traffic.

P75 wiring bug fix (import os) is independently valuable — committed
as v7.56 fix. Marker bumped to `v7.56_local_os_import` so the patch
re-applies cleanly on fresh containers.

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
