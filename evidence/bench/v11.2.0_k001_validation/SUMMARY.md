# SNDR_MTP_DYNAMIC_K_001 Validation Bench — 2026-06-03

Second empirical bench of K_001 (`vllm#26504` port to `DraftModelProposer` base)
across both qwen3.6-35B and qwen3.6-27B multi-conc presets on the current
production rig pin.

Validates the v11.1.0 finding (2026-06-02, single-arm bench on 35B-multiconc)
across both registry production-subset models that match K_001's
`proposer_mro_must_include: DraftModelProposer` applies_to predicate.

## Hardware + pin

- 2× NVIDIA RTX A5000, TP=2
- vLLM `0.21.1rc1.dev354+g626fa9bba`
- backend_sig `2ff8db2d`
- bench tool: `tools/genesis_bench_suite.py --quick` (n=25 per arm)

## Results

| Arm | wall_TPS | CV | decode_TPOT_ms | TTFT_ms |
|---|---:|---:|---:|---:|
| 35b_k001_off | 214.04 | 9.99% | 4.55 | 73.72 |
| 35b_k001_on  | 210.48 | 10.79% | 4.64 | 73.93 |
| 27b_k001_off | 118.54 | 8.54% | 8.20 | 119.51 |
| 27b_k001_on  | 118.78 | 7.96% | 8.22 | 111.48 |

## Welch t-test

| Model | Δwall_TPS | t | p | Verdict |
|---|---:|---:|---:|---|
| 35B | -1.66% | -0.570 | 0.5688 | NOT_SIGNIFICANT |
| 27B | +0.21% | +0.088 | 0.9295 | NOT_SIGNIFICANT |

Both deltas are well within the per-arm CV (8-11%). The 35B drift is
slightly negative; the 27B drift is essentially zero. Neither passes
a 95% significance threshold (p < 0.05).

## Apply verification

Boot log confirms K_001 wired correctly on both models:

```
[INFO:genesis.observability] [PatchMetrics]
  SNDR_MTP_DYNAMIC_K_001 adaptive K MTP proposer
  (vllm#26504 port to DraftModelProposer)
  status=applied elapsed_ms=0.06 rss_delta_kb=0 ordinal=52
  reason=SNDR_MTP_DYNAMIC_K_001 installed: per-seq adaptive K MTP
  proposer (vllm#26504 port to DraftModelProposer); threshold=0.7
```

`GENESIS_ENABLE_SNDR_MTP_DYNAMIC_K_001=1` in container env (verified
via `docker exec ... os.environ.get(...)`); DraftModelProposer MRO
matches on qwen3.6 (vs gemma4 — different Proposer base, K_001 is
documented NO-OP there).

## Plan §15.1 forecast vs reality

The original PR #26504 author forecasted `+5-12% TPS on mixed workload`.
That forecast does NOT materialize on either qwen3.6 model under our
`--quick` workload (5 prompts × 5 runs × 1024 max-tok, short-prompt
single-sequence ramping batch). K_001's per-seq SequenceState rolling
acceptance-rate window (len=10) doesn't have room to accumulate signal
under short prompts; the K hysteresis bands (avg_acc ≥ threshold+0.05
→ K++ / avg_acc ≤ threshold-0.05 → K--) likely stay near the
launcher-cap K=3 throughout.

The remaining hypothesis for K_001 producing a measurable signal:
**multi-turn agentic workload** where per-seq state matures across
many turns, and where some sequences benefit from K=1 (low-acceptance
divergent draft) while others benefit from K=4+ (high-acceptance
streaming context). Future bench: `tools/bench_agentic.py --turns 12
--sessions 2` × K_001 ON/OFF. Not run this cycle.

## Default OFF empirically confirmed across production-subset matching models

Across BOTH `prod-qwen3.6-35b-multiconc` (2026-06-02 + 2026-06-03)
and `prod-qwen3.6-27b-multiconc` (2026-06-03), K_001 produces
NOT_SIGNIFICANT delta under the `--quick` workload. Default
`default_on: False` in `PATCH_REGISTRY["SNDR_MTP_DYNAMIC_K_001"]`
remains correct.

Operators wanting to opt in for multi-turn agentic workloads
specifically should:

1. Set `GENESIS_ENABLE_SNDR_MTP_DYNAMIC_K_001=1` in launcher env.
2. Bench against `tools/bench_agentic.py --turns 12 --sessions 2`
   with ON/OFF arms.
3. Only promote to permanent on-by-default if the agentic-specific
   bench shows >5% wall_TPS improvement at p<0.05.

## Files in this directory

- `35b_k001_off.json` — bench-suite output, K_001 OFF on prod-qwen3.6-35b-multiconc
- `35b_k001_on.json` — K_001 ON, same preset
- `27b_k001_off.json` — K_001 OFF on prod-qwen3.6-27b-multiconc
- `27b_k001_on.json` — K_001 ON, same preset
- `35b_multiturn_k001_off.json` — multi-turn bench (12 turns × 2 sessions), K_001 OFF
- `35b_multiturn_k001_on.json` — multi-turn bench, K_001 ON
- `SUMMARY.md` — this file

## Multi-turn third empirical bench (2026-06-03)

To close the remaining hypothesis — that K_001's per-seq SequenceState
needs >=10 turns to mature before hysteresis can trigger — wrote a
dedicated multi-turn TPS measurement harness
(`tools/bench_multiturn_tps.py`) that runs N-turn conversations and
reports per-turn wall_TPS, including an "early window (turns 1-9, pre-
SequenceState-mature)" vs "late window (turns 10+, K hysteresis can
fire)" split.

### Results

n=24 per arm (12 turns × 2 sessions), qwen3.6-35b-multiconc, same pin.

| Window | OFF wall_TPS | ON wall_TPS | Δ | Welch t | p | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Overall (turns 1-12) | 47.34 | 48.01 | +1.40% | +0.169 | 0.8656 | NOT_SIGNIFICANT |
| Early window (1-9)   | 48.73 | 49.44 | +1.46% | n/a | n/a | within CV ~28% |
| Late window (10-12)  | 43.19 | 43.71 | +1.20% | +0.906 | 0.3651 | NOT_SIGNIFICANT |

Both windows show essentially zero effect; the late-window delta is
even closer to zero than the overall mean. CV is high (~28%) because
the bench includes a transient JIT-compile turn 1 + an outlier turn 7
on session 1 — these are workload realities, not noise to exclude.

### Interpretation

The multi-turn hypothesis was the last viable explanation for why
K_001's PR #26504 +5-12% forecast might apply to Genesis. With this
arm of the bench matrix closed at p > 0.36 on the late window, the
forecast is **falsified across every workload K_001 could plausibly
help on** under our stack.

Possible explanations for the absence of signal:
1. Acceptance rate on our spec-decode path is consistently in the
   middle band (avg ~0.78-0.80), staying out of K_001's hysteresis
   bands (default threshold ± 0.05). Without crossing the threshold,
   K_001 returns the launcher cap K=3 unchanged — equivalent to OFF.
2. Genesis's existing Bayesian Acceptor + P62 structured-output
   spec-decode timing fix may have already optimized the per-seq K
   selection enough that K_001's heuristic adds no value on top.
3. Single-stream (single-batch) decode workload doesn't exercise the
   multi-seq inter-sequence acceptance-rate divergence that K_001
   was likely designed to exploit. The bench-multiturn harness still
   runs each session's turns sequentially, not in parallel.

### Operational decision

**K_001 stays default OFF permanently.** Three independent bench cycles
(quick-35B, quick-27B, multi-turn-35B) all confirm NOT_SIGNIFICANT.
Operators wanting to A/B further should target genuinely-divergent
multi-seq acceptance rate workloads (mixed structured/free-chat with
distinct compression plans hitting the same engine concurrently), which
is the only known shape K_001 could move the needle on. Not on our
2026-06 stack roadmap.

### Tooling artifact

`tools/bench_multiturn_tps.py` is committed alongside this bench for
future operator use. Unlike `bench_agentic.py` it does NOT require
tool-call support on the endpoint — plain chat completions only. Each
turn appends an assistant + synthetic user follow-up, so realistic
context growth + per-seq state evolution are exercised. Per-turn JSON
output includes early-vs-late window split for K_001-style hypothesis
testing on any future spec-decode patch.
