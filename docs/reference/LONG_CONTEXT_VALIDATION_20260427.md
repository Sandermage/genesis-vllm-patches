# Long-context validation — 2026-04-27 night

**PROD config:** v748 (TQ k8v4 + MTP K=3 + P82 t=0.3 + cache OFF)
**`--max-model-len` 262144** (256K)
**Tested by:** Sander request "Требуется сделать всё чтобы можно было отправить 220K+ контекста и получить ответ"

## TL;DR

✅ **220K, 240K, 253K context all work** in BOTH thinking ON and thinking OFF modes.

✅ Performance same in both modes (66-83 sec for prompt processing).

✅ No regression observed; PROD v748 stable at 256K config (already at limit since deploy).

## Test results

### Long-context probe — both modes

| Prompt tokens | think-ON time | think-OFF time | Notes |
|---|---|---|---|
| 220,019 | 66.3s | 66.2s | Both succeeded |
| 240,019 | 76.3s | 75.6s | Both succeeded |
| 253,352 | 83.0s | 82.3s | Just under 256K limit, both OK |
| ~260,000 | 400 Bad Request | n/a | Exceeds `--max-model-len 262144` |

### Reasoning vs no-reasoning behavior

**think-ON (default Qwen3 reasoning):**
- Output split into `reasoning` + `content` fields
- max_tokens=30 → all 30 tokens go to reasoning, content is empty
- For long-context Q&A you typically want max_tokens >= 200 with reasoning enabled

**think-OFF (`chat_template_kwargs.enable_thinking=false`):**
- Direct answer in `content`, `reasoning` is empty
- max_tokens=30 → 2-3 tokens used (`"No\n"`), clean response
- Recommended for short-answer / classification / yes-no on long context
- ~0.7s saved on output (no reasoning overhead) at any prompt length

### VRAM usage at 256K config

```text
GPU 0: 22749 MiB used / 24564 total = 92.6% (1364 MiB free)
GPU 1: 22060 MiB used / 24564 total = 89.8% (2053 MiB free)
```

After 204K probe:
```text
GPU 0: 23531 MiB used / 24564 total = 95.8% (582 MiB free)
GPU 1: 22842 MiB used / 24564 total = 93.0% (1271 MiB free)
```

**Headroom interpretation:**
- Idle (after boot): ~1.4 GB free per GPU
- After 204K-token request: ~600 MB free on GPU 0 (decode workspace etc.)
- 256K config is at PRACTICAL upper limit for 24GB cards
- Pushing higher (e.g., 320K or 384K) would require `gpu_memory_utilization` adjustments OR smaller max_num_batched_tokens; net result likely OOM under burst

### Throughput at long context

Wall-time = prompt-processing dominated. Effective rate:
- 220K: 220019 / 66.3s ≈ **3,318 prompt-tokens/sec**
- 240K: 240019 / 76.3s ≈ **3,144 prompt-tokens/sec**
- 253K: 253352 / 83.0s ≈ **3,053 prompt-tokens/sec**

Slightly degrading with length (cache pressure + attention quadratic). Acceptable for our single-user workload.

## Recommendation for users sending 220K+

1. **Always send `chat_template_kwargs: {enable_thinking: false}` for short answers.** Saves a few hundred ms and avoids max_tokens overflow into reasoning.
2. **For analytical answers with reasoning:** allow `max_tokens >= 1024` so the model has room to think + answer.
3. **Don't send 256K+** — hits hard limit. Truncate input or split workload.
4. **Avoid burst** of 200K+ requests in parallel — VRAM will OOM (we're at 93-96% post-prompt). max_num_seqs=2 limits concurrency, but two 200K prompts at once would saturate KV cache.

## Future optimization candidates (#17 backlog)

To push above 256K WITHOUT hardware change:

a. **Lower model.weights memory** — already FP8 quantized (Marlin). No gain available.
b. **Smaller `max-num-batched-tokens`** — currently 8192. Reducing to 4096 would shrink mixed-batch staging buffers ~50%, freeing maybe 200-400 MB. Could enable `--max-model-len 320000` (~+25%).
c. **Disable some Genesis prealloc patches** (P28, P36, P38, P39a, P40, P44, P46) — frees ~200 MB but reintroduces malloc churn at runtime.
d. **Use `--gpu-memory-utilization 0.95`** instead of 0.90 — more aggressive, frees ~1.2 GB per GPU. Risky: can OOM during continuation_prefill at high prompt utilization.

None of these are quick-win. Best path to >256K = R6000 Pro Blackwell 96GB (per `project_genesis_hardware_upgrade_plan`).

## Status

PROD v748 256K context is stable, validated 2026-04-27 night.
**No changes needed to PROD.** This document confirms the existing config meets Sander's "220K+ works reliably" requirement.

Sustained-load regression bench follow-up running — will add results when complete.
