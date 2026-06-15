# Gemma-4 MTP — REAL dev491 A/B numbers (replaces the conflated "+65%") + 31B crash + G4_10 reframe

**Date**: 2026-06-16. Pin: dev491 image `nightly-1033ffac2`. Hardware: 2× A5000 SM 8.6, TP=2.
**Why**: operator pushback that prior study was superficial. This is the empirical, same-pin, same-context
verification that replaces the stale dev354 artifact ("+65%") with real numbers, and the live-binary
verification behind the G4_10 reframe.

## 26B-A4B no-MTP vs MTP-K3 (assistant drafter) — clean A/B, both @32K, dev491, n=3

| Workload      | no-MTP TPS | MTP-K3 TPS | Δ      | TPOT no-MTP→MTP (ms) |
|---|---:|---:|---:|---|
| thinking_off  | 119.3      | **196.9**  | **+65%** | 8.21 → 4.87 |
| thinking_on   | 119.4      | 195.6      | +64%   | 8.24 → 4.90 |
| code_gen      | 121.3      | 183.6      | +51%   | 8.13 → 5.31 |
| multi_turn    | 100.0      | 130.7      | +31%   | 8.00 → 5.09 |
| tool_call     | 88.9       | 93.7       | +5%    | 4.37 → 2.39 |
| short_chat    | 82.5       | 82.2       | ~0%    | 11.35 → 13.16 |
| long_gen      | 122.4      | 117.9      | **−4%**  | 8.12 → 8.42 |
| long_ctx_8k   | 63.8       | 59.8       | −6%    | 9.04 → 9.42 |
| long_ctx_32k  | 25.4       | 22.5       | **−11%** | 10.95 → 13.57 |

**Honest conclusion**: MTP-K3 is a big win for chat/thinking/code/multi-turn (+31% to +65%) and a small
LOSS for long-gen / long-context (−4% to −11%). The `validated_conditional` decision in the dev354 artifact
was correct in spirit. So MTP-K3 should be the default for short/medium generation (free_chat, thinking,
code, multi_turn) and gated OFF for long-context / long-gen. The headline "+65%" holds — but ONLY for
thinking/chat, not universally. (Prior session stated "+65%, make it the default" flatly — overstated;
corrected here with the full per-workload table.)
- The number was previously cited from an artifact on pin dev354/626fa9bba; this A/B confirms it reproduces
  on dev491 for thinking/chat specifically.
- Acceptance was NOT captured (the /metrics scrape for spec_decode counters returned empty — metric-name
  gap; the per-workload TPS deltas are the load-bearing evidence).

## 31B-tq-mtp-chat-k3 @64K — CRASHES on dev491 (stale overlay, not OOM/slow)

Container `Exited (1)`. Root cause:
```
ImportError: cannot import name 'get_kv_cache_capacity' from 'vllm.v1.core.kv_cache_utils'.
Did you mean: 'get_kv_cache_configs'?
```
The launcher bind-mounts `sndr/engines/vllm/patches/attention/turboquant/overlays/pr42637/kv_cache_utils.py`
over dev491's `vllm/v1/core/kv_cache_utils.py`. That overlay is a snapshot from an earlier pin and lacks
`get_kv_cache_capacity`, which dev491's own `vllm/v1/engine/core.py:45` imports → the overlay breaks
vLLM's import chain → hard boot crash. Another dev491 API drift (same class as G4_08/G4_10), but this one
CRASHES rather than silently no-ops. **31B-tq-mtp is NOT dev491-ready** until the pr42637 overlays are
refreshed to dev491's kv_cache_utils API (or the 31B runs without the TQ-KV overlay stack). Consistent with
its dev259 pin_hold. The 26B-mtp has no TQ overlays, so it is unaffected and benches fine (above).

## G4_10 reframe (commit 5b7901e8) — verified on the live binary
Live dev491 check (`docker run ... TritonAttentionBackend`): `supports_non_causal()=True`,
`supports_head_size(256)=True`, `supports_head_size(512)=True` (rule `h>=32`). So stock TRITON_ATTN does
everything the bespoke G4_10 kernel was for; the kernel was retired (it was a no-op on dev491 + redundant +
256-only + untested). G4_10 is now a clean enablement guard; G4_03's stale "TRITON_ATTN causal-only"
matrix row was corrected. EAGLE-3 drafts causally on dev491 (llm_base_proposer.py:1084,1195); only DFlash
is non-causal.

## Next
1. EAGLE-3 rig test on stock TRITON_ATTN (G4_10=1 + g4_71b/g4_75 routing) — boot + bench vs the real 26B
   MTP baseline above (thinking_off 196.9). The "+15-25% over MTP" club-3090 claim, now testable cheaply
   (no bespoke kernel).
2. Fix the pr42637 kv_cache_utils overlay for dev491 (`get_kv_cache_capacity`→`get_kv_cache_configs`) so
   31B-tq-mtp can boot on dev491 — OR test 31B-MTP without the TQ-KV overlay stack.
3. Make 26B MTP-K3 the chat-route default (workload-gated: ON for thinking/chat/code/multi_turn, OFF for
   long-context/long-gen).
