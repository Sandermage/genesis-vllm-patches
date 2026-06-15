# Gemma-4 spec-decode — EAGLE-3 detour, G4_03 root-cause, corrected plan

**Date**: 2026-06-16. Pin dev491 (image), but 26B/31B Gemma-4 still pin-held on dev259.
**Hardware**: 2× RTX A5000 24GB, SM 8.6 Ampere, TP=2, no NVLink.
**Trigger**: user directive "найти решения для ускорения гемма 4" + loop "наши ядра изучи может что-то упустили".

## TL;DR — we already had the right answer; EAGLE-3 was a wrong turn

The native **Gemma-4 MTP assistant drafter** (`Gemma4Proposer`, vllm#41745, causal → Ampere-safe)
is the correct spec-decode path for Gemma-4 on our hardware, and the project already ships it:

| Model | Profile | Drafter | Status |
|---|---|---|---|
| 26B-A4B | `gemma4-26b-mtp-chat-k3` | `/models/gemma-4-26B-A4B-it-assistant`, method=mtp K=3 | **+65% (198 vs 119.5 TPS) on dev491 — WORKS** |
| 31B | `gemma4-31b-tq-mtp-chat-k3` | `/models/gemma-4-31B-it-assistant`, method=mtp K=3 + TQ-KV @64K | assistant-drafter path; pending dev491 re-validation |

**EAGLE-3 is a dead end on Ampere+Gemma4** and the project already knew it (patch **G4_03**).

## What happened

1. Following "реализуй лучший подход", I implemented EAGLE-3 profiles (gemma4-{26b,31b}-eagle3)
   using the official RedHatAI speculator heads + vLLM's native eagle3 loader (PR#39450), reasoning
   that EAGLE-3 reads target hidden states and sidesteps the MTP-on-TQ-shared-KV 0%-collapse.
2. Rig validation (bench bqwscfp7l): **both** profiles BOOT-FAILED (UNHEALTHY 540s each). 35B PROD
   restored healthy afterward (verified end-to-end: authenticated completion `2+2`→`4`, finish=stop).
3. Boot-failure root cause (from the retained `Exited (1)` container logs):

   ```
   RuntimeError: [Genesis G4_03] Refusing EAGLE-3 drafter on Ampere SM 8.6 with Gemma 4 target.
   EAGLE-3 uses non-causal block-parallel attention. Gemma 4 has head_dim=256 (sliding) and
   head_dim=512 (global). No Ampere SM 8.6 attention backend supports both — see vllm#40382:
     FA2 / FA2_DIFFKV / FLASHINFER / TRITON_ATTN / TREE_ATTN — causal-only.
     FLEX_ATTENTION supports both but is slow enough that draft-acceptance gain is wiped out.
   RECOMMENDED — native Gemma 4 MTP assistant drafter (Gemma4Proposer):
     method: mtp / model: /models/Gemma-4-31B-it-assistant / num_speculative_tokens: 8
     ... Landed upstream via vllm#41745 (MERGED).
   ```

   `sndr/engines/vllm/patches/model_compat/gemma4/g4_03_gemma4_ampere_non_causal_drafter_guard.py:226`.

### G4_03 — the guard we missed
Default-on, `conflicts_with: [G4_10]`. Wraps `SpecDecodeBaseProposer._create_draft_vllm_config` and
refuses at config-build time when `method in ("eagle3","dflash")` (causal proposers — MTP/EAGLE-1/
draft_model/ngram — pass through). It encodes the full Ampere backend matrix (vllm#40382) and names
both the recommended path (MTP assistant drafter) and the deep fix (G4_10 — a Genesis non-causal
head_dim=256 Triton kernel, currently NOT implemented / conflicts_with G4_03).

**Lesson (iron-rule-#11 / Study-first):** the external research swept upstream PRs but did not read our
own `gemma4` patch family first. G4_03 already captured this exact analysis (verified 2026-06-11). Reading
our kernels before building EAGLE-3 would have saved the detour. "наши ядра изучи может что-то упустили" —
we missed G4_03.

## Action taken
- **Reverted** the EAGLE-3 detour (commit 23a66cbc reverts d210b86c + 418eba59): deleted both eagle3
  profiles + the eagle3 method support in `spec_decode.py` (its only consumers were these profiles) +
  the test exemption. Keeping them would be a dead path on our only hardware (every boot trips G4_03).
- Removed the two dead `Exited (1)` eagle3 containers from the rig. PROD untouched.
- The RedHatAI eagle3 speculator checkpoints remain on the rig NFS (harmless; usable only on a
  non-Ampere rig or once G4_10 lands).

## Verified findings from the parallel research + verification workflow

### 1. #43671 (EAGLE-3 aux hidden-state off-by-one) — RED HERRING for gemma4 (do NOT backport)
Two independent adversarial verifiers + synthesis agreed (`claim_holds:false`), file:line both sides:
- #43671 is **MiniMax-M2-scoped** (single-file diff to `minimax_m2.py`, OPEN). It compensates for the
  MiniMax draft head being trained against SGLang's `+1` capture convention.
- Our gemma4 heads were trained with vLLM's own Speculators library against the **same** `idx+1`-capture
  + pass-through path dev491 runs (gemma4.py:1360 `_maybe_add_hidden_state(aux, layer_idx+1, …)`); PR#39450
  logged HEALTHY acceptance (per-pos 0.823/0.638/0.486, mean 2.95) on the exact RedHatAI 31B head.
- **Applying a +1 shift to gemma4 would DESYNC train/infer and CAUSE the collapse** the claim fears.
  PR#43671's own body says the fix is intentionally local because other eagle3 models rely on current behavior.
- (Moot anyway now that EAGLE-3 is reverted — but recorded so we don't re-chase it.)

### 2. G4_08 (Gemma-4 Marlin K-dim pad MoE fallback) — silent no-op on dev491 + worse
Confirmed via dev491 checkout + AWQ dispatch path:
- **Class rename**: G4_08 looks up `CompressedTensorsMoEWNA16MarlinMethod`; dev491 renamed it to
  `CompressedTensorsWNA16MarlinMoEMethod` (compressed_tensors_moe_wna16_marlin.py:50). `getattr(...,None)`
  → `apply()` returns `("skipped", …)`. Silent no-op.
- **Method rename**: it wraps `.apply_weights`; dev491's method is `.apply` (:512) — would AttributeError
  (uncaught) if the class lookup were fixed.
- **WRONG CLASS for AWQ (decisive)**: the 26B-A4B model is AWQ-int4 → dispatches to `AWQMarlinMoEMethod`
  (awq_marlin.py:512), a different class than the `CompressedTensors…` one G4_08 patches. So G4_08 never
  unblocked K=352 for the AWQ checkpoint, even on dev259. The YAML premise (`G4_08:'1'`, `G4_02:'0'`) is false.
- **int4 stub**: `kernels/g4_kpad_moe_gemm_triton.py:375-378` `NUM_BITS==4` branch does `b*scale` with NO
  nibble unpack ("see _unpack_int4 below" — which does not exist). Garbage on real AWQ int4.
- **Boot mystery resolved**: 26B-A4B is NOT running on dev491 — YAML pins dev259 with explicit `pin_hold`
  (gemma-4-26b-a4b-it-awq.yaml:59-60). dev491 does NOT pad MoE-K natively for AWQ (awq_marlin.py:316
  2-arg `check_moe_marlin_supports_layer` + marlin_utils.py:317-318 "MoE prep does not pad yet"). So on
  a dev491 bump, K=352 is unguarded (neither G4_02 nor G4_08).

### 3. #45703 (OPEN) — native MoE Marlin K-pad — supersedes G4_08
Extends Marlin thread-tile padding to MoE incl. the **AWQ-int4** path G4_08 stubs (`_process_awq_weights_marlin`,
nibble-aware), same zero-pad-at-prep + zero-cancellation contract, native CUDA. Predecessor #45295 (dense-only)
is already in dev491. **Plan: WATCH #45703; retire G4_08 (+ re-decide G4_02) on merge-into-an-adopted-pin.**

## Corrected Gemma-4 speedup plan (next steps)

1. **26B-A4B → make `gemma4-26b-mtp-chat-k3` the chat path.** +65% already validated (assistant drafter,
   dev491). Currently `gemma4-26b-no-mtp` holds the default-role slot and the request_router gates MTP by
   workload. Update the workload-gate so free_chat/code also get MTP-K3 (the +65% covers free_chat — the old
   "MTP-off-for-chat" gate was a stale dev259-era 0%-acceptance finding). [config; needs router-policy edit]
2. **31B → re-validate `gemma4-31b-tq-mtp-chat-k3` on dev491** (assistant drafter + TQ-KV @64K). This is the
   correct path the EAGLE-3 detour bypassed; documented to boot clean at 64K on the prior pin. **Requires a
   PROD-down boot+bench cycle** (31B + 35B won't co-reside on 2×24GB) — defer to user greenlight given the
   "quick smoke only" + PROD-safety constraints.
3. **G4_08 hygiene** (patch optimization): make the silent dev491 no-op LOUD (observable warn when flag=1 but
   class/method unresolved) + correct the gemma-4-26b YAML premise; retire on #45703 merge. [code-only]
4. **G4_15/G4_24** fictional +5-10% claims already corrected (commit dcd2bd47).

## PROD status
35B `vllm-qwen3.6-35b-balanced-k3` healthy on :8102 (canonical port for start_qwen3.6-35b-balanced.sh;
:8101 idle is expected). Verified end-to-end with an authenticated completion, not just /health.
