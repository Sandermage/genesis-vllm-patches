# Adding Your Own Model Recipe

This guide walks an operator through adding a new model to Genesis end-to-end: from picking a base launch script, to wiring up the right patches, to submitting the recipe upstream so others can reuse it.

If you're new to Genesis, read [QUICKSTART.md](../QUICKSTART.md) first. If you want the patch catalog, see [PATCHES.md](../PATCHES.md). If you want the supported-versions matrix, see [docs/COMPATIBILITY.md](COMPATIBILITY.md).

---

## Quick decision tree

Before doing anything else, classify your model:

**Is it Qwen3-family (Qwen3, Qwen3-Coder, Qwen3.5, Qwen3.6, Qwen3-Next)?**
- Yes → Genesis works out of the box for the common variants. The 27B / 35B-A3B-MoE recipes ship in-tree. Other sizes (8B, 14B, 80B-Next) usually require only env-var tweaks.
- No → some patches still apply (the generic ones — see below); model-specific patches will skip via `applies_to` filters and that's fine.

**Generic patches (apply to most architectures):**
- P58 (async-scheduler placeholder fix), P66 (cudagraph capture-sizes filter), P72 (profile_run cap), P74 (chunk-clamp), P94 (spec-decode prepare_next_token_ids zero-init), P99 (workspace memoize), PN13 (CUDAGraphWrapper gc lambda arity), PN14 (TQ decode IOOB clamp — opt-in), PN17 (FA2 lse runtime clamp), PN19 (scoped max_split_size_mb).

**Model-specific patches:**
- P59-P69 family — Qwen3 reasoning parser, tool-call XML in `<think>`, MTP draft handling.
- P60, P60b, PN11, P103 — Hybrid GDN models only (Qwen3.5, Qwen3-Next).
- P87, P91 — AutoRound INT4 quantization with `group_size=128`.
- P81 — FP8 (offline or online).
- P67/P67b, P98, P101, PN8 — TurboQuant KV cache (`turboquant_k8v4`).

If you don't recognize half of these, that's fine — the launch scripts pick the right set for you.

---

## Step 1: Identify your model

Collect, before you start editing anything:

- **HuggingFace name + revision pin.** e.g. `Qwen/Qwen3-Coder-30B-A3B-Instruct@abc123`. Pin a revision — float-tags break reproducibility.
- **Architecture.** From `config.json`, the `architectures` field. Examples: `Qwen3MoeForCausalLM`, `Qwen3NextForCausalLM`, `Qwen3_5ForConditionalGeneration`, `LlamaForCausalLM`, `MistralForCausalLM`, `Gemma2ForCausalLM`.
- **Quantization.**
  - None (BF16/FP16) — biggest VRAM, no quant patches needed.
  - AutoRound INT4 — check `quantization_config.quant_method == "auto-round"`. P87, P91 apply if `group_size=128` (Marlin path); they no-op if `group_size=-1` (AllSpark path).
  - GPTQ INT4 — Marlin path, similar coverage to AutoRound.
  - AWQ INT4 — partial coverage; some patches assume Marlin and skip on AWQ.
  - FP8 offline (pre-quantized weights) — P81 applies.
  - FP8 online (`--quantization fp8`) — P81 applies, PN8 saves ~1 GiB/GPU on Ampere.
- **Hybrid attention?** `model_type` in config.json. `qwen3_5` and `qwen3_next` are hybrid (some layers GDN/Mamba, some standard attention). Pure-attention models (`qwen3_moe`, `llama`, `mistral`) are not.
- **Spec-decode option.**
  - MTP module on the HF repo (look for `model.embed_tokens` + a `mtp_*` weight prefix) → MTP supported.
  - DFlash drafter checkpoint exists separately on HF → DFlash supported (Qwen3.6-27B and 35B-A3B both have z-lab drafts).
  - Neither → use ngram (always works, gain depends on workload).

Write these five things down. The rest of this guide refers back to them.

---

## Step 2: Pick a base launch script

Genesis ships launch scripts in **two locations** for two deployment modes:

- `scripts/*.sh` — Docker container launch (recommended for most users; uses `docker run`).
- `scripts/launch/*.sh` — both Docker (`start_*`) and bare-metal (`bare_metal_*`, host shell with vLLM installed via pip) variants.

Pick the closest match to your config:

| Your config | Start with |
|---|---|
| Qwen3-family + INT4 + hybrid + 1× 24GB short ctx | `scripts/launch/start_27b_int4_no_TQ_short_single_card.sh` |
| Qwen3-family + INT4 + hybrid + 1× 24GB long ctx (256K) | `scripts/launch/start_27b_int4_no_TQ_long_256K_single_card.sh` |
| Qwen3-family + INT4 + hybrid + 2× 24GB short ctx | `scripts/start_27b_int4_fp8_e5m2_short.sh` |
| Qwen3-family + INT4 + hybrid + 2× 24GB long ctx (256K) | `scripts/start_27b_int4_fp8_e5m2_long_256K.sh` |
| Qwen3-family + INT4 + hybrid + TurboQuant k8v4 (5× KV pool) | `scripts/start_27b_int4_TQ_k8v4.sh` |
| Qwen3-family + MoE + FP8 (2× 24GB) | `scripts/start_35b_fp8_PROD.sh` |
| Qwen3-family + MoE + FP8 + 1× 48GB | `scripts/launch/start_35b_fp8_PROD_single_card.sh` |
| Coding-agent workload (DFlash drafter) | `scripts/start_27b_int4_DFLASH.sh` |
| Bare-metal (no Docker), 27B INT4 + TQ k8v4 | `scripts/launch/bare_metal_27b_int4_TQ_k8v4.sh` |
| Bare-metal (no Docker), 35B FP8 PROD | `scripts/launch/bare_metal_35b_fp8_PROD.sh` |
| Llama / Mistral / Gemma dense | `scripts/start_35b_fp8_PROD.sh` (drop MoE-specific flags) |

**Naming convention:**

- `start_*` = Docker container launch (uses `docker run`, mounts /models, applies patches inside container)
- `bare_metal_*` = host shell launch (assumes vLLM already installed via `pip install`)
- `_no_TQ_*` historical name for fp8_e5m2 KV cache (without TurboQuant) — file IS for fp8 KV
- `_TQ_k8v4_*` = TurboQuant 8-bit-key 4-bit-value KV cache (5× pool, requires P4 + P67 + P98)
- `_DFLASH_*` = DFlash speculator (coding-agent workload)
- `_single_card` suffix = TP=1 variant; without suffix = TP=2

**Rule of thumb:** start with a script that matches your *attention type* (hybrid vs. dense), *KV dtype* (auto / fp8_e5m2 / turboquant), and *card count*. The rest you'll edit in step 3.

### Step 2b: Docker compose mirror (if you don't use the bash scripts directly)

If you deploy via Docker compose (rather than `bash scripts/...`), the
bare-metal scripts won't run as-is inside the container — they expect
`pip install`-writable layers. Mirror the env vars + flags into your
compose's `environment:` and `command:` blocks. Reference: each
`start_*.sh` script lists the full env-var set baked for that config —
copy the `-e GENESIS_ENABLE_*` blocks into compose `environment:` and
the `vllm serve` flags into `command:`.

Worked compose snippet (27B + TQ k8v4 PROD, mirrors `scripts/start_27b_int4_TQ_k8v4.sh`):

```yaml
services:
  vllm-27b:
    image: vllm/vllm-openai:nightly
    environment:
      GENESIS_ENABLE_P4: "1"
      GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL: "1"
      GENESIS_ENABLE_P98: "1"
      GENESIS_ENABLE_P85: "1"
      GENESIS_ENABLE_P87: "1"
      GENESIS_ENABLE_P91: "1"
      GENESIS_ENABLE_P99: "1"
      GENESIS_ENABLE_P100: "1"
      GENESIS_ENABLE_P101: "1"
      GENESIS_ENABLE_P58_ASYNC_PLACEHOLDER_FIX: "1"
      GENESIS_ENABLE_P60_GDN_NGRAM_FIX: "1"
      GENESIS_ENABLE_P60B_TRITON_KERNEL: "1"
      GENESIS_ENABLE_P61_QWEN3_MULTI_TOOL: "1"
      GENESIS_ENABLE_P61B_STREAMING_OVERLAP: "1"
      GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING: "1"
      GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING: "1"
      GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER: "1"
      GENESIS_ENABLE_P68_AUTO_FORCE_TOOL: "1"
      GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER: "1"
      GENESIS_ENABLE_P72_PROFILE_RUN_CAP: "1"
      GENESIS_PROFILE_RUN_CAP_M: "4096"
      GENESIS_ENABLE_P74_CHUNK_CLAMP: "1"
      GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT: "1"
      GENESIS_ENABLE_PN9_INDEPENDENT_DRAFTER_ATTN: "1"
      GENESIS_ENABLE_PN11_GDN_AB_CONTIGUOUS: "1"
      GENESIS_ENABLE_PN13_CUDA_GRAPH_LAMBDA_ARITY: "1"
      GENESIS_ENABLE_PN17_FA2_LSE_CLAMP: "1"
      GENESIS_PREALLOC_TOKEN_BUDGET: "4096"
      GENESIS_BUFFER_MODE: "shared"
    command:
      - --model
      - /models/Qwen3.6-27B-int4-AutoRound
      - --kv-cache-dtype
      - turboquant_k8v4
      - --tensor-parallel-size
      - "2"
      - --speculative-config
      - '{"method":"mtp","num_speculative_tokens":3}'
      # ... copy remaining flags from scripts/start_27b_int4_TQ_k8v4.sh
```

The full source of truth for env vars is the `start_*.sh` script header — keep your compose in sync when Genesis updates the env-var set.

---

## Step 3: Configure for your model

Open the script and change these:

### Required edits

```bash
# Replace with your model
--model /path/to/your/model
--served-model-name my-model

# Memory
--gpu-memory-utilization 0.85   # bump to 0.90 if you have headroom; lower if OOM
--max-model-len 65536            # set to your target context
--max-num-seqs 4                 # lower for long ctx, raise for high concurrency

# Spec-decode (pick one — see Step 1 for which methods YOUR model supports)
--speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_min": 2, "prompt_lookup_max": 5}'
# or for MTP (REQUIRES the HF repo to carry mtp_* weight prefix — see Step 1 §4):
--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
# or for DFlash (REQUIRES separate z-lab drafter checkpoint download — see Step 1 §4):
--speculative-config '{"method": "dflash", "model": "/path/to/dflash-draft", "num_speculative_tokens": 4}'
# Note for DFlash on hybrid GDN models (Qwen3.6 family): vllm PR #40898 (DFlash SWA support)
# is OPEN as of 2026-05-01. Genesis ships PN21 partial backport — full SWA enabler awaits
# upstream merge. Without it, ~25% acceptance-length gap on long context with sliding-window
# attention layers. Track at https://github.com/vllm-project/vllm/pull/40898.

# KV cache dtype (pick one)
--kv-cache-dtype auto             # default
# --kv-cache-dtype fp8_e5m2       # 2× KV capacity, ~no quality loss
# --kv-cache-dtype turboquant_k8v4  # 5× KV capacity, requires P67/P98
```

### Genesis env flags

```bash
# Enable a baseline universal set:
export GENESIS_ENABLE_P58_ASYNC_PLACEHOLDER_FIX=1
export GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER=1
export GENESIS_ENABLE_P72_PROFILE_RUN_CAP=1
export GENESIS_ENABLE_P74_CHUNK_CLAMP=1
export GENESIS_ENABLE_P94=1
export GENESIS_ENABLE_P99=1
export GENESIS_ENABLE_PN13_CUDA_GRAPH_LAMBDA_ARITY=1
export GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP=1
export GENESIS_ENABLE_PN17_FA2_LSE_CLAMP=1
export GENESIS_ENABLE_PN19_SCOPED_MAX_SPLIT=1
```

Add the model-specific patches per step 4.

---

## Step 4: Pick patches

Refer to [PATCHES.md](../PATCHES.md) for full descriptions. The buckets:

### Universal (recommended for almost everything)

P58, P66, P72, P74, P94, P99, PN13, PN14, PN17, PN19.

These fix bugs or add safety guards that don't depend on model architecture. Default them all on.

### Qwen3-family (tool-call, streaming, MTP)

If your model is Qwen3 / Qwen3.5 / Qwen3.6 / Qwen3-Next:

- P59, P60, P60b — reasoning parser tool-call extraction (Qwen3 puts tool calls inside `<think>` blocks).
- P61, P61b, P62 — multi-tool first-occurrence, streaming overlap guard, reasoning-aware grammar.
- P64, P68, P69 — qwen3coder MTP streaming, long-ctx tool-call hardening.

### Hybrid GDN models only

Qwen3.5, Qwen3-Next:

- P60, P60b — GDN conv + SSM state corruption with ngram spec decode.
- PN11 — hybrid layer dispatch fix (a/b contiguity).
- P103 — chunked GDN fwd_h+fwd_o orchestrator (saves ~600 MiB on long ctx).

### AutoRound INT4 (Marlin path only)

If `quantization_config.group_size == 128`:

- P87 — backport of vLLM PR #40361, +24% on Ampere AutoRound INT4.
- P91 — AutoRound row-parallel scales fix (vLLM PR #39460 backport).

These are no-ops on `group_size=-1` AllSpark path.

### FP8

- P81 — FP8 hotfixes (block-scaled MM low-M decode tuning).
- PN8 — saves ~1 GiB/GPU on FP8 online quant on Ampere.

### TurboQuant KV cache

If `--kv-cache-dtype turboquant_k8v4`:

- P4 — required, removes hybrid TQ rejection.
- P67, P67b — multi-query Triton kernel (replaces upstream which gives garbage tokens under FULL cudagraph).
- P98 — required, TQ WorkspaceManager revert (else AssertionError on workspace lock).
- P101 — TQ packed-slot layout opt.
- PN8 — VRAM savings.

**Copy-paste TQ k8v4 minimal env block:**

```bash
# Required for TQ k8v4 — engine won't boot without these
export GENESIS_ENABLE_P4=1                              # removes hybrid TQ rejection
export GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1       # Genesis multi-query kernel
export GENESIS_ENABLE_P98=1                             # WorkspaceManager revert (vllm#40941 fix)
export GENESIS_ENABLE_P101=1                            # TQ packed-slot layout
export GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT=1      # ~600 MiB VRAM savings on draft
```

**Recommended additional patches for TQ k8v4 PROD (per `start_27b_int4_TQ_k8v4.sh`):**

```bash
# Performance (composes with TQ k8v4)
export GENESIS_ENABLE_P85=1                             # hybrid fine-shadow prefix cache
export GENESIS_ENABLE_P87=1                             # ~+24% on AutoRound INT4 path (Marlin)
export GENESIS_ENABLE_P91=1                             # AutoRound row-parallel scales fix
export GENESIS_ENABLE_P99=1                             # WorkspaceManager memoize
export GENESIS_ENABLE_P100=1                            # FlashInfer FULL cudagraph spec-decode

# Recommended quality patches
export GENESIS_ENABLE_P58_ASYNC_PLACEHOLDER_FIX=1
export GENESIS_ENABLE_P60_GDN_NGRAM_FIX=1
export GENESIS_ENABLE_P60B_TRITON_KERNEL=1
export GENESIS_ENABLE_P61_QWEN3_MULTI_TOOL=1
export GENESIS_ENABLE_P61B_STREAMING_OVERLAP=1
export GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING=1
export GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING=1
export GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER=1
export GENESIS_ENABLE_P68_AUTO_FORCE_TOOL=1
export GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER=1
export GENESIS_ENABLE_P72_PROFILE_RUN_CAP=1
export GENESIS_PROFILE_RUN_CAP_M=4096
export GENESIS_ENABLE_P74_CHUNK_CLAMP=1
export GENESIS_ENABLE_PN9_INDEPENDENT_DRAFTER_ATTN=1
export GENESIS_ENABLE_PN11_GDN_AB_CONTIGUOUS=1
export GENESIS_ENABLE_PN13_CUDA_GRAPH_LAMBDA_ARITY=1
export GENESIS_ENABLE_PN17_FA2_LSE_CLAMP=1
export GENESIS_PREALLOC_TOKEN_BUDGET=4096
export GENESIS_BUFFER_MODE=shared
```

> **Note**: P82 (SGLang acceptance threshold OR-clause) is **OFF by default** on the 27B PROD launch — historical bench data showed it's biased on small batch single-stream Lorbus INT4 + MTP K=3. Enable via `GENESIS_P82_THRESHOLD_SINGLE=0.3 GENESIS_ENABLE_P82=1` only after A/B on your specific workload.

### Compile / cudagraph safety

If you hit boot crashes related to torch.compile or cudagraph capture:

- P65 — switch to PIECEWISE cudagraph (workaround for FULL capture issues on hybrid).
- P99 — compile-cache safety.

See [docs/CLIFFS.md](CLIFFS.md) for the cliffs these patches address.

---

## Step 5: First boot

```bash
bash scripts/start_<your>.sh > boot.log 2>&1 &
tail -f boot.log
```

What to watch for:

1. **`[GENESIS]` summary block** — should print near the top. Every patch you enabled should be `APPLY` (success), `SKIP` (filtered out by `applies_to` — fine), or `INFO` (already-applied, fine). Anything `FAILED` is a bug, capture the line and open an issue.
2. **`Application startup complete.`** — vLLM has bound the port.
3. **`curl http://localhost:8000/v1/models`** — should return your model.
4. **Tool-call sanity** — quick smoke test:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer genesis-local" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "What time is it in Paris?"}],
    "tools": [{"type": "function", "function": {
      "name": "get_time",
      "description": "Get current time in a city",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    }}],
    "tool_choice": "auto"
  }' | jq '.choices[0].message'
```

> **API key note:** the launch scripts include `--api-key genesis-local` so the `Authorization: Bearer genesis-local` header is required. If you launched **without** `--api-key`, drop the header (and any `OPENAI_API_KEY` you set client-side). If you set a different key, replace it.

You should see a `tool_calls` field with `name: "get_time"` and `arguments: {"city": "Paris"}`. If you get garbage tokens (`<tool_call><tool_call>...`) or repetition, you've hit a cliff — see [docs/CLIFFS.md](CLIFFS.md).

5. **Bench:**

```bash
python tools/genesis_bench_suite.py \
    --base-url http://localhost:8000/v1 \
    --model my-model \
    --runs 5
```

> **Bench input format:** the script ships with default `NARR_PROMPTS` (narrative / 600-char) and `CODE_PROMPTS` (code / ~80-char) lists targeting the Qwen3-Coder chat template. If your model has a different template (e.g. Llama-3 ChatML, Mistral instruct), edit the prompt lists in `tools/genesis_bench_suite.py` to match the format your tokenizer expects — wrong template causes the bench to look slow because every reply gets a "I cannot help with that" stop after a few tokens.

Record `wall_TPS` mean, std, CV. CV under 8% is a clean run. Above that, investigate (other tenants, thermal, allocator fragmentation).

---

## Step 6: Tune and iterate

Common things to tweak after first boot:

### Slow

- **Spec-decode acceptance low?** Try a different `num_speculative_tokens` (3 is a good default; 5 helps repetitive workloads, 2 helps prose).
- **CPU dispatch overhead?** Set `VLLM_MOE_BACKEND=triton` for MoE models on v0.20+ (see Cliff 6 in [docs/CLIFFS.md](CLIFFS.md)).
- **Kernel sweep.** P67 BLOCK_KV / num_warps overrides via env: `GENESIS_P67_BLOCK_KV=64`, `GENESIS_P67_NUM_WARPS=4`. Sweep with `tools/genesis_bench_suite.py`.

### OOM

- Lower `--max-model-len` first (cuts attention metadata).
- Lower `--max-num-seqs` (cuts KV cache).
- Lower `--gpu-memory-utilization` from 0.90 → 0.85.
- Switch to `--kv-cache-dtype fp8_e5m2` (2× KV capacity).
- Switch to `--kv-cache-dtype turboquant_k8v4` (5× KV capacity, needs P4 + P67 + P98).
- Drop `--enable-prefix-caching` if your model is hybrid and on AutoRound INT4 — see Cliff 3.

### Tool-call breaks

- Verify P59-P69 family is enabled.
- Check the reasoning parser flag: `--reasoning-parser qwen3` (or `qwen3_5` for hybrid).
- If using ngram spec-decode on prose, try `prompt_lookup_min=2,max=5` instead of strict mode (see Cliff 5).

### Quality regression

- A/B with `GENESIS_DISABLE_ALL=1` to confirm the regression is patch-related.
- Bisect by disabling patch buckets one at a time.
- Open an issue with the bisect result and a reproducer.

---

## Step 7: Submit your recipe back

Once your recipe boots cleanly, passes a tool-call sanity check, and you have `n=5` bench numbers:

1. **Add the launch script.** `scripts/start_<MODEL>_<KV>_<MODE>.sh`. Make sure it's executable and self-contained (no `source ../private_env.sh` referencing files outside the repo).
2. **Update [MODELS.md](../MODELS.md).** Add a row to the table with model name, GPU, KV dtype, expected TPS, and link to your script.
3. **Open a PR.** Follow the [contributing guide](../CONTRIBUTING.md) — include `tested-on` info and the bench output.

The maintainer reviews everything personally. Turnaround is usually 24-48 hours.

---

## Worked example: adding a Llama-3 70B recipe

To show that generic patches work outside Qwen3-family, here's a minimal walkthrough.

### Identify

- **Model:** `meta-llama/Meta-Llama-3-70B-Instruct@<rev>`
- **Architecture:** `LlamaForCausalLM`
- **Quantization:** none (BF16) or AWQ INT4 if you want to fit on 1× 48GB
- **Hybrid:** no
- **Spec-decode:** ngram only (no MTP module shipped)

### Pick a base script

`start_35b_fp8_PROD.sh` — it's the closest pure-attention dense launcher in-tree. We'll strip MoE flags.

### Copy and edit

```bash
cp scripts/start_35b_fp8_PROD.sh scripts/start_llama3_70b_awq.sh
```

Edits:

```bash
--model meta-llama/Meta-Llama-3-70B-Instruct
--served-model-name llama-3-70b
--quantization awq                            # if AWQ checkpoint
--max-model-len 8192                          # Llama-3 native ctx
--max-num-seqs 8
--gpu-memory-utilization 0.90

# Spec-decode:
--speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_min": 2, "prompt_lookup_max": 5}'

# KV cache:
--kv-cache-dtype fp8_e5m2                     # 2× KV capacity, generic patch coverage
```

### Genesis env flags

Universal set only:

```bash
export GENESIS_ENABLE_P58_ASYNC_PLACEHOLDER_FIX=1
export GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER=1
export GENESIS_ENABLE_P72_PROFILE_RUN_CAP=1
export GENESIS_ENABLE_P74_CHUNK_CLAMP=1
export GENESIS_ENABLE_P94=1
export GENESIS_ENABLE_P99=1
export GENESIS_ENABLE_PN13_CUDA_GRAPH_LAMBDA_ARITY=1
export GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP=1
export GENESIS_ENABLE_PN17_FA2_LSE_CLAMP=1
export GENESIS_ENABLE_PN19_SCOPED_MAX_SPLIT=1
```

The Qwen3-specific patches (P59-P69 family) will SKIP automatically because `applies_to.model_archs` doesn't include `LlamaForCausalLM` — that's by design.

### Boot, test, bench

Same as steps 5-6 above.

### Submit

PR with the script + a row in MODELS.md + your bench numbers.

That's it. Generic patches work on Llama-3 because they're not coupled to Qwen3 internals — they fix bugs or add guards in code paths that all transformer models hit.

---

## Cross-references

- [QUICKSTART.md](../QUICKSTART.md) — getting started
- [PATCHES.md](../PATCHES.md) — full patch catalog
- [MODELS.md](../MODELS.md) — supported model table
- [docs/COMPATIBILITY.md](COMPATIBILITY.md) — vLLM pin / model / GPU support matrix
- [docs/CLIFFS.md](CLIFFS.md) — known cliffs to watch out for
- [CONTRIBUTING.md](../CONTRIBUTING.md) — how to submit your recipe
