# Genesis Glossary

A reference of terms you will encounter while running Genesis on top of vLLM. Each entry is short on purpose — enough to orient a newcomer, not a textbook chapter. Entries are grouped loosely by topic but listed alphabetically inside each section.

This is meant for first-time readers. If you already know what `MTP` or `GQA` is, skip ahead. If a term you saw in a Genesis log line is missing, please open an issue.

## Performance Metrics

### TPS (tokens per second)
End-to-end generation throughput, measured as decoded output tokens divided by wall-clock time of the request. In Genesis benchmarks we report `wall_TPS` over a fixed prompt set, with N runs and a coefficient of variation (CV) so noise is visible. Higher is better.

### TTFT (time to first token)
Latency from request submit to the first streamed token reaching the client. Dominated by prefill cost (KV-cache fill) and the first decode step. Sensitive to context length and prefix-cache hits. Lower is better.

### TPOT (time per output token)
Average inter-token latency during the decode phase, after the first token. Equal to `1 / TPS` for a single request and is what the user perceives as "typing speed". Independent of TTFT.

### CV (coefficient of variation)
Standard deviation divided by mean, expressed as a percentage. Genesis reports CV alongside every TPS number to distinguish a real win from run-to-run noise. A patch claiming +2% with CV=5% is not a real win.

## Quantization

### FP16 / BF16
Half-precision floating point formats used for weights and activations. FP16 has more mantissa bits (better precision, smaller range), BF16 has more exponent bits (wider range, used by most modern training). vLLM accepts both for unquantized models.

### FP8
8-bit floating point, two flavors `e4m3` (more precision) and `e5m2` (more range). Used for weights, activations and KV cache on Hopper and newer. On Ampere FP8 KV is supported but compute falls back to FP16/BF16 matmul.

### GPTQ
Post-training weight-only quantization technique that calibrates per-channel scales using a small dataset. Genesis runs GPTQ-4bit Qwen variants but prefers AutoRound for accuracy.

### AWQ
Activation-aware Weight Quantization. Like GPTQ but explicitly scales weights based on activation magnitudes. Common for 4-bit Qwen and Llama checkpoints.

### AutoRound INT4
Intel's quantization-aware rounding scheme producing 4-bit weights with group-size 128 (typically). Genesis-validated checkpoint is `Lorbus/Qwen3.6-27B-int4-AutoRound`. Routes through Marlin or AllSpark kernels depending on `group_size`.

### TurboQuant (k8v4, 4bit_nc, 3bit_nc)
KV-cache compression scheme upstream as `--kv-cache-dtype turboquant_*`. `k8v4` keeps keys in 8-bit and values in 4-bit (Genesis default — best quality/throughput trade). `4bit_nc` is symmetric 4-bit, `3bit_nc` is 3-bit (high accuracy loss on Qwen). Trades 5-15% throughput for 2-4× more concurrent KV slots.

## Speculative Decoding

### Speculative Decoding
A technique where a small/fast draft proposes K tokens and the large target model verifies them in one forward pass. If accepted, the model "skips ahead" K tokens for the cost of one decode step.

### Draft Model / Target Model
The draft is the small proposer (e.g. an MTP head, an n-gram lookup, or a small transformer). The target is the production model that verifies. Acceptance rate is the fraction of draft tokens that survive verification.

### Acceptance Rate
Fraction of speculative tokens accepted by the target model. Higher is better. Heavily workload-dependent: code completion gets 70-90%, free-form prose gets 30-50%. Genesis reports per-position acceptance for tuning `prompt_lookup_min`.

### MTP (Multi-Token Prediction)
A small "head" module trained jointly with the target that predicts the next K tokens directly. Qwen3.6 ships with built-in MTP. Best for chat/prose workloads.

### Eagle3
Tree-based speculative-decoding scheme using a tiny draft transformer plus a verification tree. Higher acceptance than MTP at higher cost. Genesis tracks Eagle3 but does not yet ship it as default.

### ngram (Prompt Lookup Decoding)
Zero-cost draft method that searches the prompt for matching n-grams and reuses them as speculative tokens. Excellent for code (high lexical repetition), poor for prose. Tunable via `prompt_lookup_min/max`.

### DFlash
Draft model designed for code-heavy workloads. Larger and smarter than MTP, weaker than Eagle3. HuggingFace-gated download.

## Attention Variants

### MHA (Multi-Head Attention)
Vanilla attention. Each query head has its own dedicated key and value head. Highest memory footprint per token.

### MQA (Multi-Query Attention)
All query heads share a single key and value head. Lowest memory, lowest quality. Rare in modern models.

### GQA (Grouped Query Attention)
Compromise: query heads are split into G groups, each group shares one K/V head. Qwen3.6-27B uses `GQA=24/4=6` (24 query heads, 4 KV heads, group size 6). Genesis P67 fast-path requires power-of-two group size — non-pow-2 GQA falls through to upstream until v7.63.x generalization landed.

### GDN (Gated DeltaNet)
Linear-attention variant from the FLA (Flash Linear Attention) family used in hybrid Qwen3.6 models. Replaces some softmax-attention layers with a recurrent gated state. Cheap memory, different numerics.

### Hybrid Model
A model interleaving softmax-attention layers and linear-attention (GDN) layers. Qwen3.6-27B-int4-AutoRound is hybrid. Requires special KV-cache layout in vLLM.

### FLA (Flash Linear Attention)
Open-source library of fast linear-attention kernels (DeltaNet, GLA, RWKV, Mamba2). vLLM imports FLA for GDN layers.

## Compilation & Kernels

### CUDA Graph
NVIDIA mechanism for capturing and replaying a sequence of GPU operations as a single launch. Eliminates per-op CPU overhead. vLLM uses three modes:
- `PIECEWISE` — capture small contiguous regions, safe with control flow.
- `FULL` — capture the whole forward pass; fastest but breaks on data-dependent shapes.
- `FULL_AND_PIECEWISE` — full graphs plus piecewise fallback for shape variations.

### torch.compile
PyTorch's TorchDynamo + Inductor stack that traces and compiles graphs at runtime. vLLM uses it for fused element-wise ops and select decode kernels.

### Triton
OpenAI's Python-embedded GPU kernel language. Most Genesis hand-written kernels (P67, P40, P104) are Triton. Generated PTX is cached under `~/.triton/cache`.

### Tensor Parallel (TP)
Splits each weight matrix across N GPUs. Communication via all-reduce after each layer. Genesis-validated configs: TP=1 (single 24 GiB card, smaller models) and TP=2 (dual A5000 / dual 3090 / dual 4090).

### Pipeline Parallel (PP)
Splits the model layer-wise across N GPUs. Lower bandwidth need than TP, higher latency. Genesis does not ship PP-validated configs.

### Expert Parallel (EP)
For MoE models, splits the experts across GPUs. Genesis tested EP on 35B-A3B and found it hurt our single-user workload — kept off.

## Genesis-Specific

### Patch
A unit of behavior change applied at runtime. Two kinds: `text-patch` (regex/anchor edit of vLLM source files inside the container R/W layer) and `code-patch` (Python monkey-patch in `vllm/_genesis/`).

### Anchor / Marker
A short, stable string in upstream vLLM source that a text-patch uses as an insertion point. Anchors break across vLLM versions — `applies_to` pins prevent silent drift.

### Drift
When upstream vLLM changes the code surrounding an anchor, the patch can fail to find it (visible drift) or — worse — patch the wrong place (silent drift). Genesis runs anchor-presence checks on every boot.

### Sub-Patch
A logical patch implemented as several smaller atomic edits (e.g. P64 has sub-patches A through F). Useful when a single bug touches multiple files.

### Dispatcher
The boot-time loader (`vllm/_genesis/dispatcher.py`) that consults the `PATCH_REGISTRY`, evaluates env flags, runs `applies_to` and `conflicts_with` checks, and prints `[APPLY] / [SKIP] / [REC] / [OFF]` for every patch.

### applies_to filter
Per-patch metadata declaring which vLLM commit range, model family, GPU SM, or KV-cache dtype the patch supports. Skipped automatically when the boot environment does not match.

### conflicts_with
Per-patch declaration that two patches must not be applied together. The dispatcher refuses to boot if both env flags are on.

### A3B Suffix
Qwen naming convention: `A3B` = "Activated 3 Billion". The MoE 35B model activates 3B parameters per token. Compute cost is roughly that of a 3B dense model, memory is that of a 35B model.

### Lorbus
HuggingFace user `Lorbus`, who maintains the AutoRound INT4 quantization of Qwen3.6-27B that Genesis validates against.
