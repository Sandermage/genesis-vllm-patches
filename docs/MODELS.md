# Models, configs, and the V2 layered system

Single reference for **which models Genesis targets**, **how to add
your own**, and **how the V2 layered config system works underneath**.

> See also: [`HARDWARE.md`](HARDWARE.md) for the GPU envelope,
> [`CONFIGS_AUTO.md`](CONFIGS_AUTO.md) for the machine-generated
> config inventory, [`PATCHES.md`](PATCHES.md) for what each P*/PN*
> does.

## Default model: `Qwen/Qwen3.6-35B-A3B-FP8`

**HuggingFace**: <https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8>

| Property | Value |
| --- | --- |
| Total parameters | 35.95 B |
| Active per token | ~3 B (8 of 256 experts × 512 + shared expert) |
| Architecture | `qwen3_5_moe` (hybrid GDN + full-attention, 10 full + 30 GDN layers) |
| Hidden size | 2048 |
| Layers | 40 |
| Q/KV heads | 16 / 2 (GQA 8:1) |
| Experts | 256, top-8, shared expert |
| MTP layer | 1 (`mtp_num_hidden_layers: 1`) |
| Native context | 262 144 (256K) |
| Disk size | ~36 GB FP8 |
| License | Apache-2.0 |

### Why this specific model

1. **Hardware envelope match (48 GB VRAM, 2× A5000 Ampere).**
   `qwen3_5_moe` in FP8 lands at ~36 GB on disk and ~18 GB per GPU
   under TP=2. With TurboQuant `k8v4` KV (FP8 K + 4-bit V), a single
   256K-context session fits in < 22 GB per GPU, leaving headroom for
   batch prefill. The Apr-2026 releases (DeepSeek V4 Flash 158 GB,
   GLM-5.1 754 B, Qwen3-Next-80B 80 GB) all exceed our VRAM budget
   and would require NVFP4 (Blackwell-only) or REAP-style expert
   pruning to fit.
2. **Active-parameter throughput on Ampere.** A3B = 3 B active per
   forward pass on top of a 35 B sparse backbone. On 2× A5000 (no FP8
   native compute, Marlin weight-only path), this yields ~57 tok/s
   baseline and ~216 tok/s sustained with the Genesis MTP stack on
   Wave 10. A dense 35B model would saturate the PCIe Gen4 bus and
   run 3–4× slower. A3B is the highest-throughput configuration the
   SM 8.6 generation can sustain at this parameter count.
3. **Genesis patch lock-in (317 entries in `PATCH_REGISTRY`).** ~52
   default-on entries are production-eligible. ~30 fire on 35B PROD
   in steady state; the rest are conditional on workload / hardware /
   spec method. Switching architecture (Gemma 4, DeepSeek V4, GLM 5)
   would require a new patch port (~2–3 weeks per family).
4. **Long-context budget (262 144 tokens verified).** Our agentic
   workloads routinely exceed 100 K tokens. Qwen3.6-35B-A3B's native
   256K window has been end-to-end verified at 252K under load (96%
   of cap). The closest sub-50 GB alternatives (Gemma 4 26B A4B at
   128K, Qwen3.6 dense at 32K) fall short.
5. **Tool-calling fidelity.** Qwen3 series has best-in-open-weight
   Hermes-style XML tool-call templates and a chat template that
   survives ngram speculative decoding when paired with Genesis's
   strict spec-decode config (`prompt_lookup_min=8`). Abliterated /
   distilled variants trade refusal removal for tool-call grammar
   fragility — a losing trade for an aggregator stack.
6. **Ecosystem and maintenance.** 1.2M downloads / 179 likes on HF
   as of Apr 2026; active upstream (Qwen team patches GDN/MoE bugs
   in vLLM main) means our patches track real upstream progress
   rather than diverging into a personal fork.

## Tested alternative models

### `Qwen/Qwen3.6-27B-FP8` (dense variant)

**HuggingFace**: <https://huggingface.co/Qwen/Qwen3.6-27B-FP8>

| vs default | Detail |
| --- | --- |
| Architecture | `qwen3_5` dense — same hybrid 3:1 GDN + full pattern, no MoE |
| Total params | 27.78 B (vs 35.95B) |
| Active params/token | 27.78 B (full forward — no MoE sparsity) |
| VRAM @ FP8 | ~28 GB (fits on 1× 24GB or 2× 24GB) |
| Speed estimate | ~50–65 tok/s decode (vs ~216 sustained for A3B Wave 10) |
| Patch compat | All non-MoE patches apply (MoE-only patches in `family=moe` auto-skip via dispatcher) |

**Compose template:** `compose/docker-compose.qwen3-5-dense.yml`.
**Recommendation:** validated for single-3090 setups (noonghunna).
Not recommended as primary because dense 27B is 2–2.5× slower than
the A3B baseline.

### `Lorbus/Qwen3.6-27B-int4-AutoRound` (default for 24 GB rigs)

| Property | Value |
| --- | --- |
| Architecture | `qwen3_5` dense, AutoRound INT4 |
| VRAM @ INT4 + TQ k8v4 KV | ~14 GB weights + KV that fits on 24 GB |
| Native context | 256K (320K validated experimentally) |
| Genesis presets | `a5000-2x-27b-int4-tq-k8v4`, `qa-qwen3.6-27b-tq-1x` (V2; V1 alias `a5000-1x-27b-int4-tested` retired 2026-06-01), `long-ctx-qwen3.6-27b` |

This is the default 27B preset for any 24 GB rig, single or dual GPU.
Tool-call clean rate consistently 8/8 on Wave 10 (132.93 TPS sustained
on 2× A5000).

### `google/gemma-4-26B-A4B-it`

**HuggingFace**: <https://huggingface.co/google/gemma-4-26B-A4B-it>

| Property | Value |
| --- | --- |
| Architecture | `gemma4` (128 experts, A4B = 4B active) |
| VRAM @ FP8 | ~27 GB (fits) |
| Context | 128K (vs our 262K) |
| Hybrid attention | NO — different MoE design from `qwen3_5_moe` |

**Compose template:** `compose/docker-compose.gemma4-26b-moe.yml`
(experimental). Needs new patches for Gemma 4-specific MoE layout —
most Genesis patches skip via dispatcher.

### `cyankiwi/gemma-4-31B-it-AWQ-4bit`

**HuggingFace**: <https://huggingface.co/cyankiwi/gemma-4-31B-it-AWQ-4bit>

| Property | Value |
| --- | --- |
| Architecture | `gemma4_dense` (interleaved sliding + global + p-RoPE) |
| Quant / KV | AWQ-4bit (compressed-tensors); KV path is the trade-off below |
| Spec-decode | MTP K=3 via `google/gemma-4-31B-it-assistant` (separate drafter — optimal at K=3, do not raise) |
| Published context | 256K (G4_09 chunked-prefill) |

The 31B dense Gemma runs in two KV configurations that trade throughput
against context on the same 2× A5000 rig (single-stream, pin
`0.23.1rc1.dev148+gb4c80ec0f`):

| Profile | KV plan | Context | Decode TPS | Tool-call | Best for |
| --- | --- | ---: | ---: | :---: | --- |
| `gemma4-31b-kvauto-chat` | kv-auto — uniform fp16 KV, no TurboQuant | 32K | **~70.1** | 7/7 | interactive chat / tool-use (preset `prod-gemma4-31b-kvauto-chat`) |
| `gemma4-31b-tq-mtp-chat-k3` | TurboQuant `turboquant_4bit_nc` | 64K | ~41.4 | 6/7 | long-document context (preset `prod-gemma4-31b-tq-mtp-chat-k3`) |

kv-auto is +69.6 % TPS AND better tool-call quality; the only cost is
context (64K → 32K, because uniform fp16 KV is ~4× the turboquant_4bit_nc
footprint). Pick per workload — long documents → TQ 64K; interactive chat
/ tool-use → kv-auto 32K. The operator runs both.

## Models evaluated but NOT adopted

- **`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`** — 80 GB FP8 doesn't fit
  48 GB. Would need AWQ-INT4 (no official release) + multi-node.
- **`deepseek-ai/DeepSeek-V4-Flash` (284B / 13B active)** — 158 GB on
  disk; needs 4× A100 80GB. `deepseek_v4` arch (CSA+HCA hybrid) is
  incompatible with our GDN / MoE / TQ patch families.
- **`deepseek-ai/DeepSeek-V4-Pro` (862B)** — 860 GB. Out of scope for
  any single-node Ampere setup.
- **`zai-org/GLM-5.1` (754B)** — same scale problem; needs Blackwell
  plus a REAP-pruned variant which doesn't exist yet.
- **`Infatoshi/Qwen3.6-35B-A3B-NVFP4-FP8`** — NVFP4 requires Blackwell
  sm_120; Ampere SM 8.6 is unsupported.
- **`batsclamp/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-FP8`**
  — drop-in by architecture but abliteration consistently degrades
  tool-call clean rate, and "Claude-4.7-Opus distillation" is
  undocumented training. Bench it as a 30-min blue/green if curious;
  don't switch in production without measured numbers.

## Adding a new model

### Step 1 — architecture compatibility check

```bash
huggingface-cli download <org>/<model> --local-dir /tmp/check --include="config.json"
cat /tmp/check/config.json | jq '.model_type, .architectures, .num_local_experts // "dense"'
```

| `model_type` | Genesis support | Notes |
| --- | --- | --- |
| `qwen3_5_moe` | **primary** | All families apply. |
| `qwen3_5` | good | MoE-only patches auto-skip. |
| `qwen3_next` | likely good | Most apply. |
| `gemma4` | experimental | Subset only. |
| `deepseek_v4`, `glm_moe_dsa` | NO | Different arch family. |
| `mixtral` | partial | Plain-MoE patches only. |

### Step 2 — VRAM math

```python
# Rough rule of thumb
total_disk_gb       = <from HF model card>
weights_vram        = total_disk_gb  # FP8 stays full size
kv_cache_vram_at_X  = max_model_len * num_kv_heads * head_dim * 2 * num_layers / 1e9
total_per_gpu       = (weights_vram + kv_cache_vram_at_X) / tp_size
assert total_per_gpu < 22.0  # for 24GB A5000-class
```

### Step 3 — capture a working container

If you have a hand-tuned container running:

```bash
sndr model-config new my-rig --from-running vllm-test-container
```

The captor (audit C2 closure, 2026-05-16) parses `docker inspect`
output (Entrypoint + Cmd + Env + Mounts) and reverse-engineers a
`ModelConfig` YAML under `~/.sndr/configs/my-rig.yaml`.

### Step 4 — clone the closest builtin instead

```bash
sndr model-config list
sndr model-config new my-rig --template qa-qwen3.6-27b-tq-1x  # V1 alias `a5000-1x-27b-int4-tested` retired 2026-06-01
$EDITOR ~/.sndr/configs/my-rig.yaml
```

Edit at minimum:

- `key:` — your kebab-case identifier.
- `title:` — human-readable name.
- `maintainer:` — your GitHub handle.
- `hardware.gpu_match_keys:` — lowercase substring of
  `nvidia-smi --query-gpu=name`.
- `hardware.n_gpus:` — 1, 2, 4, ...
- `hardware.min_vram_per_gpu_mib:` — actual VRAM minus a reasonable
  buffer.
- `model_path:` — full path to weights, or the HuggingFace repo id.

For lifecycle declaration:

```yaml
lifecycle: community-test           # MUST start here
community_submitted: true           # MUST be true for community-* lifecycle
verified_by: []                     # filled by `sndr model-config promote`
test_started_at: '2026-05-16'       # today's ISO date
```

### Step 5 — local validation + boot

```bash
sndr model-config validate my-rig     # ~50 ms offline lint (schema + 19 audit rules)
sndr model-config preflight my-rig    # ~3 s sanity (mounts/GPU/pin/quant args)
sndr launch my-rig                    # boot
sndr model-config verify my-rig       # bench + diff vs reference_metrics
```

If quality is in line with the template you cloned from, capture the
bench output back into the YAML:

```bash
sndr model-config bench-and-update my-rig
```

This writes a `reference_metrics:` block back into the YAML
surgically (preserves your comments + ordering).

## The V2 layered config system

Genesis V2 splits the old monolithic `model_config.yaml` into four
orthogonal layers that compose into a runtime `ModelConfig`. The
split makes it cheap to add a new model on existing hardware, a new
rig for existing models, or a patch sweep without touching production
presets.

### The four layers

```text
┌────────────────────────────┐
│  ModelDef                  │  identity + capabilities + canonical patches
│  builtin/model/<id>.yaml   │  (model-owned: dtype, kv_cache_dtype, spec_decode)
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  HardwareDef               │  rig + sizing defaults + runtime block
│  builtin/hardware/<id>.yaml│  (hardware-owned: n_gpus, vram, image, mounts)
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  ProfileDef                │  patches delta + sizing override
│  builtin/profile/<id>.yaml │  (operator-owned: enable/disable/override)
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  Preset alias (3-pointer)  │  short operator name → triplet
│  builtin/presets/<n>.yaml  │  model: ...  hardware: ...  profile: ...
└────────────────────────────┘
```

### Composition rules

The composer (`sndr/model_configs/compose.py`):

1. **Compat gate first.** `check_compat(model, hardware)` rejects
   pairings where `requires.min_gpu_count` /
   `min_total_vram_mib` / `min_cuda_capability` aren't met. Fails
   fast with an operator-facing message.
2. **Patches matrix = `model.patches` + `profile.patches_delta`**
   (apply order: `enable → disable → override`).
3. **Sizing = `profile.sizing_override` OR `hardware.sizing`** —
   operator tuning for a specific model × hardware pair wins over
   the rig default.
4. **Runtime = `hardware.runtime.default`**, optionally overridden
   by `--runtime <name>` CLI flag (must be in
   `hardware.runtime.supported`).
5. **Final result is a V1 `ModelConfig`** — the composer is a bridge,
   so the existing launcher / k8s / compose / quadlet emitters work
   unchanged.

### Layer ownership rules

Each field has a single owning layer. Cross-layer conflicts on an
owned field are rejected at load time.

| Field | Owned by | Why |
| --- | --- | --- |
| `dtype`, `kv_cache_dtype`, `spec_decode.method`, `attention_arch` | Model | Different capability = different model file. |
| `tool_call_parser`, `reasoning_parser` | Model | Tied to the checkpoint's tokenizer/template. |
| `n_gpus`, `min_vram_per_gpu_mib`, `cuda_capability_min` | Hardware | Physical rig attribute. |
| `image`, `image_digest`, `mounts`, `network` | Hardware | Container substrate. |
| `system_env` (NCCL / PYTORCH / VLLM globals) | Hardware | Stable across models on the rig. |
| `max_model_len`, `max_num_seqs`, `gpu_memory_utilization` | Hardware default, Profile override | Sizing depends on the pair. |
| `genesis_env` (patches matrix) | Model canonical, Profile delta | Patch enable/disable per profile. |

An operator who wants a different capability MUST reference a
different ModelDef, not override the existing one.

### Profile delta semantics

```yaml
patches_delta:
  enable:                              # added on top of model.patches
    GENESIS_ENABLE_PN999_NEW_FEATURE: '1'
  disable:                             # removed from model.patches
    - GENESIS_ENABLE_PN90_PROBABILISTIC_DRAFT
  override:                            # value overrides (key wins)
    GENESIS_P82_THRESHOLD_SINGLE: '0.5'
```

Apply order: `enable → disable → override`. Conflicts within a
single profile (a key in both `enable` and `disable`) raise
SchemaError at load time. Cross-profile conflicts are caught by
`make audit-configs`.

### Sizing override

When the model × hardware pairing genuinely needs different sizing
than the hardware default, the profile carries an explicit override:

```yaml
sizing_override:
  max_model_len: 78000            # tighter ctx than hardware default
  gpu_memory_utilization: 0.95    # push higher because single-stream
  max_num_seqs: 1
  max_num_batched_tokens: 4096
  enable_chunked_prefill: true
  enforce_eager: false
  disable_custom_all_reduce: true
```

### Adding a new preset

Three orthogonal questions decide which layer changes:

1. **Different checkpoint, KV format, or spec method?**
   → New `builtin/model/<id>.yaml`.
2. **Different rig (GPU count, VRAM, image digest, mounts)?**
   → New `builtin/hardware/<id>.yaml`.
3. **Same model+hardware, different patches enable/disable or sizing
   knobs?**
   → New `builtin/profile/<id>.yaml`.

Then drop a 3-pointer preset alias:

```yaml
# builtin/presets/my-rig.yaml
model: qwen3.6-35b-a3b-fp8
hardware: a5000-2x-24gbvram-16cpu-128gbram
profile: my-experimental-wave10
```

Verify with `sndr launch my-rig --preflight-only` and
`make audit-configs`.

### Discovery CLI

```bash
sndr model list-v2          # every ModelDef (canonical patches summary)
sndr model show <model-id>  # capabilities + canonical patches dump
sndr hardware list          # every rig (n_gpus, vram, cuda cap, runtime)
sndr hardware show <hw-id>  # sizing defaults + runtime block + system_env
sndr profile list           # every profile (delta counts, sizing-override flag)
sndr profile show <id>      # delta + sizing override + promotion contract
sndr profile diff <id1> <id2>
```

### V1 ↔ V2 bridge

V2 is additive: the V1 launcher path still works for legacy preset
keys (`a5000-2x-35b-prod`, `a5000-2x-27b-int4-tq-k8v4`, ...). V2
aliases (`prod-qwen3.6-35b-balanced`, `prod-qwen3.6-27b-tq-k8v4`, ...) resolve through
`registry_v2` and produce the same V1 `ModelConfig` shape that the
existing emitters already consume. A byte-identical regression test
covers each preset. V1 deprecation lands in Phase 9; V2 layered
files remain the long-term source of truth.

## Contributing a community config

Every config — yours or builtin — goes through the same 5-stage
pipeline:

```text
                  ┌──────────────────────────────────────────────┐
                  │  YAML edit / submit                          │
                  └────────────────┬─────────────────────────────┘
                                   ↓
   1. validate    schema + 19 audit rules (offline, ~50 ms)
                                   ↓
   2. preflight   env + dependencies + hardware sanity (~3 s)
                                   ↓
   3. launch      boot vLLM container with this config (~2–5 min)
                                   ↓
   4. verify      bench + compare to reference_metrics (~5–10 min)
                                   ↓
   5. promote     advance lifecycle (community-test → -dev → -prod)
```

### Lifecycle gates

| Stage | Gate |
| --- | --- |
| → `community-dev` | Successful verify on submitter rig + 1 reviewer rig. |
| → `community-prod` | ≥ 2 verified rigs, `reference_metrics` set, ≥ 7 days since `test_started_at` (cooling-off). |

Promote command:

```bash
# Maintainer side, after their rig verifies:
sndr model-config promote my-rig --to community-dev \
    --rig-tag rtx-3090-2x --handle reviewerhandle

sndr model-config promote my-rig --to community-prod
```

### Portable mounts — don't hardcode personal paths

SNDR configs travel between rigs. Use symbolic mount references:

```yaml
docker:
  mounts:
    - ${models_dir}:/models:ro
    - ${hf_cache}:/root/.cache/huggingface:ro
    - ${sndr_src}:/usr/local/lib/python3.12/dist-packages/vllm/sndr_core:ro
    # Bad — non-portable:
    # - /home/user/specific-project/models:/models:ro
```

Variables resolve via `~/.sndr/host.yaml`, auto-detected at
`install.sh` time per rig. `sndr model-config validate` (audit rule
R-019) catches typos / missing host.yaml entries BEFORE launch.

### Declaring runtime support

```yaml
deploy:
  docker: true        # default — tested on docker
  podman: true        # if you've verified on podman
  bare_metal: true    # if your config also works as native venv
                      # (recommended on Proxmox kernel 6.17.x; see TROUBLESHOOTING)
  kubernetes: true    # set true after `sndr service install --yes` lands cleanly
  lxc_proxmox: true   # set true after the runnable LXC bootstrap is validated
  default: docker
```

Operator picks runtime at launch:

```bash
sndr model-config render <key>                      # uses deploy.default
sndr model-config render <key> --runtime bare_metal
sndr model-config render <key> --runtime podman
sndr model-config render <key> --runtime kubernetes
sndr model-config render <key> --runtime lxc_proxmox
```

All five runtimes emit runnable artefacts (audit C3 + C4 closure,
2026-05-16). The `sndr service install / start / stop / status /
logs / uninstall` lifecycle is wired for every backend.

### Failure diagnostic recipes

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `SchemaError: lifecycle must be one of ...` | typo | use exact `community-test` / `community-dev` / `community-prod` |
| `R-015 reference_metrics required` | `community-prod` without metrics | run `bench-and-update` first |
| `R-018 hybrid capacity OOM` | TQ k8v4 + Hybrid + KV mismatch | reduce `max_model_len` or `max_num_seqs` |
| `unknown env var GENESIS_ENABLE_P83` | references retired/archived patch | grep `dispatcher.py` PATCH_REGISTRY for valid IDs |
| `gpu_match_keys missing` | empty list | add at least one substring like `['rtx 3090']` |
| `verified_by < 2 for community-prod` | only one rig validated | get a second rig validation before promoting |
| boot fails with `cudaErrorIllegalAddress` on GDN | Cliff 2b on hybrid model | enable PN59 in `genesis_env` |
| boot fails with `prefix-cache + DS conv` | hybrid GDN + prefix cache | drop `--enable-prefix-caching` |

See [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) for the full cliff
catalogue + named rollback procedures.

### When NOT to submit a config

- You have regressions (TPS or quality) versus the closest builtin
  config without explanation. Root-cause the regression first.
- You depend on a vLLM pin different from the Genesis-supported
  pin — write it up as an issue first.
- Your config requires patches NOT in `PATCH_REGISTRY` — submit the
  patch first via [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Discovery API (Python)

```python
from vllm.sndr_core.model_configs.registry_v2 import (
    load_alias,            # alias → V1 ModelConfig
    load_model,            # ModelDef by id
    load_hardware,         # HardwareDef by id
    load_profile,          # ProfileDef by id
    list_models,           # all model ids
    list_hardware,         # all hardware ids
    list_profiles,         # all profile ids (optional parent_model filter)
    compose_by_ids,        # (model_id, hw_id, profile_id, runtime?) → ModelConfig
)
from vllm.sndr_core.model_configs.runtime_container import (
    build_runtime_container_spec,  # ModelConfig → canonical IR
)
```

## Looking ahead

When Genesis gets NVIDIA Blackwell (RTX 6000 Pro 96 GB planned):

- DeepSeek V4 Flash AWQ-INT4 (~80 GB) becomes feasible.
- Qwen3-Next-80B-FP8 fits in a single-card setup.
- NVFP4 quants unlock larger models at the same VRAM budget.
- Genesis patch port for `deepseek_v4` arch becomes worthwhile.

Until then — Qwen3.6-35B-A3B-FP8 + Genesis patches is the empirical
best for 2× A5000-class hardware.

## Reference

- Dataclass schema: `sndr/model_configs/schema_v2.py`
- Composer: `sndr/model_configs/compose.py`
- Registry helpers: `sndr/model_configs/registry_v2.py`
- RuntimeContainerSpec: `sndr/model_configs/runtime_container.py`
- Auto-generated inventory: [`CONFIGS_AUTO.md`](CONFIGS_AUTO.md)
- Community SDK guide: [`CONTRIBUTING.md` § Community SDK](CONTRIBUTING.md)
- Rollback procedures: [`TROUBLESHOOTING.md` § Rollback playbook](TROUBLESHOOTING.md)
