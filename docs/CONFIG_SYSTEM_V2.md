# Config system V2 — layered model configuration

**Status:** Phase 3 done (11 presets migrated), Phase 7 documentation draft.
**Audience:** operators running Genesis-patched vLLM and contributors adding new presets.

V2 splits the old monolithic `model_config.yaml` into four orthogonal layers
that compose into a runtime `ModelConfig`. The split makes it cheap to add
a new model on existing hardware, a new rig for existing models, or a
patch sweep without touching production presets.

## The four layers

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

## Discovery CLI

```bash
sndr model list-v2          # every ModelDef (canonical patches summary)
sndr model show <model-id>  # capabilities + canonical patches dump
sndr hardware list          # every rig (n_gpus, vram, cuda cap, runtime)
sndr hardware show <hw-id>  # sizing defaults + runtime block + system_env
sndr profile list           # every profile (delta counts, sizing-override flag)
sndr profile list --model <model-id>   # filter to one parent model
sndr profile show <profile-id>         # delta + sizing override + promotion contract
sndr profile diff <profile-id>         # patches matrix delta vs canonical model.patches
```

## Composition rules

Composer (`vllm/sndr_core/model_configs/compose.py`):

1. **Compat gate first.** `check_compat(model, hardware)` rejects pairings
   where `requires.min_gpu_count` / `min_total_vram_mib` / `min_cuda_capability`
   aren't met. Fails fast with an operator-facing message.
2. **Patches matrix = model.patches + profile.patches_delta** (enable →
   disable → override applied in that order).
3. **Sizing = profile.sizing_override OR hardware.sizing** (operator
   tuning for a specific model×hardware pair wins over the rig default).
4. **Runtime = hardware.runtime.default**, optionally overridden by
   `--runtime <name>` CLI flag (must be in `hardware.runtime.supported`).
5. **Final result is a V1 `ModelConfig`** — composer is a bridge so the
   existing launcher / k8s / compose / quadlet emitters work unchanged.

## Layer ownership rules

Each field has a **single owning layer**. Cross-layer conflict on an
owned field is rejected at load time.

| Field | Owned by | Why |
|---|---|---|
| `dtype`, `kv_cache_dtype`, `spec_decode.method`, `attention_arch` | Model | Different capability = different model file |
| `tool_call_parser`, `reasoning_parser` | Model | Tied to the checkpoint's tokenizer/template |
| `n_gpus`, `min_vram_per_gpu_mib`, `cuda_capability_min` | Hardware | Physical rig attribute |
| `image`, `image_digest`, `mounts`, `network` | Hardware | Container substrate |
| `system_env` (NCCL/PYTORCH/VLLM globals) | Hardware | Stable across models on the rig |
| `max_model_len`, `max_num_seqs`, `gpu_memory_utilization` | Hardware default, Profile override | Sizing depends on model×hardware pair |
| `genesis_env` (patches matrix) | Model canonical, Profile delta | Patch enable/disable per profile |

Operator who wants a different capability MUST reference a different
ModelDef, not override the existing one.

## The eleven validated presets

| Alias | Model | Hardware | Profile | Lifecycle |
|---|---|---|---|---|
| `prod-35b` | qwen3.6-35b-a3b-fp8 | a5000-2x-24gb | 35b-balanced | stable |
| `prod-27b-tq` | qwen3.6-27b-int4-autoround-tq-k8v4 | a5000-2x-24gb | 27b-tq-k8v4 | stable |
| `prod-35b-dflash` | qwen3.6-35b-a3b-fp8-dflash | a5000-2x-24gb | 35b-dflash | stable |
| `prod-27b-dflash` | qwen3.6-27b-dflash | a5000-2x-24gb | 27b-dflash | stable |
| `long-ctx-27b` | qwen3.6-27b-int4-autoround-fp8kv | a5000-2x-24gb | 27b-long-ctx | stable |
| `qa-27b-tested` | qwen3.6-27b-int4-autoround-fp8kv | a5000-2x-24gb | qa-27b-fp8kv | tested (QA-only) |
| `qa-27b-tq-1x` | qwen3.6-27b-int4-autoround-tq-k8v4 | a5000-1x-24gb | qa-27b-tq-1x | tested (single-card QA) |
| `experimental-27b-tq-dflash-ab` | qwen3.6-27b-int4-autoround-tq-k8v4 | a5000-2x-24gb | ab-27b-tq-dflash | experimental |
| `example-2x-tier-aware` | qwen3.6-27b-int4-autoround-tq-k8v4 | a5000-2x-24gb | tier-aware-2x | experimental |
| `example-3090-dense-cpu-offload` | qwen3.6-7b-dense | single-3090-24gb | cpu-offload-3090 | experimental |
| `example-3090-tier-aware` | qwen3.6-27b-int4-autoround-tq-k8v4 | single-3090-24gb | tier-aware-3090 | experimental |

The release gate `make audit-configs` walks every preset and verifies
the triplet composes cleanly.

## Adding a new preset

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

Verify with `sndr launch my-rig --preflight-only` and `make audit-configs`.

## Profile delta semantics

```yaml
patches_delta:
  enable:                              # added on top of model.patches
    GENESIS_ENABLE_PN999_NEW_FEATURE: '1'
  disable:                             # removed from model.patches
    - GENESIS_ENABLE_PN90_PROBABILISTIC_DRAFT
  override:                            # value overrides (key wins)
    GENESIS_P82_THRESHOLD_SINGLE: '0.5'
```

Apply order: `enable → disable → override`. Conflicts within a single
profile (a key in both `enable` and `disable`) raise SchemaError at
load time. Cross-profile conflicts are caught by `make audit-configs`.

## Sizing override

When the model×hardware pairing genuinely needs different sizing than
the hardware default, the profile carries an explicit override:

```yaml
sizing_override:
  max_model_len: 78000           # tighter ctx than hardware default
  gpu_memory_utilization: 0.95   # push higher because single-stream
  max_num_seqs: 1                # single-stream optimized
  max_num_batched_tokens: 4096
  enable_chunked_prefill: true
  enforce_eager: false
  disable_custom_all_reduce: true
```

If `sizing_override` is absent, the composer falls back to
`hardware.sizing`. The override exists because **sizing depends on
the model AND the hardware** — a 27B model on a 1× A5000 cannot use
the same `max_model_len` as a 35B model on a 2× A5000.

## V1 ↔ V2 bridge

V2 is additive: the V1 launcher path still works for legacy preset
keys (`a5000-2x-35b-prod`, `a5000-2x-27b-int4-tq-k8v4`, etc.). V2
aliases (`prod-35b`, `prod-27b-tq`, ...) resolve through `registry_v2`
and produce the same V1 `ModelConfig` shape that the existing emitters
already consume. Byte-identical regression test covers each preset.

V1 freeze (Phase 9) deprecates V1 loader entries after Phase 8b
acceptance; the V2 layered files remain the source of truth.

## Runtime backends

The composed `ModelConfig` plus the chosen runtime feeds into a single
canonical `RuntimeContainerSpec` (Phase 4.5). Every emitter — docker
argv, docker-compose YAML, podman quadlet, kubernetes manifest — reads
the same IR. Switching `--runtime` only changes the output format; the
semantic fields (mounts, env, ports, security) are identical across
backends.

Acceptance check (Phase 4.5):

```bash
sndr launch prod-35b --dry-run --runtime docker     > /tmp/d.txt
sndr launch prod-35b --dry-run --runtime compose    > /tmp/c.txt
sndr launch prod-35b --dry-run --runtime quadlet    > /tmp/q.txt
sndr launch prod-35b --dry-run --runtime kubernetes > /tmp/k.yaml
# Semantic diff between any two outputs = 0 (only format differs).
```

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

## Reference

- Dataclass schema: `vllm/sndr_core/model_configs/schema_v2.py`
- Composer: `vllm/sndr_core/model_configs/compose.py`
- Registry helpers: `vllm/sndr_core/model_configs/registry_v2.py`
- RuntimeContainerSpec: `vllm/sndr_core/model_configs/runtime_container.py`
- Community SDK guide: `docs/COMMUNITY_PATCHES.md`
- Rollback procedures: `docs/ROLLBACK_PLAYBOOK.md`
