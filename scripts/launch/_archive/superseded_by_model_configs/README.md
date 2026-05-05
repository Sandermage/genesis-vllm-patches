# Archived launch scripts (2026-05-05)

These 17 `.sh` scripts (`start_*.sh` + `bare_metal_*.sh`) were the
historical way to launch Genesis-patched vLLM. Each script bundled
together: env vars + docker run flags + vllm serve args.

The problem: any change to a patch set, env flag, or hardware tweak
required editing the right script — and the same config existed in
two forms (start_* for docker, bare_metal_* for native), so changes
had to be duplicated.

## Replaced by `vllm/_genesis/model_configs/builtin/*.yaml`

Each YAML config captures:
- All Genesis env vars (`GENESIS_ENABLE_*`)
- All system env (`PYTORCH_*`/`VLLM_*`/`NCCL_*`/...)
- All vllm serve flags (max_model_len, util, kv_dtype, ...)
- Docker setup (image, mounts, port, network)
- Spec decode config (method, K)
- **Reference metrics** (TPS, tool-call, CV, VRAM from validated bench)
- **Verify tolerances** (drop %, min tool, max CV, max VRAM grow)
- Audit notes (gotchas, conflicts, when NOT to pick)

Single launcher: `scripts/launch.sh <config-key>`.

## Migration map

| Old script | New config key |
|---|---|
| `start_35b_fp8_PROD.sh` (+ `bare_metal_*`) | `a5000-2x-35b-prod` |
| `start_27b_int4_no_TQ_short.sh` (+ `bare_metal_*`) | `a5000-2x-27b-int4-balanced` |
| `start_27b_int4_no_TQ_short_single_card.sh` (+ `bare_metal_*`) | `a5000-1x-27b-int4-balanced` |
| `start_27b_int4_TQ_k8v4.sh` (+ `bare_metal_*`) | not migrated (broken under cudagraph FULL — see audit_rule R-010) |
| `start_27b_int4_no_TQ_long_256K.sh` (+ `bare_metal_*`) | TODO — community PR welcome |
| `start_v793_27b_PN12_test.sh` | research artefact, not migrated |

## To use a migrated config

```bash
# Browse
./scripts/launch.sh list

# Validate (offline, fast)
./scripts/launch.sh a5000-2x-35b-prod --validate

# Preflight (env check)
./scripts/launch.sh a5000-2x-35b-prod --preflight

# Launch
./scripts/launch.sh a5000-2x-35b-prod

# After it boots
python3 -m vllm._genesis.compat.cli model-config diagnose a5000-2x-35b-prod
python3 -m vllm._genesis.compat.cli model-config verify a5000-2x-35b-prod
```

## To add your own config

```bash
python3 -m vllm._genesis.compat.cli model-config new my-rig \
    --template a5000-2x-35b-prod
# Edit ~/.genesis/model_configs/my-rig.yaml
./scripts/launch.sh my-rig
# After bench, share via PR to vllm/_genesis/model_configs/community/
```

## Why these are kept (not deleted)

Historical archeology: when investigating "did config X ever work on
hardware Y", it's useful to find the exact env block + flags that were
used at the time. These files are read-only reference; do not edit.
