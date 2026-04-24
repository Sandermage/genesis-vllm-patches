# MoE Tuning Configs (community-contributed, not in upstream vLLM)

This directory holds pre-tuned Triton kernel configurations for the fused-MoE
kernel on hardware/shape combinations that vLLM upstream does **not** ship.

vLLM looks up tuned configs by filename at runtime:
`vllm/model_executor/layers/fused_moe/configs/E={experts},N={intermediate},device_name={gpu},dtype={dtype},block_shape={shape}.json`

If the matching file is present, `fused_moe` loads it and skips the on-the-fly
Triton autotuner path (which on Ampere leaves ~15-20% on the table for
small/medium batch sizes).

---

## Files here

### `E=256,N=512,device_name=NVIDIA_RTX_A5000,dtype=fp8_w8a8,block_shape=[128,128].json`

- **Source**: upstream PR [vllm-project/vllm#40129](https://github.com/vllm-project/vllm/pull/40129)
  (community-contributed, **CLOSED without merge** — maintainers declined to
  accept tuning configs for consumer/workstation Ampere cards).
- **Target shape**: MoE with `E=256` experts, `N=512` intermediate size,
  FP8 W8A8 weights, block scaling `[128,128]`.
- **Target hardware**: NVIDIA RTX A5000 (Ampere, SM 86, 24 GB).
- **Target model**: Qwen3.6-35B-A3B-FP8 — exactly our production model on
  VM 100 (192.168.1.10).
- **Batch sizes covered**: 1, 2, 4, 8, 16, 32, 64, 128, 256.
- **Measured uplift** (from PR body): +16% generation tok/s
  (~125 -> ~145 tok/s), best on small/medium batch, tapering at batch >= 128.
- **Tuning methodology**: vLLM's built-in Triton autotuner, 100 iterations per
  batch size, best config by lowest kernel time.

Key observations from the PR author for SM 86:
- `BLOCK_SIZE_M=16` is optimal — MoE routes few tokens per expert.
- `BLOCK_SIZE_K` varies 64-256 with batch size.
- `num_stages` kept conservative (1-4) because of SM 86 shared-memory limits.

---

## How to use it (operator runbook)

The file must land inside the vLLM container at the path vLLM probes:
`/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/`
(exact Python version depends on the container image — adjust if different).

### Step 1 — find the vLLM configs directory inside the container

```bash
docker exec vllm-qwen python -c \
  "import vllm, os; \
   p = os.path.join(os.path.dirname(vllm.__file__), \
   'model_executor/layers/fused_moe/configs'); \
   print(p)"
```

### Step 2 — copy the file in

From the repo root on the host (VM 100, where this repo is checked out):

```bash
docker cp \
  "vllm/_genesis/configs/moe_tuning/E=256,N=512,device_name=NVIDIA_RTX_A5000,dtype=fp8_w8a8,block_shape=[128,128].json" \
  vllm-qwen:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/
```

Replace `vllm-qwen` with the actual container name (`docker ps | grep vllm`) and
`/usr/local/lib/python3.12/...` with the path printed by Step 1 if different.

### Step 3 — restart the container so vLLM re-reads the configs dir

```bash
docker compose restart vllm-qwen
```

Or, if your compose file is separate for the vLLM service:

```bash
cd /opt/genesis/vllm && docker compose restart
```

### Step 4 — verify it loaded

Tail the logs and look for a `Using configuration from ...A5000...` line:

```bash
docker logs -f vllm-qwen 2>&1 | grep -i "configuration from\|fused_moe"
```

If you see `A5000,dtype=fp8_w8a8,block_shape=[128,128].json` referenced — it's live.
If you see `Using default MoE config` — the filename didn't match exactly
(check `E=`, `N=`, `device_name=NVIDIA_RTX_A5000` spelling; GPU names from
`torch.cuda.get_device_name(0)` must match byte-for-byte).

---

## Persistence warning

`docker cp` writes into the container's read/write layer. A `docker compose
restart` or `stop && start` keeps it. A `docker compose down && up -d` (or a
`docker rm`) **wipes it** and you must re-copy.

For permanent install, bake it into a custom image:

```Dockerfile
FROM vllm/vllm-openai:<tag>
COPY "E=256,N=512,device_name=NVIDIA_RTX_A5000,dtype=fp8_w8a8,block_shape=[128,128].json" \
     /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/
```

Or mount the configs directory as a volume in `docker-compose.yml`.

---

## Why we carry this locally

Upstream closed PR #40129 without merging (policy: no consumer-card tuning
configs in-tree). Our production Qwen3.6-35B-A3B-FP8 on 2x A5000 matches the
exact `E=256,N=512,block_shape=[128,128]` shape the PR targets, so the uplift
is directly applicable. This directory is the Genesis-side home for any other
community-contributed MoE configs we find useful.
