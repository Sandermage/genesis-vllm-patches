# Path C — Tier-aware KV Cache (PN95)

Operator-facing guide for the v7.73.x **tier-aware KV cache** (PN95).
Designed for operators running **single-3090 / single-A5000-class** GPUs
who hit OOM on long-context + vision workloads. Solves
[club-3090 issue #58](https://github.com/noonghunna/club-3090/issues/58).

---

## When you need this

You are a candidate for PN95 if **all** of these are true:

1. You're on a single 24 GiB GPU (3090, A5000, RTX 4090, L4 24G).
2. Your `max_model_len` is ≥ 100K tokens.
3. Your model is hybrid-GDN (Qwen3.6-27B/35B-A3B, Qwen3-Next, etc) —
   i.e. you have `GENESIS_ENABLE_PN59_STREAMING_GDN=1` set.
4. You serve multimodal (vision) requests.
5. You see OOM crashes after ~5-7 chat turns.

If all five match — Path C is yours. If you're on dense (non-hybrid)
models, see [Path A](#alternative-path-a-dense-only) below; it's
simpler and has been around longer.

If you don't hit OOM, you don't need Path C — its TPS regression
(~10-30% on tier-move-heavy workloads) is real and you'd be paying
for capability you don't need.

---

## How it works

Genesis ships a `TierManager` that owns a multi-tier KV cache
hierarchy: `gpu` → `cpu` (pinned RAM) → optional `nvme`.

When free GPU VRAM drops below `tier_low_water_pct`, the manager:

1. Walks tier-0 (GPU) pages.
2. **Skips MambaSpec groups** (Mamba SSM state stays on GPU — this is
   the bit upstream CPU-offload paths get wrong).
3. **Drains MM/vision pages first** (image tokens have lower
   attention re-use than text-prefix tokens, so they amortize the
   PCIe round-trip).
4. Demotes the rest LRU-order to the CPU pinned-RAM pool.
5. On hit: promotes back to GPU (cudaMemcpyAsync, separate stream
   from cudagraph capture).

The **MambaSpec exclusion is mandatory** — every other CPU offload
implementation tries to memcpy MambaSpec groups and crashes because
the layout is `(num_layers, head_dim, conv_state_dim)` instead of
the standard KV `(num_layers, num_kv_heads, head_dim, block_size)`.

---

## Quick start

### 1. Update your model config

Add a `cache_config.tiers` block. Example for single 3090 + 27B-A3B:

```yaml
cache_config:
  # PN91 single-tier defaults (back-compat)
  eviction_policy: lru

  # PN95 multi-tier extension
  tiers:
    - device: gpu
      capacity_gib: 20.0
      eviction_policy: lru
      promote_on_hit: true
      demote_threshold_pct: 0.92
      low_water_pct: 0.75
    - device: cpu
      capacity_gib: 40.0
      eviction_policy: lru
      vision_first: true        # demote image pages first
      pinned: true              # cudaMallocHost
  exclude_mamba_ssm: true       # MUST stay True on hybrid GDN
  vision_demote_first: true
  tier_low_water_pct: 0.05      # demote when <5% GPU VRAM free
  async_demote: true
```

### 2. Enable PN95 at launch

```bash
export GENESIS_ENABLE_PN95_TIER_AWARE_CACHE=1
sndr launch <your-config-key>
```

### 3. Monitor

Watch the `[PN95]` log lines on engine boot — they'll print the
TierManager stats and confirm Mamba groups are excluded. Run your
workload past the previous OOM point. Decode TPS will be 10-30%
lower than GPU-only baseline; that's the expected trade.

### 4. Verify with bench-compare

```bash
sndr bench-compare baseline.json with-pn95.json --metric wall_TPS
```

---

## Configuration reference

### `CacheTier` fields

| Field | Default | Notes |
|---|---|---|
| `device` | (required) | `'gpu'`, `'cpu'`, or `'nvme'` |
| `capacity_gib` | (required) | hard cap on this tier's allocation |
| `eviction_policy` | `lru` | `lru` / `2q` / `arc` |
| `promote_on_hit` | `True` | demoted page hit → bring back to upper tier |
| `demote_threshold_pct` | `0.92` | tier fill ratio that triggers demote |
| `low_water_pct` | `0.75` | demote until this ratio reached |
| `vision_first` | `False` | drain mm pages before text |
| `pinned` | `True` | for `cpu` tier: cudaMallocHost-backed |
| `nvme_path` | None | required when `device == 'nvme'` |

### `CacheConfig` PN95 fields

| Field | Default | Notes |
|---|---|---|
| `tiers` | `[]` | empty = PN91 single-tier behavior unchanged |
| `exclude_mamba_ssm` | `True` | MUST stay True on hybrid-GDN |
| `vision_demote_first` | `True` | sub-policy mirror |
| `tier_low_water_pct` | `0.05` | GPU free-VRAM threshold to trigger demote |
| `async_demote` | `True` | cudaMemcpyAsync vs sync |

---

## Constraints + safety

**Path A (`OffloadConfig.cpu_offload_gib`) on hybrid-GDN configs:**
the schema validator REFUSES this combination unless Path C is also
declared. Why: vLLM's stock `--cpu-offload-gb` doesn't know to skip
MambaSpec groups. Path A is dense-only; Path C is the hybrid-GDN
solution.

**Pinned-memory budget:** the CPU tier uses `cudaMallocHost`. Refuses
to enable when `host_ram_gib < 1.5 * tiers_cpu_capacity_gib`. Set
`ulimit -l unlimited` or run inside Proxmox LXC with `memlock=unlimited`.

**MTP K=3 spec-decode:** the manager keeps the last `2*(num_spec+1)`
admit-order pages in a "hot ring" that refuses demotion. This avoids
the PCIe round-trip on the verify path.

**Cudagraph capture:** demote runs on a separate CUDA stream from
cudagraph capture. `FULL_AND_PIECEWISE` mode is supported.

---

## Alternative: Path A (dense-only)

If your model is **dense** (no Mamba SSM state — Llama, Qwen2.5, Gemma,
etc), Path A is simpler:

```yaml
offload:
  cpu_offload_gib: 16.0
  swap_space_gib: 4.0
```

Translates to `--cpu-offload-gb 16 --swap-space 4` at launch. Uses
vLLM's stock CPU offload — no Genesis runtime overhead. NOT for
hybrid-GDN; the schema validator blocks that combination outright.

See [`single-3090-dense-cpu-offload-EXAMPLE.yaml`](../vllm/sndr_core/model_configs/builtin/single-3090-dense-cpu-offload-EXAMPLE.yaml).

---

## When NOT to use PN95

- 2× A5000+ rigs with TP=2: you have 48 GiB of VRAM. PN17 (FA2 LSE
  clamp) widens long-text envelope to 205K. PN95's tier-move cost is
  pure overhead at that capacity.
- Pure dense models: use Path A or vLLM's stock `--cpu-offload-gb`
  directly.
- Non-vision workloads: vision-first demote is the headline win;
  without MM pages, demote is just LRU which is no better than vLLM
  stock prefix-cache eviction.

---

## Status (v7.73.x)

Path C foundation (Days 1-4 of the 9-day implementation plan)
shipped 2026-05-09:

- Schema, tier_manager, text-patch wire-in, dispatcher entry — DONE
- Vision-token MM tagging at admit (Day 5) — DEFERRED to live
  integration session
- Mamba runtime classifier walk (Day 6) — DEFERRED
- Live 27B Lorbus bench on 2× A5000 reproducer (Day 7) — DEFERRED
- EXAMPLE config — DONE (community-test lifecycle)
- Cross-rig validation by noonghunna on actual 3090 — PENDING

For implementation deep-dive see the research design note shipped to
maintainers (internal); the public summary is in this document.
