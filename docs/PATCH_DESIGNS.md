# Patch design appendices

Technical design notes for the individual patch families that are
substantial enough to need more than a [`PATCHES.md`](PATCHES.md)
entry. Each appendix is self-contained — operators can read just the
section they need.

Sections:

1. [PN95 — tier-aware KV cache](#pn95--tier-aware-kv-cache) — Path C
   for single-3090 / single-A5000 24 GB long-context + vision rigs.
2. [GDN kernel fusion roadmap](#gdn-kernel-fusion-roadmap) — proposed
   fusion path to lower TTFT and raise prefill TPS on hybrid-GDN
   models.
3. [Qwen3 reasoning vs content streaming contract](#qwen3-reasoning-vs-content-streaming-contract)
   — the `enable_thinking` / `content=null` invariant for
   OpenAI-compatible clients on Qwen3-class models.
4. [`Genesis → sndr_core` rename (v11.0.0)](#genesis--sndr_core-rename-v1100)
   — what changed during the v10 → v11 hard flip and how to
   migrate a pre-v11 install.

## PN95 — tier-aware KV cache

Operator-facing guide for the tier-aware KV cache patch (PN95).
Designed for operators running **single-3090 / single-A5000-class**
GPUs who hit OOM on long-context + vision workloads. Solves
[club-3090 issue #58](https://github.com/noonghunna/club-3090/issues/58).

### When you need this

You are a candidate for PN95 if **all** of these are true:

1. You're on a single 24 GiB GPU (3090, A5000, RTX 4090, L4 24 G).
2. Your `max_model_len` is ≥ 100K tokens.
3. Your model is hybrid-GDN (Qwen3.6-27B / 35B-A3B, Qwen3-Next, ...)
   — i.e. you have `GENESIS_ENABLE_PN59_STREAMING_GDN=1` set.
4. You serve multimodal (vision) requests.
5. You see OOM crashes after ~5–7 chat turns.

If all five match — Path C is yours. If you're on dense (non-hybrid)
models, see [Path A](#alternative-path-a-dense-only) below; it's
simpler and has been around longer.

If you don't hit OOM, you don't need Path C — its TPS regression
(~10–30% on tier-move-heavy workloads) is real and you'd be paying
for capability you don't need.

### How it works

Genesis ships a `TierManager` that owns a multi-tier KV cache
hierarchy: `gpu` → `cpu` (pinned RAM) → optional `nvme`.

When free GPU VRAM drops below `tier_low_water_pct`, the manager:

1. Walks tier-0 (GPU) pages.
2. **Skips MambaSpec groups** (Mamba SSM state stays on GPU — this
   is the bit upstream CPU-offload paths get wrong).
3. **Drains MM/vision pages first** (image tokens have lower
   attention re-use than text-prefix tokens, so they amortise the
   PCIe round-trip).
4. Demotes the rest LRU-order to the CPU pinned-RAM pool.
5. On hit: promotes back to GPU (`cudaMemcpyAsync`, separate stream
   from cudagraph capture).

The **MambaSpec exclusion is mandatory** — every other CPU offload
implementation tries to memcpy MambaSpec groups and crashes because
the layout is `(num_layers, head_dim, conv_state_dim)` instead of
the standard KV `(num_layers, num_kv_heads, head_dim, block_size)`.

### Quick start

#### 1. Update your model config

Add a `cache_config.tiers` block. Example for single 3090 + 27B-A3B:

```yaml
cache_config:
  # Single-tier defaults (back-compat with PN91)
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

#### 2. Enable PN95 at launch

```bash
export GENESIS_ENABLE_PN95_TIER_AWARE_CACHE=1
sndr launch <your-config-key>
```

#### 3. Monitor

Watch the `[PN95]` log lines on engine boot — they print the
TierManager stats and confirm Mamba groups are excluded. Run your
workload past the previous OOM point. Decode TPS will be 10–30%
lower than GPU-only baseline; that's the expected trade.

#### 4. Verify

```bash
sndr verify <your-config-key>          # compare to reference_metrics
sndr memory --live                     # live tier breakdown
sndr patches pn95 status               # PN95 internal state
sndr patches pn95 dump --json          # full dump for triage
```

### Configuration reference

**`CacheTier` fields:**

| Field | Default | Notes |
| --- | --- | --- |
| `device` | (required) | `'gpu'`, `'cpu'`, or `'nvme'`. |
| `capacity_gib` | (required) | hard cap on this tier's allocation. |
| `eviction_policy` | `lru` | `lru` / `2q` / `arc`. |
| `promote_on_hit` | `True` | demoted page hit → bring back to upper tier. |
| `demote_threshold_pct` | `0.92` | tier fill ratio that triggers demote. |
| `low_water_pct` | `0.75` | demote until this ratio is reached. |
| `vision_first` | `False` | drain MM pages before text. |
| `pinned` | `True` | for `cpu` tier: `cudaMallocHost`-backed. |
| `nvme_path` | None | required when `device == 'nvme'`. |

**`CacheConfig` PN95 fields:**

| Field | Default | Notes |
| --- | --- | --- |
| `tiers` | `[]` | empty = PN91 single-tier behaviour unchanged. |
| `exclude_mamba_ssm` | `True` | MUST stay True on hybrid-GDN. |
| `vision_demote_first` | `True` | sub-policy mirror. |
| `tier_low_water_pct` | `0.05` | GPU free-VRAM threshold to trigger demote. |
| `async_demote` | `True` | `cudaMemcpyAsync` vs sync. |

### Constraints and safety

- **Path A (`OffloadConfig.cpu_offload_gib`) on hybrid-GDN configs.**
  The schema validator REFUSES this combination unless Path C is also
  declared. vLLM's stock `--cpu-offload-gb` doesn't know to skip
  MambaSpec groups. Path A is dense-only; Path C is the hybrid-GDN
  solution.
- **Pinned-memory budget.** The CPU tier uses `cudaMallocHost`. The
  validator refuses to enable when
  `host_ram_gib < 1.5 * tiers_cpu_capacity_gib`. Set
  `ulimit -l unlimited` or run inside Proxmox LXC with
  `memlock=unlimited`.
- **MTP K=3 spec-decode.** The manager keeps the last
  `2 × (num_spec + 1)` admit-order pages in a "hot ring" that refuses
  demotion. This avoids the PCIe round-trip on the verify path.
- **Cudagraph capture.** Demote runs on a separate CUDA stream from
  cudagraph capture. `FULL_AND_PIECEWISE` mode is supported.

### Alternative: Path A (dense-only)

For dense models (no Mamba SSM state — Llama, Qwen2.5, Gemma, …),
Path A is simpler:

```yaml
offload:
  cpu_offload_gib: 16.0
  swap_space_gib: 4.0
```

Translates to `--cpu-offload-gb 16 --swap-space 4` at launch. Uses
vLLM's stock CPU offload — no Genesis runtime overhead. NOT for
hybrid-GDN; the schema validator blocks that combination outright.

See `single-3090-dense-cpu-offload-EXAMPLE.yaml` under
`vllm/sndr_core/model_configs/builtin/`.

### When NOT to use PN95

- **2× A5000+ rigs with TP=2.** You have 48 GiB of VRAM. PN17 (FA2
  LSE clamp) widens the long-text envelope to 205K. PN95's tier-move
  cost is pure overhead at that capacity.
- **Pure dense models.** Use Path A or vLLM's stock
  `--cpu-offload-gb` directly.
- **Non-vision workloads.** Vision-first demote is the headline win;
  without MM pages, demote is just LRU which is no better than
  vLLM's stock prefix-cache eviction.

## GDN kernel fusion roadmap

Research note. The Wave 10 bench on dev371 showed that **GDN
prefill** structurally caps TTFT on conc=8 at ~237 ms (operator
target 100–120 ms) and prevents per-token prefill cost from
dropping below ~163 µs/tok on 35B-A3B-FP8. This appendix records
the proposed fusion path; status is "monitor upstream, implement
Phase 1 if upstream hasn't merged equivalent by 2026-06-01".

### Problem — 6 sequential Triton kernels × 30 GDN layers

Each GDN layer (Mamba2-style) in prefill invokes six Triton kernels
in series:

```text
mixed_qkvz, _ = in_proj_qkvz(hidden_states)        # GEMM (Marlin / cuBLAS)
ba,         _ = in_proj_ba(hidden_states)          # GEMM
[split / reshape / contiguous chain]               # — PN50 fused
torch.ops.vllm.gdn_attention_core(...) → calls FLA:
  1. chunk_local_cumsum_scalar      (Triton)
  2. chunk_scaled_dot_kkt_fwd       (Triton)
  3. solve_tril_chunk_inv           (Triton)
  4. recompute_w_u_fwd              (Triton)
  5. chunk_gated_delta_rule_fwd_h   (Triton, recurrence)
  6. chunk_fwd_o                    (Triton)
```

On **35B-A3B FP8 / 30 GDN layers**:

- 30 × 6 = **180 Triton kernel launches** per prefill forward pass.
- Each launch ~50–100 µs CPU overhead (driver + kernel dispatch +
  cmd queue).
- Total: 9–18 ms of pure overhead **just for launches**, without
  compute.

On SM 8.6 (A5000), occupancy on short kernels is insufficient to
hide the launch latency. **This is the TTFT floor.**

### What is already done (low-hanging)

- **PN50 (SGLang #21019)** — fused split / reshape / cat / `.contiguous`
  (6 ops → 1 Triton kernel).
- **PN106** — GDN scratch pool 2/2 (eliminates per-call alloc).
- **P28** — GDN `core_attn_out` prealloc.
- **P39a** — FLA `chunk_scaled_dot_kkt` persistent A pool.

All four are per-call savings; none reduces the number of kernel
launches.

### Fusion plan — 6 kernels → 3 (target)

**Phase 1 — fuse kernels 1 + 2** (`chunk_local_cumsum` +
`chunk_scaled_dot_kkt_fwd`). Dependency: cumsum result is input for
KKT. Currently separate kernels because of different grid topology
(cumsum by chunks, KKT by chunk pairs). Solution: one Triton kernel
with two program-ids — `pid_chunk` for cumsum strip, `pid_pair` for
KKT inside one chunk.

Expected: 2 launches → 1, saves ~50–100 µs × 30 layers =
**1.5–3 ms TTFT**.

**Phase 2 — fuse kernels 3 + 4** (`solve_tril_inv` +
`recompute_w_u_fwd`). Tril solve is strictly sequential by rows;
`recompute_w_u` consumes the solve output. Fuse via cooperative
warps: one warp team does back-substitution, the second team starts
the w/u computation as soon as a diagonal row finishes.

Expected: another 1–2 ms × 30 layers = **30–60 ms TTFT**.

**Phase 3 — fuse kernels 5 + 6** (`chunk_gated_delta_rule_fwd_h` +
`chunk_fwd_o`). This is the hot spot — the gated delta-rule
recurrence (kernel 5) takes 60–70% of GDN time. `chunk_fwd_o` uses
the `h` state from kernel 5. Currently separate because state is
written to global memory between iterations.

Solution: keep state in shared memory / registers across the
boundary:

```python
@triton.jit
def fused_recurrence_o(
    q, k, v, g, beta,
    o_ptr,
    T, H, D_k, D_v,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,  # chunk size = 64
):
    # State h is [D_v, D_k] — keep in registers if D_v*D_k*4 < 96 KB.
    # SM 8.6: 100 KB shared mem / SM.
    # For D_v=128, D_k=128: 64 KB → fits.
    h = tl.zeros((BLOCK_D_v, BLOCK_D_k), dtype=tl.float32)
    for c in range(num_chunks):
        # gated delta-rule update of h in registers
        # immediately compute o[c] = q[c] @ h (no global write of h)
        tl.store(o_ptr + c * BLOCK_T * D_v + ..., o_chunk)
```

Expected: −1 launch + lower memory traffic =
**3–5 ms TTFT × 30 layers = 90–150 ms**.

**Phase 4 — numerical validation.** Each fused kernel must match
unfused output bit-for-bit (FP16/BF16 tolerance 1e-3 abs, 1e-4 rel):

1. Random inputs, varying chunk count `{1, 4, 8, 16}`.
2. Edge cases: T not a multiple of `BLOCK_C`.
3. Real model: compare logits / last_hidden_states with PN50 + PN106
   + fused stack vs upstream stack — KL divergence < 1e-4.

### Risk + mitigation

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| FP precision drift on fused recurrence | high | medium | accumulate state in FP32, downcast on store |
| Shared-mem overflow on SM 8.6 (100 KB) | medium | high | conditional `BLOCK_D` split, fall back to unfused for large `head_dim` |
| Triton autotune fails to find good config | medium | medium | hand-tune for our shapes (BS=1..8, T=1..32K) |
| Inductor compile interaction | low | high | mark fused kernel as opaque via `torch.library.custom_op` |
| GDN layer count varies between models | low | low | configure via `model_config.num_gdn_layers` lookup |

### Effort + expected gains

| Phase | Effort | Risk |
| --- | --- | --- |
| Phase 1 (fuse 1+2) | 2–3 days | low |
| Phase 2 (fuse 3+4) | 3–5 days | medium (tril solve fusion tricky) |
| Phase 3 (fuse 5+6) | 5–7 days | high (recurrence in registers) |
| Phase 4 (validation) | 2–3 days | — |

Total: 12–18 working days (~3–4 weeks calendar).

If all three phases succeed:

- TTFT @ conc=8: 237 ms → estimated **150–180 ms (−25 % to −36 %)**.
- TTFT @ conc=1: 59 ms → estimated **35–45 ms (−25 %)**.
- Aggregate TPS @ conc=8: 689 → estimated **800–900 (+15 % to +30 %)**.
- Per-request TPS @ conc=1: 215 → estimated **240–260 (+12 % to
  +21 %)** — may reach the operator target of 240 TPS.

### Alternative — `torch.compile` `mode='reduce-overhead'`

On dev371, Inductor `mode='reduce-overhead'` automatically fuses
small kernels. Try `--compilation-config.optimization_level O3` and
compare; if Inductor handles it, custom fusion isn't needed. Quick
experiment: launch with `optimization_level=3`, inspect the boot
log for "Inductor cache miss" frequency, then measure TTFT/TPS.

### Open research questions

1. Lock-free per-chunk state update — possible on SM 8.6 without
   deadlock?
2. Tensor cores on FP32 accumulator: SM 8.6 has limited TF32
   throughput (none in the FP8 path); affects the recurrence kernel.
3. Memory layout: `A_ptr` / `o_ptr` — which stride pattern is
   optimal for L2? Needs a micro-benchmark.
4. FlashInfer GDN kernel — landed for Hopper SM 9.0+. Can FlashInfer's
   design be ported (without Hopper-specific instructions) to
   Ampere?

### Decision: do-not-implement until pin ≥ v0.22.x

The current vLLM pin (dev371, `0.20.2rc1`) is in active development;
upstream may land equivalent fusion in the next few months.
Trade-off:

- **Implement now**: 3–4 weeks of work, risk that upstream merges an
  equivalent.
- **Wait for v0.22**: possible upstream fusion. Pin bump in v0.22.x
  = re-evaluation.

**Decision.** Monitor upstream `vllm-project/vllm` GDN PRs until
2026-06-01. If nothing has merged, implement Phase 1 (lowest risk,
biggest win/effort ratio).

### Research sources

- FLA repo: <https://github.com/fla-org/flash-linear-attention>
- Mamba2 paper: arXiv 2405.21060 (Dao 2024).
- SGLang PR #21019 (PN50 backport — split / reshape fusion).
- Triton tutorial 09 "persistent matmul" — pattern for in-register
  state.
- vLLM #41446 (`chunk_o` scale-fold pattern — PN29 backport, opt-in).

## Qwen3 reasoning vs content streaming contract

Applies to Qwen3-6-class models (27B, 35B-A3B, ...) that support
"thinking mode". Not relevant for non-reasoning models.

### Problem

A request with a small `max_tokens` (32 / 96 / 192) on a Qwen3 model
can return `message.content = null` with the whole token budget in
`message.reasoning`. OpenAI-compatible clients that expect a
non-empty `content` see an empty response.

Symptom (live PROD smoke, 27B on dev209):

```json
{
  "finish_reason": "length",
  "message": {
    "role": "assistant",
    "content": null,
    "reasoning": "Here's a thinking process..."
  },
  "usage": {"completion_tokens": 32}
}
```

All 32 tokens went into reasoning; content is empty.

### Genesis architectural solution

The PN16 lazy-reasoner patch (v2 — see
`vllm/sndr_core/middleware/lazy_reasoner.py`) explicitly rejected
mutating `enable_thinking` through the chat-template (variant V1).
Reason: a 28 % TPS regression and 6× CV blow-up due to CUDA-graph
dispatch breakage (Wave 6 closure).

Instead PN16 v2 offers four variants:

| Variant | Env flag | Behaviour | When to enable |
| --- | --- | --- | --- |
| **V3 client override** | (default ON) | respect client `chat_template_kwargs.enable_thinking` | always |
| **V5 soft cap** | `GENESIS_PN16_MAX_THINKING_TOKENS>0` | append `<think>` budget hint to the last user message | RAG / single-shot |
| **V7 hard `max_tokens` cap** | `GENESIS_PN16_CLASSIFIER_MAX_TOKENS>0` | classifier detects a short prompt → cap `max_tokens` | short-answer / IDE agents |
| **V8 tool-think budget** | `GENESIS_PN16_TOOL_THINK_BUDGET>0` | prepend a system message `reason ≤ N tokens before tool_call` | tool-call workflows |

### Contract for clients

**Client knows reasoning is not needed (short answer / system smoke).**
Pass `enable_thinking=false` explicitly:

```python
import openai

resp = openai.chat.completions.create(
    model="qwen3.6-27b",
    messages=[{"role": "user", "content": "Say OK"}],
    max_tokens=32,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(resp.choices[0].message.content)  # → "OK"
```

`curl` equivalent:

```bash
curl -fsS -X POST http://localhost:8101/v1/chat/completions \
    -H "Authorization: Bearer genesis-local" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.6-27b",
        "messages": [{"role":"user","content":"Say OK"}],
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": false}
    }'
```

This is zero-cost — no CUDA-graph regression, no MTP draft bias
(which would require thinking-enabled traces): PN16 V3 simply
respects the explicit client override.

**Client does NOT know the context (community OpenAI-compatible
SDKs).**

- Option A — server-side V7 via boot env:

  ```bash
  docker run ... -e GENESIS_PN16_CLASSIFIER_MAX_TOKENS=128 ...
  ```

  The classifier heuristically detects "trivial" prompts and caps
  `max_tokens` to 128. The chat template is NOT mutated → CUDA-graph
  hits remain intact. Trade-off: short-question answers are capped at
  128 tokens.

- Option B — server-side default through the chat template: bake
  `enable_thinking=false` into the preset; reasoning becomes opt-in
  per request:

  ```yaml
  # vllm/sndr_core/model_configs/builtin/<preset>.yaml
  genesis_env:
    GENESIS_PN16_CLASSIFIER_MAX_TOKENS: "128"
    # or vllm-side default flag (if supported):
    # VLLM_DEFAULT_ENABLE_THINKING: "false"
  ```

### CI smoke gate

`tools/openai_smoke.py` enforces the content-not-null contract:

```bash
make smoke-content                              # default: 127.0.0.1:8101
HOST=http://your-host:8101 make smoke-content
ENABLE_THINKING=false make smoke-content        # confirms V3 path
```

Exit codes:

- `0` — content received.
- `2` — content empty (assertion failure).
- `1` / `3` — HTTP / response-shape error.

### Related files

- `vllm/sndr_core/middleware/lazy_reasoner.py` — PN16 v2 implementation.
- `vllm/sndr_core/integrations/middleware/pn16_lazy_reasoner.py` — wiring.
- `tools/openai_smoke.py` — smoke test.
- `Makefile::smoke-content` — CI gate target.

## `Genesis → sndr_core` rename (v11.0.0)

Released **2026-05-08** (hard flip). This appendix documents the
rename, the structural changes that came with it, and what an
operator on a pre-v11 install needs to do on upgrade.

> This page exists because the public README cannot reference the
> retired `vllm/_genesis` namespace by name without tripping the
> `audit-docs-stale` gate. The retired tokens are the subject of
> this document; that is the documented exception for the
> stale-token allowlist.

### Why the rename happened

Up to v10.x the Python package lived at `vllm/_genesis/`. Three
problems forced a hard rename in v11.0.0:

1. `_genesis` looked like a private vLLM module — confusing for
   vLLM maintainers and operators alike. It is not part of vLLM
   and never was.
2. The old single-file `apply_all.py` (4 542 lines) didn't scale.
   Pull requests conflicted on every change, reviews stalled, and
   there were no family-level lines of responsibility.
3. Operator UX was thin — no CLI, no schema-driven configs, no
   audit gate, no per-patch observability.

### Before vs after

| v10.x (Genesis-named) | v11.0.0 (SNDR Core) | Effect |
| --- | --- | --- |
| `vllm/_genesis/` (235 files, flat) | `vllm/sndr_core/` (family-organised) | Clear hierarchy. |
| `apply_all.py` 4 542 lines | `apply/{orchestrator,verify,shadow,_per_patch_dispatch}.py` | PRs localised. |
| Flat `_genesis/patches/` | `integrations/<family>/<patch>.py` across 21 families | Review by area. |
| `import _genesis` side-effects on boot | Lazy `vllm.sndr_core.__init__` (torch-less importable) | CI / preflight without CUDA. |
| Boot summary scattered across uvicorn INFO | Structured boot summary + per-patch `elapsed_ms` + `rss_delta` | Observability. |
| Hardcoded paths (`/home/<user>/...`, LAN IPs) | Portable env vars (`$GENESIS_MODELS_DIR`, …) | Reproducibility. |
| `patch_genesis_unified.py` shim | Removed | Cleaner. |
| `vllm/sndr_core/wiring/patch_*.py` | `vllm/sndr_core/integrations/<family>/<patch>.py` | Family taxonomy. |
| `~/.genesis/` config dir | `~/.sndr/` (legacy alias honoured) | Canonical name. |
| No CLI | `sndr launch / doctor / verify / model-config / deps / patches` | Operator UX. |
| Single-format `model_configs/*.yaml` (V1 monolithic) | V1 still works + V2 layered (`model/`, `hardware/`, `profile/`, `presets/`) | Reusable building blocks. |

### What improved

- **Single CLI entry point** — `sndr launch <preset>` replaces 18
  ad-hoc `start_*.sh` / `bare_metal_*.sh` scripts.
- **Schema-driven model configs** with `audit_rules.py` checks
  (R-001 … R-018) and a `make evidence` release gate that runs them.
- **Anchor-manifest fast-path** — text patches record the anchor
  SHA; on upstream drift the patch self-skips with a clear
  `drift_marker detected` line instead of silently breaking.
- **Per-patch observability** — `GENESIS_OBSERVABILITY=1` prints
  `elapsed_ms` and `rss_delta` for every patch on boot.
- **43-gate `make evidence`** — release-tier audit covering legacy
  imports, hardcoded paths, security scan, community gate, lifecycle
  ratchet, doc sync.
- **21-family taxonomy** — patches grouped by subsystem
  (`attention.gdn`, `spec_decode`, `kv_cache`, …) instead of one bag.

### What was removed

- `vllm/_genesis/` — entire tree, 235 files.
- `patch_genesis_unified.py` — pre-v11 back-compat shim.
- `vllm/sndr_core/wiring/patch_*.py` — replaced by canonical
  `integrations/<family>/<patch>.py`.
- 11 retired patches whose upstream-merged equivalents are now in
  the vLLM nightly pin (P94, PN9, ...).
- `vllm/sndr_core/compat/fingerprints/` — stale 3-file
  cap-detection cache.
- `sponsor-site/` — separate project; didn't belong here.

### What stayed on purpose

- The name "Genesis" in documentation, banner, and wave numbers — it
  is the project's brand. Only the Python package was renamed.
- **V1 monolithic model configs** (`a5000-2x-35b-prod.yaml`, ...) —
  they still load and pass the same audit gate as V2. New configs
  SHOULD use V2, but V1 is not forced to migrate.
- **`~/.genesis/` legacy alias** for the config dir — existing
  operators don't have to move state.

### Migration steps for a pre-v11 install

```bash
# 1. Pull the v11+ release
cd /path/to/genesis-vllm-patches
git fetch && git checkout main

# 2. Re-install the plugin so it points at sndr_core.plugin:register
pip install -e .

# 3. Rewrite any custom script that imports the retired namespace
grep -rn 'vllm\._genesis\|vllm/_genesis' your-scripts/ |
    awk -F: '{print $1}' |
    sort -u |
    xargs sed -i 's/vllm\._genesis/vllm.sndr_core/g; s|vllm/_genesis|vllm/sndr_core|g'

# 4. Verify the import path resolves
python3 -c 'import vllm.sndr_core; print(vllm.sndr_core.__file__)'

# 5. Run smoke
sndr doctor
```

There is **no back-compat alias** for the retired namespace.
`import vllm._genesis` raises `ModuleNotFoundError`. Pre-v11 launch
scripts and tools must be updated before they run on v11.0.0 or
later.

### See also

- [`../README.md`](../README.md) — current state, install,
  benchmarks.
- [`../CHANGELOG.md`](../CHANGELOG.md) — per-release detail
  (v7.x → v11.0.0+wave10).
- [`INSTALL.md`](INSTALL.md) — installer reference and upgrade
  troubleshooting.
