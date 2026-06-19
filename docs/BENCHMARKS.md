# Genesis vLLM Patches — Benchmarks

Canonical PROD bench numbers for the Genesis reference rig +
operator-facing methodology (metrics, scenarios, sharing rules).
Reproducible from any host that runs the same vLLM pin against
the listed Genesis preset. See [`HARDWARE.md`](HARDWARE.md) for the
GPU envelope and [`MODELS.md`](MODELS.md) for the model lineup.

> **Current canonical stack (v12.0.0 current registry)**
>
> - Genesis `v12.0.0` — 317 PATCH_REGISTRY entries
>   (174 full + 17 marker-only + 4 retired + 7 partial + 2 placeholder).
> - vLLM `0.23.1rc1.dev148+gb4c80ec0f` (`dev101` = previous / rollback pin).
> - Reference rig: **2× RTX A5000 24 GB** (Ampere SM 8.6),
>   driver 580.142, CUDA 13.0.2.
> - Spec-decode: MTP K=5 on Qwen 35B/27B (re-tuned 2026-06-19; Gemma drafter
>   stays K=3 — optimal for its separate drafter). Probabilistic draft
>   rejection, vllm#40269.
> - Attention: TurboQuant k8v4 KV cache + FlashAttention 2, TP=2.

## Latest PROD numbers (v12.0.0 current registry; single-stream re-benched 2026-06-19 on dev148, K=5)

The single-stream rows are the MTP K=3→K=5 re-tune (warm sweep, pin
`0.23.1rc1.dev148+gb4c80ec0f`): 35B +15.8 % vs K=3 (207→239.7), 27B +8.2 %
vs K=3 (117.7→127.4). The multi-conc rows are NOT re-benched at K=5 — they
remain the 2026-05-23 K=3 aggregate measurements.

| Model | wall_TPS | decode_TPOT | CV% | Tool-call | Method |
| --- | ---: | ---: | ---: | :---: | --- |
| **Qwen3.6-35B-A3B-FP8** (prod-qwen3.6-35b-balanced, max_num_seqs=2, K=5) | **239.7** | 3.94 ms | 4.9% | 7/7 | `genesis_bench_suite.py` single-stream warm sweep (dev148) |
| **Qwen3.6-27B-int4-AutoRound** (prod-qwen3.6-27b-tq-k8v4, K=5) | **127.4** | 7.54 ms | 8.3% | 7/7 | same single-stream warm sweep (dev148) |
| **Qwen3.6-35B-A3B-FP8** (prod-qwen3.6-35b-multiconc, max_num_seqs=8, K=3) | **672.27** agg | 33.81 ms | 1.2% | — | `tools/multi_conc_bench.py --conc 8 --rounds 5 --max-tok 1024` (non-stream aggregate, 2026-05-23) |
| **Qwen3.6-27B-int4-AutoRound** (prod-qwen3.6-27b-tq-multiconc, max_num_seqs=8, K=3) | **471.10** agg | 51.70 ms | 1.0% | — | same multi-conc harness (non-stream aggregate, 2026-05-23) |

> Multi-conc rows measure non-stream aggregate throughput across 8 concurrent
> requests (`tools/multi_conc_bench.py`). Decode TPOT is the streaming
> per-request median (33.81 ms 35B vs 51.70 ms 27B reflects per-token
> latency under contention, not per-stream throughput).
> Single-stream rows use `genesis_bench_suite.py --quick` and reflect
> the latency a single interactive client sees. Choose by use case:
> 35B-MoE wins aggregate (+42% vs 27B dense) and per-token TPOT
> (-35%); 27B wins TTFT (181 vs 254 ms) and single-stream sustained
> throughput per its preset.

### Wave 10 Δ vs Wave 8 baseline (27B PROD, same harness)

| Metric | Wave 8 (2026-05-11) | Wave 10 (2026-05-15) | Δ |
| --- | ---: | ---: | ---: |
| wall_TPS | 130.76 | **132.93** | **+1.66%** |
| decode_TPOT_ms | 7.31 | 7.27 | -0.5% (faster) |
| CV% | 5.29 | 3.5 | tightened |

**Wave 10 components on top of Wave 9**: PN116 / PN118 / PN119
TurboQuant backports, PN125–PN130 warmup-orchestrator family,
PN132 / PN133 correctness backports, PN204 GDN dual-stream
consolidation (off in single-conc; on in `prod-qwen3.6-35b-multiconc`),
PN96b Marlin MoE persistent workspace (renamed after the silent
dict-key collision with kv_cache/PN96 was fixed), PN95 tier-aware
cache wiring closure.

The 27B improvement vs Wave 8 is small but outside CV. Most of
it comes from PN122 (the renamed `SPRINT26_CG_DISPATCH_TRACE`)
no longer crashing on import: each failed `@register_patch` hook
added ~30–50 ms boot overhead and one log/exception event that
introduced jitter on the worker decode path.

## What is currently on for `prod-qwen3.6-35b-balanced`

Per Genesis structured boot summary printed once at boot end:

```text
══════════════════════════════════════════════════════════════════════
Genesis vLLM Patcher — boot summary
══════════════════════════════════════════════════════════════════════
  Genesis:  v12.0.0
  vLLM:     0.23.1rc1.dev148+gb4c80ec0f
  GPU:      2× NVIDIA RTX A5000 (sm_86)
──────────────────────────────────────────────────────────────────────
  Patches:  317 total → ~80 APPLY | ~148 SKIP
  By family (APPLY only):
    • attention.gdn          ~5
    • attention.turboquant   ~12 (incl. PN116/118/119)
    • compile_safety         ~4
    • kernels                ~3
    • kv_cache               ~6 (incl. PN95)
    • moe                    ~3 (incl. PN96b)
    • observability          ~4 (incl. PN122)
    • reasoning              ~5
    • scheduler              ~3
    • serving                ~3
    • spec_decode            ~9 (incl. PN90 probabilistic)
    • streaming              ~3
    • tool_parsing           ~6
    • worker                 ~10 (incl. PN35, warmup PN125–130)
══════════════════════════════════════════════════════════════════════
```

The complete machine-readable per-patch state lands in the proof
artefacts under `evidence/patch_proof/<id>__*.json` after a
`sndr patches release-check` run.

## Reproduction recipe

The canonical bench harness is `tools/genesis_bench_suite.py` (shim
under `tools/`, canonical source under `vllm/sndr_core/tools/`). It
reads a `ModelConfig` preset and runs five stages: short-gen TTFT,
sustained long-gen TPS, tool-call clean, multi-turn stability,
long-context probe (skippable).

```bash
# 1. Install + boot
sndr install                # or `bash install.sh --workload tool_agent -y`
sndr launch a5000-2x-35b-prod    # V1 key, or use V2 alias `prod-qwen3.6-35b-balanced`

# 2. Wait for the structured boot summary in docker logs

# 3. Run the canonical bench
python3 tools/genesis_bench_suite.py \
    --quick --ctx 8k \
    --model qwen3.6-35b-a3b \
    --out ~/.sndr/bench-results/35b_wave10.json

# 4. Verify against the preset's reference_metrics
sndr model-config verify prod-qwen3.6-35b-balanced
```

Multi-conc runs flip `max_num_seqs=8` and use the
`prod-qwen3.6-35b-multiconc` V2 alias (qwen3.6-35b-multiconc.yaml profile);
expect aggregate TPS ~675 at the cost of higher TTFT.

## Historical reference

Older points are kept for regression-detection. Wave 8 (dev93)
numbers remained the operator-facing baseline until Wave 10 confirmed
the small uplift above; Wave 7 / v7.72 (dev9) is pre-v11-rename and
is not directly comparable because the patch registry was much
smaller (134 entries vs 317 today).

### Wave 7 / v7.72 dev9 snapshot (2026-05-05, pre-v11 rename)

| Model | Sustained TPS | CV% | Cold-warm latency | Tool-call clean | Multi-turn 10/10 | VRAM steady-state |
| --- | --- | --- | --- | --- | --- | --- |
| **Qwen3.6-35B-A3B-FP8** (MoE) | 192.9 tok/s | 4.19% | 2.34s | 10/10 | 10/10 survived (avg 1.1s) | 22687 + 21998 = 44685 MiB |
| **Qwen3.6-27B-int4-AutoRound** | 95.6 tok/s | 4.04% | 4.76s | 10/10 | 10/10 survived (avg 2.3s) | 22753 + 22064 = 44817 MiB |

### Wave 8 dev93 snapshot (2026-05-11)

| Model | wall_TPS | decode_TPOT | CV% | Tool-call |
| --- | ---: | ---: | ---: | :---: |
| Qwen3.6-27B-int4-AutoRound | 132.28 | 7.31 ms | 5.29% | 8/8 |
| Qwen3.6-35B-A3B-FP8 (Sprint 1) | 241.35 | 3.85 ms | 3.02% | 7/7 |

The 35B Sprint-1 number (241 TPS) was a single-prompt cherry-pick
captured before the methodology shift to sustained sweeps — the
**239.7 TPS** single-stream K=5 figure (MTP K=3→K=5 re-tune, dev148) in
the v12.0.0 PROD-numbers table above is the correct apples-to-apples
comparison (K=3 single-stream was 207).

## Cross-rig validators (call for replication)

Genesis numbers above are 2× RTX A5000 single-rig. Cross-rig
validation requested from operators on:

- **noonghunna** (1× RTX 3090, 4× RTX 3090 club-3090) — long-time
  Cliff 2 + tool-call collaborator.
- **apnar** (1× RTX 5090, sm_120 consumer Blackwell) — first
  sm_120 production rig (club-3090#51 thread).
- **tfriedel** (4× RTX 3090) — vendors Genesis as submodule.
- **Quentin Machu** — P64 sub-patch E author + bug-class triage.
- **MidasMining**, **JartX**, **jhsmith409**, **webcodes-cz** —
  hardware variety (5090, H20, R6000 Blackwell, 8× A4000).

If you run Genesis on hardware not listed, drop a bench JSON into
`tests/integration/baselines/` (PR welcome).

## Methodology — what each metric captures

The bench harness is a single-script HTTP client
(`tools/genesis_bench_suite.py`, canonical source under
`vllm/sndr_core/tools/`) that exercises a running vLLM server through
its OpenAI-compatible endpoint. No NVML hooks, no Triton tracing, no
tokenizer surgery — it runs everywhere `requests` runs.

| Metric | What it captures | Why it matters |
| --- | --- | --- |
| **Tool-call quality** | Pass / fail count over a 4-case fixture (think on/off × Hermes-XML / OAI tools) | Catches regressions where spec-decode or a parser-side patch breaks `tool_calls` emission. Pass rate below 4/4 is the leading indicator of quality drift. |
| **Decode-only TPOT** | `(elapsed − TTFT) / (completion_tokens − 1) × 1000` ms | The fair primary speed metric for spec-decode A/B. Wall TPS conflates queue + scheduler + TTFT with decode and hides regressions. Methodology adopted from thc1006's `bench_v3_clean_ab.py`. |
| **Wall TPS** | `completion_tokens / elapsed` | End-to-end metric you'd quote for chat UX. Useful for cross-config comparison only when prompts and `max_tokens` match. |
| **TTFT** | Wall-clock to first content token | Important for chat UX, mostly irrelevant for batch. Sanity check that prefill isn't pathological. |
| **Multi-turn TTFT** | TTFT over 5 sequential same-prefix requests | Detects prefix-cache health. If turns 2–5 don't drop sharply vs turn 1, prefix caching is broken. |
| **Stability stress** | 30 iterations at standard config; checks for crash, NaN, drift | Catches memory leaks, accumulating compile-cache pressure, scheduler stalls. |
| **Context window probe** | Progressive HTTP probe at 16K → 32K → 64K → ... up to your `--ctx all` cap | Identifies the largest context that loads + decodes without OOM. Where multi-card setups commonly regress. |
| **GPU profile** | `nvidia-smi` snapshot, driver / CUDA / vLLM versions | Required to reproduce. Captured automatically into the JSON. |

### Output files

Every run produces two files per arm:

- **`<arm_name>_<timestamp>.json`** — full machine-readable record:
  per-prompt stats, every trial result, accept-rate scrape,
  configuration echo, GPU profile. Paste this into a GitHub
  Discussion.
- **`<arm_name>_<timestamp>.md`** — human-readable summary with
  tables, verdict banner, common pitfalls flagged.

When you supply two JSONs to `--compare`, the suite emits a third
file `compare_<A>_vs_<B>_<timestamp>.json` with Welch's two-sample
t-test on decode TPOT, delta in ms, delta in percent, two-sided
p-value, and a plain-English verdict
(`B FASTER by X% (p=…)` / `NOT SIGNIFICANT (p=…)` /
`INCONCLUSIVE`).

All numbers are shareable. Nothing in the JSON identifies your IP,
your local paths, or anything beyond hardware specs you chose to
include.

## Prerequisites

| What | Version | Notes |
| --- | --- | --- |
| Python | 3.10+ | stdlib + `requests` (auto-pip-installed on first run if missing) |
| vLLM | running locally OR reachable via HTTP | The bench is an HTTP client. It does not import vLLM. |
| API key | matches `--api-key` your server was started with | Default: `genesis-local` |
| `gh` CLI | optional | Only needed for `gh issue create` / `gh discussion create` if you want to script result sharing |
| Network | bench → vLLM | Same host, same VM, or LAN — pick what's natural |

No dependency on the Genesis patches themselves. You can run this
harness against vanilla vLLM, against another patch tree, or against
a hosted vLLM endpoint.

## Bench command reference

```bash
# Quick smoke test (~5 min: 5 runs × 5 prompts × 256 tokens; tool-call probe; no stress; no ctx probe)
python3 tools/genesis_bench_suite.py --quick

# Standard run (~15-30 min: 25 runs × 5 prompts × 1024 tokens; full quality battery; one ctx size)
python3 tools/genesis_bench_suite.py --mode standard --ctx 8k

# Full evaluation (~1-2 hours: long-ctx scan up to your card's ceiling, 30-iter stress)
python3 tools/genesis_bench_suite.py --mode full --ctx all

# Compare two arms (post-hoc; no server needed)
python3 tools/genesis_bench_suite.py --compare run_A.json run_B.json --compare-out delta.json

# Custom: 25 runs × 5 standard prompts × 1024 decode tokens, named arm
python3 tools/genesis_bench_suite.py \
    --runs 25 --prompts standard --max-tokens 1024 \
    --arm-name my_baseline --out my_baseline.json

# Tight CV check: 50 runs × short prompts × 256 tokens (highest-signal config for noise floor)
python3 tools/genesis_bench_suite.py --runs 50 --prompts short --max-tokens 256
```

> The exact CLI options are defined in the script — when in doubt,
> run `python3 tools/genesis_bench_suite.py --help`. This guide
> describes the intended interface; the script is the canonical
> source.

| Flag | Purpose | Default |
| --- | --- | --- |
| `--host` | vLLM HTTP host | `127.0.0.1` |
| `--port` | vLLM HTTP port | `8000` |
| `--scheme` | `http` / `https` | `http` |
| `--api-key` | server API key | `genesis-local` |
| `--model` | served-model-name (auto-discovered if omitted) | (first `/v1/models` entry) |
| `--mode` | `quick` / `standard` / `full` preset | `standard` |
| `--ctx` | context probe target: `4k` / `8k` / `16k` / ... / `all` | (mode-dependent) |
| `--runs` | trials per prompt | `25` |
| `--prompts` | `standard` (5 long prompts) / `short` (5 short prompts) | `standard` |
| `--max-tokens` | per-request decode cap | `1024` |
| `--arm-name` | label that goes in the output filename and JSON | `A` |
| `--out` | output path; defaults to `<arm>_<timestamp>.json` | auto |
| `--quiet` | suppress per-trial stdout | off |
| `--compare A.json B.json` | post-hoc Welch's t-test (no server needed) | — |
| `--compare-out` | where to write comparison JSON | stdout |

## Run scenarios — five common environments

### Scenario 1 — Bare metal Ubuntu / Debian (vLLM via pip)

CUDA + drivers installed on the host; `pip install vllm` works.

```bash
git clone https://github.com/Sandermage/genesis-vllm-patches.git
cd genesis-vllm-patches
python3 -m pip install --user requests        # bench dep

# Boot vLLM via the unified launcher (v12.0.0 canonical):
sndr launch prod-qwen3.6-35b-balanced              # 35B FP8 + MTP K=5 + TQ k8v4, latency
# OR inspect resolved args:
sndr launch prod-qwen3.6-35b-balanced --dry-run

# Wait for "Application startup complete", then bench:
python3 tools/genesis_bench_suite.py --quick --out my_first_run.json
# OR via the unified CLI:
sndr bench --quick --out my_first_run.json
```

The two forms are equivalent — the unified CLI is a thin shim over
`tools/genesis_bench_suite.py` with argv forwarded verbatim.

### Scenario 2 — Docker (`vllm/vllm-openai:nightly`)

The supported reference path. All Genesis PROD runs use this image.

```bash
git clone https://github.com/Sandermage/genesis-vllm-patches.git
cd genesis-vllm-patches
docker pull vllm/vllm-openai:nightly             # current Genesis pin (0.23.1rc1.dev148+gb4c80ec0f)

sndr launch prod-qwen3.6-35b-balanced                            # docker emission
docker logs -f vllm-server                      # wait for startup

# Bench runs OUTSIDE the container, hitting localhost:8000
python3 tools/genesis_bench_suite.py --quick --out arm_a.json
```

You don't need to install the Genesis Python plugin on the host to
run the bench — the bench only speaks HTTP.

### Scenario 3 — Proxmox VM / Ubuntu VM with passthrough GPU

Same as Scenario 1 or 2, with extra hypervisor concerns:

- **GPU passthrough** must be working. `lspci | grep -i nvidia`
  inside the VM must show your card(s); `nvidia-smi` inside the VM
  must succeed. If either fails, fix passthrough first (IOMMU
  groups, PCIe ACS override, `vfio-pci` claim) — the bench can't
  help with that.
- **Networking.** The bench can run inside the same VM as vLLM
  (use `--host 127.0.0.1`), or from another machine on your LAN
  (use `--host gpu-vm.local` or the VM's IP / DNS name). The latter
  is useful when you want to bench from a stable workstation while
  the VM reboots between arms.

```bash
# From your workstation, against vLLM on a Proxmox VM:
python3 tools/genesis_bench_suite.py \
    --host gpu-vm.local --port 8000 \
    --api-key genesis-local \
    --mode standard --ctx 8k \
    --out vm_run.json
```

For LXC-on-Proxmox specifics see
[`PATCH_DESIGNS.md` § PN95](PATCH_DESIGNS.md) and the lxc_proxmox
renderer in [`QUICKSTART.md`](QUICKSTART.md).

### Scenario 4 — WSL2

WSL2 runs Linux inside Windows; it accesses NVIDIA GPUs through the
Windows driver. Caveats:

- **CUDA driver** must be installed on Windows (NOT inside WSL).
  `nvidia-smi` inside WSL must work — if it doesn't, install the
  Windows-side driver from NVIDIA's WSL/CUDA page and restart the
  WSL distro.
- **Native Ubuntu in WSL is preferred.** `pip install vllm` inside
  WSL is the simpler path. Running Docker-in-WSL adds a
  virtualisation layer (Docker Desktop → WSL → CUDA passthrough)
  that has historically been flaky — slightly lower TPS, sporadic
  CUDA init failures vs native Ubuntu.
- **Storage** matters a lot for cold-load latency. Mount your model
  directory on the WSL filesystem (`/home/<user>/models`), not on
  `/mnt/c/...`. Crossing the Windows ↔ WSL filesystem boundary slows
  model loading by 5–30×.

```bash
nvidia-smi    # MUST succeed before anything else
```

Then proceed as in Scenario 1 (native pip install) or Scenario 2
(Docker).

### Scenario 5 — RunPod / cloud GPU rental

Any cloud GPU rental that exposes a shell + GPU works. RunPod is the
most common community choice.

```bash
# 1. Spin up a pod with a nightly-CUDA Ubuntu image and 1× / 2× of your target GPU
# 2. SSH in (or use the web terminal)
# 3. Clone + start vLLM as in Scenario 1 or 2
# 4. Port forwarding:
#    - Bench inside pod: --host 127.0.0.1, no forwarding needed
#    - Bench from laptop: in RunPod's instance UI, expose TCP port 8000.
#      RunPod gives you xyz-8000.proxy.runpod.net.
# 5. From your laptop:
python3 tools/genesis_bench_suite.py \
    --host xyz-8000.proxy.runpod.net --port 443 \
    --scheme https \
    --api-key genesis-local \
    --quick --out runpod_a.json
```

Caveats:

- Cold-start TTFT is higher than bare metal because of network
  latency. The decode TPOT number is unaffected (measured server-
  side).
- Don't share JSON outputs publicly with secrets in them. The bench
  doesn't capture them, but if you set `--api-key` to anything
  sensitive on the command line, your shell history may have it.

## Context window selection

Not every rig has VRAM for 256K. Pick the largest your card can
stably hold; if the bench hits HTTP 500 or OOM at the chosen size,
drop one row.

| Card class | VRAM | Recommended max ctx | Comment |
| --- | --- | --- | --- |
| RTX 3060 / 3070 | 8–12 GB | 4K | Single-GPU; INT4 27B only; very tight. |
| RTX 3080 / 4070 Ti | 10–16 GB | 16K | INT4 27B comfortable; FP8 35B won't fit. |
| RTX 3090 / A5000 | 24 GB | 64–128K | 27B INT4 long-ctx OR 35B FP8 short-ctx. |
| RTX 4090 | 24 GB | 128K | Similar capacity to 3090, faster decode. |
| 2× RTX 4090 / 2× A5000 | 48 GB | 256K | 27B INT4 long-ctx config; 35B FP8 stable. |
| 2× RTX 5090 | 64 GB | 256K+ | Most workloads; large headroom. |
| RTX PRO 6000 Blackwell | 96 GB | 256K–320K | Single-card 35B FP8 with full headroom. |
| H100 80 GB | 80 GB | 320K+ | Reference for 35B FP8 long-context. |
| 2× H100 / H200 | 160–192 GB | 1M+ | Frontier; bench has not been run there. |

The bench suite's `--ctx` flag accepts:

- A specific size: `--ctx 8k`, `--ctx 32k`, `--ctx 128k`, `--ctx 256k`.
- A scan: `--ctx all` walks 4K → 8K → 16K → 32K → 64K → 128K → 256K,
  stops at the first OOM / HTTP-500, and reports the largest one
  that passed.

If you don't know your card's ceiling, use `--ctx all`. The scan is
non-destructive — failures are reported, not crashed.

## Interpreting the output

The Markdown summary highlights five numbers:

**`wall_TPS` vs `decode_TPOT_ms`.**
`wall_TPS = completion_tokens / total_elapsed_seconds` is the
headline number for "how fast does my chat feel".
`decode_TPOT_ms` is ms per emitted token with TTFT subtracted —
the **fair primary metric for spec-decode A/B**. A patch can
improve `wall_TPS` by 5 % while regressing `decode_TPOT_ms` (e.g.
speeding up TTFT but slowing decode), or vice versa. When in doubt,
decode TPOT should drive your decision. If a patch only moves wall
TPS and not decode TPOT, the win is in scheduler / queueing, not
decode kernels.

**`CV` (coefficient of variation = std / mean).**

- `< 0.08`: healthy. A/B differences ≥ 5 % are real.
- `0.08 – 0.12`: borderline. Re-run with `--runs 50` for tighter
  variance, or look for background CUDA work / thermal throttling.
- `> 0.12`: noisy. Something is competing for the GPU
  (another model, a desktop session, `nvidia-smi -l 1`, a slow disk
  during prefill). A/B at this CV is unreliable.

**`TTFT_ms`.** Matters for chat UX (>200 ms is noticeable), less so
for batch. High TTFT with low decode TPOT = fast-once-going
scheduler with prefill bottleneck (common on long-context probes).
Pathologically high TTFT (>5 s on short prompts) = something is
wrong: model not finished loading, prefix-cache repeatedly
invalidated, or hitting the very first request after `vllm serve`
boot (always cold-start).

**Tool-call pass rate.**

- `4/4`: clean.
- `3/4` with failing case using `enable_thinking=true` and
  `max_tokens=300`: very likely a `max_tokens` artefact, not a
  quality regression. Re-run that case at `max_tokens=1500`.
- `3/4` reproducible across runs: real regression. Open an issue
  with the failing case's raw response body.
- `< 3/4`: severe. Stop, do not deploy. Compare patch envs against
  the last known-good config.

**`accept_rate` (spec-decode acceptance).** Visible only if vLLM
was started without `--disable-log-stats`. For MTP K=5 on
Qwen3.6-A3B: ~0.65–0.78 is typical (per-token rule). Lower = bias
against the draft heads, possibly a quality regression. Higher =
good draft alignment. For ngram strict (P77 /
`prompt_lookup_min=8`): ~0.95–1.0 on suffix-friendly workloads.
A jump in `accept_rate` post-patch is usually GOOD; a drop is
usually BAD. The exception is P82 SGLang OR-clause acceptance,
which raises `accept_rate` artificially.

## Common issues

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ConnectionRefusedError: [Errno 111]` | vLLM hasn't finished booting | Wait 2–4 min after `vllm serve`. Watch for `Application startup complete`. |
| `FAIL_HTTP_500` during long-ctx probe | OOM at the requested context | Drop `--ctx` one tier; lower `--gpu-memory-utilization` from 0.90 → 0.85; reduce `--max-num-seqs`. |
| `tool-call 3/4` consistently | `max_tokens` too small for thinking-mode tool call | Raise `--max-tokens` 300 → 1500. If still failing, real regression — file an issue. |
| `wall_TPS` varies wildly between trials | Background process competing for GPU | Stop other CUDA work. Disable `nvidia-smi -l 1` watchers. Re-run with `--runs 50`. |
| Decode TPOT fine, TTFT 5–10× higher than expected | Cold start, or prefix-cache miss every turn | First request always cold-starts; ignore. If turns 2+ are also high, check `--enable-prefix-caching` and that the same prompt prefix is reaching the server unchanged. |
| `accept_rate` is `null` in the JSON | vLLM started with `--disable-log-stats` | Drop the flag from your launch script. |
| `text_sha1` differs across trials at `temperature=0` | Spec-decode non-determinism | Expected; the decode kernels are not bitwise deterministic. Two SHAs is OK; ten different SHAs is not. |
| Bench finishes in 30 s with empty per-prompt stats | All trials failed; check JSON for `error` fields | Usually a model-name or auth-key mismatch. Verify `curl -H "Authorization: Bearer genesis-local" http://host:port/v1/models` returns 200. |

If you hit something not in this table, the JSON contains every
per-trial response (including error strings). Paste the relevant
slice into an issue and we'll triage. The full cliff catalogue is in
[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

## Sharing your results

The community is actively interested in cross-rig data.

1. **Run with `--out my_results.json`** (or any name you'll
   recognise later).
2. **Open a GitHub Discussion** at
   <https://github.com/Sandermage/genesis-vllm-patches/discussions>.
3. **Title format:** `[Bench] <model> on <GPU> — <wall_TPS> TPS`.
   Examples:
   - `[Bench] qwen3.6-35b-a3b-fp8 on 2× RTX A5000 — 239.7 TPS`
   - `[Bench] qwen3.6-27b-int4-AutoRound on 1× RTX 3090 — 88 TPS`
4. **Body should include:**
   - The Markdown summary (or paste the tables).
   - **Hardware**: CPU model, RAM, motherboard, PSU, cooling.
   - **GPU details**: driver version, CUDA version, link width
     (PCIe Gen3 / 4 / 5 x8 / x16).
   - **Container / environment**: Docker / pip / WSL / VM.
   - **Patches active**: which `GENESIS_ENABLE_PXX` envs you set
     (or "all defaults from `sndr launch prod-qwen3.6-35b-balanced`").
5. **Optionally attach the full JSON** as a code block or gist link.

We're especially interested in:

- **New GPU classes** — rare consumer cards (W7900, RTX 6000 Ada),
  datacenter cards (L40S, B200), Apple Silicon (none of these have
  been benched on Genesis yet).
- **Multi-card configs** — TP=2, TP=4, TP=8. Most data is TP=2;
  TP ≥ 4 is unexplored.
- **WSL / VM environments** — Genesis hasn't been validated under
  WSL; numbers from WSL2 + RTX 5090 would be especially welcome.
- **Quality regression reports** — if your tool-call score is below
  4/4, open an issue (not a discussion). Include the four failing-
  case logs.
- **OOM thresholds at different context sizes** — useful for
  updating the [Context window selection](#context-window-selection)
  table above.

## Privacy

The bench suite does **not** phone home, does **not** upload
anything, does **not** collect telemetry. Everything stays on your
machine in plain JSON / Markdown until you choose to share via a
GitHub Discussion or issue.

The JSON includes the hostname / IP you ran against (default
`127.0.0.1`), the model identifier returned by `/v1/models`, an
`nvidia-smi` snapshot (if available), and driver / CUDA / vLLM
versions. It does NOT include your API key (stripped before
serialising), local file paths, or anything from your shell
environment beyond what you explicitly passed as flags.

## Reference — building blocks

`tools/genesis_bench_suite.py` is the flagship community-grade
entrypoint. Under the hood it composes:

- `tools/multi_conc_bench.py` — multi-concurrency sweep
  (TTFT / TPOT / aggregate TPS at conc=1/2/4/8).
- `tools/bench_decode_tpot_clean_ab.py` — decode-only TPOT building
  block (raw bench + Welch t-test compare). Methodology originally
  adopted from
  [thc1006's `bench_v3_clean_ab.py`](https://github.com/thc1006/qwen3.6-vllm-2x3090/blob/master/scripts/bench_v3_clean_ab.py)
  — credit to them.
- `tools/progressive_context_probe.py` — context-window scan with
  PASS / FAIL per level.
- `tools/_retired/phase1_test_harness.sh` — RETIRED 2026-05-15
  (kept for archeology; superseded by `genesis_bench_suite.py` +
  `multi_conc_bench.py`).

The four PROD-ready configs launched through the unified CLI:

| Config | V2 preset |
| --- | --- |
| 35B-A3B-FP8 PROD (TQ k8v4 + MTP K=5, latency) | `sndr launch prod-qwen3.6-35b-balanced` |
| 35B-A3B-FP8 multi-conc (TQ k8v4 + max_num_seqs=8) | `sndr launch prod-qwen3.6-35b-multiconc` |
| 35B-A3B-FP8 + DFlash N=3 (latency) | `sndr launch prod-qwen3.6-35b-dflash` |
| 35B-A3B-FP8 + DFlash N=3 (multi-conc) | `sndr launch prod-qwen3.6-35b-dflash-multiconc` |
| 27B-INT4-AutoRound + TQ k8v4 (latency) | `sndr launch prod-qwen3.6-27b-tq-k8v4` |
| 27B-INT4-AutoRound + TQ k8v4 (multi-conc) | `sndr launch prod-qwen3.6-27b-tq-multiconc` |
| 27B-INT4-AutoRound + DFlash N=5 (latency) | `sndr launch prod-qwen3.6-27b-dflash` |
| 27B-INT4-AutoRound + DFlash N=5 (multi-conc) | `sndr launch prod-qwen3.6-27b-dflash-multiconc` |

V2 presets resolve to a (model, hardware, profile) triplet under
`vllm/sndr_core/model_configs/builtin/`. The legacy per-config
`start_*.sh` / `bare_metal_*.sh` scripts were moved to
`scripts/launch/_archive/` for archeology.

## If your numbers don't match the public ones

Common causes of cross-rig divergence:

- **Driver version** — 570 → 580 was a 3× win on CUDA 13.0 paths.
- **PCIe link width** — Gen3 x8 vs Gen4 x16 for TP=2 NCCL.
- **Background processes** — Plex transcode, browser GPU
  acceleration, gnome-shell on the same GPU.
- **Thermal throttling** on under-cooled cards — sustained workload
  pulls cards from boost into base clocks.
- **`PYTORCH_CUDA_ALLOC_CONF`** without `expandable_segments:True`.

A short Discussion thread with your numbers + hardware details +
active patches almost always diagnoses the gap quickly. Cross-rig
data is what makes this project work.
