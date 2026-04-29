# Genesis Benchmark Suite — Run Guide

A reproducible, environment-agnostic guide for measuring Genesis vLLM Patches performance on your own hardware and sharing the numbers with the community.

> Source of truth: `tools/genesis_bench_suite.py --help`. This guide describes the intended interface and the reasoning behind each metric. When the script's CLI flags drift from the table below, the script wins — open an issue or PR if you spot a mismatch.

---

## What this measures

The Genesis benchmark suite is a single-script harness that exercises a running vLLM server through its OpenAI-compatible HTTP endpoint and reports a small, well-defined set of metrics that have proven useful for diagnosing performance and quality regressions on Qwen3.6-A3B-class models with speculative decoding.

| Metric | What it captures | Why it matters |
|---|---|---|
| **Tool-call quality** | Pass/fail count over a 4-case fixture (think on/off × hermes-xml/oai-tools) | Catches regressions where spec-decode or a parser-side patch breaks `tool_calls` emission. A pass rate below 4/4 is the leading indicator of quality drift. |
| **Decode-only TPOT** | `(elapsed - TTFT) / (completion_tokens - 1) × 1000` ms | The fair primary speed metric for spec-decode A/B. Wall TPS conflates queue + scheduler + TTFT with decode and hides regressions. Methodology adopted from thc1006's `bench_v3_clean_ab.py`. |
| **Wall TPS** | `completion_tokens / elapsed` | The end-to-end metric you'd quote for chat UX. Useful for cross-config comparison only when prompts and `max_tokens` match. |
| **TTFT** | Wall-clock to first content token | Important for chat UX, mostly irrelevant for batch. Useful as a sanity check that prefill isn't pathological. |
| **Multi-turn TTFT** | TTFT over 5 sequential same-prefix requests | Detects prefix-cache health. If turns 2-5 don't drop sharply vs turn 1, prefix caching is broken. |
| **Stability stress** | 30 iterations at standard config; checks for crash, NaN, drift | Catches memory leaks, accumulating compile-cache pressure, scheduler stalls. |
| **Context window probe** | Progressive HTTP probe at 16K → 32K → 64K → ... up to your `--ctx all` cap | Identifies the largest context that loads + decodes without OOM. Where multi-card setups commonly regress. |
| **GPU profile** | nvidia-smi output snapshot, driver/CUDA versions, vLLM version | Required to reproduce. Captured automatically into the JSON. |

The harness is intentionally minimal — no NVML hooks, no Triton tracing, no tokenizer surgery. It runs everywhere `requests` runs.

---

## Output

Every run produces two files per arm:

- **`<arm_name>_<timestamp>.json`** — full machine-readable record. Per-prompt stats, every trial result, accept-rate scrape, configuration echo, GPU profile. This is what you paste into a GitHub Discussion.
- **`<arm_name>_<timestamp>.md`** — human-readable summary. Tables, verdict banner, common pitfalls flagged.

When you supply two JSONs to `--compare`, the suite emits a third file:

- **`compare_<A>_vs_<B>_<timestamp>.json`** — Welch's two-sample t-test on decode TPOT, delta in ms, delta in percent, two-sided p-value, plain-English verdict (`B FASTER by X% (p=…)` / `NOT SIGNIFICANT (p=…)` / `INCONCLUSIVE`).

All numbers are shareable. Nothing in the JSON identifies your IP, your local paths, or anything beyond hardware specs you chose to include.

---

## Prereqs

| What | Version | Notes |
|---|---|---|
| Python | 3.10+ | stdlib + `requests` (auto-pip-installed on first run if missing) |
| vLLM | running locally OR reachable via HTTP | The bench is an HTTP client. It does not import vLLM. |
| API key | matches the `--api-key` your server was started with | Default: `genesis-local` |
| `gh` CLI | optional | Only needed for `gh issue create` / `gh discussion create` if you want to script result sharing |
| Network | bench → vLLM | Same host, same VM, or LAN — pick what's natural |

There is **no** dependency on the Genesis patches themselves. You can run this harness against vanilla vLLM, against another patch tree, or against a hosted vLLM endpoint.

---

## How to run

Below are five common environments. Pick the one that matches your setup, follow it once, then jump to the [Bench command reference](#bench-command-reference).

### Scenario 1 — Bare metal Ubuntu/Debian (vLLM via pip)

You have CUDA + drivers installed on the host and `pip install vllm` works.

```bash
# 1. Clone Genesis patches (you only need the bench tool + a launch script)
git clone https://github.com/Sandermage/genesis-vllm-patches.git
cd genesis-vllm-patches

# 2. (Optional) install Python deps for the bench
python3 -m pip install --user requests

# 3. Start vLLM. The simplest path is to crib args from one of our launch scripts:
#       scripts/launch/start_35b_fp8_PROD.sh   (35B FP8 + spec-decode + TQ k8v4)
#       scripts/launch/start_27b_int4_no_TQ_long_256K.sh   (27B INT4 + long-context)
#    Replace --model with your local path, drop the docker bits, keep the vllm serve flags.
#
#    Example (minimal):
vllm serve \
  --model /path/to/Qwen3.6-35B-A3B-FP8 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --api-key genesis-local \
  --served-model-name qwen3.6-35b-a3b \
  --host 0.0.0.0 --port 8000 &

# 4. Wait for "Application startup complete" (usually 60-180s on first launch).
# 5. Run the bench.
python3 tools/genesis_bench_suite.py --quick --out my_first_run.json
```

Sample output paths:

- `./my_first_run.json` (full record)
- `./my_first_run.md` (summary)

### Scenario 2 — Docker (vllm/vllm-openai:nightly)

The supported reference path. All Genesis production runs use this image.

```bash
# 1. Clone the repo, pull the image
git clone https://github.com/Sandermage/genesis-vllm-patches.git
cd genesis-vllm-patches
docker pull vllm/vllm-openai:nightly

# 2. Pick a launch script and adjust paths to YOUR machine.
#    The reference scripts mount /nfs/genesis/models — change to wherever your weights live.
#    Edit:
#       -v /nfs/genesis/models:/models:ro                 → your model directory
#       -v /home/sander/.cache/huggingface:/root/...      → your HF cache (or remove)
#       -v /home/sander/genesis-vllm-patches/...:/...:ro  → your repo path
#
#    Then run it:
bash scripts/launch/start_35b_fp8_PROD.sh

# 3. Watch logs until you see "Application startup complete":
docker logs -f vllm-server-mtp-test

# 4. The bench runs OUTSIDE the container, on the host, hitting localhost:8000.
#    (The container exposes -p 8000:8000.)
python3 tools/genesis_bench_suite.py --quick --out arm_a.json
```

You don't need to install the Genesis Python plugin on the host to run the bench. The bench only speaks HTTP.

### Scenario 3 — Proxmox VM / Ubuntu VM with passthrough GPU

Same as Scenario 1 or 2, with extra hypervisor concerns:

- **GPU passthrough** must be working. `lspci | grep -i nvidia` inside the VM must show your card(s); `nvidia-smi` inside the VM must succeed. If either fails, fix passthrough first — IOMMU groups, PCIe ACS override, `vfio-pci` claim — none of which the bench can help with.
- **Networking**. The bench can run inside the same VM as vLLM (use `--host 127.0.0.1`), or from another machine on your LAN (use `--host 192.168.1.10` or whatever the VM's IP is). The latter is useful when you want to bench from a stable workstation while the VM reboots between arms.

Sample cross-VM bench command:

```bash
# From your workstation, against vLLM on a Proxmox VM:
python3 tools/genesis_bench_suite.py \
  --host 192.168.1.10 --port 8000 \
  --api-key genesis-local \
  --mode standard --ctx 8k \
  --out vm_run.json
```

Genesis production runs on Proxmox VM 100 (192.168.1.10) and the bench is invoked from the Mac workstation. This path is well-trodden.

### Scenario 4 — WSL2

WSL2 runs Linux inside Windows; it can access NVIDIA GPUs through the Windows driver. Caveats:

- **CUDA driver** must be installed on Windows (NOT inside WSL). `nvidia-smi` inside WSL must work — if it doesn't, install the Windows-side driver from NVIDIA's WSL/CUDA page and restart the WSL distro.
- **Native Ubuntu in WSL is preferred**. `pip install vllm` inside WSL is the simpler path. Running Docker-in-WSL adds a virtualization layer (Docker Desktop → WSL → CUDA passthrough) that has historically been flaky; you may see slightly lower TPS and sporadic CUDA init failures vs native Ubuntu.
- **Storage** matters a lot for cold-load latency. Mount your model directory on the WSL filesystem (`/home/<user>/models`), not on `/mnt/c/...`. Crossing the Windows ↔ WSL filesystem boundary slows model loading by 5-30×.

```bash
# Inside WSL:
nvidia-smi   # MUST succeed before anything else

# Then proceed as in Scenario 1 (native pip install) or Scenario 2 (Docker).
```

### Scenario 5 — RunPod / cloud GPU rental

Any cloud GPU rental that exposes a shell + GPU works. RunPod is the most common community choice.

```bash
# 1. Spin up a pod with a nightly-CUDA Ubuntu image and 1× or 2× of your target GPU.
# 2. SSH in (or use the web terminal).
# 3. Clone + start vLLM as in Scenario 1 or 2.
#
# 4. Port forwarding:
#    - If you only bench from inside the pod: --host 127.0.0.1, no forwarding needed.
#    - If you want to bench from your laptop: in RunPod's instance UI,
#      add an exposed TCP port pointing at 8000. RunPod gives you
#      a public hostname like xyz-8000.proxy.runpod.net.
#
# 5. From your laptop:
python3 tools/genesis_bench_suite.py \
  --host xyz-8000.proxy.runpod.net --port 443 \
  --scheme https \
  --api-key genesis-local \
  --quick --out runpod_a.json
```

Cloud caveats:

- **Cold-start TTFT** will be higher than bare metal because of network latency between you and the pod. The decode TPOT number is unaffected — that's measured on the server side.
- **Don't share JSON outputs publicly with secrets in them**. The bench doesn't capture them, but if you set `--api-key` to anything sensitive on the command line, your shell history may have it.

---

## Bench command reference

```bash
# Quick smoke test (~5 min; 5 runs × 5 prompts × 256 tokens; tool-call probe; no stress; no ctx probe)
python3 tools/genesis_bench_suite.py --quick

# Standard run (~15-30 min; 25 runs × 5 prompts × 1024 tokens; full quality battery; one ctx size)
python3 tools/genesis_bench_suite.py --mode standard --ctx 8k

# Full evaluation (~1-2 hours; includes long-ctx scan up to your card's ceiling, 30-iter stress)
python3 tools/genesis_bench_suite.py --mode full --ctx all

# Compare two arms (post-hoc; no server needed)
python3 tools/genesis_bench_suite.py --compare run_A.json run_B.json --compare-out delta.json

# Custom: 25 runs × the 5-prompt "standard" set × 1024 decode tokens, named arm
python3 tools/genesis_bench_suite.py \
  --runs 25 --prompts standard --max-tokens 1024 \
  --arm-name my_baseline --out my_baseline.json

# Tight CV check: 50 runs × short prompts × 256 tokens (highest-signal config for noise floor)
python3 tools/genesis_bench_suite.py --runs 50 --prompts short --max-tokens 256
```

> **Note on flag accuracy**
> The exact CLI options are defined in the script — when in doubt, run `python3 tools/genesis_bench_suite.py --help`. This guide describes the intended interface; the script is the canonical source. If the suite shipped with extra flags or renamed `--mode` / `--ctx` / `--prompts` since this guide was written, the `--help` output supersedes the table below.

| Flag | Purpose | Default |
|---|---|---|
| `--host` | vLLM HTTP host | `127.0.0.1` |
| `--port` | vLLM HTTP port | `8000` |
| `--scheme` | `http` / `https` | `http` |
| `--api-key` | server API key | `genesis-local` |
| `--model` | served-model-name; auto-discovered if omitted | (first `/v1/models` entry) |
| `--mode` | `quick` / `standard` / `full` preset | `standard` |
| `--ctx` | context probe target: `4k` / `8k` / `16k` / ... / `all` | (mode-dependent) |
| `--runs` | trials per prompt | `25` |
| `--prompts` | `standard` (5 long prompts) / `short` (5 short prompts) | `standard` |
| `--max-tokens` | per-request decode cap | `1024` |
| `--arm-name` | label that goes in the output filename and JSON | `A` |
| `--out` | output path; defaults to `<arm>_<timestamp>.json` | auto |
| `--quiet` | suppress per-trial stdout | off |
| `--compare A.json B.json` | post-hoc Welch's t-test; no server needed | — |
| `--compare-out` | where to write comparison JSON | stdout |

---

## Context window selection

Not everyone has the VRAM for 256K context. Pick the largest your card can stably hold; if the bench hits HTTP 500 or OOM at the chosen size, drop one row.

| Card class | VRAM | Recommended max ctx | Comment |
|---|---|---|---|
| RTX 3060 / 3070 | 8-12 GB | 4K | Single-GPU; INT4 27B only; very tight |
| RTX 3080 / 4070 Ti | 10-16 GB | 16K | INT4 27B comfortable; FP8 35B will not fit |
| RTX 3090 / A5000 | 24 GB | 64-128K | 27B INT4 long-ctx OR 35B FP8 short-ctx |
| RTX 4090 | 24 GB | 128K | Similar capacity to 3090, faster decode |
| 2× RTX 4090 / 2× A5000 | 48 GB | 256K | 27B INT4 v791b config; 35B FP8 stable |
| 2× RTX 5090 | 64 GB | 256K+ | Most workloads; large headroom |
| RTX PRO 6000 Blackwell | 96 GB | 256K-320K | Single-card 35B FP8 with full headroom |
| H100 80 GB | 80 GB | 320K+ | Reference for 35B FP8 long-context |
| 2× H100 / H200 | 160-192 GB | 1M+ | Frontier; bench has not been run there |

The bench suite's `--ctx` flag accepts:

- A specific size: `--ctx 8k`, `--ctx 32k`, `--ctx 128k`, `--ctx 256k`
- A scan: `--ctx all` walks 4K → 8K → 16K → 32K → 64K → 128K → 256K, stops at the first OOM/HTTP-500, and reports the largest one that passed

If you don't know your card's ceiling, use `--ctx all`. The scan is non-destructive — failures are reported, not crashed.

---

## Sharing your results

The community is actively interested in cross-rig data. Here's how to share well:

1. **Run with `--out my_results.json`** (or any name you'll recognize later).
2. **Open a GitHub Discussion** at https://github.com/Sandermage/genesis-vllm-patches/discussions
3. **Title format:** `[Bench] <model> on <GPU> — <wall_TPS> TPS`
   - Examples:
     - `[Bench] qwen3.6-35b-a3b-fp8 on 2× RTX A5000 — 162 TPS`
     - `[Bench] qwen3.6-27b-int4-AutoRound on 1× RTX 3090 — 88 TPS`
     - `[Bench] qwen3.6-35b-a3b-fp8 on 1× RTX PRO 6000 Blackwell — 240 TPS`
4. **Body should include:**
   - The Markdown summary section (or paste-the-tables)
   - **Hardware**: CPU model, RAM, motherboard, PSU, cooling
   - **GPU details**: driver version, CUDA version, link width (PCIe Gen3/4/5 x8/x16)
   - **Container/environment**: Docker / pip / WSL / VM
   - **Patches active**: which `GENESIS_ENABLE_PXX` envs you set (or "all defaults from `start_35b_fp8_PROD.sh`")
5. **Optionally attach the full JSON** as a code block or gist link. The JSON is small (a few hundred KB) and is the raw input for any future re-analysis.

The community is **especially** interested in:

- **New GPU classes** — rare consumer cards (W7900, RTX 6000 Ada), datacenter cards (L40S, B200), Apple Silicon (none of these have been benched on Genesis yet).
- **Multi-card configs** — TP=2, TP=4, TP=8. Most Genesis data is TP=2; TP≥4 is unexplored.
- **WSL / VM environments** — Genesis hasn't been validated under WSL. Numbers from WSL2 + RTX 5090 would be especially welcome.
- **Quality regression reports** — if your tool-call score is below 4/4, please open an issue (not just a discussion). Include the four failing-case logs.
- **OOM thresholds** at different context sizes — particularly useful for updating the [Context window selection](#context-window-selection) table above.

If you don't want to use GitHub Discussions, an issue with `[bench-share]` in the title also works.

---

## Interpreting the output

The Markdown summary highlights five numbers. Here's how to read them.

**`wall_TPS` vs `decode_TPOT_ms`**

- `wall_TPS` = `completion_tokens / total_elapsed_seconds` — the headline number for "how fast does my chat feel".
- `decode_TPOT_ms` = ms per emitted token, with TTFT subtracted out — the **fair primary metric for spec-decode A/B**.
- A patch can improve `wall_TPS` by 5% while regressing `decode_TPOT_ms` (e.g., by speeding up TTFT but slowing decode), or vice versa. When in doubt, decode TPOT is what should drive your decision.
- Genesis reports both. If a patch only moves wall TPS and not decode TPOT, the win is in scheduler/queueing, not in decode kernels.

**`CV` (coefficient of variation = std / mean)**

- `< 0.08` (8%): healthy. Your numbers are stable; A/B differences ≥ 5% are real.
- `0.08 - 0.12`: borderline. Re-run with `--runs 50` for tighter variance, or look for background CUDA work / thermal throttling.
- `> 0.12`: noisy. Something is competing for the GPU (another model, a desktop session, an `nvidia-smi -l 1` running, a slow disk during prefill). A/B comparisons at this CV are unreliable.

**`TTFT_ms` (time to first token)**

- Matters for chat UX (people notice >200ms before the first token), less so for batch.
- High TTFT with low decode TPOT = fast-once-going scheduler with prefill bottleneck. Common on long-context probes.
- Pathologically high TTFT (>5s on short prompts) = something is wrong. Check that the model finished loading, that prefix-cache isn't being repeatedly invalidated, and that you're not hitting the very first request after `vllm serve` boot (the first request always has cold-start cost).

**Tool-call pass rate `4/4`**

- `4/4`: clean. Your tool-call generation is healthy.
- `3/4` with the failing case using `enable_thinking=true` and `max_tokens=300`: very likely a `max_tokens` artifact, not a quality regression. Re-run that case at `max_tokens=1500`.
- `3/4` reproducible across runs: real regression. Open an issue with the failing case's raw response body.
- `< 3/4`: severe. Stop, do not deploy. Compare patch envs against the last known-good config.

**`accept_rate` (spec-decode acceptance)**

- Visible only if vLLM was started without `--disable-log-stats`.
- For MTP K=3 on Qwen3.6-A3B: ~0.65-0.78 is typical (per-token rule). Lower = bias against the draft heads, possibly a quality regression. Higher = good draft alignment.
- For ngram strict (P77 / `prompt_lookup_min=8`): ~0.95-1.0 on suffix-friendly workloads.
- A jump in `accept_rate` post-patch is usually GOOD; a drop is usually BAD. (The exception is P82 SGLang OR-clause acceptance, which is biased — it raises `accept_rate` artificially.)

---

## Common issues

| Symptom | Likely cause | Fix |
|---|---|---|
| `ConnectionRefusedError: [Errno 111]` | vLLM hasn't finished booting yet | Wait 2-4 min after `vllm serve`. Watch logs for `Application startup complete`. |
| `FAIL_HTTP_500` during long-ctx probe | OOM at the requested context size | Drop `--ctx` one tier; or lower `--gpu-memory-utilization` from 0.90 → 0.85; or reduce `--max-num-seqs`. |
| `tool-call 3/4` consistently | `max_tokens` too small for thinking-mode tool call | Raise `--max-tokens` from 300 to 1500. If still failing, real regression — file an issue. |
| `wall_TPS` varies wildly between trials | Background process competing for GPU | Stop other CUDA work (`nvidia-smi` to check). Disable any `nvidia-smi -l 1` watchers. Re-run with `--runs 50`. |
| Decode TPOT fine, TTFT 5-10× higher than expected | Cold start, or prefix-cache miss every turn | First request always has cold-start; ignore. If turns 2+ are also high, check `--enable-prefix-caching` and that the same prompt prefix is reaching the server unchanged. |
| `accept_rate` is `null` in the JSON | vLLM started with `--disable-log-stats` | Optional metric; safe to ignore. To capture it, drop the flag from your launch script. |
| `text_sha1` differs across trials at `temperature=0` | Spec-decode non-determinism, or seed not honored | Expected for spec-decode; the decode kernels are not bitwise deterministic. Use `--temperature 0 --seed 42` and accept that two SHAs is OK; ten different SHAs is not. |
| Bench finishes in 30 seconds with empty per-prompt stats | All trials failed; check the JSON for `error` fields | Usually a model-name or auth-key mismatch. Verify `curl -H "Authorization: Bearer genesis-local" http://host:port/v1/models` returns 200. |

If you hit something not in this table, the JSON contains every per-trial response (including error strings). Paste the relevant slice into an issue and we'll triage.

---

## Privacy note

The bench suite does **not** phone home, does **not** upload anything anywhere, does **not** collect telemetry. Everything stays on your machine in plain JSON / Markdown until you choose to share via a GitHub Discussion or issue.

The JSON does include:

- The hostname/IP you ran it against (default `127.0.0.1`)
- The model identifier returned by `/v1/models`
- A snapshot of `nvidia-smi` (if available)
- Driver/CUDA/vLLM versions

It does **not** include:

- Your API key (we strip it before serializing)
- Local file paths
- Anything from your shell environment beyond what you explicitly passed as flags

If you want to share a JSON file but feel uncertain about any field, scrub it manually — the file is small and human-readable.

---

## Reference: relationship to internal Genesis test harness

The community-facing `genesis_bench_suite.py` is a packaging of the same six tests our internal `tools/phase1_test_harness.sh` runs before promoting a Genesis patch tree:

1. `/v1/models` reachability
2. Tool-call quality probe (4 cases)
3. Decode-only TPOT bench (`bench_decode_tpot_clean_ab.py`, N=25)
4. Multi-turn TTFT probe (5 sequential same-prefix)
5. 30-iteration stability stress
6. Context window probe (256K / 280K / 300K / 317K — for our prod hardware; the suite scales these to your card)

If you want to inspect the underlying decode-only TPOT methodology, read [`tools/bench_decode_tpot_clean_ab.py`](../tools/bench_decode_tpot_clean_ab.py) — Welch's t-test, SHA1 content audit, per-prompt holds, streaming with usage. That methodology was originally adopted from [thc1006's `bench_v3_clean_ab.py`](https://github.com/thc1006/qwen3.6-vllm-2x3090/blob/master/scripts/bench_v3_clean_ab.py) and we credit them for it.

---

## Reference: launch scripts cited in this guide

The 4 PROD-ready configs ship in two flavors each — Docker (`start_*.sh`) and bare-metal (`bare_metal_*.sh`):

| Config | Docker | Bare metal |
|---|---|---|
| **35B-A3B-FP8** PROD (TQ k8v4 + MTP K=3 + PN8, 320K) | [`start_35b_fp8_PROD.sh`](../scripts/launch/start_35b_fp8_PROD.sh) | [`bare_metal_35b_fp8_PROD.sh`](../scripts/launch/bare_metal_35b_fp8_PROD.sh) |
| **27B-INT4-Lorbus** short-ctx (no TQ, fp8_e5m2, ≤8K, high TPS) | [`start_27b_int4_no_TQ_short.sh`](../scripts/launch/start_27b_int4_no_TQ_short.sh) | [`bare_metal_27b_int4_no_TQ_short.sh`](../scripts/launch/bare_metal_27b_int4_no_TQ_short.sh) |
| **27B-INT4-Lorbus** long-ctx 256K (no TQ, util 0.90) | [`start_27b_int4_no_TQ_long_256K.sh`](../scripts/launch/start_27b_int4_no_TQ_long_256K.sh) | [`bare_metal_27b_int4_no_TQ_long_256K.sh`](../scripts/launch/bare_metal_27b_int4_no_TQ_long_256K.sh) |
| **27B-INT4-Lorbus** + TurboQuant k8v4 (P98 required) | [`start_27b_int4_TQ_k8v4.sh`](../scripts/launch/start_27b_int4_TQ_k8v4.sh) | [`bare_metal_27b_int4_TQ_k8v4.sh`](../scripts/launch/bare_metal_27b_int4_TQ_k8v4.sh) |

The Docker variants bind-mount Genesis into a stock `vllm/vllm-openai:nightly` image (recommended for reproducibility).
The bare-metal variants assume vLLM is installed via `pip install vllm` and symlink Genesis `_genesis` into the existing vllm package on first run.

Internal building blocks (used by the bench suite, also runnable standalone):

- [`tools/genesis_bench_suite.py`](../tools/genesis_bench_suite.py) — **flagship community-grade entrypoint** (this guide)
- [`tools/bench_decode_tpot_clean_ab.py`](../tools/bench_decode_tpot_clean_ab.py) — decode-only TPOT building block (raw bench + Welch t-test compare)
- [`tools/progressive_context_probe.py`](../tools/progressive_context_probe.py) — context-window scan with PASS/FAIL per level
- [`tools/phase1_test_harness.sh`](../tools/phase1_test_harness.sh) — 6-test internal promotion gate

To match the exact public-benchmark numbers in [README.md § Headline numbers](../README.md#headline-numbers), use:

- 35B baseline: `start_35b_fp8_PROD.sh` or `bare_metal_35b_fp8_PROD.sh`
- 27B short-ctx: `start_27b_int4_no_TQ_short.sh`
- 27B long-ctx 256K: `start_27b_int4_no_TQ_long_256K.sh`
- 27B + TurboQuant: `start_27b_int4_TQ_k8v4.sh`

For correctness validation (apply matrix, smoke tests, pytest) — different purpose than performance bench — see [`validate_unit.sh`](../validate_unit.sh) (CPU 30 sec) / [`validate_integration.sh`](../validate_integration.sh) (GPU smoke + pytest) / [`scripts/run_validation_suite.sh`](../scripts/run_validation_suite.sh) (universal per-model).

---

## Final note

If your benchmark numbers don't match the Genesis published numbers — even on identical hardware — that **is** an interesting result, and we'd like to know about it. Common causes of cross-rig divergence we've seen so far:

- Driver version (570 → 580 was a 3× win on CUDA 13.0 paths)
- PCIe link width (Gen3 x8 vs Gen4 x16 for TP=2 NCCL)
- Background processes (Plex transcode, browser GPU acceleration, gnome-shell on the same GPU)
- Thermal throttling on under-cooled cards (sustained workload pulls cards from boost into base clocks)
- `expandable_segments:True` not being set in `PYTORCH_CUDA_ALLOC_CONF`

A short discussion thread with your numbers + hardware details + active patches almost always diagnoses the gap quickly.

Thank you for benchmarking. Cross-rig data is what makes this project work.
