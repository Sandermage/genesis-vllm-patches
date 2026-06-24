<p align="center">
  <img src="assets/logo.png" alt="Genesis vLLM Patches" width="780">
</p>

# Genesis vLLM Patches

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![vLLM pin](https://img.shields.io/badge/vllm-0.23.1rc1.dev301+g04c2a8dea-orange.svg)](https://github.com/vllm-project/vllm)
[![Patches](https://img.shields.io/badge/registry-317%20patches-green.svg)](docs/PATCHES.md)
[![SNDR Core](https://img.shields.io/badge/SNDR%20Core-v12.0.0-blue.svg)](CHANGELOG.md)
[![GPU](https://img.shields.io/badge/GPU-RTX%203090%20%7C%204090%20%7C%205090%20%7C%20A5000%20%7C%20H20%20%7C%20R6000-purple.svg)](docs/HARDWARE.md)

**Runtime patches for [vLLM](https://github.com/vllm-project/vllm) — Qwen3.6-class
inference on consumer NVIDIA Ampere / Ada / Blackwell with TurboQuant k8v4 KV
cache, MTP K=5 spec-decode, tool-calling, and 256K-class context. 316 patches
across 27 families. Apache 2.0.**

---

## What it is

A **drop-in runtime patcher** for vLLM. It pins to a specific vLLM nightly
commit and applies 317 small, surgical changes — text edits at known anchors,
class-rebind wrappers, and FastAPI middleware — that together turn an
out-of-the-box vLLM into a production-grade Qwen3.6 inference server on
*consumer* NVIDIA hardware (3090, 4090, 5090, A5000, A6000, …) where vLLM
upstream mostly targets datacenter SKUs.

It is **not** a fork of vLLM, a quantizer, a new inference engine, or a
training framework. Patches retire automatically when upstream merges the
underlying fix.

## Headline numbers (v12.0.0 current registry)

Reference rig: **2× RTX A5000 24 GB** (Ampere SM 8.6), driver 580.142,
CUDA 13.0.2, MTP K=5 + TurboQuant k8v4, TP=2.

| Model | Stock vLLM | Genesis (v12.0.0) | Δ |
| --- | ---: | ---: | ---: |
| Qwen3.6-35B-A3B-FP8 (single-conc, K=5) | ~157 t/s | **239.7 t/s** | +53 % |
| Qwen3.6-35B-A3B-FP8 (8-way multi-conc, K=3) | n/a | **~675 t/s agg** | 3.21× scaling |
| Qwen3.6-27B-int4-AutoRound (single-conc, K=5) | ~87 t/s | **127.4 t/s** | +46 % |
| Tool-call clean rate (35B / 27B) | 2–6 / 10 | **7/7 · 8/8** | qualitative |

256K context hardware-verified on both models. Full methodology, historical
comparisons, and per-rig reproduction recipes:
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

![Sustained TPS — Genesis vs stock](assets/charts/tps_genesis_vs_stock.png)

> **Pin bump 2026-06-24:** the vLLM pin is now `0.23.1rc1.dev301+g04c2a8dea`
> (image `nightly-04c2a8dea`); `dev148` is the previous / rollback pin. Smoke
> validation on the 2× A5000 rig: 35B 208 TPS + 31B 94.7 TPS boot + chat +
> tool-call. The dev301 anchor regen surfaced 5 `anchor_drift` entries:
> PN394 (#46047) + PN400 (#45656) retired on dev301; PN353A + PN382
> kept + re-anchored. The full per-model bench table below is the prior
> dev148 K=5 re-tune cycle (still the canonical sustained-bench evidence).

### Latest rig validation — 2026-06-19 (pin `0.23.1rc1.dev148+gb4c80ec0f`)

Full model-cycle re-test on the reference 2× A5000 rig after the MTP K=3→K=5 re-tune and the
dev148 pin promotion. Each model boots the Genesis apply pipeline, applies its patch set, and is
benchmarked / smoke-tested live (`tools/genesis_bench_suite.py`, single-stream warm sweep). The
35B / 27B single-stream rows are the K=5 re-tune numbers; Gemma stays K=3 (its separate drafter
is optimal at K=3). `dev101` is retained as the previous / rollback pin.

| Model | Quant / KV | Patches | Decode TPS | Tool-call | Status |
| --- | --- | ---: | ---: | :---: | --- |
| Qwen3.6-35B-A3B-FP8 | FP8 dense · TQ k8v4 · MTP K=5 | 95 | **239.7** (CV 4.9 %) | 7/7 | ✅ serving — +15.8 % vs K=3 |
| Qwen3.6-27B-int4-AutoRound | INT4 AutoRound · TQ k8v4 · MTP K=5 | 93 | **127.4** (CV 8.3 %) | 7/7 | ✅ serving — +8.2 % vs K=3 |
| Gemma-4-31B | INT4 · TQ k8v4 · MTP K=3 | 81 | — | — | ⚙️ boots + patches apply; serving needs MM-budget config (multimodal-bidirectional × spec-decode) |
| DiffusionGemma-26B-A4B-FP8 | FP8-dynamic · block-diffusion · TP=2 | 45 | coherent | — | ✅ **serving at TP=2** — `PN-FP8MOE-KPAD` (Marlin N=352) + `G4_26` (TP-vocab soft-embed); enforce-eager · max-num-seqs 2 · gpu-util 0.80 |

The 35B and 27B clear their historical peak band — the K=5 re-tune lifts single-stream decode
to 239.7 / 127.4 t/s (+15.8 % / +8.2 % vs K=3) within CV → the v12 platform carries **no decode
regression**. `PN-FP8MOE-KPAD` (backport of open vLLM
PR [#45703](https://github.com/vllm-project/vllm/pull/45703), model-agnostic Marlin-MoE
intermediate-pad) plus `G4_26` (backport of [#45774](https://github.com/vllm-project/vllm/pull/45774),
DiffusionGemma TP>1 vocab-sharded soft-embed all-gather) make
**DiffusionGemma the first block-diffusion FP8-MoE checkpoint to boot AND serve coherently
at TP=2 on consumer Ampere** without a kernel rebuild — validated 2026-06-17 (clears the
Marlin N=352 thread-tile crash, then the `probs @ embed_weight` `[131072,2816]` TP-vocab
shape mismatch; the coherent generation confirms the soft-embed all-gather yields correct
TP=2 output).

## Quick install

```bash
curl -sSL https://raw.githubusercontent.com/Sandermage/genesis-vllm-patches/main/install.sh | bash
```

The installer detects your OS / Python / GPU / vLLM presence, clones into
`~/.sndr/`, installs the plugin, writes a tailored launch script, and runs a
60-second smoke test. Five-minute walk-through and Day-1 acceptance steps:
[`docs/QUICKSTART.md`](docs/QUICKSTART.md).

To pick a different vLLM pin, workload, or non-interactive flag set:
[`docs/INSTALL.md`](docs/INSTALL.md).

## Documentation map

| If you want to... | Read |
| --- | --- |
| One-page operator manual (installer → launcher → configs → patches) | [`docs/USAGE.md`](docs/USAGE.md) |
| Install + first boot | [`docs/INSTALL.md`](docs/INSTALL.md) → [`docs/QUICKSTART.md`](docs/QUICKSTART.md) |
| Browse `sndr` commands | [`docs/CLI_REFERENCE.md`](docs/CLI_REFERENCE.md) |
| Pick a model + hardware combo | [`docs/MODELS.md`](docs/MODELS.md) + [`docs/HARDWARE.md`](docs/HARDWARE.md) |
| Tune an env-var flag | [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) |
| Browse the patch catalogue + compatibility matrix | [`docs/PATCHES.md`](docs/PATCHES.md) |
| Diagnose an OOM, cliff, or boot failure | [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) |
| Roll a broken release back | [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) |
| See current bench numbers + reproduce | [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) |
| Author a patch or community plugin | [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) |
| Sponsorship / hardware loan / business invoicing | [`docs/SPONSORS.md`](docs/SPONSORS.md) |
| Disclose a security issue | [`SECURITY.md`](SECURITY.md) |

Full docs index: [`docs/README.md`](docs/README.md).

## Contributing

Bug reports, new patches with empirical evidence, new model recipes, and
cross-rig bench reports are all welcome. The full workflow (anchor
conventions, lifecycle ratchet, pin-bump playbook, PR template) is in
[`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md). Security disclosures go
through [`SECURITY.md`](SECURITY.md).

## Credits + license

Apache-2.0 (see [`LICENSE`](LICENSE)). Per-patch attribution and upstream
PR linkage in [`docs/CREDITS.md`](docs/CREDITS.md).

Author: Sandermage (Aleksandr Barzov), Odessa, Ukraine.
Sponsorship channels (voluntary, no obligations) and hardware-loan
contact: [`docs/SPONSORS.md`](docs/SPONSORS.md).
