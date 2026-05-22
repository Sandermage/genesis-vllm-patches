# Frequently Asked Questions

Common questions from people who just discovered Genesis and want to know
what they're getting into. If your question isn't here, please open an
issue — the FAQ is updated based on actual user reports.

For deeper topics see also [`GLOSSARY.md`](GLOSSARY.md) (term definitions),
[`HARDWARE.md`](HARDWARE.md) (sizing), and [`CONFIGS.md`](CONFIGS.md)
(per-model launch flags).

### Q: What is Genesis?

A runtime patch package that layers on top of stock vLLM. It applies
text-patches and Triton kernels at boot, plus a small middleware
layer, to optimize Qwen3.6 family models on consumer Ampere/Ada/
Hopper GPUs. Think of it as "vLLM tuning pack" — not a fork.

### Q: Is Genesis a fork of vLLM?

No. Genesis runs against an unmodified vLLM commit (pinned in
[`INSTALL.md`](INSTALL.md)). Patches are applied at runtime via the
dispatcher, anchored to known commits. You can run Genesis-on /
Genesis-off with the same vLLM binary by toggling environment
variables.

### Q: Which vLLM pin does Genesis target today?

`0.20.2rc1.dev371+gbf610c2f5` (current public release, v12.0.0,
2026-05-16). Each patch declares an `applies_to` range, so newer
vLLM commits cause patches to print `[SKIP — applies_to mismatch]`
rather than crashing. Bumping the pin is a deliberate release event
documented in [`RELEASE_POLICY.md`](RELEASE_POLICY.md).

### Q: How big is the patch registry today?

**226 entries** — 173 full-implementation + 17 marker-only +
4 retired + 7 partial + 2 placeholder. The current state is always
in [`PATCHES_AUTO.md`](PATCHES_AUTO.md) (auto-generated from
`vllm/sndr_core/dispatcher/registry.py`) and the narrative
explanations in [`PATCHES.md`](PATCHES.md).

### Q: How do I update vLLM without losing patches?

Bump the `applies_to` range on each affected patch and re-run the
anchor-verification suite. Most text-patches survive minor vLLM
updates because their anchors are short and stable; some need the
anchor adjusted by a few characters. `sndr doctor` tells you which
patches drifted before you boot.

### Q: How do I enable or disable an individual patch?

Each patch is gated by a single environment variable:
`GENESIS_ENABLE_P67=1` turns it on, unset or `=0` turns it off.
The boot log prints every patch and its decision. There is no
global "enable all" switch — by design.

### Q: Which patches are ON by default?

About 52 of 226 entries are marked `default_on=True` in the
registry — production-eligible Wave 10 backports + legacy
pre-dispatcher overlays that have been validated against the
v11 baselines. The full list is in [`PATCHES_AUTO.md`](PATCHES_AUTO.md);
the policy that decides which subset is allowed in production
presets is in [`PATCH_PLAN.md`](PATCHES.md).

A fresh Genesis install without any preset still respects the
per-patch `default_on` flag; production launch scripts under
`scripts/` flip additional opt-in patches on top.

### Q: I have one RTX 3090 — what should I run?

`Qwen3.6-27B-int4-AutoRound` from Lorbus, TP=1, context up to 32K,
no prefix-caching, no DFlash. Run `sndr model-config list` to find
a 1×24GB preset; the `a5000-1x-27b-int4-tested` config is the
closest validated starting point.

### Q: I have 2× 24 GiB cards — should I run 27B or 35B?

Depends on workload. 35B-A3B-FP8 (MoE) wins on prose quality and
broad-knowledge tasks; 27B-int4 wins on tool-call reliability, long
context (320K validated), and raw TPS. If you primarily run
agentic / tool-calling pipelines, start with 27B.

### Q: Is LoRA supported?

Not actively tested. vLLM's LoRA system should work because Genesis
patches are mostly orthogonal to LoRA loading, but no Genesis-
validated LoRA recipe exists. Try it and report results.

### Q: Does streaming work?

Yes. Patch P61b adds a streaming overlap guard that fixes a slice
bug in upstream Qwen3 streaming output. Enable
`GENESIS_ENABLE_P61B=1` together with the rest of the tool-call
family if you stream tool calls.

### Q: Does tool-call work reliably?

Yes — this is one of Genesis's main focus areas. The P59 / P61 /
P62 / P64 / P68 / P69 patch family fixes upstream regressions in
Qwen3 tool-call generation, especially around `<think>` tags,
multi-tool prompts, and streaming. Enable them together via the
`tool_call_safe` recipe in [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

### Q: How do I download the DFlash draft model?

It's a gated HuggingFace repo (`z-lab/Qwen3.6-27B-DFlash`,
`z-lab/Qwen3.6-35B-A3B-DFlash`). Accept the license on the model
page, then `huggingface-cli login` with a token that has read
access. Genesis will not auto-download it for you.

### Q: What if patches break my boot?

First, look at the boot log — Genesis prints `[APPLY]` / `[SKIP]` /
`[FAIL]` for every patch with a reason string. Disable the failing
patch by unsetting its `GENESIS_ENABLE_*` flag. If you can't find a
working subset, file an issue with the full boot log; include your
vLLM commit hash, GPU model, and the model checkpoint. The
[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) walks through the
recovery procedure step by step.

### Q: How do I capture a running container into a config?

`sndr model-config new <key> --from-running <container>` (audit
C2 closure 2026-05-16). The CLI runs `docker inspect`, reverse-
engineers a `ModelConfig` YAML from the live container's
Entrypoint+Cmd+Env+Mounts, and writes it to
`~/.sndr/configs/<key>.yaml`. Review the GPU id placeholder, the
image digest, and the symbolic-mount references before launching.

### Q: How do I add my own model to Genesis?

See [`CONFIGS.md`](CONFIGS.md) for the full guide. Short version:
copy a base config via `sndr model-config new <key> --template
<existing-key>`, update model path + env vars, test boot + tool-call
sanity, submit PR with bench numbers.

### Q: MoE backend — Triton or FlashInfer?

Workload-dependent. Triton MoE is more stable on consumer Ampere/
Ada and is the Genesis default for 35B-A3B-FP8. FlashInfer MoE is
faster on Hopper/Blackwell but has had stability regressions (see
vLLM #41306). On 2× A5000, Triton wins.

### Q: Why DFlash instead of MTP?

DFlash is trained for code-heavy workloads and produces longer
accepted runs on programming tasks. MTP is built into Qwen3.6
itself and works better for chat/prose. Run both, measure
acceptance rate on your real traffic, pick the winner. Genesis
empirical numbers: MTP K=3 wins prose by ~30%, DFlash N=5 wins
code by ~50%.

### Q: Where do I see which patches were applied at boot?

The Genesis dispatcher prints a structured log block right after
vLLM model load. Look for lines starting with
`[INFO:genesis.apply_all] [Genesis] applied: P67 ...` or
`[INFO:genesis.apply_all] [Genesis] skipped: P40 (reason)`. The
full registry status with `APPLY`/`SKIP`/`FAIL` summary is also
printed at boot end. `sndr patches plan <preset>` also previews
the decision without booting.

### Q: A patch shows "SKIP" — is something broken?

Almost always no. SKIP means either you didn't enable the patch
(default), or the dispatcher decided it doesn't apply to your
environment (wrong GPU, wrong KV dtype, wrong model family).
Patches are opt-in and self-gated. Only `[FAIL]` is a real
problem.

### Q: Can I run Genesis without Docker?

Yes. Genesis is a regular Python package and patches a vLLM
installed in the same environment. The Genesis reference
deployment uses Docker for repeatability, but bare-metal pip
works too. Just remember that text-patches mutate files inside
`site-packages/vllm/` — back them up or use a venv per Genesis
version. `sndr model-config render <key> --runtime bare_metal`
emits a venv launch script.

### Q: How do I run Genesis on Kubernetes or Proxmox?

`sndr service install <key>` now wires both backends end-to-end
(audit C3 closure 2026-05-16). For k8s it renders a
Deployment+Service+ConfigMap manifest under `~/.sndr/k8s/` and
applies it with `kubectl apply` when invoked with `--yes`. For
Proxmox it emits a runnable LXC bootstrap script under
`~/.sndr/proxmox/<key>.sh` that handles `pct create` + GPU
passthrough + venv bootstrap + launch.sh in one pass.

### Q: How much performance should I expect over stock vLLM?

On the Genesis reference rig (2× A5000, Qwen3.6-27B-int4) with the
recommended patch set: roughly 25-40% TPS uplift versus the same
vLLM commit with no patches, plus tool-call reliability
improvements that don't show up in TPS numbers. Your numbers will
differ by GPU and workload — always benchmark. The current
canonical numbers are in [`BENCHMARKS.md`](BENCHMARKS.md).
