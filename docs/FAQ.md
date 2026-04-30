# Frequently Asked Questions

Common questions from people who just discovered Genesis and want to know what they're getting into. If your question isn't here, please open an issue — the FAQ is updated based on actual user reports.

For deeper topics see also `docs/GLOSSARY.md` (term definitions), `docs/HARDWARE.md` (sizing), and `docs/CONFIGS.md` (per-model launch flags).

### Q: What is Genesis?
A: A runtime patch package that layers on top of stock vLLM. It applies text-patches and Triton kernels at boot, plus a small middleware layer, to optimize Qwen3.6 family models on consumer Ampere/Ada/Hopper GPUs. Think of it as "vLLM tuning pack" — not a fork.

### Q: Is Genesis a fork of vLLM?
A: No. Genesis runs against an unmodified vLLM commit (pinned in `INSTALL.md`). Patches are applied at runtime via the dispatcher, anchored to known commits. You can run Genesis-on/Genesis-off with the same vLLM binary by toggling environment variables.

### Q: Is it compatible with vLLM v0.20.x main?
A: Genesis tracks specific vLLM commits (currently `fe9c3d6c5` and the v0.20.0 / v0.20.1rc0 tags). Each patch declares an `applies_to` range, so newer vLLM commits cause patches to print `[SKIP — applies_to mismatch]` rather than crashing. Bumping the pin is a deliberate release event.

### Q: How do I update vLLM without losing patches?
A: Bump the `applies_to` range on each affected patch and re-run the anchor-verification suite. Most text-patches survive minor vLLM updates because their anchors are short and stable; some need the anchor adjusted by a few characters. The Genesis CI doctor command (`genesis doctor`) tells you which patches drifted before you boot.

### Q: How do I enable or disable an individual patch?
A: Each patch is gated by a single environment variable: `GENESIS_ENABLE_P67=1` turns it on, unset or `=0` turns it off. The boot log prints every patch and its decision. There is no global "enable all" switch — by design.

### Q: Which patches are ON by default?
A: **None.** Every patch is opt-in. A fresh Genesis install with no env flags behaves identically to stock vLLM. Production launch scripts under `scripts/` declare exactly which patches they want.

### Q: I have one RTX 3090 — what should I run?
A: `Qwen3.6-27B-int4-AutoRound` from Lorbus, TP=1, context up to 32K, no prefix-caching, no DFlash. See `scripts/start_27b_int4_fp8_e5m2_short_single_card.sh` for a working launch line.

### Q: I have 2× 24 GiB cards — should I run 27B or 35B?
A: Depends on workload. 35B-A3B-FP8 (MoE) wins on prose quality and broad-knowledge tasks; 27B-int4 wins on tool-call reliability, long context (320K validated), and raw TPS. If you primarily run agentic / tool-calling pipelines, start with 27B.

### Q: Is LoRA supported?
A: Not actively tested. vLLM's LoRA system should work because Genesis patches are mostly orthogonal to LoRA loading, but no Genesis-validated LoRA recipe exists. Try it and report results.

### Q: Does streaming work?
A: Yes. Patch P61b adds a streaming overlap guard that fixes a slice bug in upstream Qwen3 streaming output. Enable `GENESIS_ENABLE_P61B=1` together with the rest of the tool-call family if you stream tool calls.

### Q: Does tool-call work reliably?
A: Yes — this is one of Genesis's main focus areas. The P59 / P61 / P62 / P64 / P68 / P69 patch family fixes upstream regressions in Qwen3 tool-call generation, especially around `<think>` tags, multi-tool prompts, and streaming. Enable them together via the `tool_call_safe` recipe.

### Q: How do I download the DFlash draft model?
A: It's a gated HuggingFace repo (`z-lab/Qwen3.6-27B-DFlash`, `z-lab/Qwen3.6-35B-A3B-DFlash`). Accept the license on the model page, then `huggingface-cli login` with a token that has read access. Genesis will not auto-download it for you.

### Q: What if patches break my boot?
A: First, look at the boot log — Genesis prints `[APPLY]` / `[SKIP]` / `[FAIL]` for every patch with a reason string. Disable the failing patch by unsetting its `GENESIS_ENABLE_*` flag. If you can't find a working subset, file an issue with the full boot log; include your vLLM commit hash, GPU model, and the model checkpoint.

### Q: How do I add my own model to Genesis?
A: See `docs/CONFIGS.md` for the full guide. Short version: copy a base launch script from `scripts/`, update model path + env vars, test boot + tool-call sanity, submit PR with bench numbers.

### Q: MoE backend — Triton or FlashInfer?
A: Workload-dependent. Triton MoE is more stable on consumer Ampere/Ada and is the Genesis default for 35B-A3B-FP8. FlashInfer MoE is faster on Hopper/Blackwell but has had stability regressions (see vLLM #41306). On 2× A5000, Triton wins.

### Q: Why DFlash instead of MTP?
A: DFlash is trained for code-heavy workloads and produces longer accepted runs on programming tasks. MTP is built into Qwen3.6 itself and works better for chat/prose. Run both, measure acceptance rate on your real traffic, pick the winner. Genesis empirical numbers: MTP K=3 wins prose by ~30%, DFlash N=5 wins code by ~50%.

### Q: Where do I see which patches were applied at boot?
A: The Genesis Dispatcher prints a structured log block right after vLLM model load. Look for lines starting with `[INFO:genesis.apply_all] [Genesis] applied: P67 ...` or `[INFO:genesis.apply_all] [Genesis] skipped: P40 (reason)`. The full registry status with `APPLY`/`SKIP`/`FAIL` summary is also printed at boot end.

### Q: A patch shows "SKIP" — is something broken?
A: Almost always no. SKIP means either you didn't enable the patch (default), or the dispatcher decided it doesn't apply to your environment (wrong GPU, wrong KV dtype, wrong model family). Patches are opt-in and self-gated. Only `[FAIL]` is a real problem.

### Q: Can I run Genesis without Docker?
A: Yes. Genesis is a regular Python package and patches a vLLM installed in the same environment. The Genesis reference deployment uses Docker for repeatability, but bare-metal pip works too. Just remember that text-patches mutate files inside `site-packages/vllm/` — back them up or use a venv per Genesis version. See `scripts/bare_metal_*.sh` for examples.

### Q: How much performance should I expect over stock vLLM?
A: On the Genesis reference rig (2× A5000, Qwen3.6-27B-int4) with the recommended patch set: roughly 25-40% TPS uplift versus the same vLLM commit with no patches, plus tool-call reliability improvements that don't show up in TPS numbers. Your numbers will differ by GPU and workload — always benchmark.
