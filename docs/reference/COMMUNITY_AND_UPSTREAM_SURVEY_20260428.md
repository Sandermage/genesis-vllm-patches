# Community + Upstream Survey — 2026-04-28 morning

Triple research run while Sander slept. 3 agents in parallel covered:
- noonghunna repos (downstream practitioner cluster)
- Sandermage/genesis-vllm-patches community (forks, mentions, issues)
- vLLM upstream activity last 30h

---

## A. Sandermage/genesis-vllm-patches community

### Stars + velocity

- **23 stars** (was 21 in memory) → **+2 in last 24h**
- 18 of 23 stars came in last 5 days → accelerating
- Repo age: 10 days
- Recent starrers worth noting: `chaosong` (vLLM-adjacent), `null-dev`, `flockonus`, `shawn-ms`, `NeoKactus`

### Forks

- **0 literal GitHub forks** (memory's "5 forks" was loose terminology)
- **8 derivative recipe repos** (was 5):
  - noonghunna/qwen36-27b-single-3090 (58 stars, 11 forks)
  - noonghunna/qwen36-dual-3090 (28 stars, 3 forks)
  - tribixbite/wsl-llm
  - thc1006/qwen3.6-vllm-2x3090
  - tedivm/qwen36-27b-docker
  - k0zakinio/qwen36-vllm-setup
  - **danbedford/qwen36-dual-3090-nvlink** (NEW since memory) — fork of noonghunna's dual with NVLink, merged PR #1 on 2026-04-26
  - **AlexsJones/llmfit** (NEW since memory) — benchmark cache referencing Genesis v7.14 + P65

### Issues on Sander's repo

- **#1 OPEN** clayboby (2026-04-19) "OOM with long context" — Sander already responded in detail 2026-04-22
- **#2 OPEN** masrudyn (2026-04-26) "patch_genesis_unified.py" — UNANSWERED ~48h. Draft reply prepared at `docs/upstream/draft_replies/09_genesis_issue2_masrudyn_shim_RU+EN.md` (awaiting Sander GO).

### Cross-references in vllm-project/vllm

Sander-authored work tracked: PR #40914, PR #40127, Issue #40124, Issue #40875.
Cited by: noonghunna (#40807, #40831, #40880), jhsmith409, Bot1822, ExtReMLapin.
**No vLLM core maintainer (kaichao, simon-mo, etc.) has linked or @-mentioned Genesis or Sandermage in PR descriptions.** Engagement is from peer practitioners, not core team.

---

## B. noonghunna ecosystem (recent activity since 2026-04-23)

### Recent commits worth noting (selected)

- `a260bda` (2026-04-27 noonghunna): migrated default/tools-text/longctx composes from missing `patch_genesis_unified.py` to v7.14+ modular `python3 -m vllm._genesis.patches.apply_all` — confirms our compat shim was on time
- `fed529b` (2026-04-27 noonghunna): "fix(eager): Genesis P4 IS required when using `turboquant_3bit_nc`" — community confirmed our P4 is mandatory for hybrid TQ
- `c34bbf1` (2026-04-27 noonghunna): made `patch_tolist_cudagraph.py` robust to nightly + non-docker (auto-discovers vLLM via `import vllm`, single-line regex anchors) — improvement to our external_probe pattern
- `a703bb7` / `9d17d91` (2026-04-27): `bench.sh v2` adds `wall_TPS`, `decode_TPS`, `TTFT`, `CV` — bench harness directly compatible with our genesis_bench_v4

### CRITICAL bug report — ampersandru on Issue #1

> "tool calls can be hit pretty badly... using 7.15 would just fail to output a response. Genesis 5.12 is much more stable... **In 7.15, the anchor for the `_prefill_attention` patch was not found, leaving a critical part of the vLLM engine unpatched.** In your 5.12 run, both Site A and Site B applied successfully."

**This is a serious issue:** v7.15 had anchor drift that silently failed to apply a critical patch (likely P67 hook on `_prefill_attention`). Without P67 + cudagraph spec-decode = crash OR silent corruption.

**Action item HIGH:** verify our text_patch framework LOGS clearly when anchor not found (we shipped `required=True` for sub_patches, but does the user notice in boot logs?). Also verify v7.59 anchors all apply (ampersandru's report was on v7.15 which is older).

### Other unresolved community items

- **walmis (Issue #5):** WSL2, CPU-bound, `nproc=6` in container despite 16 passed → Sander posted debug guide 2026-04-27, walmis confirmed CPU-bound, awaiting next step.
- **3dluvr** (different repo): Minachist INT8 + plain nightly hits **111-134 tok/s @ 98.6% accept** WITHOUT Genesis. Asking if our Marlin pad-sub-tile-n PR #40361 applies to W8A16 (INT8) too. **Concerning data point** — suggests our patches may not be net-positive for INT8 single-stream on 2× 3090.
- **ampersandru on Genesis 7.53 (latest):** `mean_tps=84.0, mean_q=0.98, passed=6/6` BUT "tools is still a bit broken" — Hermes + Opencode tool callers get no response from his agent.

### Cross-rig P67 validation (positive)

noonghunna dual-3090 + Genesis v7.48 + P67/P67b/P78 + 35B-A3B:
- **wall_TPS 136.87, CV 2.2%, decode 139.15, TTFT 119ms, AL 2.5, accept ~30%**

Solid cross-hardware confirmation our P67 kernel works on 3090 too.

---

## C. vLLM upstream activity last 30h

### MERGED PRs — direct relevance

- **#40941 MERGED** (2026-04-27) by bhoomit: "Share TQ dequant buffers, eliminate float16_copy" — saves 57 GB @ 1M ctx in TurboQuant continuation-prefill. **Same PR our 04-27 research already flagged.** Just bump pin to get it free.
- **#40946 MERGED** (2026-04-27) by Dao007forever: "Cap SWA/chunked-local runtime admission" — fixes long-prompt rejection on hybrid full+SWA models. MEDIUM relevance for our 320K config.
- **#40651 MERGED** (2026-04-27): MRV2 sampling acceptance gap. LOW (we're MRV1).

### OPEN PRs — backport candidates

- **#40915 OPEN** by jow-: "Redesign Qwen3 XML tool parser as character-level state machine" — direct overlap with our P58/P59/P61b/P62 stack. Could SUPERSEDE our band-aid set. **HIGH WATCH.**
- **#40962 OPEN**: "Validate post-reasoning structured output tokens in spec decode" — exact intersection of our P62. May conflict OR complement.
- **#40956 OPEN**: "correct h matrix layout in chunk_kda output kernel" — Gated DeltaNet inter-chunk h-state mismatch in FLA. **HIGH for residual ngram ceiling.** Worth A/B.
- **#41043 OPEN** by wangluochao902: "Avoid per-step numpy allocation in prepare_next_token_ids" — spec-decode hot path perf. Track for backport.
- **#40921 OPEN**: "Fix RMSNormGated input_guard torch.compile dynamo tracing" — Qwen3.5/3.6 startup. Verify on next pin bump.

### NEW issues (regression / Ampere / long-context)

- **#41014 OPEN** by hanchaoqi: SM 8.6 fp8e4nv default unsupported on Ampere. Our P3+P4 mitigates this — could earn the issue by commenting "Genesis P3+P4 is the practical workaround until upstream lands the proper fix".
- **#41031 OPEN**: AssertionError in `sampler.py:383` with `prompt_logprobs=20`. **Test on PROD** before next deploy.
- **#41022 OPEN**: FP8 logical_widths on MergedColumnParallelLinear. Check our FP8 path.

### Release tags

- **v0.20.0** tag landed 2026-04-27 17:58 UTC (commit `88d34c64`)
- **v0.20.1rc0** tag landed 2026-04-27 08:17 UTC (commit `ebf862c3`)

We're pinned to `0.19.2rc1.dev212+g8cd174fa3` (~Apr 25). v0.20.0 is 2 days ahead. **Do NOT auto-bump** — wait for changelog + RC stabilization.

### Sander's own activity status

- **PR #40914**: noonghunna confirmed cross-rig (dual 3090) on 04-26 23:19 UTC; Sander promoted draft → ready-for-review on 04-27 14:47 UTC. **No maintainer review yet** (~24h). Consider pinging spec-decode CODEOWNER if quiet 24h more.
- **Issue #38898** (Mamba conv state): Sander's 04-27 14:47 comment is the latest. No reply from NickLucche or maintainers in window.

---

## Synthesis — prioritized action items

### Tier 0 — IMMEDIATE (next session)

1. **[HIGH] Investigate ampersandru's anchor-drift report.** Verify v7.59 ALL anchors apply (not just silently no-op). Check that text_patch framework emits clear WARNING when `required=True` anchor missing. Verify P67 hook actually injects on PROD v759 (it does per boot log: "kernel_built True" + "P67 hook injected").
2. **[HIGH] Reply to Issue #2 (masrudyn).** Draft prepared. Sander GO needed.
3. **[MEDIUM] Reply to walmis (Issue #5)** — WSL2 CPU-bound. Suggest specific WSL2 tuning (`.wslconfig` `processors=16`, `memory=64GB`).

### Tier 1 — THIS WEEK

4. **[MEDIUM] Pin bump to v0.20.0 OR v0.20.1rc0** once changelog published. Drops cherry-picks for #40941 (TurboQuant share buffers — saves 57GB at 1M ctx) and #40946 (SWA cap).
5. **[MEDIUM] Read PR #40915** carefully (Qwen3 parser rewrite) — could replace 4 of our patches (P58/P59/P61b/P62). If clean replacement, retire our band-aids.
6. **[HIGH] A/B test PR #40956** (KDA h-matrix layout fix) — might address residual ngram clean-rate ceiling we documented in v7.13 strict-ngram breakthrough.
7. **[MEDIUM] Test PR #41031 reproducer** (`prompt_logprobs=20` AssertionError) on PROD — we don't currently set this but agentic clients might.

### Tier 2 — STRATEGIC

8. **[MEDIUM] 3dluvr's INT8 data point** (134 tok/s @ 98.6% accept WITHOUT Genesis on Minachist INT8 + plain nightly). Investigate whether Genesis adds value for INT8 single-stream OR is FP8/W8A8-specific. If INT8 doesn't benefit, document MODELS.md explicitly.
9. **[LOW] Comment on Issue #41014** (SM 8.6 fp8e4nv) noting Genesis P3+P4 mitigates. Earns community goodwill, brings traffic.
10. **[LOW] noonghunna's `c34bbf1`** (auto-discover vLLM via `import vllm` in tolist_cudagraph fix) — adopt for our external_probe to be more portable.
11. **[LOW] cleanup launch_scripts/test/** — 24 scripts, many overnight bisect artifacts. README inventory placed; archival still pending.

### Tier 3 — DEFER

- noonghunna NVLink variant: useful reference if Sander gets NVLink hardware (currently dual A5000 are PCIe).
- Suffix Decoding (P75) deploy variant: needs real-workload bench (agentic tool-call replay). Not standard bench.
- v756-deploy with cache: now safe via P67 safety gate, but our single-user profile doesn't benefit from cache.

---

## Files added to repo

- This survey: `docs/reference/COMMUNITY_AND_UPSTREAM_SURVEY_20260428.md`
- Issue #2 reply draft: `docs/upstream/draft_replies/09_genesis_issue2_masrudyn_shim_RU+EN.md`
- Server-side: `launch_scripts/test/README.md` + `launch_scripts/current/README.md`

## Memory entries added

(to be added to `~/.claude/projects/.../memory/`)
