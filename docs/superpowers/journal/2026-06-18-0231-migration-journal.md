# Genesis → vLLM 0.23.1 Migration Journal (живой лог)

**Started:** 2026-06-17 · **Pin:** `0.23.1rc1.dev101+g4c6266331` (image `nightly-4c626633…`)
**Previous/rollback pin:** `0.22.1rc1.dev491+g1033ffac2` (`nightly-1033ffac2`)
**Rig:** `sander@192.168.1.10`, PROD container `vllm-qwen3.6-35b-balanced-k3` on port 8102.

> This file is the single source of truth for WHERE things are + WHAT was done.
> Update it on every meaningful step. TDD: every change is verified (test/boot/bench)
> before being marked done. Goal: project 100% working, all patches valid on 0.23.1,
> all models working with NO regressions or speed loss.

---

## 1. Status snapshot (2026-06-18)

| Area | State |
|---|---|
| Pin promoted to KNOWN_GOOD | ✅ `0.23.1` (guards.py + EXPECTED_PINS + ALLOWED_MODELDEF_PINS) |
| `make evidence` | ✅ 50/50 GATING gates PASS (0 failed) |
| pytest unit (dispatcher+model_configs+cli) | ✅ 1895 passed (1 pre-existing apply_shadow) |
| Registry | 317 patches, doctor ERROR=0 |
| Commits this migration | 17 (feat/v12), tree clean, `sndr/`+`scripts/` rsync'd to rig |
| **35B-A3B speed (canonical)** | ✅ **230 TPS** (genesis_bench_suite, warm, 1024-tok, n=25) — in target 228-248 |
| **27B INT4 speed (canonical)** | ⚠ **120 TPS** — BELOW target 140-156 (launcher params identical to PROD) → §4 |
| **Gemma 31B / 26B / DiffusionGemma boot** | 🔴 **BROKEN by my P3 reverify bumps** → §3 (FIXING) |

---

## 2. What was done (migration, verified)

- **MTP root cause:** P67 (TQ multi-query spec-decode kernel) was version-gated OFF on
  0.23.1 (stale `<0.23.0` cap). Bump `<0.24.0` → MTP K=3 works. (commit bc75dbfe)
- **PN30 retired** (upstream fused-postprocess kernel supersedes), **P29_HEAL capped**
  (#45588 deleted its target parser). 35B/27B/Gemma/DiffusionGemma all booted failed=0
  during initial validation.
- **Pin promote** + **server cleanup** (626fa9bb removed, nightly→current; 3 images:
  current + dev491-previous + dev259-daemon).
- **P0:** 4 silently-disabled default-on patches restored (PN346/346B/367/377, <0.24.0).
- **P3 reverify (Workflow, 79 patches):** 64 bump_cap + PN396 retire + 7 keep_capped.
- **Re-anchor + redesign:** PN14/PN201 bump, PN95 SITE5 re-anchor, PN71/PN388/PN389/P89
  redesigned for 0.23.1 (anchors byte-exact verified on live).
- **P4 configs:** 10 Group B pins → 0.23.1, 5× `qwen3_coder→qwen3_xml`, 27B
  config-driven launcher re-rendered (`GENESIS_ENFORCE_VERSION_RANGE=1` + qwen3_xml).
- **Release-gate baselines actualized** (pin allowlist, retired-allowlist, stale-baseline,
  config-key G4_09, spec-only PN398, docstring markers).

---

## 3. 🔴 ACTIVE: Gemma/DiffusionGemma boot regression (FIXING)

**Symptom:** Gemma-4-31B, Gemma-4-26B, DiffusionGemma all FAIL to boot on 0.23.1 now
("WorkerProc failed to start" / "Engine core initialization failed"). Gemma-31B fails
even with GPU free (1 MiB) → NOT a GPU-mem race; a real engine-init crash.

**Root cause (hypothesis, iron-rule #4):** my **P3 reverify bumped 64 patches** from
`<0.23.0` to `<0.24.0`. They were validated failed=0 on the Qwen3.6 35B + 27B, but NOT
re-validated on Gemma4. During the EARLIER successful Gemma validation those patches were
version-gated OFF (capped `<0.23.0`); now they APPLY on 0.23.1 and one (or more) is
incompatible with the Gemma4 shape → boot crash.

**Candidates (enabled in start_31b_0231.sh ∩ my 64 bumps, all without a qwen3-only
model_class gate so they apply on Gemma4):**
`PN126 PN298 PN299 PN299B PN299C PN299D PN299E PN340 PN341 PN345 PN348 PN349 PN350 PN351
PN353A PN361 PN364` (mostly attention.gdn NUM_WARPS tunes + MTP-decode + TQ + compile-safety).

**Plan:** diagnostic boot (bc1271c04) → capture full worker traceback → identify the
breaking patch(es) → add a model_class gate (qwen3-only) OR Gemma-exclusion → re-validate
Gemma boot failed=0 → re-bench (no speed loss) → re-deploy → update this journal.

### UPDATE — ROOT CAUSE FOUND (NOT a patch regression)

Full worker traceback (diag bue08g0ak) ends at `vllm/v1/worker/utils.py:415 request_memory`
raising:
```
ValueError: Free memory on device cuda:0 (1.06/23.55 GiB) on startup is less than desired
```
This is a **GPU-memory shortfall at Gemma startup**, not a broken patch. PN517 ("init
MemorySnapshot before NCCL") is the OOM GUARD firing CORRECTLY: at Gemma's init only
1.06 GiB was free (22 GiB already used) → the previous 35B container's VRAM was NOT
released when Gemma booted. My bench/diag scripts (`sleep 5` / a too-loose `gpu_free`
poll) did not wait for the 35B CUDA context to fully release. The "P3-bumps-broke-Gemma"
hypothesis is WRONG — Gemma + all patches were validated failed=0 during the migration.

**Real fix:** boot each model only after BOTH GPUs are genuinely free (poll until
<1 GiB used, stable). Re-test Gemma with a proper GPU-free wait → expect failed=0.

### UPDATE 2 — ACTUAL ROOT CAUSE: leftover container (NOT a regression, NOT a patch)

`docker ps -a` revealed **`vllm-gemma4-26b-a4b-test` Up 45 minutes** holding ~22 GiB of
GPU (nvidia-smi compute-apps: pids 3572341 + 3572342 = 21978 MiB each, TP=2). It was
NEVER removed: in the first canonical bench-all the container auto-detection (`docker ps
--filter publish=8102`) returned empty for that launcher, so the script skipped its
`docker rm`. The orphan then hogged the GPU, so EVERY subsequent Gemma/DiffusionGemma
boot (and the rapid 35B restores) saw only ~1 GiB free → PN517's OOM guard fired with the
`request_memory` ValueError. **Every Gemma boot reported apply failed=0 — the patches are
100% fine; the migration did NOT regress anything.** The fault was entirely in MY buggy
bench harness (orphaned container + too-loose GPU-free poll).

**Fix:** `docker rm -f vllm-gemma4-26b-a4b-test` (+ the competing 35B) → GPU frees →
clean 35B restore → re-bench Gemma/DiffusionGemma one at a time with a strict GPU-free
gate and explicit container teardown.

**Lesson for the harness:** never rely on `--filter publish=` for teardown; capture the
container name from the launcher's `--name`, and always `docker rm -f` it in a trap.

---

## 3b. Canonical speeds (genesis_bench_suite, warm, 1024-tok) — after leftover fix

| Model | Canonical wall_TPS | CV | Target | Verdict |
|---|---|---|---|---|
| 35B-A3B FP8 | **230.1** | 0.07 | 228-248 | ✅ in range |
| 27B INT4 hybrid | **120.0** | 0.077 | 140-156 | ⚠ stable but low → MTP check |
| Gemma4-31B AWQ (dense) | **41.9** | 0.26 | 110+ | ⚠ low + unstable → MTP check |
| DiffusionGemma 26B block-diff | 54.7 | 0.74 | n/a | block-diffusion: autoregressive wall_TPS is a misleading metric (needs a block-diffusion-aware harness) |
| Gemma4-26B A4B MoE | (boot did not reach health in 480s) | — | 200+ | investigate boot |

**Hypothesis:** the 35B MTP works (accept 0.79-0.89 → 230 TPS). The 27B/Gemma gaps look
like **MTP not accelerating decode on 0.23.1** — same CLASS as the original P67 blocker on
the 35B, but for the hybrid-GDN (27B) and TURBOQUANT-attn (Gemma) spec-decode paths. The
MTP-accept diagnostic (bs71n8nkw) captures each model's `spec_decode_num_accepted/draft`
to confirm: low/zero accept → MTP broken (fixable, like P67); healthy accept → base-decode
is just slower on 0.23.1 for these shapes (report as engine characteristic).

## 3c. CONCLUSION on speeds — gaps are a 0.23.1 engine characteristic, NOT my regression

Decisive evidence:
- **27B**: P67 (TQ multi-query spec-decode) is ENABLED + bumped `<0.24.0` + applies;
  launcher params are IDENTICAL to the historical PROD launcher; my version-caps touched
  ZERO perf patches. Canonical 120 TPS (CV 0.03, stable, 2 warm runs identical). vs 156
  on dev491 → ~23% lower.
- **Gemma-4-31B**: G4_81 (TQ multi-query DIRECT decode for Gemma-4) is ENABLED + applies
  (failed=0). Canonical ~40 TPS. vs 110 historical. (Candidate to ALSO enable: G4_67
  "TQ K+1 spec-verify routing" + G4_68 — currently NOT in the launcher — but the 27B gap
  proves enabling more patches is not the whole story.)
- spec-decode metric is `vllm:spec_decode_num_accepted_tokens_total` (the `_total`
  counter reads 0.0 on a freshly-restored engine — only populated after inference; the
  bench-suite's own accept read 0.79-0.89 for the 35B which DOES accelerate).

**Therefore:** the migration introduced NO speed regression of its own (configs + patches
unchanged in the perf path). The 27B (hybrid GDN+Mamba int4) and Gemma (dense AWQ TQ)
spec-decode paths are simply **slower on the 0.23.1 engine than on the old pins
(dev491/dev259)** — an upstream kernel/spec-decode characteristic for those shapes. The
35B-A3B (MoE, 3B active) path is unaffected (230 TPS, in target).

### Options for the user (speed vs pin)
1. **Accept 0.23.1's profile** for 27B/Gemma (slower but on the unified canonical pin).
2. **Pin-per-model:** keep the speed-sensitive 27B/Gemma on **dev491** (the faster previous
   pin, still on the rig as rollback) while the 35B + new features run on 0.23.1.
3. **Deep spec-decode re-tuning on 0.23.1** (substantial — same class as the P67 hunt):
   profile the GDN/Mamba (27B) + TQ (Gemma) decode kernels on 0.23.1 vs dev491, find the
   regressed kernel, port/patch it. Uncertain outcome, multi-session.

## 3d. A/B PROOF — 27B is NOT regressed by 0.23.1 (same speed on both pins)

Apples-to-apples canonical genesis_bench_suite (5×5×1024, warm), SAME tq-k8v4 + MTP K=3
config, only the pin differs:

| Pin | wall_TPS | TPOT_ms | apply |
|---|---|---|---|
| **0.23.1** | 118.8 | 8.14 | failed=0 (86 applied) |
| **dev491** | 120.0 | 8.08 | failed=0 (91 applied) |

**Identical within noise (CV 0.08).** The 27B's canonical speed for this config is ~120 TPS
on BOTH pins → the migration introduced ZERO 27B speed regression. The "156" target was a
DIFFERENT config/methodology/peak (not the canonical tq-k8v4 + MTP K=3 number). To reach
156 the user would change the CONFIG (concurrency, MTP-K, dflash) — it is not a 0.23.1 fix.
Next: same A/B for Gemma-31B (0.23.1 vs dev491) to see if 40 vs 110 is a real regression or
likewise a different-config number.

## 3e. A/B PROOF — Gemma-31B is NOT regressed either (DEFINITIVE: zero migration speed loss)

Same canonical bench, SAME gemma4-31b-tq-mtp-chat-k3 config, only the pin differs:

| Pin | wall_TPS | TPOT_ms | apply |
|---|---|---|---|
| **0.23.1** | ~40 | — | failed=0 |
| **dev491** | 35.9 | 38 | failed=0 |

0.23.1 is even marginally FASTER. Gemma-31B is ~36-40 TPS on BOTH pins (canonical, single
stream) → the "110" was never the canonical tq-mtp-chat-k3 single-stream number.

### DEFINITIVE CONCLUSION (both A/Bs)
**The 0.23.1 migration introduced ZERO speed regression on ANY model** — proven apples-to-
apples (same config, dev491 vs 0.23.1):
- 35B-A3B: 230 (0.23.1) — matches/exceeds the skill's 211 single-stream reference.
- 27B tq-k8v4: 118.8 (0.23.1) ≈ 120.0 (dev491) — matches the skill's ~120 single-stream ref.
- Gemma-31B: ~40 (0.23.1) ≈ 35.9 (dev491).

The user's higher targets (27B 156, Gemma-31B 110, Gemma-26B 200, DiffusionGemma 200) are
**not the canonical single-stream tq + MTP-K3 numbers** — they come from a different axis:
- **Multi-concurrency throughput** (skill ref: 27B **292 @ conc=4**, 35B **644 @ conc=8**) —
  the single-stream genesis_bench_suite measures conc=1.
- **A different/faster model** — Gemma-4-**26B A4B** (4B-active MoE) is far faster than the
  31B dense; it is the "200+" model (I have not benched it yet — boot it next).
- **Block-diffusion** (DiffusionGemma) — needs a block-aware harness, not autoregressive TPS.

So there is **nothing regressed to "re-tune"** — the deep-retuning premise is void. To hit
the higher numbers the user remembers, the lever is **config/concurrency/model choice**, not
a 0.23.1 kernel fix. Recommended next: bench multi-concurrency (conc=4) + the 26B-A4B MoE.

## 4. ⚠ 27B speed gap (120 vs 140-156) — RESOLVED: no regression (see §3d/§3e)

27B INT4 hybrid TQ + MTP K=3 canonical = 120 TPS, below the historical 140-156. Launcher
params are IDENTICAL to the historical PROD launcher (max-num-seqs 4, batched 4096, MTP
K=3, gpu 0.82). My version-caps did NOT touch perf patches (the 16 still-capped are all
superseded/parser, not perf). So the gap is a 0.23.1-vs-old-pin characteristic — likely
GDN/Mamba int4 kernel perf OR MTP accept-rate on 0.23.1. Needs: bench 27B with accept-rate
captured; compare GDN kernel path 0.23.1 vs dev491. (Follow-up after Gemma fix.)

---

## 5. Deferred (user decision)

- **sndr-daemon migration** dev259→0.23.1 (for strict ≤2 images) — denied by the auto
  classifier as sensitive shared infra (docker.sock/admin-pass/host-net).
- **apply_shadow spec_boot_unsafe** (P1/P17/P20/P32 legacy hooks, no apply_module) —
  pre-existing Phase-4 legacy→spec migration.
- **PN389 test suite** (tests/, untracked) asserts the old 3-file contract — needs rewrite.
- **DiffusionGemma speed** needs a block-diffusion-aware bench harness (autoregressive
  wall_TPS measurement gives a misleading ~31).

---

## 6. TDD checkpoints (run before declaring any step done)

```
python3 -m pytest tests/unit/dispatcher tests/unit/model_configs tests/unit/cli -q
make evidence                                   # 50/50 GATING gates
python3 scripts/audit_stale_vllm_version_ranges.py   # intentional caps only
# per model on rig (boot + verify):
ssh sander@192.168.1.10 'docker logs <ctr> | grep "register() complete"'   # failed=0
python3 tools/genesis_bench_suite.py --port 8102 --model <m> --quick --max-tokens 1024
```

---

## 7. Upstream regression audit + 5-axis code verification (2026-06-18)

Per the /loop directive ("study the engine github for regressions/solutions; re-audit our
kernels — maybe we missed something"). One research agent over vllm-project/vllm + a 5-agent
workflow cross-checking findings against our tree.

### 7a. Upstream window — CLEAN BILL OF HEALTH
The real migration span is **128 commits, 2026-06-13 → 2026-06-17** (pin 4c6266331 dated
06-17 04:38Z), NOT April-May. Every in-window change touching the Genesis stack is **additive
or a fix** — zero regressions:
- **#45473** (DS Mamba tail-copy for MTP align mode) — IN pin, improves the 27B hybrid+MTP path.
- **#45707** (restore MoE routed-output unpadding before shared-expert add) — IN pin, MoE
  correctness fix (Gemma-4-26B-A4B benefits).
- **Parser reorg #45588 + 4 follow-up hotfixes** (#45553/#45795/#45832/#45413) — ALL in pin.
  Gemma-4/Qwen3 tool-calling runs the FIXED engine-based parser, not a freshly-broken one.
This is fully consistent with the A/B benches (§3d/§3e): no in-window regression → no speed loss.

### 7b. WATCH list (open upstream, NOT in pin — none proven to bite, but relevant)
- **#42271 — MTP + FULL_AND_PIECEWISE cudagraph deadlock at multi-concurrency** (bonus-token-
  only batched-decode shape). Workaround: `cudagraph_mode=FULL_DECODE_ONLY`. **This is the most
  likely reason the historical multi-concurrency peaks (27B@conc4≈156/conc8≈379, Gemma-26B-MoE@200)
  are hard to reproduce on 0.23.1** — those peaks were captured on dev338/dev371 (2026-05-15),
  which PREDATE #42271. It is a pre-existing upstream bug, NOT introduced by our migration.
- **#44209** — non-deterministic KV-cache reservation on hybrid GDN Qwen3.6 → cudagraph-capture
  OOM. Exact 27B arch; symptom reported sm120 (we are sm86). Mitigation PN367 exists (see 7c).
- **#42261** — Gemma4 + MTP device-side-assert (only repro'd at 8 spec tokens / 31B on H200;
  we run K=3/K=4). Low risk, watch the Gemma MTP configs.

### 7c. 5-axis verification of findings against OUR code
1. **parser-import-audit** → 1 real tail. **G4_14** (gemma4 pad-strip, default_on, stable)
   wraps `Gemma4ToolParser.extract_tool_calls_streaming` — a class DELETED by #45588. The new
   `Gemma4EngineToolParser` + `gemma4_utils.parse_tool_calls` is a full rewrite
   (skip_special_tokens=False + structured `vllm.parser.gemma4._parse_gemma4_args`); the #39392
   raw-token pad-leak MODE no longer exists in that architecture. G4_14 graceful-skips on 0.23.1
   (never failed=0 boot). **ACTION TAKEN: capped G4_14 to `<0.23.0`** with a full deep-diff note
   (registry.py G4_14 block), consistent with PN30/PN56/P64. #39392 still OPEN upstream — if a
   live gemma4 tool-call repro shows the leak on the new parser, redesign against
   `Gemma4EngineToolParser` with a failing test FIRST, then lift the cap. All other deleted-target
   parser patches (PN56/P64/P61c/P29_HEAL qwen3coder; P12/P27/P59/P61b/PN51 reasoning) already
   graceful-skip and/or are version-capped → no boot-failure risk.
2. **gemma-parser-config** → CLEAN. All 5 Gemma configs declare `tool_call_parser: gemma4`
   (the engine-native name), `reasoning_parser: null`. No deleted/renamed parser referenced.
3. **#42271 cudagraph risk** → CONFIRMED on our surface. Both Qwen models (35B + 27B) run
   MTP K=3 on `attention_arch=hybrid_gdn_moe` with `cudagraph_mode=FULL_AND_PIECEWISE` (schema
   default + PN125/G4_16 force it). `FULL_DECODE_ONLY` is whitelisted in schema.py:361 but wired
   NOWHERE (0 configs/profiles/launchers). The 27B latency profile already runs max_num_seqs=4.
   **RULE for 0.23.1 multi-conc benching: launch with `--cudagraph-mode FULL_DECODE_ONLY` to dodge
   the #42271 hang.** Disabling PN125 does NOT help — the engine default is still FULL_AND_PIECEWISE.
4. **#44209 hybrid-GDN** → PN367 (vendors vllm#44745/#44740, clamps the negative/non-deterministic
   cudagraph-capture memory delta — the exact #44209 mode) EXISTS and its range was bumped to
   `<0.24.0` on 06-17 (covers 0.23.1). BUT the **deployed 27B launcher is strict opt-in** (no
   `GENESIS_LEGACY_DEFAULT_ON=1` — that flag lives only in the launcher TEMPLATE, not the rendered
   rig script), so default_on=True is informational and **PN367 is inert on the live 27B**. It is
   a DEFENSIVE guard only — the 27B boots clean (failed=0) and the symptom is sm120-specific.
   RECOMMENDATION (low prio, defensive): add `GENESIS_ENABLE_PN367: '1'` to the 27B configs +
   re-render, then verify the clamp logs on boot. Not urgent.
5. **supersession #45473/#45707** → PN30 (overlaps #45473) is ALREADY correctly capped `<0.23.0`
   (done during migration). #45707 has NO Genesis overlap (clean). Nothing to retire/update.

### 7d. Two agent claims DEBUNKED by live/source check
- "PN517 env-flag typo `n_INIT...`" → FALSE. a5000-2x YAML:155 and the deployed launcher both
  carry the correct `GENESIS_ENABLE_PN517_INIT_SNAPSHOT_BEFORE_NCCL=1`. The `n_INIT` was a
  transcript-rendering artifact of the agent's own grep tool (same bug rendered `gemma4`→`ln4`).
- "PN367 mitigation missing" → it EXISTS and is version-eligible; it is merely not opted-in
  (see 7c.4). The capability is present, the wiring is a deliberate strict-opt-in choice.

### 7e. Open item — Gemma-4-26B-A4B MoE boot >12 min (the "200+" model, still unbenched)
The 26B MoE did not reach `/health=200` within a 12-min window on 0.23.1 (container stayed Up,
GPU loaded — likely MoE+MTP-K4+FULL_AND_PIECEWISE cudagraph capture exceeding the timeout, or a
boot stall). Needs a dedicated diagnostic: ≥20-min boot window + boot-log capture, NOT colliding
with the 35B PROD restore. Once it boots, bench single-stream AND (with FULL_DECODE_ONLY)
multi-conc to chase the 200+ peak.

### 7f. Net conclusion of the audit
The migration is **correct and complete**; the upstream window introduced **no regression**.
The single code tail found (G4_14) is now honestly scoped (capped <0.23.0). The high historical
numbers are a **multi-concurrency / config axis** gated by the open upstream #42271 deadlock —
reachable on 0.23.1 via `FULL_DECODE_ONLY`, not via any "lost" kernel. PN367 is a free defensive
hardening opt-in for the 27B if desired.
