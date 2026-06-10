# Genesis pin-preflight residual triage — operator action plan
**Pin:** `0.22.1rc1.dev259+g303916e93` · pristine tree `/private/tmp/candidate_pin_current/vllm` · patches repo `/Users/sander/Documents/Visual Studio Code/genesis-vllm-patches`

---

## 1. IMMEDIATE fixes (HIGH risk, verifier agreed)

### PN353A — remove self-defeating drift marker (HIGH, TOOL_BUG, verifier confirmed byte-level)
- **Problem:** marker `tq_max_kv_splits_for_cuda_graph` is pre-existing pin API (defined in pristine `config/attention.py:31`, present at target line 295, read by PN353A's own replacement) → fires on every pin; `apply()` checks markers before anchor → patch can never apply. PROD has `GENESIS_ENABLE_PN353A` set yet the patch is silently suppressed. **Live consequence:** 27B-tq-k8v4 crashes on first decode (workspace-lock `AssertionError`, `turboquant_attn.py:907`) — open P0 in `docs/superpowers/journal/2026-06-10-fleet-speeds-and-roadmap.md`.
- **Fix:** `sndr/engines/vllm/patches/attention/turboquant/pn353a_tq_builder_workspace_reserve.py` — change `upstream_drift_markers` to `["[Genesis PN353A", "def _reserve_workspace"]` (optionally + a `gh pr view 44053`-verified string from the actual diff). **No anchor change** — PN353A_OLD byte-exact at pristine `turboquant_attn.py:201-205`, count=1, `_reserve_workspace` absent upstream.
- **Verify:** re-run preflight → pristine restore → boot shows `applied: PN353A` → re-run 27B first-decode repro.
- Verifier addendum: PN118's injected text re-plants the same marker — concrete motivation for the §6 self-collision lint.

### P6 — neutralize before this pin goes live (mis-apply hazard, from newly-merged sweep)
- **Problem:** P6's drift marker `'TQFullAttentionSpec,'` (trailing comma) never matches the pristine lazy import (`platforms/interface.py:582`), while BOTH P6 anchors still match pristine → an enabled P6 would "apply" and inject a dead duplicate `elif` + redundant import on top of upstream's merged superset (vllm#39931, which also fixed our `max`→`lcm` bug at `interface.py:573-609`).
- **Fix:** hard-skip/neutralize P6 now (registry gate or marker correction in `p6_tq_block_size_align.py`); full retire in §3.

### 1b. Fix-drifts batch (MEDIUM, verifier agreed — schedule immediately after)

### PN32 — GDN chunked-prefill v3 re-anchor (keep; upstream has no T-dim chunking)
- Re-anchor `PN32_ANCHOR` on pristine `gdn/qwen_gdn_linear_attn.py:1503-1532` (new `# 2.3` block, ending at `# Init cache` persist; full text in fix sketch). Replacement updates: single-seq via `prefill_query_start_loc.shape[0] == 2`; persist into `ssm_state[prefill_state_indices]`; **new cutedsl bypass guard** (CustomOp asserts `chunk_indices/chunk_offsets` not None at lines 398-400).
- **Verifier addition (mandatory):** PN79 sub-patch 3C (`pn79_inplace_ssm_state.py:746-770`, PROD-applied atomically, boot line 113) anchors on the **identical** pristine block. v3 must compose: (a) apply-order dependency + post-PN79 anchor variant, (b) shared sub-patch, or (c) registry ordering/conflict declaration — else v3 dry-run-fails exactly like the drift it fixes.

### P59 — Qwen3 tool_call recovery re-anchor (keep; vllm#39055 still OPEN)
File `sndr/engines/vllm/patches/reasoning/p59_qwen3_reasoning_tool_call_recovery.py`:
(a) `IMPORT_OLD` → `"from collections.abc import Iterable, Sequence\nfrom typing import TYPE_CHECKING"`; (b) add variant C anchored on **P27's post-apply output** (P27 runs before P59 at boot); (c) add variant D anchored on pristine lines 142-144 for P27-absent deployments; upgrade wrap variants to require-at-least-one (today the patch would report "applied" while missing its core `</think>`-present wrap — variants A/B anchor on dead residue); delete/retire variants A/B.

### PN58 — envs.py tail re-anchor (keep; PR #40962 unmerged)
`ENVS_OLD` broke (comment grew to 3 lines + 3 new entries before `}` at pristine `envs.py:2013`). Use the **minimal tail anchor** `"VLLM_NIC_SELECTION_VARS": lambda: os.getenv("VLLM_NIC_SELECTION_VARS", ""),\n}` (verified unique, comment-churn-resistant). Other 5 sub-anchors intact. MultiFilePatchTransaction means the dead anchor currently blocks all 5 files atomically. Mark the PN58 row resolved in the preflight journal backlog (drift was already listed there — not "undocumented").

### PN288 — finish_reason override, 4-part coordinated fix (keep)
(1) New `PN288_STREAMING_OLD` = pristine `chat_completion/serving.py:821-828` (upstream removed `auto_tools_called` from the streaming condition); (2) drop `auto_tools_called` kwarg from `PN288_STREAMING_NEW` **and the except-fallback** (otherwise NameError escapes the defensive wrapper); (3) middleware `decide_streaming_finish_reason` gains `auto_tools_called: bool = False`; (4) sub-patch 2 untouched (byte-exact at 1246-1250).
- **Verifier addition:** coordinate with **P107** (its ANCHOR_OLD includes the same pristine block) — order P107-before-PN288 or coordinate anchors. **File a separate P107 defect:** its v2 injects `and not auto_tools_called` into the streaming generator where the variable no longer exists on this pin (latent NameError); its comment claiming the variable still exists is wrong.

### PN38 — delete sub-patch B, keep A/C/D (keep; PR #40425 still OPEN)
Site B is upstream-native (`get_draft_quant_config` at `qwen3_dflash.py:228`, quant_config passed at 248-254); re-applying B = duplicate kwarg = **SyntaxError**. Remove the `pN38_b_pass_quant_config` TextPatch + constants; add an upstream-presence guard (require both native lines, skip loudly if absent); A/C/D anchors verified count=1 each. Update docstring/registry credit (Site B native since dev259).

*(P85 also needs anchor fixes but the verifier refuted the classifier's recommendation — see §5.)*

---

## 2. Registry/metadata hygiene batch (single commit)

| Item | File / location | Change |
|---|---|---|
| **P26** | `registry.py:6493` + comment 6494-6499 | `superseded_by` → "PARTIAL — cu_2 half only (PR #40420); output_alloc (~32 MiB/call) NOT upstream, still applies". Guards against a future title-matching retire of a live perf win. |
| **PN71** | `registry.py:2431-2458` | `requires_patches: []` → `["P27"]` — PN71's anchor literally contains P27's injected comments. |
| **PN346** | module vs registry | `apply()` ignores registry `GENESIS_ENABLE_PN346`/`default_on=False`, honors only `GENESIS_DISABLE_PN346` → default-ON in practice (boot line 87). Align registry or module. (Found during P85 verification, §5.) |
| **P7** | `registry.py:3007` vs P7 entry | PN204 comment calls P7 "retired"; P7 entry is `lifecycle='legacy'`, `default_on=True`. Align during the P7b retire commit. |
| **PN55** | `pn55_wake_up_hybrid_kv.py:161-168` | Remove `init_fp8_kv_scales` from drift markers (name-collides with merged vllm#28783; #41602/#41896 both OPEN). ANCHOR_OLD intact at pristine 947-950. Document narrowed scope (post-#28783 only hybrid+fp8-KV+wake corner) + journal entry so next sweep sees KNOWN_DOCUMENTED. |
| **P91B** | module docstring lines 87-119 | Stale K.1.R 2026-05-28 note ("refactored out of inc.py entirely") — pattern is back at pristine `inc.py:538`. |
| **overlays** | `overlays/pr42637/single_type_kv_cache_manager.py` | Dormant 0.20-era snapshot with old `use_eagle` code, unreferenced by bind-mount table/launchers — delete or annotate so sibling-anchor sweeps stop re-flagging it. |

---

## 3. Retire queue (iron-rule-#11-verified, byte-level evidence in each case)

| Patch | Supersession evidence (verified) | Retire actions |
|---|---|---|
| **P7b** | Both anchors dead post-refactor (`get_tensor_model_parallel` 0 hits in target; 12-space pair gone). Superseded by PN204 (`maybe_execute_in_parallel` present in pin) + PN365 (single-GEMM, **applied in one PROD container** — phrase audit-accurately). Extra reason: PN204_FWD_NEW/PN365_FWD_NEW emit P7b's anchor-2 text in fallback branches → resurrection risks sibling-match. | `lifecycle='retired'`, notes citing PN204+PN365 + anchor death, move module to `_archive/`, keep `conflicts_with`. Fix P7 inconsistency (§2). |
| **PN54** | Sub-A native (`qwen_gdn:1504-1514`, gather without `.contiguous()`); sub-B LoRA branch removed (only CPU assert at 1023). Latent self-substring marker bug → feed §6 lint. | Retire per P61/PN9/PN78 convention + `superseded_by` note. Do NOT chase `b/a.contiguous()` at 942-943 (different site, real copy — PN350/PN365 territory). |
| **P78** | Upstream absorbed Sites B/C/D/E: CPU-mirror fields 190-193, build() 237-238, `prefill_max_seq` from `seq_lens_cpu` 486-489, continuation 601-610; buggy GPU `.tolist()` gone from file. | `lifecycle='retired'` + retire_note; drop P78 from `composes_with` at `registry.py:4553` **and** prose in PN353B detail at 4538-4539. |
| **P36** | WorkspaceManager `get_simultaneous` native with identical 3 buffers/shapes/invariant; patched `register_buffer` site no longer exists; boot already skips via pr40798 probe. | Retire; **confirm PR #40798 number via `gh pr view` first** (only non-byte-verified claim); check `POOL_TQ_DECODE_SHARED` has no other consumers before deleting module. |
| **P83** | `use_eagle`→`drop_eagle_block` rename + coordinator lookahead (`kv_cache_coordinator.py:643-650`, 565-571). Re-anchor UNSAFE: skip-pop breaks the monotonic-decrease convergence invariant (L588-593) + `eagle_verified` bookkeeping. Already "empirically disproven" per registry. | Retire; note phrasing: "supersedes the convergence-interaction cost; residual tail-block drop tracked via open #44986". Add `vllm_version_range` cap <0.22.1. |
| **P84** | Both sites native: scheduler `hash_block_size` param; `resolve_kv_cache_block_sizes` (GCD default + `cache_config.hash_block_size` override + divisibility ValueError). | Retire. **Caveat (verifier):** Mamba back-off (kv_cache_utils.py:639-644) precedes both GCD AND the explicit override — verify `mamba_cache_mode='align'` resolution / num_hashes>0 on the prod prefix-caching config before declaring equivalence. Cascade: re-triage P85's `requires_patches=["P84"]`. |
| **P4** | Registry's own overdue plan: vllm#39931 merged 2026-05-05, near-verbatim our `_genesis_p4_full_attention_indices` logic; boot self-skips via marker. | `lifecycle='retired'` + `vllm_version_range` cap; delete module next cleanup batch; regen `docs/PATCHES_AUTO.md`; run pin-gate + iron-rule-11 tests. Hybrid-TQ smoke with P4 OFF first. |
| **P20** | Upstream superset: per-layer `_tq_Pi_half` fp16 cache (`turboquant_attn.py:352, 789-797`) + prealloc `k_full/v_full` eliminating the `torch.cat` transient (better than ours). `implementation_status='marker_only'`, never binds. | Zero-risk retire; update `upstream_compat.py` merged_date from "OPEN (vendor fork only)". |
| **P6** | Upstream merged a corrected superset (`lcm` not `max`). | Retire after the §1 neutralization. |

**Pending decision (NOT yet in retire queue):** **PN200** — anchor now ambiguous (3× pristine / 2× post-P28); PROD-applied P28 owns the CUDA site and delivers the same buffer-reuse+zero. Per the journal corollary, range-capping ≠ retirement while launchers export the flag: either retire properly (lifecycle + remove `GENESIS_ENABLE_PN200_GDN_SCRATCH_REUSE` from `compose/prod-*.yml` + model YAML) or re-anchor as a P28 chain consumer with `conflicts_with`/`requires`. Do not silently re-anchor.

---

## 4. Document-only (no patch changes)

### P64 — KNOWN_DOCUMENTED
Serving-anchor absence is the documented post-retire steady state (journal 2026-06-09, commit 630283ac); parser payload applied at boot (line 39); P107 carries the serving-side role. Optional hygiene: delete the two retired TextPatch entries from `_make_serving_patcher` (keep constants as docs) or allowlist in preflight.

### Env-conditional false positives — patches healthy, builder env-gated before `resolve_vllm_file`
All verified: target exists, anchors byte-exact/unique on pristine where checked; "target not found" detail is a canned lie when `_unresolved_resolve_targets()==[]`.

| Patch | PROD state | Note |
|---|---|---|
| PN96, PN73, PN91, PN92, SNDR_WORKSPACE_001 | **applied at boot** | Healthy; SNDR_WORKSPACE_001 optional: drop the env early-exit from `_make_patcher` (apply() re-gates) |
| PN71 | applied | Registry hygiene in §2 (requires P27) |
| PN97, PN104, PN105, PN202 | skipped-disabled | `<0.21.0` caps deliberate ("honest last-validated record") |
| PN200 | skipped-disabled | Decision queued in §3 |

### TOOL_BUG rows — patch healthy, fix the preflight
- **PN204:** env gate fires before resolve; module's fallback already targets `gdn/qwen_gdn_linear_attn.py`; anchor unique at 920-924; drift markers absent (#42301 unmerged). PROD intentionally OFF (PN365 conflict).
- **P82:** UNBUILDABLE false positive — `threshold` has no default by design; builds via module's own `_read_threshold()/_read_min_draft_pos()`; anchor unique at `rejection_sampler.py:815-817`; PROD-applied at thr=0.3000.
- **P91B:** dual-factory alternation by design (Option A); dev371 anchor matches `inc.py:538`; keep dev338 factory while that pin stays in `KNOWN_GOOD_VLLM_PINS`.

### Consolidated preflight v1.2 fix (one PR, `tools/pin_preflight.py`)
1. Builder-None + specs carry unset `env_flag` → retry with flag forced "1" (try/finally), evaluate normally, tag `env_forced: true`. **Plumb `info["specs"]` into `evaluate_module`** (currently receives only module_name/patch_ids/candidate_root).
2. Retry still None + `_unresolved_resolve_targets()==[]` → new non-actionable `ENV_GATED_ABSTAIN` verdict; keep DRIFT_FILE_MOVED only when targets are actually missing.
3. `_unresolved_resolve_targets`: report resolved-vs-unresolved literals ("resolves via fallback: …").
4. UNBUILDABLE: resolve missing params via module `_read_<param>` convention (P82).
5. Anchor-alternation OR-groups via module manifest → `EXPECTED_ALTERNATE` (P91B).
6. All-`required=False` zero-match patchers → `KNOWN_OPTIONAL_RETIRED` (P64).
7. Unit tests in `tests/unit/tools/test_pin_preflight.py`; then **re-run the full sweep** — the 12 DRIFT_FILE_MOVED rows / 7 disabled-by-env likely hide more of this class.

---

## 5. DISAGREEMENTS (verifier refuted classifier)

### P85 — hybrid fine-shadow prefix cache (only `agree: false` row)
- **Classifier position:** REAL_DRIFT_FIX_ANCHORS; fix Site 1 only (re-anchor `cache_blocks` to upstream's new `retention_interval` signature, pristine 1211-1229); explicitly: "P85_SITE2_OLD/NEW need no change" (Site 2 byte-exact in pristine at 988-1016).
- **Verifier position:** classification correct, **recommendation materially false**. Sibling **PN346** (effectively default-on — only `GENESIS_DISABLE_PN346` is honored; applied at boot line 87, *before* P85 at line 173) rewrites a byte-identical 4-line subsequence inside P85_SITE2_OLD, inserting 12 lines mid-anchor. TextPatcher requires count==1 → the required Site 2 sub-patch fails on **every real (post-PN346) boot**, and pristine MambaManager lacks the `drop_eagle_block` guard so PN346 will apply on this pin too. Mirror of the P18B-on-PN119 class: a Site-1-only fix passes pristine preflight but fails the classifier's own smoke test.
- **Resolution (adopt verifier):** fix BOTH sites — Site 1 as sketched; Site 2 via (a) PN346-aware anchor (coarse fallback must then carry PN346's `drop_eagle_block` guard), (b) reorder P85 before PN346 + verify post-apply uniqueness, or (c) `conflicts_with` + require `GENESIS_DISABLE_PN346=1`. Fix PN346 registry mismatch (§2). **Validation must run full `apply_all` with `GENESIS_ENABLE_P85=1` WITHOUT disabling PN346.** Keep classifier's side-notes (P84 partial-absorption re-check; `cache_full_blocks` block_mask re-validation).

---

## 6. Self-collision TRUE_RISK + binding failures + newly-merged markers

### Self-collision summary — 122/122 findings TRUE_RISK (110 unique pairs, 73 patches, 0 defended)
**Mechanism:** every flagged drift marker provably enters post-apply file content (111 via replacement text, 11 via idempotency-marker substring); stock TextPatcher checks idempotency (Layer 2) before drift (Layer 3) with no `[Genesis` skip for these, and all 66 custom apply() wrappers RE-check the markers. False "upstream_merged" skips fire on: version-stamped marker bumps (e.g. P67 v-stamps, P70 v7.15, P79d v7.46, PN106 v11.3.0), partial `file_cache` divergence (P82 incident class), or a sibling baking the string (**observed PN369/P71 incident**). **Worst case:** PN54's marker is also in the pristine ANCHOR → false-skips PRE-apply (moot after §3 retire). Already-confirmed live instances: PN353A (§1), PN55 (§2).

**Actions:**
1. Ship the lint proposed in PN353A's sketch: **reject any `upstream_drift_marker` that is a substring of the patcher's own replacement text or idempotency-marker line** (registry/preflight check). Would have caught PN353A, PN54, PN55, and the PN118 marker re-planting.
2. Remediation convention for the batch: drift markers disjoint from emitted text (prefix all baked identifiers; upstream-side markers strictly upstream-only), or a general self-emitted-marker registry extending the `[Genesis` skip in Layer 3.
3. Priority: PROD-applied/default-on patches with version-stamped markers (P67×2, P70, P79d, PN106, PN202, PN293, P71, P100 customs), then the long `_genesis_*` tail.
4. Re-run the `/tmp/pn369_triage.py` scan as CI after remediation.

### Binding failures
- **G4_03 — REAL_BREAK (fix with §1b batch):** `from vllm.v1.spec_decode import eagle3` — module gone; Eagle3 lives in `eagle.py` (`EagleProposer`), wrapped method on `SpecDecodeBaseProposer` (`llm_base_proposer.py:1157`). **Silent hazard:** try/except ImportError + log.debug means an enabled G4_03 reports "applied" with wrapped_count=1 and leaves Eagle3 unguarded. Fix: repoint import to `eagle` (cls probe already includes `EagleProposer`), or better wrap `SpecDecodeBaseProposer._create_draft_vllm_config` gated on `method in ("eagle3","dflash")` (rename-proof, covers both). Also investigate the **registry default_on=True vs boot "skipped — disabled"** discrepancy, and re-validate the guard's recommendation text against the new native `Gemma4Proposer` MTP path (`gemma4.py:31`).
- **G4_07 — DORMANT (document-only):** `apply_fp8_block_linear` removed by the kernel-selection refactor (`init_fp8_linear_kernel` + `Fp8BlockScaledMMLinearKernel` subclasses); lazy import breaks only at first forward if opted in. Record in docstring/registry; pre-stage the repoint (direct `w8a8_triton_block_scaled_mm`, or a custom kernel subclass via the new API); add a fail-loud registration-time import probe. **NOT retire** — the upstream double-scale hazard (vllm#39407) persists (`compressed_tensors_w8a8_fp8.py:69-71`).

### Newly-merged markers (`sndr/engines/vllm/upstream_compat.py`)
- **PR_39953 (P16):** both halves native (int64 casts in TQ decode/store kernels; FA2 forcing at `arg_utils.py:2111-2121`); P16 doesn't exist anywhere in the live system. Mark verified-merged-in-pin; fix stale `affects_patch` → "none — P16 already removed".
- **PR_38479 (TQ substrate):** tracks upstream TurboQuant itself, not an overlap — set `already_known_merged` (merged in v0.20.0) so it stops re-triggering. **P3 stays** (anchor byte-matches pristine; e4b15 staircase still absent upstream — required on SM86). P22's case-(b) re-hook investigation stands.
- **PR_JARTX_11 (P20):** retire → §3; update merged_date.
- **PR_39931:** P4 retire → §3 (smoke with P4 OFF first). P6 → §1 neutralize + §3 retire. **P5/P5b keep-extras + fix Probe 1:** `hasattr(...turboquant.config, 'TQFullAttentionSpec')` is always False — the symbol lives in `v1/kv_cache_interface.py:327` — so the intended #39931 auto-skip never fires; fix the module path (or key on `_get_full_attention_layer_indices`) and make defer-vs-keep explicit. P9: no action (already retired). Update merged_date to 2026-05-05.

---

### Suggested execution order
1. §1 PN353A + P6 neutralize (pre-pin-live gate) → 2. §4 preflight v1.2 fix + re-sweep (flushes false positives, may surface more) → 3. §1b fix-drifts batch (PN32 w/ PN79 composition, P59, PN58, PN288 w/ P107, PN38) + §5 P85 both-sites fix → 4. §3 retire queue + §2 hygiene (one commit each, journal entries per iron rule #11) → 5. §6 self-collision lint batch + G4_03 repoint → final preflight sweep must show the 14 DRIFT_ANCHOR / 12 DRIFT_FILE_MOVED residual rows cleared or reclassified.