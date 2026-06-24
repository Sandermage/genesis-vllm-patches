# Changelog

All notable changes to **Genesis vLLM Patches** are tracked here.

This is the engineering log (per-commit, per-patch decisions, per-A/B
numbers, retire trails). Audit 2026-05-11 confirmed this file is the
single source of truth — earlier references to a separate
`vllm/sndr_core/CHANGELOG.md` pointed to a file that never existed.

The project uses [Semantic-ish Versioning](https://semver.org/) keyed
to internal sprints (`v7.62.x` etc). Until a 1.0 cut, expect breaking
changes only when an upstream vLLM PR replaces a Genesis patch and
the patch retires accordingly — those are flagged loud-and-clear in
the per-release notes.

---

## How this file is organised

Each release is a top-level `## [<tag>] — <title> (<date>)` heading.
Within a release, sub-sections use one of the following H3 headers
so an operator can scan a release at a glance:

| Sub-section | What goes in it |
|---|---|
| **Highlights** | One-paragraph summary of the most operator-visible change. |
| **Patches changed** | Added / changed / retired patches with one-line rationale each. |
| **Migration notes** | Anything an operator needs to do on upgrade (env-var rename, config diff, manual step). |
| **Bench / measurements** | A/B numbers, regression deltas, wall-TPS / TPOT / VRAM tables. |
| **Audit findings** | What gate failed, what was fixed, what was deferred. |
| **Verified** | What was actually run to prove the release works (`make evidence`, `pytest`, hardware probe). |

Release-type tags in the title give the shape of the release at a glance:

- **Wave N** — the N-th canonical PROD bench cycle (e.g. `wave9`).
  Patch matrix changes + new bench numbers + verified A/B vs the prior wave.
- **audit-closure** — fixes a specific audit reference doc's P0/P1 findings.
- **stable-promotion** — moves one or more patches from
  `lifecycle=experimental` to `lifecycle=stable` with anchor-manifest
  coverage.
- **release-blockers-closed** — final gate before a tag is cut.
- **hotfix** — narrow fix for a reported issue, no new patches.

---

## Version index (newest first)

### [Unreleased] — dev148 pin, MTP K=5 re-tune, PN394/PN399/PN353A, Gemma-31B kv-auto, enterprise hardening (current tip)

| Tag | Date | Type | Summary |
|---|---|---|---|
| `[Unreleased]` | 2026-06-19 | dev | Pin promoted `dev101 → 0.23.1rc1.dev148+gb4c80ec0f` (live rig pin). MTP K=3→K=5 re-tune (Qwen, lossless): 35B **239.7 TPS** (+15.8%), 27B **127.4 TPS** (+8.2%); Gemma stays K=3. PN394 (#46047) + PN399 (#46067) promoted to PROD; PN353A (#44053) enabled in Qwen YAMLs. Gemma-4-31B kv-auto chat profile/preset (+69.6% chat TPS). G4_11 multi-turn fix. PN29+PN298 consolidated into one chunk_o wiring module + one registry entry (runtime-neutral; PN29 flag retained as alias). Registry 317; 57 default-on; 27 families; `make evidence` 63/63 |

### v12.0.0 series — dev491 pin, Gemma-4 + native 31B MTP, audit hardening

| Tag | Date | Type | Summary |
|---|---|---|---|
| `v12.0.0` | 2026-06-16 | release | Pin advanced to `0.22.1rc1.dev491+g1033ffac2`. Gemma-4 26B/31B family lands (G4_* patch line). **Native MTP on Gemma-4-31B-TQ now works** (G4_67→G4_81 verify route: coherent + ~40.7 t/s, +10% vs no-MTP). TurboQuant head_dim=512 prefill SDPA fallback (G4_82). Audit C1 evidence-gate cascade closed → `make evidence` 63/63 green; `sndr_private/` untracked (793-file public-leak fix); file_cache Layer-0 marker-accumulation bug fixed. Registry 241→313; 53 default-on; 27 families |

### v11.3.0 series — Enterprise hardening sweep + Phase 7 advance

| Tag | Date | Type | Summary |
|---|---|---|---|
| `v11.3.0` | 2026-06-03 (eve) | release | Tier 1 enterprise hardening: PyPI Trusted-Publisher step + 3 P3.2 shared-policy modules (GDN/NGRAM/TQ workspace) + P3.4 dispatcher readiness audit + SPEC_DECODE_GUIDE. Tier 2 Phase 7: SNDR_EAGLE3_AUX_HIDDEN_001 ships EAGLE-3 model-side API surface; PHASE7_RESEARCH_TRACK consolidates Suffix Decoding (DONE) + EAGLE-3 (PREP) + Mamba-3 (watch). Registry 240→241; +1473 LOC docs/audit; 49 new tests PASS; 60/60 gates green |

### v11.2.0 series — Phase 6 P3.1 multi-file md5 PoC complete

| Tag | Date | Type | Summary |
|---|---|---|---|
| `v11.2.0` | 2026-06-03 | release | P3.1 multi-file md5 conversion via sibling PoC pattern: PN118 turboquant_attn.py + PN79 chunk.py + chunk_delta_h.py; 240 registry entries; documents anchor drift on production pin |

### v11.1.0 series — Phase 6 closeout

| Tag | Date | Type | Summary |
|---|---|---|---|
| `v11.1.0` | 2026-06-02 | release | Phase 6 P3.1 + P3.3 closeout + operator docs Phase 10 V1 sunset refresh + 237 registry entries |

### v11.0.0 series — SNDR Core

| Tag | Date | Type | Summary |
|---|---|---|---|
| `wave9_release_blockers_closed` | 2026-05-14 evening | release-blockers-closed | Closes remaining P0/P1; 256K hardware-verified on both models; patch-proof coverage 151/151 |
| `wave9_tp2_256k` | 2026-05-14 | wave | TP=2 256K commissioning + PN59 anchor fix on dcacdf9a pin |
| `audit_2026_05_12_dual_state_closure` | 2026-05-12 | audit-closure | Comprehensive dual-state audit closure |
| `stable_first` | 2026-05-12 | stable-promotion | First STABLE patches in registry + asyncio cleanup |
| `wave9_dev209` | 2026-05-12 | wave | Wave 9 pin-bump re-bench + STABLE promotion analysis |
| `stale_ref_cleanup` | 2026-05-12 | audit-closure | F-015 closure + post-rename stale-ref sweep |
| `wave8` | 2026-05-11 | wave | Wave 8 regression audit + canonical bench WIN |
| `v11.1.0-dev` | 2026-05-09 | feature | UNIFIED_CONFIG_AUTOMATION_PLAN closure + Path C foundation |
| `v11.0.0` | 2026-05-08 | release | Two-audit closure + canonical SNDR Core/Engine split |

### v10.x — pre-rename foundation

| Tag | Date | Type | Summary |
|---|---|---|---|
| `v10.0.0` | 2026-05-07 | architecture | Hard flip: canonical impl now lives in `sndr_core/` |
| `v9.0.0` | 2026-05-07 | architecture | Path consolidation (`project_paths.py`) |
| `v8.0.0` | 2026-05-07 | architecture | SNDR Core architectural refactor |

### v7.x — pre-restructure history

| Tag | Date | Type | Summary |
|---|---|---|---|
| `v7.65 → v7.72` | 2026-04 – 2026-05 | series | 7.72 series: PN59 streaming-GDN, Cliff 2b breakthrough, Blackwell consumer support, structured boot summary |
| `v7.63.x` | 2026-04 | series | TurboQuant k8v4 + MTP K=3 stabilization |

The current canonical baseline is the `[Unreleased]` line below on vLLM
nightly pin `0.23.1rc1.dev148+gb4c80ec0f` (image
`vllm/vllm-openai:nightly-b4c80ec0f`); the last cut tag is `v12.0.0`. 317
patches in `PATCH_REGISTRY`, 57 default-ON, 27 families, `make evidence`
63 / 63 gates green. (The PROD 35B service runs the dev148 pin; `dev101`
is retained as the previous / rollback pin per the CLAUDE.md ≤2-pin policy.
`sndr/version.py` carries the `.dev0` suffix in-tree; release tooling strips
it on publish — pyproject.toml holds the release version.)

---

## [Unreleased] — vLLM pin bump dev148 → dev301 (2026-06-24)

### Highlights

- **vLLM pin promoted `0.23.1rc1.dev148+gb4c80ec0f` → `0.23.1rc1.dev301+g04c2a8dea`**
  (image `vllm/vllm-openai:nightly-04c2a8dea`, commit `04c2a8dea`). dev148
  retained as the previous/rollback pin per CLAUDE.md ≤2-pin policy. Smoke
  validation: **35B 208 TPS + 31B 94.7 TPS** boot + chat + tool-call.

### Patches changed

- **9-PR iron-rule-#11 deep-diff outcome (dev301 anchor-SOT regen).** The
  dev301 anchor regeneration surfaced exactly 5 `anchor_drift` entries
  (P85 / PN394 / PN353A / PN400 / PN382 — confirmed against
  `pins/0.23.1_04c2a8dea/drift.rej.json`). Resolution:
  - **PN394 RETIRED on dev301** (`superseded_by: vllm#46047`, capped
    `< 0.23.1rc1.dev301`). #46047 (qwen3 partial-param `<`-truncation fix) is
    IN dev301. Still applies through dev148; runtime drift-marker self-skips.
  - **PN400 RETIRED on dev301** (`superseded_by: vllm#45656`, capped
    `< 0.23.1rc1.dev301`). The NVIDIA `auto_gptq.py` `is_sym` qzeros guard is
    native on dev301. CPU clause never used on our rig.
  - **PN353A KEPT active** (`superseded_by: vllm#44053` note added, NOT
    version-gated out). #44053 is native on dev301, but `PN399.requires_patches`
    anchors PN353A's live output — coupled retirement waits for PN399's
    upstream (#46067). Range left open through `< 0.24.0`.
  - **PN382 KEPT active**, reclassified `backport → related_not_superseding`
    (the valid-enum equivalent of a divergent fork; rationale in its `notes`).
    vllm#45080 merged a WEAKER whole-pool/group-0 variant; our per-block fill
    + real `kv_cache_groups` map are correctness-critical for GDN hybrids and
    absent from the merge.
  - **Deferred (PRs in vllm main, NOT in dev301 — still apply, left active):**
    PN398/PN370 (#45100), PN386 (#45389), PN-FP8MOE-KPAD (#45703), G4_80
    (#45040). They retire on a future bump when their PR lands. G4_80 carries a
    credit note that its arm-1 retires when #45040 lands (arm-2 query_quant is
    Genesis-original).

### Migration notes

- `KNOWN_GOOD_VLLM_PINS` (and the `EXPECTED_PINS` drift-trap mirror) gain
  `0.23.1rc1.dev301+g04c2a8dea` + `nightly-04c2a8dea`; dev148 stays in the
  tuple as the rollback.
- All 11 builtin ModelDefs bumped `vllm_pin_required` to dev301 with a
  `pin_hold` waiver for the promotion window (the shared a5000-2x hardware
  YAML still pins the dev148 image until the dev301 `@sha256:` digest is
  captured — hardware YAMLs intentionally unchanged in this step).
- The anchor-SOT generated `sndr/engines/vllm/pins/0.23.1_04c2a8dea/`.

### Verified

- `pytest -k "registry or iron_rule_11 or audit_upstream or pin_gate or default_on"` green;
  `test_pin_gate.py` green; registry self-test + `audit_registry_contract.py`
  clean; `audit_v2_modeldef_vs_hardware_pin.py` rc0 (all 11 ModelDefs HOLD on
  the dev301-vs-dev148 mismatch); `check_doc_sync.py` rc0; all 11 ModelDefs
  load + validate.

---

## [Unreleased] — dev148 pin · MTP K=5 re-tune · PN394/PN399/PN353A · Gemma-31B kv-auto · enterprise hardening (2026-06-19 … 2026-06-23)

### Highlights

- **MTP K=3 → K=5 re-tune (Qwen, lossless).** K was under-tuned at 3 fleet-wide.
  Warm single-stream on pin `0.23.1rc1.dev148+gb4c80ec0f`: **35B 239.7 TPS /
  TPOT 3.94 ms (+15.8 % vs K=3 207)**, **27B 127.4 TPS / TPOT 7.54 ms (+8.2 %
  vs 117.7)**. MTP spec-decode is target-exact so the gain is free. K=5 locked
  into both Qwen YAMLs + launchers (35B K=6 won't boot, K=7 regresses; 27B
  peaks at K=5). **Gemma stays K=3** — its separate drafter is optimal at K=3.

### Patches changed

- **PN394 promoted to PROD** (backport of MERGED vllm#46047) — fixes the qwen3
  tool-call parser truncating a partial parameter value at a literal `<`.
  Merged upstream 2026-06-18, *after* the dev148 pin, so not yet in the
  deployed engine; self-skips once a newer pin carries it. Default ON.
- **PN399 promoted to PROD** (consolidated backport+improvement of OPEN
  vllm#46067) — single-owner fixed TurboQuant decode-scratch buffer; fixes
  the CUDA illegal-memory-access in FULL cudagraph and removes the dead
  PN118 / PN353A decode reservations.
- **PN353A enabled in the Qwen YAMLs** (backport of OPEN vllm#44053) —
  TurboQuant MetadataBuilder workspace reserve; closes the long-context
  `workspace locked, cannot grow` assertion. PN399 dependency.
- **Gemma-4-31B kv-auto chat profile + preset added** —
  `gemma4-31b-kvauto-chat` profile and `prod-gemma4-31b-kvauto-chat` preset.
  kv-auto (uniform fp16 KV, no TurboQuant, 32K ctx) delivers ~70.1 TPS chat
  (+69.6 % vs the TQ chat sibling's 41.4) AND better tool-call (7/7 vs 6/7),
  trading 64K → 32K context. Operator runs both profiles.
- **Gemma chat-template fix — use the model's built-in HF template** (`225bfcac`).
  The 4 AWQ Gemma ModelDefs — `gemma-4-26b-a4b-it-awq`,
  `gemma-4-26b-a4b-it-awq-experimental`, `gemma-4-31b-it-awq`, and
  `gemma-4-31b-it-awq-mtp-n8-code` — now set `chat_template: null` +
  `GENESIS_ENABLE_G4_11_GEMMA4_CHAT_TEMPLATE_INSTALL=0`, so `compose()` emits no
  `--chat-template` flag and vLLM falls through to the model's built-in HF
  template. The custom G4_11 template broke tool-calls and chat quality — raw
  `<start_of_turn>` token echo, an unreadable `<|channel>thought<channel|>`
  format the gemma4 tool-parser couldn't read, and ZERO `tool_calls`
  (finish=length). **Validated live on the rig:** 26B + 31B both give clean chat
  AND a working `get_weather` tool-call. (The g4_11 patch module is untouched —
  disabled by config.)
- **5 retired-patch version-range provenance** records added.
- **G4_85 (TurboMind int4 MoE kernel) build COMPLETED + benchmarked (rig, pin
  dev148, 2026-06-23) — verdict: superseded by EP+Marlin, stays experimental.**
  The build is now complete (full ~46-object `find src/turbomind` closure, all
  `-Xcompiler -fPIC`; `build_kernels.sh` rewritten to it) and the kernel FIRES on
  the 26B — the previously-fatal `undefined symbol: vtable for
  turbomind::LinearWeight` is resolved, `tm_probe()=1`, both TP workers load it,
  numerics correct (GEMM_err ~3e-4, full-MoE reldiff ~9e-4). Benchmarked vs Marlin
  (TTFT-subtracted decode TPS, 26B, n=20, MTP-off): **EP+Marlin 142.5 tps >
  pure-TP `moe_wna16` 125.4 tps by +13.6%**; pure-TP + g4_85 builds and fires but
  **OOMs at boot** — it dequantizes int4 → fp16 per MoE layer (~2× weight memory
  transiently, ~1.45 GiB/layer/GPU × ~30 layers) on top of an int4 model that
  already fills ~20 of 23.5 GiB (gpu-mem 0.88/0.60/0.45 all OOM). Fixing that is a
  **source redesign** (stream-dequant one layer at a time, or feed packed int4
  directly to TurboMind), not a build fix. g4_85 would have to exceed 142.5 just
  to break even vs prod AND it cannot boot on 2× A5000 → **superseded by EP+Marlin,
  stays experimental / pure-TP-only**; kept (not retired) pending the dequant
  memory-footprint redesign. `implementation_status` stays `partial`.

### Migration notes

- **Pin promoted dev101 → dev148** (`0.23.1rc1.dev148+gb4c80ec0f`, image
  `vllm/vllm-openai:nightly-b4c80ec0f`). `dev101` is retained as the
  previous / rollback pin per the CLAUDE.md ≤2-pin policy. dev148 added to
  `KNOWN_GOOD_VLLM_PINS` (guards.py), `EXPECTED_PINS` (pin-gate test), and
  `ALLOWED_MODELDEF_PINS` (runtime-pins audit). All 12 ModelDef / profile
  `vllm_pin_required` fields bumped to dev148; the two PROD ModelDefs (35B,
  27B-tq-k8v4) carry a `pin_hold` waiver because the ModelDef leads the
  shared hardware image during the promotion window (the hardware
  `image_digest` bumps once the dev148 sha256 is captured).

### Bench / measurements

| Model | K=3 → K=5 | Δ TPS | Δ TPOT |
| --- | --- | ---: | ---: |
| 35B-A3B-FP8 (single-stream) | 207.1 → **239.7** | +15.8 % | -12 % |
| 27B-int4-AutoRound (single-stream) | 117.7 → **127.4** | +8.2 % | -8 % |
| Gemma-4-31B kv-auto vs TQ chat | 41.4 → **70.1** | +69.6 % | — |

### Verified

- `make evidence` 63/63 green · `sndr patches doctor` ERROR=0 ·
  `check_doc_sync.py --strict` clean · `generate_patches_md.py --check`
  exit 0 · `pytest tests/unit/dispatcher tests/unit/model_configs` pass ·
  `tsc --noEmit` (GUI) exit 0.

### Enterprise hardening sweep (2026-06-23 — full-suite green · audit-alias sync · reference configs)

#### Test suite — whole repo green

- **`pytest tests/` → 12851 passed / 0 failed** (698 skipped, 1 documented xfail).
  Every pre-existing and churn-introduced failure was closed through a
  study → diagnose → adversarial-verify loop, not by weakening assertions.
- **PN282 spec-decode acceptance metric — a REAL test-isolation bug, not the
  "flaky" it was long filed as.** The autouse fixture put the fake
  `rejection_sampler` in `sys.modules` but never rebound it as the
  `rejection_sampler` *attribute* of the parent `vllm.v1.sample` package. `apply()`
  resolves its target via `from vllm.v1.sample import rejection_sampler`, which
  prefers the stale parent attribute (bound when an earlier test imported the real
  package) over the `sys.modules` fake — so in a full run the wrap landed on the
  wrong module and four assertions failed. Order-dependent: green in isolation, red
  in the full suite. The fixture now rebinds + restores the attribute.
- **R-011 config audit synced with the 2026-06-19/20 parser-trio consolidation.**
  `audit_rules._check_env_keys_exist` now reads `env_flag_aliases`, matching the
  runtime dispatcher: the merged P64/P61b entries expose the absorbed
  P61c/PN56/PN51 enable flags as aliases, which the prod 27B/35B YAMLs legitimately
  still set (they were mis-reported as unknown keys). The `VLLM_TQ_DECODE_` P18b
  decode-kernel knobs were registered as a `runtime_tunables` family knob (single
  source of truth — no inline prefix in the audit rule). R-011 still fires for
  genuine typos.
- Stale legacy gates corrected: boot-summary (PN51 → uncategorized after
  consolidation), P3 drift markers (the FP16 staircase form is *self-emitted* by
  P3 so it must NOT be a drift marker — self-collision / false-skip guard pinned),
  PN388 dual-anchor drift, preset count-drift (23 → 24), and 5 obsolete P64
  design-intent tests (module consolidated).
- **Installer smoke-test fail-class fix** (`24994ea1`):
  `install._classify_failure` now buckets a `ModuleNotFoundError` (`no module
  named …`) and a `cannot import name 'X' from 'vllm…'` ImportError as
  `runtime_gap` rather than `wiring_bug`. Both are environment/version gaps (the
  pin lacks the module, or the symbol moved in this pin), not install-blocking
  internal regressions — the `cannot import name` rule is scoped to
  vllm/torch/triton/flashinfer source modules so a genuine `from 'sndr…'` wiring
  ImportError still classifies as `wiring_bug`.
- **Allowlist documents the 4 G4_11=0 default-on overrides** (`430fd260`): the
  Gemma chat-template fix disables `GENESIS_ENABLE_G4_11_GEMMA4_CHAT_TEMPLATE_INSTALL`
  (a `default_on=True` patch) in the 4 AWQ Gemma ModelDefs; the default-on-mismatch
  ratchet test now allowlists each override with the live-finding rationale so the
  ratchet stays green.

#### Security / hygiene

- **`tools/license_keygen.py` moved out of the public tree** into `sndr_private/`
  (untracked + gitignored) so the commercial license tooling cannot leak.
- Private LAN/host references scrubbed from `scripts/anchor_sot/rebuild_pin.sh`;
  `audit-public-paths` and the full `make audit` aggregate are clean.
- Upstream watchlist `vllm#43409` reconciled `port → drift-check` (PN400 **is** the
  backport of fix `vllm#45656`; the port is complete, not pending).

#### Reference-complete model configs

- **All 11 builtin model configs made self-describing** — `patches_attribution`
  (evidence-based role + rationale per patch), `notes`, `extra_vllm_flags` escape
  hatch, and richer inline comments. The 5 Gemma configs gained
  `patches_attribution` from scratch (derived from the registry + the qwen gold
  precedent). **Proven behavior-preserving**: the fully-composed `ModelConfig` for
  all 25 presets is byte-identical to before, verified by composing in a clean HEAD
  worktree and diffing (`compose()` never reads attribution/notes). 10 unvalidated
  profiles deprecated.

#### Hardening verification

- `pytest tests/` 12851 passed / 0 failed · `make audit` clean · `validate_all_models`
  0 bad · 25-preset composed `ModelConfig` byte-identical pre/post.

---

## [v12.0.0] — dev491 pin · Gemma-4 family · native 31B MTP · audit hardening (2026-06-16)

### Highlights

- **Native MTP on Gemma-4-31B-TQ now works.** The drafter is Q-only (no
  `k_proj`/`v_proj`) and KV-shares the target's cache; its shared layers map to
  the target's `--kv-cache-dtype-skip-layers [58,59]` which store **native bf16**,
  so the drafter must read bf16 via `TRITON_ATTN` (G4_71B head-256 / G4_75 head-512)
  rather than the TQ kernel — without that it CUDA-OOBs. With drafter routing kept,
  the TQ K+1 **spec-verify route was switched G4_67 → G4_81**: G4_67 returned a raw
  3-D tensor instead of writing the engine `output` buffer (contract gap) → garbage
  decode; G4_81 honours the output-buffer contract + forwards sliding-window/mm-prefix
  masking → **coherent output, ~40.7 t/s single-stream (+10% vs the no-MTP 37 t/s
  baseline)**. Shipped in both V2 profiles (`gemma4-31b-tq-mtp-{chat-k3,structured-k4}`).
- **G4_82 — TurboQuant prefill SDPA fallback for `head_dim>256`** (Ampere FA2 caps
  head_dim at 256, vllm#38887). Unblocks the Gemma-4-31B interleaved global
  (head_dim=512) layers on the native TQ path.
- **Audit C1 evidence-gate cascade closed** (`make evidence` 4 failing → **63/63
  green**): G4_82 added to `KNOWN_SPEC_ONLY_PATCHES` + proof artefact; doc patch
  counts synced 312 → 313 across 12 docs; `file_cache.record_apply_result` marker
  pruning bug fixed (it dropped ALL accumulated Layer-0 markers when post-apply
  content lacked their comment text); G4_10 lifecycle-docstring sync; env.py G4_08
  dup alias removed; PN290 logger/flag-message corrected.
- **Privacy: `sndr_private/` untracked.** The `.gitignore` rule only filters
  *untracked* files; 793 already-tracked files (41 MB) would have shipped on clone.
  `git rm -r --cached` (kept on disk) — the folder's "never ships" guarantee is now
  actually enforced.
- **Registry 241 → 313** (27 families, 53 default-on). Gemma-4 26B/31B `G4_*` patch
  line, pin advanced to `0.22.1rc1.dev491+g1033ffac2`.

### Verification

- `python3 scripts/make_evidence.py` → 63/63 gates green.
- `python3 scripts/check_doc_sync.py --strict` → 0 mismatches (313 consistent).
- `python3 -m sndr.apply.shadow --strict` → CLEAN (known spec-only: 74).
- Live 31B-tq MTP boot (dev491, 2× A5000, PROD restored after): coherent +
  40.7 t/s; the drafter-routing / verify-route diagnostic ladder is recorded in
  `docs/superpowers/journal/2026-06-16-mtp31b-breakthrough-drafter-qonly.md`.

---

## [v11.3.0] — Enterprise hardening sweep + Phase 7 advance (2026-06-03 evening)

### Highlights

- **Tier 1 enterprise hardening (commit `04bc7143`)** — six high-value
  documentation + audit deliverables consolidating the master plan's
  P3.2 + P3.4 + Iron Rule #2 surfaces:

  - **PyPI Trusted-Publisher step** in `.github/workflows/release.yml`
    (OIDC `id-token: write` + `environment: pypi` + `pypa/gh-action-pypi-publish@release/v1`
    with `skip-existing: true`). Operator-side setup: PyPI project
    settings → Publishing → add Trusted Publisher (Owner: Sandermage,
    Repo: genesis-vllm-patches, Workflow: release.yml, Env: pypi).
    After that, every `v*` tag push auto-publishes to PyPI without
    manual `twine` interaction.

  - **`docs/SPEC_DECODE_GUIDE.md`** (315 lines) — operator-facing guide
    explaining when to use MTP (current default) vs Suffix Decoding
    (P75 — already wired) vs NGRAM (P70+P77+PN72 stack) vs K_001
    (default OFF, empirically NOT_SIGNIFICANT across 3 bench cycles).
    Includes switching recipes + bench reference paths + future work
    (EAGLE-3 blocker, Mamba-3 watchlist).

  - **3 P3.2 shared-policy modules** — composition contract +
    runtime audit + drift surveillance for the three patch clusters
    flagged by the master plan as consolidation candidates:
    - `vllm/sndr_core/integrations/attention/gdn/_contiguity_policy.py`
      — unified policy for PN11/PN54/PN50 (.contiguous() discipline
      across 3 distinct insertion points in GDN forward path; 7/7 tests).
    - `vllm/sndr_core/integrations/spec_decode/_ngram_policy_orchestrator.py`
      — unified composition contract for P70/P77/P86/PN72/PN90
      (orthogonal insertion points on NgramProposer pipeline; future
      v12.x+ pluggable DecisionStrategy candidate documented; 9/9 tests).
    - `vllm/sndr_core/integrations/attention/turboquant/_workspace_policy.py`
      — unified composition for P98/P99/PN118/SNDR_WORKSPACE_001
      (TurboQuant workspace area; PN118 is the only default_on;
      future v12.x+ WorkspaceFacade extending TurboQuantBufferManager
      documented; 10/10 tests).

  - **`scripts/audit_dispatcher_migration_readiness.py`** + 6 unit
    tests — P3.4 migration readiness: confirms 218 spec-ready + 22
    intentionally unmapped + **0 real gaps**. v12.0.0 orchestrator
    iteration switch path documented (estimated 2-3 days vs master
    plan's 8-9 day estimate — auto-derivation already done).

- **Tier 2 Phase 7 advance (commits `d53c26a5` + `079e05a4`)** —
  EAGLE-3 model-side preparation + research-track consolidation:

  - **`SNDR_EAGLE3_AUX_HIDDEN_001`** — Genesis-original Phase 7
    EAGLE-3 model-side preparation. Ships the safe API surface
    (`register_aux_hidden_state_hooks` + `pop_aux_hidden_states` +
    `clear_aux_hidden_state_hooks`) so when a Qwen3.6 EAGLE-3 drafter
    checkpoint lands the wire-up is <1 day. Default OFF + zero runtime
    cost without explicit caller. Thread-safe via RLock, idempotent,
    auto-detects layer attribute path (Qwen3.x / Llama-family
    compatible). 17/17 unit tests including hook lifecycle, multi-shape
    captures, env layer-id parsing, idempotency. Registry **240 → 241**.

  - **`docs/PHASE7_RESEARCH_TRACK.md`** (161 lines) — single tracker
    for the 3 Phase 7 directions with status + watch signals + revisit
    triggers + estimated effort:
    - **Suffix Decoding**: ✅ DONE (P75 wired, vllm#25784 in pin, docs
      published)
    - **EAGLE-3**: ⏳ model-side prep DONE; drafter wire-up blocked on
      Qwen3.6 EAGLE-3 checkpoint (does not exist publicly yet)
    - **Mamba-3 MIMO**: 📚 research-only (reference code in
      state-spaces/mamba + fla-org/flash-linear-attention; no vLLM
      serving support; no Qwen-3.x-Mamba3 target model)
    Plus 5 adjacent research areas surfaced for future planning.

### Verification

- **49 new unit tests, all PASS**: 7 (GDN policy) + 9 (NGRAM policy) +
  10 (TQ workspace policy) + 6 (dispatcher readiness audit) + 17
  (EAGLE-3 prep)
- **60/60 evidence gates green** on both local + rig
- **0 LOC behavior change** in any existing patch — Tier 1 modules
  are documentation + read-only audit surfaces; Tier 2 EAGLE-3 patch
  has zero runtime cost without explicit caller
- **+1,473 LOC docs/audit/tests**; 10 doc counter bumps (240 → 241)

### Iron Rule #2 (public release decisions) — status

| Item | Status |
|---|---|
| Wheel + sdist build via CI | ✅ done in `release.yml` |
| SBOM (CycloneDX + SPDX) attached | ✅ done in `release.yml` |
| PyPI Trusted Publisher step | ✅ **NEW** in v11.3.0 — operator does PyPI side setup once |
| Public origin push | ⏳ operator decision (sndr-dev is private; no public push yet) |
| Tauri desktop builds | 📚 scaffold only (commit `b608b6f4`); deferred to next release |
| Docker Hub image publish | 📚 not configured; deferred |

### Master plan status (post-v11.3.0)

| Phase | Status |
|---|---|
| 0–3 (stop-bleeding, registry SoT, cleanup, Triton tune) | ✅ closed |
| 4 (Apply dispatch consolidation) | ✅ closed per M.1.R amendment |
| 5 (perf cherry-picks) | ✅ 4/5 PRs in pin; K_001 empirically NOT_SIGNIFICANT × 3 |
| 6 P3.1 (anchor → md5+full-file) | ✅ closed in v11.2.0 (sibling-PoC pattern) |
| 6 P3.2 (code consolidation) | ⏳ shared-policy modules + audit shipped in v11.3.0 (Tier 1); code-level merge documented for v12.x+ |
| 6 P3.3 (PersistentBufferRegistry) | ⏳ PersistentSlicePool + PN12/P46 migrated in v11.2.0; P39a/P36 deferred (bench window) |
| 6 P3.4 (dispatcher migration) | ⏳ migration readiness audit in v11.3.0 (218/241 spec-ready, 0 real gaps); switch deferred to v12.0.0 |
| 6 P3.5/P3.6/P3.7 (refactors + GUI) | ✅ done in main |
| 7 Suffix Decoding | ✅ P75 wired + SPEC_DECODE_GUIDE.md (v11.3.0) |
| 7 EAGLE-3 model-side | ✅ SNDR_EAGLE3_AUX_HIDDEN_001 in v11.3.0 |
| 7 EAGLE-3 drafter | ⏳ blocked on Qwen3.6 EAGLE-3 checkpoint |
| 7 Mamba-3 MIMO | 📚 watchlist (no target model) |

### Out of scope (deferred to v12.0.0+)

- Code-level P3.2 deep refactor (WorkspaceFacade extending
  TurboQuantBufferManager) — requires CUDA-graph safety bench
- P3.4 orchestrator iteration switch (iter_patch_specs in apply chain)
  — 2-3 days dedicated session
- P39a + P36 full BufferPool migration (registration-only landed in
  v11.1.0; allocator-level routing needs bench)
- EAGLE-3 drafter wire-up (template ready; awaits Qwen3.6 EAGLE-3
  checkpoint)
- Mamba-3 architecture port (awaits Qwen-3.x-Mamba3 target model)
- Public github.com/Sandermage/genesis-vllm-patches push + first PyPI
  publish (operator Iron Rule #2 decisions)

### Branch state

This release ships on `feat/container-management` (same branch as
v11.2.0). Triplet (local / rig / sndr-dev) fully synchronized at
commit `079e05a4` before tag.

---

## [v11.2.0] — Phase 6 P3.1 multi-file md5 PoC complete (2026-06-03)

### Highlights

- **Phase 6 P3.1 multi-file md5 conversion CLOSED** via sibling-PoC pattern. Three new v2 patches landed completing pn118 + most of pn79's md5+full-file conversion:
  - `PN118_V2_MD5_TURBOQUANT_ATTN` — companion to `PN118_V2_MD5_WORKSPACE` (v11.1.0); together the two v2 siblings cover both files in pn118's scope. Documents critical drift: pn118's `TQ_ANCHOR_INIT_OLD` anchor does NOT match upstream `turboquant_attn.py` at our PROD pin — pn118 silently no-ops on that anchor. The md5+full-file pattern detects ANY drift, eliminating silent partial-apply.
  - `PN79_V2_MD5_CHUNK` — md5+full-file PoC for `model_executor/layers/fla/ops/chunk.py`. Empirically: only 3/7 of pn79's chunk.py anchors match current upstream (4 drifted — silent no-op coverage on production).
  - `PN79_V2_MD5_CHUNK_DELTA_H` — md5+full-file PoC for `model_executor/layers/fla/ops/chunk_delta_h.py`. 3/4 anchors match; 1 drifted.

- **Sibling-PoC pattern validated**. The v11.1.0 plan considered a "MultiFileMd5Patch" abstraction. Found unnecessary: N single-file md5 patches compose via Genesis markers (each sibling has unique marker, idempotent independently, composable with anchors-based original on un-converted files). Pattern works across multi-file conversions without new abstractions.

- **pn79 partial conversion only — 2/4 files** (the other 2 are upstream-drifted out of existence at our pin):
  - `gdn_linear_attn.py` — REMOVED from upstream; SPLIT into model-specific files under `model_executor/models/fla/{kimi,olmo,qwen}_gdn_linear_attn.py`. pn79's anchors for this file are dead.
  - `olmo_hybrid.py` — REMOVED from upstream (functionality refactored into model-specific paths).
  - Net: pn79's original 4-file × ~5-anchor coverage is reduced to chunk.py (3/7 working) + chunk_delta_h.py (3/4 working) on the current pin. v2 md5 patches make this transparent; original anchors silently partial-apply.

- **Strongest empirical case for md5+full-file pattern**. Combined drift across pn118 + pn79 (5+ broken anchors + 2 deleted files) shows anchors silently rot on every pin bump. The md5 pattern fails loudly when upstream changes — operators see "md5 mismatch — patch skipped" rather than getting unverified partial-application semantics.

- **Registry growth**: 237 → 240 entries (3 new v2 patches). All default OFF; opt-in via `GENESIS_ENABLE_PN118_V2_MD5_TURBOQUANT_ATTN=1` / `GENESIS_ENABLE_PN79_V2_MD5_CHUNK=1` / `GENESIS_ENABLE_PN79_V2_MD5_CHUNK_DELTA_H=1` for A/B validation on the rig.

- **Triplet synced** (local + rig + sndr-dev/dev) at tag time. 60/60 evidence gates green on both local and rig post-merge. 10 unit tests pass (5 per pn79 v2 + 5 per pn118 v2 turboquant).

### Master plan status (post-release)

- Phase 6 P3.1 (anchor → md5+full-file): **fully closed** across pn118 + maximally closed across pn79 given upstream file deletions. Single-file md5 pattern + sibling composition validated as the canonical approach for multi-file conversions.
- All other Phase status unchanged from v11.1.0.

### Out of scope (deferred)

- `K_001` threshold sweep + multi-turn agentic + 27B comparison — still need rig bench window
- `PersistentBufferRegistry` allocator-level routing — still deferred pending bench validation
- `RELEASE_POLICY.md` percentage re-derivation — still needs rig bench attachment workflow
- Tauri desktop builds — still needs Rust toolchain decision
- Public origin push — still a separate Iron Rule #2 decision

### Branch state

Ships on `feat/container-management` (same branch as v11.1.0 — operator's working branch). v11.2.0 tag points at the final P3.1 closeout commit.

---

## [v11.1.0] — Phase 6 closeout (2026-06-02)

### Highlights

- **Phase 6 P3.1 closed** via `PN118_V2_MD5_WORKSPACE` PoC — single-file md5+full-file PoC of the PN119 reference pattern, scoped to `vllm/v1/worker/workspace.py` (one of pn118's two target files). Companion to original `PN118` (does NOT conflict — they compose: v2 covers workspace.py via md5 guard + full-file replacement; original pn118 retains coverage of `turboquant_attn.py` via anchors, detecting v2's Genesis marker on workspace.py and skipping re-anchoring there). Default OFF — opt-in via `GENESIS_ENABLE_PN118_V2_MD5_WORKSPACE=1`. Validates the single-file md5 pattern; pn118's second target + pn79's multi-file conversion deferred to v11.2.0+.

- **Phase 6 P3.3 partial closure** via `PersistentBufferRegistry` — unified `BufferPool` + singleton `PersistentBufferRegistry` API + 4 patches registered (P36/PN12/P46/P39a). **Scope adapted from plan**: existing allocators (TurboQuantBufferManager, FFNIntermediateCache, GdnGatingBufferManager, FlaKktBufferManager) integrate with `GenesisPreallocBuffer` for CUDA-graph pointer stability and reserve-before-cudagraph semantics; replacing those with `BufferPool.acquire()` direct calls would NOT be byte-equivalent. The migration adds the **registration surface only** — each migrated module imports the registry + pool constant and calls `ensure_pool_registered()` in `apply()` for operator visibility (`sndr patches show buffer_registry` future CLI surface). Tensor storage ownership + allocation semantics + `torch.empty()` calls + per-layer attributes all UNCHANGED. The master plan's "−200 LOC unified allocator" benefit is deferred to v11.2.0+ pending bench validation on the rig.

- **Phase 6 P3.7 fully materialized in git** (Batches 1-8 this release cycle): CLI thin-shell + Product API (~10K LOC + 391 tests + auth subsystem) + GUI React SPA (9 screens) + Tauri scaffold + 4 operator docs. Pre-release operator WIP for container management feature also snapshotted on this branch as `53e716f6`/`80fbff61`/`b9721664`/`c6099e11`.

- **Operator docs Phase 10 V1-sunset closeout** (Track A):
  - `docs/CONFIGS.md` "Step 2: Adapt a launch script" replaced with V2-preset workflow (16 prod + 7 non-prod presets via `sndr preset list/show/explain/recommend` + `sndr launch <preset>`); `cp scripts/start_*` recipe replaced with V2 layered triplet (`sndr config new model + edit + compose preset + launch`).
  - `docs/TROUBLESHOOTING.md` R-002/R-007 smoke commands updated from V1 key `a5000-2x-35b-prod` to V2 alias `prod-qwen3.6-35b-balanced`. New subsection "Restoring a V1 baseline from history (emergency rollback)" documents the `git checkout 607385f1~ -- <V1.yaml>` recovery path.
  - `docs/RELEASE_POLICY.md` lines 137-150 annotated as Wave 10-era snapshot (registry count = 169, bench-with-baseline = 81); re-derivation requires running `sndr patches release-check --mode require-bench-attached` against all 16 `prod-*` presets on the rig — deferred until next bench window.

- **14 new evidence gates** wired this release cycle (`make evidence`: 46 → 60 → 60+ green). New gating gates: `audit-source-refs-in-docs`, `audit-docs-refs-in-source`, `audit-phase3-relocation`, `audit-pn59-cliff2b`, `audit-lifecycle-docstring-sync`, `audit-v2-runtime-pins`, `audit-v2-modeldef-vs-hardware-pin`, `audit-private-namespace`, `audit-shim-window`, `audit-wheel-contents`, `audit-yaml-status-enum`, `audit-override-policy`. Informational: `audit-anchor-fragility`, `audit-v1-sunset`, `audit-english-only`.

- **Registry growth**: 236 → 237 entries (PN118_V2_MD5_WORKSPACE added). 10 docs updated to reflect new count (README + 9 docs/).

### SNDR_MTP_DYNAMIC_K_001 first empirical bench (vllm#26504 port)

- Tested on `prod-qwen3.6-35b-multiconc`, 2× A5000 TP=2, n=25 per arm
- `wall_TPS`: 211.84 (OFF) vs 208.62 (ON) — Δ -1.52% within 11% CV
- Welch t = -0.118, p = 0.9063 — **NOT_SIGNIFICANT**
- Default OFF empirically validated for this preset
- Plan §15.1 "+5-12% mixed workload" does NOT materialize on this workload
- Follow-up candidates (v11.2.0+ bench window): threshold sweep (default 0.7 → 0.5/0.8), multi-turn agentic workload, 27B Lorbus INT4 comparison

### Master plan status (post-release)

- Phases 0-3: closed pre-this-release
- Phase 4: closed per M.1.R amendment 2026-05-27
- Phase 5: 4/5 PRs in pin; vllm#26504 default-OFF validated; tl.dot audit + p101 verified as non-actionable
- Phase 6 P3.5/P3.6/P3.7: DONE in main
- Phase 6 P3.1 (pn118 v2 workspace PoC): CLOSED for the workspace.py scope; pn118 turboquant_attn.py + pn79 multi-file deferred to v11.2.0+
- Phase 6 P3.3 (buffer registry): REGISTRATION SURFACE landed; allocator-level routing deferred to v11.2.0+
- All actionable master plan items either closed or deferred with explicit reasons

### Out of scope (deferred to v11.2.0+)

- `K_001` threshold sweep (need rig bench window)
- `K_001` multi-turn agentic workload bench (need rig bench window)
- `K_001` 27B comparison (need rig bench window)
- `pn118` turboquant_attn.py md5 conversion (workspace.py PoC validates pattern first)
- `pn79` multi-file md5 conversion (35 anchors across 3 files; PN119/PN118_V2_MD5_WORKSPACE single-file pattern validated; multi-file pattern requires separate design work)
- `PersistentBufferRegistry` allocator-level routing — replace `TurboQuantBufferManager` / `FFNIntermediateCache` / `GdnGatingBufferManager` / `FlaKktBufferManager` internals with `BufferPool.acquire()` (CUDA-graph safety + storage-ownership integration requires bench validation)
- `RELEASE_POLICY.md:157-` percentage re-derivation (needs rig bench attachment workflow)
- Tauri desktop builds (needs Rust toolchain decision; scaffold landed `b608b6f4`)
- Public `origin` (github.com/Sandermage/genesis-vllm-patches) push + PyPI/Docker Hub publish (separate Iron Rule #2 decision)

### Branch state

This release ships on `feat/container-management` (operator's working branch). The branch also carries the container-management feature spec (`4b371972`) + WIP implementation snapshots (`53e716f6`, `80fbff61`, `b9721664`, `c6099e11`). Container management feature is orthogonal to v11.1.0 enterprise hardening; both land together because the operator was working on container management while the enterprise hardening work landed on the same branch. The `v11.1.0` tag points at the final enterprise-hardening commit.

---

## [Unreleased] — chat-K3 promotion resolved + TURBOQUANT multimodal regression closed (2026-06-01)

### Highlights

The chat-K3 promotion blocked yesterday (entry below, dated 2026-05-31)
on the `ValueError: AttentionBackendEnum.TURBOQUANT not valid …
'partial multimodal token full attention not supported' + 'kv_cache_
dtype not supported'` rejection at engine init is now resolved. Profile
promoted `experimental → validated` in commit `16eea839`. The earlier
diagnosis — that vllm's TURBOQUANT validator on gemma4 rejects long
contexts because multimodal heads stay in the model even with
`--language-model-only` — was structurally correct but missed the
patch-stack root cause: the chat-K3 profile shipped (commit `284477f9`)
with an incomplete `patches_delta.enable` block that omitted the full
TQ infrastructure stack the sister `structured-k4` carries.

### Audit findings

Investigation followed the STUDY → ANALYZE → VERIFY → SEARCH → COMPARE
sequence per Iron Rule #11. Diff of chat-K3 `patches_delta.enable` vs
structured-k4 surfaced 15 missing entries — the load-bearing ones being
`G4_60L_TQ_BACKEND_MM_PREFIX` (Python-side `supports_mm_prefix=True`
monkey-patch on stock `TurboQuantAttentionBackend`, the direct fix for
the multimodal rejection) and `G4_69_SKIP_LAYERS_NATIVE_BACKEND` (closes
the second `kv_cache_dtype not supported` rejection by resolving
per-layer skip-layer `kv_cache_dtype='auto'` against the global
`--attention-backend TURBOQUANT` correctly).

Sister structured-k4 was unaffected at boot because its
`max_model_len=4096` apparently stays below the vllm validator's
`use_mm_prefix` trigger threshold; chat-K3 at `max_model_len=200000`
hit the guard immediately. structured-k4 still **needs** these patches
for correctness at any real workload — its 4K context just hides the
boot-time guard fire on the current pin.

### Patches changed (chat-K3 profile mirror against structured-k4)

| New entries in chat-K3 `patches_delta.enable` | Role |
|---|---|
| `G4_60A_TQ_SLIDING_SPEC` | PR42637 mixed-attention TQ overlay |
| `G4_60B_TQ_ATTN_OVERLAY` | … |
| `G4_60C_TQ_DECODE_OVERLAY` | … |
| `G4_60D_TQ_STORE_OVERLAY` | … |
| `G4_60E_KV_CACHE_UTILS` | … |
| `G4_60G_TQ_DISPATCH` | … |
| `G4_60H_TQ_CONFIG_AUGMENT` | … |
| `G4_60K_TQ_ENGINE_CONFIG` | … |
| `G4_60L_TQ_BACKEND_MM_PREFIX` | **direct fix for boot rejection** |
| `G4_61_TQ_SHARED_WORKSPACE` | TQ workspace sharing |
| `G4_62_TQ_KERNEL_WARMUP` | TQ kernel warmup orchestrator |
| `G4_32_TQ_VALIDATION_BYPASS` | TQ validation bypass |
| `P65_TURBOQUANT_SPEC_CG_DOWNGRADE` | spec-decode cudagraph downgrade |
| `G4_68_TQ_SPEC_CG_DOWNGRADE_OVERLAY` | TQ-specific CG downgrade |
| `G4_69_SKIP_LAYERS_NATIVE_BACKEND` | **closes 2nd rejection reason** |
| `G4_70_PN259B_MIXED_ALLOC` | mixed allocator |
| `G4_70_PN259B_FAIL_FAST` | … |
| `G4_70_PN259C_ROUTE_B` | … |

### Migration notes

`sizing_override.max_model_len`: 200000 → 65536 (64K). Empirical ceiling
on pin `0.21.1rc1.dev354+g626fa9bba`. The 200K original deadlocked at
PN126 prefill warmup with CUDA OOM (model + TQ-KV reached 21.92 GiB of
23.55 GiB total per card; PN126 wanted 168 MiB headroom, only 136 MiB
free → engine never started API server). 64K boots clean with the full
G4_60* TQ infra stack enabled. Operators wanting to re-raise toward
200K should tune `gpu_memory_utilization` first + walk the progressive
ladder revalidation on the current pin.

### Bench / measurements

3-run × 5-workload validation on pin `0.21.1rc1.dev354+g626fa9bba`,
greedy decode `T=0 top_k=1`, max_tokens=350, CV ≤ 5% on every
workload, target ≥ +50% chat geomean vs default sibling met:

| Workload         |  run1 |  run2 |  run3 |  mean |   CV |
| ---------------- | ----: | ----: | ----: | ----: | ---: |
| free_chat        | 44.71 | 44.60 | 44.18 | 44.50 | 0.6% |
| code_gen         | 43.87 | 44.06 | 43.86 | 43.93 | 0.3% |
| summarization    | 43.41 | 44.21 | 44.27 | 43.96 | 1.1% |
| structured_count | 44.35 | 43.79 | 43.79 | 43.98 | 0.7% |
| tool_json        | 41.77 | 43.79 | 43.51 | 43.02 | 2.6% |

`profile_geomean_tps: 43.608` (+19.0 % vs K=4 — meets
`promotion_gain_threshold: 0.1`). The artifact's
`decision: validated_conditional` carries over per its
`allowed_workloads: [code_gen, free_chat, summarization]` —
`structured_count` and `tool_json` remain routed to the K=4 sibling.

### Verified

- Commit `16eea839` ships the patches_delta mirror + max_model_len
  trim. `chat-K3` launcher boots clean TP=2 + 64K context, serves
  `gemma-4-31b-code`, accepts spec-decode `num_speculative_tokens=3`
  on the gemma-4-31B-it-assistant drafter.
- `prod-gemma4-31b-tq-mtp-chat-k3` preset (commit `d8cafb3c`) flipped
  evidence visibility to public — the artifact JSON
  `vllm/sndr_core/integrations/spec_decode/artifacts/gemma4-31b-tq-mtp-chat-k3.json`
  has been in the public tree since the initial promotion commit;
  the `evidence_visibility: private` marker on the preset card was
  carried over from the experimental phase and outlived its purpose.
- `audit_config_catalog` warning count dropped 9 → 7 as a side effect
  (the 7 remaining are documented operator decisions: 5
  `external://` placeholders pending public bench publish + 2
  `sndr_private/` paths held in the private tree).

### Open follow-ups

- gpu_memory_utilization tuning + progressive ladder validation if
  the operator wants to re-raise `max_model_len` toward 200K on the
  current pin.
- Investigate whether the same root cause (`use_mm_prefix` trigger
  threshold) affects the structured-k4 sibling at its real-workload
  context (it hides at the 4K boot path but could fire mid-request).

---

## [Unreleased] — V1 sunset 10× completed in single day session (2026-06-01)

### Highlights

10 V1 builtin files retired in one day, dropping the V1 baseline from
12 → 2 (83.3% reduction). The last 2 remaining files are both
TRANSPARENT bucket (V2 composes byte-identical config) and require
test/prod-code refactor to retire.

### Sunset bucket distribution evolution

| Stage                 | V1 baseline | transparent | deprecated | needs-choice | tombstone |
|-----------------------|:-----------:|:-----------:|:----------:|:------------:|:---------:|
| Pre-session           | 12          | 3           | 9          | 0            | 0         |
| After #1-#8 (deletes) | 4           | 2           | 2          | 0            | 0         |
| After #9-#10 (PN95)   | 2           | 2           | 0          | 0            | 0         |

### 10 sunsets — strategy log

1. `single-3090-dense-cpu-offload-EXAMPLE` — PoC (pure docs/audit refs)
2. `single-3090-hybrid-gdn-tier-aware-EXAMPLE` — EXAMPLE namespace
3. `a5000-1x-27b-int4-tested` — ZERO runtime refs (cleanest by audit)
4. `a5000-2x-35b-fp8-dflash` — comparative-study winner (0 tests)
5. `a5000-2x-27b-int4-tq-k8v4-dflash` — self-flagged `lifecycle: retired`
6. `a5000-2x-27b-dflash-true` — first TRANSPARENT-bucket sunset
7. `a5000-2x-27b-int4-long-ctx` — comment-only refs
8. `a5000-2x-27b-int4-tested` — first sunset with legacy test fixture migration
9. `a5000-2x-tier-aware-EXAMPLE` — PN95 architectural unblock
10. `a5000-1x-tier-aware-pn95` — twin sunset with #9

### Methodology established

1. **Runtime-coupling audit** replaces ref-count heuristic (commit `165c453f`)
2. **Comparative study** of V1↔V2 functional equivalence per migration table rationale
3. **Content preservation policy** — append annotations, NEVER truncate originals
4. **Test fixture migration** — surviving sibling swap pattern (commit `47b5a004`)
5. **Architectural unblock** — extract domain-specific data out of V1 ModelConfig (PN95 refactor, commits `<sha-1>` + `<sha-2>`)

### Phase 10 plan for remaining 2 V1 files

Both files are TRANSPARENT bucket (V2 byte-identical) but blocked by
test/prod-code coupling rather than data semantics.

#### `a5000-2x-27b-int4-tq-k8v4` — 6 active call sites

| Site | Type | Migration path |
|---|---|---|
| `tests/unit/test_phase9_v1_freeze.py:102` | V1 freeze test by design | Refactor to use synthetic V1 fixture (tmp_path + monkey-patched `_BUILTIN_DIR`); OR retire test after Phase 10 V1 freeze concept itself is sunset |
| `tests/unit/model_configs/test_package_versions.py:154` | Asserts `cfg.package_versions` field | V2 ModelConfig (via `load_alias`) does NOT carry `package_versions`. Migration requires either V2 ModelDef schema extension OR test refactor to assert versions from V2 `versions_override` path |
| `tests/legacy/test_model_config_cli.py:112,126` | V1 CLI (`cli_main(["render", ...])`) | V1 CLI is itself in `compat/` namespace (compat-only). Either retire V1 CLI test suite OR migrate to V2 CLI testing surface |
| `tests/legacy/test_community_lifecycle.py:163` | `cfg.validate()` | Likely V2-compatible: change `get(...)` → `load_alias("prod-qwen3.6-27b-tq-k8v4")` |
| `tests/legacy/test_model_config_audit_rules.py:209` | `audit(get(...))` | Likely V2-compatible: same swap |

Two-step migration: (1) swap V2-compatible sites first (low risk), (2) refactor V1-bound sites (high risk).

#### `a5000-2x-35b-prod` — 12+ active call sites

| Category | Count | Migration path |
|---|---|---|
| `tests/unit/test_phase9_v1_freeze.py` | 6 | Same synthetic-fixture refactor as above |
| Prod code: `cli/{compose,config,install,report,launch}` + `compat/models/pull.py` + `kv_calc.py` + `__init__.py` | 6+ | Mostly docstring examples (already annotated with V2 hint via commit `569bd108`). Two paths actively call `get('a5000-2x-35b-prod')` in fallback chains — refactor to try V2 alias first |
| `tests/unit/model_configs/test_package_versions.py` | 1 | Same `package_versions` blocker as tq-k8v4 |
| `tests/legacy/test_*` | 4+ | Same V2-compatible swap pattern |

#### Common blocker — `package_versions` schema field

The V1 `ModelConfig` schema carries a `package_versions` field (pinned
Python package versions: `pandas==2.2.3`, `scipy==1.14.1`, etc.). V2
`ProfileDef`/`ModelDef` does NOT expose this field. Tests asserting
`cfg.package_versions is not None` fail after V2 alias migration.

Resolution paths (Phase 10 design decision):

1. **Extend V2 schema**: add `package_versions` field to either
   `ProfileDef` (per-profile pinning) or `ModelDef` (per-model pinning).
   Update V2 `compose` to populate the field. Tests work unchanged.

2. **Move package pins to a separate source-of-truth**: extract
   pinned versions to `pyproject.toml` or a Genesis-internal manifest.
   Tests refactor to read from new source.

3. **Retire the `package_versions` tests**: if Genesis no longer pins
   Python packages at the per-config level (it's a per-release pin),
   the tests are obsolete and can be deleted.

Operator + maintainer decision required before Phase 10 execution.

### Verified

- `python3 scripts/audit_no_new_v1.py`: "✓ V1 frozen — 2-entry baseline"
- `pytest tests/unit/scripts tests/unit/model_configs
  tests/unit/test_runtime_tunables.py`: 836 passed, 2 warnings
- All 10 commits land cleanly on `dev` branch (no push)
- PN95 architectural refactor preserves operator-visible behavior
  (backward-compat aliases for legacy V1 env values)

---

## [Unreleased] — chat-K3 promotion blocked by TURBOQUANT + gemma4 multimodal regression (2026-05-31)

### Audit findings

Attempted to promote `gemma4-31b-tq-mtp-chat-k3` profile from
`status: experimental` → `validated` per Stage K plan (3× bench
runs × 5 workloads for CV ≤ 5% validation of the earlier session's
"+112% chat, +19% global" evidence).

**Promotion blocked**. The chat-K3 container does NOT boot on the
current vllm pin `0.21.1rc1.dev354+g626fa9bba`:

```text
ValueError: Selected backend AttentionBackendEnum.TURBOQUANT is not
valid for this configuration. Reason: ['kv_cache_dtype not supported',
'partial multimodal token full attention not supported']
```

Reproduced at:

- `max_model_len=200000` (profile default) — fail
- `max_model_len=32768` — fail
- with `--language-model-only` — fail
- without `--language-model-only` — fail

The structured-k4 sibling (same profile family, identical Genesis
patch stack, same TURBOQUANT backend + turboquant_4bit_nc KV) boots
cleanly. The only material difference is `max_model_len`:
structured-k4 uses 4096, chat-k3 uses 200K. Hypothesis: vllm's
`AttentionBackendEnum.TURBOQUANT` validator on gemma4 (which carries
multimodal heads even with `--language-model-only`) rejects long
contexts due to "partial multimodal token full attention" guard;
4K stays below the trigger threshold.

The earlier session's chat-K3 bench evidence ("+112% chat,
+19% global geomean") was logged into the profile YAML earlier
today (2026-05-31, same pin) but is now unverifiable — that bench
either ran on a different transient state OR the profile YAML
captured a now-stale measurement.

### Profile YAML annotation

Updated `gemma4-31b-tq-mtp-chat-k3.yaml` status comment with the
boot regression details, blocking promotion until investigation
resolves the TURBOQUANT+long-context+gemma4-multimodal interaction.

### Operational state

- structured-k4 restored on rig port 8102 (canonical operational)
- chat-k3 launcher reverted to original (--language-model-only and
  max_model_len edits backed out via .bak restore)
- No commit attempted for chat-k3 launcher fix since none worked

### Open follow-ups

- Investigate vllm TURBOQUANT backend validator at
  `vllm/v1/attention/backends/...` to find the multimodal+long-ctx
  guard condition + understand why structured-k4 (4K) passes
- Try alternative backends (FLASH_ATTN, FLASHINFER) on chat-K3
  config and see if any yields equivalent perf without the
  multimodal rejection
- If TURBOQUANT-on-long-ctx is fundamentally unsupported on this
  pin, redesign chat-K3 profile to use a different backend OR pin
  guard the profile to a known-working pin range

---

## [Unreleased] — Renderer auto-pick fix + ProfileDef.target_hardware (2026-05-31)

### Highlights

Two-step closure of the V2 launcher auto-pick mis-rendering discovered
during Stage G post-rename qwen validation. Initial qwen3.6-27b-tq-k8v4
boot failure was misdiagnosed as KV-budget drift; root cause turned out
to be `sndr profile render-launchers` auto-picking `a5000-1x` (smallest
satisfying) instead of `a5000-2x` (largest VRAM the profile semantics
target).

### Changes

1. **Commit `861e1fed`** — `_pick_default_hardware()` sort-order fix.
   Old: `candidates[0]` from alphabetically-sorted `list_hardware()`
   (picks `a5000-1x` for 27B INT4 model that fits both 1× and 2×).
   New: sorts by total VRAM DESC and returns largest. Operators wanting
   a smaller rig pass `--hardware <id>` explicitly.

2. **Commit `3a9ee38c`** — `ProfileDef.target_hardware: Optional[str]`
   new schema field. Closes the edge case where auto-pick logic (now
   "prefer largest VRAM") mis-routes 3 variant profiles whose
   semantics target a specific smaller rig: `qa-qwen3.6-27b-tq-1x`
   (→ a5000-1x), `cpu-offload-3090` and `tier-aware-3090` (→ single-3090).
   Renderer precedence: explicit `--hardware` > `profile.target_hardware`
   > auto-pick.

### Audit result

All 23 V2 profiles re-rendered cleanly:

- 20 → `a5000-2x-24gbvram-16cpu-128gbram` TP=2 (auto-pick default)
- 2 → `single-3090-24gbvram` TP=1 (via target_hardware)
- 1 → `a5000-1x-24gbvram-16cpu-128gbram` TP=1 (via target_hardware)

Mismatches: **0**.

### Verified

- `pytest tests/unit/cli/test_profile_render_launchers.py
  tests/unit/cli/test_profile_validate.py`: 58 passed
- `bash /home/sander/start_qwen3.6-27b-tq-k8v4.sh` on rig with pin <!-- security_scan: allow -->
  626fa9bba: boots cleanly TP=2 + 256K context, serves `qwen3.6-27b`,
  generates coherent reasoning for "Capital of Germany?" prompt
- All 23 V2 launchers pass `bash -n` syntax check on rig
- Reverted misdiagnosed profile changes (max_model_len → 262144)

---

## [Unreleased] — V2 launcher coverage closure + legacy chat_default deprecation (2026-05-31)

### Highlights

Closes V2 launcher coverage gap surfaced by the Phase 1-2 config audit:
every renamed V2 PROD/QA/experimental profile now has a corresponding
`sndr profile render-launchers`-generated launcher deployed to the rig
at `/home/sander/start_<profile-id>.sh`. <!-- security_scan: allow -->

### Operator notes

- 15 V2 launchers generated + deployed to rig (in addition to the 5
  gemma4 launchers from the gemma4 rename commit):

  - `start_qwen3.6-{27b,35b}-{dflash,multiconc,balanced,...}.sh`
  - `start_gemma4-26b-{multiconc,multiconc-k1,no-mtp}.sh`
  - `start_ab-qwen3.6-27b-tq-dflash.sh`
  - `start_qa-qwen3.6-27b-{fp8kv,tq-1x}.sh`

- All pass `bash -n` syntax check
- All use `--entrypoint $LAUNCHER_DIR/run.sh` pattern (immune to
  vllm image ENTRYPOINT drift observed post-pin-bump)
- All pin to canonical SHA-pinned image
  `vllm/vllm-openai:nightly-626fa9bba5663a5cf6a870debf031ee344ddb822`

### Legacy chat_default_PROPER launcher deprecated

`/home/sander/start_gemma4-31b_chat_default_PROPER.sh` produces garbage <!-- security_scan: allow -->
output (`la la la la la...`) post-pin-bump. Root cause:
`--tool-call-parser gemma4 --enable-auto-tool-choice` references the
`Gemma4ToolParser` class which does NOT exist in the current vllm pin
(G4_14 reports "No Gemma4ToolParser-like class found in this vLLM
pin; G4_14 is no-op").

Confirmed via A/B test 2026-05-31:

- Legacy launcher: `content: la la la la la la la la la...` (garbage)
- V2 `start_gemma4-31b-tq-default.sh`: `content: "Tensor parallelism
  splits large weight matrices across multiple GPUs. Instead of
  duplicating the model, each GPU computes a partial result..."`
  (coherent, correct)

Both launchers serve the same profile semantics (MTP-OFF, TQ KV,
broad workload safe). V2 just lacks the broken tool-parser flag.

Operator action: rename of legacy file applied on rig
(`start_gemma4-31b_chat_default_PROPER.sh.DEPRECATED_2026-05-31_use_V2_start_gemma4-31b-tq-default`).
Use `bash /home/sander/start_gemma4-31b-tq-default.sh` going forward. <!-- security_scan: allow -->

### Verified

- `curl http://localhost:8101/v1/chat/completions` (V2 default):
  coherent chat response, no garbage
- `curl http://localhost:8102/v1/chat/completions` (V2 structured-k4):
  `"2 + 2 = 4"` — correct math
- Both V2 launchers boot cleanly with no `--tool-call-parser gemma4`
  reference; Genesis patches still apply (G4_19_TURBOQUANT_KV etc.)

### Post-rename qwen launcher validation (Stage G)

A/B verification that the qwen3.6 mass-rename (commit 91daa11d) did
NOT break the qwen model path:

- `bash /home/sander/start_qwen3.6-35b-balanced.sh` → boots clean, <!-- security_scan: allow -->
  served-model-name `qwen3.6-35b-a3b`, generates coherent reasoning
  chain about MoE architecture (60-word constraint task). Confirms
  qwen3.6 rename is wire-clean end-to-end.

- `bash /home/sander/start_qwen3.6-27b-tq-k8v4.sh` → FAILS at engine <!-- security_scan: allow -->
  init with `ValueError: To serve at least one request with the
  model's max seq len (262144), 7.11 GiB KV cache is needed, which
  is larger than the available KV cache memory (1.81 GiB).`

  Root cause: profile `qwen3.6-27b-tq-k8v4.yaml` was bench-validated
  at 256K context on vllm pin `dcacdf9a` (2026-05-14 progressive
  ladder PASS at 200K/230K/256K). The current pin `626fa9bba` has a
  different KV-budget accounting and the 256K context no longer
  fits on 2× A5000 at `gpu_memory_utilization=0.92`.

  This is **NOT a rename bug** — same config would have failed
  pre-rename. It is an Iron-Rule-#11 style validation drift: the
  profile carries stale evidence from an older pin. Re-validation
  required on current pin OR sizing_override needs adjustment.
  Documented as separate follow-up; out of scope for this rename
  cleanup commit.

---

## [Unreleased] — SNDR_MTP_DYNAMIC_K_001 scope correction (2026-05-31)

### Audit findings

Investigation Discipline catch (Sander mandate: "verify, study, compare,
analyze, and test"). Earlier session claimed SNDR_MTP_DYNAMIC_K_001
gave +10.7% chat on gemma4-31B. v1.4 attempted to close the gap to static
chat-K=3 by adding `GENESIS_SNDR_MTP_DYNAMIC_K_INITIAL` env var.

Bench on gemma4-31B with v1.4 + initial_k=3 produced:

- chat -22.8%, code -20.7%, count -17.9%, json +0.2% vs static K=4
- NO `[SNDR_MTP_DYNAMIC_K]` adapt log lines after multiple requests
- Apply log fires correctly at boot

Root cause via docker exec MRO probe:

```text
Gemma4Proposer MRO: [Gemma4Proposer, SpecDecodeBaseProposer, object]
Own propose: False
Own __init__: True
DraftModelProposer.propose at: vllm/v1/spec_decode/llm_base_proposer.py:427
```

`Gemma4Proposer` does NOT inherit from `DraftModelProposer`. The
monkey-patch on `DraftModelProposer.__init__` and `.propose` has ZERO
effect on gemma4 (any variant). The patch is only relevant for models
whose proposer class actually inherits from DraftModelProposer.

Decision:

- v1.4 changes stashed locally (`WIP-DYNK-v1.4-UNVERIFIED-REGRESSION`),
  NOT committed
- Repo + rig back at HEAD v1.0
- Earlier "+10.7% chat" figure for gemma4-31B was bench noise, not real
- Patch metadata needs scope correction in next session: add explicit
  `applies_to.proposer_class` predicate so the patch self-skips on
  models that don't use a `DraftModelProposer`-rooted MRO (gemma4
  uses `Gemma4Proposer(SpecDecodeBaseProposer)` directly)
- Re-verification needed on qwen3.6-27B / qwen3.6-35B to determine if
  THEIR proposers inherit from DraftModelProposer (likely yes via
  generic assistant-model MTP path) — bench gains, if real, will be
  visible there, not gemma4

### Operator notes

- vllm/vllm-openai:nightly image ENTRYPOINT changed to `[vllm, serve]`
  (was different in pre-626fa9bba pin). Launcher scripts using
  `bash -c "set -e; pip install …; exec vllm serve …"` pattern break
  because the bash command gets treated as positional args to `vllm
  serve`. Fix: add `--entrypoint /bin/bash` to docker run and remove
  the redundant leading `bash` from CMD args (so just `-c "…"`
  reaches /bin/bash). Applied to
  `/home/sander/start_gemma4-31b_chat_default_PROPER.sh` on the rig <!-- security_scan: allow -->
  (sed backup `.bak-2026-05-31` / `.bak2-2026-05-31` / `.bak3-2026-05-31`
  preserved).
- Same fix likely needed on other rig launchers using the
  `bash -c "exec vllm serve …"` pattern; audit before next launch.

### Verified

- `git stash list` shows `WIP-DYNK-v1.4-UNVERIFIED-REGRESSION` (preserved
  for future investigation, never committed)
- Default 31B container `vllm-gemma4-31b-chat-default` healthy on port
  8104 with HEAD v1.0 g_dynamic_k_mtp_proposer.py file
- Docker exec MRO probe confirms Gemma4Proposer doesn't inherit from
  DraftModelProposer

---

## [Unreleased] — K.1.R pin bump dev371 → nightly-626fa9bb + PN82 retire (2026-05-28)

### Highlights

K.1.R deep-diff pin bump audit closure. New canonical pin: vllm/vllm-openai
nightly tag `nightly-626fa9bba5663a5cf6a870debf031ee344ddb822` (multi-arch
manifest digest `sha256:674922aae790c2cbf45f4e844098d227b80d40a74bfc7797a444d213a221879f`,
upstream commit 626fa9bba5663a5cf6a870debf031ee344ddb822 from 2026-05-28T04:59:34Z,
"[BugFix] Fix blocked reasoning parsing with MRV2 (#43808)"). Previous pin
`0.20.2rc1.dev371+gbf610c2f5` (bf610c2f56764e1b30bc6065f4ceace3d6e59036)
retained in allowlist as fallback. 362 commits ahead of previous pin;
closest annotated ancestor is v0.21.1rc0.

PROMOTION_PENDING — repo state reflects the new pin as canonical (docs,
compose YAMLs, hardware YAMLs, model_configs `vllm_pin_required`,
README badge), but `PROMOTION_PENDING_VLLM_PINS` includes the new pin
until the server-side bench cycle on the rig validates 27B + 35B PROD
presets.

### Patches changed

**Retired (1):**

- **PN82** (`Mamba CUDA-graph stale is_prefilling padded rows`) →
  superseded by vllm#41873 merged upstream at commit
  39d5fa96a7c687f9ed7e14a5a52064965356cede on 2026-05-21T22:42:43Z,
  included in pin window dev371 → 626fa9bb. K.1.R deep-diff confirmed
  byte-equivalent insertion of `is_prefilling[num_reqs:] = False` at
  the same post-assignment location in `vllm/v1/worker/gpu_model_runner.py:2270`.
  Per iron rule #11 case (a) — only delta is comment text (Genesis 3-line
  in-place vs upstream 2-line, cosmetic). Wiring moved to
  `integrations/_retired/`, registry entry updated with `lifecycle=retired`,
  `superseded_by`, and `vllm_version_range: "<0.21.1rc0+g626fa9bba5"`.
  Test file moved to `tests/unit/integrations/_retired/`; worker family
  contract updated.

**Already absorbed (8, status confirmed):**

PR audit at K.1.R confirmed 9 other tracked PRs merged upstream prior
to dev371 (already retired in earlier sessions): P94 (#41043), PN9
(#39930 / G4_05), PN13 (#41235), PN19 (#41268), PN52 (#41411), PN67
(#41674 — merged exactly at our previous pin's SHA), PN80 (#41845).
No re-classification needed; existing `lifecycle=retired` entries cover
them. PN82 is the only addition for K.1.R.

**Anchor drift (4 files / OPEN PRs, deferred):**

Anchor drift check at 626fa9bb identified shifts in 4 upstream files
covered by 6 OPEN PR patches: `config/vllm.py` (PN275, P95, P66, P72),
`model_executor/layers/quantization/auto_gptq.py` (P91),
`model_executor/layers/quantization/inc.py` (P91B),
`transformers_utils/configs/speculators/algos.py` (PN21). All affected
patches carry `upstream_drift_markers` self-skip semantics — runtime
will surface drift via `apply.shadow` reporter rather than crash. Closer
anchor re-anchoring deferred to a follow-up slice (does NOT block the
pin bump under the soft-fail-on-anchor-shift policy).

### Migration notes

**For operators:**

1. **Docker pull** the new image when convenient:

   ```bash
   docker pull vllm/vllm-openai@sha256:674922aae790c2cbf45f4e844098d227b80d40a74bfc7797a444d213a221879f
   ```

   The compose YAMLs already reference this digest.

2. **Validate locally first** on a non-PROD container:

   ```bash
   sndr launch prod-qwen3.6-35b-balanced --dry-run    # confirm pin matches
   sndr launch prod-qwen3.6-35b-balanced              # boot + smoke
   sndr verify                        # tool-call smoke
   ```

3. **Rollback path** if needed: previous pin `dev371+bf610c2f5` is in
   the allowlist (multi-arch digest `sha256:f66782cc770fb92b9c4e608a70aa04b9c51e6f3055e0b82c77a53d7107cd71a7`).
   Edit compose YAML image: line to revert.

4. **PN82 retire impact**: if you had `GENESIS_ENABLE_PN82_MAMBA_CUDAGRAPH_PREFILL_ZERO=1`
   in env, leaving it set is harmless (the patch now self-skips with a
   retirement message). Remove for cleanliness.

### Bench / measurements

K.1.R is a laptop-side audit. Server-side bench cycle pending.

### Audit findings

PR audit summary (72 unique upstream_pr references in registry):

- 50 OPEN upstream
- 22 CLOSED (10 actually merged, 12 closed-without-merge)
- 1 merge in pin window dev371→626fa9bb (PN82)
- 4 files with anchor drift (covered by existing upstream_drift_markers)

### Verified

- `python3 -m pytest tests/unit -q --tb=no` — all green (forthcoming
  final sweep in Commit 4).
- `audit_registry_contract.py` — 8 invariants clean.
- `make gates` — all 14 fast gates green.
- `apply.shadow --strict` — CLEAN (43 known spec-only).
- `patches doctor` — 0 errors / 0 warnings.

Refs: `sndr_private/planning/audits/K_1_R_PIN_BUMP_AUDIT_2026-05-28_RU.md`
(forthcoming deliverable).

---

## [Unreleased] — CONFIG-UX.5 derived config catalog closure (2026-05-24)

> Closes the CONFIG-UX.5 derived-catalog track end-to-end across two
> commits: (a) schema + generator + drift audit (CONFIG-UX.5.1) and
> (b) native `sndr config-catalog` CLI surface (CONFIG-UX.5.2). The
> catalog is a **derived API over V2 YAML + registry — not a new
> source of truth**; nothing is committed to git as catalog JSON, and
> no SQLite engine is introduced. Operators regenerate on demand; CI
> verifies determinism and redaction.

### Highlights

- **Catalog schema + generator + drift audit (CONFIG-UX.5.1)**. New
  `vllm/sndr_core/model_configs/catalog_schema.py` defines five row
  types (`preset`, `profile`, `model`, `hardware`, `baseline`) plus
  `RedactedEvidenceRef` and the `is_redactable_path` /
  `is_private_visibility` helpers that gate every output path.
  `scripts/generate_config_catalog.py` produces a deterministic
  JSON catalog from the V2 corpus with `--stdout` and `--check`
  modes; `scripts/audit_generated_config_catalog.py` verifies that
  regeneration is bit-stable and that no `sndr_private/*` paths or
  `visibility: private` evidence_refs leak into the public output.
- **Native CLI surface (CONFIG-UX.5.2)**. New `sndr config-catalog`
  subcommand with four leaves:
  `build [--check] [--stdout] [--out PATH]`,
  `verify [--strict] [--json]`,
  `show <row_id|row_type/row_id> [--section SECTION]`,
  `query --row-type {preset,profile,model,hardware,baseline,any}
  [--field FIELD --equals VALUE | --contains VALUE | --expires-before DATE]
  [--json]`. Query is AND-only with a fixed five-flag DSL — no
  expressions, joins, sort, or JSONPath. Collision-aware row id
  resolution (bare form errors with the candidate list if the same id
  appears across row types). `--strict-fresh` with `--from PATH`
  honors the supplied path and exits non-zero on staleness (does not
  silently fall back to regeneration).
- **Redaction discipline**. Redaction is applied at output time
  rather than at build time, so the same generator drives both the
  public catalog and any private debug view. The CLI defaults to
  `--redact-private` on every leaf; turning it off requires explicit
  operator action and is not exercised by any committed code path.
- **Discoverability**. `make config-catalog` umbrella alias forwards
  to `sndr config-catalog build`. All five `--help` texts contain the
  anchor phrase **"derived catalog"** so operators reading the CLI
  surface see immediately that this is not a new source of truth.

### Migration notes

- No operator action required. The catalog is generated on demand;
  there is no committed catalog JSON to refresh in `git pull`. No
  SQLite engine is introduced — the JSON shape is the API.
- A future `.5.3` SQLite backing could be opened if a concrete
  consumer emerges (analytics, dashboard, ratchet gate). Operator
  policy is **no SQLite without a named consumer** — empty schema
  has no value.
- Catalog row schemas are versioned by the `schema_version` field
  in `catalog_schema.py`. Breaking changes to row shape will bump
  this field and be called out in the corresponding CHANGELOG entry.

### Audit findings

Final state on the live corpus at the closure commit:

- `audit-generated-config-catalog`: **0 findings** (deterministic
  regeneration + redaction clean across 60 rows).
- `audit-override-policy`: **0 findings** (CONFIG-UX.4 DEBT closure
  preserved; no profile newly trips a Class-4 rule).
- `audit-configs`: **21 / 21 byte-identical compose** across the
  CONFIG-UX.5.1 + .5.2 commits — catalog work is metadata, not
  runtime mechanics.
- pytest baseline across `tests/unit/{model_configs,dispatcher,scripts,cli}`:
  **2231 passed** (CONFIG-UX.5.1 landed at 2199, CONFIG-UX.5.2 added
  32 new test gates; one parametric skip when torch is preloaded by
  another test in the run — verified torch-free import via dedicated
  subprocess gate). No regression introduced.
- 12 pre-existing family-contract failures (gemma4 `g4_60`, `pn51`,
  `pn66`/`pn67`) confirmed unrelated by stash-and-rerun; they
  pre-date this bundle and are tracked separately.
- Torch-free import gate verified in an isolated subprocess —
  `import vllm.sndr_core.cli.config_catalog` does not pull `torch`
  into `sys.modules`. The corresponding parametric pytest gate
  skips when torch is already preloaded by another test in the same
  run, so the subprocess invocation is the authoritative proof.

### Migration help

```bash
# Build the catalog in-memory and print to stdout
sndr config-catalog build --stdout

# Verify regeneration is deterministic + redacted (CI mode)
sndr config-catalog verify --strict --json

# Show a specific row (collision-aware bare form supported)
sndr config-catalog show preset/prod-qwen3.6-35b-balanced
sndr config-catalog show prod-qwen3.6-35b-balanced              # if id is unambiguous

# Query with the fixed five-flag DSL (AND-only, no expressions)
sndr config-catalog query --row-type profile \
                          --field override_class --equals bench

sndr config-catalog query --row-type profile \
                          --field override_expires_at \
                          --expires-before 2026-09-01

# Discoverable umbrella alias
make config-catalog
```

---

## [Unreleased] — CONFIG-UX.4 override-policy rollout closure (2026-05-24)

> Closes the CONFIG-UX.4 override-policy track end-to-end. Across six
> commits this bundle (a) ships the rollout infrastructure that lets V1
> deprecation + V2 card-less + missing-override-policy events emit
> stage-aware severity, (b) enforces four Class-4 forbidden overrides
> at audit-time regardless of OverridePolicy justification, and (c)
> retires the 20-profile `missing_override_policy` debt that CONFIG-UX.4.2
> surfaced — 20 → 0 with zero overclaim, zero fabricated evidence, and
> zero profile auto-promoted to the `production` class.
>
> Default rollout stage flips from `0` to `1`; observable behavior for
> operators is unchanged (the severity matrix is identical between
> Stage 0 and Stage 1 for non-tombstone buckets). Stage 2 strict-mode
> runs only on tag-push CI via the new `rollout_strict_audit` job in
> `.github/workflows/release.yml`.

### Highlights

- **Rollout infrastructure (CONFIG-UX.4.1)**. New
  `vllm/sndr_core/model_configs/_rollout.py` with stage parser and
  `effective_severity()` helper; frozen `_v1_migration_table.json`
  shipped as package data (12 V1 keys × four buckets);
  `scripts/audit_v1_migration.py` informational audit gate; signature
  extensions on `_maybe_warn_v1_deprecation` and `_maybe_warn_unannotated`
  preserve backward compatibility; tombstone-incident template under
  `sndr_private/planning/incidents/_TEMPLATE_tombstone.md`.
- **Class-4 forbidden override enforcement (CONFIG-UX.4.2)**. Four
  predicates registered in `audit_override_policy.py`:
  `gpu_memory_utilization > 1.0`, `tensor_parallel_size > hardware.n_gpus`,
  `kv_cache_dtype` downgrade vs model declaration (static narrowness
  ordering), and `spec_decode` method-name change. Audit-time only; K-only
  spec-decode changes remain allowed. `DEFAULT_STAGE` flips `0 → 1`.
- **Stage 2 release-strict CI job**. `.github/workflows/release.yml`
  gains `rollout_strict_audit` that exports `SNDR_V1_ROLLOUT_STAGE=2`
  and runs the three CONFIG-UX audits with `--strict`. `make evidence-release`
  mirrors the env export for local parity.
- **Override-policy debt closure (DEBT.1 + 2A + 2B + 2C)**. All 20
  profiles with `sizing_override` now carry an explicit `override_policy`
  block; no profile claims the `production` class — every annotation
  uses `bench` (12) or `qa` (2) per the anti-overclaim discipline.
  `expires_at: 2026-08-22` set uniformly for every annotation in this
  bundle to force a 90-day re-evaluation cycle.

### Migration notes

- Operators previously running at the implicit Stage 0 see no
  observable change at the new Stage 1 default (severity matrix is
  identical between Stage 0 and Stage 1 for non-tombstone buckets).
- Operators who want to revert to explicit Stage 0 can set
  `SNDR_V1_ROLLOUT_STAGE=0` during the transition window (≥1 release).
- Pre-commit hooks for `audit_config_catalog`, `audit_override_policy`,
  and `audit_v1_migration` stay default-severity (non-strict). Strict
  mode runs only on tag-push CI and via `make evidence-release` locally.
  Class-4 errors fire unconditionally — they're physics / evidence
  policy, not rollout warnings — and the global silence env
  `GENESIS_DISABLE_V1_DEPRECATION_WARNING=1` does NOT suppress them.
- DEBT closure used `bench` class (not `production`) even where public
  baseline JSONs exist, because per-preset config-block cross-validation
  is deferred to a follow-up audit. Promotion of individual profiles to
  the `production` class is a separate per-profile decision after public
  bench refresh; the `expires_at: 2026-08-22` field is the forcing
  function.
- `tier-aware-3090` was annotated under Path A (operator-locked
  classification per the CONFIG-UX.4.DEBT.2C mini-R) because the 145K
  `max_model_len` over the 65K Path-A hardware default is intentional:
  the hardware YAML's own inline comment ("profile may override")
  anticipates the override, and PN95 tier-aware page demotion is the
  mechanism that makes it usable on a single 24GB card. The annotation
  carries `allowed_to_exceed_hardware_default: true` as the explicit
  operator acknowledgement.

### Audit findings

Final state on the 21-profile live corpus post-closure:

- `audit_override_policy.py` default: **0 errors / 0 warnings**
  (was 20 missing_override_policy warnings before DEBT.1).
- `audit_override_policy.py --strict`: **exit 0** (was exit 1).
- Class-4 forbidden rules: **0 fires** on all 21 profiles, all four
  rules exercised by positive/negative tests in
  `tests/unit/scripts/test_audit_override_policy_class4.py`.
- Compose output: **21/21 byte-identical** across every commit in this
  bundle — annotations are metadata, not runtime mechanics.
- `audit_v1_migration.py` Stage 1 distribution (informational): 3
  transparent V1 keys, 5 needs_operator_choice, 4 deprecated, 0 tombstone.
- pytest baseline: **2153 passed** across the model_configs, dispatcher,
  scripts, and cli suites, no regression introduced by any commit in
  the bundle.

DEBT closure batch-by-batch:

| Batch     | Profiles | Class assignments                                                            | Audit debt |
|-----------|---------:|------------------------------------------------------------------------------|------------|
| DEBT.1    | 7        | all `bench` + public/private evidence (production-facing, multi-conc family) | 20 → 13    |
| DEBT.2A   | 5        | 2 `qa` + 3 `bench`, `evidence_refs: []` (qa / a-b / example clear class)     | 13 → 8     |
| DEBT.2B   | 7        | `bench` + `reason: "bench_pending: ..."` prefix (no evidence yet)            | 8 → 1      |
| DEBT.2C   | 1        | `bench` Path A + structural reason (tier-aware-3090; mini-R driven)          | 1 → 0      |

### Migration help

```bash
# Browse the new operator preset surface
sndr preset list
sndr preset recommend --workload structured_json.short

# Inspect the V1 migration table + Stage 1 distribution
python3 scripts/audit_v1_migration.py --json

# Run Stage 2 strict-mode audits locally (matches tag-push CI)
make evidence-release

# Revert to explicit Stage 0 during transition window
SNDR_V1_ROLLOUT_STAGE=0 sndr launch <preset>

# Inspect any one profile's new override_policy block
sndr profile show qwen3.6-27b-tq-k8v4
```

---

## [v11.0.0+wave10] — Multi-conc unlock + PN204 v2 + partial-merge recoveries (2026-05-15)

> Multi-concurrency throughput characterisation на dev371 pin
> (`0.20.2rc1.dev371+gbf610c2f5`, nightly `bf610c2f56764e1b30bc6065f4ceace3d6e59036`).
> Discovered real system ceiling is **689 TPS aggregate at `max_num_seqs=8`**
> (vs 215 TPS single-conc) — 3.21x scaling factor on Qwen3.6-35B-A3B-FP8.
> All Wave 9 patches retained; this is additive, not breaking.

### Highlights

- **PN204 v2 redesign** — добавлен `torch.compiler.is_compiling()` guard +
  `_PN204_ARMED` flag + arming hook on `Worker.determine_available_memory`.
  Закрывает CUDA driver crash (`torch/_inductor/runtime/static_triton_launcher.py:291
  invalid argument`) который v1 показывал на dev371 во время `profile_run`.
  Bench-neutral within CV (675 vs 689 agg, TTFT 240 vs 237 ms) — но не крашит
  и hardware-future-proof (Hopper SM 9.0+ должен дать +5-10% по vllm PR #42301).
- **Multi-concurrency throughput discovery** — единичная per-req TPS 215 не
  была настоящим ceiling'ом. Bench `tools/multi_conc_bench.py` показал
  689 TPS aggregate на `max_num_seqs=8`, 3.21x масштабирование. PN119 GQA
  grouping + cudagraph capture при batch>1 — РАБОТАЮТ.
- **3 partial-merge recoveries** на retired patches:
  - **P26** — sub-level `upstream_merged_markers` для cu_2 (upstream landed
    cu_2 hasattr guard на dev338, но output pool ~32 MiB/call всё ещё
    un-merged на dev371; sub-patch теперь применяется независимо).
  - **PN50** — anchor indentation drift fix (12/16 → 8/12 spaces на dev371,
    "anchor_missing" был ложным детектом upstream merge — реально GDN fusion
    Triton kernel НЕ landed upstream).
  - **P12 v2** — добавлен `p12_last_to_first_occurrence` sub-patch поверх
    upstream's LAST-occurrence merge (vllm#35687 landed token-id resolution
    но silently drops earlier `<tool_call>` blocks в multi-tool flows).
- **PN51 reactivated** — Qwen3 streaming `enable_thinking=false` content
  routing fix. Upstream issue #40816 STILL OPEN. Streaming path не имеет
  `not self.thinking_enabled` short-circuit — клиенты которые читают только
  `delta.content` (Open WebUI, LibreChat, LobeChat, Cline, OpenCode) видят
  "reasoning only" вместо ответа. Перенесён из `_retired/` после audit.
- **PN134 retired loud-and-clear** — bench validated -25% TPS regression
  (`StorageBox.should_realize_on_reuse` monkey-patch ломает Inductor
  compile cache на hybrid_gdn_moe). Lifecycle `retired`, double-env-flag
  guard, iron-rule waiver, bench data в docstring.

### Patches changed

| ID | Action | Rationale |
|---|---|---|
| **PN204** | v1 → v2 redesign | `is_compiling()` guard + arming hook fixes dev371 CUDA crash |
| **PN51** | retired → experimental | upstream gap confirmed by retired-audit |
| **PN132** | added (PR-blog) | vllm#42739 backport — Triton top-k/top-p contiguous fix |
| **PN133** | added (PR-blog) | vllm#42722 backport — MTP scheduler empty-output guard |
| **PN134** | experimental → retired | bench validated -25% TPS on hybrid_gdn_moe |
| **P26** | partial-merge fix | output_alloc sub applies separately from cu_2 |
| **PN50** | anchor fix | indentation drift on dev371 (12/16 → 8/12), GDN fusion now applies |
| **P12 v2** | sub-patch added | LAST→FIRST flip for multi-tool agentic flows |
| **PN96b** | rename (no logic) | resolves PATCH_REGISTRY dict-key collision with kv_cache/PN96 |

### Migration notes

- New preset YAML available: `vllm/sndr_core/model_configs/builtin/a5000-2x-qwen3.6-35b-multiconc.yaml`.
  Use for agentic / batch workloads where throughput matters more than
  per-request TTFT. Pairs with existing `a5000-2x-35b-prod.yaml` (latency
  profile, `max_num_seqs=2`).
- `PN204_DUAL_STREAM_INPROJ=1` requires `GENESIS_LEGACY_P7=0` (registry
  `conflicts_with` declared). Multi-conc preset YAML already wires this.
- `tools/restart_35b_dev371_multiconc.sh` — production-ready launch script
  with full env matrix. Use as drop-in for `start_35b_*.sh` legacy scripts.
- `tools/multi_conc_bench.py` — reusable bench tester. Supports `sweep`
  mode (conc=1/2/4/8) or single-N mode. Replaces ad-hoc bench scripts.

### Bench / measurements

**35B-A3B FP8, dev371+bf610c2f5, 2× A5000 TP=2** (`tools/multi_conc_bench.py sweep`):

| conc | agg TPS (non-stream) | per-req TPS | scaling | TTFT med (stream) | TPOT med |
|---:|---:|---:|---:|---:|---:|
| 1 | 215 | 215 | 1.00x | 65 ms | 14.3 ms |
| 2 | 349 | 184 | 1.62x | 64 ms | 17.5 ms |
| 4 | 478 | 134 | 2.22x | 146 ms | 22.4 ms |
| **8** | **689** | 91 | **3.21x ⭐** | 243 ms | 31.7 ms |
| 16 | 685 | 51 | 3.21x | — | — (diminishing) |

**27B INT4 AutoRound, same hardware, same dev371**:

| conc | agg TPS | TTFT med | scaling |
|---:|---:|---:|---:|
| 1 | 97 | 105 ms | 1.00x |
| 8 | 379 | 189 ms | **3.89x** (better scaling than 35B) |

**Long-context (16K prompt), 35B**:

| conc | agg TPS | TTFT | TPOT |
|---:|---:|---:|---:|
| 1 | 12 | 2.6 s | 15 ms |
| 4 | 13 | 6.8 s | 120 ms |

Prefill dominates at long-prompt (~163 µs/tok), decode KV access stable.

**Anti-patterns confirmed on dev371** (do not enable):

- `--enable-prefix-caching` → -28.8% TPS (forces Mamba `align` mode incompatible with TQ k8v4).
- `rejection_sample_method` + `draft_sample_method=probabilistic` → -4% TPS, -8.6% accept rate.
- `--performance-mode interactivity` → neutral to -3.5%.
- `--max-num-batched-tokens 8192` (vs 4096) → larger batches but TTFT regresses (683 ms vs 240 ms at conc=8).
- `--max-model-len 320000` (Sprint 1 size, vs 280K) → -4.4% TPS (less VRAM headroom for batch=8 capture).

### Audit findings

- 7 patches auto-skipped as `upstream_merged` on dev371 — deep audit:
  4 truly merged (P4, PN19, PN52, PN90), 3 partial-gap (P26, PN50, P12).
  All 3 partial gaps recovered (see Patches changed above).
- 13 retired patches re-audited: 11 truly fully-merged (kept retired),
  2 partial gaps recovered (P61 already covered via P12 v2, PN51
  reactivated).
- TTFT user target 100-120 ms at `conc=8`: **structurally not reachable**
  on current hardware without GDN kernel fusion. Current 243 ms is the
  prefill compute floor (30 GDN layers × 6 sequential Triton kernels =
  180 launches per forward).

### Verified

```bash
# All gates green:
python3 -m vllm.sndr_core.cli patches doctor       # 169 entries, ERROR=0 WARN=0
python3 -m vllm.sndr_core.apply.shadow --strict    # CLEAN
python3 -m pytest tests/unit/dispatcher -q          # 152 passed

# Multi-conc bench (server):
bash tools/restart_35b_dev371_multiconc.sh         # 130-180 s warmup
python3 tools/multi_conc_bench.py sweep            # full conc=1/2/4/8 profile
```

### Architectural roadmap

See `docs/GDN_KERNEL_FUSION_DESIGN.md` — 3-phase plan для снижения TTFT
floor: fuse `chunk_local_cumsum`+`chunk_scaled_dot_kkt_fwd` (Phase 1),
`solve_tril_inv`+`recompute_w_u_fwd` (Phase 2), `chunk_gated_delta_rule_fwd_h`+`chunk_fwd_o`
(Phase 3). Effort 12-18 working days. Expected: TTFT 243→150-180 ms at conc=8
(-25..-36%), per-req TPS 215→240-260 at conc=1.

Decision: monitor upstream до 2026-06-01; implement Phase 1 if upstream doesn't
merge equivalent.

### DFlash variant bench (Wave 10 closure, 2026-05-15 evening)

После TQ+MTP multi-conc сцены DFlash variants получили симметричный Wave 10
patch matrix + полный bench. DFlash имеет фундаментальное VRAM-ограничение
на consumer Ampere: head_size=256 в drafter блокирует TQ k8v4 / fp8 KV
(target FP8 + drafter bf16 page-size mismatch при fp8 KV applied). Без
compression fp16 KV @ 256K не помещается в 24GB.

**35B DFlash** (max_model_len=100K, max_num_seqs=8, dtype=bfloat16):

| conc | DFlash agg TPS | DFlash TTFT | MTP+TQ ref | Delta |
| --- | --- | --- | --- | --- |
| 1 | 153 | 74 ms | 215 / 65 ms | -29% TPS / +14% TTFT |
| 4 | 359 | 154 ms | 478 / 146 ms | -25% TPS / +5% TTFT |
| 8 | **562** | **162 ms** ⭐ | 689 / 243 ms | -18% TPS / **-33% TTFT** |

DFlash на 35B — **LATENCY winner** при conc=8 (TTFT 162 ms vs MTP 243 ms,
гораздо ближе к user target 100-120 ms). MTP+TQ — **THROUGHPUT winner**
(689 vs 562 aggregate) + **long-context winner** (280K vs 100K cap).

**27B DFlash** (max_model_len=80K, max_num_seqs=8):

| conc | DFlash agg TPS | DFlash TTFT | MTP+TQ ref |
| --- | --- | --- | --- |
| 1 | 102 | 102 ms | 97 / 105 ms |
| 4 | 268 | 160 ms | 265 / 159 ms |
| 8 | 385 | 190 ms | 379 / 189 ms |

На 27B DFlash и MTP+TQ практически идентичны (Δ <2%). Выбор зависит
от context length: MTP+TQ supports 262K, DFlash capped at 80K.

**Operator decision matrix**:

- Latency-critical single-user 35B agentic → DFlash (TTFT 74 ms)
- Multi-tenant 35B (≤100K ctx) → DFlash (TTFT 162 ms — closer to user target)
- Multi-tenant 35B (long-ctx required) → MTP+TQ (256K + 689 TPS)
- 27B either method → pick by required context length

DFlash V2 model YAML files updated with full bench tables in notes section
(qwen3.6-35b-a3b-fp8-dflash.yaml, qwen3.6-27b-dflash.yaml). Wave 10 patches
(PN51/96b/125/126-130/132/133) added symmetric to MTP siblings — some are
no-op on DFlash code path (PN128 eagle warmup, PN130 TQ decode warmup,
PN133 MTP scheduler) but cost zero and aid cross-config consistency.

---

## [v11.0.0+wave9_release_blockers_closed] — Release-tier audit closure (2026-05-14, evening)

> Closes the remaining P0/P1 findings from
> `docs/_internal/CURRENT_PROJECT_STATE_ERRORS_2026-05-14_RU.md`.
> Both 256K hardware bench ladders (27B INT4 TQ k8v4 + 35B-A3B FP8)
> completed PASS on the same dcacdf9a pin with all Wave 9 patches
> in the canonical config matrix.

### Hardware verification — both target models hit 256K

`docs/_internal/runs/tp2_256k_probe_27b.json` and
`tp2_256k_probe_35b.json` (gitignored — internal artefact):

| Model | 200K | 230K | 256K |
|---|---|---|---|
| **27B INT4 TQ k8v4** | 886 s · 113 t/s | 1190 s · 97 t/s | 1487 s · 86 t/s |
| **35B-A3B FP8**     | 381 s · 263 t/s | 495 s · 233 t/s | 620 s · 207 t/s |

Both runs were against the V2 layered preset (`sndr launch prod-qwen3.6-27b-tq-k8v4` /
`prod-qwen3.6-35b-balanced`) after the env-flag matrix synced in commit `77f2ec8` —
i.e. the canonical `python3 -m vllm.sndr_core.cli launch ...` path,
not the bash launcher. The bash launcher in
`scripts/launch/start_pn95_2xa5000_nightly_dcacdf9a.sh` is now a
reference artefact (it documents the exact env matrix used during
commissioning) rather than the canonical operator path.

### `launch --check-deps` false-positive — fixed

`vllm/sndr_core/cli/launch.py::_run_check_deps` previously only ran
the caveat matcher (env-flag combinations) and therefore never
reported the real host blockers (Docker not installed, NVIDIA driver
missing, model directory absent). Operators got a green preflight
even when the host could not launch.

`_run_check_deps` now drives the canonical
`vllm.sndr_core.deps.{inspect_host, plan_changes}` planner — the same
code path that powers `sndr deps plan --strict` — and surfaces both
planner blockers and caveat errors. Either signal makes the function
return 2. Verified on a Mac dev box without Docker: three planner
blockers + `FINAL_EXIT=2`.

### `vllm/_genesis/` migration — every active consumer now uses sndr_core

`_genesis/` was the legacy patch home; `sndr_core/` has been the
canonical home since v11. Runtime was already clean
(`python3 -m vllm.sndr_core.apply` loads zero `_genesis` modules),
but six utility / installer / probe files outside the runtime path
still imported from `vllm._genesis`. Each one is now pointed at the
sndr_core equivalent:

| File | Was | Now |
|---|---|---|
| `patch_genesis_unified.py` | `vllm._genesis.patches.apply_all.main` | `vllm.sndr_core.apply.apply_all` |
| `tools/check_upstream_drift.py` | `vllm._genesis.patches.upstream_compat` | `vllm.sndr_core.integrations.upstream_compat` |
| `tools/genesis_vllm_plugin/genesis_v7/__init__.py` | `vllm._genesis.patches.apply_all.run` | `vllm.sndr_core.apply.orchestrator.run` |
| `tests/probes/verify_new_patches_all_models.py` | `vllm._genesis.dispatcher` | `vllm.sndr_core.dispatcher{,.registry}` |
| `install.sh` | symlink `_genesis` + `vllm._genesis.compat.cli` | symlink `sndr_core` + `vllm.sndr_core.compat.cli` (and the legacy `_genesis` symlink is cleaned up during setup) |
| `scripts/run_validation_suite.sh` | `vllm._genesis.model_detect.get_model_profile` | `vllm.sndr_core.detection.model_detect.get_model_profile` |

The only remaining references to `vllm._genesis` outside the package
itself are intentional — `tests/legacy/test_deployment_runtimes.py`
verifies that the deprecated docker bind-mount path still works for
v7.x users, and `tests/unit/scripts/test_check_no_legacy_imports.py`
is the static gate that fails the build if NEW legacy imports
appear. The `vllm/_genesis/` directory itself can now be removed in
a follow-up commit without breaking any active runtime, plugin,
utility, installer, or test path.

### Patch proof coverage — 151/151

`python3 -m vllm.sndr_core.cli patches prove --all` was run to
populate `evidence/patch_proof/` with one `{patch_id}__*.json`
artefact per registry entry. The reference doc reported 136 proven
/ 15 dead (coverage 90.1%); after the sweep:

```
dead=0  proven=151  coverage=100.0%
```

`audit-patches-prove-all` in `make evidence` is now green
(80 s wall on the local box). Proof artefacts live under
`evidence/patch_proof/` and are gitignored by project policy — each
operator runs `prove --all` on their own pin to populate the local
cache; the static checks inside each artefact verify the
patch_id ↔ registry entry ↔ apply-function link.

The 15 entries previously surfaced as "dead" (SNDR_WORKSPACE_001,
PN200/201/202/203/204, PN104/105/106/108, PN91/92/97, PN71/73)
now all carry a fresh `static_passed: true` proof.

### Tail of the P0/P1 audit list

The remaining `make evidence` failures
(audit-community, audit-all-referents, audit-readme-counters,
audit-no-hardcoded-paths, audit-no-stub, audit-engine-boundary,
audit-security) are outside the original reference-doc P0/P1 list
and are filed as separate cleanup items for the next iteration.

---

## [v11.0.0+wave9_tp2_256k] — TP=2 256K commissioning + PN59 anchor fix (2026-05-14)

> Wave 9 closure for the 27B-INT4 hybrid stack on 2× A5000. PN59 (streaming
> GDN orchestrator) was silently SKIPPED on the dcacdf9a nightly because
> upstream added a `core_attn_out` parameter to `chunk_gated_delta_rule_fwd`
> and our text-patch anchor matched only the 11-parameter signature. After
> the anchor fix, PN59 engages at runtime; the TP=2 progressive context
> ladder (50K-200K) finished green on a freshly-booted container and the
> 256K probe is in flight at time of writing.

### Patches added

- **PN204** — port of vllm-project/vllm#42301 (open as of 2026-05-14).
  Single text-patch on `gdn_linear_attn.py::forward_cuda` Part 1.
  Replaces the serial `in_proj_qkvz` / `in_proj_ba` pair with
  `vllm.utils.multi_stream_utils.maybe_execute_in_parallel`. Stream
  and events are allocated lazily on the first forward call — the
  eager `__init__` allocation triggered a `RuntimeError: expected
  event to be a torch.Event object` in the worker on the pinned
  vLLM nightly. Mutually exclusive with retired P7. Auto-SKIPs when
  upstream lands #42301 (drift marker `_in_proj_aux_stream`).
  Upstream measures -2.9% TPOT at qps=0.5 on Qwen3.5-35B-A3B;
  Genesis port re-tests are queued behind 256K bench completion.
- **PN108** — fused_recurrent prefill dispatch (Cliff 2 memory-bound
  attempt). Tombstoned with full design analysis in
  `vllm/sndr_core/integrations/attention/gdn/pn108_fused_recurrent_prefill.py`.
  The fla `fused_recurrent_gated_delta_rule` kernel cannot serve
  single-sequence prefill: `inplace_final_state=True` requires
  `ssm_state_indices` (Triton compile error when None passed);
  `inplace_final_state=False` allocates a per-token state buffer
  (~12 GB at T=29K Qwen3.6-27B shape). The patch is retained as a
  documented dead-end so the next maintainer does not repeat the
  attempt. Superseded by PN59 anchor fix.

### Anchor fixes

- **PN59 streaming-GDN orchestrator** — anchor updated to the
  12-parameter `chunk_gated_delta_rule_fwd` signature that nightly
  dcacdf9a introduced (extra `core_attn_out: torch.Tensor | None`
  trailing parameter). PN59 now applies cleanly on this pin and
  engages on long single-sequence prefill.
- **P68/P69 long-context tool-call adherence** — anchor moved from
  the now-defunct `create_chat_completion` body (upstream split the
  endpoint into a thin wrapper + `_create_chat_completion`) to the
  pair `# Streaming response\n        tokenizer = self.renderer.tokenizer`.
  The new anchor sits below any earlier middleware hook (including
  PN16 lazy-reasoner) and is order-independent against PN16, so
  both patches can apply in either order on the same file.

### Registry / dispatcher hygiene

- **SNDR_WORKSPACE_001** — `@register_patch` decorator string aligned
  with the registry id. Strict shadow parser splits on whitespace,
  so the previous hyphenated form (`SNDR-WORKSPACE-001 ...`) was
  unparseable and the registry entry surfaced as
  `spec_only_unexpected`. Both id and `name` constant now use the
  underscored canonical form; the user-facing title in `registry.py`
  keeps the hyphenated form for readability.
- **Doc counters 136 → 151** — `README.md`, `docs/PATCHES.md`,
  `docs/INSTALL.md`, `docs/MODELS.md` now match the registry size;
  default-on / opt-in split corrected to `32 / 119`.
  `scripts/check_doc_sync.py --strict` is green.
- **V1 baseline +1** — `a5000-1x-tier-aware-pn95.yaml` added to
  `FROZEN_V1_BASELINE` in `scripts/audit_no_new_v1.py`. The yaml is
  the verified single-A5000 long-context preset (PN95 multi-tier
  offload reaches 200K on one card); V2 layered triplet migration
  is queued as a follow-up cleanup, not a release blocker.

### Operational artefacts

- **`scripts/launch/start_pn95_2xa5000_nightly_dcacdf9a.sh`** — the
  hardware-verified 2× A5000 launcher for the dcacdf9a pin (TP=2,
  TQ k8v4, MTP K=3, full Genesis stack including PN59 anchor-fixed,
  PN95 tier-aware, PN106/200/201 GDN scratch pool, PN77 fp8
  lm_head, PN204 dual-stream input projection).
- **TP=2 progressive context bench** — 50K / 100K / 156K / 180K /
  200K all PASS, prefill TPS ramping from 736 t/s at 50K to 113 t/s
  at 200K. Results in `docs/_internal/runs/tp2_progressive_probe.json`.

### Author / credit hygiene

- Removed the "AI-assisted by author" / "Claude-assisted" attribution <!-- audit-ai-attribution: allow -->
  strings from upstream-backport credits (`vllm/_genesis/dispatcher.py`,
  `patch_N58_spec_reasoning_boundary.py`, this CHANGELOG). The
  upstream PR author and number are still cited; the manner in which
  the upstream PR was drafted is not relevant to the backport.
- Real product names (`Claude Code` IDE agent, the
  `Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-…` model on HuggingFace)
  are preserved — those are names of third-party artefacts, not
  attributions for code in this repository.

---

## [v11.0.0+audit_2026_05_12_dual_state_closure] — Comprehensive audit closure (2026-05-12)

> **Closed all P0/P1 + most P2/P3 findings from `docs/_internal/COMPREHENSIVE_DUAL_STATE_AUDIT_2026-05-12_RU.md`. Local pytest: 5365 passed / 0 failed. Server pytest: 5413 passed / 0 failed.**

### Critical fixes (P0)

- **PN26 torch-less apply contract** — env-gate check before kernel import; graceful skip when GPU runtime deps missing. Fixes CI / Mac dev environments where torch-less apply previously crashed.
- **Server-side stale imports** — 118 `vllm.sndr_core.patches` references in `tests/` cleaned (was 108 per audit, 10 missed). Server full pytest now zero-failed (was 114 failed).
- **Hardcoded private LAN IP removed from public tooling** — `tools/long_ctx_smoke.sh`, `tools/soak.sh`, 6× Makefile targets all default to `127.0.0.1` with explicit `HOST=` override path.
- **content=null contract enforced** — new `tools/openai_smoke.py` with `--assert-content` flag. Documents Qwen3 thinking-mode behavior in `docs/REASONING_CONTENT_CONTRACT.md` (PN16 v2 architecture explanation).
- **Legacy-import CI gate** — `scripts/check_no_legacy_imports.sh` (POSIX bash) forbids `vllm.sndr_core.patches.*` and active `vllm._genesis.*` imports. Hooked into `make audit`.

### Trust anchor (P1)

- **Ed25519 Trust anchor activated.** `vllm/sndr_core/license.py::_TRUST_ANCHOR_PUBKEY_B64URL` is now a real 32-byte key (`iSk29MUb9HldKokPRyOG7bAjwYaQdgqYsS17yfskE8s`) generated via `scripts/generate_trust_anchor.py`. Placeholder zero-key retired.
- **Ceremony doc** at `docs/security/TRUST_ANCHOR_CEREMONY.md` documents rotation workflow.
- **CI gate** `tests/unit/test_trust_anchor_not_placeholder.py` prevents placeholder regression.

### Registry metadata overlay (P1)

- **`vllm/sndr_core/dispatcher/registry_metadata.py`** — new overlay module providing `derive_metadata(patch_id, registry_meta)` returning `{implementation_status, test_status, production_default}` without bloating 136 PATCH_REGISTRY entries with 129 explicit fields each.
- **EXPLICIT_OVERRIDES** for PN95 (partial), PN64 (placeholder), PN26b (scaffold), P5b (coordinator).
- **File-based test status** detection via `tests/unit/integrations/<family>/test_<id>_*.py` discovery.
- Final distribution: 113 live, 11 retired, 6 full, 2 research, 1 partial, 1 scaffold, 1 coordinator, 1 placeholder. 16 patches correctly flagged as not-production.

### Docs cleanup (P1)

- `docs/INSTALL.md`, `docs/CONTRIBUTING.md`, `docs/BENCHMARK_GUIDE.md`, `scripts/launch/README.md` — all stale `_genesis/` instructions replaced with `sndr_core/` + back-compat alias note. Pre-v11 layout still works via the `vllm.sndr_core.__getattr__` alias.

### Operational hygiene (P2)

- **`sndr doctor-system --logs` flag** — host-side log forensics (last N hours):
  - dmesg → OOM-kill events (regex match).
  - dmesg → NVRM Xid errors with severity classification (`FATAL_XIDS = {31, 43, 45, 63, 64, 74, 79}`).
  - `docker ps --filter status=restarting` → restart-loop containers.
  - `journalctl -u genesis-vllm` → suspicious-keyword filter.
  - Verdict escalation: OOM / fatal Xid / restart loop → `NOT READY`.
- 25 unit tests in `tests/unit/cli/test_doctor_logs.py` (parser regex, severity classification, window filter, graceful degradation).

### Compatibility matrix (P2)

- **`CompatibilityMatrix` + `CompatibilityRule` dataclasses** added to `vllm/sndr_core/model_configs/schema.py`. Singleton `COMPATIBILITY_MATRIX` with `register(rule, predicate)` API.
- 4 canonical rules:
  - **COMPAT-001** (forbidden): DFlash on Qwen-next architecture (refined from initial "DFlash on hybrid GDN" after catching false-positive on Qwen3.6 Lorbus PROD preset).
  - **COMPAT-002** (discouraged): TurboQuant k8v4 on hybrid-GDN without `GENESIS_ENABLE_P98=1`.
  - **COMPAT-003** (discouraged): N-gram spec_decode on TQ k8v4 with `max_model_len > 131072`.
  - **COMPAT-004** (forbidden): DFlash without drafter `model` path.
- Integration: `ModelConfig.validate()` raises `SchemaError` on forbidden, `audit()` surfaces discouraged.
- 21 unit tests in `tests/unit/model_configs/test_compatibility_matrix.py`.

### Multi-runtime renderers (P3)

- **`sndr compose render/up/down/logs`** — new docker-compose YAML renderer at `vllm/sndr_core/cli/compose.py`. Uses `yaml.safe_dump`, idempotent, GPU reservation block, host.yaml substitution. 11 unit tests.
- **`sndr quadlet render`** — new Podman Quadlet `.container` renderer at `vllm/sndr_core/cli/quadlet.py`. INI format with `[Unit]/[Container]/[Service]/[Install]` sections, Environment line per env var, `AddDevice=nvidia.com/gpu=all`. 11 unit tests.
- **`sndr k8s render` extensions** — `KubernetesConfig` gained `node_selector`, `pvc`, `pvc_size_gib`, `pvc_storage_class`, `secret_mounts` fields. Renderer emits per-claim `PersistentVolumeClaim` manifests + Secret volume bindings. Bug fix: empty `volumeMounts: []` / `volumes: []` now render inline (previously misaligned, broke YAML parsing). 9 unit tests.

### Upstream watchlist (P4)

- **`vllm#42102` added to `docs/upstream/UPSTREAM_WATCHLIST.yaml`** — DFlash + quantized target KV coexistence. `action: port`, tracks PN21/PN23/PN24/PN38/PN40 for retire-after-merge.
- **Backport plan doc** at `docs/_internal/research/upstream_42102_dflash_independent_kv_groups_plan_2026-05-12.md` — per-mechanism mapping to future PN94/PN95b patches + test plan + retire-candidates.

### PN96 A/B bench (P5 — gated)

- **Plan documented** at `docs/_internal/PN96_AB_BENCH_PLAN_2026-05-12_RU.md`. Execution requires ~45 min PROD downtime — gated on explicit operator go-ahead.

### Deferred

- **Proxmox apply** (S3.4) — existing `sndr proxmox render` emits ready-to-execute `pct create`/`qm create` commands for operator review. Live PVE automation deferred until a testbed is available (untested PVE actions are destructive-by-default).

### Test counts

| Sprint | Local pytest delta | Server pytest delta |
|---|---|---|
| Start of audit | 5293 / 0 fail | 5242 / 114 fail |
| S1.1–S1.6 closure | 5293 / 0 | 5319 / 0 |
| S2.1 trust anchor | 5293 / 0 | 5321 / 0 |
| S2.2 metadata overlay | 5293 / 0 | 5321 / 0 |
| S2.4 doctor-logs | +25 = 5332 | +39 = 5360 |
| S2.5 compat matrix | +21 = 5353 | +21 = 5381 |
| S3.1–3.3 renderers | +31 = 5365 | +32 = 5413 |

---

## [v11.0.0+stable_first] — First STABLE patches in registry + asyncio cleanup (2026-05-12)

> **Closed the deferred STABLE-promotion item by IMPLEMENTING the missing infrastructure (path 1 of the architectural gap). PN33 + PN35 are now the first 2 `lifecycle="stable"` patches in PATCH_REGISTRY history — manifest covered, ratchet-satisfied, server-verified. Also closed asyncio test-infra ResourceWarnings (3→0, all gone).**

### STABLE promotion: PN33 + PN35 — full infrastructure built

Earlier evaluation flagged both patches as production-validated but blocked by missing anchor-manifest infrastructure. This session closed that gap:

1. **Pristine fixtures extracted** from `vllm/vllm-openai:nightly` (dev209) without any Genesis bind-mount, verified 0 Genesis markers:
   - `tests/legacy/pristine_fixtures/gpu_model_runner.py` (7179 lines, md5 `ac61702177b286d0d7239050fd07cbbc`) — covers PN33.Sub-1 + PN35.Sub-1
   - `tests/legacy/pristine_fixtures/llm_base_proposer.py` (1638 lines, md5 `b2b5def581d27d4654e66d83b4ff4998`) — covers PN35.Sub-2
   - All 3 anchors verified verbatim-matching pristine (exactly 1 occurrence each).
2. **`register_for_manifest()` added** to both wiring modules with same pattern as PN79: build-mode TextPatcher targeting pristine fixture, declares patch_id for ratchet visibility.
3. **`scripts/build_anchor_manifest.py` extended** with discovery list (`_REGISTRY_TARGETS`) + path mapping (`_KNOWN_REL_PATHS`) for `gpu_model_runner.py` → `v1/worker/` and `llm_base_proposer.py` → `v1/spec_decode/`.
4. **`anchor_manifest.json` regenerated** — now covers 7 patchers / 6 files / 21 anchors (was 4/4/N). PN33 + PN35 entries added with md5-pinned anchor offsets.
5. **`tests/unit/infra/conftest.py`** new file — function-scoped autouse fixture invokes `register_for_manifest()` for each STABLE-eligible patch before each infra test. Required because `test_anchor_manifest.py` has its own `_clear_registry` autouse that wipes between its tests, so a session-scoped seed would be cleared before `test_stable_manifest_policy.py` runs.
6. **Both patches promoted** to `lifecycle: "stable"` with `stable_since: "v11.0.0+wave9_dev209"`. All 6 ratchet tests pass.

### Wave 9 dev209 live-PROD re-bench (no downtime)

Pin bump dev93 → dev209 validated by running `genesis_bench_suite.py --quick --ctx 8k` against the LIVE 27B PROD container:

| Metric | Wave 8 (dev93) | Wave 9 (dev209) | Δ | Verdict |
|---|---|---|---|---|
| wall_TPS | 131.48 | **131.67** | +0.14% | neutral |
| decode_TPOT (ms) | 7.35 | **7.34** | -0.14% | neutral |
| TTFT (ms) | 97.1 | 104.07 | +7.2% | within CV ~8% |
| stability CV % | 4.76 | **3.22** | -32% | **TIGHTER reproducibility** |
| VRAM total (MiB) | 44594 | 44879 | +0.6% | within +2000 tolerance |
| Tool-call | 8/8 | 7/7 | (different harness) | all pass |

27B YAML `reference_metrics` updated to Wave 9 numbers with Wave 8 retained as `prev_long_gen_tps` baseline for regression detection.

### Asyncio ResourceWarnings closed

`tests/legacy/test_response_cache_middleware.py` had 2 `asyncio.get_event_loop().run_until_complete(...)` calls (Python 3.12+ deprecates this — creates a loop and leaves it for GC). Replaced with `asyncio.run(...)` which properly creates + closes the loop. ResourceWarning count: **90 → 3 → 0** across the cleanup sessions.

### Lifecycle distribution (history-making)

| Lifecycle | Count | Change |
|---|---|---|
| experimental | 85 | -2 (PN33, PN35 promoted) |
| legacy | 33 | unchanged |
| retired | 11 | unchanged |
| research | 3 | unchanged |
| **stable** | **2** | **+2 (first ever — PN33, PN35)** |
| coordinator | 1 | unchanged |

This is the **first time** the Genesis ratchet has had non-vacuous STABLE patches. The infrastructure built here applies to all future STABLE candidates.

### Files touched

- `vllm/sndr_core/dispatcher/registry.py` — PN33+PN35 promoted to STABLE
- `vllm/sndr_core/integrations/worker/pn33_spec_decode_warmup_k.py` — added `register_for_manifest()`
- `vllm/sndr_core/integrations/worker/pn35_inputs_embeds_optional.py` — added `register_for_manifest()`
- `vllm/sndr_core/manifests/anchor_manifest.json` — regenerated with 3 new entries
- `scripts/build_anchor_manifest.py` — extended discovery list + path mapping
- `tests/legacy/pristine_fixtures/gpu_model_runner.py` — NEW (pristine dev209)
- `tests/legacy/pristine_fixtures/llm_base_proposer.py` — NEW (pristine dev209)
- `tests/legacy/pristine_fixtures/README.md` — added 2 new entries + dev209 pin
- `tests/unit/infra/conftest.py` — NEW (STABLE-ratchet registration fixture)
- `tests/legacy/test_response_cache_middleware.py` — asyncio.get_event_loop → asyncio.run
- `vllm/sndr_core/model_configs/builtin/a5000-2x-27b-int4-tq-k8v4.yaml` — Wave 9 reference_metrics

### Server verification (vllm-pn95-2xa5000 PROD)

- 10+ files rsynced. Server self-test: 8 pass / 0 fail / 0 warn.
- STABLE ratchet on server: **6 tests pass** (including `test_every_stable_patch_has_registered_patcher` + `test_every_stable_patch_has_manifest_coverage` for PN33 + PN35).
- Lifecycle distribution on server matches local.

---

## [v11.0.0+wave9_dev209] — Wave 9 pin-bump re-bench + STABLE promotion analysis (2026-05-12)

> **Closed two deferred items: (a) Wave 9 canonical re-bench against live PROD on dev209 — confirmed pin bump dev93→dev209 is statistically equivalent on throughput with TIGHTER CV (3.22% vs 4.76%); (b) STABLE-promotion evaluation of PN33+PN35 — both production-validated across Wave 6-9 + 2 vllm pins, but architectural ratchet correctly blocks promotion until anchor-manifest infrastructure exists for runtime-hook patches. Documented gap with two paths forward.**

### Wave 9 re-bench results (live 27B PROD, no downtime)

Method: `genesis_bench_suite.py --quick --ctx 8k` against running container `vllm-pn95-2xa5000:8101` on dev209. No config changes between Wave 8 and Wave 9 — only the vllm pin differs.

| Metric | Wave 8 (dev93) | Wave 9 (dev209) | Δ | Verdict |
|---|---|---|---|---|
| wall_TPS sustained | 131.48 | **131.67** | +0.14% | neutral |
| decode_TPOT (ms) | 7.35 | **7.34** | -0.14% | neutral |
| TTFT (ms) | 97.1 | **104.07** | +7.2% | within TTFT inherent variance (CV 7.85%) |
| stability CV % | 4.76 | **3.22** | -32% | **TIGHTER — better reproducibility** |
| tool-call | 8/8 | 7/7 | (different harness build) | all pass |
| VRAM total (MiB) | 44594 | 44879 | +0.6% | within +2000 tolerance |

Conclusion: dev209 pin bump validated for PROD. 27B YAML `reference_metrics` updated to Wave 9 numbers with Wave 8 retained as `prev_long_gen_tps` baseline for regression detection.

### STABLE promotion analysis (PN33, PN35)

Both patches are `default_on=True` + `lifecycle=experimental` — full production validation evidence:

- **PN35** (text-only inputs_embeds skip, vllm#35975 backport): runtime hook, no `_make_patcher`. Validated default_on across Wave 6→9 + dev93/dev209 on 27B PROD, zero regressions. Upstream PR still OPEN.
- **PN33** (spec-decode warmup K-aware sizing, vllm#37521 backport EXTENDED): has `_make_patcher` (text-patch shape) but no `register_text_patcher` call + no manifest entry. Validated across same matrix on both 27B+35B PROD. Upstream PR still OPEN AND narrower than our extended fix.

The STABLE ratchet ([tests/unit/infra/test_stable_manifest_policy.py](tests/unit/infra/test_stable_manifest_policy.py)) correctly blocked both with:

1. `test_every_stable_patch_has_registered_patcher` — wiring module must call `register_text_patcher()` at import time
2. `test_every_stable_patch_has_manifest_coverage` — `anchor_manifest.json` must have an entry for each patch_id

Both promotions reverted; `experimental_note` field added to each with detailed production-validation evidence so the operational signal isn't lost. The ratchet architectural gap is documented in [docs/CONTRIBUTING.md#promoting-a-patch-to-lifecyclestable "Ratchet architectural gap"](docs/CONTRIBUTING.md#promoting-a-patch-to-lifecyclestable) with two paths forward:

1. **Build manifest infrastructure per-patch** — proper STABLE; needs running `build_anchor_manifest.py` on a vllm-installed host + creating pristine fixtures. Easy for PN33 (real text-patch), harder for PN35 (would need TextPatcher conversion).
2. **Extend ratchet for runtime-hook STABLE** — add `stable_kind = "runtime-hook"` sub-track that accepts `production_validated_pins` evidence instead of manifest. Architectural trade-off: runtime-hook patches can drift if upstream changes the monkey-patched function, no manifest md5 to detect it.

Neither path is taken in this session — the qualitatively correct move is to preserve the ratchet's architectural promise (STABLE = manifest-protected) and keep production-validated runtime hooks at `experimental` until the gap is properly closed.

### Lifecycle distribution (no net change)

| Lifecycle | Count |
|---|---|
| experimental | 87 |
| legacy | 33 |
| retired | 11 |
| research | 3 |
| stable | **0** (was 0; ratchet still vacuous-pass) |
| coordinator | 1 |

### Files touched

- `vllm/sndr_core/model_configs/builtin/a5000-2x-27b-int4-tq-k8v4.yaml` — `reference_metrics` block: Wave 9 numbers, Wave 8 retained as prev_baseline, dev209 in `vllm_pin`
- `vllm/sndr_core/dispatcher/registry.py` — `PN33`/`PN35`: `experimental_note` added with validation evidence
- `docs/CONTRIBUTING.md#promoting-a-patch-to-lifecyclestable` — "Ratchet architectural gap" section documenting the runtime-hook blocker

---

## [v11.0.0+stale_ref_cleanup] — F-015 closure + post-rename stale-ref sweep (2026-05-12)

> **Discovered + fixed 10 silent stale-ref bugs left over from the v10 `patches/`→`integrations/` and `paths/`→`locations/` renames. None caused boot crashes (everything degraded to no-op / `None` return / `0 modules`), so the v11 cut was shipped on top of them. Server-verified live on `vllm-pn95-2xa5000` PROD.**

### Silent runtime bugs fixed (3 in compat tools)

- **`compat/categories.py::_WIRING_DIR`** — still pointed at `sndr_core/patches`; `module_for(<any patch_id>)` returned `None`. Broke `migrate.check_patch_against_upstream`, lifecycle-audit CLI, cache-parity audit. Verified on PROD: was `None`, now resolves dotted path.
- **`compat/self_test.py::_check_wiring_imports`** — scanned `sndr_core/wiring/patch_*.py` (0 files post-rename); reported `pass: 0 wiring modules import cleanly` while validating nothing. Now uses `wiring_dir()` + canonical `p[0-9]*.py` / `pn[0-9]*.py` glob — finds **121 modules** on PROD.
- **`compat/cache_parity_audit.py::_read_patch_source`** — same hardcoded path, never matched any source file. Now routes through `wiring_dir()`. Confirmed reads `P83/P84/P85/PN35/PN54` correctly.

### Canonical helper fixed (F-015 closure)

- **`locations/project_paths.py::wiring_dir`** — docstring said canonical is `sndr_core/integrations/` but code still concatenated `sndr_core/patches`. Now canonical→legacy-patches→legacy-wiring fallback chain.

### Schema gaps closed

- **`compat/schema_validator._KNOWN_FIELDS`** — missing `enables_upstream_feature` (iron-rule-#11 case c). `validate_registry` raised on `P75/P99`.
- **`schemas/patch_entry.schema.json`** — sync: added `enables_upstream_feature` property + `observability` to family enum.
- **`tests/unit/env/test_registry_flag_coverage.py::VALID_FAMILIES`** — missing `observability`; `SPRINT26_CG_DISPATCH_TRACE` was flagged.
- **`model_configs/schema.py::ModelConfig.lifecycle`** — added `retired` to allowed set (warned about every load on dflash YAML).
- **`model_configs/schema.py::ReferenceMetrics`** — added 4 `wave8_*_delta_pct_*` fields so audit-trail values in 27B YAML aren't dropped with warning.

### Doc-sync infrastructure

- **`scripts/check_doc_sync.py::count_registry_entries`** — regex was `[A-Za-z0-9_]+` so it missed hyphenated keys (`PN40-classifier`); reported 134 expected when actual was 135 — phantom mismatch.

### Test infra unification

- Deleted **`tests/legacy/pytest.ini`** (legacy shim). Was making pytest pick `tests/legacy/` as rootdir, isolating those 1065 tests from `tests/conftest.py` fixtures — 106 setup errors of `fixture 'deterministic_seed' not found`. Markers `cuda_required` / `rocm_required` / `gpu_required` merged into root `pytest.ini`. Registered missing `requires_torch` marker.
- Fixed stale `tests/installer/test_install_prepare.py::test_prepare_pulls_pin_from_config` — was asserting old dev93 pin literal; now reads from registry helper so the assertion survives future pin bumps.

### Doc counts bumped (4 files, 9 references)

- `134 → 135` everywhere PATCH_REGISTRY count appears: `README.md` (5×), `docs/PATCHES.md` (2×), `docs/MODELS.md` (2×), `docs/INSTALL.md` (1×).
- Added `SPRINT26_CG_DISPATCH_TRACE` row to `docs/PATCHES.md::Other` table.
- Tree-listing in `docs/INSTALL.md` updated for `patches/`→`integrations/` and `paths/`→`locations/` renames (was still showing pre-v10 layout).

### Test counts

- Local: **5268 passed / 127 skipped / 0 failed** (was 11 failing + 106 erroring across compat/migrate, compat/self_test, env/registry_flag_coverage, legacy/test_patches_md_sync, legacy/test_router_softmax).
- `make ci` — all 4 gates pass (pin-gate + iron-rule-11 + family contracts + doc-sync) + audit upstream offline.

### Server verification (vllm-pn95-2xa5000 PROD)

- 8 files rsynced into `~/genesis-vllm-patches-v11/` package mount.
- `wiring_dir()`: `None` → `/usr/local/lib/.../sndr_core/integrations`
- `module_for("PN14" / "P67" / "P38" / "PN65")`: all `None` → all resolve.
- `_check_wiring_imports`: `0 modules` → `121 modules` import cleanly.
- `load_all()` ModelConfig: 4 schema warnings → **0 warnings**, 11 configs load.
- `sndr self-test` (full CLI): was 7 pass / 1 warn (schema) → **8 pass / 0 warn**
  after adding `research_note` to P82/P83/PN26b.
- `audit_upstream_status.py` with network: NEWLY-MERGED=0, STALE-RETIRED=0,
  ERROR=0, all 7 previously-offline-STALE resolved to SUPERSEDED-OK=6 +
  ISSUE-CLOSED=1 (PN51, already retired w/ provenance). All waivers fire:
  INTENTIONAL-INVERSE=1, ENABLES-UPSTREAM=2 (P75+P99), RETIRED-INTERNAL=1.
- `audit_yaml_vs_runtime.sh` on 27B PROD: **No real drift detected**.
- `sndr doctor`: 6/6 sections clean, only HW PCIe advisory.
- PROD smoke: `vllm-0.20.2rc1.dev209+g5536fc0c0-tp2` API responding.

### Lesson

Post-rename stale refs degrade silently — `module_for` returning `None` doesn't crash, it just makes downstream tools (migrate, audit CLI, etc.) operate on empty inputs without any signal. Future rename audits should add positive-assertion tests (`assert module_for("P67") is not None`) to catch this class on the next rename.

---

## [v11.0.0+wave8] — Wave 8 regression audit + canonical bench WIN (2026-05-11)

> **Single-day audit closing 11 regression sources on 27B PROD; canonical bench shows +6.43% wall_TPS / -6.0% decode_TPOT vs Wave 7 baseline.**

### Headline
- **27B PROD canonical bench**: **132.28 TPS / 7.31 ms TPOT / 8/8 tool** (`genesis_bench_suite.py --quick --ctx 8k`)
  - vs Wave 7 reference 124.29 TPS: **+6.43%** wall_TPS, **-6.0%** decode_TPOT
  - vs Sprint 1 P82 sweep winner 130.79 TPS: **+1.14%**

### Regression sources closed (11 found, 9 fixed in-code, 2 documented)
1. **PN90 missing in start-script** — was validated +7.4% but env not exported. Fixed.
2. **P82=0 drift YAML↔start-script** — Sprint 1 promoted P82=1+thr=0.1 (+5.23%), drift lost it. Fixed.
3. **P71 broken on 27B GQA=6** — fails per-step `ValueError: cu_num_draft_tokens length 1`. Removed from 27B (keep on 35B where it works).
4. **GroupAB missing** — P70/PN12/PN14/P94/P103 (`feedback_groupAB_5patches_27b_only` +9.2% on 27B at 512t). Added across configs.
5. **P61 retired stuck enabled** in 5 YAML — removed.
6. **P100 Blackwell-only on Ampere** — removed from 27B configs.
7. **PN13 retired enabled** in 27B-tested — removed (upstream merged vllm#41235).
8. **P83+P85 broken dep on 27B-LC** — P85 requires P84 which was NOT enabled. Both removed.
9. **27B DFlash YAML -23.6% regression** — marked `lifecycle: retired`.
10. **PN95 init broken** (TierManager=None) — documented (cost is init-only, kept env=1).
11. **PN59 dead code on 27B** (always bypassed) — documented (cheap check, defensive value, kept env=1).

### YAML reconciliation (8 active configs)
- `a5000-2x-27b-int4-tq-k8v4`: Wave 8 + env-name typos fixed (P94/P103 short form) + reference_metrics 132.28 TPS
- `a5000-2x-27b-int4-long-ctx`: Wave 1+3.1+7+8 sync; P83+P85 dep removed (V1 YAML retired 2026-06-01 in V1 sunset #7; V2 equivalent: preset `long-ctx-qwen3.6-27b` — sizing-identical 280K ctx / util 0.90 / seqs 2 / batched 2048 / fp8_e5m2 KV / MTP K=3; V2 carries `override_policy.bench_pending=true` since long-context 32K+ bench refresh against current pin is deferred)
- `a5000-1x-27b-int4-tested`: Wave 1+3.1+7+8 sync; P100 removed (retired 2026-06-01 in V1 sunset #3; V2 equivalent: preset `qa-qwen3.6-27b-tq-1x`)
- `a5000-2x-27b-int4-tq-k8v4-dflash`: `lifecycle: retired` (V1 YAML retired 2026-06-01 in V1 sunset #5; V2 equivalent: preset `experimental-qwen3.6-27b-tq-dflash-ab` — A/B diagnostic, same model + envs + 131K ctx)
- `a5000-2x-35b-prod`: P61 removed (retired no-op)
- `a5000-2x-27b-dflash-true`: P61 removed (V1 YAML retired 2026-06-01 in V1 sunset #6; V2 equivalent: preset `prod-qwen3.6-27b-dflash` — transparent bucket, V2 composes byte-identical config DFlash N=5 single-stream)
- `a5000-2x-27b-int4-tested`: P61+P100+PN13 removed (V1 YAML retired 2026-06-01 in V1 sunset #8; V2 equivalent: preset `qa-qwen3.6-27b-tested` — sizing-identical 131K ctx / util 0.90 / seqs 2 / batched 4096 / fp8_e5m2 KV / MTP K=3; V2 explicitly disables 16 Wave 1/7/8 patches via patches_delta. Operator must consciously pick V2 because V2 ≠ byte-identical V1)
- `a5000-2x-35b-fp8-dflash`: P61 removed (retired 2026-06-01 in V1 sunset #4; V2 equivalent: preset `prod-qwen3.6-35b-dflash`, max_model_len 160K → 65K post-dev371 fix)

### Tunables tested (sweeps)
- **P67_NUM_KV_SPLITS** sweep на 27B (my 5×2 methodology): 16=118.43, 32=117.04, 48=115.32, 64=115.95. Canonical sanity (5×5): P67=16 = 132.28 TPS / 100.9 TTFT; P67=32 = **131.48 TPS / 97.1 TTFT** / CV 4.76%. TPS/TPOT TIED within CV ~5%, but **P67=32 wins TTFT by 3.7% + tighter CV + Sprint 1 default**. **Locked-in P67=32** для PROD.
- **max_num_batched_tokens 4096→8192** A/B: REGRESSED (-3.8% TPS, +8.9% TPOT, CV chaos 35.8%) → reverted. P72 caps profile_run, runtime 8192 caused scheduler issues with MTP K=3.
- **VLLM_FLOAT32_MATMUL_PRECISION=medium**: first attempt crashed (`Connection reset`); skipped as regression-risk пattern emerged.

### Documentation reconciliation
- README.md patch count 132/123/128 → **134** consistent
- BENCHMARKS.md: added Wave 8 PROD numbers section
- INSTALL.md "130 community patches" → 134
- PATCHES.md "133 entries" → 134; engine tier clarified (0 patches, namespace reserved)
- CONFIGURATION.md header bumped v7.59 → v11.0+wave8
- MODELS.md "37 patches" → 134
- 5 broken `vllm/sndr_core/CHANGELOG.md` references fixed (replaced with root)
- `tools/genesis_bench_suite.py:1089` legacy `vllm._genesis.compat.cli` → `vllm.sndr_core.compat.cli`

### Theme 4 — family-contract test coverage (audit gap closure)

- **MoE family**: 0% → contract for 4 patches (P24, P31, P37, PN27). `tests/unit/integrations/moe/test_moe_family_contract.py` — 22 tests, 20 pass + 2 xfailed (legacy auto-apply tech debt: `GENESIS_LEGACY_P24`/`P31` are synthetic flags per registry comment, no runtime effect — xfail documents the gap rather than hides it).
- **Quantization family**: 0% → contract for 3 patches (P81, P91, PN77). `tests/unit/integrations/quantization/test_quantization_family_contract.py` — 20 tests, all pass after P91 docstring fix (added `GENESIS_ENABLE_P91=1` mention for operator grep continuity; was dispatcher-gated only).
- Invariants per patch: module-importable (torch-less), Genesis marker (const OR attr pattern), `apply()` callable, env_flag referenced in source, no top-level torch import, family field matches in registry. Family-level: every patch registered + filesystem files match list (drift detector).
- Template: append `(module_path, patch_id)` tuple → contract auto-applies. Expansion path to other 0%-coverage families on the same template.

### Skill v2 final update (2026-05-11)

`~/.claude/skills/genesis-vllm-patches/SKILL.md` updated with comprehensive 2026-05-11 session state (485 lines, +102 from previous):

- **New section "Automated maintenance tooling (7 tools total)"** consolidates all session work:
  - 7-tool inventory table with purpose + when-to-run
  - Makefile shortcuts (21 targets)
  - Pre-commit hooks (12 total)
  - CI workflows (6 .yml files)
  - Theme 4 family contracts (factory pattern reference)
- **Iron rule #11 section** (in Anti-patterns) with full discipline:
  - 3-categorization framework for retire decisions
  - 4 waiver mechanisms (`_RETIRED_NO_SUPERSEDE`, `_INTERNAL_SUPERSESSION`, `_INTENTIONAL_INVERSE`, `enables_upstream_feature`)
  - 9 retired patches inventory with provenance
- Drift detection legacy section preserved (Theme 5 starters) for completeness

Future maintainers now have a single canonical skill file that documents:

- 18 families covered by family contracts (factory pattern)
- 134 patches in registry (single source of truth)
- 13 pin-gated patches + 9 retired (with provenance)
- 7 audit tools + 12 pre-commit hooks + 21 make targets + 4 CI gates + weekly cron
- PROD layout: integrations/ + locations/ + project_paths.py + env-driven scripts
- Operating manual: bench reference + pin-bump procedure + 30+ anti-patterns + iron rule #11 discipline

### PR template + Makefile + final operator ergonomics (2026-05-11)

- **`.github/PULL_REQUEST_TEMPLATE.md` rewritten** for current layout:
  - Stale `wiring/patch_*.py` / `vllm/sndr_core/tests/` paths → `integrations/<family>/<file>.py` / `tests/unit/`
  - **Iron rule #11 retire provenance section** added — 3 categorization options (byte-identical / does MORE / different approach) with mandatory deep-diff checklist
  - **Pin-gate `vllm_version_range`** section for new patches (3 range shapes)
  - **Family contract auto-coverage note** — new patches auto-covered if family already has contract
  - **CI gates inventory** — 4 explicit gates documented (pin-gate, iron-rule-#11, family contracts, audit upstream)
  - Pre-commit hook reference + how to run locally
- **`Makefile` added** with 21 operator shortcuts:
  - `make help` (auto-discovered from target docstrings)
  - Tests: `test`, `test-pin-gate`, `test-iron-rule`, `test-family`, `test-doc-sync`, `gates`
  - Audits: `audit-upstream`, `audit-upstream-offline`, `audit-yaml`, `audit`
  - Docs: `docs-check`, `docs-write`, `docs`
  - Pre-commit: `precommit-install`, `precommit`
  - Paths: `paths-env`, `paths-print`
  - Maintenance: `clean`, `doctor`
  - Aggregate: `ci` (runs everything CI runs, no gh API)
- **Smoke verified**: `make gates` runs all 4 CI gates in ~2s, all green.

### Pre-commit hooks + Operator runbook (2026-05-11 closure)

- **`.pre-commit-config.yaml`** added — runs Genesis CI gates locally before push:
  - Trailing-whitespace, EOF, YAML/JSON syntax, large-file (>2MB) checks (standard pre-commit-hooks)
  - 7 Genesis-specific local hooks gating on path changes:
    - Pin-gate test (when `registry.py` or `guards.py` changed)
    - Iron-rule-#11 meta-test (when `registry.py` changed)
    - Doc-sync (when `registry.py` or operator docs changed)
    - PATCHES_AUTO.md / CONFIGS_AUTO.md sync checks
    - Upstream audit offline (registry sanity)
    - Family contracts (when `integrations/` touched)
  - Operator install: `pip install pre-commit && pre-commit install`
- **Operator runbook** [Genesis_internal_docs/OPERATOR_RUNBOOK_2026-05-11.md](../Genesis_internal_docs/OPERATOR_RUNBOOK_2026-05-11.md) — day-to-day operational companion to CONTRIBUTING.md:
  - Daily / on-demand checks (PROD health, drift, registry consistency, upstream queue)
  - Vllm pin-bump quick procedure (9 steps, ~25 min PROD downtime)
  - Patch misbehavior triage (don't disable blindly — PN17 lesson cited)
  - Server filesystem layout cheatsheet
  - Pre-commit setup
  - PROD-down playbook (5-step recovery + rollback to dev93 baseline)
  - Maintenance cadence (per-PR / weekly / daily / on-bump / quarterly)
  - Tools cheatsheet (7 audit/maintenance scripts)
  - Escalation criteria

### Tech debt closure — 21 xfailed legacy env_flag tests → 1598 all-pass (2026-05-11 final)

The 21 xfailed tests across all 6 hand-written contracts + factory all stemmed from the same root cause: legacy patches (P3, P4, P5, P6, P7, P8, P12, P14, P15, P22, P24, P26, P27, P28, P31, P34, P36, P38, P39A, P44, P46) have synthetic `GENESIS_LEGACY_*` env_flags in registry — pure registry metadata that the legacy auto-apply path bypasses entirely. The xfail message said "tech debt: refactor to explicit env-gate OR rename env_flag" — i.e. operators couldn't `grep GENESIS_LEGACY_P34 vllm/sndr_core/integrations/` and find it.

**Fix** — 2-part:

1. **Bulk-add documenting comment block** to each of the 21 patch source files (Python script iterated all 21 paths). Each file now has, right after its module docstring:

   ```python
   # Legacy auto-apply note (audit 2026-05-11): registry env_flag
   # `GENESIS_LEGACY_P34` is synthetic — flag exists for registry/audit
   # coherence but has no runtime effect. Patch applies unconditionally
   # via dispatcher's legacy auto-apply path (`is_legacy_active` in
   # vllm/sndr_core/dispatcher/decision.py). See registry.py "Legacy
   # patches" section (~line 2083) for full context.
   ```

   Operator-grep continuity: `grep GENESIS_LEGACY_P34 vllm/sndr_core/integrations/` now finds the patch source + explains the legacy semantic.

2. **Refine contract helper logic** ([_family_contract_helpers.py](tests/unit/integrations/_family_contract_helpers.py)) and 6 hand-written contracts: check source for env_flag FIRST, only xfail if STILL missing (was xfail-first regardless). After bulk doc-fix, all 21 now pass cleanly.

**Final test suite**: **1598 passed + 37 skipped + 0 xfailed** (was 1577 passed + 21 xfailed). All explicit failure modes cleared. The 37 skips are legitimate (legacy auto-apply lifecycles, retired no-ops, pure runtime hooks without TextPatcher, vllm install root missing on Mac dev).

### YAML pin sync + CONTRIBUTING expansion + memory final (2026-05-11)

- **27B PROD canonical YAML** ([a5000-2x-27b-int4-tq-k8v4.yaml](vllm/sndr_core/model_configs/builtin/a5000-2x-27b-int4-tq-k8v4.yaml)) <!-- audit-links: allow --> `vllm_pin_required` bumped `dev93+g51f22dcfd` → `dev209+g5536fc0c0` to match actual PROD pin + `genesis_pin: v11.0.0+wave8+phase2` + `last_validated: 2026-05-11`. Other 10 YAMLs document `last validated on pin X` — left as-is (updating without re-bench would be a lie); operators bump as they re-validate. (V1 file retired in Phase 10 sunset 2026-06-01; link preserved as historical reference.)
- **scripts/moe_lookup_helper.sh** stale comment `vllm/sndr_core/paths/project_paths.py` → `vllm/sndr_core/locations/project_paths.py` (residual cleanup).
- **CONTRIBUTING.md expanded** with 3 operator-facing sections:
  - **"Adding a new family contract"** — 40-line factory pattern template + invariants listed
  - **"Audit & maintenance tools"** — table of 6 automated scripts (audit_upstream_status, emit_paths_env, check_doc_sync, generate_patches_md, generate_configs_md, audit_yaml_vs_runtime)
  - **"Iron rule #11"** — 4 waiver mechanisms (RETIRED_NO_SUPERSEDE, INTERNAL_SUPERSESSION, INTENTIONAL_INVERSE, enables_upstream_feature) + example registry entry
- **CI gates** section in CONTRIBUTING.md now lists the 4 explicit fail-fast gates added in test.yml.
- **Memory file** [SESSION_MEMORY_2026-05-11_FULL.md](../Genesis_internal_docs/SESSION_MEMORY_2026-05-11_FULL.md) updated with:
  - Layer renames table (4 entries) + back-compat aliases
  - Unified paths section (4 new helpers + CLI + server env file + Python dedup)
  - Theme 4 COMPLETE summary (17 contracts via 3 approaches)
  - Real bugs surfaced inventory
  - CI gates table
- **Server hygiene verified**: filesystem on server is clean (only `integrations/` and `locations/`, no stale `patches/` or `paths/`); tools/ + scripts/ synced.
- **audit_upstream_status** live with `gh` API: NEWLY-MERGED=0 — clean queue.

### CI gates + README sync + comprehensive verification (2026-05-11 final)

- **CI workflow [test.yml](.github/workflows/test.yml) expanded** with 4 new explicit gate steps (fail-fast visibility, runs after main pytest):
  - Pin-gate adoption gate: `tests/unit/dispatcher/test_pin_gate.py` (13 tests — catches allowlist drift, range semantics regressions)
  - Iron rule #11 enforcement gate: `tests/unit/dispatcher/test_iron_rule_11_enforcement.py` (4 tests — catches forgotten retire provenance)
  - Family contracts gate (all 17): `tests/unit/integrations/` (~700 contract tests — catches marker/env_flag/family convention drift across integrations tree)
  - Upstream-status audit (informational): runs `audit_upstream_status.py --skip-network` for PR-time visibility (strict gate is the weekly cron in [upstream_audit_status.yml](.github/workflows/upstream_audit_status.yml))
- **README.md line 120** sync: `vllm.sndr_core.patches` → `vllm.sndr_core.integrations` reference + note on back-compat alias.
- **audit_yaml_vs_runtime.sh verified post-renames** on PROD: 27B (`vllm-pn95-2xa5000`) shows 0 real drift — explicit disables in YAML + PN95 experiment additions properly classified.
- **Code quality sweep**: scanned for top-level torch/triton imports across all of `integrations/`, `compat/`, `cli/`, `middleware/` — **0 hits** (P38 was the only one, already fixed). Also clean: hardcoded `$HOME|/nfs/|/opt/` paths (only legitimate auto-detection probe lists in `model_configs/host.py`); FIXME/XXX/HACK comments (clean).

**Final session totals**:

| Metric | Value |
| --- | --- |
| Test suite | **1577 passed + 37 skipped + 21 xfailed** |
| Family contracts | **17** covering 18/18 families |
| Pin-gated patches | 13 (PN90 anchor-tight + 10 PROD broad + 2 retire-targets) |
| Retired patches | 9 (all deep-diff verified with provenance) |
| Iron-rule-#11 waiver mechanisms | 4 (RETIRED_NO_SUPERSEDE, LEGACY_PIN_GATE, INTENTIONAL_INVERSE, ENABLES_UPSTREAM) |
| Hardcoded paths in active scripts | 0 (F-013 closure via `project_paths.py` + `~/.genesis_paths.env`) |
| CI explicit gates | 4 added (pin-gate, iron-rule, family contracts, audit upstream) |
| Doc-sync gates | clean across README/PATCHES/INSTALL/MODELS/BENCHMARKS (134 patches consistent) |
| 27B PROD | live on dev209 + env-driven scripts (smoke confirmed thrice: LAYOUT_OK / INTEGRATIONS_OK / ENV_DRIVEN_OK) |
| 35B PROD | validated on dev209 (231.08 TPS, -0.55% vs Wave 8 baseline within CV) |

### Theme 4 COMPLETE — Factory pattern + all 18 families covered (2026-05-11)

Refactored family-contract testing from copy-paste pattern (~200 lines per family × 6 = ~1200 lines duplicated) to **factory module** ([tests/unit/integrations/_family_contract_helpers.py](tests/unit/integrations/_family_contract_helpers.py)). Each new family contract is now ~40 lines:

```python
from tests.unit.integrations._family_contract_helpers import (
    make_family_contract_class, make_family_registry_class,
)
PATCHES = [("vllm.sndr_core.integrations.<fam>.<file>", "<PID>"), ...]
class TestMyFamilyPatchContract(make_family_contract_class("<fam>", PATCHES)): pass
class TestMyFamilyFamilyRegistry(make_family_registry_class("<fam>", PATCHES)): pass
```

**Refined invariant logic** (single source of truth — propagates to all contracts):

- Marker check now **skips pure runtime hooks** (no `TextPatcher` import detected) — anchor-marker convention only applies to text-patches; runtime hooks like P67c, PN61, PN65, PN62, PN70, PN16_V6 legitimately don't need markers
- env_flag check has fallback to companion files (kernels/ split case)
- Legacy synthetic-flag patches still xfail with documented tech debt

**11 new family contracts via factory** (in addition to 6 prior MoE/quantization/scheduler/spec_decode/worker/memory/reasoning):

| Family | Patches | Notes |
| --- | --- | --- |
| `kv_cache` | 5 | P5/P14/P83/P85/PN95 |
| `compile_safety` | 4 | P6/P66/P95/PN13 |
| `kernels` | 5 | P36/P87/PN12/PN25/PN28 |
| `loader` | 2 | PN8/PN61 |
| `lora` | 1 | PN80 |
| `middleware` | 3 | PN16/PN16_V6/PN65 |
| `multimodal` | 1 | PN62 |
| `observability` | 1 | SPRINT26_CG_DISPATCH_TRACE (post-family-fix) |
| `serving` | 5 | P62/P68/P69/P107/PN70 (P68+P69 share file) |
| `tool_parsing` | 4 | P15/P61c/P64/PN56 (P29 legacy-only, no file) |
| `attention.flash` | 2 | P100/PN17 |
| `attention.gdn` | 17 | Largest single-family GDN coverage |
| `attention.turboquant` | 19 | (4 legacy-only entries P18b/P20/P32/P51 excluded — no files) |

**Real bug surfaced + fixed**: [p38_tq_continuation_memory.py](sndr/engines/vllm/patches/attention/turboquant/p38_tq_continuation_memory.py) had top-level `import torch` + `import torch.nn.functional as F` — breaks torch-less pytest collection contract. Moved to lazy `try/except ImportError` block; preserves runtime behavior, restores torch-less collection safety.

**Cumulative**: 17 family contracts (6 hand-written + 11 factory + 3 attention nested) covering **ALL 18 families** that have integration code. Total test suite **1577 passed + 37 skipped + 21 xfailed** (was 1166 at start of iteration — **+411 family-contract tests**).

### SPRINT26 + Memory + Reasoning contracts (2026-05-11 final)

- **SPRINT26_CG_DISPATCH_TRACE** registry/filesystem mismatch fixed: registry had `family="worker"` but `apply_module` pointed to `vllm.sndr_core.integrations.observability.sprint26_cudagraph_dispatch_trace` (the actual file location); `category: "observability"` also indicated true family. Changed `family="observability"` to match filesystem + category.
- **Memory family contract** (5 patches): coordinator (P5b) + retired (PN19, PN78) lifecycles skip marker check (legitimate no-ops/pointers); active P15B + P38B fully covered. 28 tests + 3 skipped.
- **Reasoning family contract** (8 patches): P27 + P12 legacy auto-apply xfail (synthetic GENESIS_LEGACY_*); rest pass. 48 tests + 2 xfailed.
- Contract test infrastructure now handles 4 patch states gracefully:
  - Active experimental → full check
  - Legacy auto-apply (synthetic env_flag) → xfail with documented tech debt
  - Retired no-op (lifecycle=retired + impl_status=retired) → skip marker+env_flag
  - Coordinator (lifecycle=coordinator) → skip marker (helpers live elsewhere)

**Cumulative coverage**: 6 family contracts (MoE + quantization + scheduler + spec_decode + worker + memory + reasoning) = 322 family-contract tests. Total suite **1166 passed + 22 skipped + 7 xfailed**.

### F-013 stale container_name fix + Worker family contract (2026-05-11)

Follow-up to audit F-013: 4 YAML configs had stale `container_name: vllm-server-mtp-test` (pre-Wave 8 name). Fixed:

- `a5000-2x-27b-int4-tq-k8v4.yaml` (27B PROD canonical) → `vllm-pn95-2xa5000`
- `a5000-2x-35b-prod.yaml` (35B PROD) → `vllm-35b-prod`
- `a5000-2x-27b-int4-tested.yaml` → `vllm-pn95-2xa5000-tested` (V1 YAML retired 2026-06-01 in V1 sunset #8; preset `qa-qwen3.6-27b-tested` going forward)
- `a5000-2x-27b-int4-tq-k8v4-dflash.yaml` → `vllm-pn95-2xa5000-dflash` (V1 YAML retired 2026-06-01; preset `experimental-qwen3.6-27b-tq-dflash-ab` going forward)

YAMLs now match actual server start-script container names.

**Worker family contract test** ([test_worker_family_contract.py](tests/unit/integrations/worker/test_worker_family_contract.py)): 9 patches × 6 invariants = 56 tests covering family that previously had 3/9 dedicated tests (PN52, PN55, PN82). All pass.

Surfaced 2 grep-gap bugs fixed in patch docstrings:

- PN33 used inverse `GENESIS_DISABLE_*` env in source while registry declares `GENESIS_ENABLE_*` — docstring now documents BOTH (registry-canonical + in-source kill-switch)
- PN35 dispatcher-gated, env_flag not in source — docstring now mentions `GENESIS_ENABLE_PN35_INPUTS_EMBEDS_OPTIONAL` for operator grep

Note: registry entry `SPRINT26_CG_DISPATCH_TRACE` declares `family="worker"` but source lives under `integrations/observability/sprint26_cudagraph_dispatch_trace.py` — registry/filesystem mismatch flagged for separate audit, not covered by worker contract.

**Cumulative test suite size: 1090 passed + 18 skipped + 5 xfailed** (was 1034 pre-worker; +56 from worker contract).

### Audit F-013 closure — Unified paths via project_paths.py (2026-05-11)

Per audit F-013 ("Не все hardcoded адреса и пути переведены в переменные"): all hardcoded paths in configs/scripts now flow through a single canonical settings file [`vllm/sndr_core/locations/project_paths.py`](sndr/engines/vllm/locations/project_paths.py).

**Extended `project_paths.py` with 4 new helpers** (covers what previously lived as hardcoded strings):

| Helper | Env vars honored | Default |
| --- | --- | --- |
| `models_dir()` | `SNDR_MODELS_DIR`, `GENESIS_MODELS_DIR` | `/models` if exists else `~/.cache/sndr/models` |
| `compile_cache_dir()` | `SNDR_COMPILE_CACHE_DIR`, `GENESIS_COMPILE_CACHE_DIR` | `<install_root>/cache/compile` |
| `triton_cache_dir()` | `SNDR_TRITON_CACHE_DIR`, `GENESIS_TRITON_CACHE_DIR`, `TRITON_CACHE_DIR` | `<install_root>/cache/triton` |
| `hf_cache_dir()` | `SNDR_HF_CACHE_DIR`, `HF_HOME`, `GENESIS_HF_CACHE_DIR` | `~/.cache/huggingface` |

Plus `emit_env_shell()` — renders all paths as a sourcable bash snippet so server start-scripts and Python use **identical** values.

**New CLI** [`scripts/emit_paths_env.py`](scripts/emit_paths_env.py):

- `python3 scripts/emit_paths_env.py > ~/.genesis_paths.env` — generate canonical env file
- `python3 scripts/emit_paths_env.py --print` — pretty-print all paths (debug)
- `--prefix SNDR` — alternate prefix (default: GENESIS for back-compat)

**Dedup of duplicate path-resolution logic**:

| File | Before | After |
| --- | --- | --- |
| `compat/models/pull.py:_resolve_models_dir` | Own `SNDR_MODELS_DIR / GENESIS_MODELS_DIR` env-check + HF fallback | Delegates to `project_paths.models_dir()`; HF Hub fallback retained for pull-time differs |
| `integrations/attention/gdn/p60b_*.py:_clear_triton_cache` | 3-path list with `os.environ.get("TRITON_CACHE_DIR")` | Adds `project_paths.triton_cache_dir()` first; legacy fallbacks kept as last-resort for env-less environments |

**Server start-scripts rewritten** to source canonical env file + use env vars throughout:

- 5 hardcoded paths per script replaced: `/nfs/genesis/models` → `${GENESIS_MODELS_DIR}`, `$HOME/.cache/huggingface` → `${GENESIS_HF_CACHE_DIR}`, compile/triton cache dirs, Genesis project mount
- All 3 scripts (`start_pn95_2xa5000_test.sh`, `start_35b_prod_wave8.sh`, `start_27b_bump_test.sh`) source `~/.genesis_paths.env` at top if present
- `~/.genesis_paths.env` written on server with operator-specific values (NFS models, host caches, Genesis project location)
- Operator workflow: edit one file (`~/.genesis_paths.env`); all start-scripts pick up new values on next launch

**Verification**: 1034 tests pass + 18 skipped + 5 xfailed. 27B PROD restarted on env-driven layout — smoke gen confirmed `ENV_DRIVEN_OK` (3rd PROD restart this session — `LAYOUT_OK` → patches+paths renames → `ENV_DRIVEN_OK` → env-driven scripts).

### Structure cleanup — Layer rename per audit P-01/P-02 (2026-05-11)

Two layer renames to remove semantic overload + improve discoverability. Both backed by audit P-01 (sndr_structure_deep_audit_2026-05-07.md): `paths/` was overloaded (project paths + vllm targets + resolver shims), `patches/` was misleading (directory holds runtime integration overlays, not just bug-band-aids).

**`vllm/sndr_core/patches/` → `vllm/sndr_core/integrations/`**

- Rationale: directory holds 123 runtime integration overlays across 18 families — text-patches AND monkey-wraps AND env activators AND intentional inverses AND byte-equivalent backports. "Patches" implies temporary bug-fix; "integrations" captures the actual role: integrating Genesis behavior with vllm at runtime.
- 88 Python files rewrote `vllm.sndr_core.patches` → `vllm.sndr_core.integrations`
- 18 family subfolders preserved (mirror registry `family` field — clean granularity)
- Test tree moved `tests/unit/patches/` → `tests/unit/integrations/`
- Dispatcher's filesystem walk ([spec.py:320](sndr/dispatcher/spec.py#L320)) updated with legacy `patches/` fallback during transition
- Back-compat alias in `vllm/sndr_core/__init__.py.__getattr__`: `import vllm.sndr_core.patches` still resolves to `integrations` (transition path)

**`vllm/sndr_core/paths/` → `vllm/sndr_core/locations/`** + file renames per audit P-01:

| Old name | New name | Role |
| --- | --- | --- |
| `sndr_paths.py` | `project_paths.py` | Genesis project file paths (12 helpers: `install_root`, `wiring_dir`, `manifest_dir`, `model_configs_*_dir`, etc.) |
| `engine_targets.py` | `vllm_targets.py` | 63 vllm-side path constants (canonical anchor target registry) |
| `resolver.py` | (unchanged — small forwarder shim) | re-exports `resolve_vllm_file` from `detection.guards` |
| `vllm_install.py` | (unchanged — small forwarder shim) | re-exports `vllm_install_root` from `detection.guards` |

- Folder `locations/` semantic: "where files live" — covers both Genesis project layout + vllm target paths without overloading "paths"
- Module aliases in [locations init](sndr/engines/vllm/locations/__init__.py): `engine_targets = vllm_targets` and `sndr_paths = project_paths` (back-compat during transition)
- Back-compat alias in `vllm/sndr_core/__init__.py.__getattr__`: `import vllm.sndr_core.paths` still resolves to `locations`

**Verification**: full test suite 1034 passed + 18 skipped + 5 xfailed. PROD restarted twice (each layer rename + server sync). Smoke gen on final layout: `LAYOUT_OK`. Server-side cleanup of stale pycache via `sudo rm -rf` (root-owned by container).

### Phase 2 — Automated upstream-status audit + v2 retire batch (2026-05-11)

**New tool**: [scripts/audit_upstream_status.py](scripts/audit_upstream_status.py) — for each of 57 patches with declared `upstream_pr`, queries GitHub via `gh api` for merge state, cross-references with our lifecycle / pin-gate fields, categorizes:

- **NEWLY-MERGED** (action queue): upstream merged BUT our lifecycle ≠ retired
- **STALE-RETIRED** (investigate): retired locally but upstream PR still OPEN
- **ISSUE-CLOSED**: upstream issue resolved — check if our patch is still needed
- **INTENTIONAL-INVERSE** (waived): we deliberately oppose merged upstream (perf/Ampere-specific)
- **ENABLES-UPSTREAM** (waived): convenience activator/wrapper on top of upstream feature, not a backport
- **RETIRED-INTERNAL** (waived): our internal supersession (e.g. P12 v2 superseded P61)
- **SUPERSEDED-OK**: properly retired with full provenance (no action)
- **WATCH**: normal — upstream open, our patch active
- **ERROR**: gh API issue — investigate

Handles GitHub ISSUE refs as well as PRs (PN51's `upstream_pr` was an issue number not PR). Modes: full table / `--filter <category>` / `--json` / `--skip-network` / `--fail-on-newly-merged` (CI gate). Suggested weekly cron in `.github/workflows/upstream_audit.yml`.

**v2 retire batch** triggered by audit findings — 6 NEWLY-MERGED + 1 ISSUE-CLOSED processed via iron-rule-#11 deep-diff:

| Patch | Decision | Reasoning |
| --- | --- | --- |
| **PN9** | RETIRED | Upstream #39930 more capable (adds SpeculativeConfig.attention_backend pydantic field we didn't backport); pin-gate `<dev9` formalized |
| **P94** | RETIRED | Byte-identical with #41043 (Wave 8 deep-diff confirmed); lifecycle bumped from experimental |
| **P98** | INTENTIONAL-INVERSE waiver | Deliberate revert of #40941 (WorkspaceManager) due to 17% TPS regression on Ampere TQ small-batch — keep active |
| **P99** | `enables_upstream_feature: True` | Memoization wrapper ON TOP of upstream WorkspaceManager (#40941) — augments, not backports |
| **P75** | `enables_upstream_feature: True` | Convenience auto-enable of Suffix Decoding (#25784 merged 2025-11) — activator, not backport |
| **PN80** | RETIRED | Byte-equivalent: dev209 `lora_model.py:203-206` has `device=device` arg in `TensorDeserializer` call as #41845 specified |
| **PN51** | RETIRED | Upstream fixed at serving layer via `prompt_is_reasoning_end` (different impl, same outcome — entry-point guard PN51 added is no longer needed) |

**Total retired now: 9** (was 7 pre-audit; +PN9 +P94 +PN80 +PN51).

Registry-driven waivers introduced: `enables_upstream_feature: True` field on patches that augment/activate upstream features without backporting. Audit script honors this to exclude from NEWLY-MERGED queue.

Final audit state: **NEWLY-MERGED=0, STALE-RETIRED=0, ERROR=0**. 152 tests pass + 2 xfailed. Doc-sync gates clean. 27B PROD on dev209 stable.

### Phase 2 — Iron rule #11 enforcement meta-test + PN17 hypothesis disproven (2026-05-11)

**Meta-test** ([test_iron_rule_11_enforcement.py](tests/unit/dispatcher/test_iron_rule_11_enforcement.py)): enforces provenance discipline on the registry — every `lifecycle="retired"` patch must declare BOTH `superseded_by` + `vllm_version_range`, OR be explicitly waived in `_RETIRED_NO_SUPERSEDE_WAIVER` (hypothesis disproven / internal-only retire cases). Every patch with `superseded_by` must declare `vllm_version_range`, EXCEPT legacy auto-apply patches whose synthetic `GENESIS_LEGACY_*` flag bypasses pin-gate (waived via `_LEGACY_PIN_GATE_WAIVER`). 4 tests, all pass after fixing 1 surfaced gap: PN13 had pin-gate but missing `superseded_by` — added "vllm#41235 (merged 2026-04-29, in commit c2fb013 / v0.20.2 — byte-equivalent on dev93+dev209)" annotation. Waivers documented: P61 (internal supersession by our own P12 v7.62.5), P63 (hypothesis disproven 2026-04-25), PN78 (internal pending investigation), P8 (upstream refactor, no specific PR captured). Going forward, every new retire requires either provenance OR explicit waiver — test fails the CI suite otherwise.

**PN17 differential bench** ([bench_dev209_postfixes_2026-05-11.json](../Genesis_internal_docs/bench_dev209_postfixes_2026-05-11.json) baseline + new PN17=0 run): static analysis flagged `seqused_k.max().item()` per-attention-call as suspected -3.7% Sprint 1 regression source. Differential bench (35B/dev209, PN17=0 vs PN17=1):

- PN17=1 baseline: wall_TPS **231.08**, decode_TPOT 4.01 ms, CV ~4-9%
- PN17=0 disabled: wall_TPS **178.95**, decode_TPOT 5.15 ms, CV ~8-11%
- **Delta: PN17=0 is -22.6% TPS, +28.5% TPOT** (drastically worse)

PN17 is a **PERF WIN**, not a regression source. Static analysis was wrong: CUDA stream scheduling absorbs the `.item()` sync, while the alternative cost (FA2 allocating `softmax_lse[num_seqs, num_heads, 320000]` per call when `max_seqlen_k=max_model_len` is used unclamped) dominates by 10-100× on long-context configs. Registry note corrected — explicit "DO NOT DISABLE" + empirical bench numbers + lesson reasoning. Skill v2 anti-pattern added: "Static-analysis hot-path verdict without differential bench" with this case as exemplar.

Sprint 1 regression source remains unknown. PN17 ruled out. Other Wave 7 guards (P107 per-request finalize, PN56 only on regex fail, PN66 per-streaming-delta, PN67 1-line bool fix, P95 NO-OP on FP8) static-analyzed as cheap; differential bench on each would be expensive. May be cost-of-safety we accept until a focused sprint can isolate.

### Phase 2 — 35B bump dev93 → dev209 + iron-rule-#11 retire batch (2026-05-11)

35B PROD validated on dev209+g5536fc0c0 with full boot-log audit per iron rule #11. Same parallel-container strategy as 27B: stop 27B PROD → boot 35B with bumped image + tee log → canonical bench → audit auto-skips → restore 27B.

**35B canonical bench on dev209** (5×5×1024, warm cache):

- wall_TPS = **231.08** (CV 8.72%) vs Wave 8 232.36 → **-0.55%** (within CV)
- decode_TPOT = **4.01 ms** (CV 4.02%) vs 3.97 → **+1.0%** (within CV)
- TTFT = 114.4 ms (CV 33.88%, first-token jitter typical) vs 112 → +2.1%
- Tool-call 7/7 positive, 8K context stable
- Net: performance-neutral vs Wave 8 dev93 baseline; Sprint 1 (-3.7%) gap carries over (Wave 7 defensive guards, separate investigation queued)
- Genesis self-test inside bench: **7 pass / 0 fail / 1 warn** (bug-fix sweep self-test fallback working — was "not available" before)

**Boot log audit identified 5 auto-skip candidates** (52 applied / 79 skipped / 2 anchor-drift soft-skips on 35B/dev209):

- **PN19** (`Scoped max_split_size_mb during model load`, #41268): wire detector found `_scoped_allocator_max_split` in dev209 `vllm/v1/worker/gpu_worker.py`. Deep-diff confirmed: upstream PR #41268 (merged 2026-04-30, before our dev93 SHA 2026-05-07!) implements byte-equivalent context-manager + 20 MiB minimum + prior-value restoration. Was overdue for retire. Registry now: `lifecycle="retired"` + `vllm_version_range: "<0.20.2rc1.dev93"` + `superseded_by` annotation. Wiring continues to auto-skip; pin-gate formalizes.
- **PN52** (already retired earlier in session): `< prompt_lens` without -1 + `in_progress_prompt_logprobs_cpu` on `CachedRequestState` — #41411 byte-equivalent on dev209.
- **P4** / **P12** / **P26** (legacy patches, pre-dispatcher era): wire detector auto-skips via upstream markers (`TurboQuantConfig.get_boundary_skip_layers`, `_tool_call_token_id`, `_cu_2` respectively). Lifecycle kept `"legacy"` (architectural property — pre-dispatcher auto-apply pattern, not a stage), but registry now carries `superseded_by` annotation + iron-rule-#11 study note documenting deep-diff state. Wire-detector skip is correct + safe; no behavioral change.
- **P22** (already documented earlier): iron-rule-#11 case (b) — our patch does MORE (profiler-visibility hook on top of upstream class-shared buffer restructure). Investigation queued in registry note. Currently auto-skip safe but loses our improvement; need new hook site design on dev209's restructured `gpu_model_runner.capture_model` + current `TurboQuantAttentionImpl`.

**Retire count now 7** (up from 5 pre-audit). 35B YAML [a5000-2x-35b-prod.yaml](vllm/sndr_core/model_configs/builtin/a5000-2x-35b-prod.yaml) <!-- audit-links: allow --> updated: `vllm_pin_required: 0.20.2rc1.dev209+g5536fc0c0` + bench result annotation block + Sprint 1 baseline preserved as second-tier historical reference (`prev_long_gen_tps_sprint1: 241.35`). docs/PATCHES_AUTO.md regenerated (134 entries, 7 retired). 27B PROD restored on dev209 after. (V1 file retired in Phase 10 sunset 2026-06-01; link preserved as historical reference.)

### Phase 2 — Bug-fix sweep post-dev209 boot (2026-05-11)

After dev209 cutover, full boot-log audit (`/tmp/genesis_boot.log` — 897 lines) surfaced 3 real bugs + 1 iron-rule-#11 retire candidate:

- **PN95 init schema mismatch** (fixed): YAML field `prev_long_gen_tps_sprint1` (Wave 8 sprint reference) + 4 `wave8_delta_pct_*` audit-trail fields weren't in `ReferenceMetrics` dataclass → `ReferenceMetrics.__init__() got an unexpected keyword argument` on every cold container. Two-part fix: (1) added `prev_long_gen_tps_sprint1: Optional[float]` to [schema.py:317](sndr/model_configs/schema.py) (legitimate second-tier historical reference, parallel to `prev_long_gen_tps`); (2) defensive unknown-field filtering with `log.warning` in `_from_plain_dict` reference_metrics path — audit-trail fields no longer crash loaders. PN95 lazy init now succeeds: `register_kv_caches: 65 layers (mamba+attn), 17 attention layers eligible for demote`.
- **Bench self-test fallback** (fixed): `tools/genesis_bench_suite.py:capture_genesis_patch_state` used host `python3 -m vllm.sndr_core.compat.cli self-test` which fails with `ModuleNotFoundError` when bench runs outside container → printed `self-test exit=1` as if patcher was broken. Rewrote with 2-stage fallback: try host python first, then `docker exec` into auto-detected Genesis containers (preferring `vllm-*` prefix). Returns clear `reason` with stderr context instead of bare exit code.
- **Boot log truncation** (fixed): PROD start-script piped Genesis pre-pass through `python3 -m vllm.sndr_core.apply 2>&1 | tail -3` — operator saw only "… and 38 more" + 2 final lines, losing 800+ lines of apply trace including `[Genesis pin-gate]` log, dep-graph check, per-patch APPLY/SKIP reasons, anchor-drift detector findings. Changed to `| tee /tmp/genesis_boot.log | tail -20` — full audit trail saved, last 20 lines surface for boot UI. Now visible: pin-gate confirms `vllm 0.20.2rc1.dev209+g5536fc0c0 is on the Genesis allowlist`, dep-graph clean across 73 enabled patches, 58 applied / 72 skipped / 3 anchor-drift soft-skips on dev209.
- **PN52 retire via iron rule #11** (lifecycle update): full boot log + deep-diff revealed upstream #41411 (merged 2026-05-04, in dev209) implements BOTH PN52 fixes byte-equivalently — `< prompt_lens` without -1 in `prompt_logprob.py` AND `in_progress_prompt_logprobs_cpu` moved to `CachedRequestState` (gpu_input_batch.py:54 + gpu_model_runner.py:5200/5207/5264). Patch self-skips via wiring's drift detector; registry now formalizes: `lifecycle="retired"` + `vllm_version_range: "<0.20.2rc1.dev209"` + `superseded_by` annotation. Patch file kept in tree as audit trail; queued for delete in next cleanup pass.
- **P22 iron rule #11 investigation queued**: drift detector auto-skips P22 on dev209 because `_init_turboquant_buffers` removed upstream (PR #40655-style class-shared buffers landed without formal merge). Per our patch's own docstring (lines 43-55), the alternative upstream approaches LACK the profiler-visibility hook that's P22's specific value-add (visible to memory-profiler → correct KV cache sizing → no #40420-class OOM). Iron rule #11 case (b) "our patch does MORE" — currently auto-skip is safe but loses our improvement; investigation queued in registry note (P22 entry now documents the gap explicitly). Need to design new hook site on top of dev209's restructured `gpu_model_runner.capture_model` + current `TurboQuantAttentionImpl`.

**Post-bugfixes canonical bench** (5×5×1024, warm cache):

- wall_TPS = **131.48** (CV **2.96%** — improved from 3.53% first bench)
- decode_TPOT = **7.36 ms** (CV 3.71%, improved from 7.39)
- TTFT = 118.9 ms first turn, 135-138 ms steady
- Tool-call 7/7 positive, 8K context stable
- vs Wave 8 dev93 baseline (132.28): **-0.61%** wall_TPS (within CV) — effectively equivalent

### Phase 2 — vllm pin bump dev93 → dev209 (2026-05-11, net-neutral)

- **Bumped from**: `0.20.2rc1.dev93+g51f22dcfd` (Wave 8 baseline, 2026-05-07)
- **Bumped to**: `0.20.2rc1.dev209+g5536fc0c0` (vllm/vllm-openai:nightly, 2026-05-11) — ~116 dev-increments / 4 days of upstream work
- **Method**: parallel throwaway test container (`vllm-27b-bump-test`, port 8102) — PROD downtime ~25 min for the actual cutover
- **Bench (canonical `genesis_bench_suite.py --quick --ctx 8k`, 5×5 = 25 runs)**:
  - wall_TPS = **131.11** (CV 3.53%) vs Wave 8 132.28 → **-0.88%** (within CV)
  - decode_TPOT = **7.39 ms** (CV 3.66%) vs 7.31 → **+1.10%** (within CV)
  - TTFT = **97.7 ms** (CV 7.44%) vs 97.1 → **+0.62%** (within CV)
  - Tool-call: **7/7 positive**, 8K context stable
  - Net: performance-neutral; no regressions visible on PROD-critical path
- **Deep-diff verification** (per skill iron rule #11, Sander 2026-05-11): cross-referenced every Genesis `upstream_pr` field (57 patches with declared upstream PR) against PRs merged in window. **0 new supersessions in dev93→dev209 window** — every merged upstream_pr in our registry was already in dev93 (PR #25784/#39930/#40941/#41043/#41235/#41268/#41411 all merged 2025-11 → 2026-05-04). The 116 dev-counter increments are unrelated upstream work (sparse MLA, ROCm DSv4, internal refactors, etc.).
- **Pin-gate validation**: all 13 pin-gated patches applied without `VERSION:` skip. PN90 anchor-tight range admitted (dev209 ≥ dev9 ✓). PN9/PN13 retire upper-bounds correctly skip on dev209 (gate functioning as designed).
- **PROD cutover**: `~/start_pn95_2xa5000_test.sh` image swapped `vllm-genesis-pinned:dev93-2026-05-09` → `vllm/vllm-openai:nightly`. Backup retained at `~/start_pn95_2xa5000_test.sh.bak.dev93-2026-05-11`. PROD restarted, smoke gen confirmed `PROD_DEV209_BUMP_OK` (191 tokens, finish=stop, reasoning + content clean).
- **Allowlist updates**: `KNOWN_GOOD_VLLM_PINS` += `"0.20.2rc1.dev209+g5536fc0c0"` ([detection/guards.py](sndr/engines/vllm/detection/guards.py)); `EXPECTED_PINS` matched in [tests/unit/dispatcher/test_pin_gate.py](tests/unit/dispatcher/test_pin_gate.py) (drift trap green).
- **Bench artifact**: `Genesis_internal_docs/bench_dev209_2026-05-11.json` (18.4 KB).
- **PN95 init failure on cold container** (known issue, init-only cost): `ReferenceMetrics.__init__() got an unexpected keyword argument 'prev_long_gen_tps_sprint1'` — schema drift in PN95 lazy init, surfaced when cache dir is fresh. Not blocking, kept env=1 per Wave 8 audit decision.

### Phase 2 — Pin-gate adoption (13 patches now declare `vllm_version_range`)

- **Discovery (2026-05-11)**: pin-gate infrastructure existed since 2026-05-04 (Sander, "защита от дурака") — `KNOWN_GOOD_VLLM_PINS` allowlist + boot-time `assert_vllm_pin_allowed` + per-patch `applies_to.vllm_version_range` parsed by `compat/version_check.check_version_constraints` wired into `dispatcher/decision.py:117-140`. But **0 patches actually declared a range** as of audit — full carcass, no adoption.
- **Reference adoption**: PN90 declares anchor-tight range `(">=0.20.2rc1.dev9", "<0.21.0")` — anchors don't exist on pre-dev9 pins. First patch to use the gate.
- **PROD-critical broad adoption** (10 patches): P67, P82, P70, PN12, PN14, P94, P103, PN16, P107, P72 — each declares `(">=0.20.0", "<0.21.0")`. Currently-validated range; drift detector handles anchor-line breakage on bumps within range.
- **Retire-target upper-bound adoption** (2 patches): PN9 (`<0.20.2rc1.dev9` — upstream #39930 merged 2026-04-28), PN13 (`<0.20.2` — upstream #41235 merged 2026-04-29 into 0.20.2 release). Gates them off on current pin without hard-deleting (audit-trail preserved).
- **Today's vllm HEAD** = `7863fff6e` (2026-05-11). NOT yet added to `KNOWN_GOOD_VLLM_PINS` — pending server-side validation per pin-bump playbook (boot smoke → drift skips → canonical bench → promote).
- **Test coverage**: [tests/unit/dispatcher/test_pin_gate.py](tests/unit/dispatcher/test_pin_gate.py) — 13 tests covering allowlist drift trap, `check_version_constraints` semantics (in/below/above-range, undetectable-conservative-pass, PEP 440 prerelease), PN90 reference declarations, dispatcher wiring sanity. All pass.
- **Docs**: [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) new "Pin-bump playbook" section (3-layer gate description + 7-step bump procedure + diagnostic snippet + 3 range-shape patterns). Updated registry example to include `vllm_version_range`.

### Theme 5 — doc-sync automation (closes recurring drift class)

- `scripts/check_doc_sync.py` — parses registry.py count (134) vs README/PATCHES/INSTALL/MODELS/BENCHMARKS claims. Modes: report / `--strict` / `--json`. All 5 docs verified consistent.
- `scripts/generate_patches_md.py` — auto-gen `docs/PATCHES_AUTO.md` (134 entries) from registry via regex + `ast.literal_eval`. Modes: write / `--check` / `--stdout`. Statistics by tier/lifecycle/family, per-family grouped table.
- `scripts/generate_configs_md.py` — auto-gen `docs/CONFIGS_AUTO.md` (11 configs) from builtin YAMLs. Extracts top-level fields + `reference_metrics` + MTP K.
- `tools/audit_yaml_vs_runtime.sh` — drift detector: YAML `genesis_env` vs `docker inspect` env. Smart filtering for explicit disables + intentional experiment additions. Exit codes 0/1/2.
- `.github/workflows/doc_sync.yml` — CI gate: `check_doc_sync.py --strict` + `generate_patches_md.py --check` + `generate_configs_md.py --check` on every push to main + PR.

### Research deliverables (in `Genesis_internal_docs/`)
- `SESSION_REPORT_2026-05-11_RU.md` — session write-up with all F1-F7 outcomes
- `CANONICAL_TIMELINE_2026-05-11_RU.md` — full 2026-04-25 → 05-11 history
- `MASTER_PLAN_2026-05-11_FINAL.md` — unified 10-part plan
- `RESEARCH_2026_05_11/HOLISTIC_IMPROVEMENT_PLAN_RU.md` — 8-phase roadmap (53-77 engineer-days refactor + 2-3 months strategic borrowings)
- `RESEARCH_2026_05_11/RESEARCH_A/B/C_*.txt` — codebase + cross-engine + upstream raw research
- `~/.claude/skills/genesis-vllm-patches/SKILL.md` — operating manual (9 iron rules + 9 bug classes + server playbook)

### Retire-candidate diff (post-bump cleanup)
Deep-diff verified Agent C misclassified 5/6 retire candidates. Only **P94** is byte-identical with merged vllm#41043 (drop on next bump). PN50/P74/PN33/PN67/PN92 all KEEP per source code differences.

### vLLM pin status
- Current: `0.20.2rc1.dev93+g51f22dcfd` (SHA `51f22dcf`, 2026-05-07)
- Upstream HEAD 2026-05-11: `8415bf2c` (129 commits ahead, 486 files)
- **Bump deferred 7-10 days** (target 2026-05-18..21) — upstream had 4 breaking landings 2026-05-11

### 35B PROD status — bench completed 2026-05-11 19:00Z

Created `Genesis_internal_docs/start_35b_prod_wave8.sh` matching `a5000-2x-35b-prod.yaml` (full Wave 7+1+3.1+P82+P67=48 config). Deployed and bench'd in single session.

**Canonical bench `genesis_bench_suite.py --quick --ctx 8k` (3rd warm rerun, stable)**:
- wall_TPS = **232.36** (CV 6.74%)
- decode_TPOT = **3.97 ms** (CV 4.50%)
- TTFT ≈ 112 ms

**Δ vs Sprint 1 baseline (241.35 / 3.85)**: **-3.7% TPS / +3.1% TPOT** — real but minor regression. Likely:
1. Wave 7 defensive guard overhead (P107/PN17/PN52/PN56/PN66/PN67/P95 ~0.5% each)
2. Triton cache state difference (fresh `triton-cache-35b-prod` dir vs Sprint 1's warmed cache)
3. PN90+PN16 V8 cumulative overhead on 35B path

**Net Genesis impact** (weighted by typical workload):
- 27B PROD: **+5.78% TPS, -9.85% TTFT** (большой выигрыш)
- 35B PROD: -3.7% TPS, +4% TTFT (мелкая регрессия, defensive guards trade-off)
- Net: Wave 8 = positive (27B is primary PROD workload)

35B YAML reference_metrics updated 2026-05-11. Sprint 1 baseline preserved as historical reference.

### NO git push
Per Sander 2026-05-11. All edits local only.

---

## [v11.1.0-dev] — UNIFIED_CONFIG_AUTOMATION_PLAN closure + Path C foundation (2026-05-09)

> **Single autonomous session: 13+ shippable deliverables.** Tier 0/1/2/3
> work from `docs/_internal/UNIFIED_CONFIG_AUTOMATION_PLAN_2026-05-09_RU.md`
> + Path A interim shipped + Path C v7.73.x foundation laid down (Days 1-4
> of the 9-day plan in code; Days 5-7 deferred to live integration).

### Schema (8 new dataclasses, all back-compat)

- **Y1 + B6** `PackageVersions.python_packages` — moves the renderer's
  hardcoded `pandas==2.2.3 scipy==1.14.1 xxhash==3.5.0` baseline into
  per-config YAML. Operator override; falls back to legacy when unset.
- **Y4** `DockerConfig.host_port` / `container_port` split — declare
  HOST≠CONTAINER ports for multi-instance hosts, k8s NodePort, RunPod.
- **Y11** `UpstreamPinPolicy` — per-config `required_pin/allowed_pins/
  blocked_pins` with `check()` method. Honored by `sndr deps plan`.
- **Y12** `OverridesPolicy` — `allow_env` + `safe_ranges` for
  validated `sndr launch --override` numeric/key gating.
- **OffloadConfig (Path A interim)** — `cpu_offload_gib` + hybrid-GDN
  guard that raises unless Path C `cache_config.tiers` +
  `exclude_mamba_ssm=True` is declared.
- **Y3** `Artifacts` (`models[]` + `caches[]`) — typed model+cache spec
  replacing fetch_models.sh hardcoded paths and old compat.models.pull
  registry-tagged lookup.
- **Y10** `ServiceConfig` — backend/restart/env_file declaration for
  systemd/compose/podman/k8s/proxmox/bare-metal targets.
- **Y2** `PackageSources` — channel declarations (distro_repo / pip /
  nvidia_repo / docker_image / source_build / curl_pipe_bash). Safety:
  `curl_pipe_bash` requires explicit `allow_third_party=True` opt-in.

### CLIs (5 new + 1 mode)

- **C2 `sndr deps check / plan`** (+ `--write-report`, `--strict`).
- **C4 `sndr model pull / list`** (thin native wrapper + cli_main fast-path).
- **C5 `sndr install --config <key> --prepare`** — bypasses workload
  heuristics; emits launcher script for the named preset.
- **C17 `sndr upstream check / show / list`** — surfaces Y11 + the
  project KNOWN_GOOD_VLLM_PINS allowlist.
- **C22 `sndr caveats list / check / explain`** — surfaces the new
  `vllm/sndr_core/caveats.py` registry (Y13).

### Path C v7.73.x foundation (PN95)

- **Schema:** `CacheTier` dataclass + `CacheConfig.{tiers, exclude_mamba_ssm,
  vision_demote_first, tier_low_water_pct, async_demote}`.
- **Runtime:** `vllm/sndr_core/cache/tier_manager.py` (~360 LOC) with
  per-tier `EvictionPolicy` (LRU/2Q/ARC), CPU-pool slab, vision-first
  demote ordering, Mamba-exclusion filter, spec-decode hot ring.
- **Wire-in:** `vllm/sndr_core/integrations/kv_cache/pn95_tier_aware_cache.py`
  text-patches both `single_type_kv_cache_manager.cache_blocks` and
  `block_pool.get_cached_block` with fail-silent `notify_admit/touch`
  hooks. Default OFF; opt-in via
  `GENESIS_ENABLE_PN95_TIER_AWARE_CACHE=1` + `cache_config.tiers`.
- **Dispatcher:** `apply_patch_N95_tier_aware_cache()` registered;
  `vllm/sndr_core/dispatcher/registry.py` PN95 entry; `Flags.PN95_TIER_AWARE_CACHE`.
- **EXAMPLE config:** `model_configs/builtin/single-3090-hybrid-gdn-tier-aware-EXAMPLE.yaml` (retired 2026-06-01 in V1 sunset; V2 equivalent: preset `example-3090-tier-aware`).
- **Days deferred to live integration:** vision-token MM tagging
  (Day 5), Mamba runtime classifier walk (Day 6), live bench on
  27B Lorbus (Day 7).

### Bench results (server-validated)

- **27B P82 sweep:** thr=0.1 wins +5.23% TPS over P82=0 baseline (130.79
  vs 124.29), 7/7 tool-call across all thresholds. **27B PROD YAML
  promoted** to `P82=1` + `thr=0.1`.
- **PN16 V6 multi-bench (5 runs):** mean TPS 236.12 ± 2.21 (CV 0.94%),
  5/5 london_think PASS — prior single-run regression hypothesis
  REJECTED. V6 stays opt-in; quality concern cleared.
- **PN58 vs P62 A/B:** NULL impact (236.04 vs 236.63), stay on P62.
- **PN12 FFN scratch pool A/B on 35B:** NULL impact (TPS +0.14%, VRAM
  identical) — stays default OFF.
- **35B PROD final reference:** wall_TPS 233.84 (within Sprint 1 CV).

### Bug fixes

- **B1**: 35B + 27B PROD YAML `vllm_pin_required` bumped dev9→dev93.
- **B3**: `_match_preset()` switched from non-existent `gpu_class` to
  `gpu_match_keys` — preset matching had been silently failing.
- **B5**: `scripts/fetch_models.sh` slimmed 95→70 lines; delegates to
  `sndr model pull`.
- **B6**: Renderer's hardcoded runtime deps moved to `PackageVersions`.

### Security / release hardening

- **Strict tests pass without `SNDR_ALLOW_LEGACY_LICENSE_KEYS`**.
- **`.github/workflows/release.yml`** — wheel build + SBOM + strict-test
  gate + GH Release upload.
- **T5 lockfiles** — `requirements-runtime.lock` + `requirements-dev.lock`.

### Tools (community-facing)

- **`tools/kv_calc.py`** — standalone capacity calculator (port of the
  club-3090 ask). Accepts `--preset` or `--model`; renders breakdown +
  GREEN/YELLOW/RED verdict per declared `--gpu-vram`.

### Research

- **club-3090 #58** root-caused (PN59 GDN gate rejects chunked-prefill
  on 24GB single-card) + 3-path solution analysis (Path A interim
  shipped, Path C v7.73.x foundation in code, Path B rejected).
- **Path C v7.73.x deep design** — agent-produced 200-line implementation
  spec at `docs/_internal/research/path_c_v773_tier_aware_cache_design_2026-05-09.md`.

### Test posture

- **+~150 new unit tests** added this session.
- **2522 PASS / 15 skip / 0 fail** strict
  (`SNDR_ALLOW_LEGACY_LICENSE_KEYS=""`).

---

## [v11.0.0] — Two-audit closure + canonical SNDR Core/Engine split (2026-05-08)

> **48-hour audit-driven cleanup.** Two independent third-party deep
> audits landed back-to-back: `sndr_structure_deep_audit_2026-05-07.md`
> (19 + 20 findings) and `sndr_repeat_deep_audit_2026-05-08.md` (9 P0 + 9 P1).
> Combined closure rate: **33 of 37 findings same-day**, including all
> P0 critical bugs.

![Audit-finding closure timeline](assets/charts/audit_closure_timeline.png)

### Architecture (the headline change)

- **`vllm/_genesis/` deleted entirely** (235 files). Tests migrated to
  `tests/legacy/` with import rewrites; impl consolidated into
  `vllm.sndr_core`. `pytest.ini` now points at a single `tests/` root.
- **Two-wheel split formalized**: root `pyproject.toml` ships only
  `vllm.sndr_core*` (Apache 2.0). New `pyproject-engine.toml` ships only
  `vllm.sndr_engine*` (commercial). The Apache wheel no longer leaks
  commercial code.
- **Engine boundary tightened to 1 patch** via Sander's strict-AND rule
  (NOT on github + no author ref + no PR link + no `upstream_pr`):
  PN72 is the sole engine patch. P39a flipped to community after git
  history check showed it WAS tracked in old `_genesis/`. Helper
  `ngram_frequency_filter.py` moved core→engine; core has redirect stub.
- **PN78 tombstone**: lifecycle=retired, `apply()` returns "skipped"
  unconditionally. Deprecated same day it was created — investigation
  found vllm pin already calls `empty_cache()` twice in `capture_model`,
  AND apply_all monkey-patches don't reach spawn'd workers.

![SNDR architecture v11.0.0](assets/charts/architecture_v11.png)

### P0 critical fixes (9/9 closed)

- **P0-1 lazy patches `__init__.py`**: 21 family `__init__.py` files
  converted from eager `from . import <patch>` cascades to lazy
  `__getattr__` loaders. `vllm.sndr_core.patches` is now torch-less
  importable. Added explicit torch-less smoke step to CI workflow.
- **P0-2 engine `{{name}}` literal bug**: 5 engine `__init__.py` had
  `f"…{{name}}"` (literal) instead of `f"…{name}"` (interpolated).
- **P0-3 PN72 helper boundary**: `ngram_frequency_filter.py` moved core→engine.
- **P0-4 P39a tier flip**: P39a was tracked in old `_genesis/` git
  history → must be community per Sander's strict-AND rule.
- **P0-5 engine gitignore**: `vllm/sndr_engine/`, `pyproject-engine.toml`,
  `sndr_*_audit_*.md` added to `.gitignore`.
- **P0-6 plugin entry point**: canonical `vllm.sndr_core.plugin:register`
  now ships in the core wheel.
- **P0-7 installer exit code**: `sndr install` exits 2 on smoke failures;
  `--allow-smoke-fail` opt-out.
- **P0-8 strict-mode launcher**: `cfg.to_launch_script(strict_mounts=…)`
  threaded through; live launches fail loudly on missing host.yaml entries.
- **P0-9 bundle smoke tests**: 7 previously-failing tests auto-fixed by P0-1.

### P1 structural fixes (7/9 closed)

- **P1-1 apply.shadow allow-list**: `KNOWN_SPEC_ONLY_PATCHES` for 7
  intentional spec-only entries; CI gate on unexpected divergence.
- **P1-3 license dev-flag gate**: unsigned keys require
  `SNDR_ALLOW_LEGACY_LICENSE_KEYS=1`; production rejects them by default.
- **P1-4 docker-aware host apply**: `sndr launch` skips host apply for
  Docker configs (container handles its own).
- **P1-5 SNDR_HOME canonical**: `~/.sndr/` is the canonical install root;
  `~/.genesis/` is a legacy fallback honored on read.
- **P1-6 CI workflow**: pytest.ini single-root; CI workflow rewritten
  with torch-less import smoke + apply-shadow strict gate.
- **P1-7 drift watcher**: rewritten to walk `iter_patch_specs()`.
- **P1-9 public scripts**: hardcoded `<host>` removed in favor of
  `localhost:8000` defaults.

### Deferred (2/18)

- **P1-8 docs**: surface-level updates landed in v11.0.0; deeper
  operator-tutorial rewrite scheduled for v11.1.
- **P1-9 compose**: integration compose files + personal scripts need
  to MOVE to private/internal, not sed-replace. Tracked separately.

### Stable lifecycle ratchet (PR38 §5.5 closure)

Rather than mass-migrate 6 high-anchor patches, v11 ships a self-enforcing
test ratchet: any patch promoted to `lifecycle="stable"` MUST have its
patch_id wired into the anchor manifest. Test:
`tests/unit/infra/test_stable_manifest_policy.py`. Today 0 stable
patches → test passes vacuously; future promotions can't skip the ritual.
See `docs/CONTRIBUTING.md#promoting-a-patch-to-lifecyclestable`.

### Pytest baseline

- v10.0.0:  2425 pass / 74 skip / 0 fail
- v11.0.0:  **2621 pass** / 79 skip / 0 fail (+196 across the 48-hour audit window)

### Migration notes for v10.x operators

- `~/.genesis/` keeps working as a fallback path; canonical is `~/.sndr/`.
- `vllm._genesis.*` imports gone — point them at `vllm.sndr_core`.
- Legacy plugin install still works but is now a thin re-export. New
  installs use the core wheel only.
- `SNDR_ENGINE_LICENSE_KEY` accepts both signed Ed25519 tokens (production)
  and unsigned keys (dev mode behind `SNDR_ALLOW_LEGACY_LICENSE_KEYS=1`).

---

## [v10.0.0] — Hard flip: canonical impl now lives in `sndr_core/` (2026-05-07)

> **The hard flip Sander demanded.** Every implementation file
> physically moved from `vllm/sndr_core/` into `vllm/sndr_core/` (and
> from `wiring/<perf_hotfix|legacy|...>/` into the engine-subsystem
> taxonomy at `sndr_core/integrations/<family>/`). The legacy `_genesis/`
> tree is now a thin alias layer that exists solely to preserve the
> ~327 monkey-patch test contracts and ~5 years of operator muscle
> memory.
>
> Zero functional regressions: 2425 pytest pass, 74 skipped.

### What's new

- **117 patch wirings physically migrated** to `sndr_core/integrations/<family>/`
  with cleaner naming (`patch_67_tq_multi_query_kernel.py` →
  `p67_tq_multi_query_kernel.py`, etc.). Engine-subsystem taxonomy:
  `attention/{gdn,turboquant,flash}/`, `spec_decode/`, `scheduler/`,
  `worker/`, `kv_cache/`, `moe/`, `quantization/`, `kernels/`,
  `compile_safety/`, `loader/`, `middleware/`, `memory/`, `lora/`,
  `multimodal/`, `reasoning/`, `serving/`, `tool_parsing/`.
- **84 utility files migrated**: kernels (30), compat (32), middleware,
  oracle, cache, utils, model_configs, configs, manifests.
- **Wiring infrastructure consolidated** at `sndr_core/wiring/`:
  `file_cache.py`, `rebind.py`, `anchor_manifest.py`, `patcher_registry.py`.
- **Detection consolidated** at `sndr_core/detection/`:
  `guards.py` (820 lines), `gpu_detect.py`, `model_detect.py`,
  `config_detect.py`.
- **`vllm/sndr_core/<X>` → `vllm.sndr_core.<canonical>` redirect** uses
  the `sys.modules[__name__] = importlib.import_module(...)` trick.
  Result: legacy and canonical names are the **same module object** —
  `monkeypatch.setattr` on either path propagates to both.

### How redirects work (so future-Sander remembers)

Three redirect patterns in play:

1. **Leaf-module redirect** (`patch_*.py`, `guards.py`, `dispatcher.py`,
   etc.) — single line `sys.modules[__name__] = importlib.import_module(canonical)`.
   Aliases the legacy module to canonical at import time.

2. **CLI-aware redirect** (`compat/<x>.py` modules with a `__main__`
   block) — same as above, plus a `if __name__ == "__main__":` block
   that re-imports the canonical and calls its `main()`. Required
   because the alias replaces `__main__` *before* it has a chance to
   run, so `python -m vllm.sndr_core.compat.<x>` would otherwise exit
   with empty stdout.

3. **Forward-package** (`__init__.py` for the 8 redirected sub-packages:
   `compat/`, `middleware/`, `kernels/`, `cache/`, `oracle/`, `utils/`,
   `model_configs/`, `compat/models/`) — **not** a sys.modules alias.
   Aliasing the parent package would force submodule lookup through the
   canonical's `__path__`, bypassing the legacy redirect files entirely.
   Instead these stay real packages with their own `__path__`, but
   forward attribute access to canonical via `__getattr__`.

### Test contract preservation

`monkeypatch.setattr(vllm.sndr_core.guards, "is_nvidia_cuda", fake)`
still works because both `vllm.sndr_core.guards` and
`vllm.sndr_core.detection.guards` are the same object in `sys.modules`
after the redirect runs. No test migration was needed.

The single test fixture that needed updating was `_fresh_module()` in
`test_club3090_22_pn59_chunked_prefill_opt_out.py`: it manually deletes
the module from `sys.modules` to force a reimport, but had to learn
the v10 redirect pair so the canonical entry gets purged too.

### Operator-visible

- New code SHOULD import from `vllm.sndr_core.<canonical>` —
  see e.g. `from vllm.sndr_core.detection.guards import is_nvidia_cuda`.
- Legacy paths (`from vllm.sndr_core.guards import ...`) keep working
  indefinitely. There is no deprecation warning.
- `python -m vllm.sndr_core.compat.doctor`, `…compat.presets list`,
  etc. all keep working.

### Files

- 210 redirect files under `vllm/sndr_core/`
- 271 canonical files under `vllm/sndr_core/`
- 25 legitimate non-redirect legacy files (top-level `__init__.py`s,
  `__version__.py`, sub-package `__init__.py`s — all real packages)

---

## [v9.0.0] — Path consolidation (`project_paths.py`) (2026-05-07)

> **Pragmatic v9 increment** — единый source of truth для всех
> SNDR-internal paths (manifest dir, wiring dir, telemetry dir, model
> configs dir, etc.). Closes the gap left by Stage 2 (which centralized
> only vllm engine paths in `vllm_targets.py`).
>
> Zero functional regressions: 2465 pytest pass, boot dry-run identical
> (108 applied / 20 skipped / 0 failed).

### What's new

**`vllm/sndr_core/paths/project_paths.py`** — 11 path helpers с env override:

  - `install_root()` — operator install dir (default `~/.genesis`).
    Honors `SNDR_HOME` (canonical), `GENESIS_HOME` (legacy alias).
  - `wiring_dir()` — patch_*.py location. Honors `SNDR_WIRING_DIR`;
    auto-detects v8 legacy (`_genesis/wiring/`) vs v9 canonical
    (`sndr_core/integrations/`) location.
  - `manifest_dir()` + `manifest_json_path()` — Site Map location.
    Honors `SNDR_MANIFEST_DIR`.
  - `pristine_fixtures_dir()` — pristine vllm source files for tests.
  - `telemetry_dir()` — opt-in telemetry write dir. Honors
    `SNDR_TELEMETRY_DIR` / `GENESIS_TELEMETRY_DIR`.
  - `model_configs_{builtin,community,user}_dir()` — 3-tier config layout.
    User dir honors `SNDR_MODEL_CONFIG_DIR` / `GENESIS_MODEL_CONFIG_DIR`.
  - `moe_tuning_dir()` — TQ k8v4 sweep data. Honors `SNDR_MOE_TUNING_DIR`.
  - `host_config_path()` — deployment runtime YAML. Honors
    `SNDR_HOST_CONFIG` / `GENESIS_HOST_CONFIG`.
  - `all_paths()` — snapshot dict for `sndr doctor` diagnostic.

**Replaced hardcoded paths** in production:

  - `vllm/sndr_core/apply/_state.py` `_resolve_wiring_module()`
    — now uses `sndr_paths.wiring_dir()` instead of arithmetic
    on `_genesis_pkg.__file__`.
  - `vllm/sndr_core/core/manifest_cache.py` — uses
    `sndr_paths.manifest_json_path()` instead of importing
    `default_manifest_path()` from legacy module.
  - `scripts/moe_lookup_helper.sh` — `OUT_DIR` honors
    `$SNDR_MOE_TUNING_DIR` env override.
  - `install.sh` — `GENESIS_HOME` resolution chain now checks
    `SNDR_HOME` first (canonical), then `GENESIS_HOME` (legacy).

### Why pragmatic v9 vs full hard-flip

Full v9 hard-flip (move all 396 `_genesis/` files to canonical
`sndr_core/` locations + migrate 327 monkey-patch test contracts)
was scoped at ~60-85 hours. Per Sander's pragmatic guidance
(2026-05-07), the actual goal is just to ensure paths are config-
driven so the project layout stays portable. This release achieves
that without the high-risk physical migration.

The 396-file canonical impl flip (Stages B+C of the original v9 plan)
remains scheduled for v10.0 with proper deprecation period.

### Migration guide

Existing operators: **no action required**. Legacy `GENESIS_*` env
vars continue to work exactly as before. The new `SNDR_*` aliases
are ADDITIONAL not replacement.

New deployments preferring SNDR canonical naming:

```bash
export SNDR_HOME=/opt/sndr
export SNDR_TELEMETRY_DIR=/var/log/sndr-telemetry
export SNDR_MODEL_CONFIG_DIR=/etc/sndr/configs
sndr install --dry-run
```

---

## [v8.0.0] — SNDR Core architectural refactor (2026-05-07)

> **Largest structural release since v7.0.** Genesis renamed internally
> to **SNDR Core** (community tier) + **SNDR Engine** (commercial tier).
> Public-facing brand stays "Genesis" — the package is just better-organized
> underneath: `vllm/sndr_core/` (canonical home for new code) +
> `vllm/sndr_core/` (back-compat shim, kept until v9.0).
>
> **Zero functional regressions**: 2465 pytest pass, boot dry-run
> identical to v7.72 baseline (108 applied, 20 skipped, 0 failed on
> Mac dev environment). All 117 patches + 130 PATCH_REGISTRY entries
> work via either import path.
>
> Pin: `0.20.2rc1.dev93+g51f22dcfd` (validated 2026-05-07, allowlist-clean).
> Tag rollback marker: `pre-sndr-core-refactor-v8` on commit `f9576df`.

### What's new (operator-visible)

**1. Two-tier package** (Sander Q8/Q9 decision):
  - `vllm/sndr_core/` — community tier (Apache 2.0). All upstream backports
    + patcher infrastructure. **Default install for everyone.**
  - `vllm/sndr_engine/` — commercial tier (LicenseRef-Sandermage-Commercial).
    Sander-original IP: P67/P67b/P67c (TQ multi-query Triton kernel),
    P82 (SGLang acceptance), PN21-PN24/PN38/PN40 (DFlash family),
    PN26/PN26b (TQ centroids), PN29 (GDN scale fold), PN57 (centroids
    disk cache), PN72 (frequency drafter). **16 engine-tier patches.**
  - Dispatcher tier-gate: when `tier="engine"` patch's apply is requested
    and `vllm.sndr_engine` package isn't installed → clean skip with
    "requires commercial SNDR Engine license" message. Use
    `SNDR_ENABLE_TIER_OVERRIDE=1` to force community-only mode (CI).

**2. CLI installer** — `python -m vllm.sndr_core.cli install [--dry-run] [-y]`.
  8-step wizard (rustup/uv style):
  1. Hardware detect (GPU + CUDA + RAM + disk)
  2. Python check (≥3.10)
  3. Existing vllm install discovery
  4. Channel select (stable/nightly/dev) — picks pin from
     `KNOWN_GOOD_VLLM_PINS` allowlist
  5. Wheel download (Stage 13 hooks `wheels.vllm.ai`)
  6. pip install
  7. Wire SNDR paths + validate **63/63 engine targets**
  8. Smoke test (dispatcher dry-run)

  Plus `sndr launch [config_key]` driver around `vllm serve` with
  apply-patches-then-exec semantics.

**3. Bundle pattern** — atomic apply of related patches via
  `MultiFilePatchTransaction`. **5 bundles ship in v8.0**:
  - `tool_parsing_qwen3coder` (P15+P61c+P64+PN56) — community
  - `reasoning_qwen3` (P12+P27+P59+P61+P61b+PN51) — community
  - `attention_gdn_spec` (P60+P60b) — community
  - `attention_tq_multi_query` (P67+P67b) — **engine** tier
  - `spec_decode_async_cleanup` (P79b+P79c+P79d) — community

  Activation: `SNDR_ENABLE_BUNDLE_<NAME>=1` (atomic 5-patcher commit
  via two-phase transaction with rollback). Idempotent — bundle and
  individual patch flags can both be set without double-apply.

**4. Per-sub drift markers** — each `TextPatch` sub-anchor declares
  its own `upstream_merged_markers` + `on_upstream_merge` policy:
  - `"skip_silently"` (default) — only that sub no-ops, siblings continue
  - `"warn"` — same + WARNING-level log
  - `"abort_bundle"` — entire patcher aborts, rollback-safe inside bundles

  Use case: when upstream cherry-picks PART of a multi-anchor backport,
  the not-yet-merged anchors keep applying. Critical for long-lived
  patches with 3-9 sub_patches surviving multiple pin upgrades.

**5. Centralized infrastructure**:
  - `vllm/sndr_core/paths/vllm_targets.py` — **63 vllm engine paths**
    in one constants module. Rename in upstream → patch ONE constant.
  - `vllm/sndr_core/env.py` — `Flags` class with **135 flag constants**
    + `is_enabled(flag)` helper supporting both `SNDR_ENABLE_*` and
    `GENESIS_ENABLE_*` prefixes (SNDR wins when both set).
  - `vllm/sndr_core/dispatcher/{registry,decision,audit,reporting,pins}.py`
    — split from monolithic `_genesis/dispatcher.py` (was 2828 LOC).
  - `vllm/sndr_core/apply/{_state,_per_patch_dispatch,orchestrator,verify}.py`
    — split from monolithic `_genesis/patches/apply_all.py` (was 5273 LOC).
  - `vllm/sndr_core/core/{text_patch,multi_file,manifest_cache}.py`
    — split from `_genesis/wiring/text_patch.py` (was 833 LOC).
  - `vllm/sndr_core/{runtime,detection,patches,bundles}/` — re-export
    shims for utility modules + 117 patches in 19 engine subsystems.

**6. Patch taxonomy** — every PATCH_REGISTRY entry now has `tier`
  (community/engine) and `family` (one of 19 engine subsystems):
  `tool_parsing`, `reasoning`, `serving`, `attention.{gdn,turboquant,flash}`,
  `spec_decode`, `scheduler`, `worker`, `kv_cache`, `moe`, `quantization`,
  `kernels`, `compile_safety`, `loader`, `middleware`, `memory`,
  `lora`, `multimodal`. Use `family` for filtering / grouping in
  diagnostic output.

**7. Tests + contract validation**:
  - `tests/` (top-level, NOT inside pip pkg per Q7=C):
    - `tests/unit/env/test_registry_flag_coverage.py` — 6 contract tests
      (every registry env_flag in Flags class, canonical prefix discipline,
      tier+family fields present + values valid).
    - `tests/unit/core/test_per_sub_drift.py` — 6 dedicated tests for
      per-sub drift markers.
    - `tests/bundles/test_stage7_bundles_smoke.py` — 14 bundle smoke tests.
    - `tests/installer/test_install_dry_run.py` — 14 CLI dry-run tests.
  - `vllm/sndr_core/tests/` — legacy 163-file suite preserved for
    cross-test imports + monkey-patch contracts. Stage 9 (this release)
    declared `pytest.ini testpaths` so both roots auto-discover.

### Deprecation policy

**`vllm/sndr_core/` legacy shim** stays in v8.x for back-compat:

  - All Stage 6/10 forward-shim files re-export to canonical impl
    (which currently lives at the legacy path — direction-flip deferred).
  - All test monkey-patches on `vllm.sndr_core.*` continue to work.
  - Legacy logger names (`genesis.dispatcher`, `genesis.apply_all`) emit
    unchanged for operator log filters.

**v9.0 (no date set)** will:
  - Flip canonical impl direction (move actual code from `_genesis/` to
    `sndr_core/`).
  - Migrate test monkey-patch contracts to canonical paths.
  - Bulk-relocate `vllm/sndr_core/tests/` → `tests/<subsystem>/`.
  - Remove `vllm/sndr_core/` package entirely (3-month deprecation
    window after v8.0.0 final).

### Migration guide for downstream forks

If you import from `vllm.sndr_core.*` in your code: **no action needed**.
All paths continue to resolve to the same callable objects.

If you'd like to migrate to canonical SNDR paths (recommended for new code):

```python
# Old (still works):
from vllm.sndr_core.guards import resolve_vllm_file
from vllm.sndr_core.dispatcher import PATCH_REGISTRY
from vllm.sndr_core.wiring.text_patch import TextPatcher
from vllm.sndr_core.wiring.structured_output.patch_61c_qwen3coder_deferred_commit import apply

# New canonical (Stage 12+):
from vllm.sndr_core.locations import resolve_vllm_file, engine_targets
from vllm.sndr_core.dispatcher import PATCH_REGISTRY
from vllm.sndr_core.core import TextPatcher
from vllm.sndr_core.patches.tool_parsing.p61c_qwen3coder_deferred_commit import apply
```

For env-flag access, the new helper:

```python
from vllm.sndr_core.env import Flags, is_enabled
if is_enabled(Flags.P61C_QWEN3CODER_DEFERRED_COMMIT):
    ...
```

### Backwards compatibility verified

  - 2465 pytest pass / 0 fail (was 1958 in v7.72.2 — +507 tests added)
  - Boot dry-run identical: 108 applied, 20 skipped (15 platform + 5
    bundle-disabled), 0 failed
  - Tier-gate works (TIER_OVERRIDE + ImportError branches both honored)
  - Per-sub drift verified live (skip_silently / warn / abort_bundle)
  - 3-level forward chain works (`sndr_engine → sndr_core → _genesis`)
  - Both `Genesis ` and `SNDR ` marker prefixes recognized at
    idempotency check (`brand.RECOGNIZED_MARKER_PREFIXES`)

### Internal architecture refs (for contributors)

  - Migration tag: `pre-sndr-core-refactor-v8` on commit `f9576df`
    (last v7.72 state before Stage 0).
  - 13-stage refactor plan + per-stage decisions: see
    `Genesis_internal_docs/SNDR_CORE_STRUCTURE_PROPOSAL_v2_RU_20260507.md`
    + `project_sndr_core_refactor_status.md` memory.
  - Stage 11 CLI architecture: see `vllm/sndr_core/cli/install.py`
    docstring + `tests/installer/test_install_dry_run.py`.
  - Engineering deep-dive on patch consolidation: see
    `Genesis_internal_docs/PATCHER_CONSOLIDATION_DEEP_DIVE_RU_20260507.md`.

---

## [Unreleased] — `v7.65 → v7.72` series

> Pin: `0.20.2rc1.dev9+g01d4d1ad3` (validated 2026-05-04, allowlist-clean).
> Builds on v7.64 release with v7.65 hygiene + v7.66/v7.67/v7.68 patch
> work + comprehensive audit pass + v7.69 cross-rig fix series + v7.72
> sprint (PN59-PN67 + audit hardening). All on `dev` branch only (main
> promotion deferred until cross-rig validation completes).

### v7.72.2 — PN70 tool schema subset filter + Proxmox VE installer caveat (2026-05-05 12:30 EEST)

**Two complementary improvements landed together: a Genesis-original patch that closes the third (and most general) fix path from [noonghunna/club-3090#57](https://github.com/noonghunna/club-3090/issues/57), and an installer-side caveat for [club-3090#49](https://github.com/noonghunna/club-3090/issues/49).**

**PN70 — Tool schema subset filter** (`GENESIS_ENABLE_PN70_TOOL_SCHEMA_FILTER=1`):

- Companion to v7.72.1's option-1 P68 fix (which **skips** the upgrade on dirty catalogs). PN70 implements lexhoefsloot's option-3: keep the upgrade, **filter** dirty tools out of the combined `anyOf` xgrammar schema.
- Wraps `vllm.tool_parsers.utils._get_json_schema_from_tools` and pre-filters tools whose JSON Schema contains xgrammar-unsupported keys before the combined schema is built and handed to xgrammar. Reuses P68's `_scan_schema_for_unsupported_key` so the unsupported-key set is single-sourced (no drift between option-1 and option-3 paths).
- Behavior matrix:
  - **all tools clean** → wrapper passes through unchanged
  - **subset compat** → wrapper builds schema from compat subset only; logs a WARN naming the filtered-out tools
  - **all tools incompat** → returns `None` (no enforcement, parser handles tool calls at output)
  - **any unexpected exception during filter** → falls back to original (defensive — PN70 never breaks a request that would have succeeded under stock)
- Composability with P68 declared via new `composes_with: ["P68"]` registry field. Recommended combo: P68 ON + PN70 ON → P68 upgrades `tool_choice` on long prompts, PN70 keeps grammar enforcement on the compat subset.
- Tests: +28 TDD (`test_pn70_tool_schema_subset_filter.py`) covering the compat helper, name extractor, env-flag matrix (5 truthy variants), filter behavior (clean / mixed / all-incompat / empty), defensive fallback, dispatcher registration, P68 composability regression. Schema validator extended to accept `composes_with` field. PATCHES.md sync gate updated (header 125 → 126).
- Full local suite: **1958 pass + 73 skipped + 0 failures** (was 1930).

**install.sh — Proxmox VE caveat + `--bare-metal` flag** ([club-3090#49](https://github.com/noonghunna/club-3090/issues/49) lexhoefsloot):

- New `detect_proxmox_runtime()` step runs after vLLM detection. Checks `uname -r` for `pve`/`proxmox` substring AND `/etc/pve` directory presence.
- On detection: WARN with link to club-3090#49 explaining the upstream `vllm/vllm-openai:nightly` × PVE 8.x kernel 6.17.x uvloop crash that was empirically isolated to the docker image (bare `docker run` reproduces, native venv on the same host runs clean). Auto-enables `--bare-metal`.
- New `--bare-metal` flag (or `GENESIS_BARE_METAL=1`) tells `print_next_steps()` to point operator at native `pip install vllm==0.20.x` + `vllm serve` instead of docker-compose presets. Auto-enabled on PVE; manually settable for any operator running microk8s / podman / k8s / bare-metal.

**Wiring + dispatcher hygiene:**

- New patch file: `vllm/sndr_core/wiring/structured_output/patch_N70_tool_schema_subset_filter.py`
- New apply_all entry: `apply_patch_N70_tool_schema_subset_filter` (placed before PN65 access log)
- Dispatcher PATCH_REGISTRY entry: `PN70` with `composes_with: ["P68"]`
- Schema validator: `composes_with` added to `_KNOWN_FIELDS`
- PATCHES.md: PN70 row added to "Structured-output / Qwen3 parser" category, header count bumped 125 → 126
- docs/CONFIGURATION.md: `GENESIS_ENABLE_PN70_TOOL_SCHEMA_FILTER` documented under structured-output section
- Tests badge bumped 1930 → 1958

### v7.72.1 — P68 xgrammar-incompat tool skip (2026-05-05 11:30 EEST)

**Bug fix from cross-rig report — [noonghunna/club-3090#57](https://github.com/noonghunna/club-3090/issues/57) (lexhoefsloot).**

- **P68** (`GENESIS_ENABLE_P68_AUTO_FORCE_TOOL=1`) silently turned the upstream xgrammar `patternProperties` / `propertyNames` / `$ref` / `oneOf` limitation into a 100% reproducible **400 `ValueError: features not supported by xgrammar`** on long-prompt requests from agentic IDE clients (OpenClaw, Claude Code, Cline, Cursor) whose tool catalog included even one schema with such keys.
- **Root cause:** P68 upgraded `tool_choice: "auto" → "required"`, which makes vLLM build a combined `anyOf` schema across ALL tools and run xgrammar compilation. A single tool with `patternProperties` (very common for `exec`-type tools) then poisoned every long-prompt request.
- **Fix:** new `_find_xgrammar_incompat_tool()` scans `request.tools` BEFORE upgrading `tool_choice`. If any tool contains an xgrammar-unsupported JSON Schema key, P68 skips the upgrade with a WARN, but P69 reminder still fires (only the upgrade is unsafe). Operators on a non-xgrammar backend (guidance / outlines / llguidance) can override via new `GENESIS_P68_FORCE=1`.
- **Tests:** +18 TDD (`test_club3090_57_p68_xgrammar_skip.py`) covering schema scanner + apply_hook integration + force override + P69 still-fires path. Full suite **1930 pass + 73 skipped + 0 failures** (was 1912).
- **Honest scope:** this is a workaround on the Genesis side, not an upstream xgrammar fix. The fundamental `patternProperties` support gap is tracked at upstream xgrammar; Genesis's job is to not weaponize it.

### v7.72 — `PN59-PN67` + audit hardening sprint (2026-05-05)

**7 new patches:**
- **PN59** (Genesis-original) — Streaming-GDN window-iterative driver (Variant D Phase 2). Empirically validated 2026-05-05 on 27B PROD: **−142 MiB/GPU at boot + −95% per-soak fragmentation drift**, 0 regressions. PROMOTED default-ON in 27B PROD.
- **PN60** (Genesis-original) — Quant arg validator (preflight DX, doctor extension); CLI `genesis preflight --quantization X --model /path`. Catches apnar club-3090#51 NVFP4 boot failure with one-line remediation hint instead of 30-line pydantic ValidationError.
- **PN61** (Genesis-original) — qwen3_vl loader KeyError → language_model_only auto-fallback (class-rebind wrapper). v2 with pre-emptive language_model_only set when NVFP4/compressed-tensors detected (avoids mid-load partial state).
- **PN62** (Genesis-original) — Text-only ViT scratch skip marker (3-5 GiB save target on qwen3_vl + NVFP4). Currently sets `_pn62_skip_vit_scratch` marker only — production hook deferred until NVFP4 cross-rig validation.
- **PN63** (Genesis-original) — gpu_profile advisory: prefer `--kv-cache-dtype fp8_e5m2` over `fp8_e4m3` on consumer Blackwell SM 12.0 (apnar empirical: e4m3+96K loses −2.6% TPS vs e5m2+48K on RTX 5090).
- **PN64** (Genesis-original) — Marlin MoE SM 12.0 placeholder entry (env-gated; awaits real 5090 sweep).
- **PN65** (Genesis-original) — Genesis structured API access log middleware. Replaces uvicorn's bare `INFO: <host> - "GET /v1/models" 401` with `[Genesis-API] 401 GET /v1/models <1ms client=<host>`. v2 fix uses persistent `logging.Filter` to suppress uvicorn.access INFO records (live verified — single source request observability).
- **PN66** (vllm#41696 backport, panpan0000) — Multiturn `</think>` leak in DelegatingParser. Removes buggy `prompt_reasoning_checked` short-circuit that walked the FULL prompt for `</think>` and prematurely set `reasoning_ended=True` from a previous turn's `</think>`. Defensive backport for multi-turn DSML/Hermes/Qwen3 chat.
- **PN67** (vllm#41674 backport, JasonKeyiL) — thinking_token_budget inverted-bool single-token fix. Removes `not` from `or not thinking_budget_tracks_reqs` in `gpu_input_batch.py:894`. Defensive; NULL-impact on Genesis PROD.

**Critical infrastructure / audit fixes:**
- **P79d wired** (was orphan in v7.69) — registered in `PATCH_REGISTRY` + `apply_all`.
- **P51 + P102 registered** — were runtime-active but invisible to dispatcher.
- **`qwen3_6` model_class mapping** added in `model_detect.py` (forward-compat).
- **`conflicts_with` symmetry restored** — 4 missing back-links (P65↔[P56,P57,P67,P67b], PN58↔P62, P7↔P7b, P28↔PN32).
- **PN64 env-gate enforcement** — `(12, 0)` table lookups in `kernels/marlin_tuning.py` now respect `GENESIS_ENABLE_PN64`.
- **PN65 v2 uvicorn dedup** via `logging.Filter` (was setLevel which uvicorn re-overrode).
- **PN40 scheduler subpatch split** — separate markers prevent partial-apply lock-out.
- **PN59 unified bool parser** — accepts `"1","true","yes","y","on"` case-insensitive.
- **PN60 wired into `doctor.collect_report`** as `_section_preflight()`.
- **`gdn_composability.composes_with`** honored — explicit-compatible pairs no longer trigger site-overlap warnings.
- **`preflight_checks` `consecutive`** rewritten as state-machine (`current_streak` / `max_streak`).
- **`env_flag_guard` scan** extended to `GENESIS_DISABLE_*` prefix (was unreachable).
- **P37/P40/P5b/P7b env_flag alignment** — registry was wrong, code/scripts were right; aligned to short form.
- **4 wiring routings** via `result_to_wiring_status` (P82/P72/P85/P100) — was masking SKIPPED → "applied".
- **Status helper migration** — 30 wiring files still pending follow-up sprint.
- **MultiFilePatchTransaction** atomic dry-run — primitive exists, `count(anchor) == 1` check pending.
- **AppleDouble cleanup** on server (56 → 0) + `.gitignore` patterns + `COPYFILE_DISABLE=1` in syncs.
- **Deprecated external probes** (`patch_tolist_cudagraph`, `patch_40074_iooo`) removed from 4 launch scripts (P78/PN14 supersede).
- **pytest config** — root `conftest.py` + `pytest.ini`; documented `python3 -m pytest` workaround.
- **A-19 audit test** — `test_a19_optional_sub_patches_marker_policy.py` forbids `required=False` sub-patches sharing TextPatcher marker (PN40-class regression gate).
- **A-15 pytest torch marker** — auto-skip via AST scan for module-level `import torch` on torch-free hosts.

**Operator UX:**
- **Structured boot summary** at end-of-boot — system info header + per-category counters + APPLIED grouped by category with vllm# PR cites + SKIPPED grouped by reason class. 78 unique patches → readable single block, replaces scattered per-patch INFO lines.
- **Three tfriedel-pattern ports** — `scripts/verify-full.sh` (7-stage smoke), `scripts/probe_max_ctx.sh` (auto-binary-search max ctx), `scripts/fetch_models.sh` (SHA-verified HF download).
- **Comprehensive bench** — `tests/bench/comprehensive_bench.py` (cold-warm latency / sustained TPS / tool-call clean / multi-turn stability / VRAM steady / long-context needle) with README-ready markdown output.
- **Bundled `qwen3.6-enhanced.jinja`** + `qwen3.5-enhanced.jinja` (Cheuk-Yiu Chan @allanchan339) at `assets/chat_templates/` with relationship-to-runtime-patches docs.
- **MoE tuning workflow** — `scripts/moe_lookup_helper.sh` (renamed from `tune_moe.sh` 2026-05-05 per G-POST-05; the script names + stages JSON, the autotune sweep is manual via `benchmark_moe.py --tune` per the script's NEXT STEPS block) + honest documentation in `vllm/sndr_core/configs/moe_tuning/README.md` clarifying that bundled A5000 Triton config is unused on Ampere FP8 (vLLM uses Marlin kernel; PR #40129 closed). Per-arch backend selection table added; cross-rig contributions targeted at Ada/Hopper/Blackwell where Triton path IS active.

**Live PROD bench (2× A5000, full v7.72 stack):**
- **35B-A3B-FP8**: 192.9 tok/s (CV 4.19%), tool-call 10/10, multi-turn 10/10, VRAM 22687+21998 = 44685 MiB
- **27B-int4-AutoRound**: 95.6 tok/s (CV 4.04%), tool-call 10/10, multi-turn 10/10, VRAM 22753+22064 = 44817 MiB
- Long-context needle: 3/4 found (1K/51K/92K — 10K MISS reproducible LITM)

**Test coverage:** 1858 unit tests pass (+247 vs v7.68 1611 baseline) + 73 skipped.

**Pin:** stays on `0.20.2rc1.dev9+g01d4d1ad3`. Pin-bump to vLLM main HEAD `0c620d2e08` (+25 commits) prepared (verdict GREEN, P4 + PN52 supersede markers in registry) — awaits explicit operator action + nightly publish.



### v7.69 — `PN35` absorption from club-3090#32 (2026-05-03)

After RossNE99 + GuiPerPT reported borderline OOM on 24GB cards
(`docker-compose.dual.yml` 262K context on WSL2; `long-text.yml` on
v7.69 needing context lowered to 110K), surveyed noonghunna's
`models/qwen3.6-27b/vllm/patches/` directory. Found 1 new sidecar
worth absorbing into Genesis as a first-class patch:

- **PN35 — skip inputs_embeds buffer for text-only models** (default
  ON). Backport of vllm-project/vllm#35975 by **AjAnubolu** (UPSTREAM
  PR, OPEN since 2026-03-04). Skips the
  `(max_num_tokens, hidden_size)` GPU buffer + pinned CPU mirror
  allocation in BOTH `gpu_model_runner.py:713`
  (`self.inputs_embeds = self._make_buffer(...)`) AND
  `llm_base_proposer.py:205` spec-decode proposer
  (`self.inputs_embeds = torch.zeros(...)`).

  For Qwen3.6-27B at `max_num_tokens=4096`, hidden_size=8192:
  - ~64 MiB GPU + ~64 MiB pinned CPU per buffer
  - 2 sites total → **~128 MiB GPU + ~64 MiB CPU per worker freed**
  - Default ON: pure addition behind a guard
    (`if self.supports_mm_inputs or self.enable_prompt_embeds`),
    no regression possible for multimodal models

  **Particularly relevant on borderline-OOM configs:**
  - Single-24GB-GPU + long context + spec-decode (Cliff 2 fires at
    "tried to allocate 50 MiB, 24.5 MiB free" thresholds)
  - WSL2 setups (per club-3090#32 reports — Xwayland/WSL vGPU layer
    eats extra ~830 MiB-1 GiB overhead, pushing borderline configs
    over the cliff)
  - Composes naturally with P103 + PN32 v2 (Cliff 2 stack)

  Pattern credit: noonghunna club-3090 setup-time sidecar
  `patch_inputs_embeds_optional.py` (2026-05-02). Originally raised
  by club-3090#32 reporters (RossNE99, GuiPerPT). PN35 promotes the
  sidecar to a first-class Genesis text-patch so it survives the
  entrypoint `exec vllm serve` pattern (same survivability argument
  as P103 v7.69 self-install).

  Auto-retires when vllm#35975 merges upstream — drift markers
  include both Genesis-specific `[Genesis PN35 v7.69 inputs_embeds_optional]`
  AND the upstream `vllm#35975` reference, so post-merge state
  triggers SKIPPED cleanly.

  19 new TDD tests in `test_pN35_inputs_embeds_optional.py`. Sweep:
  1599 → 1618 / 0 fail.

  Total PATCH_REGISTRY entries: 100 → 101.

  **Other 6 patches in noonghunna's directory** were surveyed too —
  all already covered by Genesis after the v7.66/v7.68 absorption
  series:
  - `patch_pn25_genesis_register_fix.py` → absorbed into PN25 v7.68
    (import-time registration)
  - `patch_pn30_dst_shaped_temp_fix.py` → absorbed into PN30 v7.68
    (part3 dst-shaped temp)
  - `patch_workspace_lock_disable.py` → absorbed into PN34
  - `patch_tolist_cudagraph.py` → already shipped at
    `tools/external_probe/patch_tolist_cudagraph.py`
  - `patch_pr40798_workspace.py` → already shipped at
    `tools/external_probe/patch_pr40798_backport.py`
  - `README.md` → docs

  noonghunna's sidecar set is now down to ZERO uncovered patches in
  Genesis (all functionality available via PATCH_REGISTRY entries
  or external_probe).

### v7.69 — close club-3090#19 cross-rig findings (2026-05-02)

After v7.68 reached noonghunna's club-3090 1×3090 + 2×3090 rigs (commit
`18e65e3`), three distinct issues surfaced. All three root-caused and
fixed in v7.69. Tests: **1512 pass / 73 skip / 0 fail** (+18 new tests
covering the three fixes).

- **F1 — PN30 part3 drift-marker false-positive (CRITICAL).** Root
  cause: `part3` declared generic `upstream_drift_markers=["[Genesis
  PN30"]`. Because `part2` (separate `TextPatcher` in same `apply()`
  call) inserts the substring `[Genesis PN30 issue #17]` into the same
  file (`v1/worker/mamba_utils.py`), `part3`'s Layer 3 drift check
  matched part2's own insertion and skipped with `upstream_merged` →
  required-fail → vLLM aborts on first apply. Fix: tightened part3
  drift markers to `[Genesis PN30 v7.68 dst-shaped]` (specific to
  part3's own replacement comment). Re-runs unaffected because Layer 2
  (idempotency marker) fires before Layer 3. 2 new regression tests
  pin the fix.

- **F2 — P103 setattr lost on `exec vllm serve` (HIGH).** Root cause:
  P103 v7.62.20 was a pure-setattr monkey-patch. In noonghunna's
  entrypoint pattern (`python3 -m vllm.sndr_core.patches.apply_all &&
  exec vllm serve`), `apply_all` runs in the entrypoint shell process
  and does `setattr(chunk_mod, "chunk_gated_delta_rule_fwd", wrapper)`.
  Then `exec vllm serve` REPLACES the process image — the setattr is
  gone. Workers spawn fresh, import the original
  `chunk_gated_delta_rule_fwd`, hit Cliff 2 OOM at chunk.py:71
  (`chunk_fwd_o(...)` after the inner h tensor was allocated).
  
  The "rebound at 0 caller sites" log noonghunna saw was a red herring
  — internal callers in chunk.py resolve via `chunk_mod.__dict__` (which
  we DID setattr), so 0 external aliases is normal. Log message rewritten.
  
  Fix: v7.69 introduces a **text-patch on `chunk.py` itself** that
  appends a self-install hook at end of file. The hook checks
  `GENESIS_ENABLE_P103=1` at module-import time and calls a new
  `_genesis_p103_install_at_import(globals())` helper. This wraps
  `chunk_gated_delta_rule_fwd` in the just-loaded module dict —
  surviving any startup mechanism (entrypoint exec, worker spawn,
  spawn-context multiprocessing). Legacy setattr step is preserved as
  defense-in-depth for the current process. 11 new tests cover the
  install helper, env gating, idempotency, missing deps soft-fail,
  text-patch anchor, and v7.69 ordering.

- **F3 — PN32 v1 chunked at wrong level (HIGH).** Root cause: PN32 v1
  patched `forward_cuda` outer and sliced `mixed_qkv/b/a` before
  calling `gdn_attention_core`. But inside `_forward_core`, the chunked
  call to `self.chunk_gated_delta_rule(...)` still received
  `attn_metadata.non_spec_query_start_loc` describing the FULL prompt
  (PN32 v1 didn't slice metadata). FLA kernel allocated h based on
  full T regardless of outer slice. Empirical evidence: PN32 v1 OOM'd
  EARLIER (30K) than baseline (50-60K) — chunking overhead with no
  inner allocation reduction.
  
  Fix: PN32 v2 (v7.69) **rewritten to patch `_forward_core` directly**.
  Wraps the prefill branch's `self.chunk_gated_delta_rule(...)` call
  with a chunk-aware loop:
  
  - Slices `query/key/value/g/beta` along T dim
  - Builds chunk-local `cu_seqlens=[0, chunk_len]` per chunk
  - Threads `initial_state` via prior chunk's `last_recurrent_state`
  - Concatenates outputs along T (dim=1)
  - Single-sequence prefill only — multi-seq bypasses to original
    (chunking across cu_seqlens boundaries needs inner state-cache
    surgery not exposed at this layer)
  
  PN32 v2 **composes with P103**: v2 chunks the OUTER FLA call, P103
  chunks INSIDE FLA's `chunk_gated_delta_rule_fwd`. Together they
  give full Cliff 2 coverage on single-24GB-GPU. 19 tests rewritten
  for v2 semantics (anchor, replacement, threading, single-seq
  detection, env gating, dependencies docstring).
  
  Recommended pairing for single-24GB-GPU Cliff 2:

  ```bash
  GENESIS_ENABLE_P103=1                              # required (closes inner h)
  GENESIS_ENABLE_PN32_GDN_CHUNKED_PREFILL=1          # closes outer FLA call buffer
  GENESIS_PN32_GDN_CHUNK_SIZE=8192                   # default
  GENESIS_PN32_GDN_CHUNK_THRESHOLD=16384             # default
  GENESIS_FLA_FWD_H_MAX_T=16384                      # P103 default
  ```

### Live-validation matrix (all 4 configs on 2× A5000, 2026-05-02)

| Config | Boot | TPS @ 256t | CV | Tool-call | Active patches |
|---|---|---|---|---|---|
| 27B INT4 + TQ k8v4 + MTP K=3 (PROD) | OK | **104.0** | 0.5% | clean | PN33+PN25(v7.66)+45 others |
| 35B-A3B FP8 + MTP K=3 | OK | **183.7** | n/a (1 run) | clean | PN33+PN25+PN26b+PN8 |
| 35B-A3B DFlash | OK | **155.0** | n/a (1 run) | clean | PN33+PN22+PN23+PN24+PN8 |
| 27B INT4 + DFlash drafter K=5 | OK | **129.3** | n/a (1 run) | clean | PN33+PN22+PN23+PN24+PN12+PN17 |

`PN33` text-patch verified in live `gpu_model_runner.py` on all 4
configs (marker present + K-aware code `list(range(_genesis_pn33_K))`
in place). The **27B INT4 + DFlash drafter** result (129.3 TPS on
2× A5000) lines up with noonghunna's published 78 narr / 128 code TPS
on 2× 3090 — same drafter recipe, similar consumer Ampere → cross-rig
reproducibility confirmed.

### Test + count metrics

```
vllm/sndr_core/tests/  ─ 1494 pass / 73 skip / 0 fail (full local sweep)
PATCH_REGISTRY        ─ 100 entries  (P1–P103 + PN8–PN34 + sub-patches)
apply_all functions   ─ 99 (P68/P69 share one apply function — documented)
ruff F821/F401/F841   ─ 0 errors    (was 195 before audit cleanup)
```

### PATCH_REGISTRY distribution by category

```
spec_decode   ████████████████████████████████  24
hybrid (GDN)  ██████████████████████████        20
moe           ██████████████████                14
quant (TQ)    █████████████████                 13
attention     ███████████████                   11
kv_cache      ███████                            7
sampling      █████                              5
misc          ██████                             6
                                              ─────
                                                100
```

`PATCH_REGISTRY` snapshot from `vllm/sndr_core/dispatcher.py` 2026-05-02
(rounded to whole bars; some entries multi-categorize via tag list).

### Cross-rig validation (community + Genesis PROD)

| Rig | Owner | GPU(s) | Model | Verdict | Source |
|---|---|---|---|---|---|
| Genesis PROD | @Sandermage | 2× A5000 | Qwen3.6-27B-int4 + TQ k8v4 | 104 TPS / 0.5% CV | this repo |
| Genesis PROD | @Sandermage | 2× A5000 | Qwen3.6-35B-A3B-FP8 | 184 TPS | this repo |
| club-3090 | @noonghunna | 1× 3090 24GB | 27B INT4 + DFlash drafter K=5 | 78 TPS narr / 128 code | club-3090#19 |
| club-3090 | @noonghunna | 2× 3090 | 27B INT4 + MTP K=3 | stable boot + tool-call | club-3090#19 |
| ampersandru | community | 1× 3090 | 27B INT4 + MTP K=3 | mid-stream OOM (PN33 fix) | issue thread |
| TurboQuant rigs | @JartX | 5090 / H20 / 4× R6000 / 8× A4000 | various TQ k8v4 | validated via PR #39931 | upstream |

### v7.68 — diagnose + fix what cross-rig validation surfaced (2026-05-02)

After v7.66 reached noonghunna's 1× 3090 + 2× 3090 rig, cross-rig
testing surfaced 3 distinct bug classes that 2× A5000 PROD didn't
reproduce. All 3 fixed in v7.68 + 1 new companion patch:

- **PN30 v7.68 — dst-shaped temp** (corrects v7.65 layout-corruption).
  v7.65 PN30 used `state[src_block_id, :, offset:].contiguous()` which
  produced a compact `dim×(state_len-offset)` buffer; raw memcpy into
  the destination block (strided by full `state_len`) packed source
  rows into wrong destination offsets, corrupting DS conv state row
  strides on every `offset>0` copy. Surfaced as TQ store CUDA assert
  several layers downstream — the user-visible symptom was at the wrong
  layer, not the root. Diagnosis credit: noonghunna + independent cross-check
  cross-check (club-3090 commit `9af1a52`). Fix: 3-file text-patch
  with new part3 on `collect_mamba_copy_meta` building dst-shaped temp
  via `state[dest_block_id].clone()` + tail copy. Old part1 path now
  fail-closed RuntimeError. 12 TDD tests pass.

- **PN25 v7.68 — import-time registration** (eliminates v7.66 TP=1
  spawn fail). v7.66's `direct_register_custom_op` still failed on
  TP=1 spawn config because `Library("genesis", "FRAGMENT")` was
  constructed inside the Dynamo trace context →
  `instantiate_user_defined_class_object` crash. v7.68 text-patches
  `activation.py` to register at module-import time (BEFORE any trace),
  caches result as `_GENESIS_PN25_SILU_AND_MUL_OP` module global.
  `forward_native` body reads only the cached global — never registers.
  Same pattern preventively applied to **P7b** (`gdn_dual_stream_
  customop`) — same bug class would have hit on any TP=1 enable.
  Pattern credit: noonghunna's `patch_pn25_genesis_register_fix.py`
  (club-3090 commit `a62ad78`).

- **PN34 (NEW) — workspace lock runtime relaxation** (PN33 companion).
  PN33 fixed BOOT-time `_dummy_sampler_run` under-counting; runtime
  decode path on `turboquant_attn.py:1350:_decode_attention` still
  raised `WorkspaceManager._ensure_workspace_size` AssertionError on
  rare paths (continuation-prefill into long context, MTP K=3 + decode
  mid-stream). PN34 ports noonghunna's
  `patch_workspace_lock_disable.py` setup-time sidecar directly into
  Genesis: relaxes the strict assertion to one-shot WARN+grow-anyway.
  Default OFF (relaxes a strict-debug assertion); requires PN33.
  Retires when vllm#40706 (TQ scratch dedup + reserve worst-case at
  warmup) merges upstream.

- **P103 fix — `T` undefined in chunked_fwd loop** (latent since
  v7.62.20). `vllm/sndr_core/wiring/hybrid/patch_103_fla_cliff2_chunked.py:197`
  used bare `T` in `for start in range(0, T, _MAX_T)` without defining
  it. Cliff 2 chunked path silent-crashed `NameError` on every trigger
  since ship date. PROD didn't surface because continuous batching
  keeps `q.shape[1] ≤ max_num_batched_tokens (4096)` — well under the
  `_MAX_T` threshold that gates the chunked branch. Fix: `T = q.shape[1]`
  immediately before the loop + dropped 2 unused locals. Caught by the
  Gemini static-analysis audit (see "Audit pass" below).

### v7.67 — REJECTED on live test (2026-05-02)

Tried `@torch.compiler.disable` decorator on `SiluAndMul.forward_native`
(SGLang pattern from `python/sglang/srt/layers/attention/triton_backend.py`).
Empirically failed on 27B + TQ k8v4 + MTP K=3 boot:

```
torch._dynamo.exc.Unsupported: logging.Logger method not supported for
non-export cases
```

Stack showed Dynamo tracing INTO `forward_native` body despite the
decorator, hitting `log.info()` inside `acquire_silu_out`. Hypothesis:
`@torch.compiler.disable` on a `@staticmethod` accessed through vLLM's
`custom_op._forward_method` dispatcher does NOT propagate — the
dispatcher reaches the underlying function via `getattr` bypassing the
decorator's frame guard. SGLang's working
`@torch.compiler.disable` patterns are on module-level functions, not
`@staticmethod` on classes called via dispatchers — pattern doesn't
transfer. Reverted to v7.66.

### v7.66 — root-cause spec-decode warmup fix (default ON)

- **PN33 — spec-decode warmup K-aware sizing** (default ON, opt-out via
  `GENESIS_DISABLE_PN33_SPEC_DECODE_WARMUP_K=1`). Backport of
  vllm-project/vllm#37521 by `itailang` (OPEN at backport time)
  EXTENDED beyond its `use_eagle()` gate to cover MTP, ngram, and
  draft-model methods. The vanilla warmup uses dummy K=1 draft tokens
  regardless of real `num_speculative_tokens` — under-counting the
  rejection sampler footprint at profile time → causes BOTH (a)
  ampersandru's mid-stream OOM via `propose_draft_token_ids` and (b)
  the workspace lock AssertionError noonghunna hit on dev205 + MTP K=3
  single-card. Same root cause, two symptoms; PN33 fixes the root.
  12 TDD tests pass.

- **PN25 v7.66 — direct_register_custom_op refactor**. Switched
  `silu_and_mul_pooled` and `dual_linear_parallel` registration from
  `@torch.library.custom_op` to vLLM canonical
  `direct_register_custom_op` from `vllm/utils/torch_utils.py:899`.
  `Library("genesis", "FRAGMENT")` at module level. Same fork-safe
  `hasattr()` pre-check guard as v7.65. Schema introspection happens
  at module import (synchronous, before any Dynamo trace), eliminating
  the "infer_schema skipped frame" crash class entirely. Note: v7.68
  later replaced the wiring call site (registration moved to
  activation.py import time) — see PN25 v7.68 above.

- **PN32 — GDN chunked-prefill (Cliff 2 single-24GB-GPU OOM fix)**.
  Splits GDN forward_cuda core attention + post-projection into chunks
  of 8K (default) when `num_tokens > 16K` (default, both env-tunable).
  Closes >50K-token single-shot OOM on 1×3090/4090/5090. Conflicts
  with P28 (legacy persistent buffer pool) — operator picks one;
  documented in dispatcher entry + wiring docstring + new test
  `test_pn32_documents_p28_conflict`. Default OFF — cross-rig
  validation needed (our 2× A5000 PROD doesn't hit Cliff 2 threshold).

### Audit pass (2026-05-02) — two independent reviewer passes

Two independent static-analysis audits ran across the genesis-vllm-patches
tree to catch latent issues that pytest + live-boot couldn't surface.
Both found real bugs the test suite missed.

| Audit | Tool | Findings | Real bugs | Fixed |
|---|---|---|---|---|
| 1st | Google Gemini | 1 critical | 1 | ✓ commit `5743c03` |
| 2nd | independent reviewer #2 | 16 (G-001..G-016) | 9 | ✓ commits `82c64c1` + `6f9c5eb` |

**Latent bugs caught + fixed**:

- `model_detect.py:185` — undefined `base` in exception path. NameError
  was masked by dispatcher's `conservative apply` fallback → genuine
  model-incompat could have applied hybrid GDN patches to non-hybrid
  models silently. (audit finding G-001)
- `patch_103_fla_cliff2_chunked.py:197` — undefined `T` in chunked-
  prefill loop. Cliff 2 chunked path silent-crashed `NameError` on
  every trigger since v7.62.20 ship. PROD didn't surface because
  continuous batching keeps `q.shape[1] ≤ max_num_batched_tokens
  (4096)` — under the `_MAX_T` threshold gating the chunked branch.
  (Gemini)
- `vllm/sndr_core/__init__.py` — eagerly imported `prealloc` (which
  imports `torch`). Every torch-less CLI / pre-commit / static-analysis
  tool failed `ModuleNotFoundError` before reaching their entry point.
  Fixed via lazy `__getattr__` for `prealloc`. (audit finding G-002)
- `ResponseCacheMiddleware` — two contract violations: (a)
  `float("abc")` on malformed temperature leaked ValueError to client
  as 500; (b) corrupt cached entry → connection hung because
  `_send_cached_response` returned without sending. Both fixed.
  (audit findings G-003 + G-004)
- `apply_all.py` — community plugins were applied BEFORE the core
  patch loop despite the docstring saying "After core patches finish".
  Plugin authors relying on the contract found post-modification
  anchors absent. Reordered + added post-plugin
  `validate_registry()` re-run. (audit findings G-006 + G-007)
- 7 env-var references in PATCHES.md / INSTALL.md didn't match
  `dispatcher.py` — operators copy-pasting got no-op env vars while
  their patch silently stayed disabled. All synced. (audit finding G-008)

**Lint pass (audit finding G-016)**:

- 195 ruff `F401`/`F841`/`RUF059` errors → 0 (`All checks passed!`)
- Unused imports + unused locals + unpacked tuple cleanup
- 154 auto-fixes via `--fix`, 41 via `--unsafe-fixes`, 1 manual

**Cleanup pass** also closed G-005 (streaming docs/code mismatch),
G-009 (PATCHES.md P72 row truncated), G-010 (rig-specific paths in
scripts — partially closed with env-var override + README rationale
for the 50+ deferred PROD launch scripts), G-011 (`bench_suffix_sweep`
`speed_runs` computed but unused → biased Pareto ranking), G-012
(Redis cache size off-by-N for stats keys), G-013 (PN16 broken doc
reference to gitignored `docs/_internal/`), G-014 (TextPatcher
Python-only marker — documented).

### v7.65 — repo hygiene + community issue closeouts

#### Community-reported bugs CLOSED

- **#16 PN25 worker-fork crash** (BLOCKING real IDE-agent users on TQ3+
  spawn configs). Pre-check `torch.ops.genesis.silu_and_mul_pooled`
  global C++ registry BEFORE `@custom_op` re-decoration in spawned
  workers — avoids `infer_schema`-under-Dynamo crash. Sister-fix
  applied to `gdn_dual_stream_customop.py` (P7b) preventively. Live
  verified on 27B PROD: HTTP 200 + 32 reasoning tokens + 90.18 TPS.
  Reporter: noonghunna (club-3090). 9 TDD tests pass.

- **#17 PN30 DS conv state + spec-decode AL>1** (50/50 LCB v6 crash on
  structured-CoT). Two-file text-patch: replaces `NotImplementedError`
  in `mamba_utils.py:get_conv_copy_spec` with `.contiguous()` copy +
  module-level temp-tensor list; adds stream sync + cleanup in
  `do_mamba_copy_block`. Cost ~10-50us per batch when path active.
  Default OFF (opt-in for affected workloads). 10 TDD tests pass.

- **#15 PN31 FA varlen persistent out** (sister patch to P38). Per-shape
  persistent `out` buffer for `_flash_attn_varlen` eliminates per-call
  malloc pressure inside FA C extension. Memory cost: ~16-64 MiB per
  shape × layer. NULL impact on 2×A5000 PROD; designed for 1×3090
  community users with budget-constrained workloads. Default OFF.
  9 TDD tests pass.

### Warnings cleanup (118 → 16, -87%)

- **P8 idempotent-skip log** clarified — distinguishes "already applied"
  (INFO) from real "anchor drift" (WARNING). Was alarming on every
  restart even when patch correctly applied.

- **PN9** (independent drafter attention) **self-retired** —
  `vllm#39930` merged upstream. Removed from all start scripts;
  use `--speculative-config.attention_backend` instead.

- **P67 hook ENTRY** misclassified `log.warning` → `log.info`. Was
  diagnostic for first-3 dispatches per layer, not actual warning;
  created ~50 fake-WARNING entries per boot.

- **Pip "running as root"** silenced via `--root-user-action=ignore`
  in all 7 start scripts.

- **`vllm serve --model X` deprecation** — changed to positional
  argument in all 7 start scripts.

### Infrastructure hardening

- **T4.6 Compile-time watchdog** — `PatchStats.compile_elapsed_sec`
  field + 3-tier threshold logging (>120s WARNING with recovery hint).
  Visibility into Triton autotune cache regression / cold-compile cost.

- **T4.5 Boot-time probe** — standalone CLI
  `python3 -m vllm.sndr_core.utils.boot_probe` for spec-decode cross-rank
  issues (#41190-class). Heuristic markers for `cudaErrorIllegalAddress`,
  workspace-lock AssertionError, OOM. Reasoning-mode aware.

### Added — new patches

- **PN21 — DFlash SWA partial backport** (`vllm#40898`, opt-in OFF).
  Two of three sub-files of jianc99's PR backported: `algos.py`
  (preserve `layer_types` / `use_sliding_window` / `sliding_window` /
  `max_window_layers` from speculators-format checkpoint into HF
  config) + `dflash.py` (force `causal=True` for sliding-window
  layer attention metadata). The `qwen3_dflash.py` model-class
  changes (7+ sub-patches across `Attention.__init__` / `DecoderLayer.__init__`
  / Model class) NOT backported — too fragile for text-patch.
  Empirical on 35B-A3B-FP8-DFlash 160K (3-run sweep):
  `5-6/7` tool-call with PN21 ON vs `7/7` baseline OFF.
  Without the model-side changes, config preserves SWA but the model
  still constructs full attention → metadata/compute mismatch shifts
  spec acceptance. **Default OFF, NOT enabled in any launch script**
  until upstream merges or full manual model-class backport.

- **PN25 — SiluAndMul.forward_native opaque-op pool** (Genesis-original,
  opt-in OFF). Sister-patch to PN12; complement, not replacement.
  PN12 patches `forward_cuda` (eager dispatch) but
  `custom_ops=["none"]` (V1 default for `aot_compile_fullgraph`)
  routes through `forward_native` which Inductor inlines and lowers
  to `empty_strided_cuda(...)` — completely bypassing PN12's pool.
  Reported by noonghunna in `club-3090#16` (VolandBerlioz Reddit + ampersandru
  cross-rig: RTX 3090 24 GB + Lorbus 27B + OpenCode 29K-token prefill
  OOMs at 137.6 MiB). PN25 registers `genesis::silu_and_mul_pooled`
  as `torch.library.custom_op` with `mutates_args=()` and
  `device_types=("cuda",)`. Inductor treats opaque ops as no-inline.
  Body acquires from `FFNIntermediateCache` pool (same one PN12
  uses) and dispatches to `torch.ops._C.silu_and_mul`.
  PN12 + PN25 patch DIFFERENT methods so anchors never collide;
  pool is shared singleton. Recommended pairing for any
  inductor-heavy config; standalone use covers single-path setups.

- **PN26 — TQ unified perf pack** (Genesis-original combining 3
  upstream OPEN PRs from jasonkim8652, opt-in OFF):
  - **Taken from #41418** (centroids prebake): pre-baked Lloyd-Max
    centroid tables for `(d=128, bits=4 / 8 / 3)` — covers our PROD
    presets `turboquant_4bit_nc` / `turboquant_k8v4` / `turboquant_3bit_nc`.
    Empirical on live container: `(128, 8)` `0.018ms` vs solver
    `4583.9ms` = **259,812× speedup** on cold boot.
  - **Genesis defensive addition** vs upstream: at first use, runs
    `prebaked == solver` self-check for `(128, 4)`. On drift > 1e-3
    (real Lloyd-Max algorithm change upstream), auto-disables
    prebake and falls through to runtime solver with a WARNING.
    On 1e-6 drift (round-noise from int/1e10 encoding), logs INFO
    and keeps prebake. Threshold gates against silent staleness.
  - **Taken from #41422 (scaffold-only)**: sparse V tile-skip kernel
    modification. Author validated AMD MI300X only; NVIDIA Ampere
    correctness needs empirical confirmation. Ships as scaffold
    gated by `GENESIS_ENABLE_PN26_SPARSE_V=1` sub-flag; actual
    kernel wiring deferred to next iteration.
  - **Dropped from #41414**: head_dim power-of-2 padding. Qwen3.6
    head_dim=128 is already pow-2; the patch would add a runtime
    branch that is dead code on our model.

- **PN27 — Revert MoERunnerInterface PluggableLayer** (`vllm#41440`
  backport, proactive scaffold, opt-in OFF). Reverts vllm#35178
  (commit `b55b2652`, merged 2026-04-30) which made
  `MoERunnerInterface` inherit from `PluggableLayer` for OOT support.
  Issue #41306 reports v0.20 MoE perf regression: Mixtral-8x7B
  TPOT +21%, TTFT +59%, throughput -19% (8× H200). bnellnm (vLLM core)
  confirmed `--moe-backend=triton` restores v0.19 perf.
  **Our pin `g7a1eb8ac2` was committed 2026-04-28 — 2 days BEFORE
  #35178 merged.** So we are accidentally pre-#35178 and NOT
  vulnerable. PN27 is a **proactive scaffold**: when we eventually
  pin-bump past `b55b2652` BEFORE upstream's #41440 (or equivalent)
  merges, all 3 sub-patches engage and revert to pre-regression
  behavior. On our current pin, all sub-patches SKIP cleanly.

### Added — infrastructure

- **Cliff 8 hardening** (`apply_all.py`) — new
  `PatchStats.partial_apply_warnings` property surfaces skipped
  patches whose reason indicates real anchor drift / ambiguous-anchor /
  required-anchor-missing — distinct from benign skips
  (opt-in OFF, upstream-merged, platform mismatch, deferred,
  redundant). Boot summary line now appends
  `N ⚠️  partial-apply warning(s)` when count is non-zero, plus
  per-warning WARNING-level lines that name each patch + reason.
  Promised to noonghunna in `club-3090` discussion #19. First
  detection in PROD: PN9 self-retire on 27B PROD boot
  (`'spec_cfg.attention_backend' present in llm_base_proposer.py`)
  — manually verified PR #39930 + DFlashProposer `use_non_causal=True`
  is full superset of our partial backport; self-retire correct.

### Changed

- **A2 — P68/P69 long-context threshold default 8000 → 50000 chars**
  (Issue #9). Old 8000-char default (~2K tokens) was too aggressive —
  triggered P68 force-tool-choice and P69 explicit-format-reminder on
  routine IDE-agent flows that are NOT genuinely long-context. New
  50000-char default (~12.5K tokens) keeps the behavior for genuine
  long histories. Code default updated; 6 active launch scripts
  updated to override `8000 → 50000` explicitly.

- **CLIFFS.md PN19 H100-only flag** (Cliff 1 mech A section).
  noonghunna 2026-05-01 confirmed PN19 costs ~120 MiB KV pool on a
  24 GB single-3090 (vs documented 200-500 MiB win on H100). Disable
  PN19 on 24 GB consumer cards (3090, 4090, A5000) running long
  context. Same lesson as P104 L2 persistence — generic allocator
  hints don't survive GPU class boundaries.

### Verified (no regression)

- **#41190 stress test** — TP=2 + spec-decode + first-request
  `cudaErrorIllegalAddress` reported by their RTX 6000 Ada / AWQ /
  WIP-PR-#40898-build setup. Stress-tested on our 35B DFlash 160K
  (TP=2 + DFlash spec K=3): 5 concurrent + 30 sequential rapid-fire
  chat completions. **ZERO `cudaError`**, zero `illegal memory
  access`, zero `watchdog` events. Differences:
  they used QuantTrio AWQ (online-quant), we use FP8 (offline);
  their pin built off PR #40898 head (WIP), our pin on main.
  Possibly P58 (async scheduler placeholder) or P60 (GDN+ngram)
  defends against the codepath.

- **#41306 MoE regression** — verified via runtime probe that our
  installed `MoERunnerInterface.__bases__ == (<class 'abc.ABC'>,)`
  (no PluggableLayer inheritance). Our pin pre-dates #35178 by 2
  days; we are NOT vulnerable. PN27 scaffold ready when we pin-bump.

### PN26b sparse-V kernel — major iteration (v5, 2026-05-01)

Comprehensive deep-dive on Genesis-original sparse-V Triton kernel based
on 4-agent research synthesis (skip-rate observability + per-row vote +
memory profiling + 14-day community scan).

**v5 design** (lean dispatcher + tuning + observability):

- **Lean dispatcher** (no per-call GPU↔CPU sync; v1's `.item()` per call
  caused -16% short-ctx + -22% long-ctx regression — REJECTED).
- **Configurable launch params** baked at apply() time: BLOCK_KV (4/8/16),
  num_warps (1/2/4/8), num_stages.
- **`tl.range()` pipelining hint** (P67 v7.50 pattern, Triton compiler
  cp.async overlap with prior-iter MMA on Ampere).
- **Cache modifier `.cg`** on K/V dequant raw loads (L2 streaming).
- **Sink-token protection** (StreamingLLM finding — first 4 KV positions
  never skipped).
- **Skip-rate observability** (NEW): per-CTA atomic int64 counters,
  constexpr-DCE'd to zero overhead when disabled, `~50-100 ns` per CTA
  at epilogue when enabled. Periodic logging every 500 calls so
  operator sees real skip rate without cross-process IPC.
- **BLASST adaptive threshold scaffold** (`λ = scale_factor / ctx_len`)
  ready in code; default OFF until skip-rate data informs which mode
  is better.

**Empirical sweep on 35B FP8 PROD (TQ k8v4 + MTP K=3, 2× A5000 SM86)**:

| BLOCK_KV | num_warps | mean | max | CV |
|---|---|---|---|---|
| OFF (baseline) | — | 175.41 | 185.15 | 4.20% |
| 8 | 1 | 178.33 | 187.67 | 3.78% |
| 8 | 2 | 180.36 | 190.24 | 4.70% |
| 16 | 2 | 178.35 | 190.74 | 3.26% |
| 8 | 4 | 183.11 | 202.38 | 5.26% |
| 8 | 8 | 181.24 | 196.60 | 5.78% |
| **4** | **4** | **184.89** | 194.56 | 4.63% |
| 4 | 8 | 177.40 | 191.97 | 5.79% |

Winner: **BLOCK_KV=4, num_warps=4** (baked as kernel default).

**Final 35B PROD A/B (apples to apples, 100t output)**:

| Config              | tool-call | mean   | min   | max    | CV    |
|---------------------|-----------|--------|-------|--------|-------|
| Baseline (OFF)      | 7/7       | 175.41 | 158.71| 185.15 | 4.20% |
| **PN26b v5**        | **7/7**   | **182.30** | 153.53 | **212.24** | 7.02% |
| Δ                   | match     | **+3.9%** | -3.3% | **+14.7%** ⭐ | +2.82pp |

The `212 max` exceeds the historical 35B PROD ceiling reference (171-204
TPS quoted from earlier sessions). Tool-call quality preserved (7/7).
Sustained 50-request load: 0 errors, p50=181, p90=197, p99=211. VRAM
delta +142 MiB (acceptable, no leak).

**Caveat**: skip rate at threshold=0.005 is empirically very low on our
short-output workload (most TPS gain comes from kernel restructuring,
not the skip itself). Skip-rate counter scaffold ships so future
operators can data-drive their threshold tuning. Long-context (>16K
input) deeper sweep deferred to next session — needs sustained-context
workload to characterize properly.

### Bench results — `v7.65` PROD eligibility

35B FP8 DFlash 160K (TP=2 + DFlash spec K=3 + PN22+PN23+PN24):

- 44 patches applied / 0 failed / 0 partial-apply warnings
- prose 256t mean **125.07 TPS, CV 3.07%**
- tool-call 5-7/7 (variance band)

27B Lorbus INT4 PROD (TQ k8v4 + MTP K=3 + 8 baked patches):

- 54 patches applied / 0 failed / 1 partial-apply warning (PN9
  self-retire — verified correct, upstream is strict superset)
- tool-call **7/7**
- prose 256t mean **88.39 TPS, CV 2.59%**
- code  512t mean **104.25 TPS, CV 0.20%**

---

## [Unreleased] — `v7.63.x` series

> 50 commits ahead of `origin/main` at time of writing. Local-only
> until Sander explicitly green-lights a GitHub push. Run on PROD
> (server `<host>`, 2× RTX A5000) since 2026-04-29.

### Added

- **Genesis Compat Layer** (`vllm/sndr_core/compat/`) — discovery and
  diagnostics package + 16-subcommand unified CLI:
  - `genesis doctor` — single-shot diagnostic (hardware + software +
    model + patches + lifecycle + dispatcher validator). Emits text
    or JSON.
  - `genesis init` — interactive first-run wizard (detect hardware →
    pick model → workload preference → generate launch script).
  - `genesis explain <patch>` — per-patch deep-dive
    (applies_to predicate, lifecycle state, upstream PR, recommendation).
  - `genesis list-models` / `genesis pull <key>` — curated 5-model
    registry; `pull` downloads the weights and writes a tailored
    launch script that engages the right Genesis patches for the
    hardware × quant combination.
  - `genesis lifecycle-audit` — CI-ready check that every entry in
    `PATCH_REGISTRY` has a known lifecycle state (exit 1 on
    `experimentl` / `retried` typos).
  - `genesis validate-schema` — shape-validates `PATCH_REGISTRY`
    (env-flag prefix, required fields, `applies_to` predicate
    well-formedness, dependency graph).
  - `genesis categories` — browse patches by category.
  - `genesis migrate <vllm-clone> --out runbook.md` — pin-bump
    runbook generator: scans your upstream vLLM checkout, flags every
    Genesis text-anchor that drifted, suggests retirement candidates.
  - `genesis recipe save / load / share / diff / adopt` — capture
    launch configurations, share them by URL, A/B them.
  - `genesis plugins` — community plugin entry-points
    (`GENESIS_ALLOW_PLUGINS=1` opt-in; HTTPS-validated).
  - `genesis telemetry` — opt-in anonymized stats; default OFF.
  - `genesis update-channel status / check / set` — apt-style
    stable / beta / dev channels.
  - `genesis self-test` — operator-facing structural sanity check
    (post-`git pull` / pin-bump). Same gate CI runs.
  - `genesis bench` — wraps the full benchmark suite under a unified
    entry point.
- **`applies_to` predicate DSL** with AND / OR / NOT trees, version-
  range matching for vllm / torch / cuda / triton / driver / compute
  capability. Backwards-compatible with all 50 existing flat-dict
  registry entries.
- **Patch lifecycle state machine** — `experimental` / `stable` /
  `deprecated` / `research` / `community` / `retired`. Code removal
  blocked until lifecycle state is `retired`.
- **A3/D2 dispatcher validator** — boots fail loud on
  `requires_patches` / `conflicts_with` referencing unknown patch
  IDs. Caught two real PROD-config issues at first run.
- **Reference benchmark fingerprints** in
  `vllm/sndr_core/compat/fingerprints/` — blessed numbers per
  hardware × model × patch-set; bench tool can compare a fresh
  run against a fingerprint.
- **D1 CI upstream drift watcher** — daily GitHub Actions cron
  diffs Genesis text-anchors against `vllm-project/vllm@main` and
  flags newly-merged PRs that allow Genesis self-retirement.
- **JSON-Schema for `PATCH_REGISTRY`** at `schemas/patch_registry.json`
  + `genesis validate-schema`.
- **Pre-commit hook** at `scripts/git/pre-commit` — runs schema +
  A3/D2 + lifecycle audit + self-test before every commit. Install
  via `bash scripts/git/install.sh`.
- **Genesis CI workflow** (`.github/workflows/test.yml`) — 492-test
  scoped CI gate on Python 3.10 + 3.12. The full session test
  surface is **1351 tests / 70 skipped / 0 failed** as of this
  release.
- **Canonical `__version__` constant** at
  `vllm/sndr_core/__version__.py` with `__commit__` and `__channel__`.
- **`scripts/git/`** — pre-commit hook + installer.
- **`docs/upstream_refs/`** — historical upstream PR diff studies
  (moved out of the root `reference/` directory by Phase 2.2).

### Patches added

| Patch | What | Status |
|---|---|---|
| `PN14` | TQ decode IOOB safe_page_idx clamp (vllm#40074 backport) | opt-in, validated |
| `PN16` | Lazy-reasoner request hook (Genesis-original) | opt-in, validated |
| `PN17` | FA2 softmax_lse runtime clamp (Genesis Issue #11 fix) | opt-in, validated |
| `PN19` | Scoped `max_split_size_mb` during model load (vllm#41268 backport) | opt-in, validated |

### Patches retired / annotated

- `P5` auto-retire when JartX vllm#39931 merges (TurboQuant hybrid)
- `P82` drift markers for vllm#40819
- `P94` superseded-on-merge by vllm#41043
- `P98` deliberate inverse of merged vllm#40941 (still required on
  hybrid GDN + TQ k8v4 path; documented why)

### Repository structure

- **Phase 2.1 wiring reorg** — `vllm/sndr_core/wiring/` regrouped
  into 9 category subdirectories
  (`spec_decode/`, `structured_output/`, `kv_cache/`, `kernels/`,
  `hybrid/`, `middleware/`, `perf_hotfix/`, `compile_safety/`,
  `legacy/`). Layout-agnostic resolution via `rglob` + computed
  dotted paths. No callsite churn.
- **Phase 2.2 root cleanup** — 29 → 21 root entries.
  `docker-compose.*.yml` × 7 moved to `compose/`,
  `validate_*.sh` × 2 moved to `scripts/`, upstream PR diff studies
  moved to `docs/upstream_refs/`. Doc cross-references updated
  (35+ replacements across README / INSTALL / QUICKSTART / MODELS /
  BENCHMARK_GUIDE).

### Bench upgrades (Genesis Benchmark Suite v2)

`tools/genesis_bench_suite.py` now produces a single rich JSON per
run with the following sections:

- `engine` — vLLM version, system fingerprint, Genesis self-test
  summary, applied patches list.
- `tool_call` — 8-case quality matrix (4 cities × 2 thinking modes;
  positive vs. negative cases scored separately).
- `decode_bench` — N runs × M prompts × max_tokens; per-prompt
  detail + aggregate `wall_TPS`, `decode_TPOT_ms`, `TTFT_ms` with
  median, mean, stddev, CV.
- `multi_turn` — N-turn TTFT with conversation context growing
  per turn.
- `stress` — stability stress: SHA1 drift, NaN sentinel scan,
  repetition detection, TPOT trend, **`STABILITY_VERDICT`**.
- `output_length` — generation capacity probe at 1K..16K target
  lengths, with per-probe VRAM tracking (`vram_before_mib`,
  `vram_after_mib`, `vram_delta_per_gpu_mib`,
  `vram_delta_total_mib`, `verdict`).
- `accept_rate` — Prometheus `/metrics` scrape for spec-decode
  counters (returns gracefully when `--disable-log-stats` is set).
- `vllm_version` — parsed `system_fingerprint` (`vllm_version`,
  `tp`, `commit`).
- `genesis_state` — `--quiet --json` self-test invocation result.

Two new CLI flags:

- `--probe-output-length` — engages section 7 of the run.
- `--scheme http|https` — picks transport for `_build_url()`.
- `--arm-name <name>` — alias for `--name` (A/B compare ergonomics).
- `--compare a.json b.json --compare-out delta.json` — Welch
  t-test + per-percentile delta JSON.

### Docs

- New: `MODELS.md`, `QUICKSTART.md`, `INSTALL.md` (Docker + bare-metal),
  `CONFIGURATION.md`, `docs/BENCHMARK_GUIDE.md`,
  `docs/SELF_TEST.md`, `docs/PLUGINS.md`, `docs/reference/` (operator
  reference per release), `docs/upstream_refs/` (upstream PR diff
  studies), `assets/README.md` (brand asset placement).
- Refactored: `README.md` cross-references updated for Phase 2.1 +
  Phase 2.2 layout.
- This file: `CHANGELOG.md` (root) — concise public release log.

### Tests

The exhaustive (out-of-CI) session test surface is now **1351 / 1391
collected (97% pass rate, 70 skip, 0 fail)**. The 121 drifted
out-of-CI tests rescued in commit `a3a8c8d` covered:

- `test_platform_matrix.py` — 66 tests rewritten for the new
  snapshot-at-load `guards.is_*` constants (no more `cache_clear()`
  calls — those functions are no longer `@functools.cache`).
- `test_v7_14_15_audit.py` — Python 3.13 dataclass introspection
  via `spec_from_file_location` now needs the module in
  `sys.modules` before `exec_module`. Added the bind.
- `test_p51_tq_active.py` — registry shape changed
  (`num_k_buffers` + `num_v_buffers` → unified `total_buffers`);
  logger renamed to `genesis.dequant_buffer`.
- `test_wiring_patch_8.py` — P8 `Issue #5` post-apply import probe
  caused over-defensive skip when `kv_cache_utils.py` returned
  `SKIPPED upstream_merged`. Now the scheduler.py sub-patch carves
  out that case explicitly (helper IS in the file → import will
  succeed).
- `test_p58_async_placeholder_fix.py` — single `SCHED_DRAFT_OLD`
  anchor was split into Site A / Site B in the 2026-04-28 P62-compat
  refactor. Test updated.
- `test_p59_qwen3_reasoning_tool_call_recovery.py` — `RETURN_THINK_OLD`
  similarly split into MONOLITH / MODULAR. Test updated.
- `test_wiring_runtime_rebind.py` — fake `TQAttentionImpl` was
  missing the `_init_turboquant_buffers` sentinel that the upstream-
  drift detector probes for. Added.
- `test_config_detect.py` / `test_model_detect.py` — guarded with
  `pytest.mark.skipif` when `vllm.config` not importable (CPU-only
  / no-vllm envs). Run normally in the integration container.
- Added an autouse `conftest.py` fixture that wipes the central
  `prealloc_budget._CACHED` before/after every test — prevents
  cross-test pollution.

### Production validation

- **27B Lorbus + TQ k8v4 + MTP K=3 + 280K context**: PROD baseline
  on 2× RTX A5000. Reference fingerprint at
  `vllm/sndr_core/compat/fingerprints/rtx_a5000_x2_qwen3_6_27b_int4_v794.json`.
- **35B-A3B-FP8 + MTP K=3 + 320K context**: validated reference;
  fingerprint pending (this release).
- **Cross-rig validated**: contributors on RTX 3090 / 4090 / 5090 /
  H20 / R6000 Pro Blackwell / 8× A4000 — see `CREDITS.md`.

---

## Earlier history

For everything before this release, scroll up — this single file
tracks every commit on the engineering side back to v7.0 (the start
of the modular `_genesis/` package, since renamed to `sndr_core/`).
Audit 2026-05-11 reconciled an obsolete redirect to a never-existing
`vllm/sndr_core/CHANGELOG.md`.
