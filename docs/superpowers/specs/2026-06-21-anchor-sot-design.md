# Per-Pin Anchor Source-of-Truth + Consolidation + Auto-Detect — Design

**Date:** 2026-06-21 · **Status:** approved (operator), pending spec review
**Author context:** Genesis vLLM overlay (~316 patches, ~180 anchor-bearing modules) on pin dev148 (`0.23.1rc1.dev148+gb4c80ec0f`), 2×A5000 rig `sander@192.168.1.10`.

## 1. Goal

On a vLLM pin bump, re-anchoring drifts across up to ~180 patch modules by hand. Make a **single per-pin source-of-truth file** the authoritative store of every patch's anchor address, so a bump becomes: regenerate automatically for patches that still match, and hand-edit **only the genuinely-drifted anchors, in one file**. Plus: **consolidate** duplicate/near-duplicate patches to shrink the anchor surface, and wire the existing **auto-detect** of model + vLLM version + hardware into the manifest selection.

### 1.1 Operator hard-requirements (first-class acceptance gates)
- **R1 — 100% coverage.** The system must enumerate and verify **ALL** patches and **ALL** target files. No hand-typed subset; no sampling. Discovery covers every anchor-bearing module (~180) and every vLLM file they touch.
- **R2 — true drift.** Drift detection must see the **real** difference between our anchor and the **live** upstream source — actual byte/structural change, not a heuristic guess. A patch is classified `ok` / `anchor_drift` / `upstream_merged` / `version_gated` from the real pristine engine source, never assumed.
- **R3 — server-tested to 100%.** Every phase is validated **on the rig** against the live pin: the full round-trip (build manifest → boot → apply-via-manifest → md5 verify → model serves correctly) must pass at 100%, with no regression vs the current `applied=89 / failed=0` baseline and no TPS/quality loss.

## 2. Background — what already exists vs the gap

A 6-agent code study (2026-06-21) established the system is **~70% pre-built in disconnected pieces**; the work is **inversion + wiring**, not greenfield.

**Already built (reuse/extend, do not rebuild):**
- **Apply engine** — `sndr/engines/vllm/wiring/anchor_manifest.py` (`compute_anchor_meta`: `byte_offset` + `byte_length` + `anchor_md5` + `replacement_md5` + `md5_pristine`; `assemble` / `validate_manifest_schema` / `verify_manifest_against_source`) + Layer-4.5 `_try_apply_via_manifest` (`sndr/kernel/text_patch.py:164`, called at `:420`): splices by `byte_offset` in reverse order with 7 abstain gates + md5 verify, falling back to exact-match Layer 5 (`:524`). The position-stable byte_offset+md5 design (cites rust-analyzer / Triton #2597) is correct.
- **Full discovery** — `tools/check_upstream_drift.py` enumerates every `PatchSpec` via `dispatcher.spec.iter_patch_specs()`, builds each patcher (`_build_patcher_for_module` / `_make_patcher`), scans `sub_patches[].anchor` against a **pristine clone** with full runtime parity (Layer-2 marker idempotency, `upstream_drift_markers`, version-gate), and classifies each patch. It already covers all 180 — it just never persists the result.
- **Per-pin dir convention** — `sndr/engines/vllm/pins/0.21.1_626fa9bba/` and `0.22.1_da1daf40b/` (one dir per pin) + `_normalize_pin` (`adapter.py:157`, `base.py:226`) mapping `0.22.1rc1.dev195+gda1daf40b → 0.22.1_da1daf40b`.
- **Auto-detect** — version (`detection/guards.py: get_vllm_full_version_string` + 60-entry `KNOWN_GOOD_VLLM_PINS`); model (`model_detect.get_model_profile()`: MoE/hybrid/quant/model_class probed from the loaded checkpoint, per-process cached); hardware (`gpu_arch_profile._detect()`).
- **Consolidation playbook** — 4 consolidated modules already shipped (e.g. `p64_p61c_pn56_qwen3coder`), with documented engine invariants. Two highest-value drift reductions are **staged but not cut over**: `pn79_v2_md5_chunk*.py` (replaces the **108-anchor** `pn79_inplace_ssm_state.py`) and `pn118_v2_md5_*.py` (17 anchors).
- **Fragility scorer** — `scripts/audit_anchor_fragility.py` (AST-walks anchor constants, scores by line count).

**The real gap:**
- **G1 — Inversion.** Anchor TEXT lives inline in 180 modules; both "manifests" are derived. The per-pin file must become **authoritative** (carry offset+md5+anchor-text+replacement-ref); inline anchor demoted to bootstrap/fallback.
- **G2 — Coverage 2%→100%.** Apply-wired JSON covers 4 patches; builder uses a hand-typed `_REGISTRY_TARGETS`/`_KNOWN_REL_PATHS`. Must auto-enumerate all 180 — the discovery code already exists in `check_upstream_drift.py`; refactor it into a shared module both the builder and drift-checker import. **(directly satisfies R1)**
- **G3 — Per-pin storage.** Good JSON schema lives at one `default_manifest_path()`; `load_manifest_for_pins` invalidates the whole manifest on pin mismatch. Relocate the schema into `pins/<pin>/anchors.json`; pin-mismatch → select a different file, not disable-for-everyone.
- **G4 — Resolver unwired to boot.** `_normalize_pin` yields the dir name but is never called from `kernel/manifest.py`; `is_pin_supported()` hardcodes `True`, `list_supported_pins()` returns `()`. ~5-line completions.
- **G5 — No build→audit→apply pipeline.** The tools are disconnected siblings. Need the glue (one `make rebuild-pin`).
- **G6 — No drift-surface metric / consolidation gate.** `audit_anchor_fragility` scores anchor size but never tallies anchors-per-target-file joined with `default_on`/`lifecycle`.

## 3. Locked decisions (operator)
1. **Authority model = authoritative + fallback.** Manifest is the primary anchor source; inline anchor stays in the module as bootstrap/fallback. Incremental, safe, reversible.
2. **Format = JSON** `pins/<pin>/anchors.json` (reuse the proven `anchor_manifest.py` schema + engine as-is; machine-generated).
3. **First win = consolidation md5 cutovers first** (`pn79_v2`, then `pn118_v2` after prod A/B) — removes the largest drift contributor before it enters the manifest.
4. **Fuzzy/libCST relocator = later (YAGNI).** Ship exact-match (byte+md5) SoT first; add offline fuzzy relocation only if the first real bump produces too many manual re-anchors.

## 4. Architecture — 6 components (each one responsibility)

1. **`pins/<pin>/anchors.json`** — per-pin authoritative store. Per entry keyed by `patch_id` + sub-patch name: `{ target_rel, byte_offset, byte_length, anchor_md5, replacement_md5, md5_pristine, anchor_text, replacement_ref }`. `anchor_text` retained for bootstrap/fallback (decision #1). Coarse per-file md5 (from the existing YAML idea) kept for fast whole-file drift screening.
2. **`anchor_discovery.py`** (new shared module) — extract the enumerator from `check_upstream_drift.py` (`iter_patch_specs → _build_patcher_for_module → patcher.target_file / sub_patches`). One function: *given a (pristine) source tree, yield every (patch_id, sub, target_rel, anchor, replacement) for all anchor-bearing patches.* Imported by both the generator and the drift-checker. **Deletes the hardcoded `_REGISTRY_TARGETS`/`_KNOWN_REL_PATHS`.** Satisfies R1 (100% coverage).
3. **Generator** (`build_anchor_manifest.py`, rewritten) — runs `anchor_discovery` against a **pristine vLLM clone of the target pin**, computes `compute_anchor_meta` for every anchor, classifies each `{ok, anchor_drift, upstream_merged, version_gated}` from the **real** pristine source (R2), and writes `pins/<pin>/anchors.json` for the `ok` set + a `.rej` report for the `anchor_drift` set.
4. **Boot resolver** (`kernel/manifest.py`, wired) — at apply time: `_normalize_pin(guards.get_vllm_full_version_string())` → load `pins/<pin>/anchors.json`; complete `is_pin_supported` (dir exists) + `list_supported_pins` (glob `pins/`); pin-mismatch selects a different file (G3). Auto-detect (component 6) feeds it.
5. **Bump pipeline** (`make rebuild-pin`) — the "edit one file" realization: discovery + classify + auto-regenerate `ok`, emit `.rej` for `anchor_drift`; operator re-anchors only the drift set **in the one new pin file**; `make audit-pin` validates the result before promotion. `upstream_merged_markers` stay the mechanism, so the per-pin file only carries anchors for patches **not** absorbed upstream.
6. **Auto-detect glue** (existing, wired) — version + model (`model_detect.get_model_profile`) + hardware (`gpu_arch_profile`) feed component 4's manifest selection and the dispatcher's existing `applies_to` gating. No operator declaration of model/hw.

**Parallel workstream 0 — consolidation** (shrinks the surface components 1-5 must track): cut over the staged md5 migrations + fold file-coincident clusters; `audit_anchor_fragility.py` extended with an anchors-per-target-file × `default_on`/`lifecycle` metric + a "safe-to-merge" gate.

## 5. Data flow

- **Boot/apply:** detect pin → `is_pin_supported`? → load `pins/<pin>/anchors.json` → Layer-4.5 splice by `byte_offset`+md5 (7 abstain gates) → on md5 miss, fall back to inline exact-match (Layer 5). Result identical to today's `applied=N / failed=0`, faster (offset splice, no per-boot scan).
- **Bump:** new pin → `make rebuild-pin` → pristine clone → `anchor_discovery` (R1) → per-anchor classify from real source (R2) → write `pins/<new_pin>/anchors.json` (ok) + `.rej` (drift) → operator re-anchors the drift set in one file → `make audit-pin` → rig validation (R3) → promote.

## 6. Phasing + per-phase acceptance (all gated on R3 server-test)

- **Ф0 — Consolidation-first.**
  - Cut over `pn79_v2_md5` (108 anchors → ~4 md5 sentinels; `default_on=False` → low risk). Retire the splice original after parity test.
  - Defer `pn118_v2_md5` (17 anchors) until a **prod A/B** (it is `default_on=True`).
  - Extend `audit_anchor_fragility.py` with the anchors-per-file × default_on metric + safe-to-merge gate.
  - **Accept:** local — pytest + `patches doctor` green; anchor count drops by ~104. Rig (R3) — boot 35B with pn79_v2 active (if applicable) OR confirm pn79 stays default-OFF + 27B (its target) A/B if enabled; `applied`/`failed=0` unchanged, no TPS loss.
- **Ф1 — Discovery module.** Extract `anchor_discovery.py`; `check_upstream_drift.py` + the generator both import it. **Accept:** `anchor_discovery` enumerates **all 180** anchor-bearing patches (R1) — a test asserts count == `iter_patch_specs` anchor-bearing count; drift-checker output unchanged.
- **Ф2 — Storage + generator.** Generator writes `pins/<pin>/anchors.json` for all `ok` patches against the dev148 pristine clone; classification from real source (R2). **Accept:** the generated dev148 manifest covers 100% of `ok` patches; a `.rej` lists any drift; round-trip test (build → apply-via-manifest → md5 verify) passes for every entry.
- **Ф3 — Runtime wiring.** Boot resolver loads `pins/<pin>/anchors.json`; per-pin select + fallback; auto-detect glue. **Accept (R3, the big one):** boot the live 35B on the rig with the manifest path active → **every** manifest-covered patch applies via Layer 4.5 (md5-verified), the rest fall back cleanly, `applied=89 / failed=0` preserved, 7×6→correct, TPS within noise of 216 baseline. 100% or it does not ship.
- **Ф4 — Bump pipeline.** `make rebuild-pin` + `make audit-pin` + `.rej` report; CI wiring. **Accept:** a dry-run "bump" against a second pin (e.g. the resident `:nightly` if it differs, or a synthetic) produces a correct ok/drift split with zero false-`ok` (R2).

## 7. Testing / verification strategy
- **TDD:** each phase test-first. Core round-trip test: `build manifest from pristine → apply via manifest → assert replacement_md5 matches → assert byte-identical to the inline-anchor result`.
- **R1 test:** `len(anchor_discovery(tree)) == count of anchor-bearing patches in iter_patch_specs()` — fails if any patch is missed.
- **R2 test:** inject a synthetic upstream drift (mutate a pristine file), assert the classifier reports `anchor_drift` for exactly the affected patch and `ok` for the rest — no false `ok`.
- **R3 (server):** every phase boots the live model on the rig and proves `applied=N / failed=0` + correct generation + TPS parity. The Ф3 gate is the full 100% apply-via-manifest proof.
- Existing gates stay green throughout: `patches doctor`, the dispatcher baseline snapshots, `apply.shadow`.

## 8. Safety / invariants
- Consolidation honors the two engine invariants (qwen3coder docstring): **one TextPatcher per absorbed patch** (failure isolation) + **distinct marker per absorbed patch** (no Layer-2 idempotency cross-shadow).
- Consolidation touches only `default_on=False`/experimental; `default_on=True` prod patches (P28/P46/P7/P29, pn118) are excluded from merges and require prod A/B.
- Authoritative+fallback (decision #1): inline anchors stay as bootstrap → any manifest miss falls back to today's behavior. The change can never make a patch silently not apply — md5 verify + fallback guarantee it.
- The per-pin file carries anchors **only** for patches whose content upstream did **not** merge (`upstream_merged_markers`).

## 9. Out of scope (this spec)
- Fuzzy/libCST structural relocator (decision #4 — later, only if needed).
- Switching the runtime engine to Comby/ast-grep (rejected — external binary on the pinned container, slower 180-patch startup).
- A full GUI for the manifest.

## 10. Critical files
```
sndr/kernel/text_patch.py                         # Layer 4.5 apply engine (extend select-by-pin)
sndr/engines/vllm/wiring/anchor_manifest.py       # schema + compute_anchor_meta (reuse)
sndr/engines/vllm/anchor_discovery.py             # NEW shared enumerator (Ф1)
sndr/engines/vllm/pins/<pin>/anchors.json         # NEW per-pin SoT (Ф2)
sndr/engines/vllm/kernel/manifest.py (or equiv)   # boot resolver wiring (Ф3)
scripts/build_anchor_manifest.py                  # rewritten generator (Ф2)
tools/check_upstream_drift.py                      # imports anchor_discovery (Ф1)
scripts/audit_anchor_fragility.py                 # + anchors-per-file metric (Ф0)
sndr/.../patches/.../pn79_v2_md5_chunk*.py         # cut over (Ф0)
Makefile                                          # rebuild-pin / audit-pin targets (Ф4)
```
