# Sprint Report — 2026-04-27 Phase 4 Round 10 (P5 conditional + community drafts + strategic findings)

**Запрос:** Backport PR #26807 (GDN all mode) + open upstream issue + P5 conditional + run tests + re-research vllm/noonghunna repos.

**Результат:** P5 conditional реализован + tested (НЕ был блокером); 4 drafts готовы; PR #26807 backport REJECTED после анализа (high risk, marginal benefit); найден vllm#38898 (наш точный bug, 24 дня silent — highest-leverage place to comment).

---

## Что сделано

### 1. P5 conditional (✅ implemented)

**File:** `vllm/_genesis/wiring/patch_5_page_size.py` (`apply()` function)

Added env-gated guard at apply() entry:
```python
if os.environ.get("GENESIS_DISABLE_P5", "").strip().lower() in ("1", "true", "yes", "on"):
    return "skipped", (
        "GENESIS_DISABLE_P5=1 set — deferring KV page-size unification "
        "to upstream's `_align_hybrid_block_size` (PR #25752/#30877). "
        "Required for compatibility with --mamba-cache-mode={align|all}."
    )
```

Operator sets `GENESIS_DISABLE_P5=1` alongside `--mamba-cache-mode=align/all`.

**Tested via v757** (= v755 align+MTP + GENESIS_DISABLE_P5=1). Result:
- Container booted ✓
- P5 SKIPPED in apply_all logs ✓
- BUT `hit_tokens=0 STILL` for align+MTP combo
- bs=2832 STILL (came from upstream `_align_hybrid_block_size`, not from P5)

**Conclusion:** P5 was NOT the blocker. The block_size=2832 is upstream's behavior for our model. Real blocker remains align+spec-decode incompatibility (PR #30877 disable / Issue #38898).

P5 conditional kept as **safe option** for users who want to defer to upstream's alignment logic when running with mamba prefix cache modes. Documented as opt-in env. No regression.

### 2. PR #26807 backport — REJECTED after deep analysis

**Investigation:**
- Fetched full diff: 1222 lines, 14 files
- Files: 6 are kernel internals (FLA chunk ops, gdn_linear_attn +443 lines, gdn_attn backend +177, causal_conv1d)
- Status: OPEN, dirty (needs rebase since 2026-03-30), CI not green, no maintainer review
- Author already disclaimed spec-decode compat (`--mamba-cache-mode=all` + spec_config = ValueError)

**Cost vs benefit assessment:**

| Aspect | Verdict |
|---|---|
| Backport effort | 1-2 days carefully + thorough P67/TQ regression testing |
| Kernel collision risk with P67 | HIGH — touches FLA + gdn ops where our P67 lives nearby |
| Production benefit | MARGINAL — `align` already gives hits=2768; `all` would add ~5-15% more granularity |
| Spec-decode unlock | NONE — same explicit forbid as align mode |
| Status uncertainty | HIGH — if upstream rebases / restructures, we re-do backport |

**Recommendation:** **DEFER.** Comment on PR #26807 offering to test when merged — low effort, high signal value. Wait for upstream merge → bump pin → no backport needed.

### 3. Upstream community engagement — 4 drafts ready

`docs/upstream/draft_replies/`:

- **`04_vllm_38898_align_spec_decode_RU+EN.md`** ⭐ HIGHEST LEVERAGE
  - Issue #38898 (NickLucche, 2026-04-03, OPEN, **24 days silent, ZERO comments**)
  - Exact NotImplementedError our 9-round investigation hit
  - Our 5K identical reproducer (hit_tokens=0 with MTP, hit_tokens=2768 without) is missing real-world data
  - Comment closes 24-day silence + offers labor

- **`05_noonghunna_dual3090_issue1_v2_RU+EN.md`** HIGH urgency
  - noonghunna posted 35B-A3B bench 2026-04-27 01:00 (wall_TPS 136.87, CV 2.2%)
  - Direct apples-to-apples cross-rig data on Sander's exact model class
  - Reply ack + side-by-side table + ask to test P82 on his rig

- **`06_vllm_40914_ack_cross_rig_RU+EN.md`** HIGH urgency
  - Acknowledge noonghunna's cross-rig validation on Sander's own PR #40914
  - Two-rig table (A5000 + 3090, 27B + 35B-A3B variants both verified)
  - Promote out of draft → tag @LucasWilkinson @WoosukKwon

- **`07_noonghunna_single3090_issue2_compat_RU+EN.md`** MEDIUM urgency
  - radojko asked for backward-compat shim for `patch_genesis_unified.py`
  - Plan: tiny shim file + Migration section in README
  - Cheap goodwill, prevents footgun for v7.13 followers

All drafts include English + Russian + Ukraine/AI disclaimer per `feedback_github_comment_style` memory rule. **NOT POSTED** — awaiting Sander review per `feedback_no_push_without_explicit_approval` rule.

### 4. Fresh upstream / community signals

**vllm#26201 (Mamba2 cache tracking) follow-ups:**
- ✅ #25752 Mamba2 APC merged
- ✅ #26377 Mamba1 APC merged
- ⏳ **#26807 GDN APC `all` mode OPEN** (simondanielsson, dirty)
- ✅ #27339 supports_mamba_prefix_caching Protocol merged
- ✅ #27813 ShortConv PC merged
- ⚠️ **#38898 OPEN** — exact match for our bug, zero engagement
- ✅ PR #40454 "Default to align for spec decode" merged 2026-04-21 (in our pin) — workaround, not real fix
- ⏳ #39463 ngram spec decode hybrid GDN fix — adjacent
- ⏳ #39562 MambaManager assertion fix
- ⏳ #40384 Exclude O(1) Mamba groups

**vllm#40914 (Sander's PR):**
- noonghunna posted cross-rig confirmation 2026-04-26 23:19
- Both 27B and 35B-A3B variants verified at v7.48
- Ready to promote out of draft

**noonghunna momentum (last 48h):**
- 5 commits in single-3090 (v7.14 compose, eager.yml, minimal.yml, eager fix)
- 8 commits in dual-3090 (turbo.yml, dflash.yml, **p67-bench-35b-a3b.yml** ⭐, NVLink prep)
- bench.sh v2 with CV metric **specifically for Sander's PR #40914**
- Community contributors: walmis, radojko, danbedford, ampersandru, 3dluvr — Genesis ecosystem growing

### 5. v0.20.0 pin bump assessment

vLLM v0.20.0 dropped 2026-04-27 09:05 UTC.
- PyTorch 2.11 + CUDA 13 + Transformers v5 — breaking
- DeepSeek V4 + FlashAttention 4 default MLA + TurboQuant 2-bit KV (#38479)
- TQ 2-bit overlaps our P67/P78 surface — needs A/B before adopting

**Recommendation: NO BUMP today.** v748 healthy on dev205. Re-evaluate after community shakes out CUDA 13 / transformers v5 regressions (1 week+).

---

## Production state

**v748 (cache OFF + MTP K=3 + P82 t=0.3) running healthy on prod.** All findings preserved as opt-in research artifacts in code (P83/P84/P85 default OFF + P5 conditional via env).

---

## Files in this delta

- `vllm/_genesis/wiring/patch_5_page_size.py` (modified — P5 conditional)
- `scripts/launch/start_v757_align_mtp_no_p5.sh` (new — test reference)
- `docs/upstream/draft_replies/04_vllm_38898_align_spec_decode_RU+EN.md` (new)
- `docs/upstream/draft_replies/05_noonghunna_dual3090_issue1_v2_RU+EN.md` (new)
- `docs/upstream/draft_replies/06_vllm_40914_ack_cross_rig_RU+EN.md` (new)
- `docs/upstream/draft_replies/07_noonghunna_single3090_issue2_compat_RU+EN.md` (new)
- This sprint report

---

## Recommended next actions for Sander (priority order)

1. **Review + post #04 vllm#38898** — highest leverage, closes 24-day silence, our reproducer is exactly the missing piece
2. **Review + post #06 vllm#40914** — promote out of draft, two-rig validation table ready
3. **Review + post #05 noonghunna/dual-3090#1** — directly responds to his 35B-A3B bench data
4. **Review + post #07 noonghunna/single-3090#2** — implement compat shim if approved
5. **Watch PR #26807** — if merges, test on our rig + bump pin
6. **Wait for #38898 maintainer response** (1-2 weeks); if silent, escalate by tagging @njhill / @WoosukKwon
7. **Test v0.20.0 in 1 week** — after community CUDA 13 shakeout

---

## Honest summary

- **v748 prod stable**, no regression introduced
- **P5 conditional was wrong hypothesis** — not the blocker, but kept as safe option
- **Real blocker: align+spec-decode incompatibility** — upstream-tracked at #38898 + #26807
- **Community engagement is the highest-leverage path** — comment on #38898, ack noonghunna's cross-rig
- **PR #26807 backport rejected** — too risky for marginal benefit; better to wait for upstream merge

**Total round 10 work:** 4 community drafts + 1 conditional patch + 1 test + comprehensive analysis. **Zero regressions, zero pushes.**
