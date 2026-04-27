# Upstream Posting Log — 2026-04-27 (Round 10 community engagement)

Sander reviewed and approved 4 drafts for posting on 2026-04-27. All 4 posted successfully via `gh` CLI. PR #40914 promoted from draft → Ready-for-Review.

## Posts

### 1. ✅ vllm-project/vllm#38898 (NickLucche, OPEN, was 24-day silent)

- **Comment:** https://github.com/vllm-project/vllm/issues/38898#issuecomment-4327928957
- **Source draft:** `04_vllm_38898_align_spec_decode_RU+EN.md`
- **Strategic value:** HIGHEST LEVERAGE — closed 24-day silence with our exact reproducer (5K identical: hit_tokens=2768 without MTP, 0 with MTP K=3). NickLucche filed the kernel-level NotImplementedError; we provided the missing real-world data point.
- **Watch for:** maintainer engagement within 1-2 weeks. If silent, escalate by tagging @njhill or @WoosukKwon.

### 2. ✅ vllm-project/vllm#40914 (Sander's K+1 spec-verify routing PR)

- **Comment:** https://github.com/vllm-project/vllm/pull/40914#issuecomment-4327933886
- **Source draft:** `06_vllm_40914_ack_cross_rig_RU+EN.md`
- **Action:** also promoted PR from draft → Ready-for-Review via `gh pr ready 40914`
- **Strategic value:** two-rig validation table (A5000 + 3090, 27B + 35B-A3B variants) + tagged @LucasWilkinson @WoosukKwon for spec-decode-side review
- **Watch for:** CODEOWNER review responses

### 3. ✅ noonghunna/qwen36-dual-3090#1 (cross-rig collab thread)

- **Comment:** https://github.com/noonghunna/qwen36-dual-3090/issues/1#issuecomment-4327942956
- **Source draft:** `05_noonghunna_dual3090_issue1_v2_RU+EN.md`
- **Strategic value:** Acknowledged noonghunna's apples-to-apples 35B-A3B bench (wall_TPS 136.87, CV 2.2%), side-by-side table vs A5000, asked him to test P82 cross-rig
- **Watch for:** noonghunna's P82 cross-rig bench results

### 4. ✅ noonghunna/qwen36-27b-single-3090#2 (v7.13 → v7.14 migration / compat)

- **Comment:** https://github.com/noonghunna/qwen36-27b-single-3090/issues/2#issuecomment-4327948337
- **Source draft:** `07_noonghunna_single3090_issue2_compat_RU+EN.md` (adapted to current thread state — noonghunna had already provided detailed migration guide)
- **Strategic value:** Confirmed migration is on Genesis side, promised compat shim within 24h, also offered diagnostic help to @darknotevil's OOM follow-up
- **Watch for:** users adopting the compat shim once shipped

## Implementations following from posts

### `patch_genesis_unified.py` shim (NEW)

Shipped at Genesis repo root per commitment in post #4 above. Thin wrapper invokes `vllm._genesis.patches.apply_all.main()` with a deprecation warning. Allows existing downstream compose files (noonghunna repos, tedivm, etc.) to keep working without modification while users migrate to the modern invocation pattern.

## Open follow-ups (next 1-2 weeks)

1. **vllm#38898 maintainer response** — if silent, escalate
2. **vllm#40914 reviewer response** — CODEOWNER feedback expected
3. **noonghunna P82 cross-rig bench** — pending his time
4. **darknotevil OOM follow-up** — wait for his launch command + log paste, then diagnose
5. **PR #26807 upstream merge** — watch for merge → bump pin → adopt GDN all mode automatically
6. **vllm v0.20.0** — re-evaluate pin bump in 1 week after community CUDA 13 shakeout

## Memory rules followed

- ✅ All drafts had English + Russian versions per `feedback_github_comment_style`
- ✅ Ukraine + AI translation disclaimer included in each post
- ✅ All posts received explicit Sander approval BEFORE posting (per `feedback_no_push_without_explicit_approval`)
- ✅ No AI co-author attribution in posts (per `feedback_no_ai_credit_in_public`)
- ✅ Genesis repo URLs are pushed (verified before posting per `feedback_verify_links_before_posting`)
- ✅ Genesis "Sander voice" tone: warm, conversational, factual, no corporate stiffness
