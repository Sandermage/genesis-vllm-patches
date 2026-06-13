# 2026-06-11 — Pin-bump dev259 → dev491: anchor-validated, promotion BLOCKED on stream tool-calls

## Target
- FROM: vllm 0.22.1rc1.dev259+g303916e93 (current PROD, 2026-06-08)
- TO:   vllm 0.22.1rc1.dev491+g1033ffac2 (candidate, 232 commits newer)
  digest sha256:779772129ce2cbd64329e370aed9dd8f27ffea9b8eb69038e9a2d5ee5791202d

## Status: ANCHOR-VALIDATED + BOOTS CLEAN, but PROMOTION DEFERRED

### What passed
1. **Preflight diagnostic** (pin_preflight vs dev491 pristine): of all 307
   patches, only **8 DRIFT_ANCHOR + 1 binding-fail** drifted across 232
   commits. PN351's proactive dual-anchor (batch-3) showed
   EXPECTED_ALTERNATE — no work needed. The predicted #45171 harmony
   landmine manifested as P107 drift.
2. **Fix-loop** (commit 82d17174): 9 patches dual-anchored
   (P58/P62/P87/P88/P89/P107/PN378/PN380 re-anchored vs BOTH pristine
   trees; G4_07 binding repointed). Re-sweep: dev491 DRIFT_ANCHOR 8→0,
   dev259 DRIFT_ANCHOR stays 0 (PROD anchors intact). lint-drift 0 on
   both trees, 516 tests green.
3. **Smoke boot** (with the full 137-var PROD env-file): dev491 boots
   **99-111 applied / 0 FAILED** — the dual-anchored patches apply
   cleanly on dev491. Non-stream tool-calls WORK (finish=tool_calls,
   get_weather extracted). No NameError (P107 re-anchor good).

### The blocker — streaming tool-calls regressed on dev491
With the SAME launcher (--tool-call-parser qwen3_xml --reasoning-parser
qwen3) that makes dev259 stream tool-calls work, dev491 returns the
tool XML as `delta.content` with `finish_reason=stop` and ZERO
`delta.tool_calls` — the parse_delta dead-zone class fixed on dev259 is
BACK on dev491. This is an upstream behavior change in the 232-commit
window (the studied PRs #45389/#45310/#45479/#45464 all touched the
streaming tool-parser / DelegatingParser path). Tool-calls are the
critical agent hot path, and streaming is the live path → promotion
BLOCKED until adapted.

### Rollback (clean)
PROD restored on dev259 (health 200, stream tool-calls verified working
= 3 delta.tool_calls). dev259 container kept throughout; nightly-303916e93
image preserved. Two PROD-down cycles total (~10 min each): the first
exposed a launcher/YAML env drift (137 -e vars set at docker run, absent
from the launcher — fold them in for reproducibility); the second
exposed the stream-tool-call regression.

## Next (to complete the promotion)
1. **Investigate the dev491 parse_delta streaming regression**: diff
   dev259 vs dev491 `parser/abstract_parser.py` (DelegatingParser.
   parse_delta / _in_tool_call_phase) + `tool_parsers/` + the
   chat_completion streaming generator. Find WHICH of the 232 commits
   changed the reasoning_ended / tool-phase gating, and adapt (likely a
   new Genesis patch or extending PN386). The studied #45389 (PN386,
   already vendored) + #45310 + #45479 are the prime suspects.
2. Re-smoke with the fix; if stream + non-stream tools both green +
   bench within CV of dev259 baseline (250/250/217.6 TPS) → promote:
   YAML pins/EXPECTED_PINS/ALLOWED_MODELDEF_PINS/anchor-manifest →
   dev491; fold the 137 env vars into the launcher; retire/version-cap
   P87+PN378+P26 (upstream-merged on dev491); tag rotation
   (nightly-303916e93 → previous, delete 626fa9bb per max-2 policy).
3. The fix-loop dual-anchors mean PROD can stay on dev259 indefinitely
   with zero risk while the stream-tools fix is developed — the bump is
   ready except this one runtime gap.

## Pin-bump system verdict
232 upstream commits → 9 patches needed re-anchoring (caught in minutes
by pin_preflight), and the only promotion blocker is a genuine runtime
behavior change that NO static tool could have caught — exactly the
gap the smoke-boot leg exists to find. The "painless pin transition"
goal is substantially met: the drift surface was mapped and fixed
automatically; only the runtime adaptation remains.

---

## Update (2026-06-14, post-PN392 server validation attempt)

### Root cause CONFIRMED (deep-diff dev259 vs dev491 pristine)
vllm#45171-era refactor **deleted `tool_parsers/qwen3xml_tool_parser.py`**
and **remapped `qwen3_xml` → `Qwen3CoderToolParser`** in
`tool_parsers/__init__.py`. The coder parser is single-emission
(emits ≤1 structural delta per call, returns to advance assuming
token-by-token feeding); the dev491 unified streaming path feeds the
WHOLE `<tool_call>…</tool_call>` XML as one delta at the reasoning→tool
boundary → the parser flips its start-flag and returns emitting ZERO
`delta.tool_calls`. Verified the re-anchored P107/P89/PN288 are NOT
implicated, and `parse_delta`/`_in_tool_call_phase`/the qwen3 reasoning
parser are byte-identical between pins.

### PN392 fix (commit a3b84468) — server validation INCONCLUSIVE
PN392 (runtime wrap of `extract_tool_calls_streaming` on both
Qwen3Coder + Qwen3XML classes, draining the single-emission core to
coalesce deltas) passes 11 TDD tests + all repo gates (registry 308).
But the dev491+PN392=1 smoke-boot STILL showed the streaming tool-call
returning the raw XML as `delta.content` with `finish_reason=stop` and
0 `delta.tool_calls`. Non-stream tool-calls + reasoning split WORK on
dev491.

### Open questions for the next focused (live, INFO-logged) iteration
1. **Did PN392 actually apply?** The PROD env sets
   `VLLM_LOGGING_LEVEL=WARNING`, which MASKS the INFO-level
   `applied: PN392` line — so the empty grep is inconclusive. PN287
   (same `applies_to.tool_call_parser` gate) applies on dev259, so the
   gate is not the blocker. Re-smoke with `VLLM_LOGGING_LEVEL=INFO` to
   confirm PN392's apply + whether its class-wrap took effect on the
   live parser instance.
2. **If PN392 applied but the symptom persists**, the failure layer is
   DEEPER than the single-emission drain: the content shows the FULL
   XML leaking as content, suggesting the streaming generator may not
   be routing to `extract_tool_calls_streaming` at all (the
   reasoning→tool phase transition, or the coder parser's
   buffer-until-complete behavior swallowing the whole delta). Trace
   `chat_completion_stream_generator` on dev491 with the wrap active.
3. **PN374** (qwen3xml quoted-key) targets the now-deleted
   `qwen3xml_tool_parser.py` → dormant on dev491; re-target to
   `qwen3coder_tool_parser.py` in the same adaptation pass.

### Pin-bump state
ANCHOR-VALIDATED (both pins DRIFT=0, fix-loop 82d17174) + BOOTS CLEAN
(99/0 failed). PROD stays on dev259 at zero risk (dual-anchors). The
ONLY remaining promotion blocker is the streaming-tool-call fix, which
needs ONE live INFO-logged iteration to either confirm PN392 works or
locate the deeper streaming-dispatch layer. NOT a rollback — the
adaptation is 95% done; this is the last 5%.
