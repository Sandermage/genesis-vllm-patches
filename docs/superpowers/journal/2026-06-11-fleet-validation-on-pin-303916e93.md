# 2026-06-11 — Fleet validation on pin 303916e93 (boot / speed / stability / tool-calls)

Mandate: implement the preflight-triage residuals and validate EVERY
fleet model on the current pin — launch, speed, stability, no
regression on tool-calls and other response modes.

Instruments: `sndr/extras/tools/genesis_bench_suite.py --quick`
(canonical; includes 7-case tool-call battery, multi-turn TTFT,
n=25 decode bench) + `tools/genesis_chat_matrix_bench.py` (9 chat
variants). Suite-to-suite comparisons only (iron rule #9).

## Fleet results

| Model | Boot (applied/failed) | Suite wall_TPS | Tool-calls | Notes |
|---|---|---:|---|---|
| **Qwen3.6-27B** int4 TQ k8v4 + MTP K=3 | 109 / 0 | **120.9** (CV 2.7%) vs 119.5 baseline 06-03 → +1.2%, no regression | **7/7** (was: NO PARSER AT ALL) | TPOT 7.6ms (thinking_off), reasoning split per YAML, 256K ctx |
| **Gemma-4-26B-A4B** AWQ + MTP K=3 | 57-62 / 0 — FIRST boot on this pin | 112.8-117.4 (CV 26-28% = answer-length artifact) | **7/7** (gemma4 parser, native in pin) | TPOT 6.0ms ≈165 TPS decode-class; TTFT 73ms CV 5% |
| **Gemma-4-31B** AWQ dense | 56 / 0 (degraded profile) | **78.4** (kv auto, 32K ctx) | **7/7** | TQ profile BLOCKED on pin (below); degraded is 3× FASTER than old-pin TQ baseline (25.8 TPS / TPOT 33.6ms) |
| **Qwen3.6-35B-A3B** FP8 (PROD) | 105 / 0 restored | per-variant matrix re-probed post-restore | qwen3_xml + PN287 v2 | **PN353A applied for the first time ever** after marker fix |

## 27B config drift found and fixed (Class-1, severe)

The "breakthrough" container from 2026-06-10 ran with **7 of 141**
prescribed env keys and NO tool parser:

- YAML prescribes 94 patch flags + 47 hardware system_env keys,
  `tool_call_parser: qwen3_coder`, `enable_auto_tool_choice`,
  `reasoning_parser: qwen3`, PN119='1'. The hand-built launcher had
  none of that. The YAML even prescribed `shm_size: 8g` — the
  container had 64MB and crashed NCCL the moment the full env enabled
  `NCCL_P2P_DISABLE=1` (fixed via ipc=host, matching the 35B pattern).
- Launcher regenerated FROM configs (hardware system_env + model
  patches block). Full stack boots 0-failed; suite shows no speed
  regression vs the canonical baseline; tools went 0 → 7/7.
- P18B-on-PN119 chain CONFIRMED live: on the first full-stack boot
  P18B failed in the APIServer pass (anchors not yet created), then
  applied in the EngineCore pass after PN119 wrote the GQA/MHA
  branches. Two-pass convergence; topo-sort by requires_patches is
  the proper fix (SNDR_TOPO_SORT_SPECS=1, needs registry sync first).

## Gemma 31B + TurboQuant: BLOCKED on this pin (real pin-bump gap)

`ValueError: Selected backend AttentionBackendEnum.TURBOQUANT is not
valid for this configuration. Reason: ['kv_cache_dtype not supported',
'partial multimodal token full attention not supported']`

- New upstream validity gate (`v1/attention/backend.py
  validate_configuration`): TQ backend lacks `supports_mm_prefix()`
  and the dtype check fails despite `turboquant_4bit_nc` being in the
  backend's ClassVar list (root cause of the dtype reason NOT yet
  established — needs in-container debug of which class/dtype the
  validator actually sees).
- G4_17 / G4_23 no-op on this pin: `Gemma4ForConditionalGeneration` /
  vision-tower classes RENAMED upstream — G4 family needs a class-map
  refresh (same class as the G4_08
  `CompressedTensorsMoEWNA16MarlinMethod not found` warning on 26B).
- Old Gemma containers cannot be reused: they bind-mount pr42637
  overlay files from the RENAMED repo layout (`vllm/sndr_core/...`) —
  starting them would shadow upstream files with empty dirs.
- Mitigation shipped: degraded profile (kv auto, no TQ backend,
  32K ctx) — 78.4 TPS, tools 7/7, 3× the old TQ speed. Cost: context
  65K → 32K. Rollback path for full 65K+TQ: previous pin
  (nightly-626fa9bb images + exited containers kept).
- **G4 migration session needed:** TQ-backend mm_prefix support
  decision (correctness of partial-mm attention over quantized KV),
  dtype validity root-cause, G4_17/G4_23/G4_08 class repoints.

## Server hygiene

- Pin policy restored: deleted dev338, dev371, v0.22.0, da1daf40
  images + 11 stale test containers (~115 GB reclaimed). Server now
  holds exactly current (nightly + nightly-303916e93) and previous
  (nightly-previous + explicit-hash) — per CLAUDE.md policy.
- a5000-2x hardware YAML digest bumped to d892cc… (current pin) —
  promotion-hygiene backfill.
- Repo state bundle-synced to the rig as `refs/local-sync/fleet-0611`;
  runtime-relevant modules (registry, PN353A, P6, chat matrix tool)
  rsynced into the live overlay.

## Instrument fixes

- `genesis_chat_matrix_bench.py`: accept the vllm 0.22.x SSE rename
  `reasoning_content` → `reasoning` (7/9 variants false-failed on the
  first reasoning-parser-enabled model).
- Note for all custom benches: with a reasoning parser active,
  scripts that count only `delta.content` read 0 tokens. The suite
  (usage-based counting) is unaffected.

## Follow-ups (queued, see 2026-06-11-preflight-residual-triage-action-plan.md)

1. Fix-drifts batch: PN32 (w/ PN79 composition), P59, PN58, PN288
   (+P107 defect), PN38 (sub-B delete — SyntaxError hazard).
2. P85 both-sites fix (PN346 interplay) + PN346 default-ON mismatch.
3. Retire queue: P7b, PN54, P78, P36, P83, P84, P4, P20, P6 — each
   with iron-rule-#11 evidence already collected.
4. 122 self-collision TRUE_RISK markers (73 patches) — lint +
   remediation convention.
5. G4_03 repoint (eagle3 module gone) + G4_07 staged repoint.
6. Preflight v1.2: ENV_GATED_ABSTAIN verdict, env-forced builder
   retry, _read_<param> UNBUILDABLE resolution, alternation groups.
7. G4/Gemma migration session for TQ-on-31B (above).
8. 26B/31B launcher YAML round-trip: fold the degraded-31B decision
   and the text-only flags back into the model YAMLs with bench
   receipts.

## Addendum (same session): streaming tool-calls were broken on 35B — found and fixed

The post-restore probe exposed TWO live streaming defects the
non-stream tests never see:

1. **P107 NameError at stream end** (predicted by the triage
   verifier): v2's injected detector referenced `auto_tools_called`,
   which on this pin exists only in the NON-streaming generator.
   Every tool-bearing streamed request errored at stream end. Fixed:
   P107 v3 (clause dropped, comment corrected, live file surgically
   patched + restart) — commit 78e35b78.

2. **Raw-XML passthrough on streamed tool-calls** (Class-11 /
   club-3090 #145 symptom): non-stream parsed perfectly, streams
   returned the tool-call XML as `content` with `finish_reason=stop`
   and ZERO `delta.tool_calls`. Root cause (upstream dead-zone in
   `parser/abstract_parser.py parse_delta`): `_in_tool_call_phase`
   requires `state.reasoning_ended=True`; with a tool parser but NO
   reasoning parser, the flag is set only via the
   `prompt_token_ids is not None` init check — which our streams
   never hit → BOTH phases inactive → "pass through as content".
   The 27B (with `--reasoning-parser qwen3` per YAML) parsed streams
   fine — the reasoning phase machine sets `reasoning_ended` on
   `</think>`.
   **Fix:** added `--reasoning-parser qwen3` to the 35B launcher —
   which the 35B model YAML (line 54) prescribed all along. Same
   Class-1 config-drift family as everything else found today.
   Verified: 5 `delta.tool_calls` chunks, `finish_reason=tool_calls`,
   0 NameError, non-stream unchanged.

Lesson for the playbook: tool-call validation MUST include a
streaming leg — the suite's 7/7 battery and non-stream probes both
passed while streams were broken. Upstream issue candidate: tool-only
parser + absent prompt_token_ids = permanent passthrough (consider
filing vllm issue / Genesis patch for `reasoning_ended=True` default
when `_reasoning_parser is None`).
