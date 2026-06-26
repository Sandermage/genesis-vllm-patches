# Genesis quality-gate harness — methodology

The Genesis quality gate is a public, runnable quality standard for any
Genesis-served OpenAI-compatible endpoint. It exists to answer two questions
that a plain "does it boot?" smoke test cannot:

1. **Where is the wall?** Boundary probes push the KV-cache and
   prefill-activation paths until something gives, and report the *real*
   fillable ceiling — not the advertised `--max-model-len`.
2. **Did the patch actually help?** A green run on a patched config does not by
   itself prove the patches are load-bearing — topology alone (e.g. TP=2) can
   take a failure mode off the table. The gate makes that distinction explicit
   and, for the soak, measurable.

It targets the Genesis bug classes directly — the named cliffs in
[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) ("Named cliffs") — and every failing
probe points at the responsible Genesis patch ID, not a generic "check the
logs".

## Credit and provenance

This harness is an **extended port of club-3090's public test suite**
([github.com/noonghunna/club-3090](https://github.com/noonghunna/club-3090),
MIT-licensed). We adapted, and built on top of, three of their scripts:

| club-3090 source | What we adapted |
|---|---|
| `scripts/verify-stress.sh` | The 8-probe boundary ladder: NIAH needle rungs from ~10K up to ~0.92×n_ctx with per-rung VRAM-margin capture and false-ceiling detection; the ~25K-token tool-prefill OOM probe; the IDE-agent / multi-turn / LCB / reasoning shapes; the "defer Cliff-2 territory to last so an OOM doesn't cascade" probe ordering. |
| `scripts/soak-test.sh --continuous` + `scripts/soak-helper.py` | The continuous (ramping-context) multi-turn soak that reaches ~22-25K accumulated tokens by turn 5 — the workload shape that surfaces multi-turn VRAM accretion — and the soak verdict (silent-empty discriminator, TPS retention, VRAM-growth threshold). |
| `scripts/bench.sh` | The bench methodology we already follow (3 warmup + 5 measured, fixed prompts, temperature 0.6, narrative + code prompt pairing) — see [`BENCHMARKS.md`](BENCHMARKS.md). |

Their **"PASS ≠ patches load-bearing"** discipline (club-3090 issue #140) is
adopted directly and is the single most important idea we took from them.
noonghunna's broader testing and bug-isolation methodology is also credited in
[`CREDITS.md`](CREDITS.md).

## What we changed for the Genesis stack

The upstream harness is excellent at finding *that* something broke. The Genesis
port adds *why, where, and which patch*:

- **Cliff-ID + patch-ID attribution.** Every failure signature is
  cross-referenced to the Genesis cliff taxonomy AND the owning patch. A 500 on
  the LCB probe does not say "check logs" — it says "Cliff 2 / 2a (GDN fwd_h),
  owned by **P103**, lower `--gpu-memory-utilization` or route to a TP=2 preset".
  The full table lives in `tools/quality_gate/probes.py::CLIFF_MAP`; the cliffs
  themselves are documented in [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).
- **Tested core, thin shell.** The part that is easy to get subtly wrong — probe
  shapes, recall checking, ladder construction, and the verdict thresholds —
  lives in `tools/quality_gate/` (`probes.py`, `soak.py`, `runner.py`) and is
  pinned by `tests/unit/quality_gate/`. The bash drivers
  (`scripts/verify_stress.sh`, `scripts/soak_continuous.sh`) are thin
  orchestrators. A full live run needs the GPU rig; the request-generation and
  verdict logic do not, and that is exactly what the unit tests cover.
- **Executable patch attribution.** `scripts/soak_continuous.sh --strip-overlays`
  runs the same soak twice — overlays ON, then OFF — and diffs the two verdicts
  into a `LOAD_BEARING` / `TOPOLOGY_SIDESTEP` / `NOT_LOAD_BEARING` result that
  names the patch under test. This turns the club-3090 #140 *discipline* into a
  *command*.
- **Preset-aware.** Drivers accept `PRESET=<key>` and resolve the endpoint and
  served-model name from `sndr launch --dry-run` (best-effort), so the gate runs
  against exactly what an operator launched.

## The probes and the cliffs they target

`scripts/verify_stress.sh` runs eight probes in order. Cliff-2 territory is
deferred to the end so an OOM there does not cascade-fail the cheaper probes.

| # | Probe | Shape | Targets |
|---|---|---|---|
| 1 | NIAH small rungs | needle at ~50% depth, scale 150 / 450 (~10K / 30K) | mid-context attention quality below the cliffs |
| 2 | Tool-response prefill OOM | multi-turn with a ~25K-token mock tool message + tool def + `tool_choice=auto` | **Cliff 1** (FA2 softmax_lse / FFN activation peak) — **PN17** |
| 3 | IDE-agent one-shot | ~5K-char Cline/OpenCode preamble + 10 tool schemas, `tool_choice=none` to force the reasoning path | **Cliff 1 mech B** (inductor FFN intermediate leak) |
| 4 | Multi-turn agent | sys + tools + 4-turn history (assistant tool_call → tool reply → follow-up) | a different inductor compile path than probe 3 |
| 5 | LCB-coding shape | LeetCode-style problem + structured plan, `max_tokens=4096` | DS conv-state crash class (Cliff 3-adjacent) |
| 6 | Reasoning-heavy | math proof, `max_tokens=8192` | spec-decode acceptance-length collapse / mamba cache-mode interactions |
| 7 | NIAH large rungs | scale 900 / 1400 (~60K / 90K) | **Cliff 2 / 2a** (GDN `chunk_gated_delta_rule_fwd_h` OOM) — **P103** |
| 8 | Context ceiling ladder | staggered NIAH from ~95K up to 0.92×n_ctx, per-rung VRAM capture | the **false-ceiling** class — config advertises N but fills << N |

### NIAH ladder + the false-ceiling detector (probe 8)

Each rung places a fresh random needle (`<colour> <animal> <number>`) at ~50%
depth and asks for it back. The ladder is calibrated against the **live
tokenizer** with a small probe (filler-scale → tokens), not a hardcoded
chars/token guess, and steps up in `CEILING_STEP_TOKENS` increments to
`CEILING_FRACTION × n_ctx`. Each rung captures free VRAM before/after, so the
result is a **margin curve**, not a single pass/fail.

The ladder **stops at the first failing rung — that depth is the real ceiling.**
A config that boots at 262K, pre-reserves its KV pool, and passes a fixed-depth
90K needle can still wall at ~125K under a scaled fill (the flash-attention
transient scratch at high fill scales with *populated* context, not with what
was reserved at boot). The fixed-depth needle never reaches the wall; the scaled
ladder does. A thin VRAM margin at the deepest passing rung is itself a failure —
sustained agent load carries prompt-cache + checkpoint overhead the single-shot
NIAH does not exercise.

### Verdict semantics for the NIAH rungs

- **PASS** — HTTP 200 and the needle was recalled.
- **WARN** (`△`) — HTTP 200 but recall missed. This is an *attention-quality*
  ceiling, **not** a system fault: the engine filled the context and answered.
  The ladder records it and moves on.
- **SKIP** (`⊘`) — HTTP 400 because the depth exceeds `--max-model-len` (a clean
  engine rejection). On the ceiling ladder a 400 is disambiguated: if the rung
  *target* was below `n_ctx`, that is a harness sizing bug (FAIL), not a skip.
- **FAIL** (`✗`) — HTTP 500 / timeout / crash, carrying the Genesis cliff + patch.

## Cliff → patch map

The full machine-readable map is `tools/quality_gate/probes.py::CLIFF_MAP`. In
summary:

| Failure signature | Cliff | Patch | First move |
|---|---|---|---|
| FA2 softmax_lse over-allocation at long ctx | Cliff 1 | PN17 | enable PN17; on 24 GB consumer Ampere also disable PN19 |
| GDN `fwd_h` `(B,NT,H,V,K)` blow-up on a single >50K prompt | Cliff 2 / 2a | P103 | enable P103 + `GENESIS_FLA_FWD_H_MAX_T=16384`, mem-util ~0.85, or route to TP=2 |
| GDN multi-turn VRAM accretion after ~4-5 ramping turns | Cliff 2b | PN59 | enable PN59 streaming-GDN + allocator hardening; TP=2 sidesteps |
| TurboQuant + spec-verify K+1 + FULL cudagraph tool-call cascade | Cliff 3 / 4 | P67 | confirm P67 compiles (GQA=6 needs the non-pow-2 generalisation) |
| HTTP 200 + zero completion tokens (silent-empty) | silent-empty | P67 / PN30 | xgrammar mask / spec-decode empty-draft / `<think>` exhaustion |
| No HTTP response (timeout / OOM-killed) | engine-down | — | inspect logs, restart, re-run |

See [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) "Named cliffs" for the full
mechanism, impact, and fix of each.

## The soak — Cliff 2b and the "PASS ≠ load-bearing" discipline

`scripts/soak_continuous.sh` runs the **only** probe that surfaces Cliff 2b: GDN
multi-turn VRAM accretion under an accumulating-context agentic conversation
(the hermes / openhands / Cline shape). Each session is **one** ramping
multi-turn coding conversation — growing synthetic tool results push accumulated
context to ~22-25K tokens by turn 5. Fresh / reset-each-turn fixtures do **not**
trigger this class, which is why continuous mode exists.

### Soak verdict

PASS means *no failure signal on the test sample*:

- request errors / stream interruptions: 0
- silent-empty turns (HTTP 200 + 0 completion tokens): < 50%
- max VRAM growth from the **warm** baseline (after session 1, so prefix-cache
  fill is not mistaken for accretion): < `SOAK_MAX_GROWTH_MIB` (default 200 MiB)
- decode-TPS retention (first sessions vs last): ≥ 80%

The silent-empty discriminator is **genuine `completion_tokens == 0`**, not the
`decode_tps == 0` proxy — the proxy false-flags fast tool-call turns that
*did* produce output.

### What PASS does NOT mean

A clean soak proves the configuration is **stable end-to-end at this depth**. It
does **not** prove that the overlay patches in the config are doing the work:

- **Topology can sidestep the failure mode.** Cliff 2b mitigations target
  single-card 24 GB pressure. A **TP=2** preset shards the GDN state across cards
  and structurally escapes Cliff 2b *regardless of which patches load*. A PASS on
  a dual preset says nothing about whether PN59 is load-bearing there.
- **The workload may not be deep enough.** Continuous mode ramps to ~22-25K
  tokens; deeper-context regimes can still fail.

This is the club-3090 #140 trap. To attribute a patch, you must compare against a
run with the overlay removed.

### `--strip-overlays` — attribution as a command

`scripts/soak_continuous.sh --strip-overlays` runs the soak twice and diffs the
verdicts:

1. **overlays ON** (the config as launched) → `on` verdict.
2. **overlays OFF** — the operator relaunches the *same* config with Genesis
   disabled (`GENESIS_ENABLE=0`), so the soak measures the engine *without* the
   patches under test → `stripped` verdict.
3. **attribute** (`tools/quality_gate/soak.py::attribution_delta`):

   | ON | STRIPPED | topology | verdict |
   |---|---|---|---|
   | PASS | **FAIL** | any | **LOAD_BEARING** — the patch did the work |
   | PASS | PASS | TP ≥ 2 | **TOPOLOGY_SIDESTEP** — TP=2 took the failure off the table; re-run on TP=1 to attribute |
   | PASS | PASS | TP = 1 | **NOT_LOAD_BEARING** for this workload — ramp deeper before clearing the patch |
   | FAIL | any | any | **INCONCLUSIVE** — get the ON config green first |

This is exactly the rigor club-3090 applies and asks contributors to apply: prove
the patch is load-bearing, do not assume it from a green run.

## Running it

Both drivers auto-detect the endpoint and served model, accept a `PRESET`, and
support `--help` and `--dry-run`. `--dry-run` exercises the full harness wiring —
ladder construction, payload generation, and verdict logic with cliff/patch
attribution — **without sending a single request**, so the gate is verifiable on
a laptop. A full live run needs the GPU rig.

```bash
# Boundary / stress gate against a running config:
scripts/verify_stress.sh
URL=http://localhost:8000 MODEL=qwen3.6-27b scripts/verify_stress.sh
PRESET=<your-preset> scripts/verify_stress.sh
scripts/verify_stress.sh --dry-run         # plan + wiring check, no requests

# Cliff-2b soak:
scripts/soak_continuous.sh                  # single soak (PASS proves stability only)
scripts/soak_continuous.sh --strip-overlays # ON-vs-OFF patch attribution
scripts/soak_continuous.sh --dry-run        # plan + wiring check, no requests
```

Useful environment overrides are documented in each script's `--help`
(`SKIP_LONGCTX`, `CEILING_FRACTION`, `CEILING_STEP_TOKENS`, `VRAM_MARGIN_MB`,
`SOAK_MAX_GROWTH_MIB`, `ATTR_PATCH`, `ATTR_TP`, …). The defaults target
Qwen3.6-27B on a single 3090; override for other VRAM classes.

## Files

- `scripts/verify_stress.sh` — boundary / stress driver (8 probes).
- `scripts/soak_continuous.sh` — Cliff-2b soak + `--strip-overlays` attribution.
- `tools/quality_gate/probes.py` — probe payloads, ladder, recall, NIAH/HTTP
  verdicts, and the cliff→patch `CLIFF_MAP`.
- `tools/quality_gate/soak.py` — continuous-ramp fixtures, soak verdict, and the
  attribution delta.
- `tools/quality_gate/runner.py` — the JSON CLI the bash drivers call.
- `tests/unit/quality_gate/` — unit tests for the probe generation and verdict
  logic (run with `python3 -m pytest tests/unit/quality_gate/`).

## Testing status

The probe-generation and verdict logic are unit-tested and run in CI without a
GPU. A full live boundary + soak run requires the rig; the harness is built so
that everything *except* the live HTTP send is verifiable offline, and
`--dry-run` confirms the wiring end-to-end before you spend rig time.
