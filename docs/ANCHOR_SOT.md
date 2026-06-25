# Anchor Source-of-Truth (SOT) + Pin-Bump Tooling — Operator Manual

**Audience**: operators bumping the vLLM pin, or auditing whether the
committed patch matrix still matches what the engine actually ships.

This is the operator how-to for the **Phase-4 anchor source-of-truth**
system: the per-pin manifests under `sndr/engines/vllm/pins/<pin>/`, the
`make rebuild-pin` / `audit-pin` / `bump-preflight` / `summarize-rej`
targets, and the `scripts/anchor_sot/` tooling behind them. It is the
companion to [`PIN_BUMP_PLAYBOOK.md`](PIN_BUMP_PLAYBOOK.md) (the canonical
end-to-end bump procedure) — read the playbook for the full bump flow; read
this for the anchor-SOT regen + drift-gate mechanics.

> Current pin: `0.23.1rc1.dev424+g3f5a1e173` (`dev301` =
> `0.23.1rc1.dev301+g04c2a8dea` = rollback). The dev301 → dev424 bump
> (2026-06-25) was the first to dogfood this tooling end-to-end.

---

## 1. What a per-pin anchor SOT is

Genesis patches apply by matching **anchors** — small, unique text regions
in upstream vLLM source. When upstream refactors a file, an anchor can
move, change, or vanish; the patch then silently stops applying (a benign
"skip"), and a performance optimization can quietly become a no-op. The
anchor SOT exists to make that class **loud**, not silent.

For each supported pin we commit a directory:

```
sndr/engines/vllm/pins/<version>_<short-sha>/
├── anchors.json      # the per-file, per-patch anchor manifest (ground truth)
└── drift.rej.json    # the reject/triage report for that pin
```

Current on-disk pins:

| Dir | Pin | Role |
| --- | --- | --- |
| `0.23.1_3f5a1e173` | `0.23.1rc1.dev424+g3f5a1e173` | **current** |
| `0.23.1_04c2a8dea` | `0.23.1rc1.dev301+g04c2a8dea` | **previous / rollback** |
| `0.23.1_b4c80ec0f` | `0.23.1rc1.dev148+gb4c80ec0f` | dropped (kept for diff history) |
| `0.22.1_da1daf40b` | `0.22.1…` | older |
| `0.21.1_626fa9bba` | `0.21.1…` | older |

### `anchors.json`

A dict with `manifest_version`, `generated_at`, `generated_by`, `pins`
(the engine version this manifest describes), and `files` — keyed by the
upstream source path, each carrying the per-patch anchors that resolved in
that file. It is the **pristine** picture: generated from the bare image's
source tree, never from a running PROD container's read-write layer (which
carries stale markers from older repo states and would mask anchor rot).

### `drift.rej.json`

The triage report for the pin. Top-level fields:

- **`counts`** — anchor outcomes by class, e.g.
  `{"ok": 144, "anchor_drift": 4, "optional_absent": 12, "retired": 30,
  "upstream_merged": 1, "version_gated": 13}`.
- **`coverage`** — `{"discovered": N, "ok": N, "rejected": N}`.
- **`merge_status`** — per-patch upstream merge tri-state
  (`merged` / `open` / `unknown`), the input to the iron-rule-#11 retire
  decision.
- **`genuine_anchor_drift`** — the patches whose required anchor is
  genuinely gone on this pin (the rows that need a re-derived anchor; this
  is the list that must be **empty or explained** before promotion).
- **`dependency_breakage`** — `edges` between a retired/skipped patch and
  its still-active **dependents** (the silent-breakage class — see §4).

---

## 2. Makefile targets (the operator surface)

| Target | What it does |
| --- | --- |
| `make rebuild-pin SSH_HOST=<user@host>` | Regenerate the per-pin SOT **on the rig** and pull it back. Syncs `scripts/anchor_sot/` + `sndr/` to the rig, runs the 2-step regen (running-container discovery + bare-image pristine source), writes `sndr/engines/vllm/pins/<pin>/anchors.json` + `drift.rej.json`, rsyncs them back. Review + commit the result. |
| `make audit-pin SSH_HOST=<user@host>` | Verify the **committed** manifest still matches a fresh rig regen (R2 drift gate). Regenerates from the live engine and diffs vs committed, ignoring timestamps. Use it to catch a pin that drifted under you without a deliberate bump. |
| `make summarize-rej [PIN=<dir>]` | Human-readable summary of `drift.rej.json`: counts by status, the merge tri-state, and retire-broken dependents. `PIN=<dir>` for one pin; omit for all. |
| `make bump-preflight OLD=<old_pin_dir> NEW=<new_pin_dir>` | The **bump gate**. Diffs the OLD vs NEW manifests and exits non-zero if any HIGH-severity perf dependent broke on the new pin. See §4. |

`CONTAINER` and `IMAGE` default to the running 35B preset / `:nightly`;
override them to target a different container or candidate image. The
regen never touches PROD beyond a read-only discovery pass.

The underlying scripts live in `scripts/anchor_sot/`
(`rebuild_pin.sh`, `audit_pin.sh`, `build_manifest.py`, `compare_manifest.py`,
`bump_preflight.py`, `summarize_rej.py`, `discover.py`, `pristine_dump.py`).

---

## 3. The pin-bump procedure, end-to-end

The full validate-before-promote flow lives in
[`PIN_BUMP_PLAYBOOK.md`](PIN_BUMP_PLAYBOOK.md). The anchor-SOT steps slot
into it as follows:

1. **Operator authorizes the bump**, naming the target pin (e.g. "bump to
   the dev424 nightly"). Per the pin policy, **never** chase a newer
   upstream build proactively — no `docker pull` without an explicit
   instruction.
2. **Candidate image must already be on the rig.** If it is absent, the
   extractor exits and prints the exact `docker pull` command for the
   operator to run deliberately.
3. **Regenerate the new pin's SOT**: `make rebuild-pin SSH_HOST=…`. This
   writes `sndr/engines/vllm/pins/<new-pin>/`.
4. **Read the triage**: `make summarize-rej PIN=<new-pin-dir>`. Resolve
   every `genuine_anchor_drift` row (re-derive the anchor from the pristine
   candidate source — playbook §3). Triage every `upstream_merged` /
   `merge_status: merged` row with an iron-rule-#11 deep diff (playbook §4)
   — never retire on a PR title alone.
5. **Run the bump gate**: `make bump-preflight OLD=<old-pin-dir>
   NEW=<new-pin-dir>`. It must exit 0 (or you must clear every flagged perf
   dependent with a canonical A/B — see §4).
6. **Boot smoke on a throwaway container** + tokenizer-fingerprint gate +
   canonical bench (playbook §5/§5b/§6). Never on PROD.
7. **Promote** (playbook §7): add the pin to `KNOWN_GOOD_VLLM_PINS`
   (`sndr/engines/vllm/detection/guards.py`), pair-update `EXPECTED_PINS`
   in `tests/unit/dispatcher/test_pin_gate.py` (`make test-pin-gate`), bump
   `vllm_pin_required` in the model YAMLs + README badge + CHANGELOG, and
   bump the validated patches' `applies_to` upper bounds (with boot-log
   proof — no blanket bumps).
8. **Tag rotation** (playbook §8): re-tag `:nightly` to the new pin, keep
   the previous pin's hash tag for rollback during the validation window,
   then delete the oldest so the server holds **at most current +
   previous** (the ≤2-pin policy).

---

## 4. What `bump-preflight` detects (the silent-regression class)

`bump-preflight` exists because the **dev148 → dev301 bump lacked it** and
a real regression slipped through clean. The post-mortem: PN353A retired,
PN399's anchor went missing, PN399 was recorded as a *benign skip*, its
decode-scratch optimization silently no-op'd, and a **-5.5% TPS regression**
hit the 35B with `genuine_drift = 0` (i.e. the naive drift check looked
clean). The gate makes that loud. It reports, between the OLD and NEW pin
manifests:

- **(a) Newly retired / version-gated-out patches** on the new pin (vs the
  old pin).
- **(b) Their broken dependents** — the retire-impact detector edges
  (`dependency_breakage.edges`), each carrying the dependent's id,
  category, lifecycle, default-on flag, and a `detail` string, ranked by
  severity.
- **(c) Perf-landmines** — PERF-tier patches that moved `ok → skip/drift`
  between the two pins (a perf optimization that silently went dead).
- **(d) A reminder** that any perf-tier delta requires a canonical A/B
  (`tools/genesis_bench_suite.py`, iron rule #9) before the bump is
  trusted.

**Exit codes**: `0` = no HIGH-severity perf dependent broken (clean /
advisory); `1` = ≥1 HIGH-severity perf dependent broken — run a canonical
A/B before bumping; `2` = usage / unreadable input.

The same edge data is surfaced live in the GUI (Patches → retire-impact /
Anchors tab) and via `sndr/engines/vllm/retire_impact.py`.

---

## 5. Worked example — the dev424 SOT

`make summarize-rej PIN=0.23.1_3f5a1e173` on the current pin reports
(`drift.rej.json`):

- `coverage`: 204 discovered, 144 ok, 60 rejected.
- `counts`: `ok 144`, `anchor_drift 4`, `optional_absent 12`,
  `retired 30`, `upstream_merged 1`, `version_gated 13`.
- `genuine_anchor_drift`: 4 entries — each was re-derived from the pristine
  dev424 source and confirmed before promotion.
- `dependency_breakage.edges`: e.g. `PN399` (stability, experimental,
  default-off) flagged as a dependent to verify.

Because every `anchor_drift` row was either re-anchored or confirmed as an
intentional retirement, and `bump-preflight OLD=…04c2a8dea NEW=…3f5a1e173`
exited clean, the dev424 pin was promoted with decode carried forward from
the validated dev148 baseline (no regression).

---

## 6. Routine drift audit (between bumps)

You do not need a bump to use this system. To confirm the live engine still
matches the committed SOT (catches an undeclared pin drift):

```bash
make audit-pin SSH_HOST=<user@host>
```

A non-empty diff means the running engine no longer matches the committed
manifest — investigate before it silently disables a `default_on` patch.

---

## See also

- [`PIN_BUMP_PLAYBOOK.md`](PIN_BUMP_PLAYBOOK.md) — **canonical** end-to-end
  pin-bump procedure (preflight → fix-drifts → deep-diff → smoke →
  tokenizer gate → bench → promotion → tag rotation).
- [`guides/PIN_UPGRADE.md`](guides/PIN_UPGRADE.md) — short operator-policy
  summary + universal launcher template; points back here and to the
  playbook.
- `sndr/engines/vllm/detection/guards.py` — `KNOWN_GOOD_VLLM_PINS`.
- `tests/unit/dispatcher/test_pin_gate.py` — `EXPECTED_PINS` drift trap.
- `tests/unit/anchor_sot/` — the anchor-SOT + retire-impact test suite.
