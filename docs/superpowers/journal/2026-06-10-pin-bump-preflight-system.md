# 2026-06-10 — Pin-bump preflight system v1 → v1.1 (enterprise pin transitions)

Mandate: make pin-to-pin transitions painless ("переход на новые пины
был не больным"). Result: a one-command preflight that maps everything
that will break on a candidate pin BEFORE touching PROD, validated
against the current pin with an oracle, and already paying for itself.

## Shipped (commits dbc8ff8d, 31515b15, bd150d81 on sndr-dev)

1. `tools/extract_candidate_tree.sh` — pristine tree from a candidate
   image (docker create/cp/rm, NO pull, no GPU) + PROVENANCE.json with
   the real in-image `vllm.__version__`.
2. `tools/pin_preflight.py` — read-only verdict engine over all patch
   specs against an alternate root (`guards.vllm_install_root` seam,
   `GENESIS_NO_PATCH_CACHE=1`). Verdicts: OK / DRIFT_ANCHOR /
   CHAINED_ANCHOR / AMBIGUOUS_ANCHOR / DRIFT_FILE_MOVED (+moved-to
   grep) / UPSTREAM_MERGED / SUB_UPSTREAM_MERGED / STALE_RESIDUE /
   UNBUILDABLE / IMPORT_FAIL / RUNTIME_BINDING (+4 binding
   sub-verdicts). Pure Layer-5 mirror — never calls `.apply()`.
3. `docs/PIN_BUMP_PLAYBOOK.md` — extract → preflight → fix loop →
   iron-rule-#11 deep-diff → smoke → bench → promotion → tag rotation;
   9 empirical failure classes.
4. `make preflight CANDIDATE_ROOT=… [JSON_OUT=…]`; 60 unit tests (TDD,
   incl. Layer-5 behavior parity vs the real
   `TextPatcher._apply_layer5_legacy`).

## Finding 1 — version-range enforcement is two-tier (34-patch triage)

Validation flagged 34 patches with `<0.22.0` ranges as out-of-range on
the promoted 0.22.1 pin. Live PROD applies P67/P82/P70/P72/PN130
anyway. In-container probe: `check_version_constraints` correctly says
False, but `should_apply` returns True with "opt-in env".

Root cause is DOCUMENTED dispatcher semantics
(`dispatcher/decision.py` `should_apply` rule 1): a truthy env flag on
an opt-in patch overrides `applies_to`, version range included. Only
`default_on=True` patches are strictly gated.

Boot-log triage of all 34 (iron rule #11 — no blanket bumps):

- 26 applied on 0.22.1 (boot-log `applied` / `already applied`) →
  ranges bumped to `<0.23.0`.
- 7 disabled-by-env (PN104 PN105 PN125 PN200 PN202 PN203 PN97) → caps
  kept (honest last-validated record).
- 1 upstream-merged PN90 → cap kept (intentional retirement gate,
  double-defended by its drift marker).
- All 34 are `default_on=False` → nothing was silently disabled.

Corollary now in the playbook: **range-capping is NOT a retirement
mechanism** while launchers still export the env flag — real
retirement = `lifecycle: retired` + flag removal from launch configs.

Preflight now reports `out_of_range_detail` split STRICT_SKIP vs
ENV_OVERRIDE_POSSIBLE.

## Finding 2 — P18B "drift" was a patch chain, not drift

P18B_TEXT read as DRIFT_ANCHOR on pristine (every anchor absent) while
live says idempotent. The initial hypothesis (rw-layer-masked latent
drift) was WRONG: PN119's bundled diff CREATES the 12-space GQA/MHA
launcher blocks P18B anchors on. Pristine = single 8-space launcher.
"Fixing" P18B's anchors to pristine would have broken the chain —
classic iron-rule-#11 save by reading before changing.

PN119's md5 gate verified healthy: pristine 0.22.1 md5 equals
`PN119_PRE_PATCH_MD5` (the constant's comment said dev338 — upstream
simply hasn't touched the file between pins; comment refreshed).

Fixes:
- `CHAINED_ANCHOR` verdict + post-sweep reclassify pass (missing
  anchors all found in a same-target sibling's replacement output →
  informational, `chained_on` names the provider).
- Native md5-diff patch evaluation (PN119 class) via the
  `*_PRE_PATCH_MD5` / `*_DIFF_PATH` / `_target_path()` convention —
  also feeds the diff's post-apply text to the chain pass.
- Registry: `P18B_TEXT.requires_patches = ["PN119"]`.
- BONUS: the chain pass discovered a second undocumented chain —
  **P7 anchors on PN365's output** (qwen_gdn_linear_attn.py).

## Current-pin sweep state (after fixes)

204 modules / 232 rows: OK=114, RUNTIME_BINDING=85, DRIFT_ANCHOR=14,
DRIFT_FILE_MOVED=12, CHAINED_ANCHOR=2, UPSTREAM_MERGED=3,
SUB_UPSTREAM_MERGED=1, UNBUILDABLE=1; actionable=32 (current-pin
DRIFT/MOVED rows are mostly known-documented: P64 serving refactor,
gdn/-split). out_of_range residual = 8, all ENV_OVERRIDE_POSSIBLE.

## v1.2 backlog

- SHA-window gh triage (auto-join `compare/<old>...<new>` PRs with
  registry `upstream_pr`).
- In-container liveness leg for RUNTIME_BINDING rows (static symbol
  presence ≠ call-site contract).
- Server-env probes for env-conditional builders (P18B anchors derive
  from `resolve_decode_tune()` — Mac defaults vs server env).
- Triage the 14 remaining DRIFT_ANCHOR + 12 DRIFT_FILE_MOVED rows on
  the current pin (most known; P36/P78/P83/P85/P59/PN58/P84/PN288/
  PN38/P91B/PN32/PN54/P7b need the fix-drifts loop or retire
  decisions). PN58 RESOLVED 2026-06-11: envs.py tail re-anchor
  (minimal entry+brace anchor at pristine envs.py:2012-2013, count==1;
  the single dead envs anchor was atomically blocking all 5 PN58 files
  via MultiFilePatchTransaction — other 5 sub-anchors verified intact.
  Drift evidence in the module's Sub-A drift-history comment).
- 127 undefended self-collision markers (PN369 class) — lint-driven
  cleanup batch.
