# Genesis vLLM Patches — rollback playbook

**Owner:** sandermage
**Last updated:** 2026-05-12
**Source spec:** PROJECT_ROADMAP_V2 §9 production-ready item 9 + §6.4

This playbook defines **named rollback procedures** for every major
feature shipped by V2 and the community SDK. Each procedure has:

1. **Trigger** — what symptom indicates rollback is needed.
2. **Revert command** — exact git/file operations.
3. **Smoke command** — proves V1 still works after rollback.
4. **Evidence** — what to record in the evidence ledger.

The playbook is **operator-runnable** — every command is copy-pasteable
on either local or server. It is NOT generated. Drift between this file
and reality is a release blocker (§6.4 release gate).

---

## R-001 — V2 alias resolution broken (`sndr launch prod-35b` fails)

**Trigger:** `sndr launch prod-35b --preflight-only` returns non-zero
exit code with `SchemaError` or `KeyError` from the V2 registry.
Symptom typically appears after a YAML edit in `model_configs/builtin/`
or after a vllm pin bump that renames an existing field.

**Revert command:**

```bash
# Disable V2 resolution path; launcher falls back to V1 preset keys.
git revert --no-edit <SHA_OF_REGISTRY_V2_LANDING>
# Or, if revert is risky:
export GENESIS_DISABLE_V2_ALIAS=1
```

**Smoke command:**

```bash
# Use the V1 preset key directly. V1 path was never removed.
sndr launch a5000-2x-35b-prod --preflight-only
# Expect: rc=0, preflight green, V1 docker emitter prints the same args
#         the operator used pre-V2.
```

**Evidence:** append entry to `ROADMAP_EVIDENCE_LEDGER` with the
rollback command + smoke output + the offending YAML field that broke
V2. Open a finding via the project's external-findings pipeline (see
`EXTERNAL_FINDINGS.md` for maintainer notes) if upstream vLLM caused
the break.

---

## R-002 — Community SDK rejecting a known-good patch

**Trigger:** `sndr patches validate plugins/community/PN<n>` fails on
a patch that previously validated cleanly. Often happens after the
shared validator (`vllm/sndr_core/community/validator.py`) is tightened.

**Revert command:**

```bash
# Disable community SDK pipeline; operator can still run patches via
# direct env flag (the pre-SDK workflow).
export GENESIS_DISABLE_COMMUNITY_SDK=1

# Or, if a specific validator rule is at fault, narrow the disable:
export GENESIS_SDK_SKIP_VALIDATOR=anchor_md5
```

**Smoke command:**

```bash
# Patch is still applied via env flag (pre-SDK path):
GENESIS_ENABLE_PN<n>=1 sndr launch a5000-2x-35b-prod --preflight-only
# Expect: rc=0, patch fires per apply.shadow strict
```

**Evidence:** record the validator rule that rejected the patch +
operator decision (loosen rule vs fix patch).

---

## R-003 — RuntimeCommandSpec emitter divergence (Phase 4.5 regression)

**Trigger:** `sndr launch --dry-run --runtime docker` output differs
semantically from `sndr launch --dry-run --runtime compose` for the
same alias. Or: docker emitter no longer matches pre-refactor golden
output for a known V2 alias.

**Revert command:**

```bash
# Each emitter has a "use legacy direct-read" fallback flag during the
# Phase 4.5 stabilization window:
export SNDR_EMITTER_LEGACY=1
# This routes emitters back to raw ModelConfig.docker / hardware.runtime
# instead of going through RuntimeCommandSpec.
```

**Smoke command:**

```bash
sndr launch prod-35b --dry-run --runtime docker > /tmp/dry-docker.txt
sndr launch prod-35b --dry-run --runtime compose > /tmp/dry-compose.txt
diff <(cat /tmp/dry-docker.txt | sort) <(cat /tmp/dry-compose.txt | sort)
# Expect: only format differences (one is YAML, one is argv), no
#         semantic differences (same env, same mounts, same ports).
```

**Evidence:** capture the diff that triggered rollback + the alias that
exposed it. Open a finding for the upstream cause.

---

## R-004 — `sndr memory explain` mis-predicting OOM

**Trigger:** `sndr memory explain --profile <alias>` reports `SAFE` but
launching `sndr launch <alias>` hits CUDA OOM during profile_run.
Or reports `OOM_RISK` for a profile that operator runs successfully.

**Revert command:**

```bash
# Memory explain is informational only — does NOT gate launch.
# Simply ignore its output until calibration is refreshed.
# If a script blocks on its output, disable:
export SNDR_MEMORY_EXPLAIN_GATING=off
```

**Smoke command:**

```bash
# Launch unaffected — memory explain never blocks runtime.
sndr launch prod-35b --preflight-only
```

**Evidence:** record the actual VRAM usage from `nvidia-smi` vs the
predicted MiB. Update `tools/memory_explain_calibration/v1.yaml` with
the new datapoint via PR.

---

## R-005 — Patch proof gate falsely failing release

**Trigger:** `sndr patches prove --all` reports a stable patch as
missing proof, but operator has manual evidence that it works.

**Revert command:**

```bash
# Either:
# A. Add a time-bounded waiver:
cat > evidence/patch_proof/_waivers/PN<n>.yaml <<EOF
patch_id: PN<n>
owner: sandermage
reason: "anchor drift across vllm 0.20.2 → 0.21.0 rebase; re-anchor pending"
expiry: '2026-06-01'
risk: low
rollback: "revert SHA <X>; patch already default_off in profile Y"
EOF

# B. Or downgrade the patch from `stable` to `beta` until proof refreshed:
# (edit plugins/community/PN<n>/manifest.yaml)
#   implementation_status: beta
```

**Smoke command:**

```bash
sndr patches prove --all
# Expect: rc=0 with the new waiver acknowledged, OR reduced threshold
#         under beta+experimental tier requirement.
```

**Evidence:** waiver file is the evidence; ledger entry records the
operator decision + chosen path (A waiver vs B downgrade).

---

## R-006 — Server diverged from local mid-sync

**Trigger:** `make audit-dirty-state-release` fails on server with
server-only tracked modified files. The §5 safe-sync recipe Step 1
caught it; Step 2 would have overwritten them.

**Revert command:**

```bash
# DO NOT proceed with rsync. Capture server-only state first.
ssh server 'cd /path/to/genesis-vllm-patches && \
  git stash push -m "pre-sync server snapshot $(date -Iseconds)" && \
  git log -5 --oneline'

# Decide per-file: pull back to local, commit on server, or archive.
ssh server 'cd /path/to/... && git stash show -p > /tmp/server-changes.patch'
scp server:/tmp/server-changes.patch /tmp/server-changes.patch
# Review on local, decide, then either:
git apply /tmp/server-changes.patch     # if changes wanted locally
# OR
ssh server 'cd /path/to/... && git stash pop && git add -A && git commit -m "..."'
```

**Smoke command:**

```bash
# After reconciliation, dirty-state release check passes on BOTH hosts:
make audit-dirty-state-release
ssh server 'cd /path/to/... && make audit-dirty-state-release'
```

**Evidence:** ledger entry `convergence-recovery` with the divergence
size + the operator decision + restored `git rev-parse HEAD` on both
hosts.

---

## R-007 — V1 preset stops working after Phase 9 freeze

**Trigger:** Operator runs `sndr launch a5000-2x-35b-prod` (V1 key)
post-Phase-9 and gets `DeprecationWarning` followed by failure. This
means V1 deprecation was upgraded to removal too aggressively.

**Revert command:**

```bash
# Phase 9 freeze adds DeprecationWarning + no-new-V1 CI gate, but does
# NOT remove V1 loader. If V1 loader is gone, revert the freeze SHA:
git revert --no-edit <SHA_OF_PHASE_9_FREEZE>
```

**Smoke command:**

```bash
# V1 preset key still resolves:
sndr launch a5000-2x-35b-prod --preflight-only
# Expect: rc=0 with DeprecationWarning printed but launch succeeds.
```

**Evidence:** ledger entry with the SHA that broke V1 path + decision
to defer V1 removal to a later release tag.

---

## R-008 — License/security gate locking out unlicensed core

**Trigger:** Fresh install of public-core repo refuses to launch with
`License required` error. Public core MUST work without license.

**Revert command:**

```bash
# Public core never requires a license. If a license check appeared,
# the offending module is in vllm/sndr_core/license/. Revert:
git revert --no-edit <SHA_OF_BLOCKING_LICENSE_CHECK>

# Or short-term disable:
export SNDR_LICENSE_REQUIRED=0
```

**Smoke command:**

```bash
sndr license status --json | jq -e '.core == "public (unlicensed)"'
sndr launch prod-35b --preflight-only
# Both must work with no license file present.
```

**Evidence:** ledger entry + open a §6.10 audit-public-docs check —
any wording that implies license-gated public features must be redacted.

---

## Cross-cutting rollback principles

1. **Never lose operator data.** Stash/snapshot before any destructive
   operation. The §5 safe-sync recipe is the template.
2. **V1 path is the floor.** V1 monolithic presets remain bench-tested
   for the duration of V2 work. Any V2 feature failure → fall back to
   V1 + smoke + evidence.
3. **Evidence first, fix second.** Every rollback creates a ledger
   entry. The entry is the input to the post-mortem that prevents
   re-occurrence.
4. **Time-box the waiver.** No "permanent" waiver. Every waiver has an
   expiry ≤30 days from creation.
5. **Public core stays public.** No rollback procedure introduces a
   private dependency or telemetry.

---

## Validation

This playbook is validated by `make audit-rollback-playbook` (new in
Phase 7), which checks:

- every named procedure has Trigger / Revert / Smoke / Evidence sections;
- every revert command is parseable shell;
- every smoke command references a real CLI surface (exists in
  `vllm/sndr_core/cli/`);
- every feature shipped after this file's `Last updated` has a
  corresponding R-XYZ entry.

Drift = release blocked.
