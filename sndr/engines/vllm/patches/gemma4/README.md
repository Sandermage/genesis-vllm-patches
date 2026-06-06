# `integrations/gemma4/` — overlay path compatibility shim

This directory exists for **one and only one reason**: to host the
`upstream_overlay_pr42637` symlink that redirects the historical
overlay path to its canonical post-Phase-2.4 home at
`../attention/turboquant/overlays/pr42637/`.

## Why this shim exists

40+ hand-launchers under `~/start_g4_*.sh` (operator-side, not in
this repo) reference the historical path via:

```bash
OVL=${GENESIS_REPO}/vllm/sndr_core/integrations/gemma4/upstream_overlay_pr42637
```

The β'-A canonical hand-launcher `~/start_g4_betaA_k1.sh` has a
session-locked md5 invariant (`085cdccee8c80aa9b70aef1092d4dfae`)
that bench reproducibility depends on. This symlink is what
preserves that invariant without forcing every launcher to be
re-baselined when the Phase 2.4 relocation moved
`pr42637/` from `integrations/gemma4/upstream_overlay_pr42637/`
to `integrations/attention/turboquant/overlays/pr42637/`.

## Do NOT add anything else here

The Phase 3 relocation audit
(`scripts/audit_phase3_relocation.py:check_r1_gemma_whitelist`) was
relaxed to permit exactly **two entries** in this directory:

1. `README.md` — this file.
2. `upstream_overlay_pr42637` — a symlink whose target must resolve
   to `vllm/sndr_core/integrations/attention/turboquant/overlays/pr42637`.

**Anything else fails R1.** Specifically:

- A `.py` file (any name) → R1 violation.
- A different symlink name → R1 violation.
- A symlink with a different target → R1 violation.
- A subdirectory → R1 violation.
- Any other file (`.DS_Store`, `__pycache__`, etc.) → R1 violation.

The audit's intent is intact: **no Gemma-specific CODE belongs in
this directory**. The carve-out only tolerates the path-compat
shim for historical hand-launchers.

## When this shim can retire

The original two technical prerequisites have landed:

1. `Phase 7.G4.G4_19C-FULLGRAPH-AUDIT` closed the profile-local
   G4_19C handling that the β'-A hand-launcher workflow depended on.
2. `Phase 7.G4.WORKLOAD-GATE-POLICY.IMPLEMENT` gave the V2 preset
   path routing coverage for the K=1 / K=4 / multi-conc /
   structured-JSON workload matrix.

Those closures are necessary but no longer sufficient. The shim is
still retained because the historical launcher path remains an
operator contract until the hand-launcher workflow is formally retired
or re-baselined. In particular, the β'-A canonical launcher md5 above
is still a standing reproducibility invariant.

Retire this directory only in a dedicated overlay-path retirement
phase, after all of the following are true:

1. Every active `~/start_g4_*.sh` workflow that still references
   `integrations/gemma4/upstream_overlay_pr42637` has either been
   replaced by V2 `sndr launch` output or explicitly archived.
2. The β'-A launcher md5 invariant has either been retired by the
   operator or re-baselined against the canonical
   `attention/turboquant/overlays/pr42637/` path.
3. Both primary and rig worktrees are already converged on the same
   dev commit, so removing the shim cannot be reintroduced by a stale
   integration branch.

**The overlay itself is treated as project-owned canonical code**
under `attention/turboquant/overlays/pr42637/` (verbatim copy of
vllm PR #42637 source). Retirement of THIS shim directory is NOT
gated on the upstream PR42637 merging — it's gated only on the two
operator-controlled launcher conditions above. The «wait for
upstream merge» framing that previous revisions of this file
carried has been retired (2026-05-28); see
`sndr_private/planning/audits/LOCAL_PR42637_CLOSURE_R_2026-05-28_RU.md`.

The retirement commit must delete this README, delete the symlink,
remove the R1 carve-out in `scripts/audit_phase3_relocation.py`, and
update the R1 tests to make `integrations/gemma4/` forbidden again.

## History

- 2026-05-22 — Phase 2.4 (`a57257be`) relocated `pr42637/` overlay
  files to `attention/turboquant/overlays/pr42637/`.
- 2026-05-22 — Phase 2.5 deleted `integrations/gemma4/` from the
  tracked tree.
- 2026-05-23 — Phase 7.G4.B1.1 created an untracked rig-local
  symlink shim at this path to preserve β'-A hand-launcher md5
  invariant.
- 2026-05-23 — Phase 7.G4.OVERLAY-PATH-CONSISTENCY committed the
  shim as a tracked symlink + this README + R1 carve-out, making
  laptop and rig identical.
- 2026-05-25 — RIG-INTEGRATION.WRITE.1 converged primary and rig on
  the same dev commit. The shim remains intentionally retained only
  for the historical launcher path contract described above.
