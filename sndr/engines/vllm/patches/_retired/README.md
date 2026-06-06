# `_retired/` — archived patch wirings

Genesis patch lifecycle has 4 active phases (`experimental`, `validated`,
`legacy`, `community-experimental`) and 1 terminal phase: `retired`.
Once a patch is retired, its registry entry stays as-is for audit trail,
but the on-disk wiring module moves here.

## Policy

A patch transitions to `_retired/` when **one** of these conditions holds:

1. **Upstream merged the same fix.** The patch is now redundant after a
   specific vLLM pin bump (e.g. `vllm_version_range: ">=0.20.2rc1.dev209,<inf"`).
2. **Hypothesis disproven empirically.** A research-track patch was
   benchmarked and shown to not provide the expected gain (or to regress).
   Registry retains `retired_waiver: True` + brief explanation.
3. **Duplicate of another active patch.** Two patches with overlapping
   functionality where one is consolidated into the other. Registry
   retains `superseded_by: "<other_pid>"`.
4. **Deprecated mechanism.** Patch is replaced by a new, more robust
   approach (e.g. `P65 → P67` workaround→root-cause).

## Registry contract for retired patches

Every retired entry must have:

- `lifecycle: "retired"` — drives dispatcher skip + audit gates
- At least one of:
  - `superseded_by: "<other_pid>"` — names the replacement
  - `retired_waiver: True` + `credit` text — explains why retired without replacement
  - `vllm_version_range: ">=X,<Y"` — version window where active (for drift safety)
- `apply_module` — pointing to `vllm.sndr_core.integrations._retired.<file>`
  (so `audit_registry_contract.py` can validate import path).

## Why keep registry entries (vs delete)?

1. **Audit trail** — boot logs show "patch X skipped (retired since pin Y)"
2. **Drift detection** — operator running against an old pin gets explicit
   warning that retired patch was last active on different pin.
3. **Cross-rig community evidence** — patches retired locally may still
   be active on community forks; registry IDs stay stable for issue ref.

## Current contents

(Filled by Phase 2 cleanup 2026-05-14)

| Patch ID | Moved | Reason |
|---|---|---|
| (see registry.py `lifecycle: retired` entries) | | |

## How to add a patch here

1. `git mv vllm/sndr_core/integrations/<family>/<file>.py vllm/sndr_core/integrations/_retired/<file>.py`
2. Update registry entry:
   ```python
   "lifecycle": "retired",
   "apply_module": "vllm.sndr_core.integrations._retired.<file>",
   "superseded_by": "<other_pid>",  # or retired_waiver: True
   "vllm_version_range": ">=X,<Y",  # version window where active
   ```
3. Run `python3 scripts/audit_registry_contract.py` — should stay green
4. Run `make evidence` — should stay 40/40
