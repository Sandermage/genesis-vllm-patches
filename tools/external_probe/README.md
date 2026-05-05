# external_probe — pre-Genesis text-patch probes

This directory holds standalone Python scripts that text-patch the
installed vLLM source **before** Genesis `apply_all` runs. They exist
for upstream PRs whose anchor lives on **pristine** upstream source
(i.e. they need to land before Genesis text-patches that overlap the
same region).

After v7.62.x most of these probes have **proper Genesis-package
equivalents** in `vllm/_genesis/wiring/patch_*.py`. The remaining files
here are kept for compatibility with the 35B FP8 PROD launch path that
hasn't been migrated yet.

## Migration status (2026-04-29)

| File | Genesis equivalent | Status | Action |
|---|---|---|---|
| `patch_40074_iooo.py` | **PN14** `wiring/patch_N14_tq_decode_oob_clamp.py` | **REDUNDANT** | Drop the `python3 /external_probe/patch_40074_iooo.py` line from launch scripts; rely on Genesis apply_all (`GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP=1`). PN14's drift-marker `safe_page_idx` makes it self-skip if external_probe applied first. |
| `patch_tolist_cudagraph.py` | **P78** `wiring/patch_78_tolist_cudagraph_guard.py` | **REDUNDANT** | Drop the launch line; rely on Genesis apply_all (`GENESIS_ENABLE_P78_TOLIST_CAPTURE_GUARD=1`). |
| `patch_pr40798_backport.py` | none yet (config_detect.py probes for it) | **STILL NEEDED** | Keep as-is. PR #40798 has no Genesis backport — config_detect's `_probe_pr40798_active()` checks for the marker this probe writes. |

## Running them

If your launch script uses external_probe, run them BEFORE
`python3 -m vllm._genesis.patches.apply_all`:

```bash
python3 /external_probe/patch_pr40798_backport.py || echo "PR40798 probe failed (non-fatal)"
python3 -m vllm._genesis.patches.apply_all
```

## When to delete a probe

A probe is safe to delete from this directory when ALL three are true:

1. Its Genesis equivalent has shipped to `wiring/`
2. The active launch scripts have been updated to drop the
   `python3 /external_probe/<probe>.py` call
3. No CI/test references the probe

For coordination across hardware tiers, prefer **deprecation notices**
over deletion — if some other operator is using the probe directly,
deleting silently breaks their stack. See the deprecation comment at
the top of each redundant file.

## When to add a new probe

You should NOT add a new probe here unless:

- The fix anchors on **upstream-pristine** source AND
- It must apply before a Genesis text-patch that overlaps the same region
- AND there's no path to do the same via Genesis dispatcher (`should_apply` + `applies_to`)

The default home for new fixes is `vllm/_genesis/wiring/patch_*.py`,
not this directory.

## D4 task tracker

The D4 migration item in the Genesis sprint is the migration of the
remaining external_probe content into proper Genesis-package patches.
Last updated: 2026-04-29 v7.62.x.

- ✅ #40074 → PN14 (commit `0d92e5b`)
- ✅ tolist → P78 (already shipped pre-2026-04-25)
- 🟡 #40798 → no Genesis patch yet; config_detect probes for marker.
  Defer migration until upstream PR resolution: if #40798 merges
  upstream, this probe + the config_detect logic both retire
  automatically. If it doesn't merge in 2 weeks, write a proper
  PNXX wiring patch.
