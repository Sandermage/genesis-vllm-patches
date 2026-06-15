# Compat-mount (`vllm/sndr_core`) removal — BLOCKED on a rig deployment migration

**Date**: 2026-06-15
**Goal (operator)**: "все файлы были внутри папки sndr и не создавали нечего за ней" — stop
the container from creating a `vllm/sndr_core` path; everything under `sndr/`.

## What was done (code side — DONE)

All RUNTIME consumers of `vllm.sndr_core` in the repo were eliminated and committed
(28bb2fe1, a03f0c8d, bfa1df50): schema resolution → `sndr.schemas`; tier_manager metric
key → `sndr.cache._pn95_runtime` (also fixed a latent dropped-counter bug); `_resolve_patches_dir`
canonical-only; docstrings; a validation script's in-container import; the stray repo-root
`vllm/sndr_core/` shim tree removed. `pyproject.toml` already declares the modern entry-point
`genesis_v7 = "sndr.plugin:register"` (v12.0.0), and `sndr/plugin.py::register()` is the real
plugin (imports the actual patch modules). **The local code is v12-clean.**

## Why the mount can't be removed yet — the boot test caught it

Booted the 27B WITHOUT the `vllm/sndr_core` overlay mount. Result: health=200 and
`import sndr` works, BUT:
```
Failed to load plugin genesis_v7 → ModuleNotFoundError: No module named 'vllm.sndr_core'
… no "Genesis Results:" line … smoke chat → HTTP 500 (EngineCore crash, no patches applied)
```
Root cause: the mount is **load-bearing**, via TWO deployment-level (not source-level)
dependencies my "0 source-imports" scan didn't cover:

1. **The installed plugin entry-point is STALE on the rig.** The rig's
   `~/genesis-vllm-patches/pyproject.toml` still says
   `genesis_v7 = "vllm.sndr_core.plugin:register"` and there is a stale
   `vllm_sndr_core-11.0.0.dist-info/entry_points.txt` with the same. So `pip install -e` at
   boot registers the OLD entry-point, which needs `vllm.sndr_core` — provided only by the
   compat mount. (Local pyproject is already migrated; the RIG checkout is a stale v11.)
2. **The 35B PROD launcher applies patches via `python3 -m vllm.sndr_core.apply`** (explicit
   legacy module), and mounts the rig's **stale v11 `vllm/sndr_core/` SOURCE tree**
   (`/home/sander/genesis-vllm-patches/vllm/sndr_core` still exists on the rig). The 27B
   launcher instead mounts the v12 `sndr/` source onto BOTH `dist-packages/sndr` and
   `…/vllm/sndr_core`, so `vllm.sndr_core` resolves to the v12 `sndr/` — which is why the
   27B runs v12 code while the 35B runs the stale v11 tree. **The rig is a v11/v12 mix.**

## The actual fix (a staged rig deployment migration — needs a focused, operator-aware run)

This is NOT a code-only cleanup; it changes how PROD launches, so stage it carefully:

1. **Refresh the rig checkout to v12**: full sync of `~/genesis-vllm-patches` (bring the
   modern `pyproject.toml`; REMOVE the stale `vllm/sndr_core/` source tree). Do this while
   no container is live-mounting it, or restart after.
2. **Purge the stale dist-info**: remove `vllm_sndr_core-11.0.0.dist-info` (and any
   `vllm.sndr_core`-keyed entry-point) so `pip install -e` registers ONLY
   `genesis_v7 = sndr.plugin:register`. Verify in-container:
   `grep -r genesis_v7 …/dist-packages/*.dist-info/entry_points.txt` → must be `sndr.plugin:register`.
3. **Re-render the launchers** (27B + 35B + Gemma-4) from the current profiles so they: mount
   ONLY `…/dist-packages/sndr` (drop the `vllm/sndr_core` overlay), and apply patches via the
   plugin (or `python3 -m sndr.apply` / `python3 -m sndr.cli apply` if an explicit apply is
   kept) — NOT `python3 -m vllm.sndr_core.apply`.
4. **Then** land the local code change (already drafted/validated by an agent, reverted as
   premature): drop the mount from `cli/legacy/profile.py`, `audit_launch_coverage.py`
   `REQUIRED_MOUNTS` (6→5), the 3 hardware YAMLs, the 2 samples, re-point `deployment.py`'s
   daemon-derivation from `/vllm/sndr_core$` to `/dist-packages/sndr$`, and update the 3
   affected unit tests. Boot-test mount-free on the 27B FIRST, then the 35B PROD.

## Status

- Code v12-clean + committed. Mount removal reverted (premature; the rig isn't migrated).
- Rig restored to known-good (27B + mount, v12 path). The mount stays until the migration.
- The boot-test discipline worked: it caught a load-bearing dependency a static scan missed.
  The "0 runtime imports" claim was true for SOURCE imports but missed the packaging
  entry-point + the explicit `-m vllm.sndr_core.apply` in the stale 35B launcher.
