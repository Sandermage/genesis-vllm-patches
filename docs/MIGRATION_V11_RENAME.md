# v11.0.0 — `Genesis → sndr_core` package rename

Released **2026-05-08** (hard flip). This appendix documents the
rename, the structural changes that came with it, and what an
operator on a pre-v11 install needs to do on upgrade.

> ⚠ This page exists because the public README cannot reference
> the retired `vllm/_genesis` namespace by name without tripping
> the `audit-docs-stale` gate. The retired tokens are the subject
> of this document; that is the documented exception for the
> stale-token allowlist.

---

## Why the rename happened

Up to v10.x the Python package lived at `vllm/_genesis/`. Three
problems forced a hard rename in v11.0.0:

1. `_genesis` looked like a private vLLM module — confusing for
   vLLM maintainers and operators alike. It is not part of vLLM
   and never was.
2. The old single-file `apply_all.py` (4 542 lines) did not scale.
   Pull requests conflicted on every change, reviews stalled, and
   there were no family-level lines of responsibility.
3. Operator UX was thin — no CLI, no schema-driven configs, no
   audit gate, no per-patch observability.

## Before vs After

| v10.x (Genesis-named) | v11.0.0 (SNDR Core) | Effect |
|---|---|---|
| `vllm/_genesis/` (235 files, flat) | `vllm/sndr_core/` (family-organized) | Clear hierarchy |
| `apply_all.py` 4 542 lines | `apply/{orchestrator,verify,shadow,_per_patch_dispatch}.py` | PRs localised |
| Flat `_genesis/patches/` | `integrations/<family>/<patch>.py` across 21 families | Review by area |
| `import _genesis` side-effects on boot | Lazy `vllm.sndr_core.__init__` (torch-less importable) | CI / preflight without CUDA |
| Boot summary scattered across uvicorn INFO | Structured boot summary + per-patch `elapsed_ms` + `rss_delta` | Observability |
| Hardcoded paths (`/home/<user>/...`, LAN IPs) | Portable env vars (`$GENESIS_MODELS_DIR`, …) | Reproducibility |
| `patch_genesis_unified.py` shim | Removed | Cleaner |
| `vllm/sndr_core/wiring/patch_*.py` | `vllm/sndr_core/integrations/<family>/<patch>.py` | Family taxonomy |
| `~/.genesis/` config dir | `~/.sndr/` (legacy alias `~/.genesis/` honoured) | Canonical name |
| No CLI | `sndr launch / doctor / verify / model-config / deps / patches` | Operator UX |
| Single-format `model_configs/*.yaml` (V1 monolithic) | V1 still works + V2 layered (`model/`, `hardware/`, `profile/`, `presets/`) | Reusable building blocks |

## What improved

- **Single CLI entry point** — `sndr launch <preset>` replaces 18
  ad-hoc `start_*.sh` / `bare_metal_*.sh` scripts.
- **Schema-driven model configs** with 16 `audit_rules.py` checks
  (R-001 … R-016) and a `make evidence` release gate that runs them.
- **Anchor-manifest fast-path** — text patches now record the
  anchor SHA; on upstream drift the patch self-skips with a clear
  `drift_marker detected` line instead of silently breaking.
- **Per-patch observability** — `GENESIS_OBSERVABILITY=1` prints
  `elapsed_ms` and `rss_delta` for every patch on boot.
- **40-gate `make evidence`** — release-tier audit covering legacy
  imports, hardcoded paths, security scan, community gate, lifecycle
  ratchet, doc sync. 40 / 40 green at HEAD.
- **21 family taxonomy** — patches grouped by subsystem
  (`attention.gdn`, `spec_decode`, `kv_cache`, …) instead of one
  bag.
- **Signed-token license gate** — Ed25519 hook for future
  commercial overlay (`vllm.sndr_engine` namespace; currently
  empty).

## What was removed

- `vllm/_genesis/` — entire tree, 235 files (commit `776aa32b`).
- `patch_genesis_unified.py` — pre-v11 back-compat shim.
- `vllm/sndr_core/wiring/patch_*.py` — replaced by canonical
  `integrations/<family>/<patch>.py`.
- 11 retired patches whose upstream-merged equivalents are now in
  the vLLM nightly pin (P94, PN9, …).
- `vllm/sndr_core/compat/fingerprints/` — stale 3-file
  cap-detection cache.
- `sponsor-site/` — separate project, did not belong in this repo.

## What stayed (on purpose)

- The **name "Genesis"** in documentation, banner, wave numbers —
  it is the project's brand. Only the Python package was renamed.
- **V1 monolithic model configs** (`a5000-2x-35b-prod.yaml`, …) —
  they still load and pass the same audit gate as V2. New configs
  SHOULD use V2, but V1 is not forced to migrate.
- **`~/.genesis/` legacy alias** for the config dir — existing
  operators do not have to move state.

## Migration steps for a pre-v11 install

```bash
# 1. Pull the v11+ release
cd /path/to/genesis-vllm-patches
git fetch && git checkout main

# 2. Re-install plugin so it points at sndr_core.plugin:register
pip install -e .

# 3. Rewrite any custom script that imports the retired namespace
grep -rn 'vllm\._genesis\|vllm/_genesis' your-scripts/ |
  awk -F: '{print $1}' |
  sort -u |
  xargs sed -i 's/vllm\._genesis/vllm.sndr_core/g; s|vllm/_genesis|vllm/sndr_core|g'

# 4. Verify the import path resolves
python3 -c 'import vllm.sndr_core; print(vllm.sndr_core.__file__)'

# 5. Run smoke
python3 -m vllm.sndr_core.cli doctor
```

> ⚠ There is **no back-compat alias** for the retired namespace.
> `import vllm._genesis` raises `ModuleNotFoundError`. Pre-v11
> launch scripts and tools must be updated before they will run on
> v11.0.0 or later.

## See also

- [README.md](../README.md) — current state, install, benchmarks.
- [CHANGELOG.md](../CHANGELOG.md) — per-release detail
  (v7.x → v11.0.0+wave9).
- [docs/INSTALL.md](INSTALL.md) — installer reference, including the
  troubleshooting tree for upgrade paths.
