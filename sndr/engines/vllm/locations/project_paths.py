# SPDX-License-Identifier: Apache-2.0
"""SNDR Core internal paths — single source of truth.

Companion to `engine_targets.py` (vllm engine paths). This module
centralizes ALL paths that are INSIDE the SNDR Core / Genesis project
itself (not the patched vllm tree):

  - Where wiring patch files live
  - Where Site Map manifest JSON lives
  - Where pristine fixtures live (test reference content)
  - Where operator config + telemetry write
  - Where model_configs YAML files live (builtin/community/user)

Why centralize:
  Before this module existed, paths like `vllm/_genesis/wiring/` or
  `~/.genesis/telemetry/` were hardcoded across 5+ files. Renaming
  `_genesis` to anything else (or moving the project to `~/.sndr/`)
  required grep-replace across the codebase. Now those refs point
  here; one constant change propagates everywhere.

Env overrides recognized (all optional):
  SNDR_HOME           — operator-local install dir (default: ~/.sndr)
  SNDR_WIRING_DIR     — explicit wiring path (default: auto-detect)
  SNDR_MANIFEST_DIR   — explicit manifests path
  SNDR_TELEMETRY_DIR  — explicit telemetry write dir
  SNDR_MODEL_CONFIG_DIR — operator-local model_configs override

Legacy aliases (back-compat with v7.x env names):
  GENESIS_HOME, GENESIS_MODEL_CONFIG_DIR, GENESIS_TELEMETRY_DIR

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────
# Helpers — env override with legacy fallback
# ─────────────────────────────────────────────────────────────────────

def _env_path(*names: str) -> Optional[Path]:
    """Return Path from first non-empty env var in `names`, else None.

    Order: SNDR_* preferred, GENESIS_* (legacy alias) fallback.
    """
    for name in names:
        val = os.environ.get(name, "").strip()
        if val:
            return Path(val).expanduser()
    return None


def _package_root() -> Path:
    """Return absolute path to `sndr/engines/vllm/` itself (this package).

    Used by other helpers to derive sibling paths like
    `vllm/sndr_core/integrations/` (v10 canonical) or the legacy
    `vllm/_genesis/wiring/` fallback during ongoing transition.
    """
    return Path(__file__).resolve().parent.parent


def _vllm_namespace_root() -> Path:
    """Return absolute path to `vllm/` (the namespace package parent).

    Examples:
      `_package_root()`         → /repo/sndr/engines/vllm
      `_vllm_namespace_root()`  → /repo/vllm
    """
    return _package_root().parent


# ─────────────────────────────────────────────────────────────────────
# 1. Operator-local install / config root
# ─────────────────────────────────────────────────────────────────────

def install_root() -> Path:
    """Operator-local install dir.

    P1-5 (audit 2026-05-08): canonical default is `~/.sndr/`. Falls
    back to `~/.genesis/` only when the legacy dir exists alone, so
    v7.x operators don't get bumped to a fresh empty home.

    Honored env: `SNDR_HOME` (canonical), `GENESIS_HOME` (legacy alias).
    Used by:
      - install.sh: clone target.
      - cli/install.py: where to write user-local configs.
      - telemetry: storage dir parent.
    """
    p = _env_path("SNDR_HOME", "GENESIS_HOME")
    if p is not None:
        return p
    sndr_default = Path.home() / ".sndr"
    if sndr_default.is_dir():
        return sndr_default
    legacy_default = Path.home() / ".genesis"
    if legacy_default.is_dir():
        return legacy_default
    return sndr_default  # canonical for write-mode callers


# ─────────────────────────────────────────────────────────────────────
# 2. Wiring dir (where patch_*.py files live)
# ─────────────────────────────────────────────────────────────────────

def wiring_dir() -> Optional[Path]:
    """Directory containing patch wiring files.

    Honored env: `SNDR_WIRING_DIR` (explicit override).

    Precedence (v12.0 post-rename 2026-06-04):
      1. ``sndr/engines/vllm/patches/`` — canonical v12 home for all
         patch implementations under the multi-engine refactor.
      2. ``vllm/sndr_core/integrations/`` — v11 canonical, retained
         as fallback during the v12.x shim window.
      3. ``vllm/sndr_core/patches/`` — pre-v11 legacy.
      4. ``vllm/_genesis/wiring/`` — pre-v8 legacy.

    Bug v12.0 (2026-06-04): the pre-rename code used
    ``_vllm_namespace_root() / "sndr_core" / "integrations"`` but the
    ``_package_root()`` helper computes ``Path(__file__).parent.parent``,
    which after the refactor resolves to ``sndr/engines/vllm/`` rather
    than ``vllm/sndr_core/`` — so the legacy probe targeted a path that
    never exists. Now we first probe the new v12 location relative to
    this file, then fall back to the absolute v11 location via a
    repo-root walk.
    """
    p = _env_path("SNDR_WIRING_DIR")
    if p is not None and p.is_dir():
        return p

    # v12 canonical: sndr/engines/vllm/patches/ — relative to this file
    # at sndr/engines/vllm/locations/project_paths.py.
    v12_canonical = Path(__file__).resolve().parent.parent / "patches"
    if v12_canonical.is_dir():
        return v12_canonical

    # v11 legacy fallback: vllm/sndr_core/integrations/.
    # _package_root() of THIS file is sndr/engines/vllm/, so walk up to
    # find the repo root and resolve the legacy tree from there.
    here = Path(__file__).resolve()
    # Walk up looking for a "vllm/sndr_core" sibling — handles both
    # editable install (repo root) and wheel-mounted layouts.
    for ancestor in here.parents:
        canonical = ancestor / "sndr" / "engines" / "vllm" / "patches"
        if canonical.is_dir():
            return canonical
        legacy_root = ancestor / "vllm" / "sndr_core"
        if legacy_root.is_dir():
            canonical = legacy_root / "integrations"
            legacy_patches = legacy_root / "patches"
            legacy_wiring = ancestor / "vllm" / "_genesis" / "wiring"
            if canonical.is_dir():
                return canonical
            if legacy_patches.is_dir():
                return legacy_patches
            if legacy_wiring.is_dir():
                return legacy_wiring
            break
    return None


# ─────────────────────────────────────────────────────────────────────
# 3. Manifest dir (Site Map anchor_manifest.json + pristine fixtures)
# ─────────────────────────────────────────────────────────────────────

def manifest_dir() -> Path:
    """Directory containing `anchor_manifest.json` + pristine fixtures.

    Honored env: `SNDR_MANIFEST_DIR`.

    Default precedence (v12 rename 2026-06-04+):
      1. `sndr/manifests/` — canonical v12 home (this package root).
      2. `vllm/sndr_core/manifests/` — v11 legacy fallback via
         repo-root walk (pre-v12 checkouts only).

    Bug v12.0 (same class as the `wiring_dir` fix): the pre-rename code
    derived the path from `_vllm_namespace_root()`, which after the
    refactor resolves to `sndr/engines/` — so both probes targeted
    paths that never exist. Resolve relative to this file instead.
    """
    p = _env_path("SNDR_MANIFEST_DIR")
    if p is not None:
        return p

    # v12 canonical: sndr/manifests/ — this file lives at
    # sndr/engines/vllm/locations/project_paths.py, so the package
    # root `sndr/` is parents[3] (valid for editable + wheel layouts).
    v12_canonical = Path(__file__).resolve().parents[3] / "manifests"
    if v12_canonical.is_dir():
        return v12_canonical

    # v11 legacy fallback: vllm/sndr_core/manifests/ (pre-v12 trees).
    for ancestor in Path(__file__).resolve().parents:
        legacy = ancestor / "vllm" / "sndr_core" / "manifests"
        if legacy.is_dir():
            return legacy

    return v12_canonical  # canonical for write-mode callers


def manifest_json_path() -> Path:
    """Path to the active anchor_manifest.json (Site Map data)."""
    return manifest_dir() / "anchor_manifest.json"


def pristine_fixtures_dir() -> Path:
    """Directory containing pristine vllm source files (test references).

    Used by:
      - `core/manifest_cache.py` — md5-verify pristine vs installed.
      - `tests/legacy/pristine_fixtures/` (current canonical layout;
        `tests/unit/integrations/<family>/` holds active contracts).
    """
    p = _env_path("SNDR_PRISTINE_FIXTURES_DIR")
    if p is not None:
        return p
    repo_root = _vllm_namespace_root().parent
    canonical = repo_root / "tests" / "legacy" / "pristine_fixtures"
    if canonical.is_dir():
        return canonical
    return _vllm_namespace_root() / "_genesis" / "tests" / "pristine_fixtures"


# ─────────────────────────────────────────────────────────────────────
# 4. Telemetry / observability output
# ─────────────────────────────────────────────────────────────────────

def telemetry_dir() -> Path:
    """Where opt-in telemetry reports are written.

    Honored env: `SNDR_TELEMETRY_DIR` (preferred),
                 `GENESIS_TELEMETRY_DIR` (legacy).
    Default: `<install_root()>/telemetry/`.
    """
    p = _env_path("SNDR_TELEMETRY_DIR", "GENESIS_TELEMETRY_DIR")
    if p is not None:
        return p
    return install_root() / "telemetry"


# ─────────────────────────────────────────────────────────────────────
# 5. Model configs (3-tier: builtin / community / user)
# ─────────────────────────────────────────────────────────────────────

def model_configs_builtin_dir() -> Path:
    """Built-in shipped model configs (read-only).

    v10 (2026-05-07): canonical home moved to `sndr_core/model_configs/`;
    legacy `_genesis/model_configs/` kept as duplicate during transition.
    Falls back to legacy if canonical is absent.
    """
    vllm = _vllm_namespace_root()
    canonical = vllm / "sndr_core" / "model_configs" / "builtin"
    if canonical.is_dir():
        return canonical
    return vllm / "_genesis" / "model_configs" / "builtin"


def model_configs_community_dir() -> Path:
    """Community-contributed model configs (PR'd).

    v10 (2026-05-07): canonical home moved to `sndr_core/model_configs/`;
    legacy fallback preserved during transition.
    """
    vllm = _vllm_namespace_root()
    canonical = vllm / "sndr_core" / "model_configs" / "community"
    if canonical.is_dir():
        return canonical
    return vllm / "_genesis" / "model_configs" / "community"


def model_configs_user_dir() -> Path:
    """Operator-local model configs (never committed).

    Honored env: `SNDR_MODEL_CONFIG_DIR` (preferred),
                 `GENESIS_MODEL_CONFIG_DIR` (legacy).
    Default: `<install_root()>/model_configs/`.
    """
    p = _env_path("SNDR_MODEL_CONFIG_DIR", "GENESIS_MODEL_CONFIG_DIR")
    if p is not None:
        return p
    return install_root() / "model_configs"


# ─────────────────────────────────────────────────────────────────────
# 6. MoE tuning configs (TQ k8v4 sweep data)
# ─────────────────────────────────────────────────────────────────────

def moe_tuning_dir() -> Path:
    """Directory containing fused-MoE Triton kernel tuning configs.

    Canonical home: `sndr_core/configs/moe_tuning/`. The legacy
    `_genesis/configs/moe_tuning/` fallback was removed in v11 along
    with the rest of `vllm/_genesis/` (commit 776aa32b). Override via
    `SNDR_MOE_TUNING_DIR` if operators want to ship their own tuned
    configs from outside the package.
    """
    p = _env_path("SNDR_MOE_TUNING_DIR")
    if p is not None:
        return p
    vllm = _vllm_namespace_root()
    return vllm / "sndr_core" / "configs" / "moe_tuning"


# ─────────────────────────────────────────────────────────────────────
# 7. Plugin host config
# ─────────────────────────────────────────────────────────────────────

def host_config_path() -> Path:
    """Where `host.yaml` lives (deployment runtime config).

    Honored env: `SNDR_HOST_CONFIG`.
    Default: `<install_root()>/host.yaml`.
    """
    p = _env_path("SNDR_HOST_CONFIG", "GENESIS_HOST_CONFIG")
    if p is not None:
        return p
    return install_root() / "host.yaml"


# ─────────────────────────────────────────────────────────────────────
# 8. Model weights + runtime caches (host-side; bridged to YAML + scripts)
# ─────────────────────────────────────────────────────────────────────
#
# 2026-05-11 audit F-013 closure: model weight paths + runtime caches
# (torch_compile, triton, huggingface) were hardcoded across server
# start-scripts, YAML configs, and shell launchers. Centralized here
# so that:
#
#   - Python code reads them via these helpers
#   - YAML configs use `${models_dir}` / `${compile_cache_dir}` style
#     env-var substitution that resolves to the SAME canonical values
#   - Server bash scripts can `source ~/.genesis_paths.env` (rendered
#     by `sndr doctor --emit-paths-env`) and reuse identical values
#
# Env vars (all optional; canonical SNDR_* + legacy GENESIS_* aliases):
#
#   SNDR_MODELS_DIR / GENESIS_MODELS_DIR
#       Host-side model weights root. Default: /models when present,
#       else ~/.cache/sndr/models. Volume-mounted into container at
#       /models per YAML convention.
#
#   SNDR_COMPILE_CACHE_DIR / GENESIS_COMPILE_CACHE_DIR
#       torch.compile / inductor cache. Default:
#       <install_root>/cache/compile/.
#
#   SNDR_TRITON_CACHE_DIR / GENESIS_TRITON_CACHE_DIR
#       Triton kernel cache. Default: <install_root>/cache/triton/.
#
#   SNDR_HF_CACHE_DIR / HF_HOME (HuggingFace canonical) / GENESIS_HF_CACHE_DIR
#       HuggingFace cache. Default: ~/.cache/huggingface (HF default).

def models_dir() -> Path:
    """Host-side root where model weight checkpoints live.

    Container-side path is conventionally `/models/<model_name>` via
    volume mount; YAML configs reference this as `${models_dir}` so
    operators can override per host without editing builtin YAMLs.

    Honored env (in order): `SNDR_MODELS_DIR`, `GENESIS_MODELS_DIR`.
    Default: `/models` if it exists, else `~/.cache/sndr/models`.
    """
    p = _env_path("SNDR_MODELS_DIR", "GENESIS_MODELS_DIR")
    if p is not None:
        return p
    canonical = Path("/models")
    if canonical.is_dir():
        return canonical
    return Path.home() / ".cache" / "sndr" / "models"


def compile_cache_dir() -> Path:
    """torch.compile / Inductor cache (vllm `--cache_dir` for `torch_compile_cache`).

    Honored env: `SNDR_COMPILE_CACHE_DIR`, `GENESIS_COMPILE_CACHE_DIR`.
    Default: `<install_root()>/cache/compile/`.
    """
    p = _env_path("SNDR_COMPILE_CACHE_DIR", "GENESIS_COMPILE_CACHE_DIR")
    if p is not None:
        return p
    return install_root() / "cache" / "compile"


def triton_cache_dir() -> Path:
    """Triton kernel cache directory.

    Honored env: `SNDR_TRITON_CACHE_DIR`, `GENESIS_TRITON_CACHE_DIR`,
                 `TRITON_CACHE_DIR` (Triton's own canonical env var).
    Default: `<install_root()>/cache/triton/`.
    """
    p = _env_path(
        "SNDR_TRITON_CACHE_DIR",
        "GENESIS_TRITON_CACHE_DIR",
        "TRITON_CACHE_DIR",
    )
    if p is not None:
        return p
    return install_root() / "cache" / "triton"


def hf_cache_dir() -> Path:
    """HuggingFace cache directory (model downloads + datasets).

    Honored env: `SNDR_HF_CACHE_DIR`, `HF_HOME` (HuggingFace canonical),
                 `GENESIS_HF_CACHE_DIR`.
    Default: `~/.cache/huggingface` (HF's own default).
    """
    p = _env_path("SNDR_HF_CACHE_DIR", "HF_HOME", "GENESIS_HF_CACHE_DIR")
    if p is not None:
        return p
    return Path.home() / ".cache" / "huggingface"


# ─────────────────────────────────────────────────────────────────────
# 9. CLI / Boot diagnostic — print all paths
# ─────────────────────────────────────────────────────────────────────

def all_paths() -> dict[str, Path | None]:
    """Snapshot of all SNDR Core internal paths. Used by `sndr doctor`."""
    return {
        "install_root":              install_root(),
        "wiring_dir":                wiring_dir(),
        "manifest_dir":              manifest_dir(),
        "manifest_json_path":        manifest_json_path(),
        "pristine_fixtures_dir":     pristine_fixtures_dir(),
        "telemetry_dir":             telemetry_dir(),
        "model_configs_builtin_dir": model_configs_builtin_dir(),
        "model_configs_community_dir": model_configs_community_dir(),
        "model_configs_user_dir":    model_configs_user_dir(),
        "moe_tuning_dir":            moe_tuning_dir(),
        "host_config_path":          host_config_path(),
        # Section 8 (audit F-013 closure 2026-05-11):
        "models_dir":                models_dir(),
        "compile_cache_dir":         compile_cache_dir(),
        "triton_cache_dir":          triton_cache_dir(),
        "hf_cache_dir":              hf_cache_dir(),
    }


def emit_env_shell(prefix: str = "GENESIS") -> str:
    """Render canonical path values as a sourcable shell snippet.

    Operator workflow (audit F-013 closure 2026-05-11):
      1. Run `python3 -m sndr.engines.vllm.locations.project_paths --emit-env > ~/.genesis_paths.env`
      2. In start-scripts: `source ~/.genesis_paths.env` BEFORE launching docker
      3. Docker mounts read the same canonical values:
           -v "${GENESIS_MODELS_DIR}":/models:ro
           -v "${GENESIS_COMPILE_CACHE_DIR}":/root/.cache/vllm/torch_compile_cache
           -v "${GENESIS_TRITON_CACHE_DIR}":/root/.triton/cache
           -v "${GENESIS_HF_CACHE_DIR}":/root/.cache/huggingface

    Single source of truth: edit env vars OR project_paths.py defaults;
    both Python (via this module) and bash (via the rendered env file)
    pick up identical values.
    """
    p = all_paths()
    lines = [
        "# Genesis canonical paths — rendered from project_paths.py",
        "# Source this file BEFORE running start-scripts:",
        "#   source ~/.genesis_paths.env",
        "# Override any value by setting the env var BEFORE sourcing.",
        "",
        f"export {prefix}_INSTALL_ROOT={str(p['install_root'])!r}",
        f"export {prefix}_MODELS_DIR={str(p['models_dir'])!r}",
        f"export {prefix}_COMPILE_CACHE_DIR={str(p['compile_cache_dir'])!r}",
        f"export {prefix}_TRITON_CACHE_DIR={str(p['triton_cache_dir'])!r}",
        f"export {prefix}_HF_CACHE_DIR={str(p['hf_cache_dir'])!r}",
        f"export {prefix}_MODEL_CONFIG_DIR={str(p['model_configs_user_dir'])!r}",
        f"export {prefix}_TELEMETRY_DIR={str(p['telemetry_dir'])!r}",
        "",
        "# vllm-side env passthroughs (canonical names where they exist):",
        f"export TRITON_CACHE_DIR={str(p['triton_cache_dir'])!r}",
        f"export HF_HOME={str(p['hf_cache_dir'])!r}",
    ]
    return "\n".join(lines) + "\n"


__all__ = [
    "install_root",
    "wiring_dir",
    "manifest_dir",
    "manifest_json_path",
    "pristine_fixtures_dir",
    "telemetry_dir",
    "model_configs_builtin_dir",
    "model_configs_community_dir",
    "model_configs_user_dir",
    "moe_tuning_dir",
    "host_config_path",
    # Section 8 (audit F-013 closure 2026-05-11):
    "models_dir",
    "compile_cache_dir",
    "triton_cache_dir",
    "hf_cache_dir",
    "all_paths",
    "emit_env_shell",
]


# CLI entry point lives at `scripts/emit_paths_env.py` (avoids the
# runpy warning that fires when a module already loaded via package
# import is then run as `python -m`). Operator workflow:
#   python3 scripts/emit_paths_env.py > ~/.genesis_paths.env
#   source ~/.genesis_paths.env
