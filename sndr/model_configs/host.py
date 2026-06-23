# SPDX-License-Identifier: Apache-2.0
"""Host config (`~/.genesis/host.yaml`) + auto-detection of common paths.

W-runtime 2026-05-06: Genesis configs reference paths via symbolic vars
(`${models_dir}`, `${hf_cache}`, etc.) so they're portable across operator
rigs. Per-host concrete paths live in `~/.genesis/host.yaml`, auto-detected
at install / first-run by scanning common locations.

Variables recognized:
  models_dir     — where model weights live (HF format dirs)
  hf_cache       — HuggingFace cache root (downloaded files)
  triton_cache   — Triton kernel cache (persistent across container restarts)
  compile_cache  — vLLM torch.compile cache (persistent compile artifacts)
  sndr_src       — checkout of genesis-vllm-patches/sndr (RO mount
                   source). `sndr_src` is the canonical variable name (v12);
                   `genesis_src` is accepted as a legacy alias for back-compat
                   with existing host.yaml files (key alias-in at load time;
                   the GENESIS_SRC env var also remains accepted). Physical
                   path points at sndr_core after v11 (_genesis dir removed
                   2026-05-08).
  plugin_src     — operator-side path to the sndr repo ROOT that the launch
                   renderer bind-mounts at /plugin and (under
                   SNDR_DEV_INSTALL_PLUGIN=1) pip-installs editable so its
                   `vllm.general_plugins` entry-point (genesis_v7 =
                   sndr.plugin:register) registers INSIDE the serving
                   container — this is what makes vllm serve re-apply
                   runtime monkey-patches in-process. Must be the repo root
                   (with the root pyproject.toml), NOT the empty legacy
                   tools/genesis_vllm_plugin subdir. Set via SNDR_PLUGIN_SRC
                   / GENESIS_PLUGIN_SRC env var, or relies on the default
                   candidate-path search below.

Operator can override any auto-detected value by editing the YAML directly.
Configs reference these via `${name}` in mount strings; render layer
resolves them via `resolve_symbolic_mounts(mounts, host.paths)`.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sndr.model_configs.schema import SchemaError


@dataclass
class HostConfig:
    """Per-host resolved paths for symbolic mount references."""
    paths: dict[str, str] = field(default_factory=dict)

    def get(self, name: str) -> Optional[str]:
        return self.paths.get(name)


# ─── Path detection ──────────────────────────────────────────────────────


# Common locations to probe for each variable. Order = priority
# (first-found wins). User can override by editing host.yaml.
#
# generic-only defaults. Operator-specific
# `/nfs/genesis/...` and `~/Genesis_Project/...` paths removed — they
# leaked one operator's deployment topology as a "blessed" default.
# The probe list now contains only OS-conventional locations; sites
# that need custom paths set them via `host.yaml` or env vars.
_DEFAULT_MODELS_CANDIDATES = [
    "/srv/models",
    "/data/models",
    "/opt/models",
    "/var/lib/models",
    str(Path.home() / "models"),
    str(Path.home() / ".cache/genesis/models"),
]

_DEFAULT_TRITON_CACHE_CANDIDATES = [
    str(Path.home() / ".cache/triton"),
    str(Path.home() / ".sndr/cache/triton"),
    str(Path.home() / ".genesis/cache/triton"),  # legacy alias
    "/var/cache/genesis/triton",
]

_DEFAULT_COMPILE_CACHE_CANDIDATES = [
    str(Path.home() / ".cache/vllm/torch_compile_cache"),
    str(Path.home() / ".sndr/cache/vllm-compile"),
    str(Path.home() / ".genesis/cache/vllm-compile"),  # legacy alias
    "/var/cache/genesis/vllm-compile",
]

# builtin configs declare per-config cache
# subdirectories like `triton-cache-qwen3.6-35b-dflash` for bench reproducibility.
# `cache_root` is the parent of all per-config cache subdirs.
_DEFAULT_CACHE_ROOT_CANDIDATES = [
    str(Path.home() / ".sndr/cache"),
    str(Path.home() / ".cache/genesis"),
    str(Path.home() / ".genesis/cache"),  # legacy alias
    "/var/cache/genesis",
]

# sndr_src — the overlay PACKAGE dir, RO-bind-mounted into the container's
# dist-packages/sndr so `import sndr` resolves. v12 (2026-06-23): the package
# moved vllm/sndr_core/ -> sndr/. The UNIFIED ROOT BUG fix (2026-06-22) below
# correctly retargeted _DEFAULT_PLUGIN_SRC_CANDIDATES to the repo root but left
# THESE pointing at the retired vllm/sndr_core subdir, so the auto-probed RO
# mount source did not exist on a clean v12 checkout (operators had to set
# sndr_src / SNDR_CORE_SRC by hand). Point at the v12 `sndr/` package dir.
_DEFAULT_SNDR_SRC_CANDIDATES = [
    str(Path.home() / "genesis-vllm-patches/sndr"),
    "/opt/genesis-vllm-patches/sndr",
    str(Path.home() / ".genesis/genesis-vllm-patches/sndr"),
]

# Plugin source candidates — the directory the launch renderer bind-mounts
# at /plugin and (under SNDR_DEV_INSTALL_PLUGIN=1) pip-installs editable so
# its `vllm.general_plugins` entry-point registers IN the serving container.
#
# UNIFIED ROOT BUG fix (2026-06-22): these MUST point at the sndr REPO ROOT
# (the dir whose root `pyproject.toml` registers
# `genesis_v7 = "sndr.plugin:register"`), NOT the legacy
# `tools/genesis_vllm_plugin` SUBDIR. The subdir has no installable sndr
# package of its own, so installing it never registered the canonical
# in-process plugin and runtime monkey-patches (incl. g4_85) never fired
# in `vllm serve`. Installing the repo root writes the `sndr.plugin:register`
# entry-point into site-packages so `load_general_plugins()` loads it.
#
# Set SNDR_PLUGIN_SRC / GENESIS_PLUGIN_SRC env var to override.
_DEFAULT_PLUGIN_SRC_CANDIDATES = [
    str(Path.home() / "genesis-vllm-patches"),
    "/opt/genesis-vllm-patches",
    str(Path.home() / ".genesis/genesis-vllm-patches"),
]


# Operator-facing env knobs that pre-empt directory probing. When set
# and pointing at an existing absolute path, the value wins regardless
# of the auto-detection order. The aliases keep v7.x deployments
# working while encouraging the SNDR_* canonical names.
_ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "models_dir":    ("SNDR_MODELS_DIR", "GENESIS_MODELS_DIR"),
    "hf_cache":      ("SNDR_HF_CACHE", "HF_HOME", "HUGGINGFACE_HUB_CACHE"),
    "triton_cache":  ("SNDR_TRITON_CACHE", "GENESIS_TRITON_CACHE"),
    "compile_cache": ("SNDR_COMPILE_CACHE", "GENESIS_COMPILE_CACHE"),
    # `sndr_src` is the canonical key (v12). The legacy `GENESIS_SRC` env var
    # is retained in the tuple so v7.x/v11 deployments keep resolving; the
    # legacy `genesis_src` host.yaml key is aliased-in at load time
    # (see load_host_config).
    "sndr_src":      ("SNDR_CORE_SRC", "GENESIS_SRC"),
    "plugin_src":    ("SNDR_PLUGIN_SRC", "GENESIS_PLUGIN_SRC"),
    "cache_root":    ("SNDR_CACHE_ROOT", "GENESIS_CACHE_ROOT"),
}


def _env_lookup(var: str) -> Optional[str]:
    """Return an absolute, existing-directory env override for `var`,
    or None when no recognised env is set or the value points at a
    non-existent path.

    Validation is intentionally strict — symlinks resolve via `is_dir()`
    so an env that names a missing or relative path is ignored rather
    than silently producing a stub mount.
    """
    for name in _ENV_OVERRIDES.get(var, ()):
        raw = os.environ.get(name)
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.is_absolute() and path.is_dir():
            return str(path)
    return None


def detect_paths(
    models_candidates: Optional[list[str]] = None,
    hf_cache_candidates: Optional[list[str]] = None,
    triton_cache_candidates: Optional[list[str]] = None,
    compile_cache_candidates: Optional[list[str]] = None,
    sndr_src_candidates: Optional[list[str]] = None,
    plugin_src_candidates: Optional[list[str]] = None,
    cache_root_candidates: Optional[list[str]] = None,
    create_missing_caches: bool = False,
) -> dict[str, str]:
    """Auto-detect per-host paths by probing common locations.

    Returns dict {var_name: absolute_path} for variables where a candidate
    location was found. Variables with no found candidate are omitted —
    operator needs to fill them manually in host.yaml.

    Env overrides (`SNDR_MODELS_DIR`, `SNDR_HF_CACHE`, `SNDR_CACHE_ROOT`,
    etc., with `GENESIS_*` aliases) pre-empt the candidate lists when
    the named env points at an existing absolute path.

    Args:
        models_candidates: override default models_dir search list
        hf_cache_candidates: override hf_cache search (default = $HOME/.cache/huggingface)
        triton_cache_candidates: override triton_cache search
        compile_cache_candidates: override compile_cache search
        sndr_src_candidates: override sndr_src search
        plugin_src_candidates: override plugin_src search
        create_missing_caches: if True, mkdir cache dirs that don't yet exist
            at the FIRST candidate location (useful at install-time)
    """
    out: dict[str, str] = {}

    # models_dir — env override first, then candidates. We're not
    # creating model dirs auto-magically.
    env_path = _env_lookup("models_dir")
    if env_path:
        out["models_dir"] = env_path
    else:
        cands = models_candidates if models_candidates is not None else _DEFAULT_MODELS_CANDIDATES
        for c in cands:
            if Path(c).is_dir():
                out["models_dir"] = c
                break

    # hf_cache — env override first, then ~/.cache/huggingface default.
    env_path = _env_lookup("hf_cache")
    if env_path:
        out["hf_cache"] = env_path
    else:
        if hf_cache_candidates is None:
            hf_cache_candidates = [str(Path.home() / ".cache" / "huggingface")]
        for c in hf_cache_candidates:
            if Path(c).is_dir():
                out["hf_cache"] = c
                break
        if "hf_cache" not in out and create_missing_caches and hf_cache_candidates:
            first = Path(hf_cache_candidates[0])
            first.mkdir(parents=True, exist_ok=True)
            out["hf_cache"] = str(first)

    # triton_cache — env override → candidates → optional create.
    env_path = _env_lookup("triton_cache")
    if env_path:
        out["triton_cache"] = env_path
    else:
        cands = triton_cache_candidates if triton_cache_candidates is not None else _DEFAULT_TRITON_CACHE_CANDIDATES
        for c in cands:
            if Path(c).is_dir():
                out["triton_cache"] = c
                break
        if "triton_cache" not in out and create_missing_caches and cands:
            first = Path(cands[0])
            first.mkdir(parents=True, exist_ok=True)
            out["triton_cache"] = str(first)

    # compile_cache — env override → candidates → optional create.
    env_path = _env_lookup("compile_cache")
    if env_path:
        out["compile_cache"] = env_path
    else:
        cands = compile_cache_candidates if compile_cache_candidates is not None else _DEFAULT_COMPILE_CACHE_CANDIDATES
        for c in cands:
            if Path(c).is_dir():
                out["compile_cache"] = c
                break
        if "compile_cache" not in out and create_missing_caches and cands:
            first = Path(cands[0])
            first.mkdir(parents=True, exist_ok=True)
            out["compile_cache"] = str(first)

    # sndr_src — env override → candidates. RO mount source, never auto-created.
    env_path = _env_lookup("sndr_src")
    if env_path:
        out["sndr_src"] = env_path
    else:
        cands = sndr_src_candidates if sndr_src_candidates is not None else _DEFAULT_SNDR_SRC_CANDIDATES
        for c in cands:
            if Path(c).is_dir():
                out["sndr_src"] = c
                break

    # plugin_src — env override → candidates. RO mount source.
    env_path = _env_lookup("plugin_src")
    if env_path:
        out["plugin_src"] = env_path
    else:
        cands = plugin_src_candidates if plugin_src_candidates is not None else _DEFAULT_PLUGIN_SRC_CANDIDATES
        for c in cands:
            if Path(c).is_dir():
                out["plugin_src"] = c
                break

    # cache_root — env override → candidates → optional create.
    env_path = _env_lookup("cache_root")
    if env_path:
        out["cache_root"] = env_path
    else:
        cands = cache_root_candidates if cache_root_candidates is not None else _DEFAULT_CACHE_ROOT_CANDIDATES
        for c in cands:
            if Path(c).is_dir():
                out["cache_root"] = c
                break
        if "cache_root" not in out and create_missing_caches and cands:
            first = Path(cands[0])
            first.mkdir(parents=True, exist_ok=True)
            out["cache_root"] = str(first)

    return out


# ─── YAML I/O ────────────────────────────────────────────────────────────


def _default_host_yaml_path() -> Path:
    """Resolve operator-config root + /host.yaml.

    P1-5 (audit 2026-05-08): canonical env is now `SNDR_HOME`,
    legacy alias `GENESIS_HOME` honored for back-compat. Default
    operator root is `~/.sndr` (was `~/.genesis`); `~/.genesis/host.yaml`
    still resolves IF it's the only one present, so v7.x operators
    aren't forced to migrate immediately.

    Resolution order:
      1. $SNDR_HOME/host.yaml (canonical)
      2. $GENESIS_HOME/host.yaml (legacy alias)
      3. ~/.sndr/host.yaml (canonical default)
      4. ~/.genesis/host.yaml (legacy default — fallback only)
    """
    sndr_home = os.environ.get("SNDR_HOME") or os.environ.get("GENESIS_HOME")
    if sndr_home:
        return Path(sndr_home) / "host.yaml"
    sndr_default = Path.home() / ".sndr" / "host.yaml"
    if sndr_default.is_file():
        return sndr_default
    legacy_default = Path.home() / ".genesis" / "host.yaml"
    if legacy_default.is_file():
        return legacy_default
    # Neither exists — return the canonical path so write-mode callers
    # create the new layout, not the legacy one.
    return sndr_default


def load_host_config(path: Optional[Path] = None) -> HostConfig:
    """Load host config from YAML. Returns empty HostConfig if file absent."""
    p = path if path is not None else _default_host_yaml_path()
    if not p.is_file():
        return HostConfig()
    try:
        import yaml
    except ImportError:
        raise SchemaError(
            "host.py requires PyYAML to load host.yaml. Install pyyaml."
        )
    data = yaml.safe_load(p.read_text()) or {}
    if not isinstance(data, dict):
        raise SchemaError(
            f"host.yaml at {p} must be a mapping, got {type(data).__name__}"
        )
    # UX warn: detect the flat-schema mistake where path-like keys live
    # at the top level instead of under `paths:`. Silent fallback to the
    # default candidate list otherwise (e.g. /opt/models) cost ~30 min
    # of operator debugging on 2026-05-22. Warning only — no behaviour
    # change, no auto-migration.
    if "paths" not in data:
        canonical_keys = set(_ENV_OVERRIDES.keys())
        leaked = sorted(set(data.keys()) & canonical_keys)
        if leaked:
            print(
                f"[host.yaml] WARN: top-level path-like key(s) "
                f"{leaked} found in {p} but no 'paths:' mapping. "
                f"These keys are IGNORED; the loader falls back to "
                f"default candidate directories. Wrap them under a "
                f"top-level 'paths:' block to take effect, e.g.:\n"
                f"  paths:\n"
                f"    models_dir: /your/models/dir\n"
                f"    ...",
                file=sys.stderr,
            )
    paths = data.get("paths", {})
    if not isinstance(paths, dict):
        raise SchemaError(
            f"host.yaml.paths at {p} must be a mapping, got {type(paths).__name__}"
        )
    paths = _normalize_legacy_keys(dict(paths))
    return HostConfig(paths=paths)


def _normalize_legacy_keys(paths: dict[str, str]) -> dict[str, str]:
    """Alias deprecated host.yaml path keys onto their canonical names.

    v12 renamed the host source-checkout mount var `genesis_src` → `sndr_src`.
    Existing operator host.yaml files still use the legacy `genesis_src:` key,
    so alias it in: if `genesis_src` is present but `sndr_src` is not, copy the
    value under `sndr_src` so downstream consumers resolve under the canonical
    key. The legacy key is left in place so nothing that still reads it breaks.
    """
    if "genesis_src" in paths and "sndr_src" not in paths:
        paths["sndr_src"] = paths["genesis_src"]
    return paths


def save_host_config(hc: HostConfig, path: Optional[Path] = None) -> Path:
    """Write host config as YAML. Creates parent dir if needed."""
    p = path if path is not None else _default_host_yaml_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
    except ImportError:
        # Manual YAML emit for the simple case (paths: {k: v}) — no deps
        lines = ["# Genesis host config — auto-detected by install / first-run.",
                 "# Override values by editing this file.",
                 "paths:"]
        for k, v in sorted(hc.paths.items()):
            lines.append(f"  {k}: {v}")
        p.write_text("\n".join(lines) + "\n")
        return p
    data = {"paths": dict(hc.paths)}
    p.write_text(yaml.safe_dump(data, sort_keys=True, default_flow_style=False))
    return p


def detect_and_save(
    path: Optional[Path] = None,
    create_missing_caches: bool = False,
) -> tuple[HostConfig, Path]:
    """Detect paths + save to host.yaml. Convenience for install/first-run."""
    paths = detect_paths(create_missing_caches=create_missing_caches)
    hc = HostConfig(paths=paths)
    saved_path = save_host_config(hc, path)
    return hc, saved_path
