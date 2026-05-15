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
  genesis_src    — checkout of genesis-vllm-patches/vllm/sndr_core (RO mount source). Variable name kept "genesis_src" for legacy compat with existing host.yaml; physical path points at sndr_core after v11 (_genesis dir removed 2026-05-08).
  plugin_src     — operator-side path to the optional Genesis vllm-plugin
                   pip package (separate from sndr_core). Set via
                   SNDR_PLUGIN_SRC / GENESIS_PLUGIN_SRC env var, or relies
                   on the default candidate-path search below.

Operator can override any auto-detected value by editing the YAML directly.
Configs reference these via `${name}` in mount strings; render layer
resolves them via `resolve_symbolic_mounts(mounts, host.paths)`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from vllm.sndr_core.model_configs.schema import SchemaError


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
# subdirectories like `triton-cache-35b-dflash` for bench reproducibility.
# `cache_root` is the parent of all per-config cache subdirs.
_DEFAULT_CACHE_ROOT_CANDIDATES = [
    str(Path.home() / ".sndr/cache"),
    str(Path.home() / ".cache/genesis"),
    str(Path.home() / ".genesis/cache"),  # legacy alias
    "/var/cache/genesis",
]

_DEFAULT_GENESIS_SRC_CANDIDATES = [
    str(Path.home() / "genesis-vllm-patches/vllm/sndr_core"),
    "/opt/genesis-vllm-patches/vllm/sndr_core",
    str(Path.home() / ".genesis/genesis-vllm-patches/vllm/sndr_core"),
]

# Plugin source candidates. Operator-side path — sndr_core does NOT
# import this package at Python level (the canonical plugin entry-point
# is vllm.sndr_core.plugin:register). These candidates are only used
# by Docker mount renderers when the operator wants to bind-mount the
# plugin source into the container for editable installs.
# Set SNDR_PLUGIN_SRC / GENESIS_PLUGIN_SRC env var to override.
_DEFAULT_PLUGIN_SRC_CANDIDATES = [
    str(Path.home() / "genesis-vllm-patches/tools/genesis_vllm_plugin"),
    "/opt/genesis-vllm-patches/tools/genesis_vllm_plugin",
    str(Path.home() / ".genesis/genesis-vllm-patches/tools/genesis_vllm_plugin"),
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
    "genesis_src":   ("SNDR_CORE_SRC", "GENESIS_SRC"),
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
    genesis_src_candidates: Optional[list[str]] = None,
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
        genesis_src_candidates: override genesis_src search
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

    # genesis_src — env override → candidates. RO mount source, never auto-created.
    env_path = _env_lookup("genesis_src")
    if env_path:
        out["genesis_src"] = env_path
    else:
        cands = genesis_src_candidates if genesis_src_candidates is not None else _DEFAULT_GENESIS_SRC_CANDIDATES
        for c in cands:
            if Path(c).is_dir():
                out["genesis_src"] = c
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
    paths = data.get("paths", {})
    if not isinstance(paths, dict):
        raise SchemaError(
            f"host.yaml.paths at {p} must be a mapping, got {type(paths).__name__}"
        )
    return HostConfig(paths=dict(paths))


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
