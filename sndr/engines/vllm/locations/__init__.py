# SPDX-License-Identifier: Apache-2.0
"""SNDR Core file location resolution — public API.

Renamed 2026-05-11 from `paths/` per audit P-01: the old name was
overloaded (project paths + vllm targets + resolver shims mixed).
`locations/` captures the unified semantic — "where files live" both
inside Genesis and inside the patched vllm install.

This subpackage centralizes all path resolution logic:

  - `vllm_install_root()` — discovers the installed vllm package directory.
  - `resolve_vllm_file(rel_path)` — converts a vllm-relative path string
    into an absolute Path object pointing at the file inside the installed
    vllm package. Returns None if vllm is not discoverable.
  - `vllm_targets` module — single source of truth for ALL vllm engine
    file paths that SNDR Core integrations modify. 63 constants.
    (Renamed from `engine_targets` 2026-05-11 — `vllm_` prefix is more
    explicit since these are vllm-side paths, not engine-agnostic.)
  - `project_paths` module — paths INSIDE the Genesis/SNDR project
    itself (wiring_dir, manifest_dir, model_configs_*_dir, etc.).
    (Renamed from `sndr_paths` 2026-05-11.)

Stage 2 design (Sander Q3 mixed approach 2026-05-07):

    Top-level `locations/vllm_targets.py` is the canonical registry.
    Per-subsystem helper re-exports (e.g. `attention/turboquant/_paths.py`)
    are added ONLY when there's clear utility — they import from the
    top-level registry, never duplicate path strings.

Why centralize:
    Before Stage 2, each of 102 integration files contained its own
    `resolve_vllm_file("tool_parsers/qwen3coder_tool_parser.py")` string.
    When upstream renamed (e.g. `tool_parsers/` → `tool_parsing/` in
    a future vllm version), updating required grep+replace across all
    102 files. Now ONE constant in `vllm_targets.py` propagates the
    rename + boot-time validation flags any orphans.

Migration status:
    Stage 2 (CURRENT) — registry created with all 63 paths. Existing
        integrations still hard-code their paths (zero functional change).
    Stage 6        — integrations migrate to import from `vllm_targets`.
    Stage 13       — boot-time audit becomes mandatory; orphan paths
        block release.

Back-compat: `from vllm.sndr_core.paths import ...` continues to work
via lazy alias in `vllm/sndr_core/__init__.py.__getattr__`. Same for
`engine_targets` and `sndr_paths` symbol aliases below.
"""
from . import vllm_targets  # noqa: F401  (vllm engine paths — Stage 2)
from . import project_paths  # noqa: F401  (SNDR-internal paths — v9.0)
from .resolver import resolve_vllm_file  # noqa: F401
from .vllm_install import vllm_install_root  # noqa: F401

# Back-compat aliases — old import names continue to resolve
engine_targets = vllm_targets  # legacy alias (pre-2026-05-11)
sndr_paths = project_paths  # legacy alias (pre-2026-05-11)

__all__ = [
    "vllm_targets",
    "project_paths",
    "resolve_vllm_file",
    "vllm_install_root",
    # Back-compat aliases:
    "engine_targets",
    "sndr_paths",
]
