# SPDX-License-Identifier: Apache-2.0
"""Genesis model_configs — vetted, reproducible launch configurations.

A model_config is the **single source of truth** for one (model × hw ×
workload) combination. It captures EVERYTHING needed to:

  1. Launch the model (vLLM args + Genesis env + system env + Docker)
  2. Verify it works (`reference_metrics` + `verify_tolerances`)
  3. Track provenance (`verified_on`, `last_validated`, pin info)

Layout:
    builtin/    — ships with the patcher; modified only via PR review
    community/  — community-contributed verified configs
    user/       — operator's local configs (loaded if present, gitignored)

Use (V2 path, canonical — V1 monolithic preset tier 100% retired
2026-06-01 after the Phase 10 sunset cascade):
    from sndr.model_configs.registry_v2 import load_alias
    cfg = load_alias('prod-qwen3.6-35b-balanced')  # V2 preset alias
    print(cfg.to_launch_script())

The legacy V1 `get()` path still resolves any operator-local YAMLs
under `community/` or `user/` tiers and emits a DeprecationWarning
for visibility; the builtin/ V1 layer is gone.

CLI: `python3 -m sndr.compat.model_config_cli list/show/render/launch/verify`
"""
from .schema import (
    ModelConfig,
    ReferenceMetrics,
    VerifyTolerances,
    HardwareSpec,
    SpecDecodeConfig,
    DockerConfig,
    SchemaError,
    load_yaml,
    dump_yaml,
    validate,
)
from .registry import load_all, get, list_keys

__all__ = [
    "ModelConfig", "ReferenceMetrics", "VerifyTolerances",
    "HardwareSpec", "SpecDecodeConfig", "DockerConfig",
    "SchemaError", "load_yaml", "dump_yaml", "validate",
    "load_all", "get", "list_keys",
]
