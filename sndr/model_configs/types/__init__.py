# SPDX-License-Identifier: Apache-2.0
"""``model_configs.types`` — sub-component dataclasses for ModelConfig.

M.5.1 (2026-05-27): relocated from the 2768-LOC ``model_configs/schema.py``
monolith into thematic modules so each dataclass lives next to a focused
test surface. ``model_configs/schema.py`` re-exports every public name
listed in ``__all__`` below, so the historical import paths
(``from sndr.model_configs.schema import HardwareSpec`` etc.)
continue to resolve.

Subsequent M.5 phases (M.5.2: emitter extraction, M.5.3: ModelConfig
slim-down) reuse this package for their own pure-data surfaces.
"""
from __future__ import annotations

from ._base import SchemaError
from .artifacts import (
    ArtifactCache,
    ArtifactModel,
    Artifacts,
    PatchAttribution,
    _PATCH_ROLES,
)
from .cache import CacheConfig, CacheTier, OffloadConfig
from .compatibility import (
    COMPATIBILITY_MATRIX,
    CompatibilityMatrix,
    CompatibilityRule,
    _is_qwen_next,
    _kv_cache_dtype,
    _spec_decode_method,
    _uses_hybrid_gdn,
)
from .docker import DeploymentConfig, DockerConfig, resolve_symbolic_mounts
from .hardware import HardwareSpec
from .metrics import (
    ConfigConstraints,
    ReferenceMetrics,
    RiskScore,
    VerifyTolerances,
)
from .packages import (
    OverridesPolicy,
    PackageSource,
    PackageSources,
    PackageVersions,
    UpstreamPinPolicy,
)
from .runtime import (
    BootstrapConfig,
    GpuTuningConfig,
    KubernetesConfig,
    ObservabilityConfig,
    ProxmoxConfig,
    ServiceConfig,
)
from .spec_decode import SpecDecodeConfig

__all__ = (
    "SchemaError",
    # hardware
    "HardwareSpec",
    # spec_decode
    "SpecDecodeConfig",
    # docker
    "DockerConfig",
    "DeploymentConfig",
    "resolve_symbolic_mounts",
    # runtime
    "KubernetesConfig",
    "ProxmoxConfig",
    "BootstrapConfig",
    "GpuTuningConfig",
    "ObservabilityConfig",
    "ServiceConfig",
    # packages
    "PackageSource",
    "PackageSources",
    "PackageVersions",
    "UpstreamPinPolicy",
    "OverridesPolicy",
    # cache
    "CacheTier",
    "CacheConfig",
    "OffloadConfig",
    # metrics
    "ReferenceMetrics",
    "VerifyTolerances",
    "ConfigConstraints",
    "RiskScore",
    # artifacts
    "ArtifactModel",
    "ArtifactCache",
    "Artifacts",
    "PatchAttribution",
    "_PATCH_ROLES",
    # compatibility
    "CompatibilityRule",
    "CompatibilityMatrix",
    "COMPATIBILITY_MATRIX",
    "_uses_hybrid_gdn",
    "_spec_decode_method",
    "_kv_cache_dtype",
    "_is_qwen_next",
)
