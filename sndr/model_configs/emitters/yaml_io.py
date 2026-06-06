# SPDX-License-Identifier: Apache-2.0
"""YAML I/O — ``dump_yaml`` / ``load_yaml`` / dict reconstruction.

Pure functions over ``ModelConfig``. Previously top-level helpers in
``model_configs/schema.py``; relocated in M.5.2. Behaviour unchanged.

The reconstruction path (``_from_plain_dict``) intentionally remains
defensive against ``ReferenceMetrics`` schema drift (audit-trail YAML
fields like ``wave8_delta_pct_*`` accumulate over time and should not
break load).
"""
from __future__ import annotations

import logging
from dataclasses import asdict, fields
from typing import TYPE_CHECKING

from ..types import (
    ArtifactCache,
    ArtifactModel,
    Artifacts,
    BootstrapConfig,
    CacheConfig,
    CacheTier,
    ConfigConstraints,
    DeploymentConfig,
    DockerConfig,
    GpuTuningConfig,
    HardwareSpec,
    KubernetesConfig,
    ObservabilityConfig,
    OffloadConfig,
    OverridesPolicy,
    PackageSource,
    PackageSources,
    PackageVersions,
    ProxmoxConfig,
    ReferenceMetrics,
    RiskScore,
    SchemaError,
    ServiceConfig,
    SpecDecodeConfig,
    UpstreamPinPolicy,
    VerifyTolerances,
)

if TYPE_CHECKING:
    from ..schema import ModelConfig

# Preserve historical logger name so operator filter rules keep matching.
log = logging.getLogger("genesis.model_configs.schema")


def dump_yaml(cfg: "ModelConfig") -> str:
    """Serialize ModelConfig → YAML string."""
    import yaml
    cfg.validate()
    d = to_plain_dict(cfg)
    return yaml.safe_dump(d, sort_keys=False, allow_unicode=True,
                          default_flow_style=False)


def load_yaml(text: str) -> "ModelConfig":
    """Parse YAML string → ModelConfig with full validation."""
    import yaml
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise SchemaError("YAML must be a mapping at top level")
    return from_plain_dict(raw)


def validate_cfg(cfg: "ModelConfig") -> "ModelConfig":
    """Validate ModelConfig in-place; raise SchemaError on issues.

    Returns the validated config for chainable use.
    """
    cfg.validate()
    return cfg


def to_plain_dict(cfg: "ModelConfig") -> dict:
    """Render ModelConfig as a plain ``dict`` (YAML-safe)."""
    return asdict(cfg)


def from_plain_dict(d: dict) -> "ModelConfig":
    """Reconstruct ModelConfig from plain dict (post-YAML-load)."""
    # Lazy import: keep this module cheap to load on hosts that only
    # need the emitter side without YAML round-trip.
    from ..schema import ModelConfig

    known = {f.name for f in fields(ModelConfig)}
    unknown = set(d.keys()) - known
    if unknown:
        raise SchemaError(
            f"unknown field(s) in ModelConfig YAML: {sorted(unknown)}. "
            f"Known: {sorted(known)}"
        )

    # Sub-component reconstruction
    if "hardware" in d and isinstance(d["hardware"], dict):
        d["hardware"] = HardwareSpec(**d["hardware"])
    if "spec_decode" in d and isinstance(d["spec_decode"], dict):
        d["spec_decode"] = SpecDecodeConfig(**d["spec_decode"])
    if "docker" in d and isinstance(d["docker"], dict):
        d["docker"] = DockerConfig(**d["docker"])
    if "reference_metrics" in d and isinstance(d["reference_metrics"], dict):
        # Defensive: filter unknown fields with a warning rather than
        # crash. Transient audit-trail fields (e.g. wave8_delta_pct_*)
        # accumulate in YAMLs as human-readable provenance and shouldn't
        # block PN95 lazy init or `verify` loads.
        rm_known = {f.name for f in fields(ReferenceMetrics)}
        rm_raw = d["reference_metrics"]
        rm_unknown = set(rm_raw.keys()) - rm_known
        if rm_unknown:
            log.warning(
                "ReferenceMetrics: ignoring unknown YAML field(s) %s "
                "(treated as audit-trail metadata, not loaded into dataclass). "
                "If a field is real schema, add it to ReferenceMetrics.",
                sorted(rm_unknown),
            )
        d["reference_metrics"] = ReferenceMetrics(
            **{k: v for k, v in rm_raw.items() if k in rm_known}
        )
    if "verify_tolerances" in d and isinstance(d["verify_tolerances"], dict):
        d["verify_tolerances"] = VerifyTolerances(**d["verify_tolerances"])
    if "constraints" in d and isinstance(d["constraints"], dict):
        d["constraints"] = ConfigConstraints(**d["constraints"])
    if "risk_score" in d and isinstance(d["risk_score"], dict):
        d["risk_score"] = RiskScore(**d["risk_score"])
    if "cache_config" in d and isinstance(d["cache_config"], dict):
        cc = dict(d["cache_config"])
        # Path C: reconstruct nested CacheTier list
        if "tiers" in cc and isinstance(cc["tiers"], list):
            cc["tiers"] = [
                CacheTier(**t) if isinstance(t, dict) else t
                for t in cc["tiers"]
            ]
        d["cache_config"] = CacheConfig(**cc)
    if "package_versions" in d and isinstance(d["package_versions"], dict):
        d["package_versions"] = PackageVersions(**d["package_versions"])
    if "upstream" in d and isinstance(d["upstream"], dict):
        d["upstream"] = UpstreamPinPolicy(**d["upstream"])
    if "overrides" in d and isinstance(d["overrides"], dict):
        d["overrides"] = OverridesPolicy(**d["overrides"])
    if "offload" in d and isinstance(d["offload"], dict):
        d["offload"] = OffloadConfig(**d["offload"])
    if "service" in d and isinstance(d["service"], dict):
        d["service"] = ServiceConfig(**d["service"])
    if "gpu_tuning" in d and isinstance(d["gpu_tuning"], dict):
        d["gpu_tuning"] = GpuTuningConfig(**d["gpu_tuning"])
    if "observability" in d and isinstance(d["observability"], dict):
        d["observability"] = ObservabilityConfig(**d["observability"])
    if "kubernetes" in d and isinstance(d["kubernetes"], dict):
        d["kubernetes"] = KubernetesConfig(**d["kubernetes"])
    if "proxmox" in d and isinstance(d["proxmox"], dict):
        d["proxmox"] = ProxmoxConfig(**d["proxmox"])
    if "bootstrap" in d and isinstance(d["bootstrap"], dict):
        d["bootstrap"] = BootstrapConfig(**d["bootstrap"])
    if "package_sources" in d and isinstance(d["package_sources"], dict):
        ps = dict(d["package_sources"])
        if "sources" in ps and isinstance(ps["sources"], list):
            ps["sources"] = [
                PackageSource(**s) if isinstance(s, dict) else s
                for s in ps["sources"]
            ]
        d["package_sources"] = PackageSources(**ps)
    if "artifacts" in d and isinstance(d["artifacts"], dict):
        a = dict(d["artifacts"])
        if "models" in a and isinstance(a["models"], list):
            a["models"] = [
                ArtifactModel(**m) if isinstance(m, dict) else m
                for m in a["models"]
            ]
        if "caches" in a and isinstance(a["caches"], list):
            a["caches"] = [
                ArtifactCache(**c) if isinstance(c, dict) else c
                for c in a["caches"]
            ]
        d["artifacts"] = Artifacts(**a)
    if "deploy" in d and isinstance(d["deploy"], dict):
        # W-runtime 2026-05-06: deploy block reconstruction
        # Filter to known DeploymentConfig fields (skip KNOWN_RUNTIMES tuple)
        dep_fields = {f.name for f in fields(DeploymentConfig)}
        dep_data = {k: v for k, v in d["deploy"].items() if k in dep_fields}
        d["deploy"] = DeploymentConfig(**dep_data)

    cfg = ModelConfig(**d)
    cfg.validate()
    return cfg
