# SPDX-License-Identifier: Apache-2.0
"""CONFIG-UX.5.1 — generated catalog row schema.

Catalog is a **derived API, not a new source of truth** (operator-locked
architectural principle, CONFIG-UX.5.1 scope). These dataclasses describe
the shape of `build/config_catalog/config_catalog.json` rows — one row
per operator-visible artifact (preset / profile / model / hardware /
baseline).

Source of truth remains:
  - `vllm/sndr_core/model_configs/builtin/**/*.yaml` (V2 YAML tree)
  - `tests/integration/baselines/*.json` (committed public bench data)

This module declares the schema only — no I/O, no parsing.
`scripts/generate_config_catalog.py` reads the source tree and produces
rows; `scripts/audit_generated_config_catalog.py` checks the derived
catalog against fresh regeneration to detect drift.

Redaction discipline (operator-locked):
  - Generated rows MUST NOT include `sndr_private/` paths
  - MUST NOT include local absolute paths (`/Users/...`, `/home/...`)
  - Private evidence_refs are replaced with `{redacted: true, ...}`
    markers — the raw path is never serialised into the public catalog

Schema versioning: bump `SCHEMA_VERSION` when adding/removing fields.
Backward-compat consumers should accept unknown future fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


SCHEMA_VERSION: int = 1


# Row types are exhaustive: one CatalogRow subtype per source-tree
# artifact category. Operator decision §10.1 — add row types in a
# minor schema bump (e.g. `evidence_ledger`, `bench_run`) without
# breaking v1 consumers.
RowType = Literal["preset", "profile", "model", "hardware", "baseline"]
ROW_TYPES: tuple[str, ...] = (
    "preset", "profile", "model", "hardware", "baseline",
)


# Baseline match quality per operator §10.1: not strict-equality
# because baseline JSONs match model families more loosely than
# preset/config blocks.
MatchQuality = Literal["exact_preset", "model_only", "family_only", "none"]
MATCH_QUALITIES: tuple[str, ...] = (
    "exact_preset", "model_only", "family_only", "none",
)


# ─── Redacted evidence marker ───────────────────────────────────────────────


@dataclass(frozen=True)
class RedactedEvidenceRef:
    """Replacement for private evidence_ref in generated catalog output.

    Generator emits this in place of the raw path when
    `evidence_visibility=private` so the public JSON catalog never
    exposes `sndr_private/...` paths or local filesystem locations.
    """
    type: str                         # bench / smoke / structured_eval / ...
    redacted: bool = True             # always True in serialised form
    visibility: str = "private"
    note: str = "private evidence — not exposed in generated catalog"


# ─── Common base ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CatalogRowBase:
    """Shared fields across all row types.

    Operator-locked field set (CONFIG-UX.5.1):
      schema_version, row_type, id, source_path, status, family,
      tags, updated_from_git_commit, payload (type-specific dataclass).

    Per-type rows extend this base with their typed payload.
    """
    schema_version: int
    row_type: RowType
    id: str                           # primary key per row_type
    source_path: str                  # repo-relative path to source file
    source_sha256: str                # sha256 of source file (drift detection)
    status: Optional[str]             # status field (preset.card.status, profile.status, ...)
    family: Optional[str]             # routing_family (preset) / model_family (model) / etc.
    tags: list[str]                   # free-form labels for downstream queries
    updated_from_git_commit: Optional[str]  # short SHA last-touching source
    generated_at: str                 # ISO timestamp (UTC) of catalog build


# ─── Preset row ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PresetRow(CatalogRowBase):
    """Preset → composed runtime + card metadata.

    Operator-required fields (CONFIG-UX.5.1):
      composed_sha256, card.status, workload_allow/deny,
      primary_metric, fallback_preset, default_for_family.
    """
    model_id: str
    hardware_id: str
    profile_id: Optional[str]
    composed_key: str
    composed_sha256: str              # sha256 of dump_yaml(composed_cfg) — golden-baseline integration
    has_card: bool
    # Flattened card surface (subset per operator §10.1)
    card_title: Optional[str]
    card_status: Optional[str]
    card_audience: Optional[str]
    card_mode: Optional[str]
    card_workload_allow: list[str]
    card_workload_deny: list[str]
    card_K: Optional[int]
    card_routing_family: Optional[str]
    card_default_for_family: bool
    card_fallback_preset: Optional[str]
    card_primary_metric_kind: Optional[str]
    card_primary_metric_value: Optional[float]
    card_evidence_visibility: Optional[str]
    card_evidence_ref_count: int
    # Evidence refs: public paths verbatim, private replaced with RedactedEvidenceRef
    card_evidence_refs: list[dict]


# ─── Profile row ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProfileRow(CatalogRowBase):
    """Profile → sizing + override_policy + Class-4 cleanliness.

    Operator-required fields (CONFIG-UX.5.1):
      override_policy, sizing_override, class4_clean.
    """
    parent_model: str
    role: Optional[str]
    # Sizing override (full flattened)
    sizing_max_model_len: Optional[int]
    sizing_max_num_seqs: Optional[int]
    sizing_max_num_batched_tokens: Optional[int]
    sizing_gpu_memory_utilization: Optional[float]
    sizing_enable_chunked_prefill: Optional[bool]
    sizing_enforce_eager: Optional[bool]
    # Override policy (operator-required at .5.1)
    has_override_policy: bool
    override_class: Optional[str]
    override_reason: Optional[str]
    override_evidence_ref_count: int
    override_evidence_visibility: Optional[str]
    override_expires_at: Optional[str]
    override_allowed_to_exceed_hardware_default: bool
    # Class-4 audit verdict at generation time
    class4_clean: bool                # True if 0 Class-4 violations on this profile
    # Patches delta summary (counts only — full delta lives in YAML)
    patches_enable_count: int
    patches_disable_count: int
    patches_override_count: int


# ─── Model row ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelRow(CatalogRowBase):
    """Model → identity + capabilities + KV/spec-decode summary."""
    title: str
    quantization: Optional[str]
    kv_cache_dtype: Optional[str]
    spec_decode_method: Optional[str]
    spec_decode_K: Optional[int]
    enable_auto_tool_choice: Optional[bool]
    tool_call_parser: Optional[str]
    vllm_pin_required: Optional[str]
    genesis_pin_min: Optional[str]
    enabled_patches_count: int


# ─── Hardware row ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HardwareRow(CatalogRowBase):
    """Hardware → rig identity + default sizing + runtime block summary."""
    title: str
    n_gpus: int
    gpu_match_keys: list[str]
    min_vram_per_gpu_mib: Optional[int]
    cuda_capability_min: Optional[list[int]]
    sizing_max_model_len: Optional[int]
    sizing_max_num_seqs: Optional[int]
    sizing_gpu_memory_utilization: Optional[float]
    runtime_default: Optional[str]


# ─── Baseline row ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BaselineRow(CatalogRowBase):
    """Public bench baseline → match quality vs corpus.

    Operator-locked (CONFIG-UX.5.1 §2.3): baselines link to model
    families via `match_quality` enum, not strict-equality. A baseline
    `qwen3.6-27b` model field matches multiple presets at the
    `family_only` level (TQ k8v4, DFlash, FP8 KV variants all share
    the qwen3.6-27b family); only when the baseline carries
    preset-id-or-config-block evidence is it `exact_preset` match.
    """
    bench_model: str                  # raw `model` field from baseline JSON
    bench_vllm_pin: Optional[str]
    bench_ctx: Optional[int]
    bench_max_tokens: Optional[int]
    bench_prompts_set: Optional[str]
    bench_runs: Optional[int]
    # Match quality vs corpus presets/models
    match_quality: MatchQuality
    matched_model_ids: list[str]      # which V2 model_ids this baseline references
    matched_preset_ids: list[str]     # which V2 preset_ids cite this baseline


# ─── Public API ─────────────────────────────────────────────────────────────


def is_private_visibility(visibility: Optional[str]) -> bool:
    """Helper: is this evidence visibility considered private?

    `mixed` evidence visibility is treated as PARTIALLY public: per-ref
    visibility decides; generator iterates and redacts only the
    `private` ones.
    """
    return visibility == "private"


def is_redactable_path(path: str) -> bool:
    """Helper: would this path leak private/local-machine info if
    written to a public catalog JSON?

    Used by the generator to enforce redaction discipline. Patterns
    that trigger redaction:
      - starts with `sndr_private/`
      - absolute local paths (`/Users/...`, `/home/...`, `/tmp/...`)
    Public-safe forms (NOT redacted):
      - repo-relative paths (`tests/integration/baselines/...`,
        `docs/...`, `vllm/...`)
      - `external://...` schemes
    """
    if not path:
        return False
    if path.startswith("sndr_private/"):
        return True
    # Absolute local paths
    for prefix in ("/Users/", "/home/", "/tmp/", "/var/"):
        if path.startswith(prefix):
            return True
    return False


__all__ = [
    "SCHEMA_VERSION",
    "ROW_TYPES", "RowType",
    "MATCH_QUALITIES", "MatchQuality",
    "RedactedEvidenceRef",
    "CatalogRowBase",
    "PresetRow", "ProfileRow", "ModelRow", "HardwareRow", "BaselineRow",
    "is_private_visibility", "is_redactable_path",
]
