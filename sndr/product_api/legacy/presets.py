# SPDX-License-Identifier: Apache-2.0
"""Pure-data Product API for V2 preset catalog and recommendations.

The CLI renders this information for humans; GUI/web callers need the same
operator-product data as typed results, without stdout, argparse, or
``SystemExit``. This module keeps imports torch-free and confines V2 registry
loading to function bodies.
"""
from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from typing import Any, Optional


STATUS_RANK: dict[str, int] = {
    "production": 0,
    "production_candidate": 1,
    "internal_validated": 2,
    "bench_pending": 3,
    "experimental": 4,
    "qa": 5,
    "example": 6,
    "historical": 7,
    "tombstone": 8,
}


class PresetProductAPIError(Exception):
    """Base class for preset Product API failures."""


class PresetNotFoundError(PresetProductAPIError):
    """Raised when a preset cannot be loaded from the V2 registry."""

    def __init__(self, preset_id: str, reason: str):
        self.preset_id = preset_id
        self.reason = reason
        super().__init__(f"preset {preset_id!r}: {reason}")


class PresetComposeError(PresetProductAPIError):
    """Raised when a preset loads but cannot compose to runtime config."""

    def __init__(self, preset_id: str, reason: str):
        self.preset_id = preset_id
        self.reason = reason
        super().__init__(f"compose failed for preset {preset_id!r}: {reason}")


class UnknownWorkloadError(PresetProductAPIError):
    """Raised when recommend receives a non-canonical workload string."""

    def __init__(self, workload: str):
        self.workload = workload
        super().__init__(
            f"--workload {workload!r} is not in KNOWN_WORKLOADS and is not "
            "a valid `custom:<slug>` form"
        )


@dataclass(frozen=True)
class PresetLoadIssue:
    """One non-fatal preset load issue collected during corpus scans."""

    preset_id: str
    error_type: str
    message: str


@dataclass(frozen=True)
class PresetRecord:
    """JSON-safe operator-facing preset record."""

    id: str
    model: str
    hardware: str
    profile: Optional[str]
    runtime: Optional[str]
    has_card: bool
    card: dict[str, Any]


@dataclass(frozen=True)
class PresetListResult:
    """Filtered preset catalog result."""

    filters: dict[str, Optional[str]]
    matched: int
    total: int
    presets: tuple[PresetRecord, ...]
    load_errors: tuple[PresetLoadIssue, ...] = ()


@dataclass(frozen=True)
class PresetRecommendation:
    """One ranked recommendation."""

    id: str
    rank: int
    model: str
    hardware: str
    profile: Optional[str]
    runtime: Optional[str]
    card: dict[str, Any]


@dataclass(frozen=True)
class PresetRecommendResult:
    """Recommendation result for a workload query."""

    query: dict[str, Any]
    results: tuple[PresetRecommendation, ...]
    total_matches: int
    total_candidates: int


@dataclass(frozen=True)
class PresetExplainResult:
    """Machine-readable preset explain payload."""

    id: str
    card: dict[str, Any]
    composed: dict[str, Any]
    fallback_diff: Optional[dict[str, Any]]


def card_to_dict(card: Any) -> dict[str, Any]:
    """Serialize a PresetCard dataclass to a plain dict."""
    if card is None:
        return {}
    return dataclasses.asdict(card)


def preset_to_record(alias_id: str, preset_def: Any) -> PresetRecord:
    """Convert a loaded ``PresetDef`` into a GUI-safe record."""
    return PresetRecord(
        id=alias_id,
        model=preset_def.model,
        hardware=preset_def.hardware,
        profile=preset_def.profile,
        runtime=preset_def.runtime,
        has_card=preset_def.has_card(),
        card=card_to_dict(preset_def.card),
    )


def load_preset(preset_id: str) -> Any:
    """Load one typed ``PresetDef`` without surfacing deprecation warnings."""
    from sndr.model_configs.registry_v2 import load_preset_def

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return load_preset_def(preset_id)
    except Exception as exc:
        raise PresetNotFoundError(
            preset_id,
            f"{type(exc).__name__}: {exc}",
        ) from exc


def load_corpus() -> tuple[tuple[str, Any], ...]:
    """Load all builtin presets as ``(preset_id, PresetDef)`` tuples.

    Invalid rows are skipped here to preserve the historical CLI behavior.
    Use :func:`list_presets` to receive structured load errors.
    """
    from sndr.model_configs.registry_v2 import list_presets as list_ids

    out: list[tuple[str, Any]] = []
    for preset_id in list_ids():
        try:
            out.append((preset_id, load_preset(preset_id)))
        except PresetNotFoundError:
            continue
    return tuple(out)


def _load_corpus_with_errors() -> tuple[tuple[tuple[str, Any], ...], tuple[PresetLoadIssue, ...], int]:
    from sndr.model_configs.registry_v2 import list_presets as list_ids

    preset_ids = list_ids()
    loaded: list[tuple[str, Any]] = []
    errors: list[PresetLoadIssue] = []
    for preset_id in preset_ids:
        try:
            loaded.append((preset_id, load_preset(preset_id)))
        except PresetNotFoundError as exc:
            errors.append(
                PresetLoadIssue(
                    preset_id=preset_id,
                    error_type=type(exc).__name__,
                    message=exc.reason,
                )
            )
    return tuple(loaded), tuple(errors), len(preset_ids)


def passes_list_filters(
    preset_def: Any,
    *,
    family: Optional[str] = None,
    workload: Optional[str] = None,
    hardware: Optional[str] = None,
    mode: Optional[str] = None,
    status: Optional[str] = None,
) -> bool:
    """Return True if a preset matches every list filter."""
    card = preset_def.card

    if hardware and preset_def.hardware != hardware:
        return False
    if family:
        if card is None or card.routing_family != family:
            return False
    if mode:
        if card is None or card.mode != mode:
            return False
    if status:
        if card is None or card.status != status:
            return False
    if workload:
        if card is None or workload not in card.workload_allow:
            return False
    return True


def list_presets(
    *,
    family: Optional[str] = None,
    workload: Optional[str] = None,
    hardware: Optional[str] = None,
    mode: Optional[str] = None,
    status: Optional[str] = None,
) -> PresetListResult:
    """Return filtered preset records for catalog/list views."""
    corpus, load_errors, total = _load_corpus_with_errors()
    matches = [
        (preset_id, preset_def)
        for preset_id, preset_def in corpus
        if passes_list_filters(
            preset_def,
            family=family,
            workload=workload,
            hardware=hardware,
            mode=mode,
            status=status,
        )
    ]
    records = tuple(
        preset_to_record(preset_id, preset_def)
        for preset_id, preset_def in matches
    )
    return PresetListResult(
        filters={
            "family": family,
            "workload": workload,
            "hardware": hardware,
            "mode": mode,
            "status": status,
        },
        matched=len(records),
        total=total,
        presets=records,
        load_errors=load_errors,
    )


def get_preset(preset_id: str) -> PresetRecord:
    """Return one preset record by id."""
    return preset_to_record(preset_id, load_preset(preset_id))


def drill_field(obj: Any, path: str) -> Any:
    """Walk a dot-path through nested dataclasses, lists, tuples and dicts."""
    cur = obj
    walked: list[str] = []
    for seg in path.split("."):
        walked.append(seg)
        if seg.isdigit() and isinstance(cur, (list, tuple)):
            idx = int(seg)
            if idx >= len(cur):
                raise KeyError(
                    f"{'.'.join(walked)}: list index {idx} out of range "
                    f"(len={len(cur)})"
                )
            cur = cur[idx]
        elif isinstance(cur, dict):
            if seg not in cur:
                raise KeyError(f"{'.'.join(walked)}: key {seg!r} not in dict")
            cur = cur[seg]
        elif dataclasses.is_dataclass(cur):
            if not hasattr(cur, seg):
                raise KeyError(
                    f"{'.'.join(walked)}: attribute {seg!r} not on "
                    f"{type(cur).__name__}"
                )
            cur = getattr(cur, seg)
        else:
            raise KeyError(
                f"{'.'.join(walked)}: cannot drill into "
                f"{type(cur).__name__} (segment {seg!r})"
            )
    return cur


def compose_for(preset_id: str) -> Any:
    """Compose a preset to the V1 ModelConfig runtime shape."""
    from sndr.model_configs.registry_v2 import load_alias

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return load_alias(preset_id)
    except Exception as exc:
        raise PresetComposeError(
            preset_id,
            f"{type(exc).__name__}: {exc}",
        ) from exc


def composed_summary(cfg: Any) -> dict[str, Any]:
    """Pull GUI/CLI-relevant fields from a composed V1 ModelConfig."""
    spec_decode = cfg.spec_decode
    return {
        "composed_key": cfg.key,
        "kv_cache_dtype": cfg.kv_cache_dtype,
        "max_model_len": cfg.max_model_len,
        "max_num_seqs": cfg.max_num_seqs,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
        "spec_decode_method": spec_decode.method if spec_decode else None,
        "spec_decode_K": (
            spec_decode.num_speculative_tokens if spec_decode else None
        ),
        "enabled_patches_count": sum(
            1 for value in (cfg.genesis_env or {}).values()
            if str(value) in ("1", "true", "True")
        ),
    }


def summarize_diff(cfg_a: Any, cfg_b: Any, fallback_id: str) -> dict[str, Any]:
    """Summarize runtime deltas between a preset and its fallback."""
    a = composed_summary(cfg_a)
    b = composed_summary(cfg_b)
    diffs: list[str] = []
    for key in (
        "max_model_len",
        "max_num_seqs",
        "gpu_memory_utilization",
        "kv_cache_dtype",
        "spec_decode_method",
        "spec_decode_K",
        "enabled_patches_count",
    ):
        if a.get(key) != b.get(key):
            diffs.append(
                f"{key:24s} this={a[key]}  vs  fallback={b[key]}"
            )
    if not diffs:
        diffs.append("(no field-level differences in summary view)")
    return {"fallback_preset": fallback_id, "diffs": tuple(diffs)}


def explain_preset(preset_id: str) -> PresetExplainResult:
    """Return card + composed runtime + fallback diff for one preset."""
    preset_def = load_preset(preset_id)
    cfg = compose_for(preset_id)

    fallback_summary: Optional[dict[str, Any]] = None
    if preset_def.has_card() and preset_def.card.fallback_preset:
        fallback_id = preset_def.card.fallback_preset
        try:
            fallback_cfg = compose_for(fallback_id)
            fallback_summary = summarize_diff(cfg, fallback_cfg, fallback_id)
        except PresetProductAPIError as exc:
            fallback_summary = {
                "fallback_preset": fallback_id,
                "error": str(exc),
            }

    return PresetExplainResult(
        id=preset_id,
        card=card_to_dict(preset_def.card),
        composed=composed_summary(cfg),
        fallback_diff=fallback_summary,
    )


def passes_recommend_filters(
    _alias_id: str,
    preset_def: Any,
    *,
    workload: str,
    hardware: Optional[str],
    concurrency: Optional[int],
) -> bool:
    """Filter rules per CONFIG-UX plus workload deny safety."""
    card = preset_def.card
    if card is None:
        return False
    if workload in card.workload_deny:
        return False
    if workload not in card.workload_allow:
        return False
    if hardware is not None and preset_def.hardware != hardware:
        return False
    if concurrency is not None:
        if card.concurrency is None:
            return False
        if not (card.concurrency.min <= concurrency <= card.concurrency.max):
            return False
    if card.status == "tombstone":
        return False
    return True


def recommend_sort_key(alias_id: str, preset_def: Any) -> tuple[Any, ...]:
    """Stable ranking: status, family default, metric desc, id."""
    card = preset_def.card
    status_rank = STATUS_RANK.get(card.status, 99) if card else 99
    not_default = 0 if (card and card.default_for_family) else 1
    metric_value = (
        -(card.primary_metric.value)
        if (card and card.primary_metric and card.primary_metric.value is not None)
        else 0.0
    )
    return (status_rank, not_default, metric_value, alias_id)


def recommend_presets(
    *,
    workload: str,
    hardware: Optional[str] = None,
    concurrency: Optional[int] = None,
    top: int = 5,
) -> PresetRecommendResult:
    """Return ranked annotated presets for a workload query."""
    from sndr.model_configs.preset_schema import is_known_workload

    if not is_known_workload(workload):
        raise UnknownWorkloadError(workload)
    if top < 0:
        raise ValueError("top must be >= 0")

    matches = [
        (alias_id, preset_def)
        for alias_id, preset_def in load_corpus()
        if passes_recommend_filters(
            alias_id,
            preset_def,
            workload=workload,
            hardware=hardware,
            concurrency=concurrency,
        )
    ]
    matches.sort(key=lambda item: recommend_sort_key(item[0], item[1]))
    selected = matches[:top]
    results = tuple(
        PresetRecommendation(
            id=alias_id,
            rank=index,
            model=preset_def.model,
            hardware=preset_def.hardware,
            profile=preset_def.profile,
            runtime=preset_def.runtime,
            card=card_to_dict(preset_def.card),
        )
        for index, (alias_id, preset_def) in enumerate(selected, start=1)
    )
    return PresetRecommendResult(
        query={
            "workload": workload,
            "hardware": hardware,
            "concurrency": concurrency,
            "top": top,
        },
        results=results,
        total_matches=len(results),
        total_candidates=len(matches),
    )
