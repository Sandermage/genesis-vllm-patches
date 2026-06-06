# SPDX-License-Identifier: Apache-2.0
"""CONFIG-UX.1 — typed PresetCard / PresetDef / EvidenceRef schema.

Lives next to (not inside) `schema_v2.py` because PresetCard is an
**operator-product concern** distinct from V2-layer mechanics:

  schema_v2.py        ModelDef / HardwareDef / ProfileDef / PatchManifest
                      (runtime mechanics — patches, sizing, runtime)
  preset_schema.py    PresetDef + PresetCard + EvidenceRef + status enums
                      (operator product — workload contract, evidence,
                      fallback chain, do-not-use)

Backwards compatibility (CONFIG-UX.1 scope):

  Legacy preset YAML (3-pointer) without `card:` continues to load and
  compose byte-identically with the prior `load_alias()` path. The card
  is fully optional. When absent, the loader synthesises a minimal
  placeholder card with `status=experimental` so downstream consumers
  always see a typed PresetDef.

Strict validation only fires for `status ∈ {production, production_candidate}`
to preserve future-safe enum expansion. Other statuses pass through with
permissive validation (presence checks only, no cross-field enforcement).

See `sndr_private/planning/audits/CONFIG_UX_R_2026-05-24_RU.md` §2 + §13
for the full schema + locked scope.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from .schema import SchemaError
from .schema_v2 import _check_id


# ─── Enums (typed Literal aliases) ──────────────────────────────────────────


# CONFIG-UX.R §2.2 — 9 statuses. Strict validation only for production-facing.
PRESET_STATUSES = (
    "experimental",
    "bench_pending",
    "internal_validated",
    "production_candidate",
    "production",
    "historical",
    "tombstone",
    "example",
    "qa",
)
PresetStatus = Literal[
    "experimental", "bench_pending", "internal_validated",
    "production_candidate", "production", "historical",
    "tombstone", "example", "qa",
]

PRESET_AUDIENCES = ("operator", "dev", "bench", "qa", "internal")
PresetAudience = Literal["operator", "dev", "bench", "qa", "internal"]

PRESET_MATURITIES = ("draft", "validated", "tested")
PresetMaturity = Literal["draft", "validated", "tested"]

PRESET_MODES = (
    "throughput",
    "structured_throughput",
    "latency",
    "long_context",
    "tool_agent",
)
PresetMode = Literal[
    "throughput", "structured_throughput",
    "latency", "long_context", "tool_agent",
]

EVIDENCE_VISIBILITIES = ("public", "private", "mixed")
EvidenceVisibility = Literal["public", "private", "mixed"]

EVIDENCE_TYPES = (
    "bench",
    "smoke",
    "tool_call_eval",
    "structured_eval",
    "regression",
)
EvidenceType = Literal[
    "bench", "smoke", "tool_call_eval", "structured_eval", "regression",
]

PRIMARY_METRIC_KINDS = (
    "agg_TPS",
    "TPOT_ms",
    "TTFT_ms",
    "acceptance_rate",
    "tool_call_success_pct",
)
PrimaryMetricKind = Literal[
    "agg_TPS", "TPOT_ms", "TTFT_ms",
    "acceptance_rate", "tool_call_success_pct",
]


# CONFIG-UX.3 (2026-05-24) — frozen workload taxonomy with `custom:<slug>`
# escape hatch. `sndr preset recommend` only ranks presets whose
# `workload_allow` contains an exact match with the operator's --workload
# argument. Unknown / non-canonical workload strings either match by
# `custom:<slug>` exact-string or are rejected.
KNOWN_WORKLOADS = (
    "free_chat",
    "structured_json.short",
    "structured_json.long",
    "tool_call.short",
    "tool_call.long",
    "summarization",
    "code_gen",
    "long_context_qa",
)
_CUSTOM_WORKLOAD_RE = re.compile(
    r"^custom:[a-z0-9][a-z0-9._-]*[a-z0-9]$|^custom:[a-z0-9]$"
)


def is_known_workload(w: str) -> bool:
    """True if `w` is in the canonical taxonomy or a `custom:<slug>` form.

    Used by `sndr preset recommend` to filter operator input. Card YAML
    files can include any string in `workload_allow` / `workload_deny`;
    this function only gates the *recommend* surface.
    """
    if w in KNOWN_WORKLOADS:
        return True
    return bool(_CUSTOM_WORKLOAD_RE.match(w))


# CONFIG-UX.R §6.3 — statuses required to satisfy "production-grade" strict
# validation. Other statuses pass through permissively.
_STRICT_VALIDATION_STATUSES = frozenset({"production", "production_candidate"})


# ─── Sub-dataclasses ────────────────────────────────────────────────────────


@dataclass
class EvidenceRef:
    """One bench / smoke / eval artefact backing a preset claim.

    Path is either repo-relative (validated against filesystem by audit
    gate, NOT here) OR an `external://` URL bypassing filesystem check.
    Visibility overrides card-level `evidence_visibility` when set —
    `audit_config_catalog.py` cross-validates the two layers agree.
    """
    type: EvidenceType
    path: str
    visibility: Optional[EvidenceVisibility] = None
    note: Optional[str] = None

    def validate(self) -> None:
        if self.type not in EVIDENCE_TYPES:
            raise SchemaError(
                f"evidence_ref.type={self.type!r} must be one of {EVIDENCE_TYPES}"
            )
        if not self.path:
            raise SchemaError("evidence_ref.path required")
        if self.visibility is not None and self.visibility not in EVIDENCE_VISIBILITIES:
            raise SchemaError(
                f"evidence_ref.visibility={self.visibility!r} must be one of "
                f"{EVIDENCE_VISIBILITIES} or null"
            )


@dataclass
class PrimaryMetric:
    """Headline metric operator sees first in `sndr preset show` output."""
    kind: PrimaryMetricKind
    value: float
    source: str
    measured_at: Optional[str] = None  # ISO date

    def validate(self) -> None:
        if self.kind not in PRIMARY_METRIC_KINDS:
            raise SchemaError(
                f"primary_metric.kind={self.kind!r} must be one of "
                f"{PRIMARY_METRIC_KINDS}"
            )
        if not self.source:
            raise SchemaError("primary_metric.source required")


@dataclass
class ConcurrencyEnvelope:
    """Tested concurrency range for this preset."""
    min: int = 1
    max: int = 1
    canonical: int = 1

    def validate(self) -> None:
        if not (1 <= self.min <= self.canonical <= self.max):
            raise SchemaError(
                f"concurrency invariant violated: 1 ≤ min({self.min}) ≤ "
                f"canonical({self.canonical}) ≤ max({self.max})"
            )


@dataclass
class ContextEnvelope:
    """Context-length operating envelope (typical input/output)."""
    max_model_len: Optional[int] = None
    typical_input_tokens: Optional[int] = None
    typical_output_tokens: Optional[int] = None

    def validate(self) -> None:
        for fname in ("max_model_len", "typical_input_tokens", "typical_output_tokens"):
            v = getattr(self, fname)
            if v is not None and v <= 0:
                raise SchemaError(
                    f"context.{fname}={v} must be > 0 if set"
                )


@dataclass
class DoNotUseCondition:
    """One anti-pattern with human-readable condition + rationale."""
    condition: str
    reason: str


# ─── PresetCard ─────────────────────────────────────────────────────────────


@dataclass
class PresetCard:
    """Operator-product contract for a preset.

    Card describes WHAT the preset is for (workload, hardware envelope,
    K/MTP policy, evidence, fallback, do-not-use) — not HOW it composes
    (the 3-pointer model/hardware/profile triplet is implementation).

    Field requirements vary by `status`:

      - `production` / `production_candidate` — strict validation
        (workload_allow, workload_deny, concurrency, K, routing_family,
        primary_metric, evidence_refs, audience, mode all required;
        validate_for_status() enforces).
      - All other statuses — permissive (presence checks only).

    See CONFIG_UX_R §2.2 / §2.3 for the full required/optional matrix.
    """

    # — Identity (required at all statuses)
    title: str
    summary: str
    status: PresetStatus

    # — Lifecycle context
    audience: Optional[PresetAudience] = None
    maturity: Optional[PresetMaturity] = None

    # — Workload contract
    mode: Optional[PresetMode] = None
    workload_allow: list[str] = field(default_factory=list)
    workload_deny: list[str] = field(default_factory=list)

    # — Operating envelope
    concurrency: Optional[ConcurrencyEnvelope] = None
    K: Optional[int] = None
    context: Optional[ContextEnvelope] = None

    # — Routing metadata (consumer-side; ProfileDef.routing remains source of truth)
    routing_family: Optional[str] = None
    default_for_family: bool = False
    fallback_preset: Optional[str] = None

    # — Evidence
    primary_metric: Optional[PrimaryMetric] = None
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    evidence_visibility: Optional[EvidenceVisibility] = None

    # — Operator guidance
    tradeoffs: list[str] = field(default_factory=list)
    do_not_use: list[DoNotUseCondition] = field(default_factory=list)

    # — Provenance (auto-populated; optional)
    card_version: int = 1
    card_updated: Optional[str] = None
    card_author: Optional[str] = None

    # — Tombstone metadata (status=tombstone only)
    tombstone_reason: Optional[str] = None

    def validate(self) -> None:
        """Shape-only validation — runs for every loaded card regardless
        of status. Field-value checks beyond shape live in
        `validate_for_status` (semantic) so partial cards synthesised
        for legacy 3-field presets don't trip cross-field rules before
        status assignment.
        """
        if not self.title:
            raise SchemaError("card.title required")
        if not self.summary:
            raise SchemaError("card.summary required")
        if self.status not in PRESET_STATUSES:
            raise SchemaError(
                f"card.status={self.status!r} must be one of {PRESET_STATUSES}"
            )
        if self.audience is not None and self.audience not in PRESET_AUDIENCES:
            raise SchemaError(
                f"card.audience={self.audience!r} must be one of {PRESET_AUDIENCES}"
            )
        if self.maturity is not None and self.maturity not in PRESET_MATURITIES:
            raise SchemaError(
                f"card.maturity={self.maturity!r} must be one of {PRESET_MATURITIES}"
            )
        if self.mode is not None and self.mode not in PRESET_MODES:
            raise SchemaError(
                f"card.mode={self.mode!r} must be one of {PRESET_MODES}"
            )
        if self.evidence_visibility is not None and self.evidence_visibility not in EVIDENCE_VISIBILITIES:
            raise SchemaError(
                f"card.evidence_visibility={self.evidence_visibility!r} must be one of "
                f"{EVIDENCE_VISIBILITIES}"
            )
        if self.K is not None and self.K < 1:
            raise SchemaError(f"card.K={self.K} must be >= 1 if set (or null)")
        if self.concurrency is not None:
            self.concurrency.validate()
        if self.context is not None:
            self.context.validate()
        if self.primary_metric is not None:
            self.primary_metric.validate()
        for i, ev in enumerate(self.evidence_refs):
            try:
                ev.validate()
            except SchemaError as e:
                raise SchemaError(f"card.evidence_refs[{i}]: {e}") from e


# ─── Validation per status (semantic, not shape) ────────────────────────────


def validate_for_status(card: PresetCard, preset_id: str) -> list[str]:
    """Status-aware semantic validation. Returns list of error messages.

    Called by audit_config_catalog.py (CONFIG-UX.audit phase) — NOT during
    load. Loader emits a warning for missing-card; audit gate emits errors
    for production-grade presets failing this check.

    Permissive for non-production statuses to preserve future-safe enum
    expansion (e.g. operator can add `example`/`qa` cards without filling
    every required-for-prod field).
    """
    errors: list[str] = []
    status = card.status

    if status not in _STRICT_VALIDATION_STATUSES:
        # Permissive path — only shape was checked. No semantic enforcement.
        return errors

    # Strict path — production / production_candidate.
    def _require(field_name: str, value: Any, *, allow_empty_list: bool = False) -> None:
        if value is None:
            errors.append(
                f"preset {preset_id!r} (status={status}): card.{field_name} required"
            )
            return
        if isinstance(value, list) and not allow_empty_list and not value:
            errors.append(
                f"preset {preset_id!r} (status={status}): card.{field_name} "
                f"must be non-empty list"
            )

    _require("audience", card.audience)
    _require("mode", card.mode)
    _require("workload_allow", card.workload_allow)
    _require("workload_deny", card.workload_deny)
    _require("concurrency", card.concurrency)
    _require("K", card.K)
    _require("routing_family", card.routing_family)
    _require("primary_metric", card.primary_metric)
    _require("evidence_refs", card.evidence_refs)

    # CONFIG-UX.R §2.4 rule 1: production+operator requires ≥1 public evidence.
    if (
        status == "production"
        and card.audience == "operator"
        and card.evidence_visibility in (None, "private")
    ):
        public_refs = [
            ev for ev in card.evidence_refs
            if ev.visibility == "public"
        ]
        if not public_refs and card.evidence_visibility != "public":
            errors.append(
                f"preset {preset_id!r}: status=production + audience=operator "
                f"requires at least one evidence_ref with visibility=public OR "
                f"card.evidence_visibility ∈ {{public, mixed}}; got "
                f"evidence_visibility={card.evidence_visibility!r} and no public refs"
            )

    # CONFIG-UX.R §2.4 rule 4: K>1 production preset must declare fallback_preset.
    if card.K is not None and card.K > 1 and not card.fallback_preset:
        errors.append(
            f"preset {preset_id!r}: K={card.K} (>1) requires card.fallback_preset "
            f"(pointer to K=1 preset of same routing_family)"
        )

    return errors


# ─── PresetDef ──────────────────────────────────────────────────────────────


@dataclass
class PresetDef:
    """Typed wrapper for `builtin/presets/<alias>.yaml`.

    Legacy 3-pointer presets (model + hardware + profile + optional runtime,
    no `card:`) continue to load through this wrapper — the card field is
    `None` for them. The loader emits a one-time DeprecationWarning per
    unannotated preset suggesting CONFIG-UX.2 annotation work.

    Note: `id` is the filename stem (alias), not a separate YAML field.
    `kind: preset` is conventional; loader doesn't require it for legacy
    compatibility but emits warning if absent.
    """
    id: str
    model: str
    hardware: str
    profile: Optional[str] = None
    runtime: Optional[str] = None
    card: Optional[PresetCard] = None

    # Optional schema-version field — defaulted for legacy compat.
    schema_version: Optional[int] = None
    kind: Optional[str] = None

    def validate(self) -> None:
        _check_id(self.id, "preset.id")
        if not self.model:
            raise SchemaError(f"preset {self.id!r}: model pointer required")
        if not self.hardware:
            raise SchemaError(f"preset {self.id!r}: hardware pointer required")
        # Pointer IDs follow V2 id grammar.
        _check_id(self.model, f"preset {self.id!r}.model")
        _check_id(self.hardware, f"preset {self.id!r}.hardware")
        if self.profile is not None:
            _check_id(self.profile, f"preset {self.id!r}.profile")
        if self.card is not None:
            self.card.validate()

    def has_card(self) -> bool:
        return self.card is not None


# ─── YAML → PresetDef parser ────────────────────────────────────────────────


_ALIAS_FILENAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*[a-z0-9]$|^[a-z0-9]$")


def parse_preset_yaml(alias_id: str, data: dict) -> PresetDef:
    """Build a typed PresetDef from a YAML-loaded dict.

    `alias_id` is the filename stem (no `.yaml`). Validates ID shape
    before parsing so a malformed filename produces a clear error.

    Card parsing recursively builds nested dataclasses (EvidenceRef,
    PrimaryMetric, ConcurrencyEnvelope, ContextEnvelope, DoNotUseCondition).
    Unknown keys are silently ignored to preserve forward-compat with
    future card schema extensions — they'll start being honored when
    the corresponding dataclass field is added.
    """
    if not _ALIAS_FILENAME_RE.match(alias_id):
        raise SchemaError(
            f"preset alias filename {alias_id!r} must match V2 id grammar "
            f"(lowercase, alphanumerics + `.`, `_`, `-`, start/end alnum)"
        )

    card_data = data.get("card")
    card: Optional[PresetCard] = None
    if card_data is not None:
        if not isinstance(card_data, dict):
            raise SchemaError(
                f"preset {alias_id!r}: card must be a mapping (got {type(card_data).__name__})"
            )
        card = _parse_card(alias_id, card_data)

    return PresetDef(
        id=alias_id,
        model=data.get("model", ""),
        hardware=data.get("hardware", ""),
        profile=data.get("profile"),
        runtime=data.get("runtime"),
        card=card,
        schema_version=data.get("schema_version"),
        kind=data.get("kind"),
    )


def _parse_card(preset_id: str, data: dict) -> PresetCard:
    """Build a PresetCard from a YAML mapping."""
    # Required fields surface as SchemaError via PresetCard.validate() — here
    # we just thread defaults so missing-but-required fields land in the
    # dataclass as falsy values rather than throwing AttributeError.
    return PresetCard(
        title=data.get("title", ""),
        summary=data.get("summary", ""),
        status=data.get("status", "experimental"),
        audience=data.get("audience"),
        maturity=data.get("maturity"),
        mode=data.get("mode"),
        workload_allow=list(data.get("workload_allow") or []),
        workload_deny=list(data.get("workload_deny") or []),
        concurrency=_parse_concurrency(data.get("concurrency")),
        K=data.get("K"),
        context=_parse_context(data.get("context")),
        routing_family=data.get("routing_family"),
        default_for_family=bool(data.get("default_for_family", False)),
        fallback_preset=data.get("fallback_preset"),
        primary_metric=_parse_primary_metric(data.get("primary_metric")),
        evidence_refs=_parse_evidence_refs(data.get("evidence_refs")),
        evidence_visibility=data.get("evidence_visibility"),
        tradeoffs=list(data.get("tradeoffs") or []),
        do_not_use=_parse_do_not_use(data.get("do_not_use")),
        card_version=int(data.get("card_version", 1)),
        card_updated=data.get("card_updated"),
        card_author=data.get("card_author"),
        tombstone_reason=data.get("tombstone_reason"),
    )


def _parse_concurrency(data: Any) -> Optional[ConcurrencyEnvelope]:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise SchemaError(
            f"card.concurrency must be a mapping (got {type(data).__name__})"
        )
    return ConcurrencyEnvelope(
        min=int(data.get("min", 1)),
        max=int(data.get("max", 1)),
        canonical=int(data.get("canonical", data.get("max", 1))),
    )


def _parse_context(data: Any) -> Optional[ContextEnvelope]:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise SchemaError(
            f"card.context must be a mapping (got {type(data).__name__})"
        )
    return ContextEnvelope(
        max_model_len=data.get("max_model_len"),
        typical_input_tokens=data.get("typical_input_tokens"),
        typical_output_tokens=data.get("typical_output_tokens"),
    )


def _parse_primary_metric(data: Any) -> Optional[PrimaryMetric]:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise SchemaError(
            f"card.primary_metric must be a mapping (got {type(data).__name__})"
        )
    return PrimaryMetric(
        kind=data.get("kind", ""),
        value=float(data.get("value", 0.0)),
        source=data.get("source", ""),
        measured_at=data.get("measured_at"),
    )


def _parse_evidence_refs(data: Any) -> list[EvidenceRef]:
    if data is None:
        return []
    if not isinstance(data, list):
        raise SchemaError(
            f"card.evidence_refs must be a list (got {type(data).__name__})"
        )
    out: list[EvidenceRef] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise SchemaError(
                f"card.evidence_refs[{i}] must be a mapping "
                f"(got {type(item).__name__})"
            )
        out.append(EvidenceRef(
            type=item.get("type", ""),
            path=item.get("path", ""),
            visibility=item.get("visibility"),
            note=item.get("note"),
        ))
    return out


def _parse_do_not_use(data: Any) -> list[DoNotUseCondition]:
    if data is None:
        return []
    if not isinstance(data, list):
        raise SchemaError(
            f"card.do_not_use must be a list (got {type(data).__name__})"
        )
    out: list[DoNotUseCondition] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise SchemaError(
                f"card.do_not_use[{i}] must be a mapping "
                f"(got {type(item).__name__})"
            )
        cond = item.get("condition", "")
        reason = item.get("reason", "")
        if not cond or not reason:
            raise SchemaError(
                f"card.do_not_use[{i}]: both `condition` and `reason` required"
            )
        out.append(DoNotUseCondition(condition=cond, reason=reason))
    return out


# ─── Synthetic card for legacy 3-pointer presets ────────────────────────────


def synth_card_for_legacy(preset_id: str) -> PresetCard:
    """Build a minimal placeholder card for a 3-pointer preset without `card:`.

    Used by the loader to ensure downstream consumers always see a typed
    PresetCard. Status `experimental` + audience `dev` — operator must
    annotate (CONFIG-UX.2 work) to upgrade to production-facing statuses.
    """
    return PresetCard(
        title=f"(unannotated preset {preset_id!r})",
        summary=(
            "Legacy 3-pointer preset without operator card. "
            "Annotate via CONFIG-UX.2 to enable preset list/show/recommend."
        ),
        status="experimental",
        audience="dev",
    )


__all__ = [
    # Enums + literal tuples
    "PRESET_STATUSES", "PresetStatus",
    "PRESET_AUDIENCES", "PresetAudience",
    "PRESET_MATURITIES", "PresetMaturity",
    "PRESET_MODES", "PresetMode",
    "EVIDENCE_VISIBILITIES", "EvidenceVisibility",
    "EVIDENCE_TYPES", "EvidenceType",
    "PRIMARY_METRIC_KINDS", "PrimaryMetricKind",
    "KNOWN_WORKLOADS", "is_known_workload",
    # Dataclasses
    "EvidenceRef", "PrimaryMetric",
    "ConcurrencyEnvelope", "ContextEnvelope", "DoNotUseCondition",
    "PresetCard", "PresetDef",
    # Functions
    "validate_for_status",
    "parse_preset_yaml",
    "synth_card_for_legacy",
]
