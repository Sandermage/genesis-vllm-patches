# SPDX-License-Identifier: Apache-2.0
"""Layered model_config V2 schema (PROJECT_ROADMAP_V2, Phase 1).

Four orthogonal layers compose into a runtime `ModelConfig` (V1 shape):

  • ModelDef         — identity + capabilities + canonical patches set
  • HardwareDef      — rig + sizing knobs + runtime block (docker/podman/bare)
  • ProfileDef       — patches delta (test → promote workflow)
  • PatchManifest    — community plugin metadata (per-patch, lives next to plugin code)

Composition is driven by `compose.py`. Each layer owns specific fields;
cross-layer conflicts on owned fields raise SchemaError at load time
(Q1 decision: ownership-based merge, not override-based).

Backwards compatibility: composed result is a V1 `ModelConfig`, so the
existing runtime (`CompatibilityMatrix`, launch, k8s/compose/quadlet
renderers, tests) continues working without changes.

See `docs/_internal/PROJECT_ROADMAP_V2_2026-05-12_RU.md` § 4 for the full
finalized architecture and Q1-Q7 operator decisions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

# Reuse V1 helper types where possible (HardwareSpec, SpecDecodeConfig,
# PatchAttribution, ...). This keeps a single source of truth for
# sub-component validation rules. PatchAttribution is imported through
# the V1 module so V1 ModelConfig and V2 ModelDef share the same dataclass
# — the same instance survives `compose()` without round-tripping.
from .schema import (
    HardwareSpec,
    PatchAttribution,
    SchemaError,
    SpecDecodeConfig,
)

SCHEMA_VERSION_V2 = 2


# ─── Common validators ────────────────────────────────────────────────────


_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*[a-z0-9]$|^[a-z0-9]$")
# Same restriction we already use for V1 keys, just split out so V2
# loaders can validate without re-implementing.


def _check_id(value: str, field_name: str) -> None:
    """Layer IDs follow the same kebab-case rule as V1 keys."""
    if not value:
        raise SchemaError(f"{field_name} required")
    if not _ID_RE.match(value):
        raise SchemaError(
            f"{field_name}={value!r} must be lowercase alphanumerics with "
            "`.`, `_`, or `-` (start/end alphanumeric)"
        )


# Patch IDs follow Genesis convention: P or PN, then digits, optional suffix.
# Examples: P67, P67b, PN116, PN119, PN16_V6. Uppercase distinguishes from
# model/hardware/profile IDs which use kebab-case lowercase.
_PATCH_ID_RE = re.compile(r"^P[N]?[0-9]+[A-Za-z0-9_]*$")


def _check_patch_id(value: str, field_name: str) -> None:
    """Patch IDs use uppercase P[N]?\\d+ convention (P67, PN119, PN16_V6)."""
    if not value:
        raise SchemaError(f"{field_name} required")
    if not _PATCH_ID_RE.match(value):
        raise SchemaError(
            f"{field_name}={value!r} must match pattern P[N]?\\d+[A-Za-z0-9_]* "
            "(e.g. P67, P67b, PN119, PN16_V6)"
        )


def _check_kind(value: str, expected: str) -> None:
    if value != expected:
        raise SchemaError(
            f"`kind: {value!r}` does not match this loader (expected {expected!r})"
        )


def _check_schema_version(value: int) -> None:
    if value != SCHEMA_VERSION_V2:
        raise SchemaError(
            f"schema_version={value} unsupported (expected {SCHEMA_VERSION_V2})"
        )


# ─── ModelDef ──────────────────────────────────────────────────────────────


@dataclass
class ModelCapabilities:
    """Inherent model capabilities — these change only when the model itself
    changes (different checkpoint, different architecture). Operators must
    NOT override these from hardware/profile layers; a different capability
    set means a different ModelDef entry.
    """
    attention_arch: Literal[
        "dense", "hybrid_gdn_moe", "hybrid_mamba", "moe",
    ]
    tool_call_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None
    enable_auto_tool_choice: bool = True
    spec_decode: Optional[SpecDecodeConfig] = None
    kv_cache_dtype: Optional[str] = None

    def validate(self) -> None:
        if self.spec_decode is not None:
            self.spec_decode.validate()


@dataclass
class ModelRequires:
    """Hardware preconditions a ModelDef declares. Composer rejects
    incompatible (model, hardware) pairs at load time using these."""
    min_total_vram_mib: int = 0
    min_gpu_count: int = 1
    min_cuda_capability: Optional[tuple[int, int]] = None
    # Block-list of attention-arch markers a rig may declare it can't run.
    rig_arch_blocklist: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.min_gpu_count < 1:
            raise SchemaError("min_gpu_count must be >= 1")
        if self.min_total_vram_mib < 0:
            raise SchemaError("min_total_vram_mib must be >= 0")


@dataclass
class ModelVersions:
    """Version pins for this model's canonical configuration."""
    genesis_pin_min: Optional[str] = None
    vllm_pin_required: Optional[str] = None
    reference_metrics_ref: Optional[str] = None


# PatchAttribution + _PATCH_ROLES live in V1 schema.py (re-exported above)
# so V1 ModelConfig and V2 ModelDef share the exact same dataclass.
# Phase A moved the definition into V2; Phase B lifted it into V1.


@dataclass
class ModelDef:
    """Identity + capabilities + canonical patches for a single model.

    Owns: model_path, dtype, parsers, spec_decode method, kv_cache_dtype,
    canonical patches matrix, version pins. Stable across rigs."""
    schema_version: int
    kind: Literal["model"]
    id: str
    title: str
    maintainer: str
    last_validated: str
    license: str

    model_path: str
    served_model_name: Optional[str] = None
    quantization: Optional[str] = None
    dtype: str = "float16"
    trust_remote_code: bool = True

    capabilities: ModelCapabilities = field(
        default_factory=lambda: ModelCapabilities(attention_arch="dense"),
    )
    requires: ModelRequires = field(default_factory=ModelRequires)
    versions: ModelVersions = field(default_factory=ModelVersions)

    # Canonical patches matrix — string-valued env knobs.
    # A profile delta can disable / enable / override entries here.
    patches: dict[str, str] = field(default_factory=dict)

    # optional structured rationale keyed by registry patch ID
    # (e.g. `PN204`, not the env-flag name). Stored alongside `patches`
    # so a model's "what" + "why" stay in one file. Empty dict is the
    # default for legacy YAMLs that haven't been backfilled yet.
    # Consumed by `sndr patches plan --explain` (Phase B) and the
    # compose policy filter (Phase C). compose() itself ignores this
    # field — Phase A is additive and non-breaking.
    patches_attribution: dict[str, PatchAttribution] = field(default_factory=dict)

    # Optional Jinja chat-template override (`--chat-template <path>`).
    # When set, the launch renderer bind-mounts the host-resolved file
    # into the container at /chat_templates/<basename> and emits the
    # CLI flag. Supports `${chat_templates_dir}/...` symbolic refs that
    # resolve via host.yaml. Use for models where the upstream
    # tokenizer-bundled chat_template.jinja has known bugs (e.g.
    # qwen3.6-27b club-3090 disc #53 — assistant branch does not close
    # </think> before <tool_call>, tools stop firing in multi-turn
    # agentic traces).
    chat_template: Optional[str] = None

    notes: list[str] = field(default_factory=list)

    def validate(self) -> None:
        _check_schema_version(self.schema_version)
        _check_kind(self.kind, "model")
        _check_id(self.id, "model.id")
        if not self.model_path:
            raise SchemaError("model.model_path required")
        if not self.title or not self.maintainer:
            raise SchemaError("model requires title + maintainer")
        self.capabilities.validate()
        self.requires.validate()
        for k, v in self.patches.items():
            if not isinstance(k, str) or not k:
                raise SchemaError(f"model.patches key {k!r} must be non-empty str")
            if not isinstance(v, str):
                raise SchemaError(
                    f"model.patches[{k!r}] value must be str (got {type(v).__name__})"
                )
        for pid, attr in self.patches_attribution.items():
            # Key must be a canonical Genesis patch ID (P{N}? + digits).
            # Same validator the PatchManifest layer uses — keeps the
            # cross-reference between attribution and registry tight.
            _check_patch_id(pid, f"patches_attribution[{pid!r}]")
            if not isinstance(attr, PatchAttribution):
                raise SchemaError(
                    f"patches_attribution[{pid!r}] must be PatchAttribution "
                    f"(got {type(attr).__name__})"
                )
            attr.validate(key=pid)
        if self.chat_template is not None and not isinstance(self.chat_template, str):
            raise SchemaError(
                f"model.chat_template must be str | None (got "
                f"{type(self.chat_template).__name__})"
            )


# ─── HardwareDef ──────────────────────────────────────────────────────────


@dataclass
class RuntimeDockerBlock:
    """Docker-runtime-specific knobs (image, ports, mounts)."""
    image: str
    image_digest: Optional[str] = None
    container_name_template: str = "vllm-{model_id}"
    host_port: int = 8000
    container_port: int = 8000
    shm_size: str = "8g"
    network: Optional[str] = None
    mounts: list[str] = field(default_factory=list)
    extra_run_flags: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.image:
            raise SchemaError("runtime.docker.image required")
        for port_name in ("host_port", "container_port"):
            p = getattr(self, port_name)
            if not isinstance(p, int) or not (1 <= p <= 65535):
                raise SchemaError(f"runtime.docker.{port_name} out of range")


@dataclass
class RuntimeBareMetalBlock:
    """Bare-metal-runtime-specific knobs."""
    venv_path: str = ""              # operator-supplied at install time
    systemd_unit_template: Optional[str] = None  # path to template

    def validate(self) -> None:
        # Bare-metal block is optional in many rigs; no required fields.
        pass


@dataclass
class RuntimeBlock:
    """Hardware-layer runtime block. Default + per-runtime config.

    Q-runtime decision (PROJECT_ROADMAP_V2 § 1): runtime placement is the
    hardware layer because it's a property of the rig, not patches testing.
    CLI `--runtime <name>` overrides default; must appear in `supported`.
    """
    default: Literal["docker", "podman", "bare-metal"] = "docker"
    supported: list[str] = field(
        default_factory=lambda: ["docker"],
    )
    docker: Optional[RuntimeDockerBlock] = None
    podman: Optional[RuntimeDockerBlock] = None  # podman uses docker-compatible shape
    bare_metal: Optional[RuntimeBareMetalBlock] = None

    _ALLOWED = ("docker", "podman", "bare-metal")

    def validate(self) -> None:
        if self.default not in self._ALLOWED:
            raise SchemaError(
                f"runtime.default={self.default!r} must be one of {self._ALLOWED}"
            )
        if self.default not in self.supported:
            raise SchemaError(
                f"runtime.default={self.default!r} not in runtime.supported"
            )
        for r in self.supported:
            if r not in self._ALLOWED:
                raise SchemaError(
                    f"runtime.supported has unknown {r!r}; allowed: {self._ALLOWED}"
                )
        # Per-runtime block validation: if a runtime is declared supported,
        # the matching block must exist + validate.
        if "docker" in self.supported and self.docker is not None:
            self.docker.validate()
        if "podman" in self.supported and self.podman is not None:
            self.podman.validate()
        if "bare-metal" in self.supported and self.bare_metal is not None:
            self.bare_metal.validate()


@dataclass
class HardwareSizing:
    """Operator-tuned sizing knobs for this rig. Composer applies them
    verbatim into the final V1 ModelConfig."""
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 2
    max_num_batched_tokens: int = 4096
    enable_chunked_prefill: bool = True
    enforce_eager: bool = False
    disable_custom_all_reduce: bool = True

    def validate(self) -> None:
        if self.max_model_len < 1:
            raise SchemaError("sizing.max_model_len must be >= 1")
        if not (0.0 < self.gpu_memory_utilization <= 1.0):
            raise SchemaError("sizing.gpu_memory_utilization must be in (0, 1]")


@dataclass
class HardwareDef:
    """Rig identity + sizing knobs + runtime block.

    Owns: hardware (HardwareSpec), sizing, runtime, deploy defaults,
    system_env. Stable across models."""
    schema_version: int
    kind: Literal["hardware"]
    id: str
    title: str
    maintainer: str

    hardware: HardwareSpec
    sizing: HardwareSizing = field(default_factory=HardwareSizing)
    runtime: RuntimeBlock = field(default_factory=RuntimeBlock)

    system_env: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def validate(self) -> None:
        _check_schema_version(self.schema_version)
        _check_kind(self.kind, "hardware")
        _check_id(self.id, "hardware.id")
        if not self.title or not self.maintainer:
            raise SchemaError("hardware requires title + maintainer")
        self.hardware.validate()
        self.sizing.validate()
        self.runtime.validate()


# ─── ProfileDef ───────────────────────────────────────────────────────────


@dataclass
class PatchesDelta:
    """Three explicit actions on the canonical patches dict + optional
    attribution override layer.

    Order applied by composer: enable → disable → override → attribution.
    Conflicts within a profile (enable + disable same key) raise SchemaError.

    the ``attribution`` map lets a
    profile override ModelDef.patches_attribution per patch ID at
    compose time. Use case: the long-ctx profile flags PN204 as
    load_bearing (model marked it optional_perf because the latency
    profile doesn't need it), or an A/B profile downgrades a patch
    from defensive to suspected_regression during a validation window.
    Override is per-entry full replacement, not field-level merge.
    """
    enable: dict[str, str] = field(default_factory=dict)
    disable: list[str] = field(default_factory=list)
    override: dict[str, str] = field(default_factory=dict)
    attribution: dict[str, "PatchAttribution"] = field(default_factory=dict)

    def validate(self) -> None:
        enabled = set(self.enable)
        disabled = set(self.disable)
        conflict = enabled & disabled
        if conflict:
            raise SchemaError(
                f"profile patches_delta: keys appear in BOTH enable and "
                f"disable: {sorted(conflict)}"
            )
        for d in self.disable:
            if not isinstance(d, str) or not d:
                raise SchemaError("profile patches_delta.disable entries must be non-empty strings")
        for src_name, src in (("enable", self.enable), ("override", self.override)):
            for k, v in src.items():
                if not isinstance(v, str):
                    raise SchemaError(
                        f"profile patches_delta.{src_name}[{k!r}] must be str"
                    )
        # validate the optional attribution override map.
        # Key shape mirrors ModelDef.patches_attribution: keys are
        # canonical patch IDs (P[N]?\\d+[A-Za-z0-9_]*), values are
        # PatchAttribution entries (role enum + role-conditional aux
        # fields). _check_patch_id() enforces the key contract; the
        # entry-level role check delegates to PatchAttribution.validate.
        for pid, attr in self.attribution.items():
            _check_patch_id(pid, f"profile patches_delta.attribution[{pid!r}]")
            if not isinstance(attr, PatchAttribution):
                raise SchemaError(
                    f"profile patches_delta.attribution[{pid!r}] must be "
                    f"PatchAttribution (got {type(attr).__name__})"
                )
            attr.validate(key=pid)


@dataclass
class ProfilePromotion:
    """Acceptance criteria + target for promote workflow."""
    validation_required: list[str] = field(default_factory=list)
    promote_to: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class ProfileVersionsOverride:
    """Optional pin overrides applied on top of model.versions."""
    vllm_pin_required: Optional[str] = None
    genesis_pin: Optional[str] = None


@dataclass
class CompressionPlanConfig:
    """Per-layer KV compression plan owned by the profile.

    Used by the structured-role profile to declare which layers must
    stay in their native (uncompressed) dtype because they are
    KV-sharing sources for an MTP drafter. Layers NOT listed here
    use ``default_kv_dtype``.

    Composer expands this into the corresponding env that the
    existing G4_60K / PN247 reader honors
    (currently ``GENESIS_G4_TQ_FORCE_SKIP_LAYERS``); when the
    reader migrates to the SNDR canonical name the composer emits
    only the SNDR variant.

    The strategy enum is constrained to ``per_layer`` for v1 — other
    strategies (global, role_based) can be added when there's a
    second concrete use case.
    """
    native_source_layers: list[int] = field(default_factory=list)
    default_kv_dtype: Optional[str] = None
    strategy: Literal["per_layer"] = "per_layer"

    def validate(self) -> None:
        seen: set[int] = set()
        for i, layer in enumerate(self.native_source_layers):
            if not isinstance(layer, int) or layer < 0:
                raise SchemaError(
                    f"profile.compression_plan.native_source_layers[{i}]="
                    f"{layer!r} must be a non-negative int"
                )
            if layer in seen:
                raise SchemaError(
                    f"profile.compression_plan.native_source_layers contains "
                    f"duplicate index {layer}"
                )
            seen.add(layer)
        if self.default_kv_dtype is not None and not isinstance(
            self.default_kv_dtype, str
        ):
            raise SchemaError(
                "profile.compression_plan.default_kv_dtype must be a str when set"
            )
        if self.strategy != "per_layer":
            raise SchemaError(
                f"profile.compression_plan.strategy={self.strategy!r} "
                f"must be 'per_layer' (only supported value in v1)"
            )


@dataclass
class BackendPlanConfig:
    """Attention backend assignments owned by the profile.

    Names map to the v1 attention-backend enum string forms
    (TURBOQUANT / TRITON_ATTN / FLASH_ATTN / etc.). The composer is
    NOT a validator of which combinations are runtime-safe; that
    contract belongs to the spec_decode planner + safety guard
    (PN271b/PN274). The composer only propagates the operator's
    declared intent into env / engine-config.

    P1.8 (2026-05-21): adds ``drafter_kv_sharing`` to declare whether
    the drafter shares physical KV with the target (β'-A K=4 needs
    ``physical``; native default is ``disabled``). This replaces the
    operator-implicit knowledge that hand-written launchers must set
    ``GENESIS_ENABLE_G4_76_DISABLE_DRAFTER_KV_SHARING=0`` to opt into
    native sharing. The mapping provider's predicate requires this
    env to be ``0`` for the artifact lookup to succeed, but the
    Gemma4 mapping provider hardcodes ``default="1"`` (disable kv
    sharing). Profiles that need the validated β'-A path must
    declare ``drafter_kv_sharing: physical`` so compose emits the
    env explicitly.
    """
    target_default: Optional[str] = None
    target_native_layers: Optional[str] = None
    drafter_sliding: Optional[str] = None
    drafter_full: Optional[str] = None
    # P1.8: drafter KV sharing policy.
    #   physical  → drafter shares target's KV slots; native β'-A K=4 path;
    #                emits GENESIS_ENABLE_G4_76_DISABLE_DRAFTER_KV_SHARING=0
    #                (mapping provider's predicate reads this as "kv_sharing_on")
    #   disabled  → drafter uses its own KV; mapping provider default behavior;
    #                no env emission (relies on runtime default = disable)
    #   None       → profile is silent on this dimension; runtime default applies
    drafter_kv_sharing: Optional[Literal["physical", "disabled"]] = None

    def validate(self) -> None:
        for name in (
            "target_default", "target_native_layers",
            "drafter_sliding", "drafter_full",
        ):
            val = getattr(self, name)
            if val is not None and (not isinstance(val, str) or not val):
                raise SchemaError(
                    f"profile.backend_plan.{name}={val!r} must be a non-empty str"
                )
        if self.drafter_kv_sharing is not None and (
            self.drafter_kv_sharing not in ("physical", "disabled")
        ):
            raise SchemaError(
                f"profile.backend_plan.drafter_kv_sharing="
                f"{self.drafter_kv_sharing!r} must be one of "
                f"('physical', 'disabled') or None"
            )


@dataclass
class RoutingConfig:
    """Operator-declared workload intent for the profile.

    At runtime the spec-decode router intersects this list with the
    artifact's ``allowed_workloads`` to compute the effective allow
    set; the artifact is the source of truth on what's been bench-
    validated, this field is what the operator claims the profile
    is intended for.
    """
    intended_workloads: list[str] = field(default_factory=list)

    def validate(self) -> None:
        seen: set[str] = set()
        for i, cls in enumerate(self.intended_workloads):
            if not isinstance(cls, str) or not cls:
                raise SchemaError(
                    f"profile.routing.intended_workloads[{i}]={cls!r} "
                    f"must be a non-empty str"
                )
            if cls in seen:
                raise SchemaError(
                    f"profile.routing.intended_workloads contains duplicate "
                    f"{cls!r}"
                )
            seen.add(cls)


@dataclass
class ValidationArtifactRef:
    """Reference to a spec-decode functional artifact that bench-
    validated this profile's runtime semantics.

    Pin-invariant by design: artifact's ``config_hash`` is computed
    over (model_id, kv_plan, K, drafter_backend) excluding the vLLM
    pin, so this reference stays valid across pin bumps. Strict pin
    matching is opt-in via ``sndr profile validate --strict-pin``
    (later).
    """
    artifact_id: str
    config_hash: str

    def validate(self) -> None:
        _check_id(self.artifact_id, "profile.validation.artifact_id")
        if not isinstance(self.config_hash, str):
            raise SchemaError(
                "profile.validation.config_hash must be a string"
            )
        h = self.config_hash.strip()
        if not h:
            raise SchemaError(
                "profile.validation.config_hash must be a non-empty hex string"
            )
        if not re.fullmatch(r"[0-9a-fA-F]+", h):
            raise SchemaError(
                f"profile.validation.config_hash={h!r} must be hex "
                f"(letters [0-9a-f])"
            )


# Allowed values for ProfileDef.role.
# - default:    role producing the production-safe upstream (TQ-only, MTP OFF)
# - structured: role producing the MTP / spec-decode upstream
# - gateway:    role producing the FastAPI reverse-proxy in front of the
#               above pair
# None preserves the existing 17 builtin profiles which are pure
# (model × hardware) operator-tuning presets.
PROFILE_ROLES = ("default", "structured", "gateway")


@dataclass
class ProfileDef:
    """Patches delta layered on top of a specific model's canonical set.

    Owns: patches_delta, optional sizing tweaks (operator tuning for the
    (model × hardware) pair), optional version overrides. Does NOT touch
    identity / capabilities — those are model-owned.

    Lifecycle: experimental → validated → promoted (delta merged into model).

    Why sizing_override here (not in hardware): sizing knobs depend on the
    SPECIFIC model on a specific rig (35B on 2×A5000 → max_num_seqs=2;
    27B on same rig → max_num_seqs=4). Profile is the right "operator
    tuning" layer; hardware just declares physical capacity.

    Runtime-role fields (P1.1, 2026-05-20)
    -------------------------------------
    The optional fields below extend ProfileDef from "tuning preset" to
    "runtime role". A profile with ``role=None`` (the default) behaves
    exactly as the prior 17 builtin profiles do — patches_delta +
    sizing_override only. A profile with ``role`` set additionally
    declares spec-decode / compression / backend / routing / validation
    semantics that the composer renders into the launcher env.

    Why optional: all 17 existing builtin profiles set role=None
    implicitly (the field is absent in their YAMLs) and continue to
    work unchanged. See SNDR_RUNTIME_PROFILES_DESIGN_DECISIONS_2026-05-20
    §4 for the full design.

    The ``spec_decode_override`` field reuses V1 ``SpecDecodeConfig``
    rather than introducing a dedicated dataclass — it's literally a
    profile-side spec-decode config (method, K, drafter, sample rules)
    so the V1 type is the right shape and gives single-source-of-truth
    semantics.
    """
    schema_version: int
    kind: Literal["profile"]
    id: str
    parent_model: str
    maintainer: str
    status: Literal["experimental", "validated", "promoted"] = "experimental"
    created: Optional[str] = None

    patches_delta: PatchesDelta = field(default_factory=PatchesDelta)
    sizing_override: Optional["HardwareSizing"] = None
    versions_override: Optional[ProfileVersionsOverride] = None
    promotion: Optional[ProfilePromotion] = None

    role: Optional[Literal["default", "structured", "gateway"]] = None
    spec_decode_override: Optional[SpecDecodeConfig] = None
    compression_plan: Optional[CompressionPlanConfig] = None
    backend_plan: Optional[BackendPlanConfig] = None
    routing: Optional[RoutingConfig] = None
    validation: Optional[ValidationArtifactRef] = None

    def validate(self) -> None:
        _check_schema_version(self.schema_version)
        _check_kind(self.kind, "profile")
        _check_id(self.id, "profile.id")
        if not self.parent_model:
            raise SchemaError("profile.parent_model required (must reference a ModelDef.id)")
        _check_id(self.parent_model, "profile.parent_model")
        if not self.maintainer:
            raise SchemaError("profile.maintainer required")
        self.patches_delta.validate()
        if self.status not in ("experimental", "validated", "promoted"):
            raise SchemaError(
                f"profile.status={self.status!r} must be experimental|validated|promoted"
            )

        # Runtime-role fields — all Optional, default None preserves prior behavior.
        if self.role is not None and self.role not in PROFILE_ROLES:
            raise SchemaError(
                f"profile.role={self.role!r} must be one of {PROFILE_ROLES} or None"
            )
        if self.spec_decode_override is not None:
            self.spec_decode_override.validate()
        if self.compression_plan is not None:
            self.compression_plan.validate()
        if self.backend_plan is not None:
            self.backend_plan.validate()
        if self.routing is not None:
            self.routing.validate()
        if self.validation is not None:
            self.validation.validate()


# ─── PatchManifest (community SDK) ───────────────────────────────────────


@dataclass
class PatchCompatibility:
    """Per-patch version / arch gates (research lesson 5).

    Used by community SDK validator + composer to decide if a patch
    is eligible on the current (model, hardware) pair.
    """
    min_vllm_pin: Optional[str] = None
    max_vllm_pin: Optional[str] = None
    min_sndr_core_version: Optional[str] = None
    max_sndr_core_version: Optional[str] = None
    model_arch_required: list[str] = field(default_factory=list)
    cuda_capability_min: Optional[tuple[int, int]] = None


@dataclass
class PatchAnchor:
    """Structural anchor (research lesson 7).

    Anchors on qualified names + surrounding context, NEVER raw line numbers.
    """
    id: str
    context_before: str
    context_after: str
    operation: Literal[
        "wrap_return", "replace_block", "insert_before", "insert_after",
    ]
    what_we_do: str


@dataclass
class PatchTargetFile:
    """Upstream source location a patch modifies."""
    path: str                      # repo-relative
    target_module: str             # dotted module path
    target_callable: Optional[str] = None
    context_md5: Optional[str] = None
    pristine_fixture: Optional[str] = None
    anchors: list[PatchAnchor] = field(default_factory=list)


@dataclass
class PatchManifest:
    """Self-describing community patch (PROJECT_ROADMAP_V2 § 4.5).

    Lives next to the plugin code at `plugins/community/<user>/<id>/manifest.yaml`.
    Loaded by the community SDK validator; composer consults compatibility +
    conflicts when assembling the final patches set.
    """
    schema_version: int
    kind: Literal["patch"]
    id: str
    namespace: str

    title: str
    maintainer: str
    version: str                   # semver (Q6 decision)
    license: str
    created: Optional[str] = None

    lifecycle: Literal[
        "community-test", "community-validated", "promoted", "retired",
    ] = "community-test"
    # Supplement §4 (no-stub policy): two orthogonal axes — implementation
    # readiness (what the code does) and publish state (whether it should
    # appear in any release registry). `draft` patches MUST NOT ship in
    # tracked builtin registry; CLI surfaces them only as work-in-progress.
    implementation_status: Literal[
        "experimental", "beta", "stable", "deprecated", "disabled",
    ] = "experimental"
    publish_state: Literal[
        "draft", "review", "published", "rejected",
    ] = "draft"

    type: Literal["runtime_hook", "text_patch", "composite"] = "runtime_hook"
    family: str = "other"
    env_flag: Optional[str] = None
    default_on: bool = False

    compatibility: PatchCompatibility = field(default_factory=PatchCompatibility)
    target_files: list[PatchTargetFile] = field(default_factory=list)

    conflicts_with: list[str] = field(default_factory=list)
    requires_patches: list[str] = field(default_factory=list)
    marker_attr: Optional[str] = None

    entry_points: dict[str, str] = field(default_factory=dict)
    tests_required: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    _SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[+\-][A-Za-z0-9.\-]+)?$")

    def validate(self) -> None:
        _check_schema_version(self.schema_version)
        _check_kind(self.kind, "patch")
        _check_patch_id(self.id, "patch.id")
        if not self.namespace.startswith("community/") and self.namespace not in (
            "official", "core",
        ):
            raise SchemaError(
                f"patch.namespace={self.namespace!r} must start with `community/` "
                f"or be one of `official`/`core`"
            )
        if not self._SEMVER_RE.match(self.version):
            raise SchemaError(
                f"patch.version={self.version!r} must be semver (MAJOR.MINOR.PATCH)"
            )
        if self.default_on and not self.env_flag:
            raise SchemaError(
                "patch.default_on=True requires an env_flag (otherwise operator can't disable)"
            )
        if self.type == "text_patch" and not self.target_files:
            raise SchemaError("patch.type='text_patch' requires target_files")
        if self.type == "runtime_hook" and self.entry_points.get("apply") is None:
            raise SchemaError(
                "patch.type='runtime_hook' requires entry_points.apply"
            )
        # Supplement §4 (no-stub policy): a `default_on` patch must be at
        # least `published` — operators won't accept a `draft` that auto-runs.
        if self.default_on and self.publish_state != "published":
            raise SchemaError(
                f"patch.default_on=True with publish_state={self.publish_state!r}: "
                "auto-enabled patches must be `published` (release-ready)"
            )

    def is_release_eligible(self) -> bool:
        """Supplement §4 gate: only `published` patches ship in tracked
        registries. `draft`/`review`/`rejected` stay in operator playground."""
        return self.publish_state == "published"


__all__ = [
    "SCHEMA_VERSION_V2",
    # ModelDef tree
    "ModelDef", "ModelCapabilities", "ModelRequires", "ModelVersions",
    # HardwareDef tree
    "HardwareDef", "HardwareSizing", "RuntimeBlock",
    "RuntimeDockerBlock", "RuntimeBareMetalBlock",
    # ProfileDef tree
    "ProfileDef", "PatchesDelta",
    "ProfilePromotion", "ProfileVersionsOverride",
    # PatchManifest tree
    "PatchManifest", "PatchCompatibility",
    "PatchAnchor", "PatchTargetFile",
]
