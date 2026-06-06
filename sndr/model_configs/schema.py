# SPDX-License-Identifier: Apache-2.0
"""ModelConfig schema — comprehensive, YAML-backed, validatable.

Every field needed to reproduce + verify a Genesis launch lives here.
No "stuff scattered across launch scripts" — schema is the contract.

M.5.1 (2026-05-27): sub-component dataclasses (HardwareSpec /
SpecDecodeConfig / DockerConfig / … / CompatibilityMatrix /
PatchAttribution — 23 classes in total) were relocated into the
``vllm.sndr_core.model_configs.types`` package and re-exported below.
Historical import paths continue to resolve unchanged:

  * ``from vllm.sndr_core.model_configs.schema import HardwareSpec``
  * ``from vllm.sndr_core.model_configs.schema import SchemaError``
  * ``from vllm.sndr_core.model_configs.schema import COMPATIBILITY_MATRIX``

``ModelConfig`` itself + YAML I/O + emitter methods
(``to_launch_script`` / ``_build_vllm_cmd`` / ``_build_docker_cmd``)
remain in this module for now; M.5.2 + M.5.3 will further decompose
them.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

# ─── Public re-exports from types/ (M.5.1 relocation) ─────────────────
# Every symbol below was previously defined inline in this module.
# Identity is preserved (single SchemaError class, single
# COMPATIBILITY_MATRIX singleton) so ``isinstance`` checks across the
# refactor still resolve to the same class object.
from .types import (  # noqa: F401  (re-exports for back-compat)
    SchemaError,
    HardwareSpec,
    SpecDecodeConfig,
    DockerConfig,
    DeploymentConfig,
    resolve_symbolic_mounts,
    KubernetesConfig,
    ProxmoxConfig,
    BootstrapConfig,
    GpuTuningConfig,
    ObservabilityConfig,
    ServiceConfig,
    PackageSource,
    PackageSources,
    PackageVersions,
    UpstreamPinPolicy,
    OverridesPolicy,
    CacheTier,
    CacheConfig,
    OffloadConfig,
    ReferenceMetrics,
    VerifyTolerances,
    ConfigConstraints,
    RiskScore,
    ArtifactModel,
    ArtifactCache,
    Artifacts,
    PatchAttribution,
    _PATCH_ROLES,
    CompatibilityRule,
    CompatibilityMatrix,
    COMPATIBILITY_MATRIX,
)

log = logging.getLogger("genesis.model_configs.schema")


SCHEMA_VERSION_CURRENT = 1


# ─── Top-level ModelConfig ────────────────────────────────────────────


@dataclass
class ModelConfig:
    """Complete launch + verify contract for one (model × hw × workload)."""
    # Identity
    key: str                                  # kebab-case stable id
    title: str                                # human-readable
    description: str                          # 1-2 sentences
    schema_version: int                       # bump on breaking changes
    maintainer: str                           # github user
    model_path: str                           # /models/...

    # Hardware (required)
    hardware: HardwareSpec = field(default_factory=lambda: HardwareSpec(
        gpu_match_keys=[], n_gpus=0, min_vram_per_gpu_mib=0,
    ))

    # Provenance
    last_validated: Optional[str] = None      # ISO date
    genesis_pin: Optional[str] = None         # commit SHA
    vllm_pin_required: Optional[str] = None   # exact match check

    # Model
    served_model_name: Optional[str] = None
    quantization: Optional[str] = None
    kv_cache_dtype: Optional[str] = None

    # vLLM serve flags (canonical)
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 2
    max_num_batched_tokens: int = 4096
    enable_chunked_prefill: bool = True
    dtype: str = "float16"
    enforce_eager: bool = False
    disable_custom_all_reduce: bool = True
    language_model_only: bool = True
    trust_remote_code: bool = True

    # Structured output
    enable_auto_tool_choice: bool = True
    tool_call_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None

    # Spec decode
    spec_decode: Optional[SpecDecodeConfig] = None

    # Genesis env (P*, PN*, GENESIS_*)
    genesis_env: dict[str, str] = field(default_factory=dict)

    # structured rationale for entries in
    # genesis_env, keyed by registry patch ID (e.g. ``PN204``). Carried
    # through compose() from ModelDef.patches_attribution so the
    # patch_plan resolver and `sndr patches plan --explain` can read
    # role/note/bench_evidence without re-loading the V2 layer. Empty
    # dict is the default for legacy configs that pre-date Phase A.
    patches_attribution: dict[str, "PatchAttribution"] = field(default_factory=dict)

    # System env (PYTORCH_*, VLLM_*, NCCL_*, OMP_*, CUDA_*, TRITON_*)
    system_env: dict[str, str] = field(default_factory=dict)

    # Extra vLLM flags not covered by canonical fields
    vllm_extra_args: list[str] = field(default_factory=list)

    # CUDA graph capture mode. Genesis stack standardizes on
    # FULL_AND_PIECEWISE (vllm default) — both the FULL graph for
    # decode-only batches and PIECEWISE for mixed prefill/decode.
    # Documented as a typed field so it can never be silently dropped
    # from a config; not rendered as a CLI flag because the current
    # vllm pin (0.20.2rc1.dev9) doesn't expose `--cudagraph-mode`.
    # Override only with `enforce_eager: true` as fallback.
    cudagraph_mode: str = "FULL_AND_PIECEWISE"

    # Docker (if absent, render as bare-metal launch)
    docker: Optional[DockerConfig] = None

    # Multi-runtime support (W-runtime 2026-05-06).
    # Default deploy block = docker-only, matching all builtin configs.
    # Configs that ALSO support k8s / podman / lxc / bare-metal flip the
    # respective flag to True. Launcher picks runtime via deploy.default
    # OR `genesis model-config render <key> --runtime <name>` explicitly.
    deploy: DeploymentConfig = field(default_factory=DeploymentConfig)

    # API
    api_key: str = "genesis-local"
    host: str = "0.0.0.0"

    # Reference + tolerances
    reference_metrics: Optional[ReferenceMetrics] = None
    verify_tolerances: VerifyTolerances = field(
        default_factory=VerifyTolerances)

    # ── Community lifecycle (Audit W-A 2026-05-06) ──
    # Flags configs originating from community PRs (vs builtin). Required
    # to be True when lifecycle ∈ {community-test, community-dev, community-prod}.
    community_submitted: bool = False
    # List of verification entries — format: "<rig-tag>@<github-handle>-<ISO-date>".
    # Example: ["rtx-a5000@sandermage-2026-05-06", "rtx-3090@noonghunna-2026-05-08"]
    # community-prod requires ≥2 distinct entries (cross-rig validation).
    verified_by: list[str] = field(default_factory=list)
    # ISO date when this config was first promoted to community-test.
    # Used to gate community-prod promotion (≥7 days stability window).
    test_started_at: Optional[str] = None

    # T1.8 (audit closure §7.2): hardware + flag constraints. The
    # launcher evaluates these against detected hardware BEFORE rendering
    # vllm serve. Missing/None means "no constraint declared".
    constraints: Optional[ConfigConstraints] = None

    # T2.1 (vllm#40270 / PN91): KV cache eviction policy. Default None
    # means "use vLLM stock LRU"; set this to swap in our 2Q or ARC
    # policy via PN91 patch.
    cache_config: Optional[CacheConfig] = None

    # Y1 (UNIFIED_CONFIG plan 2026-05-09): in-container package pins.
    # Default None means "renderer uses the hardcoded legacy baseline"
    # (pandas==2.2.3 scipy==1.14.1 xxhash==3.5.0). Configs that declare
    # this block override the baseline. See PackageVersions docstring
    # for B6 / supply-chain context.
    package_versions: Optional[PackageVersions] = None

    # Y11 (UNIFIED_CONFIG plan 2026-05-09): per-config vLLM pin policy.
    # When set, the launcher checks the running vLLM pin against
    # `upstream.required_pin` / `allowed_pins` / `blocked_pins`
    # BEFORE starting vllm. Empty/None → defer to KNOWN_GOOD_VLLM_PINS
    # project-wide allowlist (legacy behavior).
    upstream: Optional[UpstreamPinPolicy] = None

    # Y12 (UNIFIED_CONFIG plan 2026-05-09): runtime override safety.
    # Declares which env vars are safe for `sndr launch --override
    # KEY=VAL` and what numeric ranges are acceptable. Empty/None →
    # no overrides accepted (safe default).
    overrides: Optional[OverridesPolicy] = None

    # club-3090 #58 Path A (UNIFIED_CONFIG plan 2026-05-09): VRAM→CPU
    # spillover knobs (interim). Translates to `--cpu-offload-gb` at
    # render time. Don't use on hybrid-GDN configs (Mamba SSM state
    # crash — see research report). Path C (v7.73.x) extends this
    # block with tier-aware CacheConfig.
    offload: Optional[OffloadConfig] = None

    # Y3 (UNIFIED_CONFIG plan 2026-05-09): model + cache artifact specs.
    # Replaces fetch_models.sh hardcoded paths and old compat.models.pull
    # registry-tagged lookup. Drives `sndr model pull` + `sndr deps plan`
    # + container mount generation.
    artifacts: Optional[Artifacts] = None

    # Y10 (UNIFIED_CONFIG plan 2026-05-09): service-management contract.
    # Drives `sndr service install/start/stop` (Tier 4 CLI). Empty/None
    # → operator runs the bash script directly without service registration.
    service: Optional[ServiceConfig] = None

    # Y2 (UNIFIED_CONFIG plan 2026-05-09): package-source declarations.
    # Drives `sndr deps install` source-policy: prefer official distro
    # repos; refuse curl|bash unless explicitly opted in.
    package_sources: Optional[PackageSources] = None

    # Y8 (UNIFIED_CONFIG plan 2026-05-09): GPU tuning policy.
    # Drives `sndr tune` (Tier 4 CLI). Power/clocks gated behind
    # explicit unsafe_apply=true. Default fields are safe-only.
    gpu_tuning: Optional[GpuTuningConfig] = None

    # Y14 (UNIFIED_CONFIG plan 2026-05-09): observability declarations.
    # Drives memory_trace + cudagraph dispatch trace + per-patch telemetry.
    observability: Optional[ObservabilityConfig] = None

    # Y5 (UNIFIED_CONFIG plan 2026-05-09): Kubernetes deployment contract.
    # Drives `sndr k8s render/apply/status` (Tier 4 CLI). None → not k8s-ready.
    kubernetes: Optional[KubernetesConfig] = None

    # Y6 (UNIFIED_CONFIG plan 2026-05-09): Proxmox deployment contract.
    # Drives `sndr proxmox doctor/render/apply` (Tier 4 CLI).
    proxmox: Optional[ProxmoxConfig] = None

    # Y7 (UNIFIED_CONFIG plan 2026-05-09): universal-installer driver.
    # Drives `sndr bootstrap apply --scope` (Tier 4 CLI).
    bootstrap: Optional[BootstrapConfig] = None

    # T1.8 (audit closure §7.2): per-dimension risk score for `sndr
    # model-config score <key>` and dashboard ranking. Optional;
    # `derive_overall()` produces a single 0-100 number.
    risk_score: Optional[RiskScore] = None

    # Provenance + notes
    verified_on: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    workload_tag: Optional[str] = None  # 'balanced' / 'long_context' / ...
    lifecycle: str = "stable"
    # lifecycle values:
    #   experimental    — under active dev, not bench-validated yet
    #   tested          — kept for QA/regression testing; NOT a recommended
    #                     production option; excluded from "working configs"
    #                     comparisons by design
    #   stable          — bench-validated; production-ready (built-in tier)
    #   deprecated      — outgoing; kept for migration only
    #   community-test  — JUST submitted via community PR; awaiting initial verify
    #   community-dev   — verified once on submitter rig; awaiting cross-rig
    #   community-prod  — cross-verified ≥2 rigs; ≥7 days stable; reference set
    # See docs/MODEL_CONFIG_LAUNCHER.md → "Community lifecycle" for the
    # full promotion gate and `genesis model-config promote` CLI flow.

    # ── Validation + audit ──

    def validate(self) -> None:
        """Hard schema check — raises SchemaError on any violation.

        M.5.3 restructure (2026-05-27): the original 158-LOC method is
        split into private named helpers below. Error messages,
        exception class, and check ordering are preserved byte-identical
        so existing callers + tests see no behavioural delta.
        """
        self._validate_identity()
        self._validate_community_lifecycle()
        self._validate_cudagraph_mode()
        self._validate_sub_components()
        self._validate_path_c_tier_guard()
        self._validate_compatibility_matrix()

    # ── validate() helpers (M.5.3 internal split) ──

    def _validate_identity(self) -> None:
        """Identity fields: key, schema_version, title, description,
        maintainer, model_path, lifecycle enum."""
        if not self.key:
            raise SchemaError("ModelConfig.key required")
        if self.schema_version != SCHEMA_VERSION_CURRENT:
            raise SchemaError(
                f"ModelConfig.schema_version must be {SCHEMA_VERSION_CURRENT} "
                f"(got {self.schema_version})"
            )
        if not re.match(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$", self.key):
            raise SchemaError(
                f"ModelConfig.key must be kebab-case "
                f"(lowercase letters/digits/hyphens), got '{self.key}'"
            )
        if not self.title or not self.description or not self.maintainer:
            raise SchemaError(
                "ModelConfig requires title, description, maintainer"
            )
        if not self.model_path:
            raise SchemaError("ModelConfig.model_path required")
        if self.lifecycle not in (
            "experimental", "stable", "deprecated", "tested", "retired",
            "community-test", "community-dev", "community-prod",
        ):
            raise SchemaError(
                f"ModelConfig.lifecycle must be one of experimental/stable/"
                f"deprecated/tested/retired/community-test/community-dev/"
                f"community-prod (got '{self.lifecycle}')"
            )

    def _validate_community_lifecycle(self) -> None:
        """Community-lifecycle gates (W-A 2026-05-06): the
        ``community_submitted`` flag and the community-prod
        promotion gates."""
        community_states = {"community-test", "community-dev", "community-prod"}
        if self.community_submitted and self.lifecycle not in community_states:
            raise SchemaError(
                f"ModelConfig.community_submitted=True requires lifecycle ∈ "
                f"{sorted(community_states)} (got '{self.lifecycle}'). "
                f"If this is a builtin config, set community_submitted=False; "
                f"otherwise fix lifecycle to a community-* state."
            )
        if self.lifecycle == "community-prod":
            if not self.reference_metrics:
                raise SchemaError(
                    "ModelConfig.lifecycle='community-prod' requires "
                    "reference_metrics to be set (capture via "
                    "`genesis model-config bench-and-update <key>`)."
                )
            if len(self.verified_by) < 2:
                raise SchemaError(
                    f"ModelConfig.lifecycle='community-prod' requires ≥2 "
                    f"distinct verified_by entries (cross-rig validation). "
                    f"Got {len(self.verified_by)} entries: {self.verified_by}."
                )

    def _validate_cudagraph_mode(self) -> None:
        """``cudagraph_mode`` enum check."""
        valid_cg = {"NONE", "PIECEWISE", "FULL", "FULL_AND_PIECEWISE",
                    "FULL_DECODE_ONLY"}
        if self.cudagraph_mode not in valid_cg:
            raise SchemaError(
                f"ModelConfig.cudagraph_mode must be one of "
                f"{sorted(valid_cg)} (got '{self.cudagraph_mode}')"
            )

    def _validate_sub_components(self) -> None:
        """Delegate validation to each present sub-component dataclass.

        Order preserved from the pre-M.5.3 monolith so any test that
        depends on which error surfaces first sees the same path. The
        ``OffloadConfig`` branch additionally runs the Path A hybrid-GDN
        guard inline because the guard reads ``offload.cpu_offload_gib``
        AND ``cache_config.tiers`` together.
        """
        self.hardware.validate()
        if self.spec_decode is not None:
            self.spec_decode.validate()
        if self.docker is not None:
            self.docker.validate()
        self.deploy.validate()  # W-runtime 2026-05-06
        self.verify_tolerances.validate()
        if self.constraints is not None:
            self.constraints.validate()
        if self.risk_score is not None:
            self.risk_score.validate()
        if self.cache_config is not None:
            self.cache_config.validate()
        if self.package_versions is not None:
            self.package_versions.validate()
        if self.upstream is not None:
            self.upstream.validate()
        if self.overrides is not None:
            self.overrides.validate()
        if self.offload is not None:
            self.offload.validate()
            self._validate_offload_hybrid_gdn_guard()
        if self.artifacts is not None:
            self.artifacts.validate()
        if self.service is not None:
            self.service.validate()
        if self.package_sources is not None:
            self.package_sources.validate()
        if self.gpu_tuning is not None:
            self.gpu_tuning.validate()
        if self.observability is not None:
            self.observability.validate()
        if self.kubernetes is not None:
            self.kubernetes.validate()
        if self.proxmox is not None:
            self.proxmox.validate()
        if self.bootstrap is not None:
            self.bootstrap.validate()

    def _validate_offload_hybrid_gdn_guard(self) -> None:
        """Hybrid-GDN guard (Path A): CPU offload + hybrid GDN crashes
        in vLLM/SGLang/LMCache. Detect by PN59 streaming-GDN env
        being set on this config (canonical hybrid signal)."""
        uses_hybrid_gdn = (
            "1" == self.genesis_env.get(
                "GENESIS_ENABLE_PN59_STREAMING_GDN", "")
        )
        # Path C relaxation (PN95 v7.73.x): Path A is gated unless
        # cache_config.tiers is declared AND exclude_mamba_ssm=True.
        # PN95's tier manager filters MambaSpec groups out of the
        # demote candidate set, so SSM state never gets touched.
        path_c_active = (
            self.cache_config is not None
            and self.cache_config.tiers
            and self.cache_config.exclude_mamba_ssm
        )
        if (uses_hybrid_gdn and self.offload.cpu_offload_gib > 0
                and not path_c_active):
            raise SchemaError(
                "OffloadConfig.cpu_offload_gib > 0 is incompatible "
                "with hybrid-GDN models (PN59 enabled). Mamba SSM "
                "state lives outside the KV pool and CPU offload "
                "crashes upstream. See "
                "docs/_internal/research/club3090_issue58_long_ctx_"
                "vision_oom_2026-05-09.md for the full analysis. "
                "v7.73.x Path C lifts this restriction — declare "
                "`cache_config.tiers` with `exclude_mamba_ssm: true` "
                "(default true) to use the PN95 tier manager that "
                "filters MambaSpec groups out of demotion."
            )

    def _validate_path_c_tier_guard(self) -> None:
        """Path C: hybrid-GDN configs that opt INTO PN95 tiers MUST
        keep ``exclude_mamba_ssm=True`` (refusing to override is a
        deliberate safety belt — the validator should never let a bad
        config reach the dispatcher)."""
        uses_hybrid_gdn = (
            "1" == self.genesis_env.get(
                "GENESIS_ENABLE_PN59_STREAMING_GDN", "")
        )
        if (uses_hybrid_gdn and self.cache_config is not None
                and self.cache_config.tiers
                and not self.cache_config.exclude_mamba_ssm):
            raise SchemaError(
                "CacheConfig.exclude_mamba_ssm=False is incompatible "
                "with hybrid-GDN models (PN59 enabled). PN95 must "
                "exclude MambaSpec groups from demotion or the SSM "
                "state corrupts. Either remove `cache_config.tiers` "
                "(disables Path C) OR set `exclude_mamba_ssm: true`."
            )

    def _validate_compatibility_matrix(self) -> None:
        """S2.5 (2026-05-12): CompatibilityMatrix forbidden rules as
        hard error. Discouraged rules surface via ``audit()`` as soft
        warnings; see :func:`model_config_audit.audit_model_config`."""
        forbidden, _ = COMPATIBILITY_MATRIX.evaluate(self)
        if forbidden:
            lines = [
                f"[{rule.id}] {rule.title}: {msg} → {rule.mitigation}"
                for rule, msg in forbidden
            ]
            raise SchemaError(
                "CompatibilityMatrix violations:\n  - "
                + "\n  - ".join(lines)
            )

    def audit(self) -> list[str]:
        """Soft warnings for risky-but-not-invalid configurations.

        Thin delegation to
        :func:`vllm.sndr_core.model_configs.model_config_audit.audit_model_config`
        — operator-visible message wording lives there.
        """
        from .model_config_audit import audit_model_config

        return audit_model_config(self)

    # ── Render (M.5.2: thin delegations to ``model_configs.emitters``) ──

    def to_launch_script(
        self,
        host_paths: Optional[dict[str, str]] = None,
        *,
        strict_mounts: bool = False,
    ) -> str:
        """Render this config as an executable bash launch script.

        Thin delegation to
        :func:`vllm.sndr_core.model_configs.emitters.render_launch_script`
        — see that function for the full docstring + behaviour notes
        (P0-8 / F-016 history etc.).
        """
        from .emitters import render_launch_script

        return render_launch_script(
            self, host_paths, strict_mounts=strict_mounts,
        )

    def _build_vllm_cmd(self) -> list[str]:
        """vllm serve command parts (without exec/docker prefix).

        Thin delegation to
        :func:`vllm.sndr_core.model_configs.emitters.build_vllm_cmd`.
        Kept as a method for back-compat with existing internal callers.
        """
        from .emitters import build_vllm_cmd

        return build_vllm_cmd(self)

    def _build_docker_cmd(
        self,
        vllm_parts: list[str],
        host_paths: Optional[dict[str, str]] = None,
        *,
        strict_mounts: bool = False,
    ) -> str:
        """Render docker run command embedding the vllm serve.

        Thin delegation to
        :func:`vllm.sndr_core.model_configs.emitters.build_docker_cmd`.
        """
        from .emitters import build_docker_cmd

        return build_docker_cmd(
            self, vllm_parts, host_paths=host_paths,
            strict_mounts=strict_mounts,
        )


# ─── YAML I/O (M.5.2: thin re-exports from ``model_configs.emitters``) ──
#
# The implementations live in ``emitters.yaml_io``; the names below are
# preserved at module scope so existing import paths continue to work:
#
#   from vllm.sndr_core.model_configs.schema import dump_yaml, load_yaml
#   from vllm.sndr_core.model_configs import dump_yaml, load_yaml
#
# The private helpers ``_to_plain_dict`` / ``_from_plain_dict`` /
# ``_shell_quote`` retain their leading-underscore names here too, for
# any third-party callers that imported them by their pre-M.5.2 names.
from .emitters import dump_yaml, load_yaml  # noqa: E402,F401
from .emitters import from_plain_dict as _from_plain_dict  # noqa: E402,F401
from .emitters import shell_quote as _shell_quote  # noqa: E402,F401
from .emitters import to_plain_dict as _to_plain_dict  # noqa: E402,F401
from .emitters import validate_cfg as validate  # noqa: E402,F401
