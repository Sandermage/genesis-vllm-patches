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
import shlex
from dataclasses import asdict, dataclass, field, fields
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
        """Hard schema check — raises SchemaError on any violation."""
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

        # ── Community lifecycle gates (W-A 2026-05-06) ──
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
        valid_cg = {"NONE", "PIECEWISE", "FULL", "FULL_AND_PIECEWISE",
                    "FULL_DECODE_ONLY"}
        if self.cudagraph_mode not in valid_cg:
            raise SchemaError(
                f"ModelConfig.cudagraph_mode must be one of "
                f"{sorted(valid_cg)} (got '{self.cudagraph_mode}')"
            )

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
            # Hybrid-GDN guard (Path A): CPU offload + hybrid GDN crashes
            # in vLLM/SGLang/LMCache. Detect by PN59 streaming-GDN env
            # being set on this config (canonical hybrid signal).
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
        # Path C: hybrid-GDN configs that opt INTO PN95 tiers MUST keep
        # exclude_mamba_ssm=True (refusing to override is a deliberate
        # safety belt — the validator should never let a bad config
        # reach the dispatcher).
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

        # S2.5 (2026-05-12): CompatibilityMatrix forbidden rules как hard
        # error. Discouraged уходят в audit() как soft warnings.
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

        Examples: TQ k8v4 + hybrid model without P98, --enable-prefix-
        caching on hybrid GDN, etc. Operator can choose to ignore.
        """
        warnings: list[str] = []
        # TQ k8v4 + hybrid GDN model needs P98 (vs vllm#40941 lock).
        # Hybrid GDN models: 27B Lorbus int4, NOT 35B-A3B-FP8 (dense MoE).
        # Detection: PN59_STREAMING_GDN=1 in env is the canonical signal —
        # operator only enables PN59 on hybrid models.
        if self.kv_cache_dtype == "turboquant_k8v4":
            pn59_on = self.genesis_env.get(
                "GENESIS_ENABLE_PN59_STREAMING_GDN") == "1"
            int4_lorbus = "int4" in self.model_path.lower() and \
                "AutoRound" in self.model_path
            if (pn59_on or int4_lorbus) and \
                    "GENESIS_ENABLE_P98" not in self.genesis_env:
                warnings.append(
                    "P98 should be enabled for TQ k8v4 + hybrid GDN model "
                    "(WorkspaceManager fix vs vllm#40941). "
                    "Add GENESIS_ENABLE_P98=1 to genesis_env."
                )
        # Reference metrics expected for stable lifecycle
        if self.lifecycle == "stable" and self.reference_metrics is None:
            warnings.append(
                "stable lifecycle should have reference_metrics — "
                "operators can't run `verify` without baseline values."
            )
        # S2.5 (2026-05-12): CompatibilityMatrix discouraged rules.
        _, discouraged = COMPATIBILITY_MATRIX.evaluate(self)
        for rule, msg in discouraged:
            warnings.append(f"[{rule.id}] {rule.title}: {msg}")
        return warnings

    # ── Render ──

    def to_launch_script(
        self,
        host_paths: Optional[dict[str, str]] = None,
        *,
        strict_mounts: bool = False,
    ) -> str:
        """Render this config as an executable bash launch script.

        Output is either docker-based (if self.docker set) or bare-metal
        depending on the config. Either way: env vars exported, vllm
        serve called with all flags.

        Args:
            host_paths: optional mapping of symbolic mount variables
                to absolute paths. Used to resolve `${models_dir}`,
                `${hf_cache}` etc. in `docker.mounts`. If None, tries
                to load `host.yaml` lazily — but only if any mount
                actually contains a `${var}` reference. Configs with
                fully-absolute mounts work without a host config.
            strict_mounts: if True, raise SchemaError on any unresolved
                `${var}` reference in `docker.mounts`. Live launch
                paths should pass True so missing host.yaml entries
                fail loudly instead of producing an unbootable script.
                Default False — `--dry-run` previews can leave
                placeholders for documentation purposes.

        P0-8 fix (audit 2026-05-08): `strict_mounts` threaded through
        so live launch ≠ dry-run preview. Previously both paths used
        strict=False and a Docker config with missing host.yaml
        rendered an unbootable script with literal `${models_dir}`.

        F-016 fix (audit 2026-05-07): previously `_build_docker_cmd`
        embedded mounts as-is, so configs using symbolic refs got
        `${models_dir}` literally in the docker cmd → boot failure.
        """
        lines = [
            "#!/usr/bin/env bash",
            "# Generated by Genesis model_config:",
            f"#   key:           {self.key}",
            f"#   title:         {self.title}",
            f"#   maintainer:    {self.maintainer}",
            f"#   schema_v:      {self.schema_version}",
        ]
        if self.last_validated:
            lines.append(f"#   last_validated: {self.last_validated}")
        if self.genesis_pin:
            lines.append(f"#   genesis_pin:   {self.genesis_pin}")
        if self.vllm_pin_required:
            lines.append(f"#   vllm_pin:      {self.vllm_pin_required}")
        if self.reference_metrics:
            rm = self.reference_metrics
            lines.append(
                f"#   reference:     {rm.long_gen_sustained_tps:.1f} TPS "
                f"sustained / {rm.tool_call_score} tool / "
                f"CV {rm.stability_cv_pct:.2f}% / "
                f"VRAM {rm.vram_total_mib} MiB"
            )
        for note in self.notes:
            lines.append(f"#   note: {note}")
        lines.extend(["", "set -euo pipefail", ""])

        # System env
        if self.system_env:
            lines.append("# System env")
            for k, v in sorted(self.system_env.items()):
                lines.append(f'export {k}={_shell_quote(v)}')
            lines.append("")

        # Genesis env
        if self.genesis_env:
            lines.append("# Genesis patcher env")
            for k, v in sorted(self.genesis_env.items()):
                lines.append(f'export {k}={_shell_quote(v)}')
            lines.append("")

        # Build vllm serve cmd
        cmd_parts = self._build_vllm_cmd()

        # Docker or bare-metal launch
        if self.docker:
            lines.append("# Docker launch")
            docker_cmd = self._build_docker_cmd(
                cmd_parts, host_paths=host_paths,
                strict_mounts=strict_mounts,
            )
            lines.append(docker_cmd)
        else:
            lines.append("# Bare-metal launch")
            lines.append("exec " + " \\\n  ".join(cmd_parts))

        return "\n".join(lines) + "\n"

    def _build_vllm_cmd(self) -> list[str]:
        """vllm serve command parts (without exec/docker prefix)."""
        parts = [
            "vllm serve",
            f"--model {_shell_quote(self.model_path)}",
            f"--tensor-parallel-size {self.hardware.n_gpus}",
            f"--gpu-memory-utilization {self.gpu_memory_utilization}",
            f"--max-model-len {self.max_model_len}",
            f"--max-num-seqs {self.max_num_seqs}",
            f"--max-num-batched-tokens {self.max_num_batched_tokens}",
            f"--dtype {_shell_quote(self.dtype)}",
        ]
        if self.kv_cache_dtype:
            parts.append(f"--kv-cache-dtype {_shell_quote(self.kv_cache_dtype)}")
        if self.quantization:
            parts.append(f"--quantization {_shell_quote(self.quantization)}")
        if self.served_model_name:
            parts.append(f"--served-model-name {_shell_quote(self.served_model_name)}")
        if self.tool_call_parser:
            parts.append(f"--tool-call-parser {_shell_quote(self.tool_call_parser)}")
        if self.reasoning_parser:
            parts.append(f"--reasoning-parser {_shell_quote(self.reasoning_parser)}")
        if self.enable_chunked_prefill:
            parts.append("--enable-chunked-prefill")
        if self.enforce_eager:
            parts.append("--enforce-eager")
        if self.disable_custom_all_reduce:
            parts.append("--disable-custom-all-reduce")
        if self.language_model_only:
            parts.append("--language-model-only")
        if self.trust_remote_code:
            parts.append("--trust-remote-code")
        if self.enable_auto_tool_choice:
            parts.append("--enable-auto-tool-choice")
        parts.append(f"--api-key {_shell_quote(self.api_key)}")
        parts.append(f"--host {_shell_quote(self.host)}")
        if self.docker:
            # Y4: pass container-side port to vllm serve (the port it
            # listens on inside the container). Falls back to legacy
            # `port` field when host_port/container_port are not split.
            parts.append(f"--port {self.docker.effective_container_port()}")
        if self.spec_decode:
            parts.append(
                f"--speculative-config '{self.spec_decode.to_vllm_arg()}'"
            )
        for extra in self.vllm_extra_args:
            parts.append(extra)
        # club-3090 #58 Path A: cpu offload knobs become engine flags.
        # OffloadConfig.validate() already blocked hybrid-GDN combos.
        if self.offload is not None:
            parts.extend(self.offload.to_vllm_args())
        return parts

    def _build_docker_cmd(
        self,
        vllm_parts: list[str],
        host_paths: Optional[dict[str, str]] = None,
        *,
        strict_mounts: bool = False,
    ) -> str:
        """Render docker run command embedding the vllm serve.

        Mounts containing `${var}` symbolic references are resolved
        through `host_paths` (or lazy-loaded `host.yaml`) before being
        embedded in the docker `-v` flags. Mounts that are fully
        absolute paths pass through unchanged.

        Args:
            strict_mounts: when True, raises SchemaError on any
                unresolved `${var}`. Set by `to_launch_script` for the
                live launch path (P0-8 audit 2026-05-08).
        """
        d = self.docker
        # Resolve symbolic mounts. Lazy-load host.yaml only if any mount
        # actually uses `${var}` — configs with fully absolute mounts
        # don't need a host config to render.
        #
        # Resolution order when host_paths is None:
        #   1. ~/.sndr/host.yaml (explicit operator config)
        #   2. host.detect_paths() (auto-probe common locations)
        #   3. unresolved → SchemaError with actionable message
        # Step 2 lets tests + dev machines render without setting up
        # host.yaml. detect_paths() probes _DEFAULT_*_CANDIDATES and
        # returns absolute paths for those that exist on this host.
        # Variables it can't find stay unresolved → SchemaError, which
        # is the correct outcome (operator must fix host.yaml).
        resolved_mounts = list(d.mounts)
        needs_resolution = any("${" in m for m in d.mounts)
        if needs_resolution:
            if host_paths is None:
                # Lazy import: host.py touches PyYAML, keep it off the
                # cold path for callers that pass host_paths explicitly.
                from .host import load_host_config, detect_paths
                merged: dict[str, str] = {}
                try:
                    merged.update(detect_paths())
                except Exception:
                    pass
                try:
                    merged.update(load_host_config().paths)
                except Exception:
                    pass
                host_paths = merged
            # P0-8 (audit 2026-05-08): live launch passes strict_mounts=
            # True so unresolved vars raise SchemaError with a clear
            # "fix host.yaml" message. `--dry-run` paths use False to
            # preserve the preview-with-placeholders behavior.
            resolved_mounts = resolve_symbolic_mounts(
                d.mounts, host_paths, strict=strict_mounts,
            )

        lines = [
            f"docker rm -f {_shell_quote(d.container_name)} 2>/dev/null || true",
            "",
            "docker run -d \\",
            f"  --name {_shell_quote(d.container_name)} \\",
            "  --entrypoint /bin/bash \\",
            f"  --gpus {_shell_quote(d.gpus)} \\",
            f"  --shm-size={_shell_quote(d.shm_size)} \\",
        ]
        if d.memory_limit:
            lines.append(f"  --memory={_shell_quote(d.memory_limit)} \\")
        if d.network:
            lines.append(f"  --network {_shell_quote(d.network)} \\")
        # Y4: HOST:CONTAINER port mapping. Falls back to legacy
        # `port:port` when host_port/container_port are not split.
        lines.append(
            f"  -p {d.effective_host_port()}:{d.effective_container_port()} \\"
        )
        for m in resolved_mounts:
            lines.append(f"  -v {_shell_quote(m)} \\")
        for f in d.extra_run_flags:
            lines.append(f"  {f} \\")
        # Env vars
        for k, v in sorted(self.system_env.items()):
            lines.append(f'  -e {k}={_shell_quote(v)} \\')
        for k, v in sorted(self.genesis_env.items()):
            lines.append(f"  -e {k}={_shell_quote(v)} \\")
        # Image + cmd
        lines.append(f"  {_shell_quote(d.effective_image_ref())} \\")
        # Bash -c with canonical apply + exec vllm serve.
        # POSIX-escape single quotes inside the inner cmd so the outer
        # single-quoted -c '...' wrapper survives JSON args like
        # --speculative-config '{"method":"mtp",...}'.
        cmd = " ".join(vllm_parts)
        cmd_escaped = cmd.replace("'", "'\\''")
        # Build the bash bootstrap. If the operator mounts the genesis
        # plugin source at /plugin, install it in editable mode so its
        # `vllm.general_plugins` entry point auto-loads inside every
        # vllm worker process. Without this, patches only apply via the
        # explicit `apply` invocation — plugin-only paths (boot
        # banner, config detection) won't fire.
        has_plugin = any(
            ":/plugin" in m or m.endswith("/plugin")
            for m in d.mounts
        )
        # P0-8 (audit 2026-05-08): single canonical apply entrypoint.
        # The legacy apply-all fallback was a no-op
        # (module never existed in v10/v11) and silently masked any
        # apply failure as a successful sub-shell. Now the call is
        # direct — boot fails loudly if sndr_core is unimportable.
        apply_step = "python3 -m vllm.sndr_core.apply 2>&1 | tail -5"
        # P1-7 fix (audit 2026-05-08) + B6 (UNIFIED_CONFIG plan 2026-05-09):
        # runtime deps inside the container are pinned. Y1 introduced
        # `package_versions.python_packages` as the canonical source of
        # truth — when present it wins; otherwise the legacy hardcoded
        # baseline below is used. Operators can opt out via
        # `SNDR_DEV_INSTALL_RUNTIME_DEPS=1` for editable / dev workflows.
        runtime_deps = ""
        if self.package_versions is not None:
            runtime_deps = self.package_versions.to_pip_args()
        if not runtime_deps:
            runtime_deps = "pandas==2.2.3 scipy==1.14.1 xxhash==3.5.0"
        # DA-008 fix (audit 2026-05-08): production launch path NO LONGER
        # depends on the `/plugin` mount being present.
        #
        # Rationale: in production, `vllm-sndr-core` should already be
        # installed inside the container (via the wheel pip-installed at
        # image build time, or via a base image including it). Mounting
        # `/plugin` and pip-install'ing it at every container start is:
        #   - non-reproducible (whatever is in the operator's local repo wins);
        #   - slow (pip install adds ~10-30s to cold boot);
        #   - a supply-chain risk (operator's local edits become live).
        #
        # The new contract:
        #   - Production: the canonical apply step (`python3 -m
        #     vllm.sndr_core.apply`) is the ONLY thing run. If
        #     vllm-sndr-core isn't installed, the call fails loudly.
        #   - Dev: opt in to the legacy `/plugin` install via
        #     `SNDR_DEV_INSTALL_PLUGIN=1`. The original behavior is
        #     preserved verbatim under the env gate.
        #
        # `has_plugin` (presence of `/plugin` in mounts) used to
        # automatically TRIGGER the install. Now it just makes the dev
        # install POSSIBLE; the env flag must also be set.
        bootstrap_parts = ["set -euo pipefail"]
        # Optional dev-mode pinned runtime deps (P1-7).
        bootstrap_parts.append(
            'if [ "${SNDR_DEV_INSTALL_RUNTIME_DEPS:-0}" = "1" ]; then '
            f'pip install --quiet {runtime_deps} 2>&1 | tail -2; '
            'fi'
        )
        # Optional dev-mode plugin install (DA-008).
        if has_plugin:
            bootstrap_parts.append(
                'if [ "${SNDR_DEV_INSTALL_PLUGIN:-0}" = "1" ]; then '
                "cp -r /plugin /tmp/genesis_vllm_plugin && "
                "pip install --quiet --disable-pip-version-check "
                "--root-user-action=ignore --no-deps -e "
                "/tmp/genesis_vllm_plugin 2>&1 | tail -2; "
                'fi'
            )
        # Canonical apply step (always runs).
        bootstrap_parts.append(apply_step)
        bootstrap_parts.append(f"exec {cmd_escaped}")
        bootstrap = "; ".join(bootstrap_parts)
        lines.append(f"  -c '{bootstrap}'")
        return "\n".join(lines)


# ─── YAML I/O ─────────────────────────────────────────────────────────


def dump_yaml(cfg: ModelConfig) -> str:
    """Serialize ModelConfig → YAML string."""
    import yaml
    cfg.validate()
    d = _to_plain_dict(cfg)
    return yaml.safe_dump(d, sort_keys=False, allow_unicode=True,
                          default_flow_style=False)


def load_yaml(text: str) -> ModelConfig:
    """Parse YAML string → ModelConfig with full validation."""
    import yaml
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise SchemaError("YAML must be a mapping at top level")
    return _from_plain_dict(raw)


def validate(cfg: ModelConfig) -> ModelConfig:
    """Validate ModelConfig in-place; raise SchemaError on issues.
    Returns the validated config for chainable use."""
    cfg.validate()
    return cfg


# ─── Internal helpers ─────────────────────────────────────────────────


def _shell_quote(value: str) -> str:
    """Quote a value so generated shell commands preserve it exactly."""
    return shlex.quote(str(value))


def _to_plain_dict(cfg: ModelConfig) -> dict:
    return asdict(cfg)


def _from_plain_dict(d: dict) -> ModelConfig:
    """Reconstruct ModelConfig from plain dict (post-YAML-load)."""
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
