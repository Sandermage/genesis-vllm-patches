# SPDX-License-Identifier: Apache-2.0
"""ModelConfig schema — comprehensive, YAML-backed, validatable.

Every field needed to reproduce + verify a Genesis launch lives here.
No "stuff scattered across launch scripts" — schema is the contract.
"""
from __future__ import annotations

import json
import logging
import re
import shlex
from dataclasses import asdict, dataclass, field, fields
from typing import Any, ClassVar, Optional

log = logging.getLogger("genesis.model_configs.schema")


SCHEMA_VERSION_CURRENT = 1


class SchemaError(ValueError):
    """Raised when a ModelConfig (or sub-component) fails validation."""


# ─── Sub-components ───────────────────────────────────────────────────


@dataclass
class HardwareSpec:
    """GPU + system requirements for the config to apply cleanly."""
    gpu_match_keys: list[str]   # ['rtx a5000', 'a100']
    n_gpus: int
    min_vram_per_gpu_mib: int
    cuda_capability_min: Optional[tuple[int, int]] = None  # (8, 6) for Ampere

    def validate(self) -> None:
        if not self.gpu_match_keys:
            raise SchemaError("HardwareSpec.gpu_match_keys must be non-empty")
        if self.n_gpus < 1:
            raise SchemaError(
                f"HardwareSpec.n_gpus must be >= 1 (got {self.n_gpus})"
            )
        if self.min_vram_per_gpu_mib < 1:
            raise SchemaError(
                "HardwareSpec.min_vram_per_gpu_mib must be > 0"
            )


@dataclass
class SpecDecodeConfig:
    """Speculative decoding setup."""
    method: str  # 'mtp' / 'eagle' / 'ngram' / 'dflash'
    num_speculative_tokens: int
    # Path to a separate drafter model. Required for `dflash`/`eagle` (where the
    # drafter is a distinct checkpoint from the target). For `mtp`/`ngram` keep
    # None — vllm uses the target model's own MTP head / its own n-gram cache.
    model: Optional[str] = None

    # Probabilistic draft-probs propagation (vllm dev338+ native).
    #
    # The upstream rejection_sampler now natively supports the
    # `min(1, target_p / draft_p)` accept rule when both fields below are
    # set on the SpeculativeConfig — there is a runtime gate at
    # `LLMBaseProposer._enable_probabilistic_draft_probs` that requires
    # BOTH `rejection_sample_method == "standard"` AND
    # `draft_sample_method == "probabilistic"`. Without those, the
    # drafter falls back to greedy sampling and the verifier gets
    # `None` for draft_probs, so the accept rule degrades to the
    # straight equality check and we lose ~7 % accept_rate / TPS on
    # MTP K=3.
    #
    # This is the upstream-merged replacement for our Genesis PN90
    # backport (which text-patched the rejection_sampler call site
    # itself). PN90 self-retired on dev338 because upstream
    # restructured the anchor — exposing these fields here is the
    # supported migration path.
    #
    # Defaults stay None to keep V1 configs that do not set them
    # bit-identical to their prior behaviour. Operators opt-in via
    # the model YAML's `spec_decode:` block.
    rejection_sample_method: Optional[str] = None  # "standard" | None
    draft_sample_method: Optional[str] = None      # "probabilistic" | None

    # P1.7c (2026-05-20): drafter attention backend — selects the
    # attention kernel the drafter runs with. vLLM v1 SpeculativeConfig
    # accepts an `attention_backend` key; the validated β'-A K=4
    # configuration explicitly sets this to FLASH_ATTN so the drafter
    # routes head_size 256 / 512 layers through G4_71b / G4_75
    # respectively. Without this key the drafter falls back to vLLM's
    # auto-pick (typically TURBOQUANT on a TQ engine) which causes a
    # KV layout / kernel mismatch and breaks acceptance.
    #
    # Known values: FLASH_ATTN | TRITON_ATTN | TURBOQUANT | None.
    # Default None preserves backward compat: pre-P1.7c configs that
    # never set this field render bit-identically to their prior
    # behaviour. Operator opts in via the spec_decode block on
    # ModelDef or via ProfileDef.spec_decode_override.
    attention_backend: Optional[str] = None

    def validate(self) -> None:
        valid_methods = {"mtp", "eagle", "ngram", "dflash"}
        if self.method not in valid_methods:
            raise SchemaError(
                f"SpecDecodeConfig.method must be one of {valid_methods}, "
                f"got '{self.method}'"
            )
        if self.num_speculative_tokens < 1:
            raise SchemaError(
                "SpecDecodeConfig.num_speculative_tokens must be >= 1"
            )
        if self.method in ("dflash", "eagle") and not self.model:
            raise SchemaError(
                f"SpecDecodeConfig.model is required for method='{self.method}' "
                f"(drafter is a separate checkpoint from the target model)"
            )
        # The native probabilistic draft path is documented in
        # vllm.v1.spec_decode.llm_base_proposer at the
        # `_enable_probabilistic_draft_probs` gate. The known-good values
        # at the dev338 pin are listed below; new values become possible
        # if upstream extends the enum without our knowing.
        valid_rejection = {None, "standard"}
        valid_draft_sample = {None, "probabilistic"}
        if self.rejection_sample_method not in valid_rejection:
            raise SchemaError(
                "SpecDecodeConfig.rejection_sample_method must be one of "
                f"{valid_rejection}, got {self.rejection_sample_method!r}"
            )
        if self.draft_sample_method not in valid_draft_sample:
            raise SchemaError(
                "SpecDecodeConfig.draft_sample_method must be one of "
                f"{valid_draft_sample}, got {self.draft_sample_method!r}"
            )
        # P1.7c: attention_backend validates against the known set of
        # vLLM v1 attention backends. None means "do not emit the key"
        # (engine auto-picks).
        valid_attn_backend = {
            None, "FLASH_ATTN", "TRITON_ATTN", "TURBOQUANT",
        }
        if self.attention_backend not in valid_attn_backend:
            raise SchemaError(
                "SpecDecodeConfig.attention_backend must be one of "
                f"{valid_attn_backend}, got {self.attention_backend!r}"
            )

    def to_vllm_arg(self) -> str:
        """Format for --speculative-config flag."""
        d: dict = {
            "method": self.method,
            "num_speculative_tokens": self.num_speculative_tokens,
        }
        if self.model:
            d["model"] = self.model
        # Probabilistic draft-probs path — only emit when both halves of
        # the gate are set, so older vllm pins that lack the upstream
        # native path do not see unknown keys in the config blob.
        if self.rejection_sample_method is not None:
            d["rejection_sample_method"] = self.rejection_sample_method
        if self.draft_sample_method is not None:
            d["draft_sample_method"] = self.draft_sample_method
        # P1.7c: drafter attention backend (FLASH_ATTN, TRITON_ATTN,
        # TURBOQUANT). Emitted only when set; absent key = vLLM auto-pick.
        if self.attention_backend is not None:
            d["attention_backend"] = self.attention_backend
        return json.dumps(d)


@dataclass
class DeploymentConfig:
    """Multi-runtime support — operator picks one at launch.

    2026-05-06 (W-runtime): Genesis configs were docker-only by design.
    Community feedback (noonghunna club-3090 docs/CONTAINER_RUNTIMES.md):
    operators run on Docker, Podman, microk8s, Proxmox LXC + bare-metal
    venv (Proxmox kernel 6.17.x footgun workaround). The schema captures
    which runtimes a config supports so the launcher can render the
    appropriate artifact and `--runtime` CLI flag picks one explicitly.
    """
    # Per-runtime support flags. Operator can render+launch any runtime
    # where the corresponding flag is True. Default = docker only (matches
    # all builtin configs as of 2026-05-06).
    docker: bool = True
    podman: bool = False           # mostly compat with docker (COMPOSE_BIN override)
    kubernetes: bool = False       # microk8s / k3s / k8s — manual translation per noonghunna disc#48
    lxc_proxmox: bool = False      # Proxmox VE LXC — see kernel 6.17.x footgun in noonghunna CONTAINER_RUNTIMES.md
    bare_metal: bool = False       # native venv (Proxmox workaround + minimal-deps option)

    # Default runtime to use when launcher called without --runtime override.
    # Must satisfy `getattr(self, default) is True` (validated below).
    default: str = "docker"

    # Known runtime names — single source of truth for valid `default` values
    # AND for `--runtime` CLI choices. Update if adding a runtime.
    # ClassVar = not a dataclass field, not serialized to YAML, not in __init__.
    KNOWN_RUNTIMES: ClassVar[tuple] = (
        "docker", "podman", "kubernetes", "lxc_proxmox", "bare_metal",
    )

    def supported_runtimes(self) -> list[str]:
        """List of runtimes this config can launch on."""
        return [r for r in self.KNOWN_RUNTIMES if getattr(self, r) is True]

    def validate(self) -> None:
        if self.default not in self.KNOWN_RUNTIMES:
            raise SchemaError(
                f"DeploymentConfig.default='{self.default}' not in known "
                f"runtimes {self.KNOWN_RUNTIMES}"
            )
        supported = self.supported_runtimes()
        if not supported:
            raise SchemaError(
                "DeploymentConfig: at least one runtime must be True. "
                "Got all runtimes False — config can't launch anywhere."
            )
        if not getattr(self, self.default):
            raise SchemaError(
                f"DeploymentConfig.default='{self.default}' not supported by "
                f"this config (deploy.{self.default}=False). "
                f"Supported runtimes: {supported}"
            )


def resolve_symbolic_mounts(
    mounts: list[str],
    host_paths: dict[str, str],
    strict: bool = True,
) -> list[str]:
    """Expand `${var}` references in mount strings via host_paths.

    Each user has different paths (`/mnt/models` vs `/data/models`,
    `~/.cache/huggingface` vs `/var/cache/huggingface`). Configs use
    symbolic refs so they're portable across rigs. Host-specific paths live
    in `~/.sndr/host.yaml` (auto-detected at install or first-run).

    Absolute paths (no `${var}`) pass through unchanged.

    Args:
        mounts: list of "<host_path>:<container_path>[:ro|rw]" strings,
            host_path may contain `${var}` references
        host_paths: dict mapping symbolic var names to absolute paths
        strict: if True (default), raise on unknown var. If False, leave
            `${var}` literal in the output — useful for `render` previews
            on machines that don't have a complete host config yet.

    Returns:
        list of mount strings with all `${var}` resolved (or left as
        literals when `strict=False` and the var is missing)

    Raises:
        SchemaError: if `strict=True` and a referenced var is missing
            from host_paths
    """
    import re
    out = []
    var_pat = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
    for m in mounts:
        def _sub(match):
            var_name = match.group(1)
            if var_name not in host_paths:
                if not strict:
                    return match.group(0)  # leave ${var} literal
                raise SchemaError(
                    f"resolve_symbolic_mounts: unknown variable "
                    f"'{var_name}' in mount '{m}'. "
                    f"Available: {sorted(host_paths.keys())}. "
                    f"Update host config (~/.sndr/host.yaml) or "
                    f"config YAML to use absolute path."
                )
            return host_paths[var_name]
        out.append(var_pat.sub(_sub, m))
    return out


@dataclass
class DockerConfig:
    """Docker container setup."""
    image: str
    container_name: str
    # Port contract — Y4 (UNIFIED_CONFIG plan 2026-05-09):
    #   - `port`             : legacy single-port shorthand. When set,
    #                          host_port and container_port both default
    #                          to it. Existing YAML configs keep working.
    #   - `host_port`        : optional explicit host-side port. Wins
    #                          over `port` when set.
    #   - `container_port`   : optional explicit container-side port
    #                          (the value passed to `vllm serve --port`).
    #                          Wins over `port` when set.
    # Use the new fields when host port differs from container port (eg.
    # multi-instance hosts running several models on the same image,
    # k8s pods with NodePort, RunPod with fixed external port).
    port: int = 8000
    host_port: Optional[int] = None
    container_port: Optional[int] = None
    shm_size: str = "8g"
    memory_limit: Optional[str] = None  # '64g'
    network: Optional[str] = None
    gpus: str = "all"
    mounts: list[str] = field(default_factory=list)
    extra_run_flags: list[str] = field(default_factory=list)

    # T1.6 (audit closure §7.4): pinned image digest. When set, the
    # launcher will refuse to launch unless the local image's
    # `RepoDigests` resolves to this exact digest, OR the operator
    # passes `--strict-image=off` to opt out. Empty/None means "no
    # digest pin" — same behavior as before T1.6.
    #
    # Format: full sha256 digest as printed by `docker inspect -f
    # '{{index .RepoDigests 0}}' <image>` (e.g.
    # "vllm/vllm-openai@sha256:abc123…"). Tag-only references are
    # rejected because tags are mutable and defeat the pin's purpose.
    image_digest: Optional[str] = None

    def effective_host_port(self) -> int:
        """Y4: returns host_port if explicitly set, else `port`."""
        return self.host_port if self.host_port is not None else self.port

    def effective_container_port(self) -> int:
        """Y4: returns container_port if explicitly set, else `port`."""
        return (self.container_port
                if self.container_port is not None else self.port)

    def effective_image_ref(self) -> str:
        """Return the immutable image reference when a digest pin exists."""
        return self.image_digest or self.image

    def validate(self) -> None:
        if not self.image:
            raise SchemaError("DockerConfig.image required")
        if not self.container_name:
            raise SchemaError("DockerConfig.container_name required")
        for name, val in (
            ("port", self.port),
            ("host_port", self.host_port),
            ("container_port", self.container_port),
        ):
            if val is None:
                continue
            if not isinstance(val, int):
                raise SchemaError(
                    f"DockerConfig.{name} must be int (got {type(val).__name__})"
                )
            if not (1 <= val <= 65535):
                raise SchemaError(
                    f"DockerConfig.{name} must be in 1..65535 (got {val})"
                )
        if self.image_digest is not None:
            if not isinstance(self.image_digest, str):
                raise SchemaError(
                    "DockerConfig.image_digest must be a string"
                )
            if "@sha256:" not in self.image_digest:
                raise SchemaError(
                    "DockerConfig.image_digest must include '@sha256:' "
                    "(use `docker inspect` RepoDigests, not a tag-only "
                    f"ref). Got: {self.image_digest!r}"
                )


@dataclass
class ReferenceMetrics:
    """Empirically-measured baseline for `verify` to compare against.

    Required fields are core: long_gen TPS, tool quality, stability CV,
    VRAM, and pins (the things `verify` always checks). Optional fields
    (short_gen_tps, concurrent_4_total_s) are richer benchmarks captured
    by `genesis_bench_suite.py` but not by the lightweight
    `verify.bench_metrics()` used by `bench-and-update`.
    """
    measured_at: str  # ISO-8601
    bench_method: str
    long_gen_sustained_tps: float
    long_gen_mean_lat_s: float
    tool_call_score: str  # '10/10'
    stability_mean_s: float
    stability_cv_pct: float
    vram_used_mib_per_gpu: list[int]
    vram_total_mib: int
    genesis_pin: str
    vllm_pin: str
    short_gen_tps: Optional[float] = None
    concurrent_4_total_s: Optional[float] = None
    # Empirical-bake (Genesis Phase D 2026-05-07): per-config measured mamba
    # REQUEST_CONSTANT state size, in MiB. When set, R-018 audit rule uses
    # this exact value instead of the 250 MiB heuristic — gives precise
    # capacity-overflow detection per model architecture. Set automatically
    # by `genesis bench-and-update --measure-mamba-state` post-warmup, or
    # manually via empirical observation of boot logs.
    # NULL/unset → R-018 falls back to 250 MiB conservative default.
    mamba_state_mib_per_request: Optional[float] = None

    # Wave 1+2 canonical genesis_bench_suite
    # output adds richer per-component metrics. All optional — old
    # configs without these fields still load. New canonical bench
    # writes them for future regression detection.
    decode_tpot_ms: Optional[float] = None       # pure decode time (no TTFT)
    ttft_ms: Optional[float] = None              # time to first token
    spec_accept_rate: Optional[float] = None     # MTP acceptance ratio (0..1)
    # Historical reference for regression triage. When the current
    # wave's bench drops below `prev_long_gen_tps` by more than the
    # tolerance, it's a real regression vs the prior known-good.
    prev_long_gen_tps: Optional[float] = None
    prev_genesis_pin: Optional[str] = None
    prev_vllm_pin: Optional[str] = None
    # Second-tier historical reference (Wave 8 closure 2026-05-11):
    # the optimization-sprint winner from the prior wave. Lets the
    # current bench compare against BOTH the previous-baseline AND
    # the previous-best-sweep. Pin-bump candidates that match sprint1
    # within tolerance are net-neutral (acceptable); regressions
    # vs sprint1 but improvements vs baseline are still net wins.
    prev_long_gen_tps_sprint1: Optional[float] = None
    # Wave 8 delta annotations (audit trail — 2026-05-11). Strings
    # (e.g. '+5.78%') because YAML carries the formatted value the
    # operator reviewed at the wave-close meeting. Loaded as-is and
    # surfaced in `verify` output without numeric recomputation.
    wave8_delta_pct_vs_wave7: Optional[str] = None
    wave8_delta_pct_vs_sprint1: Optional[str] = None
    wave8_decode_tpot_delta_pct_vs_sprint1: Optional[str] = None
    wave8_ttft_delta_pct_vs_sprint1: Optional[str] = None
    # Wave 9 delta annotations (audit trail — 2026-05-12). Same pattern
    # as wave8_* — pre-formatted strings. Populated on the 35B PROD
    # config after the dev93→dev209 pin-bump A/B re-bench surfaced a
    # -2.82% A3B-FP8 regression that was absent on 27B hybrid GDN.
    wave9_delta_pct_vs_sprint1: Optional[str] = None
    wave9_decode_tpot_delta_pct_vs_sprint1: Optional[str] = None
    wave9_ttft_delta_pct_vs_sprint1: Optional[str] = None


@dataclass
class KubernetesConfig:
    """Y5 (UNIFIED_CONFIG plan 2026-05-09): Kubernetes deployment contract.

    Drives `sndr k8s render/apply/status` (Tier 4 CLI). Declares the
    bare-minimum shape: namespace, image, GPU resource, volumes,
    service exposure, probes.

    Fields:
      - flavor: 'microk8s-single-node' | 'generic-single-node' | 'generic-multinode'
      - namespace: k8s namespace
      - image / image_pull_policy: container image
      - gpu_resource_name: 'nvidia.com/gpu' (default) or hostpath alt
      - gpu_count: requested GPU count
      - runtime_class_name: 'nvidia' typically (RuntimeClass)
      - storage: PVC declarations (model weights, hf cache, ...)
      - service_type: 'NodePort' | 'ClusterIP' | 'LoadBalancer'
      - service_node_port: when type=NodePort
      - liveness_initial_delay / readiness_initial_delay: probe timings
      - notes: free-form
    """
    flavor: str = "microk8s-single-node"
    namespace: str = "genesis"
    image: str = ""
    image_pull_policy: str = "IfNotPresent"
    gpu_resource_name: str = "nvidia.com/gpu"
    gpu_count: int = 1
    runtime_class_name: str = "nvidia"
    storage: dict[str, str] = field(default_factory=dict)
    service_type: str = "ClusterIP"
    service_node_port: Optional[int] = None
    liveness_initial_delay: int = 600
    readiness_initial_delay: int = 600
    notes: str = ""

    # S3.3 (audit P3-3, 2026-05-12): дополнительные deployment knobs.
    # nodeSelector — labels на pod'е, чтобы scheduler выбрал нужную node
    # (например `gpu-class=a5000`). Пустой dict → omit nodeSelector
    # block из рендера.
    node_selector: dict[str, str] = field(default_factory=dict)

    # PVC bindings — `pvc_name -> mount_path`. Когда заданы, renderer
    # эмитит PersistentVolumeClaim resources (operator должен иметь
    # соответствующие PVs или storageClass) И mounts через
    # `volumes[].persistentVolumeClaim.claimName`. Если использовать
    # hostPath (legacy `storage`) — оставить pvc пустым.
    pvc: dict[str, str] = field(default_factory=dict)

    # PVC sizing (GiB per claim). Default 100 GiB on каждом claim'е если
    # не указано. Структура: `{claim_name: size_gib}`.
    pvc_size_gib: dict[str, int] = field(default_factory=dict)

    # PVC storageClassName. Пусто → operator должен иметь default class.
    pvc_storage_class: str = ""

    # Secret references — `secret_name -> mount_path`. Полезно для
    # bearer tokens / wandb keys. Содержимое секрета НЕ генерируется
    # renderer'ом — operator создаёт его вручную через `kubectl create
    # secret generic …`.
    secret_mounts: dict[str, str] = field(default_factory=dict)

    _VALID_FLAVORS = (
        "microk8s-single-node", "generic-single-node", "generic-multinode",
    )
    _VALID_PULL = ("Always", "IfNotPresent", "Never")
    _VALID_SVC = ("NodePort", "ClusterIP", "LoadBalancer")

    def validate(self) -> None:
        if self.flavor not in self._VALID_FLAVORS:
            raise SchemaError(
                f"KubernetesConfig.flavor must be one of "
                f"{self._VALID_FLAVORS} (got {self.flavor!r})"
            )
        if self.image_pull_policy not in self._VALID_PULL:
            raise SchemaError(
                f"KubernetesConfig.image_pull_policy must be one of "
                f"{self._VALID_PULL} (got {self.image_pull_policy!r})"
            )
        if self.service_type not in self._VALID_SVC:
            raise SchemaError(
                f"KubernetesConfig.service_type must be one of "
                f"{self._VALID_SVC} (got {self.service_type!r})"
            )
        if self.gpu_count < 0:
            raise SchemaError(
                f"KubernetesConfig.gpu_count must be >= 0 "
                f"(got {self.gpu_count})"
            )
        if (self.service_type == "NodePort"
                and self.service_node_port is not None
                and not (30000 <= self.service_node_port <= 32767)):
            raise SchemaError(
                f"KubernetesConfig.service_node_port must be in "
                f"30000..32767 for NodePort (got {self.service_node_port})"
            )
        if self.liveness_initial_delay < 0 or self.readiness_initial_delay < 0:
            raise SchemaError(
                "KubernetesConfig.*_initial_delay must be >= 0"
            )


@dataclass
class ProxmoxConfig:
    """Y6 (UNIFIED_CONFIG plan 2026-05-09): Proxmox deployment contract.

    Drives `sndr proxmox doctor/render/apply/status` (Tier 4 CLI).

    Three modes:
      - 'lxc': run inside an unprivileged LXC container (preferred for
               GPU stack — bare-metal venv inside LXC; Docker-inside-
               LXC is experimental/risky)
      - 'vm': run inside a Proxmox VM (Docker-on-VM safe)
      - 'host': bare-metal on the PVE host itself (least-isolated,
                expert-only — voids Proxmox upgrade safety)

    Fields:
      - mode: 'lxc' | 'vm' | 'host'
      - api_endpoint: PVE API URL
      - target_node: PVE node hostname
      - container_id_or_vmid: 100-999900
      - gpu_passthrough: True if GPU is passed through to LXC/VM
      - runtime: container runtime inside the LXC/VM ('docker' | 'podman'
                  | 'venv' | 'system_python')
      - notes: free-form
    """
    mode: str = "lxc"
    api_endpoint: Optional[str] = None
    target_node: Optional[str] = None
    container_id_or_vmid: Optional[int] = None
    gpu_passthrough: bool = True
    runtime: str = "venv"
    notes: str = ""

    _VALID_MODES = ("lxc", "vm", "host")
    _VALID_RUNTIMES = ("docker", "podman", "venv", "system_python")

    def validate(self) -> None:
        if self.mode not in self._VALID_MODES:
            raise SchemaError(
                f"ProxmoxConfig.mode must be one of {self._VALID_MODES} "
                f"(got {self.mode!r})"
            )
        if self.runtime not in self._VALID_RUNTIMES:
            raise SchemaError(
                f"ProxmoxConfig.runtime must be one of "
                f"{self._VALID_RUNTIMES} (got {self.runtime!r})"
            )
        # Safety: Docker-inside-LXC is risky; flag in notes
        if self.mode == "lxc" and self.runtime == "docker":
            # Not a hard error — let operators opt in — but the
            # validator should at least flag it. We use lazy
            # self.notes append (validate() must NOT mutate) so we
            # surface via raise only when notes don't acknowledge.
            if "docker-inside-lxc" not in (self.notes or "").lower():
                # Soft signal via SchemaError — operator can add the
                # acknowledgement string to notes to opt in.
                raise SchemaError(
                    "ProxmoxConfig.mode='lxc' + runtime='docker' is "
                    "experimental/risky (PVE LXC + nested Docker has "
                    "GPU passthrough quirks). Add the literal string "
                    "'docker-inside-lxc' to notes to acknowledge + opt in."
                )
        if (self.container_id_or_vmid is not None
                and not (100 <= self.container_id_or_vmid <= 999_900)):
            raise SchemaError(
                f"ProxmoxConfig.container_id_or_vmid must be 100..999900 "
                f"(got {self.container_id_or_vmid})"
            )


@dataclass
class BootstrapConfig:
    """Y7 (UNIFIED_CONFIG plan 2026-05-09): universal-installer driver.

    Operators run `sndr bootstrap apply --scope <X>` and this block
    declares which scopes are enabled + apply_policy.

    Fields:
      - scopes: list of {'os-packages', 'gpu-runtime', 'python-runtime',
                          'container-runtime', 'model-artifacts', 'service'}
                 — the operator's "what to bootstrap" matrix
      - apply_policy: 'ask' (default) | 'auto-yes' | 'never'
      - privilege: 'sudo' | 'root' | 'user' — what privilege the
                     bootstrapper is allowed to acquire
      - rollback_on_failure: True
      - notes: free-form
    """
    scopes: list[str] = field(default_factory=list)
    apply_policy: str = "ask"
    privilege: str = "sudo"
    rollback_on_failure: bool = True
    notes: str = ""

    _VALID_SCOPES = (
        "os-packages", "gpu-runtime", "python-runtime",
        "container-runtime", "model-artifacts", "service", "all",
    )
    _VALID_POLICY = ("ask", "auto-yes", "never")
    _VALID_PRIV = ("sudo", "root", "user")

    def validate(self) -> None:
        if not isinstance(self.scopes, list):
            raise SchemaError("BootstrapConfig.scopes must be list[str]")
        for s in self.scopes:
            if s not in self._VALID_SCOPES:
                raise SchemaError(
                    f"BootstrapConfig.scopes contains invalid scope "
                    f"{s!r}; valid: {self._VALID_SCOPES}"
                )
        if self.apply_policy not in self._VALID_POLICY:
            raise SchemaError(
                f"BootstrapConfig.apply_policy must be one of "
                f"{self._VALID_POLICY} (got {self.apply_policy!r})"
            )
        if self.privilege not in self._VALID_PRIV:
            raise SchemaError(
                f"BootstrapConfig.privilege must be one of "
                f"{self._VALID_PRIV} (got {self.privilege!r})"
            )


@dataclass
class GpuTuningConfig:
    """Y8 (UNIFIED_CONFIG plan 2026-05-09): GPU tuning policy.

    Declares which GPU-side knobs the launcher should apply at boot.
    Safety: ONLY `persistence_mode`, `ulimits`, `shm_size`, and
    `vllm_args` are safe-by-default. `power_limit` and `clocks`
    require explicit `unsafe_apply: true` because they can throttle
    cards or void warranties on consumer hardware.

    Fields:
      - persistence_mode: bool — `nvidia-smi -pm 1` (recommended on)
      - power_limit_watts: int — `nvidia-smi -pl` (UNSAFE — needs opt-in)
      - clocks_mhz: dict gfx_max/mem_max — `nvidia-smi -lgc/-lmc` (UNSAFE)
      - ulimits: dict — locked memory + open files
      - transparent_hugepages: 'never' | 'madvise' | 'always'
      - unsafe_apply: bool — gate for power_limit / clocks
    """
    persistence_mode: Optional[bool] = None
    power_limit_watts: Optional[int] = None
    clocks_gfx_mhz: Optional[int] = None
    clocks_mem_mhz: Optional[int] = None
    ulimits: dict[str, str] = field(default_factory=dict)
    transparent_hugepages: Optional[str] = None
    unsafe_apply: bool = False
    notes: str = ""

    _VALID_THP = (None, "never", "madvise", "always")

    def validate(self) -> None:
        if (self.power_limit_watts is not None
                or self.clocks_gfx_mhz is not None
                or self.clocks_mem_mhz is not None) and not self.unsafe_apply:
            raise SchemaError(
                "GpuTuningConfig: power_limit_watts/clocks_gfx_mhz/"
                "clocks_mem_mhz require explicit unsafe_apply=true. "
                "These can throttle consumer cards or void warranties."
            )
        if self.power_limit_watts is not None and self.power_limit_watts < 50:
            raise SchemaError(
                f"GpuTuningConfig.power_limit_watts must be >= 50W "
                f"(got {self.power_limit_watts})"
            )
        if (self.clocks_gfx_mhz is not None
                and not (100 <= self.clocks_gfx_mhz <= 4000)):
            raise SchemaError(
                f"GpuTuningConfig.clocks_gfx_mhz must be in 100..4000 "
                f"(got {self.clocks_gfx_mhz})"
            )
        if self.transparent_hugepages not in self._VALID_THP:
            raise SchemaError(
                f"GpuTuningConfig.transparent_hugepages must be one of "
                f"{self._VALID_THP[1:]} or unset"
            )


@dataclass
class ObservabilityConfig:
    """Y14 (UNIFIED_CONFIG plan 2026-05-09): observability declarations.

    Drives memory_trace + cudagraph dispatch tracking + per-patch
    apply telemetry — all already exist in `vllm/sndr_core/observability/`.
    This block makes them declarative per-config instead of pure env-var.

    Fields:
      - memory_trace.enabled: bool
      - memory_trace.csv_path: str
      - cudagraph_dispatch_trace: bool
      - per_patch_telemetry: bool
    """
    memory_trace_enabled: bool = False
    memory_trace_csv_path: Optional[str] = None
    cudagraph_dispatch_trace: bool = False
    per_patch_telemetry: bool = True
    notes: str = ""

    def validate(self) -> None:
        if (self.memory_trace_enabled and not self.memory_trace_csv_path):
            raise SchemaError(
                "ObservabilityConfig.memory_trace_enabled=True requires "
                "memory_trace_csv_path"
            )
        if self.memory_trace_csv_path is not None:
            if not isinstance(self.memory_trace_csv_path, str):
                raise SchemaError("memory_trace_csv_path must be string")


@dataclass
class PackageSource:
    """Y2 (UNIFIED_CONFIG plan 2026-05-09): one channel/source declaration.

    Operators declare WHERE each runtime dependency comes from
    (distro repo / pip channel / source build / docker image / NVIDIA
    repo). Drives `sndr deps install` policy: refuse to `curl|bash`
    unless explicitly opted in, prefer official distro repos by default.

    Fields:
      - name: 'docker' | 'nvidia_container_toolkit' | 'python' | 'vllm' | ...
      - kind: 'distro_repo' | 'pip' | 'docker_image' | 'nvidia_repo' |
              'github_release' | 'source_build' | 'curl_pipe_bash'
      - channel: free-form (e.g. 'stable', 'nightly', 'main')
      - allow_third_party: True if non-official upstream sources are OK
        (e.g. unofficial Docker repo, pre-release pip indices)
      - notes: free-form
    """
    name: str
    kind: str
    channel: str = "stable"
    allow_third_party: bool = False
    notes: str = ""

    _VALID_KINDS = (
        "distro_repo", "pip", "docker_image", "nvidia_repo",
        "github_release", "source_build", "curl_pipe_bash",
    )

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise SchemaError("PackageSource.name must be non-empty string")
        if self.kind not in self._VALID_KINDS:
            raise SchemaError(
                f"PackageSource.kind must be one of {self._VALID_KINDS} "
                f"(got {self.kind!r})"
            )
        # SAFETY: curl|bash is opt-in via allow_third_party=True and
        # never default. Document the risk for the operator.
        if self.kind == "curl_pipe_bash" and not self.allow_third_party:
            raise SchemaError(
                f"PackageSource.kind='curl_pipe_bash' for {self.name!r} "
                f"requires allow_third_party=True (explicit opt-in to "
                f"running an arbitrary upstream script as root)"
            )


@dataclass
class PackageSources:
    """Y2 container — list of declared sources, indexed by name."""
    sources: list[PackageSource] = field(default_factory=list)

    def validate(self) -> None:
        if not isinstance(self.sources, list):
            raise SchemaError("PackageSources.sources must be list")
        names = []
        for s in self.sources:
            s.validate()
            if s.name in names:
                raise SchemaError(
                    f"PackageSources.sources duplicate name {s.name!r}"
                )
            names.append(s.name)

    def get(self, name: str) -> Optional[PackageSource]:
        for s in self.sources:
            if s.name == name:
                return s
        return None


@dataclass
class ServiceConfig:
    """Y10 (UNIFIED_CONFIG plan 2026-05-09): service-management contract.

    Declares which service backend the launcher should target when
    operator runs `sndr service install`. Multiple backends are
    supported per-config; the launcher picks based on `--runtime` (W-runtime).

    Fields:
      - backend: 'systemd' | 'docker_compose' | 'podman_quadlet' |
                 'kubernetes' | 'proxmox' | 'bare_metal'
      - service_name: short-name for the unit (e.g. 'genesis-35b-prod')
      - user: OS user to run as (systemd User=, docker --user)
      - working_dir: cwd for the service
      - env_file: path to file with extra env vars
      - logs_dir: where stdout/stderr land
      - restart: 'always' | 'on-failure' | 'no'
      - notes: free-form
    """
    backend: str = "docker_compose"
    service_name: str = ""
    user: Optional[str] = None
    working_dir: Optional[str] = None
    env_file: Optional[str] = None
    logs_dir: Optional[str] = None
    restart: str = "on-failure"
    notes: str = ""

    _VALID_BACKENDS = (
        "systemd", "docker_compose", "podman_quadlet",
        "kubernetes", "proxmox", "bare_metal",
    )
    _VALID_RESTART = ("always", "on-failure", "no", "unless-stopped")

    def validate(self) -> None:
        if self.backend not in self._VALID_BACKENDS:
            raise SchemaError(
                f"ServiceConfig.backend must be one of "
                f"{self._VALID_BACKENDS} (got {self.backend!r})"
            )
        if self.restart not in self._VALID_RESTART:
            raise SchemaError(
                f"ServiceConfig.restart must be one of "
                f"{self._VALID_RESTART} (got {self.restart!r})"
            )
        if self.service_name and not isinstance(self.service_name, str):
            raise SchemaError("ServiceConfig.service_name must be string")


@dataclass
class ArtifactModel:
    """Y3 (UNIFIED_CONFIG plan 2026-05-09): one model artifact spec.

    Declares a HuggingFace-resolvable model + its local path + verify
    rules. Replaces the old `fetch_models.sh` hardcoded paths and the
    legacy `compat.models.pull` registry-tagged lookup with a typed,
    config-owned spec.

    Fields:
      - hf_id: HuggingFace repo ID (e.g. 'Qwen/Qwen3.6-27B-int4-AutoRound').
      - local_dir: absolute or `${var}` path where weights land.
      - revision: HF revision (commit SHA or tag). Defaults to 'main'.
      - gated: True if the repo requires HF token. Drives token-prompt UX.
      - required_files: glob patterns that MUST exist after pull
        (e.g. ['config.json', '*.safetensors']).
      - min_total_gib: minimum total local size to consider 'pulled OK'.
      - notes: free-form operator notes.
    """
    hf_id: str
    local_dir: str
    revision: str = "main"
    gated: bool = False
    required_files: list[str] = field(default_factory=lambda: ["config.json"])
    min_total_gib: float = 0.0
    notes: str = ""

    def validate(self) -> None:
        if not isinstance(self.hf_id, str) or "/" not in self.hf_id:
            raise SchemaError(
                f"ArtifactModel.hf_id must be 'org/repo' (got {self.hf_id!r})"
            )
        if not isinstance(self.local_dir, str) or not self.local_dir.strip():
            raise SchemaError(
                "ArtifactModel.local_dir must be non-empty string"
            )
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise SchemaError(
                "ArtifactModel.revision must be non-empty string"
            )
        if not isinstance(self.required_files, list):
            raise SchemaError(
                "ArtifactModel.required_files must be list[str]"
            )
        if self.min_total_gib < 0:
            raise SchemaError(
                f"ArtifactModel.min_total_gib must be >= 0 "
                f"(got {self.min_total_gib})"
            )

    def verify(self, *, base_path: Optional[str] = None) -> list[str]:
        """Returns a list of human-readable verification problems.

        Empty list = artifact is present + complete on disk.
        `base_path` overrides `${var}` lookup for tests; production
        callers resolve via host.yaml first.
        """
        from pathlib import Path
        problems: list[str] = []
        local = Path(base_path or self.local_dir).expanduser()
        if not local.is_dir():
            return [f"local_dir does not exist: {local}"]
        # Required files (glob match)
        for pattern in self.required_files:
            matches = list(local.rglob(pattern)) if "*" in pattern else (
                [local / pattern] if (local / pattern).exists() else []
            )
            if not matches:
                problems.append(
                    f"required file {pattern!r} not found under {local}"
                )
        # Min total size
        if self.min_total_gib > 0:
            total = sum(
                f.stat().st_size for f in local.rglob("*")
                if f.is_file()
            )
            total_gib = total / (1 << 30)
            if total_gib < self.min_total_gib:
                problems.append(
                    f"local size {total_gib:.2f} GiB < min_total_gib "
                    f"{self.min_total_gib:.2f} GiB"
                )
        return problems


@dataclass
class ArtifactCache:
    """Y3 (UNIFIED_CONFIG plan 2026-05-09): one cache artifact spec.

    Used for `huggingface_hub`, `triton`, `torch_compile`, `safetensors`
    caches. Drives `sndr deps plan` to know which on-disk caches the
    config expects + lets the launcher mount them when running in
    container mode.

    Fields:
      - kind: 'huggingface_hub' | 'triton' | 'torch_compile' | 'safetensors'
              | 'compile_cache' | 'other'
      - path: absolute or ${var} path to the cache directory.
      - persistent: True if the cache should survive container restarts
        (mount as named volume / host path), False for ephemeral.
      - notes: free-form.
    """
    kind: str
    path: str
    persistent: bool = True
    notes: str = ""

    _VALID_KINDS = (
        "huggingface_hub", "triton", "torch_compile", "compile_cache",
        "safetensors", "other",
    )

    def validate(self) -> None:
        if self.kind not in self._VALID_KINDS:
            raise SchemaError(
                f"ArtifactCache.kind must be one of {self._VALID_KINDS} "
                f"(got {self.kind!r})"
            )
        if not isinstance(self.path, str) or not self.path.strip():
            raise SchemaError("ArtifactCache.path must be non-empty string")


@dataclass
class Artifacts:
    """Y3 (UNIFIED_CONFIG plan 2026-05-09): container for model + cache specs.

    Top-level holder so YAML can express both lists in one block:

        artifacts:
          models:
            - hf_id: Qwen/Qwen3.6-27B-int4-AutoRound
              local_dir: /models/Qwen3.6-27B-int4-AutoRound
              required_files: [config.json, "*.safetensors"]
              min_total_gib: 14.0
          caches:
            - kind: huggingface_hub
              path: ~/.cache/huggingface
              persistent: true
            - kind: triton
              path: ${cache_root}/triton-cache-v11
    """
    models: list[ArtifactModel] = field(default_factory=list)
    caches: list[ArtifactCache] = field(default_factory=list)

    def validate(self) -> None:
        if not isinstance(self.models, list):
            raise SchemaError("Artifacts.models must be list[ArtifactModel]")
        if not isinstance(self.caches, list):
            raise SchemaError("Artifacts.caches must be list[ArtifactCache]")
        for m in self.models:
            m.validate()
        for c in self.caches:
            c.validate()


@dataclass
class OffloadConfig:
    """club-3090 #58 Path A (UNIFIED_CONFIG plan 2026-05-09): VRAM→CPU/disk
    spillover knobs (interim). Surfaces vLLM's existing
    `--cpu-offload-gb` flag in a typed schema slot AND reserves
    namespace for future tier-aware Genesis-original CacheConfig
    extension (Path C, planned for v7.73.x).

    Today this block translates to one engine arg:
      `--cpu-offload-gb <cpu_offload_gib>`

    Future fields (planned, not yet wired):
      - `tiers: list[CacheTier]`      — hierarchical cache (gpu/cpu/nvme)
      - `vision_token_demote_first`   — image tokens evicted first
      - `exclude_mamba_ssm`           — keep Mamba SSM state on GPU
        (mandatory for hybrid-GDN; vLLM/SGLang/LMCache all crash on
        hybrid-GDN offload — see club-3090 #58 research report)

    Hybrid-GDN guard: when set on a config whose `kv_cache_dtype`
    indicates hybrid GDN (turboquant_k8v4 + GDN model), `validate()`
    raises with a precise pointer to the research report. Operators
    on dense models can use this freely.
    """
    cpu_offload_gib: float = 0.0
    swap_space_gib: float = 0.0
    notes: str = ""

    def validate(self) -> None:
        if not isinstance(self.cpu_offload_gib, (int, float)):
            raise SchemaError(
                "OffloadConfig.cpu_offload_gib must be number (got "
                f"{type(self.cpu_offload_gib).__name__})"
            )
        if self.cpu_offload_gib < 0:
            raise SchemaError(
                "OffloadConfig.cpu_offload_gib must be >= 0 (got "
                f"{self.cpu_offload_gib})"
            )
        if not isinstance(self.swap_space_gib, (int, float)):
            raise SchemaError(
                "OffloadConfig.swap_space_gib must be number"
            )
        if self.swap_space_gib < 0:
            raise SchemaError(
                "OffloadConfig.swap_space_gib must be >= 0"
            )

    def to_vllm_args(self) -> list[str]:
        """Render as vllm engine flags. Empty list when offload is disabled."""
        args: list[str] = []
        if self.cpu_offload_gib > 0:
            args.append(f"--cpu-offload-gb {self.cpu_offload_gib:g}")
        if self.swap_space_gib > 0:
            args.append(f"--swap-space {self.swap_space_gib:g}")
        return args


@dataclass
class UpstreamPinPolicy:
    """Y11 (UNIFIED_CONFIG plan 2026-05-09): per-config vLLM pin policy.

    Operators can declare which vLLM pins this config has been
    validated against (`allowed_pins`) and which pins are known to
    break it (`blocked_pins`). The launcher consults this BEFORE
    starting vllm; a blocked pin aborts with a precise error
    pointing at the relevant `notes` entry.

    `required_pin` is the strict equivalent of `vllm_pin_required` at
    the top level — when set, only that exact pin is allowed (subset
    of `allowed_pins`). Use this for stable / community-prod configs.

    Empty allowlist + empty blocklist = legacy "warn-only" behavior;
    KNOWN_GOOD_VLLM_PINS still enforces project-wide allowlist.
    """
    required_pin: Optional[str] = None
    allowed_pins: list[str] = field(default_factory=list)
    blocked_pins: list[str] = field(default_factory=list)
    notes: str = ""

    def validate(self) -> None:
        for name, lst in (("allowed_pins", self.allowed_pins),
                          ("blocked_pins", self.blocked_pins)):
            if not isinstance(lst, list):
                raise SchemaError(
                    f"UpstreamPinPolicy.{name} must be list[str]"
                )
            for p in lst:
                if not isinstance(p, str) or not p.strip():
                    raise SchemaError(
                        f"UpstreamPinPolicy.{name} entries must be "
                        f"non-empty strings (got {p!r})"
                    )
        # Cross-check: required_pin can't be in blocked_pins.
        if self.required_pin and self.required_pin in self.blocked_pins:
            raise SchemaError(
                f"UpstreamPinPolicy.required_pin {self.required_pin!r} "
                f"is also listed in blocked_pins"
            )
        # Overlap: allowed ∩ blocked must be empty.
        overlap = set(self.allowed_pins) & set(self.blocked_pins)
        if overlap:
            raise SchemaError(
                f"UpstreamPinPolicy: pins in both allowed_pins and "
                f"blocked_pins: {sorted(overlap)}"
            )

    def check(self, running_pin: Optional[str]) -> Optional[str]:
        """Returns a violation message string if `running_pin` is rejected.

        Returns None if the pin is allowed (or no policy is declared).
        Order of decision:
          1. blocked_pins → reject
          2. required_pin set → must equal it
          3. allowed_pins set → must be in the list
          4. otherwise → allow (defer to KNOWN_GOOD_VLLM_PINS)
        """
        if not running_pin:
            return None
        if running_pin in self.blocked_pins:
            note = f" — {self.notes}" if self.notes else ""
            return (
                f"vllm pin {running_pin!r} is in this config's "
                f"upstream.blocked_pins{note}"
            )
        if self.required_pin and running_pin != self.required_pin:
            return (
                f"vllm pin {running_pin!r} != upstream.required_pin "
                f"{self.required_pin!r}"
            )
        if self.allowed_pins and running_pin not in self.allowed_pins:
            return (
                f"vllm pin {running_pin!r} not in upstream.allowed_pins "
                f"{sorted(self.allowed_pins)}"
            )
        return None


@dataclass
class OverridesPolicy:
    """Y12 (UNIFIED_CONFIG plan 2026-05-09): runtime override safety.

    Operators can declare which env vars are safe to override at
    `sndr launch --override KEY=VAL` time, and what numeric ranges
    are acceptable. Safety: prevents an operator from setting
    `GENESIS_P67_NUM_KV_SPLITS=999` and silently destroying TPS, or
    from setting `GENESIS_PN16_TOOL_THINK_BUDGET=-1` and crashing
    the request middleware.

    `allow_env` is a list of env-var keys (regex-free literal match)
    that may be overridden. Vars not in the list are rejected.

    `safe_ranges` maps env-var key → (min_str, max_str). The launcher
    parses the override value as int OR float and rejects out-of-range.
    Strings in min/max so YAML parses naturally; coerced lazily.
    """
    allow_env: list[str] = field(default_factory=list)
    safe_ranges: dict[str, list[str]] = field(default_factory=dict)
    notes: str = ""

    def validate(self) -> None:
        if not isinstance(self.allow_env, list):
            raise SchemaError("OverridesPolicy.allow_env must be list[str]")
        for k in self.allow_env:
            if not isinstance(k, str) or not k.strip():
                raise SchemaError(
                    f"OverridesPolicy.allow_env entries must be non-empty "
                    f"strings (got {k!r})"
                )
        if not isinstance(self.safe_ranges, dict):
            raise SchemaError("OverridesPolicy.safe_ranges must be dict")
        for k, rng in self.safe_ranges.items():
            if not isinstance(rng, list) or len(rng) != 2:
                raise SchemaError(
                    f"OverridesPolicy.safe_ranges[{k!r}] must be a "
                    f"[min, max] 2-list (got {rng!r})"
                )
            for v in rng:
                try:
                    float(v)
                except (TypeError, ValueError) as e:
                    raise SchemaError(
                        f"OverridesPolicy.safe_ranges[{k!r}] bound "
                        f"{v!r} is not numeric"
                    ) from e

    def check(self, key: str, value: str) -> Optional[str]:
        """Returns violation msg if (key,value) override is rejected.

        Returns None if accepted. Order:
          1. allow_env is empty → reject (no overrides allowed)
          2. key not in allow_env → reject
          3. key in safe_ranges → value must parse as number AND lie in [min,max]
          4. key in allow_env but no range → accept (string-only override)
        """
        if not self.allow_env:
            return "overrides not enabled (allow_env is empty)"
        if key not in self.allow_env:
            return (
                f"override key {key!r} not in allow_env "
                f"(allowed: {sorted(self.allow_env)})"
            )
        if key in self.safe_ranges:
            try:
                v = float(value)
            except (TypeError, ValueError):
                return (
                    f"override {key}={value!r} not numeric — range "
                    f"{self.safe_ranges[key]} requires a number"
                )
            lo, hi = float(self.safe_ranges[key][0]), float(self.safe_ranges[key][1])
            if not (lo <= v <= hi):
                return (
                    f"override {key}={value!r} outside safe range "
                    f"[{lo}, {hi}]"
                )
        return None


@dataclass
class PackageVersions:
    """Y1 (UNIFIED_CONFIG plan 2026-05-09): in-container package pins.

    Operators declare the runtime python packages the container needs
    (alongside vLLM itself). The renderer honors `python_packages`
    when SNDR_DEV_INSTALL_RUNTIME_DEPS=1 is set inside the container,
    rather than the renderer hardcoding versions in a string literal
    (B6 fix: previous renderer baked `pandas==2.2.3 scipy==1.14.1
    xxhash==3.5.0` into every config).

    All fields optional. If `python_packages` is empty/None, the
    renderer falls back to the legacy hardcoded baseline so existing
    YAML configs that don't declare this block keep working.

    Future blocks (planned per UNIFIED_CONFIG_AUTOMATION_PLAN §Y1):
      - vllm:           {channel: stable|tested|nightly|local, version}
      - torch:          {version}
      - flashinfer:     {version, source}
      - triton:         {version}
      - transformers:   {version}
    """
    python_packages: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def validate(self) -> None:
        for name, pin in self.python_packages.items():
            if not isinstance(name, str) or not name.strip():
                raise SchemaError(
                    f"PackageVersions.python_packages keys must be non-empty "
                    f"strings (got {name!r})"
                )
            if not isinstance(pin, str) or not pin.strip():
                raise SchemaError(
                    f"PackageVersions.python_packages[{name!r}] must be a "
                    f"non-empty version string (got {pin!r})"
                )
            # Operators must pin exactly — supply chain integrity.
            # Allow `==`, `===`, or bare version (we add `==` if bare).
            # Reject bare ranges like `>=2.0`.
            stripped = pin.strip()
            if any(stripped.startswith(op) for op in (">=", "<=", ">", "<", "~=")):
                raise SchemaError(
                    f"PackageVersions.python_packages[{name!r}]={pin!r} "
                    f"must be an exact pin (use 'X.Y.Z' or '==X.Y.Z'); "
                    f"version ranges are not allowed in production configs."
                )

    def to_pip_args(self) -> str:
        """Render as space-joined `name==version` arguments for `pip install`.

        Empty dict → empty string (renderer treats as "fallback to legacy").
        Bare version values get `==` prefix; explicit `==X.Y` passed through.
        """
        if not self.python_packages:
            return ""
        parts: list[str] = []
        for name, pin in self.python_packages.items():
            stripped = pin.strip()
            if stripped.startswith("=="):
                parts.append(f"{name}{stripped}")
            else:
                parts.append(f"{name}=={stripped}")
        return " ".join(parts)


@dataclass
class CacheTier:
    """Path C v7.73.x (PN95): one level of the KV cache hierarchy.

    Lower-index tiers are closer to compute (typically tier 0 = GPU,
    tier 1 = CPU pinned RAM, tier 2 = NVMe). Each tier carries its own
    capacity and eviction policy; demote crosses tiers, evict drops
    from the bottom tier.

    Operators declare tiers in `cache_config.tiers`. Empty list →
    PN91 single-tier behavior (zero impact for existing PROD configs).

    Field semantics:
      - `device`: 'gpu' | 'cpu' | 'nvme'
      - `capacity_gib`: hard cap on this tier's allocation
      - `eviction_policy`: forwarded to make_policy() per-tier
      - `promote_on_hit`: demoted page hit → bring back to upper tier
      - `demote_threshold_pct`: tier fill ratio that triggers demote
        (default 0.92 — start demoting when 92% full)
      - `low_water_pct`: demote until this ratio reached (default 0.75)
      - `vision_first`: if True, evict mm/image pages first
      - `pinned`: cpu tier uses cudaMallocHost-backed memory (default True)
      - `nvme_path`: required when device == 'nvme'
    """
    device: str
    capacity_gib: float
    eviction_policy: str = "lru"
    promote_on_hit: bool = True
    demote_threshold_pct: float = 0.92
    low_water_pct: float = 0.75
    vision_first: bool = False
    pinned: bool = True
    nvme_path: Optional[str] = None
    notes: str = ""

    def validate(self) -> None:
        from vllm.sndr_core.cache.eviction_policies import list_policies
        valid_devices = {"gpu", "cpu", "nvme"}
        if self.device not in valid_devices:
            raise SchemaError(
                f"CacheTier.device must be one of {sorted(valid_devices)} "
                f"(got {self.device!r})"
            )
        if not isinstance(self.capacity_gib, (int, float)):
            raise SchemaError("CacheTier.capacity_gib must be number")
        if self.capacity_gib <= 0:
            raise SchemaError(
                f"CacheTier.capacity_gib must be > 0 "
                f"(got {self.capacity_gib})"
            )
        valid_pol = list_policies()
        if self.eviction_policy not in valid_pol:
            raise SchemaError(
                f"CacheTier.eviction_policy must be one of {valid_pol} "
                f"(got {self.eviction_policy!r})"
            )
        if not (0.0 < self.demote_threshold_pct <= 1.0):
            raise SchemaError(
                f"CacheTier.demote_threshold_pct must be in (0,1] "
                f"(got {self.demote_threshold_pct})"
            )
        if not (0.0 <= self.low_water_pct < 1.0):
            raise SchemaError(
                f"CacheTier.low_water_pct must be in [0,1) "
                f"(got {self.low_water_pct})"
            )
        if self.low_water_pct >= self.demote_threshold_pct:
            raise SchemaError(
                f"CacheTier.low_water_pct ({self.low_water_pct}) must be "
                f"strictly less than demote_threshold_pct "
                f"({self.demote_threshold_pct})"
            )
        if self.device == "nvme" and not self.nvme_path:
            raise SchemaError(
                "CacheTier.nvme_path is required when device == 'nvme'"
            )


@dataclass
class CacheConfig:
    """T2.1 (vllm#40270 backport / PN91) + Path C v7.73.x (PN95):
    pluggable KV cache eviction with optional multi-tier hierarchy.

    PN91 fields (single-tier, back-compat):
      - `eviction_policy`: 'lru' | '2q' | 'arc'
      - `arc_capacity`: ARC capacity (entries). Ignored for LRU/2Q.
      - `q2_a1_ratio`: 2Q probationary ratio. Ignored for LRU/ARC.

    PN95 / Path C fields (multi-tier extension):
      - `tiers`: ordered list of CacheTier (empty = PN91 single-tier).
      - `exclude_mamba_ssm`: keep Mamba SSM state on GPU. MUST stay
        True on hybrid-GDN models (relaxes the Path A guard via this
        flag instead of blocking the config outright).
      - `vision_demote_first`: image/MM pages demoted before text.
      - `tier_low_water_pct`: GPU free-VRAM threshold to trigger demote
        (e.g. 0.05 = start demoting when free VRAM < 5%).
      - `async_demote`: cudaMemcpyAsync vs sync (default True).

    Back-compat: `tiers == []` → no PN95 behavior at all; existing
    PROD configs are unaffected.
    """
    # ── PN91 single-tier (back-compat) ──
    eviction_policy: str = "lru"
    arc_capacity: int = 4096
    q2_a1_ratio: float = 0.25
    notes: str = ""
    # ── PN95 / Path C multi-tier extensions ──
    tiers: list[CacheTier] = field(default_factory=list)
    exclude_mamba_ssm: bool = True
    vision_demote_first: bool = True
    tier_low_water_pct: float = 0.05
    async_demote: bool = True

    def validate(self) -> None:
        from vllm.sndr_core.cache.eviction_policies import list_policies
        valid = list_policies()
        if self.eviction_policy not in valid:
            raise SchemaError(
                f"CacheConfig.eviction_policy must be one of {valid} "
                f"(got {self.eviction_policy!r})"
            )
        if self.arc_capacity <= 0:
            raise SchemaError(
                "CacheConfig.arc_capacity must be > 0 "
                f"(got {self.arc_capacity})"
            )
        if not (0.0 < self.q2_a1_ratio < 1.0):
            raise SchemaError(
                f"CacheConfig.q2_a1_ratio must be in (0,1) "
                f"(got {self.q2_a1_ratio})"
            )
        if not (0.0 <= self.tier_low_water_pct < 1.0):
            raise SchemaError(
                f"CacheConfig.tier_low_water_pct must be in [0,1) "
                f"(got {self.tier_low_water_pct})"
            )
        # Multi-tier shape constraints
        for t in self.tiers:
            t.validate()
        if self.tiers:
            gpu_tiers = [t for t in self.tiers if t.device == "gpu"]
            if len(gpu_tiers) > 1:
                raise SchemaError(
                    f"CacheConfig.tiers may declare at most one gpu tier "
                    f"(got {len(gpu_tiers)})"
                )
            if len(self.tiers) >= 2:
                cpu_tiers = [t for t in self.tiers if t.device == "cpu"]
                if len(cpu_tiers) != 1:
                    raise SchemaError(
                        f"CacheConfig.tiers requires exactly one cpu tier "
                        f"when len(tiers) >= 2 (got {len(cpu_tiers)})"
                    )


@dataclass
class ConfigConstraints:
    """T1.8 (audit closure §7.2): hardware + flag constraints.

    Operators can declare invariants the launcher must check BEFORE
    starting vllm: minimum GPU VRAM/count, PCIe topology requirements,
    forbidden flags. The launch-time check fails loudly with a precise
    error pointing at the violating field, instead of failing
    mysteriously deep in vllm boot.

    All fields are optional; absence means "no constraint declared".
    """
    min_gpu_memory_gib: Optional[int] = None
    min_gpu_count: Optional[int] = None
    pcie_ok: bool = True
    nvlink_recommended: bool = False
    forbidden_flags: list[str] = field(default_factory=list)
    required_kernel_modules: list[str] = field(default_factory=list)
    notes: str = ""

    def validate(self) -> None:
        if (self.min_gpu_memory_gib is not None
                and self.min_gpu_memory_gib <= 0):
            raise SchemaError(
                "ConfigConstraints.min_gpu_memory_gib must be > 0 "
                f"(got {self.min_gpu_memory_gib})"
            )
        if (self.min_gpu_count is not None
                and self.min_gpu_count <= 0):
            raise SchemaError(
                "ConfigConstraints.min_gpu_count must be > 0 "
                f"(got {self.min_gpu_count})"
            )
        for flag in self.forbidden_flags:
            if not isinstance(flag, str):
                raise SchemaError(
                    "ConfigConstraints.forbidden_flags must be list[str]"
                )

    def check(self, *, hw, vllm_extra_args: list[str]) -> list[str]:
        """Evaluate constraints against (hw, vllm_extra_args).

        Returns a list of human-readable violation messages. Empty list
        means "all constraints satisfied". The launcher consults this
        and aborts if any violation surfaces.
        """
        violations: list[str] = []
        if self.min_gpu_count is not None and hw is not None:
            n = int(getattr(hw, "n_gpus", 0) or 0)
            if n < self.min_gpu_count:
                violations.append(
                    f"min_gpu_count={self.min_gpu_count} but hardware.n_gpus={n}"
                )
        if self.min_gpu_memory_gib is not None and hw is not None:
            mib = int(getattr(hw, "min_vram_per_gpu_mib", 0) or 0)
            gib = mib / 1024
            if gib < self.min_gpu_memory_gib:
                violations.append(
                    f"min_gpu_memory_gib={self.min_gpu_memory_gib} but "
                    f"hardware.min_vram_per_gpu_mib={mib} ({gib:.1f} GiB)"
                )
        flat_args = " ".join(vllm_extra_args)
        for forbidden in self.forbidden_flags:
            if forbidden in flat_args or forbidden in vllm_extra_args:
                violations.append(
                    f"forbidden flag {forbidden!r} present in vllm_extra_args"
                )
        return violations


@dataclass
class RiskScore:
    """T1.8 (audit closure §7.2): per-config risk dimensions.

    Operators (or `sndr model-config score`) populate these to give a
    reviewer a glanceable verdict before running. Each field is a
    0-100 score where 0 = no risk and 100 = will-definitely-blow-up.
    `derive_overall()` computes a weighted sum so dashboards can rank
    configs.

    Dimensions:
      - memory_safety:    KV/scratch/CUDA-graph headroom on declared HW.
      - tool_call:        Empirical tool-parser stability (10/10 = 0).
      - spec_decode:      MTP/ngram acceptance variance + GDN risk.
      - upstream_drift:   How many declared patches have upstream PRs.
      - deployment_ready: Cross-rig + soak-time signal.
    """
    memory_safety: int = 0
    tool_call: int = 0
    spec_decode: int = 0
    upstream_drift: int = 0
    deployment_ready: int = 0
    notes: str = ""

    def validate(self) -> None:
        for name in ("memory_safety", "tool_call", "spec_decode",
                     "upstream_drift", "deployment_ready"):
            v = getattr(self, name)
            if not isinstance(v, int):
                raise SchemaError(
                    f"RiskScore.{name} must be int 0-100 (got {type(v).__name__})"
                )
            if not (0 <= v <= 100):
                raise SchemaError(
                    f"RiskScore.{name} must be in [0,100] (got {v})"
                )

    def derive_overall(self) -> int:
        """Weighted aggregate of the five dimensions, 0-100.

        Weights reflect production impact: memory_safety + deployment_ready
        get the largest weight because they predict launch-success;
        tool_call is medium because operators can fix at request time;
        spec_decode + upstream_drift are auxiliary.
        """
        weights = {
            "memory_safety": 30,
            "deployment_ready": 25,
            "tool_call": 20,
            "spec_decode": 15,
            "upstream_drift": 10,
        }
        total = sum(
            getattr(self, k) * w
            for k, w in weights.items()
        )
        return total // sum(weights.values())


@dataclass
class CompatibilityRule:
    """S2.5 декларативное правило совместимости.

    Зачем
    -----
    Раньше известные несовместимости были разбросаны по `validate()` и
    `audit()` методам ModelConfig'а. Это работает, но новые operator'ы не
    могут одним взглядом увидеть, "какие комбинации опций безопасны".
    `CompatibilityMatrix` собирает все правила в одном месте, а UI/CLI
    может рендерить их в виде таблицы.

    Семантика
    ---------
    Каждое правило содержит:

      • `id` — стабильный идентификатор (`COMPAT-XXX`).
      • `severity` — `"forbidden"` (hard error в validate()) или
        `"discouraged"` (soft warning в audit()).
      • `predicate(cfg) → bool` — True если конфиг попадает под правило.
      • `message` — человекочитаемое объяснение, что не так и почему.
      • `mitigation` — что сделать, чтобы стало корректно.
      • `references` — docs / issue links для дополнительного контекста.

    Не дублирует существующие inline checks — добавляет НОВЫЕ декларации
    и предоставляет агрегатный view для CLI.
    """
    id: str
    severity: str  # "forbidden" | "discouraged"
    title: str
    message: str
    mitigation: str
    references: list[str] = field(default_factory=list)
    # predicate хранится не в dataclass поле (нельзя сериализовать в YAML);
    # его регистрирует `CompatibilityMatrix` рядом с метадатой.

    def validate(self) -> None:
        if not self.id:
            raise SchemaError("CompatibilityRule.id required")
        if self.severity not in ("forbidden", "discouraged"):
            raise SchemaError(
                "CompatibilityRule.severity must be 'forbidden' or "
                f"'discouraged' (got '{self.severity}')"
            )
        if not self.title or not self.message or not self.mitigation:
            raise SchemaError(
                "CompatibilityRule requires title, message, mitigation"
            )


class CompatibilityMatrix:
    """S2.5 — registry известных правил совместимости + predicate'ов.

    Использование
    -------------

      from vllm.sndr_core.model_configs.schema import COMPATIBILITY_MATRIX
      forbidden, discouraged = COMPATIBILITY_MATRIX.evaluate(cfg)
      for rule, _msg in forbidden:
          # hard error
      for rule, _msg in discouraged:
          # soft warning

    Rules добавляются через `register(rule, predicate)`. Predicate
    получает целиком ModelConfig и возвращает True если правило сработало.

    Иммутабельность: предполагается единственный экземпляр модуля
    (`COMPATIBILITY_MATRIX`) с фиксированным набором правил, известным
    на момент загрузки. Тесты могут создавать собственные instance для
    изоляции (см. `test_compatibility_matrix.py`).
    """

    def __init__(self) -> None:
        self._rules: list[tuple[CompatibilityRule, Any]] = []

    def register(self, rule: CompatibilityRule, predicate) -> None:
        rule.validate()
        if any(r.id == rule.id for r, _ in self._rules):
            raise SchemaError(
                f"CompatibilityMatrix: duplicate rule id '{rule.id}'"
            )
        self._rules.append((rule, predicate))

    def rules(self) -> list[CompatibilityRule]:
        """Все зарегистрированные правила (для CLI rendering)."""
        return [r for r, _ in self._rules]

    def evaluate(
        self, cfg: "ModelConfig",
    ) -> tuple[list[tuple[CompatibilityRule, str]],
               list[tuple[CompatibilityRule, str]]]:
        """Прогоняет все predicate'ы по cfg.

        Returns (forbidden_violations, discouraged_violations) — каждый
        элемент `(rule, human_message)`. Caller сам решает escalation.
        """
        forbidden: list[tuple[CompatibilityRule, str]] = []
        discouraged: list[tuple[CompatibilityRule, str]] = []
        for rule, pred in self._rules:
            try:
                if pred(cfg):
                    bucket = (forbidden if rule.severity == "forbidden"
                              else discouraged)
                    bucket.append((rule, rule.message))
            except Exception as exc:
                # Predicate exception не должен ронять validate всего конфига —
                # operator увидит warning в логе и сможет починить правило.
                log.warning(
                    "CompatibilityMatrix rule %s predicate raised %r — "
                    "treating as not-applicable",
                    rule.id, exc,
                )
        return forbidden, discouraged


# ──── Predicate helpers (общие проверки для правил) ────────────────────

def _uses_hybrid_gdn(cfg: "ModelConfig") -> bool:
    """Hybrid GDN признак — PN59 streaming-GDN env установлен."""
    return cfg.genesis_env.get("GENESIS_ENABLE_PN59_STREAMING_GDN") == "1"


def _spec_decode_method(cfg: "ModelConfig") -> Optional[str]:
    return cfg.spec_decode.method if cfg.spec_decode else None


def _kv_cache_dtype(cfg: "ModelConfig") -> Optional[str]:
    return cfg.kv_cache_dtype


# ──── Сами правила ─────────────────────────────────────────────────────

_COMPAT_DFLASH_ON_QWEN_NEXT = CompatibilityRule(
    id="COMPAT-001",
    severity="forbidden",
    title="DFlash speculative decode на Qwen-next архитектуре",
    message=(
        "spec_decode.method='dflash' заблокирован для Qwen-next "
        "архитектуры (upstream Qwen3-next): MTP head Qwen-next модели "
        "fused в main model особым образом, который мешает external "
        "drafter speculation. См. audit P2-2 + vllm#42102 для деталей. "
        "Для других hybrid-GDN моделей (Qwen3.6-27B Lorbus etc.) "
        "DFlash работает с отдельным drafter checkpoint."
    ),
    mitigation=(
        "Используйте method='mtp' (Qwen-next's own MTP head — "
        "intended path) или 'ngram'. Если DFlash обязателен — "
        "переключите на model_path с dense-transformer (Qwen3.6-35B-"
        "A3B-FP8) или Qwen3.6 hybrid (27B Lorbus с separate drafter)."
    ),
    references=["docs/PATCHES.md#PN59", "vllm-project/vllm#42102"],
)


_COMPAT_TQK8V4_ON_HYBRID_GDN_NO_P98 = CompatibilityRule(
    id="COMPAT-002",
    severity="discouraged",
    title="TurboQuant k8v4 на hybrid-GDN без P98 lock",
    message=(
        "kv_cache_dtype='turboquant_k8v4' + hybrid-GDN модель без "
        "явного включения P98 (vs vllm#40941 lock) может выдать "
        "non-deterministic prefill в long-context. P98 закрывает race "
        "condition в quantized KV write path."
    ),
    mitigation=(
        "Добавьте `GENESIS_ENABLE_P98=1` в genesis_env "
        "ИЛИ снимите turboquant_k8v4 для hybrid-GDN configs."
    ),
    references=[
        "docs/PATCHES.md#P98",
        "docs/_internal/research/club3090_issue58_long_ctx_vision_oom_2026-05-09.md",
    ],
)


_COMPAT_NGRAM_ON_TQK8V4_LONG_CTX = CompatibilityRule(
    id="COMPAT-003",
    severity="discouraged",
    title="N-gram spec_decode на TQ k8v4 long-context",
    message=(
        "spec_decode.method='ngram' + kv_cache_dtype='turboquant_k8v4' "
        "+ max_model_len > 131072 показал в стресс-тестах падение "
        "acceptance rate с 0.62 до 0.41 после ~10K tokens (cache "
        "thrashing). Для long-context используйте MTP — он не зависит "
        "от prefix cache."
    ),
    mitigation=(
        "Замените method='ngram' на 'mtp' для max_model_len > 131072. "
        "Если ngram необходим (workload без MTP head), уменьшите "
        "max_model_len ≤ 131072."
    ),
    references=["docs/COOKBOOK.md#ngram-vs-mtp"],
)


_COMPAT_DFLASH_REQUIRES_DRAFTER_PATH = CompatibilityRule(
    id="COMPAT-004",
    severity="forbidden",
    title="DFlash без указания drafter model",
    message=(
        "spec_decode.method='dflash' требует отдельный drafter "
        "checkpoint (поле `model`). Без него vllm падает при инициа­"
        "лизации speculative decoder. Это дублирует SpecDecodeConfig."
        "validate() но проверяется и в matrix для глобальной видимости."
    ),
    mitigation=(
        "Укажите `spec_decode.model: /path/to/dflash-drafter` ИЛИ "
        "смените метод на 'mtp' (использует MTP head самой модели)."
    ),
    references=["docs/PATCHES.md#dflash"],
)


COMPATIBILITY_MATRIX = CompatibilityMatrix()
def _is_qwen_next(cfg: "ModelConfig") -> bool:
    """Detect Qwen-next architecture by model_path substring.

    Qwen-next (upstream Qwen3-next) — distinct from Qwen3.6 hybrid
    Mamba (Lorbus). Detected purely by path naming convention.
    """
    p = (cfg.model_path or "").lower()
    return "qwen-next" in p or "qwen3-next" in p


COMPATIBILITY_MATRIX.register(
    _COMPAT_DFLASH_ON_QWEN_NEXT,
    lambda c: _spec_decode_method(c) == "dflash" and _is_qwen_next(c),
)
COMPATIBILITY_MATRIX.register(
    _COMPAT_TQK8V4_ON_HYBRID_GDN_NO_P98,
    lambda c: (
        _kv_cache_dtype(c) == "turboquant_k8v4"
        and _uses_hybrid_gdn(c)
        and c.genesis_env.get("GENESIS_ENABLE_P98") != "1"
    ),
)
COMPATIBILITY_MATRIX.register(
    _COMPAT_NGRAM_ON_TQK8V4_LONG_CTX,
    lambda c: (
        _spec_decode_method(c) == "ngram"
        and _kv_cache_dtype(c) == "turboquant_k8v4"
        and c.max_model_len > 131072
    ),
)
COMPATIBILITY_MATRIX.register(
    _COMPAT_DFLASH_REQUIRES_DRAFTER_PATH,
    lambda c: (
        _spec_decode_method(c) == "dflash"
        and c.spec_decode is not None
        and not c.spec_decode.model
    ),
)


@dataclass
class VerifyTolerances:
    """Acceptable drift before `verify` returns failure."""
    tps_drop_pct_max: float = 5.0       # fail if drop >5%
    tool_call_min: str = "9/10"          # fail if <9/10
    stability_cv_pct_max: float = 6.0    # fail if jitter doubles
    vram_increase_mib_max: int = 2000    # fail if VRAM grew >2 GB

    def validate(self) -> None:
        if self.tps_drop_pct_max < 0:
            raise SchemaError(
                "VerifyTolerances.tps_drop_pct_max must be >= 0"
            )
        if self.stability_cv_pct_max < 0:
            raise SchemaError(
                "VerifyTolerances.stability_cv_pct_max must be >= 0"
            )


# ─── PatchAttribution — structured rationale for ModelDef.patches ─────
#
# Phase A (2026-05-16) introduced patches_attribution on V2 ModelDef.
# Phase B (this commit) lifts the dataclass into V1 schema.py so V1
# ModelConfig can also carry it through compose() into the runtime
# pipeline (patch_plan resolver, sndr patches plan --explain,
# sndr compose render --policy). The dataclass is the same one V2
# uses — schema_v2.py re-exports for back-compat.

_PATCH_ROLES: tuple[str, ...] = (
    "load_bearing",
    "defensive",
    "optional_perf",
    "suspected_regression",
    "no_op",
    "unknown",
)


@dataclass
class PatchAttribution:
    """Why a patch is in the model's canonical set.

    Keyed by registry patch ID (e.g. ``PN204``), not by env-flag name.
    Stored on both ModelDef (authoring) and V1 ModelConfig (after compose)
    so the resolver in `patch_plan.py` can run on the same object the
    rest of the runtime pipeline already consumes.

    Role-conditional required fields (validated at .validate() time):

      load_bearing | suspected_regression  →  ``note`` required
      optional_perf                        →  ``bench_evidence`` required
      defensive | no_op | unknown          →  no auxiliary fields required
    """
    role: str
    note: Optional[str] = None
    bench_evidence: Optional[str] = None
    candidate_when: Optional[dict[str, Any]] = None

    def validate(self, *, key: str) -> None:
        if self.role not in _PATCH_ROLES:
            raise SchemaError(
                f"patches_attribution[{key!r}].role={self.role!r} not in "
                f"{_PATCH_ROLES}"
            )
        if self.role in ("load_bearing", "suspected_regression") and not self.note:
            raise SchemaError(
                f"patches_attribution[{key!r}]: role={self.role!r} requires "
                f"a non-empty `note` (reviewers need to see the rationale)"
            )
        if self.role == "optional_perf" and not self.bench_evidence:
            raise SchemaError(
                f"patches_attribution[{key!r}]: role='optional_perf' requires "
                f"a non-empty `bench_evidence` (perf claims need an anchor)"
            )
        if self.candidate_when is not None and not isinstance(self.candidate_when, dict):
            raise SchemaError(
                f"patches_attribution[{key!r}].candidate_when must be dict | None "
                f"(got {type(self.candidate_when).__name__})"
            )


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
