# SPDX-License-Identifier: Apache-2.0
"""V2 layered config composer (PROJECT_ROADMAP_V2, Phase 1).

`compose(model_id, hardware_id, profile_id=None, runtime=None)` produces
a V1 `ModelConfig` from V2 layered definitions:

    final = ModelDef + HardwareDef + ProfileDef_delta + Runtime_choice

The result is byte-equivalent to a legacy combined YAML — this is the
migration safety net (Phase 2 acceptance: `compose(...).to_dict()` must
match legacy `cfg.to_dict()` per preset).

Conflict semantics (Q1 operator decision):
  • Ownership-based, not override-based.
  • Each field has a single owning layer.
  • Cross-layer conflict on an owned field → SchemaError at load time.
  • Operator who wants a different capability must reference a different
    ModelDef, not override the existing one from the hardware/profile.

Profile delta semantics (Q2 + § 4.3):
  • enable → disable → override applied in that order.
  • Conflicts within a profile (key in both enable and disable) caught
    by `PatchesDelta.validate()`.

Runtime semantics (Q-runtime in § 1):
  • Runtime lives in HardwareDef; CLI `--runtime` overrides default.
  • Override must appear in `hardware.runtime.supported`; else SchemaError.
"""
from __future__ import annotations

from typing import Any, Optional

from .schema import (
    DockerConfig,
    ModelConfig,
    SchemaError,
)
from .schema_v2 import (
    HardwareDef,
    ModelDef,
    PatchesDelta,
    ProfileDef,
    RuntimeBlock,
)


__all__ = [
    "compose",
    "apply_patches_delta",
    "check_compat",
    "render_compression_env",
    "render_backend_env",
    "BACKEND_PLAN_EMISSION_MAP",
]


# ─── Backend-plan single-source-of-truth emission map (P1.8) ────────────
#
# Maps (BackendPlanConfig.field_name, declared_value) → emitted envs
# (a dict of {env_name: env_value} or None when no env is needed).
#
# This is BOTH the compose-time emission source AND the render-launchers
# strict consistency check input. cli/profile.py imports this map for
# its consistency validation; never define a parallel map elsewhere.
#
# Adding a new (field, value) requires:
#   1. Adding the entry here
#   2. Confirming the env(s) actually exist in some Genesis runtime path
#      (or set the value to None for CLI-arg / config-time concerns)
#   3. A unit test that the render emits the expected env(s)
#
# Unknown (field, value) pairs raise SchemaError in
# _validate_backend_plan_consistency() (cli/profile.py).
BACKEND_PLAN_EMISSION_MAP: dict[tuple[str, str], dict[str, str] | None] = {
    # target_default → vLLM CLI flag --attention-backend; no env needed
    ("target_default", "TURBOQUANT"): None,
    ("target_default", "TRITON_ATTN"): None,
    ("target_default", "FLASH_ATTN"): None,
    # target_native_layers → handled via compression_plan skip-list
    # auto-emit (SNDR_G4_TQ_FORCE_SKIP_LAYERS + GENESIS legacy alias)
    ("target_native_layers", "TRITON_ATTN"): None,
    ("target_native_layers", "FLASH_ATTN"): None,
    # drafter_sliding head_size=256 → G4_71b reroutes to Triton
    ("drafter_sliding", "TRITON_ATTN"): {
        "GENESIS_ENABLE_G4_71B_DRAFTER_SLIDING_TRITON": "1",
    },
    # drafter_full head_size=512 → G4_75 reroutes to Triton
    ("drafter_full", "TRITON_ATTN"): {
        "GENESIS_ENABLE_G4_75_DRAFTER_HEAD512_TRITON": "1",
    },
    # P1.8: drafter_kv_sharing
    #   physical → emit G4_76_DISABLE_DRAFTER_KV_SHARING=0 (BOTH SNDR
    #              canonical + GENESIS legacy alias). Required for the
    #              validated β'-A K=4 path; the Gemma4 mapping provider
    #              reads default="1" so explicit "0" is needed to flip
    #              kv_sharing_on=True in artifact_lookup_keys().
    ("drafter_kv_sharing", "physical"): {
        "SNDR_ENABLE_G4_76_DISABLE_DRAFTER_KV_SHARING": "0",
        "GENESIS_ENABLE_G4_76_DISABLE_DRAFTER_KV_SHARING": "0",
    },
    #   disabled → no env emission; runtime default ("1" = disable) applies
    ("drafter_kv_sharing", "disabled"): None,
}


def render_backend_env(profile: Optional[ProfileDef]) -> dict[str, str]:
    """Render ``profile.backend_plan`` declarations into env vars.

    Iterates the BackendPlanConfig fields, looks up each (field, value)
    pair in BACKEND_PLAN_EMISSION_MAP, merges any returned env dicts.
    Unknown (field, value) pairs are caught by
    _validate_backend_plan_consistency() in cli/profile.py — this
    function trusts that validation has run; missing keys here mean
    "no env emission" rather than an error.
    """
    if profile is None or profile.backend_plan is None:
        return {}
    out: dict[str, str] = {}
    for field_name in (
        "target_default", "target_native_layers",
        "drafter_sliding", "drafter_full",
        "drafter_kv_sharing",
    ):
        value = getattr(profile.backend_plan, field_name)
        if value is None:
            continue
        envs = BACKEND_PLAN_EMISSION_MAP.get((field_name, value))
        if envs is None:
            continue
        out.update(envs)
    return out


# ─── Patches delta application ───────────────────────────────────────────


def apply_patches_delta(
    canonical: dict[str, str],
    delta: PatchesDelta,
) -> dict[str, str]:
    """Merge a profile's patches_delta on top of model.patches.

    Order: enable → disable → override. Returns a NEW dict; does not
    mutate `canonical`. Delta validation (intra-profile conflicts) is
    expected to have run before this call (loader invokes it).
    """
    result = dict(canonical)
    for k, v in delta.enable.items():
        result[k] = v
    for k in delta.disable:
        result.pop(k, None)
    for k, v in delta.override.items():
        result[k] = v
    return result


# ─── Runtime-role (P1.2) — compression plan → env emission ───────────────


def render_compression_env(profile: Optional[ProfileDef]) -> dict[str, str]:
    """Render `profile.compression_plan.native_source_layers` into the
    env that the existing G4_60K / PN247 reader honors.

    Currently the reader at
    `vllm/sndr_core/integrations/gemma4/g4_60k_arg_utils.py:198`
    looks for ``GENESIS_G4_TQ_FORCE_SKIP_LAYERS`` directly. We emit
    BOTH the SNDR canonical form (forward-compatible operator surface)
    AND the GENESIS legacy alias (so the existing reader still picks
    it up). When the reader is migrated to accept the SNDR canonical
    via the standard env helper, the GENESIS line should be dropped
    in a follow-up.

    Returns an empty dict for any of:
      * profile is None
      * profile.compression_plan is None
      * compression_plan.native_source_layers is empty
    """
    if profile is None or profile.compression_plan is None:
        return {}
    layers = profile.compression_plan.native_source_layers
    if not layers:
        return {}
    csv = ",".join(str(int(layer)) for layer in layers)
    return {
        "SNDR_G4_TQ_FORCE_SKIP_LAYERS": csv,    # canonical (forward-compatible)
        "GENESIS_G4_TQ_FORCE_SKIP_LAYERS": csv,  # legacy alias (current reader)
    }


def _resolve_kv_cache_dtype(
    model: ModelDef, profile: Optional[ProfileDef],
) -> Optional[str]:
    """Resolve the effective kv_cache_dtype for the composed V1 ModelConfig.

    P1.7a: when a ProfileDef declares ``compression_plan.default_kv_dtype``
    and the parent ``model.capabilities.kv_cache_dtype`` is **neutral**
    (``None`` or ``"auto"``), the profile's concrete dtype is promoted
    into the composed ModelConfig. Without this, a structured-role
    profile with ``default_kv_dtype: turboquant_4bit_nc`` on top of a
    Gemma 4 ModelDef whose ``kv_cache_dtype: auto`` would silently
    render as ``--kv-cache-dtype auto`` and let vLLM pick something
    other than TQ — silently breaking the validated path.

    Concrete parent + matching profile is allowed (rendered dtype is
    the agreed value). Concrete parent + diverging profile is rejected
    by ``_check_compression_kv_dtype_compat`` BEFORE this function runs,
    so the only branches we hit here are:
      * profile None or no compression_plan      → parent value
      * profile has default_kv_dtype + parent None/"auto" → profile value
      * profile has default_kv_dtype + parent concrete (same) → either
        (we return profile since it's been checked equal to parent)
    """
    model_dtype = model.capabilities.kv_cache_dtype
    if profile is None or profile.compression_plan is None:
        return model_dtype
    profile_dtype = profile.compression_plan.default_kv_dtype
    if profile_dtype is None:
        return model_dtype
    if model_dtype is None or model_dtype == "auto":
        # Neutral parent — profile owns the choice (P1.2b semantics)
        return profile_dtype
    # Concrete parent + concrete profile: equality already enforced by
    # _check_compression_kv_dtype_compat; safe to return either.
    return profile_dtype


def _check_compression_kv_dtype_compat(
    model: ModelDef, profile: ProfileDef,
) -> None:
    """If profile declares default_kv_dtype, it must match model's
    canonical kv_cache_dtype — UNLESS the model is dtype-neutral
    (kv_cache_dtype in {None, "auto"}).

    Neutral semantics (P1.2b):
      * ``None``  — ModelDef does not declare a concrete kv_cache_dtype
                    (rare; mostly community models that defer to vLLM).
      * ``"auto"`` — ModelDef explicitly declares "let runtime decide";
                    this is the production-default for Gemma 4 + Qwen
                    in their base ModelDef YAMLs because the concrete
                    dtype is a workload/profile decision (TQ vs native
                    vs FP8), not an inherent model property.

    In both neutral cases the profile is free to set
    compression_plan.default_kv_dtype to a concrete value (e.g.
    "turboquant_4bit_nc" for the structured β'-A K=4 profile).

    For a non-neutral model_dtype (e.g. "fp8_e5m2", "turboquant_4bit_nc"),
    the profile MUST match — divergence is an ownership violation:
    the model says "I run with this dtype" and the profile cannot
    override that without changing parent_model.
    """
    plan = profile.compression_plan
    if plan is None or plan.default_kv_dtype is None:
        return
    model_dtype = model.capabilities.kv_cache_dtype
    # Neutral parent dtype — profile owns the choice.
    if model_dtype is None or model_dtype == "auto":
        return
    if plan.default_kv_dtype != model_dtype:
        raise SchemaError(
            f"profile {profile.id!r} compression_plan.default_kv_dtype="
            f"{plan.default_kv_dtype!r} disagrees with parent "
            f"model.capabilities.kv_cache_dtype={model_dtype!r}. "
            f"The profile cannot override the model's concrete KV "
            f"dtype; remove the profile field, change parent_model, "
            f"or set parent model.kv_cache_dtype='auto' if the dtype "
            f"is actually a workload/profile decision."
        )


# ─── Compatibility check ────────────────────────────────────────────────


def check_compat(model: ModelDef, hardware: HardwareDef) -> Optional[str]:
    """Return an error message if (model, hardware) are incompatible,
    else None. Used by composer pre-merge to fail fast with a clear
    operator-facing message.
    """
    req = model.requires
    hw = hardware.hardware
    n_gpus = int(hw.n_gpus or 0)
    if n_gpus < req.min_gpu_count:
        return (
            f"model {model.id!r} requires min_gpu_count={req.min_gpu_count} "
            f"but hardware {hardware.id!r} has n_gpus={n_gpus}"
        )
    total_vram = n_gpus * int(hw.min_vram_per_gpu_mib or 0)
    if total_vram < req.min_total_vram_mib:
        return (
            f"model {model.id!r} requires min_total_vram_mib={req.min_total_vram_mib} "
            f"but hardware {hardware.id!r} provides {total_vram} MiB total"
        )
    if req.min_cuda_capability and hw.cuda_capability_min:
        if tuple(hw.cuda_capability_min) < tuple(req.min_cuda_capability):
            return (
                f"model {model.id!r} requires CUDA capability "
                f">= {req.min_cuda_capability}, hardware has "
                f"{hw.cuda_capability_min}"
            )
    return None


def _check_profile_targets_model(profile: ProfileDef, model: ModelDef) -> None:
    if profile.parent_model != model.id:
        raise SchemaError(
            f"profile {profile.id!r} targets parent_model={profile.parent_model!r}, "
            f"not the model {model.id!r} passed to compose()"
        )


# ─── Runtime resolution ──────────────────────────────────────────────────


def _resolve_runtime(
    runtime_block: RuntimeBlock,
    runtime_override: Optional[str],
) -> str:
    """Decide which runtime variant to use.

    Default = `runtime_block.default`. Override = CLI `--runtime <name>`,
    must appear in `runtime_block.supported` else SchemaError.
    """
    chosen = runtime_override or runtime_block.default
    if chosen not in runtime_block.supported:
        raise SchemaError(
            f"runtime {chosen!r} not in supported set {runtime_block.supported}; "
            f"add it to hardware.runtime.supported or pick a different rig"
        )
    return chosen


def _render_docker_config(
    runtime_block: RuntimeBlock,
    runtime: str,
    model_id: str,
) -> Optional[DockerConfig]:
    """Build a V1 DockerConfig from V2 RuntimeBlock when runtime is docker/podman."""
    if runtime not in ("docker", "podman"):
        return None
    block = runtime_block.docker if runtime == "docker" else runtime_block.podman
    if block is None:
        raise SchemaError(
            f"runtime={runtime!r} chosen but hardware.runtime.{runtime} block missing"
        )
    container_name = block.container_name_template.replace("{model_id}", model_id)
    return DockerConfig(
        image=block.image,
        container_name=container_name,
        port=block.host_port,            # legacy fallback
        host_port=block.host_port,
        container_port=block.container_port,
        shm_size=block.shm_size,
        network=block.network,
        mounts=list(block.mounts),
        extra_run_flags=list(block.extra_run_flags),
        image_digest=block.image_digest,
    )


# ─── Top-level compose ──────────────────────────────────────────────────


def _merged_attribution(
    model: ModelDef,
    profile: Optional[ProfileDef],
) -> dict[str, Any]:
    """merge ModelDef.patches_attribution with optional
    ProfileDef.patches_delta.attribution.

    Semantics: per-key full replacement (the profile entry overrides
    the model entry in its entirety; partial field merge is not
    supported — keeps the data model simple and the diff explicit).
    Profile entries for patches absent in model attribution are
    additive. Model entries absent from the profile pass through
    unchanged.

    Order: profile wins on conflicts, same precedence as enable /
    disable / override above.
    """
    merged: dict[str, Any] = dict(model.patches_attribution)
    if profile is not None and profile.patches_delta is not None:
        merged.update(profile.patches_delta.attribution)
    return merged


def compose(
    model: ModelDef,
    hardware: HardwareDef,
    profile: Optional[ProfileDef] = None,
    *,
    runtime_override: Optional[str] = None,
) -> ModelConfig:
    """Build a V1 ModelConfig from V2 layered definitions.

    Args:
        model: validated ModelDef.
        hardware: validated HardwareDef.
        profile: optional ProfileDef whose parent_model == model.id.
        runtime_override: CLI `--runtime <name>` override; must be in
            `hardware.runtime.supported`.

    Raises:
        SchemaError on profile→model mismatch, incompatible model/hw pair,
        unsupported runtime, or internal validation failures.

    Returns:
        Composed V1 ModelConfig ready for the existing runtime
        (launch, k8s/compose/quadlet renderers, CompatibilityMatrix).
    """
    # 1. Pre-merge gates (fail fast with clear error messages).
    err = check_compat(model, hardware)
    if err:
        raise SchemaError(err)
    if profile is not None:
        _check_profile_targets_model(profile, model)

    # 2. Resolve runtime + render docker block if applicable.
    runtime = _resolve_runtime(hardware.runtime, runtime_override)
    docker_cfg = _render_docker_config(hardware.runtime, runtime, model.id)

    # 3. Compose patches matrix.
    if profile is not None:
        patches = apply_patches_delta(model.patches, profile.patches_delta)
    else:
        patches = dict(model.patches)

    # 3b. P1.2 — runtime-role compression plan → env emission.
    # Compression plan declares which target layers must stay native
    # (uncompressed) because they're KV-sharing sources for an MTP
    # drafter. The composer renders this into the existing reader's
    # env (GENESIS legacy alias + SNDR canonical, both for the
    # one-release migration window).
    if profile is not None and profile.compression_plan is not None:
        _check_compression_kv_dtype_compat(model, profile)
        compression_env = render_compression_env(profile)
        for k, v in compression_env.items():
            # If the operator already wrote one of these into patches_delta,
            # respect it (operator intent wins). Else add.
            patches.setdefault(k, v)

    # 3c. P1.8 — runtime-role backend plan → env emission.
    # backend_plan declares per-role attention backends; emission map
    # in BACKEND_PLAN_EMISSION_MAP above (used by both compose-time
    # emission AND render-launchers consistency check — single source
    # of truth). Same operator-wins setdefault discipline as 3b.
    if profile is not None and profile.backend_plan is not None:
        backend_env = render_backend_env(profile)
        for k, v in backend_env.items():
            patches.setdefault(k, v)

    # 4. Resolve versions (profile override wins).
    vllm_pin = model.versions.vllm_pin_required
    genesis_pin = model.versions.genesis_pin_min
    if profile is not None and profile.versions_override is not None:
        if profile.versions_override.vllm_pin_required:
            vllm_pin = profile.versions_override.vllm_pin_required
        if profile.versions_override.genesis_pin:
            genesis_pin = profile.versions_override.genesis_pin

    # 4b. Resolve sizing — profile.sizing_override wins (operator tuning
    # for this (model, hardware) pair); otherwise hardware.sizing defaults.
    sizing = hardware.sizing
    if profile is not None and profile.sizing_override is not None:
        sizing = profile.sizing_override

    # 4c. Optional --chat-template override → vllm_extra_args. The
    # template path is the container-side path (typically under
    # /models/<checkpoint>/...) so the operator does not have to add
    # a new mount slot; the existing ${models_dir}:/models:ro mount
    # exposes it. Added 2026-05-14 for the qwen3.6-27b template fix
    # (club-3090 disc #53 — assistant branch did not close </think>
    # before <tool_call>; tools stopped firing in agentic traces).
    vllm_extra_args: list[str] = []
    if getattr(model, "chat_template", None):
        vllm_extra_args.extend(["--chat-template", model.chat_template])

    # 5. Composed key for downstream identification.
    # V1 ModelConfig.key requires strict kebab-case `^[a-z0-9-]+$` —
    # no dots, no underscores. V2 IDs allow dots (e.g. `qwen3.6-fp8`),
    # so we sanitize by replacing `.` and `_` with `-` and joining
    # segments with `--`. Result for V2 ID `qwen3.6-fp8` + hardware
    # `a5000-2x` + profile `wave9-test` is `qwen3-6-fp8--a5000-2x--wave9-test`.
    def _v1_key(segment: str) -> str:
        return segment.replace(".", "-").replace("_", "-")

    composed_key = (
        f"{_v1_key(model.id)}--{_v1_key(hardware.id)}"
        + (f"--{_v1_key(profile.id)}" if profile is not None else "")
    )

    # 6. Build the V1 ModelConfig. Keeping field assignment explicit so
    # the byte-identical regression test can pinpoint any drift.
    return ModelConfig(
        # Identity
        key=composed_key,
        title=f"{model.title} on {hardware.title}",
        description=(
            f"V2-composed: model={model.id} + hardware={hardware.id}"
            + (f" + profile={profile.id}" if profile else "")
        ),
        schema_version=1,                # V1 shape for downstream runtime
        maintainer=model.maintainer,
        model_path=model.model_path,

        # Hardware
        hardware=hardware.hardware,

        # Provenance
        last_validated=model.last_validated,
        genesis_pin=genesis_pin,
        vllm_pin_required=vllm_pin,

        # Model
        served_model_name=model.served_model_name,
        quantization=model.quantization,
        # P1.7a: profile.compression_plan.default_kv_dtype promotes to
        # cfg.kv_cache_dtype when parent ModelDef is neutral (auto/None).
        kv_cache_dtype=_resolve_kv_cache_dtype(model, profile),

        # vLLM serve flags (sizing resolved with profile override above)
        max_model_len=sizing.max_model_len,
        gpu_memory_utilization=sizing.gpu_memory_utilization,
        max_num_seqs=sizing.max_num_seqs,
        max_num_batched_tokens=sizing.max_num_batched_tokens,
        enable_chunked_prefill=sizing.enable_chunked_prefill,
        dtype=model.dtype,
        enforce_eager=sizing.enforce_eager,
        disable_custom_all_reduce=sizing.disable_custom_all_reduce,
        language_model_only=True,
        trust_remote_code=model.trust_remote_code,

        # Capabilities (model-owned, with optional profile spec_decode override)
        # P1.2: profile.spec_decode_override (V1 SpecDecodeConfig type) takes
        # precedence over model.capabilities.spec_decode when set on a
        # runtime-role profile. Used by the structured-role profile to
        # declare K / drafter backend semantics distinct from the default
        # role on the same parent model.
        enable_auto_tool_choice=model.capabilities.enable_auto_tool_choice,
        tool_call_parser=model.capabilities.tool_call_parser,
        reasoning_parser=model.capabilities.reasoning_parser,
        spec_decode=(
            profile.spec_decode_override
            if profile is not None and profile.spec_decode_override is not None
            else model.capabilities.spec_decode
        ),

        # Patches matrix
        genesis_env=patches,
        # copy attribution from ModelDef.
        # Phase D extension: profile.patches_delta.attribution overlays
        # per-key full replacements on top of the model's map. Same
        # merge semantics as enable/override on the patches dict —
        # profile takes precedence.
        patches_attribution=_merged_attribution(model, profile),
        system_env=dict(hardware.system_env),

        # Extra CLI flags (currently only --chat-template; see 4c above)
        vllm_extra_args=vllm_extra_args,

        # Docker
        docker=docker_cfg,

        # API + host (defaults; V2 doesn't introduce a separate concept)
        api_key="genesis-local",
        host="0.0.0.0",
    )
