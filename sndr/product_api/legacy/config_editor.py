# SPDX-License-Identifier: Apache-2.0
"""Read-only V2 configuration catalog and compose preview for GUI editors."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import difflib
import hashlib
import os
import shutil
from pathlib import Path
from typing import Any, Optional

from sndr.model_configs.schema import SchemaError

from .presets import composed_summary


@dataclass(frozen=True)
class V2ConfigItem:
    """One editable V2 catalog entity for the GUI config editor."""

    id: str
    kind: str
    title: str
    source: str
    summary: str = ""
    status: str = ""
    parent_model: Optional[str] = None
    model: Optional[str] = None
    hardware: Optional[str] = None
    profile: Optional[str] = None
    runtime: Optional[str] = None
    fields: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class V2ConfigCatalog:
    """All V2 config layers needed by the graphical editor."""

    models: tuple[V2ConfigItem, ...]
    hardware: tuple[V2ConfigItem, ...]
    profiles: tuple[V2ConfigItem, ...]
    presets: tuple[V2ConfigItem, ...]


@dataclass(frozen=True)
class V2ConfigPreview:
    """Read-only composed preview for a model/hardware/profile selection."""

    selection: dict[str, Optional[str]]
    compatible: bool
    status: str
    messages: tuple[str, ...]
    composed: dict[str, Any]
    draft_yaml: str


@dataclass(frozen=True)
class V2ConfigPlan:
    """Write-safe plan for a V2 preset draft.

    The plan is intentionally non-mutating: GUI clients can show validation,
    diff and target metadata before any future apply endpoint is allowed to
    touch operator-local config files.
    """

    plan_id: str
    preset_id: str
    selection: dict[str, Optional[str]]
    target_path: str
    backup_path: Optional[str]
    action: str
    read_only: bool
    apply_enabled: bool
    valid: bool
    blocked_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    steps: tuple[str, ...]
    diff_lines: tuple[str, ...]
    draft_yaml: str


@dataclass(frozen=True)
class V2ConfigApplyResult:
    """Outcome of an operator-local V2 preset write.

    Writes are confined to ``model_configs_user_dir()/presets``. The repo
    builtin catalog and any remote server are never touched.
    """

    plan_id: str
    preset_id: str
    target_path: str
    backup_path: Optional[str]
    action: str
    written: bool
    bytes_written: int
    status: str  # "applied" | "blocked" | "conflict"
    message: str
    blocked_reasons: tuple[str, ...]


@dataclass(frozen=True)
class UserPreset:
    """A preset YAML discovered in the operator-local config dir."""

    id: str
    path: str
    model: Optional[str]
    hardware: Optional[str]
    profile: Optional[str]
    runtime: Optional[str]
    size_bytes: int


import threading as _threading
import time as _time

# The V2 catalog reads + parses every model/hardware/profile/preset YAML, so it
# is the slowest read endpoint (~0.5s). Configs change rarely (and only via the
# apply path), so we cache it with a short TTL and let apply invalidate it. The
# daemon also warms this at startup so the first GUI load is instant.
_CATALOG_CACHE: dict[str, Any] = {"data": None, "ts": 0.0}
_CATALOG_LOCK = _threading.Lock()
_CATALOG_TTL = 60.0


def invalidate_v2_config_catalog() -> None:
    """Drop the cached catalog (call after a config apply changes the YAMLs)."""
    with _CATALOG_LOCK:
        _CATALOG_CACHE["data"] = None


def collect_v2_config_catalog(*, max_age: float = _CATALOG_TTL) -> V2ConfigCatalog:
    """Return V2 models, hardware, profiles and presets as GUI records (cached)."""
    with _CATALOG_LOCK:
        cached = _CATALOG_CACHE["data"]
        if cached is not None and (_time.time() - _CATALOG_CACHE["ts"]) < max_age:
            return cached
    result = _build_v2_config_catalog()
    with _CATALOG_LOCK:
        _CATALOG_CACHE["data"] = result
        _CATALOG_CACHE["ts"] = _time.time()
    return result


def _build_v2_config_catalog() -> V2ConfigCatalog:
    """Read + parse the V2 corpus from disk (the uncached build)."""
    from sndr.model_configs.registry_v2 import (
        list_hardware,
        list_models,
        list_presets,
        list_profiles,
        load_hardware,
        load_model,
        load_preset_def,
        load_profile,
    )

    models = tuple(
        _model_item(model_id, load_model(model_id))
        for model_id in list_models()
    )
    hardware = tuple(
        _hardware_item(hardware_id, load_hardware(hardware_id))
        for hardware_id in list_hardware()
    )
    profiles = tuple(
        _profile_item(profile_id, load_profile(profile_id))
        for profile_id in list_profiles()
    )
    presets = tuple(
        _preset_item(preset_id, load_preset_def(preset_id))
        for preset_id in list_presets()
    )
    return V2ConfigCatalog(
        models=models,
        hardware=hardware,
        profiles=profiles,
        presets=presets,
    )


def preview_v2_config(
    *,
    model_id: str,
    hardware_id: str,
    profile_id: Optional[str] = None,
    runtime: Optional[str] = None,
) -> V2ConfigPreview:
    """Compose a selected V2 triplet without writing any config files."""
    from sndr.model_configs.registry_v2 import (
        compose_by_ids,
        load_profile,
    )

    messages: list[str] = []
    compatible = True
    if profile_id:
        try:
            profile = load_profile(profile_id)
            if profile.parent_model != model_id:
                compatible = False
                messages.append(
                    "Profile parent_model does not match selected model."
                )
        except SchemaError as exc:
            compatible = False
            messages.append(str(exc))

    composed: dict[str, Any] = {}
    status = "ready"
    if compatible:
        try:
            composed = composed_summary(
                compose_by_ids(
                    model_id=model_id,
                    hardware_id=hardware_id,
                    profile_id=profile_id or None,
                    runtime=runtime or None,
                )
            )
            messages.append("Selection composes successfully.")
        except Exception as exc:
            compatible = False
            status = "error"
            messages.append(f"{type(exc).__name__}: {exc}")
    else:
        status = "blocked"

    return V2ConfigPreview(
        selection={
            "model": model_id,
            "hardware": hardware_id,
            "profile": profile_id,
            "runtime": runtime,
        },
        compatible=compatible,
        status=status,
        messages=tuple(messages),
        composed=composed,
        draft_yaml=_draft_yaml(
            model_id=model_id,
            hardware_id=hardware_id,
            profile_id=profile_id,
            runtime=runtime,
        ),
    )


def plan_v2_config_edit(
    *,
    preset_id: Optional[str],
    model_id: str,
    hardware_id: str,
    profile_id: Optional[str] = None,
    runtime: Optional[str] = None,
) -> V2ConfigPlan:
    """Build a non-mutating save plan for a V2 preset draft."""
    from sndr.engines.vllm.locations.project_paths import model_configs_user_dir
    from sndr.model_configs.schema_v2 import _check_id

    safe_preset_id = (preset_id or _default_draft_preset_id(
        model_id=model_id,
        hardware_id=hardware_id,
        profile_id=profile_id,
    )).strip()

    blocked: list[str] = []
    warnings: list[str] = []
    target_file_id = safe_preset_id
    try:
        _check_id(safe_preset_id, "preset.id")
    except SchemaError as exc:
        blocked.append(str(exc))
        target_file_id = "invalid-preset-id"

    preview = preview_v2_config(
        model_id=model_id,
        hardware_id=hardware_id,
        profile_id=profile_id,
        runtime=runtime,
    )
    if not preview.compatible:
        blocked.extend(preview.messages)

    target = model_configs_user_dir() / "presets" / f"{target_file_id}.yaml"
    backup = target.with_suffix(".yaml.bak") if target.exists() else None
    current_lines = _read_text_lines(target)
    draft_lines = preview.draft_yaml.splitlines()
    action = "update" if target.exists() else "create"

    if not target.parent.exists():
        warnings.append(
            f"Target directory does not exist yet: {target.parent}"
        )
    warnings.append(
        "Plan is read-only. Apply requires an explicit future config apply endpoint."
    )

    diff_lines = tuple(difflib.unified_diff(
        current_lines,
        draft_lines,
        fromfile=str(target) if current_lines else "/dev/null",
        tofile=str(target),
        lineterm="",
    ))
    plan_id = "cfgplan_" + hashlib.sha256(
        "\n".join([
            safe_preset_id,
            model_id,
            hardware_id,
            profile_id or "",
            runtime or "",
            preview.draft_yaml,
        ]).encode("utf-8")
    ).hexdigest()[:12]

    valid = not blocked
    return V2ConfigPlan(
        plan_id=plan_id,
        preset_id=safe_preset_id,
        selection=preview.selection,
        target_path=str(target),
        backup_path=str(backup) if backup else None,
        action=action,
        read_only=True,
        apply_enabled=False,
        valid=valid,
        blocked_reasons=tuple(blocked),
        warnings=tuple(warnings),
        steps=(
            "Validate V2 layer ids and profile/model compatibility.",
            "Render deterministic preset YAML draft.",
            "Compare draft against operator-local target file.",
            "Require explicit apply endpoint before writing any file.",
        ),
        diff_lines=diff_lines,
        draft_yaml=preview.draft_yaml,
    )


def get_v2_layer(kind: str, layer_id: str) -> dict[str, Any]:
    """Return the full, JSON-safe definition of one V2 layer.

    Exposes every parameter of a model/hardware/profile (capabilities,
    requires, versions, sizing, runtime, patch matrix, deltas) so the GUI can
    render and inspect the complete config — not just the dashboard subset.
    """
    from sndr.model_configs.registry_v2 import (
        load_hardware,
        load_model,
        load_preset_def,
        load_profile,
    )

    normalized = kind.lower()
    if normalized == "model":
        obj: Any = load_model(layer_id)
        source_layer = "model"
    elif normalized == "hardware":
        obj = load_hardware(layer_id)
        source_layer = "hardware"
    elif normalized == "profile":
        obj = load_profile(layer_id)
        source_layer = "profile"
    elif normalized == "preset":
        obj = load_preset_def(layer_id)
        source_layer = "presets"
    else:
        raise ValueError(f"Unknown layer kind: {kind}")
    return {
        "kind": normalized,
        "id": layer_id,
        "source": _source_path(source_layer, layer_id),
        "definition": dataclasses.asdict(obj),
    }


@dataclass(frozen=True)
class V2LayerApplyResult:
    """Outcome of writing an edited layer YAML to the operator-local dir."""

    kind: str
    layer_id: str
    target_path: str
    backup_path: Optional[str]
    action: str
    written: bool
    bytes_written: int
    status: str  # "applied" | "blocked" | "conflict"
    message: str
    blocked_reasons: tuple[str, ...]


_LAYER_SUBDIRS = {"model": "model", "hardware": "hardware", "profile": "profile", "preset": "presets"}


def apply_v2_layer(*, kind: str, layer_id: str, yaml_text: str) -> V2LayerApplyResult:
    """Write an edited layer definition to ``model_configs_user_dir()/<kind>``.

    Operator-local only — never the repo builtin catalog, never a remote host.
    Atomic write with backup + an exclusive lock; refuses bad kind/id/empty body.
    """
    from sndr.engines.vllm.locations.project_paths import model_configs_user_dir
    from sndr.model_configs.schema_v2 import _check_id

    normalized = kind.lower()
    safe_id = (layer_id or "").strip()
    blocked: list[str] = []
    if normalized not in _LAYER_SUBDIRS:
        blocked.append(f"Unknown layer kind: {kind}")
    if not (yaml_text or "").strip():
        blocked.append("Refusing to write an empty document.")
    if normalized in _LAYER_SUBDIRS:
        try:
            _check_id(safe_id, f"{normalized}.id")
        except SchemaError as exc:
            blocked.append(str(exc))

    def _result(*, status, written, message, target, backup, bytes_written, action, reasons):
        return V2LayerApplyResult(
            kind=normalized, layer_id=safe_id, target_path=str(target) if target else "",
            backup_path=backup, action=action, written=written, bytes_written=bytes_written,
            status=status, message=message, blocked_reasons=tuple(reasons),
        )

    if blocked:
        return _result(status="blocked", written=False, message="Refusing to write.",
                       target=None, backup=None, bytes_written=0, action="create", reasons=blocked)

    target = model_configs_user_dir() / _LAYER_SUBDIRS[normalized] / f"{safe_id}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_suffix(".yaml.lock")
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(fd)
    except FileExistsError:
        return _result(status="conflict", written=False, message=f"Another write holds the lock: {lock}",
                       target=target, backup=None, bytes_written=0, action="create", reasons=())

    try:
        action = "update" if target.exists() else "create"
        backup_path: Optional[str] = None
        if target.exists():
            backup = target.with_suffix(".yaml.bak")
            shutil.copy2(target, backup)
            backup_path = str(backup)
        content = yaml_text if yaml_text.endswith("\n") else yaml_text + "\n"
        tmp = target.with_suffix(".yaml.tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, target)
        bytes_written = len(content.encode("utf-8"))
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass

    return _result(
        status="applied", written=True,
        message=f"Wrote {normalized}/{safe_id} to the operator-local config dir.",
        target=target, backup=backup_path, bytes_written=bytes_written, action=action, reasons=(),
    )


def apply_v2_config_plan(
    *,
    preset_id: Optional[str],
    model_id: str,
    hardware_id: str,
    profile_id: Optional[str] = None,
    runtime: Optional[str] = None,
    expected_plan_id: Optional[str] = None,
) -> V2ConfigApplyResult:
    """Write a validated V2 preset draft into the operator-local config dir.

    Safety contract:

      * writes only under ``model_configs_user_dir()/presets`` — never the
        repo builtin catalog and never a remote server;
      * refuses to write when the plan is invalid/blocked;
      * refuses when ``expected_plan_id`` (the plan the operator previewed)
        no longer matches the freshly derived plan — prevents applying a
        stale preview;
      * backs up an existing target to ``<name>.yaml.bak`` before writing;
      * writes atomically via a temp file + ``os.replace``;
      * guards concurrent writers with an exclusive ``.lock`` file.
    """
    plan = plan_v2_config_edit(
        preset_id=preset_id,
        model_id=model_id,
        hardware_id=hardware_id,
        profile_id=profile_id,
        runtime=runtime,
    )
    target = Path(plan.target_path)

    def _result(*, status: str, written: bool, message: str,
                backup: Optional[str], bytes_written: int,
                reasons: tuple[str, ...]) -> V2ConfigApplyResult:
        return V2ConfigApplyResult(
            plan_id=plan.plan_id,
            preset_id=plan.preset_id,
            target_path=str(target),
            backup_path=backup,
            action=plan.action,
            written=written,
            bytes_written=bytes_written,
            status=status,
            message=message,
            blocked_reasons=reasons,
        )

    if not plan.valid:
        return _result(
            status="blocked",
            written=False,
            message="Plan is not valid; refusing to write.",
            backup=None,
            bytes_written=0,
            reasons=plan.blocked_reasons,
        )
    if expected_plan_id is not None and expected_plan_id != plan.plan_id:
        return _result(
            status="conflict",
            written=False,
            message="Plan changed since preview; re-plan before applying.",
            backup=None,
            bytes_written=0,
            reasons=(),
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_suffix(".yaml.lock")
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(fd)
    except FileExistsError:
        return _result(
            status="conflict",
            written=False,
            message=f"Another apply holds the lock: {lock}",
            backup=None,
            bytes_written=0,
            reasons=(),
        )

    try:
        backup_path: Optional[str] = None
        if target.exists():
            backup = target.with_suffix(".yaml.bak")
            shutil.copy2(target, backup)
            backup_path = str(backup)
        content = plan.draft_yaml
        if not content.endswith("\n"):
            content += "\n"
        tmp = target.with_suffix(".yaml.tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, target)
        bytes_written = len(content.encode("utf-8"))
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass

    return _result(
        status="applied",
        written=True,
        message=f"Wrote '{plan.preset_id}' to operator-local config dir.",
        backup=backup_path,
        bytes_written=bytes_written,
        reasons=(),
    )


def list_user_presets() -> tuple[UserPreset, ...]:
    """List preset YAMLs in the operator-local config dir (read-only)."""
    from sndr.engines.vllm.locations.project_paths import model_configs_user_dir

    presets_dir = model_configs_user_dir() / "presets"
    if not presets_dir.is_dir():
        return ()
    records: list[UserPreset] = []
    for path in sorted(presets_dir.glob("*.yaml")):
        fields = _scan_preset_yaml(path)
        records.append(
            UserPreset(
                id=path.stem,
                path=str(path),
                model=fields.get("model"),
                hardware=fields.get("hardware"),
                profile=fields.get("profile"),
                runtime=fields.get("runtime"),
                size_bytes=path.stat().st_size,
            )
        )
    return tuple(records)


def _scan_preset_yaml(path: Path) -> dict[str, Optional[str]]:
    """Pull top-level scalar fields from a preset YAML without full parse.

    Drafts may be partial, so we avoid schema validation here and only read
    the unindented ``key: value`` lines the editor cares about.
    """
    fields: dict[str, Optional[str]] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return fields
    for line in text.splitlines():
        if not line or line[0] in (" ", "\t", "#"):
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        if key in {"model", "hardware", "profile", "runtime"}:
            fields[key] = value.strip() or None
    return fields


def _model_item(model_id: str, obj: Any) -> V2ConfigItem:
    fields = dataclasses.asdict(obj)
    caps = fields.get("capabilities", {}) or {}
    requires = fields.get("requires", {}) or {}
    versions = fields.get("versions", {}) or {}
    spec = caps.get("spec_decode", {}) or {}
    return V2ConfigItem(
        id=model_id,
        kind="model",
        title=str(fields.get("title") or model_id),
        source=_source_path("model", model_id),
        summary=str(fields.get("model_path") or ""),
        status=str(fields.get("last_validated") or ""),
        fields={
            "served_model_name": fields.get("served_model_name"),
            "dtype": fields.get("dtype"),
            "quantization": fields.get("quantization"),
            "attention_arch": caps.get("attention_arch"),
            "kv_cache_dtype": caps.get("kv_cache_dtype"),
            "tool_call_parser": caps.get("tool_call_parser"),
            "reasoning_parser": caps.get("reasoning_parser"),
            # Spec-decode + pin alignment — previously reachable only via the
            # v2Layer (full-definition) call, so every catalog/inspector view but
            # models-workbench was blind to which pin a config needs and its
            # drafter. Surfaced here so config/preset views show pin alignment.
            "spec_decode_method": spec.get("method"),
            "spec_decode_drafter": spec.get("model"),
            "vllm_pin_required": versions.get("vllm_pin_required"),
            "pin_hold": versions.get("pin_hold"),
            "reference_metrics_ref": versions.get("reference_metrics_ref"),
            "min_gpu_count": requires.get("min_gpu_count"),
            "min_total_vram_mib": requires.get("min_total_vram_mib"),
            "patch_count": len(fields.get("patches") or {}),
        },
    )


def _hardware_item(hardware_id: str, obj: Any) -> V2ConfigItem:
    fields = dataclasses.asdict(obj)
    hardware = fields.get("hardware", {})
    sizing = fields.get("sizing", {})
    runtime = fields.get("runtime", {})
    return V2ConfigItem(
        id=hardware_id,
        kind="hardware",
        title=str(fields.get("title") or hardware_id),
        source=_source_path("hardware", hardware_id),
        summary=(
            f"{hardware.get('n_gpus', '-')} GPU / "
            f"{hardware.get('min_vram_per_gpu_mib', '-')} MiB min VRAM"
        ),
        runtime=runtime.get("default"),
        fields={
            "n_gpus": hardware.get("n_gpus"),
            "cuda_capability_min": hardware.get("cuda_capability_min"),
            "max_model_len": sizing.get("max_model_len"),
            "max_num_seqs": sizing.get("max_num_seqs"),
            "gpu_memory_utilization": sizing.get("gpu_memory_utilization"),
            "runtime_default": runtime.get("default"),
            "runtime_supported": runtime.get("supported"),
        },
    )


def _profile_item(profile_id: str, obj: Any) -> V2ConfigItem:
    fields = dataclasses.asdict(obj)
    sizing = fields.get("sizing_override") or {}
    delta = fields.get("patches_delta") or {}
    return V2ConfigItem(
        id=profile_id,
        kind="profile",
        title=profile_id,
        source=_source_path("profile", profile_id),
        status=str(fields.get("status") or ""),
        parent_model=fields.get("parent_model"),
        summary=str(fields.get("created") or ""),
        fields={
            "max_model_len": sizing.get("max_model_len"),
            "max_num_seqs": sizing.get("max_num_seqs"),
            "gpu_memory_utilization": sizing.get("gpu_memory_utilization"),
            "enable_delta": len(delta.get("enable") or {}),
            "disable_delta": len(delta.get("disable") or {}),
            "override_delta": len(delta.get("override") or {}),
        },
    )


def _preset_item(preset_id: str, obj: Any) -> V2ConfigItem:
    card = dataclasses.asdict(obj.card) if obj.card else {}
    return V2ConfigItem(
        id=preset_id,
        kind="preset",
        title=str(card.get("title") or preset_id),
        source=_source_path("presets", preset_id),
        summary=str(card.get("summary") or ""),
        status=str(card.get("status") or ""),
        model=obj.model,
        hardware=obj.hardware,
        profile=obj.profile,
        runtime=obj.runtime,
        fields={
            "mode": card.get("mode"),
            "routing_family": card.get("routing_family"),
            "fallback_preset": card.get("fallback_preset"),
            "workload_allow": card.get("workload_allow") or [],
            "workload_deny": card.get("workload_deny") or [],
        },
    )


def _source_path(layer: str, item_id: str) -> str:
    from sndr.model_configs.registry_v2 import REPO_ROOT_HINT

    filename = f"{item_id}.yaml"
    if layer == "presets":
        path = REPO_ROOT_HINT / "builtin" / "presets" / filename
    else:
        builtin = REPO_ROOT_HINT / "builtin" / layer / filename
        community = REPO_ROOT_HINT / "community" / layer / filename
        path = builtin if builtin.exists() else community
    return str(Path(path))


def _draft_yaml(
    *,
    model_id: str,
    hardware_id: str,
    profile_id: Optional[str],
    runtime: Optional[str],
) -> str:
    lines = [
        "schema_version: 2",
        "kind: preset",
        f"model: {model_id}",
        f"hardware: {hardware_id}",
    ]
    if profile_id:
        lines.append(f"profile: {profile_id}")
    if runtime:
        lines.append(f"runtime: {runtime}")
    lines.extend(
        [
            "card:",
            "  title: Draft preset",
            "  summary: Generated by SNDR GUI V2 config editor preview.",
            "  status: experimental",
            "  audience: operator",
            "  maturity: draft",
        ]
    )
    return "\n".join(lines)


def _default_draft_preset_id(
    *,
    model_id: str,
    hardware_id: str,
    profile_id: Optional[str],
) -> str:
    parts = ["gui-draft", model_id, hardware_id]
    if profile_id:
        parts.append(profile_id)
    raw = "-".join(parts)
    return raw.replace("_", "-")[:96].strip(".-")


def _read_text_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8").splitlines()


__all__ = [
    "UserPreset",
    "V2ConfigApplyResult",
    "V2LayerApplyResult",
    "apply_v2_layer",
    "V2ConfigCatalog",
    "V2ConfigItem",
    "V2ConfigPlan",
    "V2ConfigPreview",
    "apply_v2_config_plan",
    "collect_v2_config_catalog",
    "get_v2_layer",
    "list_user_presets",
    "plan_v2_config_edit",
    "preview_v2_config",
]
