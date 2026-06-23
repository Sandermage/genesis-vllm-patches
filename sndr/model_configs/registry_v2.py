# SPDX-License-Identifier: Apache-2.0
"""V2 layered registry — YAML loaders + alias resolver (PROJECT_ROADMAP_V2 Phase 1).

Discovery layout (per § 4.4):

  sndr/model_configs/builtin/
  ├── model/<id>.yaml         → ModelDef
  ├── hardware/<id>.yaml      → HardwareDef
  ├── profile/<id>.yaml       → ProfileDef
  └── presets/<alias>.yaml    → triplet {model, hardware, profile?, runtime?}

  sndr/model_configs/community/
  ├── hardware/<id>.yaml      → HardwareDef (community-tier)
  └── profile/<id>.yaml       → ProfileDef (community-tier)

`load_alias(name)` resolves a preset alias → composed V1 ModelConfig.
`compose_by_ids(model_id, hw_id, profile_id, runtime)` is the
non-alias entry point used by `sndr launch --model X --hardware Y ...`.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Optional

from .schema import ModelConfig, SchemaError
from .schema_v2 import (
    HardwareDef,
    ModelDef,
    PatchManifest,
    ProfileDef,
)
from .preset_schema import PresetDef, parse_preset_yaml
from .compose import compose


__all__ = [
    "REPO_ROOT_HINT",
    "load_model",
    "load_hardware",
    "load_profile",
    "load_alias",
    "load_preset_def",
    "compose_by_ids",
    "list_models",
    "list_hardware",
    "list_profiles",
    "list_presets",
    "load_patch_manifest",
]


# CONFIG-UX.1 — one-time warning per unannotated preset, similar shape
# to V1 `_maybe_warn_v1_deprecation`. Operators see one warning per
# preset per process; CI sweeps that exercise many presets don't flood.
_UNANNOTATED_PRESET_WARNED: set[str] = set()


def _maybe_warn_unannotated(
    preset_id: str,
    *,
    stage: Optional[int] = None,
) -> None:
    """Emit a stage-aware CONFIG-UX hint for a card-less preset.

    Backwards-compatible signature: positional `preset_id` arg unchanged;
    `stage` is keyword-only with default = read from env via
    `_rollout.rollout_stage()`.

    Severity is resolved per CONFIG_UX_R §6.1 + CONFIG_UX_4_R §2.2:

      - prod-* preset (id starts with "prod-"):
          bucket = card_less_prod → WARN at Stage 0-2 (default),
          ERROR at Stage 3+ (raises RuntimeError).

      - non-prod preset (example-*, qa-*, experimental-*, long-ctx-*, ...):
          bucket = card_less_non_prod → INFO indefinitely (silenced).
          CONFIG-UX.2b will annotate these separately.

    Once-per-process tracking + GENESIS_DISABLE_V1_DEPRECATION_WARNING
    escape hatch preserved (does NOT silence ERROR severity).
    """
    from ._rollout import effective_severity, is_disabled
    if preset_id in _UNANNOTATED_PRESET_WARNED:
        return
    _UNANNOTATED_PRESET_WARNED.add(preset_id)

    bucket = "card_less_prod" if preset_id.startswith("prod-") else "card_less_non_prod"
    severity = effective_severity(
        bucket=bucket,  # type: ignore[arg-type]
        stage=stage,
    )

    if severity == "info" or (severity == "warn" and is_disabled()):
        return

    msg = (
        f"V2 preset {preset_id!r} lacks operator `card:` annotation. "
        f"Add a card to enable `sndr preset list/show/explain/recommend` "
        f"(CONFIG-UX.2). Legacy 3-pointer load path remains supported."
    )

    if severity == "error":
        raise RuntimeError(msg)

    import warnings
    warnings.warn(msg, DeprecationWarning, stacklevel=3)


# Resolved at import time so tests can monkeypatch.
_PKG_ROOT = Path(__file__).resolve().parent
REPO_ROOT_HINT = _PKG_ROOT


# ─── Helpers ─────────────────────────────────────────────────────────────


def _yaml_safe_load(path: Path) -> dict:
    try:
        import yaml
    except ImportError as e:  # pragma: no cover — pyyaml is a hard dep
        raise RuntimeError(
            "V2 registry requires `pyyaml` — `pip install pyyaml`"
        ) from e
    if not path.is_file():
        raise SchemaError(f"V2 YAML not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SchemaError(f"{path}: top-level YAML must be a mapping")
    return data


def _resolve_field_type(cls, field_name: str):
    """Resolve a dataclass field's runtime type, unwrapping Optional / Union.

    PEP 563 `from __future__ import annotations` defers annotation
    resolution → `dataclasses.Field.type` is a string. We use
    `typing.get_type_hints()` to materialise it once per class, then
    strip Optional/Union[X, None] down to X for nested-dataclass detection.
    """
    import typing
    try:
        hints = typing.get_type_hints(cls)
    except Exception:
        return None
    t = hints.get(field_name)
    if t is None:
        return None
    # Optional[X] = Union[X, None] → return X if exactly one non-None arg.
    origin = typing.get_origin(t)
    if origin is typing.Union:
        args = [a for a in typing.get_args(t) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return t


def _dataclass_from_dict(cls, data: dict):
    """Construct a dataclass instance from a YAML-loaded dict.

    Recursively materialises nested dataclass fields. PEP 563 annotations
    are resolved via `_resolve_field_type` so we get real classes, not
    string forward references.

    List/tuple fields of dataclass element type are also materialised
    (e.g. `target_files: list[PatchTargetFile]`).
    """
    if not dataclasses.is_dataclass(cls):
        return data
    import typing
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = _resolve_field_type(cls, f.name)
        if value is None:
            kwargs[f.name] = None
            continue
        # Nested dataclass {dict} → recurse.
        if isinstance(value, dict) and dataclasses.is_dataclass(ftype):
            kwargs[f.name] = _dataclass_from_dict(ftype, value)
            continue
        # list[Dataclass] → recurse per element.
        if isinstance(value, list) and ftype is not None:
            origin = typing.get_origin(ftype)
            if origin in (list, tuple):
                args = typing.get_args(ftype)
                if args and dataclasses.is_dataclass(args[0]):
                    kwargs[f.name] = [
                        _dataclass_from_dict(args[0], v) if isinstance(v, dict) else v
                        for v in value
                    ]
                    continue
        # dict[str, Dataclass] → recurse per value.
        # added so ModelDef.patches_attribution
        # `dict[str, PatchAttribution]` materialises through YAML load.
        if isinstance(value, dict) and ftype is not None:
            origin = typing.get_origin(ftype)
            if origin is dict:
                args = typing.get_args(ftype)
                if len(args) == 2 and dataclasses.is_dataclass(args[1]):
                    kwargs[f.name] = {
                        k: (_dataclass_from_dict(args[1], v) if isinstance(v, dict) else v)
                        for k, v in value.items()
                    }
                    continue
        kwargs[f.name] = value
    return cls(**kwargs)


# ─── Layer loaders ───────────────────────────────────────────────────────


def _builtin_dir(layer: str) -> Path:
    return _PKG_ROOT / "builtin" / layer


def _community_dir(layer: str) -> Path:
    return _PKG_ROOT / "community" / layer


def load_model(model_id: str) -> ModelDef:
    """Load `builtin/model/<id>.yaml` into a validated ModelDef."""
    path = _builtin_dir("model") / f"{model_id}.yaml"
    data = _yaml_safe_load(path)
    obj = _dataclass_from_dict(ModelDef, data)
    obj.validate()
    return obj


def load_hardware(hw_id: str) -> HardwareDef:
    """Load `builtin/hardware/<id>.yaml` → HardwareDef. Falls back to
    `community/hardware/<id>.yaml` if not in builtin (Q3 hybrid)."""
    candidates = [
        _builtin_dir("hardware") / f"{hw_id}.yaml",
        _community_dir("hardware") / f"{hw_id}.yaml",
    ]
    for p in candidates:
        if p.is_file():
            data = _yaml_safe_load(p)
            obj = _dataclass_from_dict(HardwareDef, data)
            obj.validate()
            return obj
    raise SchemaError(
        f"hardware {hw_id!r} not found in builtin/ or community/ directories"
    )


def load_profile(profile_id: str) -> ProfileDef:
    """Load `builtin/profile/<id>.yaml` → ProfileDef. Falls back to
    `community/profile/<id>.yaml`."""
    candidates = [
        _builtin_dir("profile") / f"{profile_id}.yaml",
        _community_dir("profile") / f"{profile_id}.yaml",
    ]
    for p in candidates:
        if p.is_file():
            data = _yaml_safe_load(p)
            obj = _dataclass_from_dict(ProfileDef, data)
            obj.validate()
            return obj
    raise SchemaError(
        f"profile {profile_id!r} not found in builtin/ or community/"
    )


def load_patch_manifest(path: Path) -> PatchManifest:
    """Load a `plugins/community/<user>/<id>/manifest.yaml` (community SDK).

    Path is explicit because community plugins live outside the
    model_configs tree (see PROJECT_ROADMAP_V2 § 4.4).
    """
    data = _yaml_safe_load(path)
    obj = _dataclass_from_dict(PatchManifest, data)
    obj.validate()
    return obj


# ─── Listing ─────────────────────────────────────────────────────────────


def _list_yaml_ids(directory: Path) -> list[str]:
    if not directory.is_dir():
        return []
    return sorted(
        p.stem for p in directory.glob("*.yaml")
        if p.is_file() and not p.stem.startswith("_")
    )


def list_models() -> list[str]:
    return _list_yaml_ids(_builtin_dir("model"))


def list_hardware() -> list[str]:
    return sorted(set(
        _list_yaml_ids(_builtin_dir("hardware"))
        + _list_yaml_ids(_community_dir("hardware"))
    ))


def list_profiles(parent_model: Optional[str] = None) -> list[str]:
    """List profile ids; optionally filter to those whose `parent_model`
    matches the given model id (loads each profile to inspect)."""
    ids = sorted(set(
        _list_yaml_ids(_builtin_dir("profile"))
        + _list_yaml_ids(_community_dir("profile"))
    ))
    if parent_model is None:
        return ids
    out: list[str] = []
    for pid in ids:
        try:
            if load_profile(pid).parent_model == parent_model:
                out.append(pid)
        except SchemaError:
            continue
    return out


def _user_presets_dir() -> Optional[Path]:
    """Operator-local preset dir (``model_configs_user_dir()/presets``).

    Presets written by the GUI / Product API land here. They participate in
    listing and resolution so the operator's edits take effect. Returns None
    if the location cannot be resolved (keeps builtin-only behavior intact).
    """
    try:
        from sndr.engines.vllm.locations.project_paths import model_configs_user_dir

        return model_configs_user_dir() / "presets"
    except Exception:
        return None


def _preset_path(alias: str) -> Path:
    """Resolve a preset alias to a YAML path, operator-local dir taking
    precedence over the builtin catalog (operator edits win)."""
    user_dir = _user_presets_dir()
    if user_dir is not None:
        candidate = user_dir / f"{alias}.yaml"
        if candidate.is_file():
            return candidate
    return _alias_dir() / f"{alias}.yaml"


def list_presets() -> list[str]:
    """List preset alias ids: builtin catalog plus operator-local presets.

    Presets are operator-facing catalog entries, so GUI/Product API callers
    need the same stable listing primitive that models, hardware, and
    profiles already expose. Operator-local presets (written under
    ``model_configs_user_dir()/presets``) are included so the GUI edit loop is
    closed: a saved preset shows up in the catalog and composes.
    """
    ids = set(_list_yaml_ids(_alias_dir()))
    user_dir = _user_presets_dir()
    if user_dir is not None:
        ids.update(_list_yaml_ids(user_dir))
    return sorted(ids)


# ─── Alias + compose entry points ────────────────────────────────────────


def _alias_dir() -> Path:
    return _PKG_ROOT / "builtin" / "presets"


def load_alias(alias: str) -> ModelConfig:
    """Resolve `presets/<alias>.yaml` → composed V1 ModelConfig.

    Two YAML shapes accepted (CONFIG-UX.1):

    1) Legacy 3-pointer (backwards-compat — all 21 builtin presets):

         model:    <model_id>            # required
         hardware: <hardware_id>         # required
         profile:  <profile_id>          # optional
         runtime:  <runtime>             # optional

       Loader emits a one-time DeprecationWarning suggesting `card:`
       annotation (CONFIG-UX.2 work). Composition path unchanged.

    2) Card-annotated (CONFIG-UX.1 forward-shape):

         model: ...
         hardware: ...
         profile: ...
         card:
           title: ...
           summary: ...
           status: production | production_candidate | ...

       Card validated for shape during load; semantic validation
       (`validate_for_status`) runs in audit_config_catalog.py
       (CONFIG-UX.audit phase), not here.

    Composition path is IDENTICAL between the two shapes — card metadata
    is operator-product concern; runtime mechanics live in model/hardware/
    profile triplet and are unaffected by the card.
    """
    preset = load_preset_def(alias)
    if not preset.has_card():
        _maybe_warn_unannotated(alias)
    return compose_by_ids(
        model_id=preset.model,
        hardware_id=preset.hardware,
        profile_id=preset.profile,
        runtime=preset.runtime,
    )


# Per-alias parse cache for preset defs, keyed by alias and validated against the
# resolved file's mtime. The GUI's overview/presets/catalog endpoints each parse
# the whole preset set per request (~tens of ms); this serves repeated reads from
# memory while a file edit (new mtime) is picked up live. Preset defs are treated
# as read-only, so sharing the cached object is safe. Bounded by preset count.
_PRESET_DEF_CACHE: dict[str, tuple[int, "PresetDef"]] = {}


def load_preset_def(alias: str) -> PresetDef:
    """Load `presets/<alias>.yaml` as a typed PresetDef (CONFIG-UX.1).

    Used by tools that need the parsed card (CLI surface in CONFIG-UX.3,
    audit gates in CONFIG-UX.audit). For composition use `load_alias`.

    Legacy 3-pointer presets load as PresetDef with `card=None`. Caller
    can call `synth_card_for_legacy(alias)` to materialise a placeholder
    card if a typed surface is required downstream.

    Cached per-alias and invalidated when the resolved YAML's mtime changes.
    """
    path = _preset_path(alias)
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        mtime = -1
    cached = _PRESET_DEF_CACHE.get(alias)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    data = _yaml_safe_load(path)
    preset = parse_preset_yaml(alias, data)
    # Validate shape only — semantic checks deferred to audit gate.
    preset.validate()
    if not preset.model or not preset.hardware:
        raise SchemaError(
            f"preset {alias!r}: `model:` and `hardware:` are required pointers"
        )
    _PRESET_DEF_CACHE[alias] = (mtime, preset)
    return preset


def compose_by_ids(
    model_id: str,
    hardware_id: str,
    profile_id: Optional[str] = None,
    runtime: Optional[str] = None,
) -> ModelConfig:
    """Load each layer by id and produce the composed V1 ModelConfig."""
    model = load_model(model_id)
    hardware = load_hardware(hardware_id)
    profile = load_profile(profile_id) if profile_id else None
    return compose(model, hardware, profile, runtime_override=runtime)
