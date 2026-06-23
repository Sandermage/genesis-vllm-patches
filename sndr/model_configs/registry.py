# SPDX-License-Identifier: Apache-2.0
"""Registry — discover ModelConfigs from builtin/, community/, user/ dirs.

Builtin (`sndr/model_configs/builtin/*.yaml`) ships with
the patcher. Community (`community/*.yaml`) is PR'd and reviewed.
User (`~/.sndr/model_configs/*.yaml` or `$SNDR_MODEL_CONFIG_DIR`) is
operator-local, never committed. Legacy `GENESIS_*` paths remain as
fallback aliases during the migration window.

`get(key)` searches all three in priority order: user > community >
builtin. This lets operators override builtin configs without forking.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .schema import ModelConfig, load_yaml, SchemaError


_BUILTIN_DIR = Path(__file__).parent / "builtin"
_COMMUNITY_DIR = Path(__file__).parent / "community"


def _user_dir() -> Path:
    """Resolve the operator-local config dir.

    P1-5 (audit 2026-05-08): canonical env is `SNDR_MODEL_CONFIG_DIR`,
    legacy alias `GENESIS_MODEL_CONFIG_DIR` honored. Default root is
    `~/.sndr/model_configs/` with fallback to `~/.genesis/model_configs/`
    when the legacy dir exists alone.

    Order:
      1. $SNDR_MODEL_CONFIG_DIR (canonical override)
      2. $GENESIS_MODEL_CONFIG_DIR (legacy alias)
      3. $SNDR_HOME/model_configs/  (canonical, derived from operator root)
      4. $GENESIS_HOME/model_configs/  (legacy alias, derived)
      5. ~/.sndr/model_configs/  (canonical default)
      6. ~/.genesis/model_configs/  (legacy default — only if it exists)
    """
    from sndr.engines.vllm.locations.project_paths import model_configs_user_dir
    return model_configs_user_dir()


def _load_dir(d: Path, source_label: str) -> dict[str, ModelConfig]:
    """Load every *.yaml from `d`, indexed by config.key.

    Keys collisions within the same dir → SchemaError. Corrupt YAMLs
    → logged + skipped (not raised) so a bad community config doesn't
    blow up the whole registry.
    """
    out: dict[str, ModelConfig] = {}
    if not d.is_dir():
        return out
    for yaml_path in sorted(d.glob("*.yaml")):
        try:
            cfg = load_yaml(yaml_path.read_text())
        except SchemaError as e:
            import logging
            logging.getLogger("genesis.model_configs").warning(
                "[model_configs] skipping %s/%s — schema error: %s",
                source_label, yaml_path.name, e,
            )
            continue
        if cfg.key in out:
            raise SchemaError(
                f"duplicate model_config key '{cfg.key}' in "
                f"{source_label}/ (file: {yaml_path.name})"
            )
        out[cfg.key] = cfg
    return out


def load_all() -> dict[str, ModelConfig]:
    """Return merged registry: user overrides community overrides builtin."""
    builtin = _load_dir(_BUILTIN_DIR, "builtin")
    community = _load_dir(_COMMUNITY_DIR, "community")
    user = _load_dir(_user_dir(), "user")
    # Merge in priority order
    merged: dict[str, ModelConfig] = {}
    for src in (builtin, community, user):
        merged.update(src)
    return merged


def get(key: str) -> Optional[ModelConfig]:
    """Get one config by key, or None if not found in any tier.

    Phase 9 (V1 freeze): when the resolved key comes from the V1
    monolithic top-level `builtin/*.yaml` tier AND a V2 alias with the
    same shape exists (`prod-qwen3.6-35b-balanced` ↔ `a5000-2x-35b-prod`), this function
    emits a single `DeprecationWarning`. The warning is informational
    only — V1 path keeps working through Phase 9 sunset; Phase 10 will
    sunset V1 loader after operator confirmation.
    """
    cfg = load_all().get(key)
    if cfg is not None and source_of(key) == "builtin":
        # Top-level builtin = V1 monolithic preset. V2 layered tiers
        # live under builtin/model/, builtin/hardware/, etc. — those
        # don't surface via load_all() because _load_dir uses *.yaml
        # (not recursive). So a hit on the top-level builtin tier is
        # by definition V1.
        _maybe_warn_v1_deprecation(key)
    return cfg


_V1_DEPRECATION_WARNED: set[str] = set()


def _lookup_v1_bucket(key: str) -> str:
    """Resolve a V1 key to its migration bucket (CONFIG-UX.4.1).

    Reads `_v1_migration_table.json`; defensive default is
    "needs_operator_choice" for keys not in the table (should not
    happen — audit_no_new_v1.py freezes the baseline). Cached
    once-per-process via the module-level `_V1_BUCKET_CACHE`.
    """
    cache = _v1_bucket_cache()
    return cache.get(key, "needs_operator_choice")


_V1_BUCKET_CACHE: dict[str, str] | None = None


def _v1_bucket_cache() -> dict[str, str]:
    """Lazy-load the migration table → key→bucket mapping."""
    global _V1_BUCKET_CACHE
    if _V1_BUCKET_CACHE is not None:
        return _V1_BUCKET_CACHE
    import json
    table_path = Path(__file__).parent / "_v1_migration_table.json"
    try:
        with table_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("entries", {})
        out = {
            k: v.get("bucket", "needs_operator_choice")
            for k, v in entries.items() if isinstance(v, dict)
        }
    except (OSError, ValueError, KeyError):
        # Defensive — degraded mode (empty table) still works; all V1
        # keys resolve to needs_operator_choice (warn at Stage 0/1).
        out = {}
    _V1_BUCKET_CACHE = out
    return out


def _maybe_warn_v1_deprecation(
    key: str,
    *,
    bucket: Optional[str] = None,
    stage: Optional[int] = None,
) -> None:
    """Emit a stage-aware deprecation event per V1 preset key.

    Backwards-compatible signature: positional `key` arg unchanged;
    `bucket` and `stage` are keyword-only with defaults that match
    the prior behavior at Stage 0/1 (warning emission).

    Severity is resolved via `_rollout.effective_severity()`:
      - transparent / needs_operator_choice / deprecated at Stage 0-2:
          DeprecationWarning (current behavior).
      - tombstone (any stage):
          RuntimeError raised with the migration-table rationale.
      - Stage 3+ for non-transparent buckets:
          RuntimeError raised.

    Once-per-key-per-process tracking preserved.

    Set `GENESIS_DISABLE_V1_DEPRECATION_WARNING=1` to silence emitted
    warnings (does NOT silence ERROR severity at Stage 3+).
    """
    from ._rollout import effective_severity, is_disabled
    if key in _V1_DEPRECATION_WARNED:
        return
    _V1_DEPRECATION_WARNED.add(key)

    if bucket is None:
        bucket = _lookup_v1_bucket(key)

    severity = effective_severity(
        bucket=bucket,  # type: ignore[arg-type]
        stage=stage,
    )

    if severity == "info" or (severity == "warn" and is_disabled()):
        # info severity never emits; warn severity silenced by escape hatch.
        return

    msg = (
        f"V1 monolithic preset key {key!r} (bucket={bucket}) is deprecated. "
        f"Prefer a V2 alias under model_configs/builtin/presets/ "
        f"(see `sndr preset list` / `sndr preset recommend`). "
        f"V1 loader stays functional during the Phase 9 freeze window."
    )

    if severity == "error":
        # Stage 3+ for non-transparent buckets, or tombstone at any stage.
        raise RuntimeError(msg)

    # severity == "warn"
    import warnings
    warnings.warn(msg, DeprecationWarning, stacklevel=3)


def list_keys() -> list[str]:
    """Sorted list of all available config keys."""
    return sorted(load_all().keys())


def source_of(key: str) -> Optional[str]:
    """Return 'builtin' / 'community' / 'user' for the tier providing key."""
    if key in _load_dir(_user_dir(), "user"):
        return "user"
    if key in _load_dir(_COMMUNITY_DIR, "community"):
        return "community"
    if key in _load_dir(_BUILTIN_DIR, "builtin"):
        return "builtin"
    return None


def path_for(key: str) -> Optional[Path]:
    """Return the YAML file path that provides `key`, or None.

    Walks user → community → builtin in that priority. Caller can write
    back into that file (e.g. `bench-and-update` updating reference_metrics).
    """
    for d in (_user_dir(), _COMMUNITY_DIR, _BUILTIN_DIR):
        if not d.is_dir():
            continue
        for yaml_path in sorted(d.glob("*.yaml")):
            try:
                cfg = load_yaml(yaml_path.read_text())
            except SchemaError:
                continue
            if cfg.key == key:
                return yaml_path
    return None
