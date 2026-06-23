# SPDX-License-Identifier: Apache-2.0
"""`sndr config keys` — canonical env-key registry (Roadmap §6.7).

Mitigates risk R2 (env-key drift): YAML configs can name env keys that
the apply.py runtime silently ignores. This CLI exposes the canonical
union of every key the codebase actually knows about, so:

  • Operators can grep for a key's origin (which patch introduced it).
  • CI can run `sndr config keys validate <yaml>` to catch typos before
    they ship to a release.

Canonical sources (union):

  1. `dispatcher.registry.PATCH_REGISTRY` — `env_flag` field per patch.
     This is the primary enable/disable surface.
  2. V2 `builtin/model/*.yaml` `patches:` blocks — captures tuning knobs
     like `GENESIS_P67_NUM_KV_SPLITS`, `GENESIS_PN16_TOOL_THINK_BUDGET`
     that aren't primary toggles but live in production model configs.
  3. V1 monolithic `builtin/*.yaml` `genesis_env:` blocks — legacy
     coverage so we don't break operators still on V1 paths during
     the Phase 9 warn-only freeze window.

`subcommands`:

  sndr config keys list [--source <s>] [--json]
  sndr config keys describe <KEY> [--json]
  sndr config keys validate <yaml-file> [--json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import _io


__all__ = [
    "add_argparser",
    "run_list",
    "run_describe",
    "run_validate",
    "load_canonical_registry",
]


def add_argparser(subparsers: Any) -> None:
    """Phase 4 deferred — config-keys is top-level (not under `sndr config`)
    to avoid colliding with the existing `sndr config` (preset browser).
    Naming follows the `config-keys` hyphenation convention from
    `sndr bench-validate` / `sndr bench-compare`."""

    p_list = subparsers.add_parser(
        "config-keys-list",
        help="Enumerate every recognized env key (R2 mitigation, §6.7).",
        description=(
            "Canonical union of PATCH_REGISTRY env_flags + V2 model.patches "
            "keys + V1 genesis_env keys. Use to grep for a key's provenance."
        ),
    )
    p_list.add_argument("--source", default=None,
                        choices=("registry", "v2", "v1"),
                        help="Filter to one canonical source.")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=run_list)

    p_d = subparsers.add_parser(
        "config-keys-describe",
        help="Show origin + provenance for one env key.",
    )
    p_d.add_argument("key", help="Env key to describe (e.g. GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL).")
    p_d.add_argument("--json", action="store_true")
    p_d.set_defaults(func=run_describe)

    p_v = subparsers.add_parser(
        "config-keys-validate",
        help="Walk a YAML file and reject unknown env keys.",
    )
    p_v.add_argument("yaml_file",
                     help="Path to a model_config / V2 model / profile YAML.")
    p_v.add_argument("--json", action="store_true")
    p_v.set_defaults(func=run_validate)


# ─── Canonical-registry builder ────────────────────────────────────────


def _collect_registry_env_flags() -> dict[str, dict[str, str]]:
    """Walk PATCH_REGISTRY and return `{key: {patch_id, family, source}}`."""
    try:
        from sndr.dispatcher.registry import PATCH_REGISTRY
    except ImportError:
        return {}
    out: dict[str, dict[str, str]] = {}
    for patch_id, meta in PATCH_REGISTRY.items():
        flag = meta.get("env_flag")
        # `env_flag_aliases` (2026-06-19): when two patches sharing one engine
        # file are consolidated into one registry entry, the absorbed patch's
        # env flag is retained as a recognized alias so existing builtin YAMLs
        # keep working. Register the alias(es) as canonical config keys too.
        flags = [flag] if flag else []
        flags.extend(meta.get("env_flag_aliases", ()) or ())
        for f in flags:
            out[f] = {
                "source": "registry",
                "patch_id": patch_id,
                "family": meta.get("family", "?"),
                "default_on": str(bool(meta.get("default_on", False))),
                "lifecycle": meta.get("lifecycle", "?"),
            }
    return out


def _collect_v2_keys() -> dict[str, dict[str, str]]:
    """Walk every V2 `builtin/model/*.yaml` patches block."""
    out: dict[str, dict[str, str]] = {}
    try:
        from sndr.model_configs.registry_v2 import (
            list_models, load_model,
        )
    except ImportError:
        return {}
    for model_id in list_models():
        try:
            model = load_model(model_id)
        except Exception:
            continue
        for k in model.patches.keys():
            existing = out.get(k)
            if existing is None:
                out[k] = {"source": "v2", "first_seen_in": model_id}
            # No "double-counting" needed — second hit just confirms the
            # key is used; first_seen_in stays stable for grep-friendliness.
    return out


# Genesis policy / runtime keys that are NOT patch toggles. They live in
# system_env and are consumed by detection/guards/orchestrator. Source-grepped
# from `vllm/sndr_core/{detection,apply,oracle,observability}` 2026-05-13.
# Add entries here when a new policy/runtime knob lands.
_POLICY_KEYS: dict[str, dict[str, str]] = {
    "GENESIS_VLLM_PIN_POLICY": {
        "source": "policy",
        "owner_module": "sndr.engines.vllm.detection.guards",
        "description": "vllm pin enforcement mode (strict|warn|off)",
    },
    # Gemma 4 declarative profile fields (compression_plan,
    # backend_plan) get rendered into env vars by compose() — register
    # them here so audit-v2-env-keys recognises them on resolved
    # presets. These are consumed by integrations/model_compat/gemma4/
    # patches G4_60K (TQ engine config skip-list union) and
    # G4_76 (drafter KV-sharing disable toggle).
    "GENESIS_G4_TQ_FORCE_SKIP_LAYERS": {
        "source": "policy",
        "owner_module": "sndr.engines.vllm.patches.model_compat.gemma4",
        "description": (
            "Comma-separated layer indices forced to native bf16 "
            "(skipped by TQ KV cache). Emitted by compose() from "
            "profile.compression_plan.native_source_layers."
        ),
    },
    "GENESIS_G4_09_CHUNK_SIZE": {
        "source": "policy",
        "owner_module": "sndr.engines.vllm.patches.model_compat.gemma4",
        "description": (
            "Tunable SWA->global prefill chunk size for G4_09 (gemma4 "
            "sliding-window global-prefill chunker). Integer; read by "
            "g4_09_gemma4_swa_global_prefill_chunker._chunk_size() and "
            "clamped to a safe range. Optional override — the patch bakes "
            "a default when unset (audit P0 follow-up tunable)."
        ),
    },
    "SNDR_G4_TQ_FORCE_SKIP_LAYERS": {
        "source": "policy",
        "owner_module": "sndr.engines.vllm.patches.model_compat.gemma4",
        "description": (
            "SNDR_-canonical alias of GENESIS_G4_TQ_FORCE_SKIP_LAYERS "
            "(one-release migration window)."
        ),
    },
    "SNDR_ENABLE_G4_76_DISABLE_DRAFTER_KV_SHARING": {
        "source": "policy",
        "owner_module": "sndr.engines.vllm.patches.model_compat.gemma4",
        "description": (
            "SNDR_-canonical alias of "
            "GENESIS_ENABLE_G4_76_DISABLE_DRAFTER_KV_SHARING. "
            "0 = drafter shares physical KV with target (β'-A K=4 "
            "validated path), 1 = drafter uses logical/separate KV."
        ),
    },
    "GENESIS_P67_NUM_WARPS": {
        "source": "policy",
        "owner_module": "sndr.engines.vllm.kernels_legacy.p67_multi_query_kernel",
        "description": (
            "Caps the P67 TQ multi-query stage-2 decode kernel num_warps "
            "(default 8 for SM>=8, else 4). The a5000-2x hardware YAML caps "
            "it to 4 to avoid SMEM spill-to-local on A5000 (SM8.6) where "
            "8 warps + the split-M accumulator can exceed the 100KB/SM "
            "budget. Unset = upstream default."
        ),
    },
    "GENESIS_ENFORCE_VERSION_RANGE": {
        "source": "policy",
        "owner_module": "sndr.dispatcher.decision",
        "description": (
            "Gates per-patch vllm_version_range enforcement at apply time "
            "(default OFF). Set to 1 on dev491 so out-of-range wraps "
            "correctly SKIP instead of corrupting the engine-native parser."
        ),
    },
}


def _collect_policy_keys() -> dict[str, dict[str, str]]:
    return {k: dict(v) for k, v in _POLICY_KEYS.items()}


def _collect_v1_keys() -> dict[str, dict[str, str]]:
    """Walk every V1 monolithic `builtin/*.yaml` genesis_env block."""
    out: dict[str, dict[str, str]] = {}
    try:
        from sndr.model_configs.registry import (
            _BUILTIN_DIR,
        )
    except ImportError:
        return {}
    import yaml
    if not _BUILTIN_DIR.is_dir():
        return out
    for fp in sorted(_BUILTIN_DIR.glob("*.yaml")):
        try:
            data = yaml.safe_load(fp.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        env = data.get("genesis_env", {}) or {}
        if not isinstance(env, dict):
            continue
        for k in env.keys():
            if k not in out:
                out[k] = {"source": "v1", "first_seen_in": fp.stem}
    return out


def load_canonical_registry() -> dict[str, dict[str, str]]:
    """Merge all canonical sources. Last assignment wins, so explicit
    PATCH_REGISTRY metadata keeps provenance even if a key also appears
    in V2 / V1 yaml or the policy allowlist."""
    policy = _collect_policy_keys()
    v1 = _collect_v1_keys()
    v2 = _collect_v2_keys()
    registry = _collect_registry_env_flags()
    merged: dict[str, dict[str, str]] = {}
    for src in (v1, v2, policy, registry):  # registry wins on collision
        for k, meta in src.items():
            merged[k] = dict(meta)
    return merged


# ─── list ──────────────────────────────────────────────────────────────


def run_list(opts: argparse.Namespace) -> int:
    canon = load_canonical_registry()
    if opts.source:
        canon = {k: v for k, v in canon.items() if v.get("source") == opts.source}

    rows = sorted(canon.items())
    if opts.json:
        print(json.dumps(
            {"keys": [{"key": k, **meta} for k, meta in rows],
             "count": len(rows)},
            indent=2, sort_keys=True,
        ))
        return 0

    title = "sndr config-keys list"
    if opts.source:
        title += f"  (filter: source={opts.source})"
    print(title)
    print("─" * 70)
    if not rows:
        print("  (no keys match)")
        return 0
    # Group by source for human-readable output.
    by_src: dict[str, list[tuple[str, dict[str, str]]]] = {}
    for k, m in rows:
        by_src.setdefault(m.get("source", "?"), []).append((k, m))
    for src_name in ("registry", "v2", "v1", "?"):
        if src_name not in by_src:
            continue
        section = by_src[src_name]
        print()
        print(f"  ── source={src_name} ({len(section)} keys) ──")
        for k, m in section:
            if src_name == "registry":
                print(f"    {k}")
                print(f"        patch={m['patch_id']}  family={m['family']}  "
                      f"default_on={m['default_on']}  lifecycle={m['lifecycle']}")
            else:
                print(f"    {k}  (first_seen_in={m.get('first_seen_in', '?')})")
    print()
    print(f"  Total: {len(rows)} canonical keys")
    return 0


# ─── describe ──────────────────────────────────────────────────────────


def _suggest_keys(query: str, canon: dict, limit: int = 5) -> list[str]:
    """Token-overlap suggester. Splits both query and candidate keys
    on `_` and ranks by shared-token count. Substring matches still
    weigh in as a tiebreaker. Difflib.SequenceMatcher gives a final
    fallback for keys that have no shared tokens but look similar."""
    import difflib

    q_tokens = {t for t in query.upper().split("_") if t}
    scored: list[tuple[int, float, str]] = []
    for k in canon.keys():
        k_tokens = {t for t in k.upper().split("_") if t}
        overlap = len(q_tokens & k_tokens)
        ratio = difflib.SequenceMatcher(None, query.upper(), k.upper()).ratio()
        if overlap == 0 and ratio < 0.55:
            continue
        scored.append((overlap, ratio, k))
    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
    return [k for _ov, _r, k in scored[:limit]]


def run_describe(opts: argparse.Namespace) -> int:
    canon = load_canonical_registry()
    key = opts.key
    if key not in canon:
        suggestions = _suggest_keys(key, canon)
        if opts.json:
            print(json.dumps(
                {"key": key, "known": False, "suggestions": suggestions},
                indent=2, sort_keys=True,
            ))
        else:
            print(f"sndr config-keys describe {key!r}")
            print("─" * 60)
            print("  ✗ unknown key")
            if suggestions:
                print("  Did you mean:")
                for s in suggestions:
                    print(f"    • {s}")
        return 1

    meta = canon[key]
    if opts.json:
        print(json.dumps({"key": key, "known": True, **meta},
                         indent=2, sort_keys=True))
        return 0

    print(f"sndr config-keys describe '{key}'")
    print("─" * 60)
    print(f"  source: {meta.get('source')}")
    for k, v in sorted(meta.items()):
        if k == "source":
            continue
        print(f"  {k}: {v}")
    return 0


# ─── validate ──────────────────────────────────────────────────────────


def _extract_keys_from_yaml(path: Path) -> list[str]:
    """Pull every env-keylike entry from a YAML file. We look at the
    standard places: `genesis_env`, `system_env`, V2 model `patches:`,
    V2 profile `patches_delta` (enable/disable/override)."""
    import yaml
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as e:
        raise RuntimeError(f"could not parse {path}: {e}") from None
    if not isinstance(data, dict):
        return []
    keys: set[str] = set()

    for top in ("genesis_env", "system_env"):
        block = data.get(top, {}) or {}
        if isinstance(block, dict):
            keys.update(block.keys())

    # V2 model: `patches: {KEY: VALUE}`.
    block = data.get("patches", {}) or {}
    if isinstance(block, dict):
        keys.update(block.keys())

    # V2 profile: `patches_delta.{enable, disable, override}`.
    delta = data.get("patches_delta", {}) or {}
    if isinstance(delta, dict):
        for sub in ("enable", "override"):
            sub_block = delta.get(sub, {}) or {}
            if isinstance(sub_block, dict):
                keys.update(sub_block.keys())
        disable = delta.get("disable", []) or []
        if isinstance(disable, list):
            for v in disable:
                if isinstance(v, str):
                    keys.add(v)

    return sorted(keys)


def run_validate(opts: argparse.Namespace) -> int:
    path = Path(opts.yaml_file)
    if not path.is_file():
        _io.warn(f"file not found: {path}")
        return 2

    try:
        keys = _extract_keys_from_yaml(path)
    except RuntimeError as e:
        _io.warn(str(e))
        return 2

    canon = load_canonical_registry()
    unknown: list[str] = [
        k for k in keys
        if k not in canon
        # Don't flag standard non-Genesis env (allow PYTORCH_*, VLLM_*, etc.).
        and (k.startswith("GENESIS_") or k.startswith("SNDR_"))
    ]
    non_genesis_keys = [k for k in keys if not (k.startswith("GENESIS_") or k.startswith("SNDR_"))]

    if opts.json:
        print(json.dumps({
            "file": str(path),
            "total_keys": len(keys),
            "genesis_keys": len(keys) - len(non_genesis_keys),
            "non_genesis_keys": len(non_genesis_keys),
            "unknown_keys": unknown,
            "passed": not unknown,
        }, indent=2, sort_keys=True))
        return 0 if not unknown else 1

    print(f"sndr config-keys validate {path}")
    print("─" * 60)
    print(f"  total keys:        {len(keys)}")
    print(f"  Genesis/SNDR keys: {len(keys) - len(non_genesis_keys)}")
    print(f"  unknown keys:      {len(unknown)}")
    print()
    if unknown:
        print("  ✗ unknown keys (Genesis/SNDR prefix, not in canonical registry):")
        for k in unknown:
            print(f"    {k}")
        print()
        print("  Hint: run `sndr config-keys describe <KEY>` for suggestions.")
        return 1
    print("  ✓ all Genesis/SNDR keys present in canonical registry")
    return 0
