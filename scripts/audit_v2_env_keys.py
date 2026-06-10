#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§4.2 V2 cross-layer env-key consistency gate — `make audit-v2-env-keys`.

Walks every Genesis/SNDR env key referenced across the V2 config layers
(model + profile + resolved-alias) and verifies each appears in the
canonical env-key registry (`vllm.sndr_core.cli.config_keys.load_canonical_registry`).

What this catches that the existing gates don't:

  • `audit-configs`        — confirms each preset alias COMPOSES, but
    not that the composed `genesis_env` only contains canonical keys.
  • `sndr config-keys validate <file>` — single-file, no CI sweep.
  • E22 `audit-launch-coverage` — system_env only, not patches matrix.

A typo in `profile.patches_delta.enable.GENESIS_ENABLE_P9999_TYPO: '1'`
composes cleanly (it's just a dict entry), survives `audit-configs`,
and only fails at runtime when the orchestrator can't resolve P9999.
This gate surfaces it at PR time.

The audit walks three layers:

  1. **model** layer — `model_configs/builtin/model/*.yaml`'s `patches:` matrix
  2. **profile** layer — `model_configs/builtin/profile/*.yaml`'s
     `patches_delta.{enable, disable, override}` blocks
  3. **resolved alias** layer — for each preset under `presets/*.yaml`,
     `load_alias()` produces a composed `genesis_env`; we walk its keys.

Only Genesis/SNDR-prefixed keys are checked (non-Genesis keys like
`PYTORCH_*` or `VLLM_*` are operator-tunable and live outside the
canonical registry by design).

Exit codes:
  0 — every Genesis/SNDR key across every layer is in the canonical registry
  1 — at least one layer has unknown keys
  2 — internal error (composer failure, YAML parse, etc.)
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "sndr" / "model_configs" / "builtin" / "model"
PROFILE_DIR = REPO_ROOT / "sndr" / "model_configs" / "builtin" / "profile"
PRESETS_DIR = REPO_ROOT / "sndr" / "model_configs" / "builtin" / "presets"


# Ensure repo root is importable when run via `make` from elsewhere.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _is_genesis_key(k: str) -> bool:
    return k.startswith("GENESIS_") or k.startswith("SNDR_")


# ─── Result types ─────────────────────────────────────────────────────


@dataclass
class LayerEntry:
    """One file or alias's audit result."""
    layer: str             # "model" | "profile" | "resolved-alias"
    label: str             # filename or alias id
    total_keys: int
    genesis_keys: int
    unknown_keys: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def passed(self) -> bool:
        return not self.error and not self.unknown_keys


# ─── Per-layer walkers ────────────────────────────────────────────────


def _walk_model_layer(canon: dict) -> list[LayerEntry]:
    from sndr.cli.legacy.config_keys import _extract_keys_from_yaml
    out: list[LayerEntry] = []
    if not MODEL_DIR.is_dir():
        return out
    for yp in sorted(MODEL_DIR.glob("*.yaml")):
        try:
            keys = _extract_keys_from_yaml(yp)
        except RuntimeError as e:
            out.append(LayerEntry("model", yp.name, 0, 0, error=str(e)))
            continue
        genesis = [k for k in keys if _is_genesis_key(k)]
        unknown = [k for k in genesis if k not in canon]
        out.append(LayerEntry(
            layer="model",
            label=yp.name,
            total_keys=len(keys),
            genesis_keys=len(genesis),
            unknown_keys=sorted(unknown),
        ))
    return out


def _walk_profile_layer(canon: dict) -> list[LayerEntry]:
    from sndr.cli.legacy.config_keys import _extract_keys_from_yaml
    out: list[LayerEntry] = []
    if not PROFILE_DIR.is_dir():
        return out
    for yp in sorted(PROFILE_DIR.glob("*.yaml")):
        try:
            keys = _extract_keys_from_yaml(yp)
        except RuntimeError as e:
            out.append(LayerEntry("profile", yp.name, 0, 0, error=str(e)))
            continue
        genesis = [k for k in keys if _is_genesis_key(k)]
        unknown = [k for k in genesis if k not in canon]
        out.append(LayerEntry(
            layer="profile",
            label=yp.name,
            total_keys=len(keys),
            genesis_keys=len(genesis),
            unknown_keys=sorted(unknown),
        ))
    return out


def _walk_resolved_aliases(canon: dict) -> list[LayerEntry]:
    """For each preset, resolve via `load_alias()` and walk the composed
    `genesis_env` map. Catches typos that survive composition (e.g. a
    profile delta `override:` that adds a non-canonical key)."""
    out: list[LayerEntry] = []
    try:
        from sndr.model_configs.registry_v2 import load_alias
    except ImportError as e:
        out.append(LayerEntry(
            "resolved-alias", "<import>", 0, 0,
            error=f"registry_v2 not importable: {e}",
        ))
        return out

    if not PRESETS_DIR.is_dir():
        return out
    for ap in sorted(PRESETS_DIR.glob("*.yaml")):
        alias = ap.stem
        try:
            cfg = load_alias(alias)
        except Exception as e:
            out.append(LayerEntry(
                "resolved-alias", alias, 0, 0,
                error=f"compose failed: {type(e).__name__}: {e}",
            ))
            continue
        keys = list(cfg.genesis_env.keys())
        genesis = [k for k in keys if _is_genesis_key(k)]
        unknown = [k for k in genesis if k not in canon]
        out.append(LayerEntry(
            layer="resolved-alias",
            label=alias,
            total_keys=len(keys),
            genesis_keys=len(genesis),
            unknown_keys=sorted(unknown),
        ))
    return out


def audit_v2_env_keys() -> list[LayerEntry]:
    """Run all three layer walkers + return combined per-entry results."""
    try:
        from sndr.cli.legacy.config_keys import load_canonical_registry
    except ImportError as e:
        return [LayerEntry(
            "<bootstrap>", "load_canonical_registry", 0, 0,
            error=f"config_keys module not importable: {e}",
        )]
    canon = load_canonical_registry()
    return (
        _walk_model_layer(canon)
        + _walk_profile_layer(canon)
        + _walk_resolved_aliases(canon)
    )


# ─── Renderers ────────────────────────────────────────────────────────


def _render_text(results: list[LayerEntry]) -> str:
    lines = []
    lines.append(f"audit-v2-env-keys: {len(results)} entry(ies) across 3 layers")
    lines.append("─" * 70)

    by_layer: dict[str, list[LayerEntry]] = {}
    for r in results:
        by_layer.setdefault(r.layer, []).append(r)

    for layer in ("model", "profile", "resolved-alias"):
        rs = by_layer.get(layer, [])
        if not rs:
            continue
        lines.append(f"  ── {layer} layer ({len(rs)} entries) ──")
        for r in rs:
            sym = "✓" if r.passed else "✗"
            tail = ""
            if r.error:
                tail = f"  ! {r.error}"
            elif r.unknown_keys:
                tail = f"  ✗ {len(r.unknown_keys)} unknown key(s)"
            lines.append(
                f"    {sym} {r.label:38s}  "
                f"({r.genesis_keys} Genesis/SNDR keys){tail}"
            )
            for u in r.unknown_keys[:5]:
                lines.append(f"          ⚠ {u}")
            if len(r.unknown_keys) > 5:
                lines.append(f"          ... ({len(r.unknown_keys) - 5} more)")

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total_unknown = sum(len(r.unknown_keys) for r in results)
    lines.append("─" * 70)
    lines.append(
        f"  {passed}/{len(results)} entries clean  "
        f"({total_unknown} unknown Genesis/SNDR key(s) total across {failed} entries)"
    )
    if failed:
        lines.append("")
        lines.append(
            "  ✗ Fix: either add the key to the canonical registry "
            "(`sndr config-keys describe <KEY>` for hints) or correct "
            "the typo in the YAML."
        )
    return "\n".join(lines)


def _render_json(results: list[LayerEntry]) -> str:
    payload = {
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "by_layer": {
            layer: {
                "entries": sum(1 for r in results if r.layer == layer),
                "passed":  sum(1 for r in results if r.layer == layer and r.passed),
                "failed":  sum(1 for r in results if r.layer == layer and not r.passed),
                "unknown_total": sum(
                    len(r.unknown_keys) for r in results if r.layer == layer
                ),
            }
            for layer in ("model", "profile", "resolved-alias")
        },
        "entries": [
            {
                "layer": r.layer,
                "label": r.label,
                "total_keys": r.total_keys,
                "genesis_keys": r.genesis_keys,
                "unknown_keys": r.unknown_keys,
                "passed": r.passed,
                "error": r.error or None,
            }
            for r in results
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true",
                    help="Machine-readable JSON output.")
    ap.add_argument("--layer", default=None,
                    choices=["model", "profile", "resolved-alias"],
                    help="Limit to one layer (debugging).")
    args = ap.parse_args()

    try:
        results = audit_v2_env_keys()
    except Exception:
        traceback.print_exc()
        return 2

    if args.layer:
        results = [r for r in results if r.layer == args.layer]

    if args.json:
        print(_render_json(results))
    else:
        print(_render_text(results))

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
