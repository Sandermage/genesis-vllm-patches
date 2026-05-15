#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 7 release gate — `make audit-configs`.

Walks every preset alias under `model_configs/builtin/presets/` and
verifies that the (model, hardware, profile) triplet it references
composes cleanly into a V1 `ModelConfig`. Compose drift = release blocked.

This is the byte-equivalence safety net that catches:

  - A profile that references a non-existent parent_model id.
  - A hardware id renamed without updating the preset that pointed at it.
  - A composer regression that breaks one alias while others stay green.
  - YAML edit that flipped a field type and silently broke the resolver.

Exit code:
  0 — every preset composes.
  1 — at least one preset failed to compose (details in stdout).
  2 — internal error (registry helpers unavailable, etc).
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

# Ensure the repo root is on sys.path so `vllm.sndr_core.*` resolves
# whether the script is invoked via `make audit-configs` (cwd=repo) or
# directly from elsewhere. Idempotent.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _alias_ids() -> list[str]:
    """Enumerate every preset alias YAML under builtin/presets/."""
    presets_dir = (
        REPO_ROOT
        / "vllm" / "sndr_core" / "model_configs" / "builtin" / "presets"
    )
    if not presets_dir.is_dir():
        return []
    return sorted(p.stem for p in presets_dir.glob("*.yaml")
                  if p.is_file() and not p.stem.startswith("_"))


def _verify_alias(alias_id: str) -> tuple[bool, str]:
    """Try to compose one alias. Return (ok, summary_line)."""
    try:
        from vllm.sndr_core.model_configs.registry_v2 import load_alias
    except ImportError as e:
        return False, f"registry_v2 not importable: {e}"
    try:
        cfg = load_alias(alias_id)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    # Lightweight invariant checks beyond a successful load.
    if not cfg.key:
        return False, "composed config has empty key"
    # V2 compose separator: was `__` (legacy), changed to `--` 2026-05-14
    # to satisfy V1 ModelConfig.key kebab-case regex. Accept both forms
    # for backward-compat during transition.
    if "--" not in cfg.key and "__" not in cfg.key:
        return False, f"composed key {cfg.key!r} missing -- (or legacy __) separator"
    if cfg.hardware.n_gpus < 1:
        return False, f"hardware.n_gpus={cfg.hardware.n_gpus} (invalid)"
    if cfg.max_model_len < 1:
        return False, f"max_model_len={cfg.max_model_len} (invalid)"
    return True, (
        f"ctx={cfg.max_model_len} seqs={cfg.max_num_seqs} "
        f"patches={len(cfg.genesis_env)} gpus={cfg.hardware.n_gpus}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true",
                    help="Emit JSON summary.")
    args = ap.parse_args()

    alias_ids = _alias_ids()
    if not alias_ids:
        sys.stderr.write(
            "audit-configs: no presets found under "
            "vllm/sndr_core/model_configs/builtin/presets/\n"
        )
        return 2

    results = []
    failures = 0
    for alias in alias_ids:
        ok, summary = _verify_alias(alias)
        results.append({"alias": alias, "ok": ok, "summary": summary})
        if not ok:
            failures += 1

    if args.json:
        print(json.dumps(
            {"total": len(alias_ids), "failures": failures, "results": results},
            indent=2, sort_keys=True,
        ))
    else:
        print(f"audit-configs: {len(alias_ids)} presets")
        print("─" * 70)
        for r in results:
            sym = "✓" if r["ok"] else "✗"
            print(f"  {sym} {r['alias']:38s} {r['summary']}")
        print()
        if failures == 0:
            print(f"  ✓ all {len(alias_ids)} presets compose cleanly")
        else:
            print(f"  ✗ {failures}/{len(alias_ids)} preset(s) failed to compose")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
