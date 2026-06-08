# SPDX-License-Identifier: Apache-2.0
"""GDN contiguity policy — unified documentation + runtime audit.

Phase 6 P3.2 (master plan §3.2) — Genesis has THREE patches that touch the
GDN forward-path `.contiguous()` discipline. Each targets a DIFFERENT
upstream branch and addresses a different problem class:

  PN11  — defensive stride-safety fix in `fix_query_key_value_ordering()`:
          adds .contiguous() after reshape of b + a when np/ng == 1.
          PROBLEM CLASS: stride correctness (vllm#41142 backport).
          Affects models where num_v_heads == num_k_heads.

  PN54  — redundancy cleanup in TWO sites:
          Sub-A (HIGH impact): removes ssm_state[indices].contiguous().
                Advanced indexing already allocates fresh; FLA's
                @input_guard guarantees downstream contiguity. Saves one
                full ssm_state-shape copy per prefill batch.
          Sub-B (LOW impact): removes .contiguous() on b, a after
                ba.chunk(2, dim=-1) in the LoRA branch (in_proj_qkv
                code path). chunk on last dim returns contiguous halves.
          PROBLEM CLASS: redundancy / VRAM allocator pressure.

  PN50  — kernel-fusion replacement of the Qwen3.5/3.6 contiguous-projection
          else: branch. Replaces a 9-line split/reshape/cat/contiguous chain
          with one Triton kernel `fused_qkvzba_split_reshape_cat_contiguous`.
          As a side effect REMOVES the b.contiguous() and a.contiguous()
          calls in that branch.
          PROBLEM CLASS: kernel-count reduction (SGLang#21019 backport).

Why this module exists
----------------------
1. **Drift surveillance** — the three patches target different upstream
   lines, but a future upstream refactor could merge or rename them.
   `audit_contiguity_state()` below scans the upstream file at boot time
   and reports which contiguity-fix patches are reachable on the current
   pin. Operators can call it via `sndr patches show gdn_contiguity`.

2. **Composability documentation** — PN54-SubB and PN50 BOTH touch
   `.contiguous()` after `chunk(2, dim=-1)` but in MUTUALLY EXCLUSIVE
   branches:
     PN54-SubB: `hasattr(self, "in_proj_qkv")` LoRA branch (Qwen3.5 only).
     PN50:      The non-LoRA `else:` branch (Qwen3.5/3.6).
   Genesis PROD uses neither LoRA nor Qwen3.5, so the two patches don't
   collide. However, if either branch becomes the default in a future
   upstream refactor, the composition would need to be re-audited.

3. **Operator visibility** — without this module, the GDN contiguity
   policy is spread across three docstrings. Centralizing here makes
   it visible to `sndr explain gdn_contiguity` and surfaces the
   composition contract in one place.

Patch composition matrix (current PROD)
----------------------------------------

| Patch | Default | Branch | Site | Composable with |
|---|---|---|---|---|
| PN11  | OFF | `fix_query_key_value_ordering()` | b, a reshape | PN54, PN50 |
| PN54-SubA | OFF | `ssm_state` indexing | ssm_state.contiguous() | PN11, PN50 |
| PN54-SubB | OFF | LoRA `in_proj_qkv` chunk | b, a chunk | PN11, PN50 (mutually exclusive branches) |
| PN50      | OFF | Qwen3.5/3.6 else branch | full kernel fusion | PN11, PN54 |

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa
Status: v11.2.0+ P3.2 consolidation surface (documentation + audit)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger("genesis.gdn.contiguity_policy")


__all__ = [
    "CONTIGUITY_PATCH_IDS",
    "audit_contiguity_state",
    "describe_policy",
    "verify_patch_composition",
]


CONTIGUITY_PATCH_IDS = ("PN11", "PN54", "PN50")
"""All patches participating in the unified GDN contiguity policy."""


_POLICY_SUMMARY = """\
GDN contiguity policy (Phase 6 P3.2)

The three patches PN11 / PN54 / PN50 share a common goal: ensure the GDN
forward path neither (a) hits a stride-mismatch crash nor (b) allocates
redundant .contiguous() copies that pressure the VRAM allocator. They
target THREE DISTINCT upstream lines and compose freely on Genesis PROD
(neither LoRA nor Qwen3.5; no branch overlap in practice).

Defaults: all three are default_on=False. Enable per workload:
  GENESIS_ENABLE_PN11_GDN_AB_CONTIGUOUS=1         # stride safety (np/ng=1)
  GENESIS_ENABLE_PN54_GDN_CONTIGUOUS_DEDUP=1      # ssm_state + LoRA chunk
  GENESIS_ENABLE_PN50_GDN_FUSED_PROJ=1            # Qwen3.5/3.6 kernel fusion

To diagnose which are currently reachable on the live pin, call:
  python -c 'from sndr.engines.vllm.patches.attention.gdn._contiguity_policy import audit_contiguity_state; print(audit_contiguity_state())'
"""


def describe_policy() -> str:
    """Return the human-readable policy summary."""
    return _POLICY_SUMMARY


def verify_patch_composition() -> dict:
    """Report whether the 3 contiguity patches are mutually composable
    on the current registry. Returns:

      {
        "composable": bool,
        "patches": {
          "PN11": {"lifecycle": str, "default_on": bool, "env_enabled": bool},
          ...
        },
        "conflicts": [...],
      }
    """
    try:
        from sndr.dispatcher.registry import PATCH_REGISTRY
    except ImportError:
        # Registry not importable (vllm not in env) — return minimal info
        return {
            "composable": True,
            "patches": {pid: {"registry_present": False} for pid in CONTIGUITY_PATCH_IDS},
            "conflicts": [],
            "warning": "PATCH_REGISTRY not importable; cannot audit composition",
        }

    patches: dict = {}
    conflicts: list[str] = []
    for pid in CONTIGUITY_PATCH_IDS:
        entry = PATCH_REGISTRY.get(pid)
        if entry is None:
            patches[pid] = {"registry_present": False}
            conflicts.append(f"{pid} not in PATCH_REGISTRY")
            continue
        env_flag = entry.get("env_flag")
        env_enabled = (
            env_flag is not None
            and os.environ.get(env_flag, "").strip() in ("1", "true", "True")
        )
        patches[pid] = {
            "registry_present": True,
            "lifecycle": entry.get("lifecycle"),
            "default_on": entry.get("default_on", False),
            "env_flag": env_flag,
            "env_enabled": env_enabled,
            "family": entry.get("family"),
        }
        # Check conflicts_with
        cw = entry.get("conflicts_with", [])
        for other in CONTIGUITY_PATCH_IDS:
            if other != pid and other in cw:
                conflicts.append(f"{pid}.conflicts_with contains {other}")

    return {
        "composable": not conflicts,
        "patches": patches,
        "conflicts": conflicts,
    }


def audit_contiguity_state(verbose: bool = False) -> dict:
    """Runtime audit — report contiguity policy state + which patches
    are reachable on the current upstream file at this pin.

    Returns dict with `composition` (from verify_patch_composition) +
    `reachability` (best-effort grep of upstream gdn_linear_attn.py for
    each patch's anchor signature) + `summary`.

    Best-effort: if the upstream file isn't resolvable from the current
    process (e.g. vllm not installed), reachability reports unknown.
    """
    composition = verify_patch_composition()

    reachability: dict = {}
    try:
        # Resolve upstream gdn_linear_attn.py (may have moved in
        # post-v11.x upstream refactors per PN79 v2 scout)
        import importlib
        import importlib.util
        spec = importlib.util.find_spec("vllm")
        if spec is None or spec.origin is None:
            raise ImportError("vllm not importable")
        vllm_root = os.path.dirname(spec.origin)

        # Try canonical locations in priority order
        candidates = [
            os.path.join(
                vllm_root, "model_executor", "layers", "fla", "ops",
                "gdn_linear_attn.py",
            ),
            # Post-PN79-v2 upstream split (model-specific files)
            os.path.join(
                vllm_root, "model_executor", "models", "fla",
                "qwen_gdn_linear_attn.py",
            ),
        ]
        upstream_file: Optional[str] = None
        for c in candidates:
            if os.path.isfile(c):
                upstream_file = c
                break

        if upstream_file is None:
            reachability["error"] = "no upstream gdn_linear_attn.py found"
        else:
            content = ""
            try:
                with open(upstream_file, encoding="utf-8") as f:
                    content = f.read()
            except OSError as e:
                reachability["error"] = f"read failed: {e}"

            if content:
                reachability["upstream_file"] = upstream_file
                # PN11 anchor signature: fix_query_key_value_ordering
                reachability["PN11_anchor"] = "fix_query_key_value_ordering" in content
                # PN54-SubA anchor: ssm_state advanced-index + .contiguous
                reachability["PN54_SubA_anchor"] = (
                    "ssm_state[" in content and ".contiguous()" in content
                )
                # PN54-SubB anchor: in_proj_qkv LoRA branch
                reachability["PN54_SubB_anchor"] = "in_proj_qkv" in content
                # PN50 anchor: ba.chunk(2, dim=-1)
                reachability["PN50_anchor"] = "ba.chunk" in content or (
                    ".chunk(2, dim=-1)" in content
                )
    except Exception as e:
        reachability["error"] = f"audit failed: {e}"

    composable = composition.get("composable", False)
    reachable = any(
        v for k, v in reachability.items()
        if k.endswith("_anchor") and isinstance(v, bool)
    )
    summary = {
        "composable": composable,
        "any_anchor_reachable": reachable,
        "all_patches_in_registry": all(
            p.get("registry_present", False)
            for p in composition["patches"].values()
        ),
    }

    if verbose:
        log.info("GDN contiguity audit summary: %s", summary)

    return {
        "policy_summary": _POLICY_SUMMARY if verbose else None,
        "composition": composition,
        "reachability": reachability,
        "summary": summary,
    }


def main_cli() -> int:
    """Print a human-readable audit report — used by `sndr explain
    gdn_contiguity` and ad-hoc operator diagnostics."""
    print(_POLICY_SUMMARY)
    print()
    result = audit_contiguity_state(verbose=False)
    print("Composition:")
    for pid, p in result["composition"]["patches"].items():
        print(
            f"  {pid}: registry_present={p.get('registry_present')} "
            f"lifecycle={p.get('lifecycle')} "
            f"default_on={p.get('default_on')} "
            f"env_enabled={p.get('env_enabled')}"
        )
    conflicts = result["composition"].get("conflicts") or []
    if conflicts:
        print(f"Conflicts: {conflicts}")
    print()
    print("Reachability (anchors in upstream):")
    for k, v in result["reachability"].items():
        print(f"  {k}: {v}")
    print()
    print(f"Summary: {result['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
