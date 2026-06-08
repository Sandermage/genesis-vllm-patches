# SPDX-License-Identifier: Apache-2.0
"""TurboQuant workspace policy — unified documentation + runtime audit.

Phase 6 P3.2 (master plan §3.2) — Genesis carries FOUR patches in the
TurboQuant workspace area (`vllm/v1/worker/workspace.py` +
`vllm/v1/attention/backends/turboquant_attn.py`). They target distinct
upstream code paths but share a common semantic surface: "acquire +
slice + grow + fallback on locked-undersized workspace".

The four patches
----------------

  P98   — TQ WorkspaceManager revert (PROD-default surface):
          Text-patches `turboquant_attn.py` to bypass the upstream
          WorkspaceManager indirection added in vllm#40941 (which
          regressed TPS 17% on Ampere small-batch). Falls back to the
          OLD per-layer `getattr(layer, "_tq_mid_o_buf", ...)` pattern.
          PROBLEM CLASS: perf regression revert (workspace manager
          indirection too expensive on Ampere).
          INSERTION POINT: turboquant_attn.py decode + prefill paths.

  P99   — WorkspaceManager memoization:
          Adds an `_genesis_p99_cache` dict keyed by
          (shapes_and_dtypes, ubatch_id, workspace_ptr) on the
          WorkspaceManager instance; subsequent calls return a copy of
          the cached list (tensors are views). Saves the list-comp
          overhead of `get_simultaneous()` after the first call per
          unique key.
          PROBLEM CLASS: redundant work elimination.
          INSERTION POINT: WorkspaceManager.get_simultaneous() body.

  PN118 — TQ workspace graceful-fallback:
          Backport of vllm#42551 (jasonboukheir). Adds
          `try_get_simultaneous()` that returns None when growth would
          be needed on a LOCKED workspace; caller falls back to
          torch.empty (graceful degradation). Also adds `reserve()`
          for pre-sizing all ubatch slots.
          PROBLEM CLASS: missing-feature backport (locked-workspace
          graceful path).
          INSERTION POINT: WorkspaceManager class + turboquant_attn.py
          callsites.

  SNDR_WORKSPACE_001 — Workspace grow-after-lock:
          Genesis-original. Replaces the AssertionError on
          locked-workspace-growth path with warn + allow-grow. PROD has
          observed this exact crash class on long-context tool-call
          warmup; the warn-and-continue path keeps the request alive
          while logging the diagnostic.
          PROBLEM CLASS: defensive Genesis-original (closes a crash
          class not in any upstream PR).
          INSERTION POINT: WorkspaceManager lock guard.

Composition contract
--------------------

| Patch | Default | Target file | Compose with | Notes |
|---|---|---|---|---|
| P98               | OFF | turboquant_attn.py | P99, PN118 | reverts WorkspaceManager hot-path; P99's memo cache stays valid |
| P99               | OFF | workspace.py       | P98, PN118 | memoize wins on cache-hit; P98 makes it irrelevant for decode |
| PN118             | ON  | both               | P98, P99   | PROD-default; graceful fallback when locked-undersized |
| SNDR_WORKSPACE_001 | OFF | workspace.py       | PN118      | defensive — kicks in BEFORE PN118 graceful-fallback path on actual grow |

PN118 is the ONLY patch in this group with `default_on=True`. The others
are operator opt-in.

Why this module exists
----------------------

1. **Composition documentation** — without this file, the workspace
   policy is spread across 1030 LOC across 4 patches. Centralizing
   here surfaces:
     - the composition contract (PN118 + P99 + P98 all valid together)
     - the mutually-exclusive insertion points (none collide)
     - the default-on vs opt-in matrix

2. **Drift surveillance** — provides `audit_workspace_state()` to
   report reachability of WorkspaceManager + turboquant_attn classes
   on the current pin. Useful when upstream refactors move these.

3. **Operator visibility** — `sndr patches show workspace_policy`
   calls into this module to surface a one-line view of all 4 patches.

4. **NOT a refactor of the patch LOGIC.** Each patch's apply()
   function is unchanged. This module is documentation + audit only.
   A future v12.x+ might extend `TurboQuantBufferManager` (already a
   partial facade for K/V dequant buffers) with workspace-aware
   methods that wrap workspace.py logic; that refactor is deferred
   until rig bench confirms byte-equivalence under cudagraph capture
   (the workspace area is the hottest path in TurboQuant decode).

Future consolidation candidate (v12.x+)
---------------------------------------

A unified `WorkspaceFacade` class extending `TurboQuantBufferManager`
could host:

    def acquire_decode_workspace(B, Hq, S, D, query_dtype, query_device):
        '''P98+P99 unified: revert-or-cache decode buffers.'''
        ...

    def try_acquire_decode_workspace_locked(...):
        '''PN118 graceful fallback wrap.'''
        ...

    def acquire_prefill_workspace(buf_shape, device):
        '''P98 dequant revert logic.'''
        ...

    def check_workspace_lock_growth(...):
        '''SNDR_WORKSPACE_001 lock-guard decision.'''
        ...

Estimated saving: ~600 LOC across the 4 patches (mostly TextPatcher
anchor boilerplate that would become method bodies). RISK: this is
the hottest path in TurboQuant decode. Migration requires:
  (a) Byte-equivalent verification via unit tests (achievable)
  (b) Full rig bench on PROD presets (35B + 27B multiconc)
  (c) CUDA-graph capture safety — pointer stability across replays

Deferred to v12.0.0 release scope.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa
Status: v11.2.0+ P3.2 documentation + audit (no behavior change)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger("genesis.turboquant.workspace_policy")


__all__ = [
    "WORKSPACE_POLICY_PATCH_IDS",
    "audit_workspace_state",
    "describe_policy",
    "verify_patch_composition",
]


WORKSPACE_POLICY_PATCH_IDS = ("P98", "P99", "PN118", "SNDR_WORKSPACE_001")
"""All patches in the unified TurboQuant workspace policy."""


_POLICY_SUMMARY = """\
TurboQuant workspace policy (Phase 6 P3.2)

Genesis carries 4 patches in the TurboQuant workspace area. They target
distinct upstream code paths and compose freely:

  P98               — revert vllm#40941 WorkspaceManager indirection
                      (Ampere TPS regression workaround)
  P99               — memoize WorkspaceManager.get_simultaneous() output
  PN118             — graceful fallback when workspace locked-undersized
                      (default_on=True — PROD baseline)
  SNDR_WORKSPACE_001 — Genesis-original: warn-and-grow on lock conflict
                      (closes the AssertionError crash class)

PN118 is the only one default_on. The others are operator opt-in. They
do NOT need a runtime facade object on v11.2.0+ Genesis — composition
is operationally orthogonal (different insertion points). A future
v12.x+ may consolidate into a WorkspaceFacade class extending
TurboQuantBufferManager.

To diagnose live state on a running container:
  python -c 'from sndr.engines.vllm.patches.attention.turboquant._workspace_policy import audit_workspace_state; import json; print(json.dumps(audit_workspace_state(), indent=2, default=str))'
"""


def describe_policy() -> str:
    return _POLICY_SUMMARY


def verify_patch_composition() -> dict:
    """Verify all 4 workspace patches are present, composable, and
    correctly default_on configured (PN118 should be the only default_on)."""
    try:
        from sndr.dispatcher.registry import PATCH_REGISTRY
    except ImportError:
        return {
            "composable": True,
            "patches": {
                pid: {"registry_present": False}
                for pid in WORKSPACE_POLICY_PATCH_IDS
            },
            "conflicts": [],
            "warning": "PATCH_REGISTRY not importable; audit limited",
        }

    patches: dict = {}
    conflicts: list[str] = []
    default_on_count = 0
    for pid in WORKSPACE_POLICY_PATCH_IDS:
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
        is_default_on = entry.get("default_on", False)
        if is_default_on:
            default_on_count += 1
        patches[pid] = {
            "registry_present": True,
            "lifecycle": entry.get("lifecycle"),
            "default_on": is_default_on,
            "env_flag": env_flag,
            "env_enabled": env_enabled,
            "family": entry.get("family"),
            "upstream_pr": entry.get("upstream_pr"),
        }
        cw = entry.get("conflicts_with") or []
        for other in WORKSPACE_POLICY_PATCH_IDS:
            if other != pid and other in cw:
                conflicts.append(f"{pid}.conflicts_with contains {other}")

    return {
        "composable": not conflicts,
        "patches": patches,
        "conflicts": conflicts,
        "default_on_count": default_on_count,
    }


def audit_workspace_state(verbose: bool = False) -> dict:
    """Full runtime audit — composition + reachability of upstream
    workspace classes."""
    composition = verify_patch_composition()

    reachability: dict = {
        "workspace_manager_class": False,
        "turboquant_attn_module": False,
    }
    try:
        from vllm.v1.worker.workspace import WorkspaceManager  # noqa: F401
        reachability["workspace_manager_class"] = True
    except Exception as e:
        reachability["workspace_manager_error"] = str(e)
    try:
        import vllm.v1.attention.backends.turboquant_attn  # noqa: F401
        reachability["turboquant_attn_module"] = True
    except Exception as e:
        reachability["turboquant_attn_error"] = str(e)

    any_enabled = any(
        p.get("env_enabled", False)
        for p in composition["patches"].values()
    ) or any(
        p.get("default_on", False)
        for p in composition["patches"].values()
    )

    summary = {
        "composable": composition.get("composable", False),
        "any_patch_active": any_enabled,
        "all_patches_in_registry": all(
            p.get("registry_present", False)
            for p in composition["patches"].values()
        ),
        # Operational invariant: PN118 is the only default_on in PROD;
        # if this changes, operators should re-bench.
        "default_on_count": composition.get("default_on_count", 0),
    }

    if verbose:
        log.info("Workspace policy audit summary: %s", summary)

    return {
        "policy_summary": _POLICY_SUMMARY if verbose else None,
        "composition": composition,
        "reachability": reachability,
        "summary": summary,
    }


def main_cli() -> int:
    print(_POLICY_SUMMARY)
    print()
    result = audit_workspace_state(verbose=False)
    print("Composition:")
    for pid, p in result["composition"]["patches"].items():
        print(
            f"  {pid}: lifecycle={p.get('lifecycle')} "
            f"default_on={p.get('default_on')} "
            f"env_enabled={p.get('env_enabled')}"
        )
    if result["composition"]["conflicts"]:
        print(f"Conflicts: {result['composition']['conflicts']}")
    print()
    print("Reachability:")
    for k, v in result["reachability"].items():
        print(f"  {k}: {v}")
    print()
    print(f"Summary: {result['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
