# SPDX-License-Identifier: Apache-2.0
"""Patch proof gate (§6.8 / R1 mitigation): static-check coverage for
every PATCH_REGISTRY entry, with proof artefacts under
`evidence/patch_proof/<patch_id>__<vllm_pin>.json`.

A "patch proof" is the operator-verifiable claim that a patch is wired
up correctly: registered, importable, no orphans, env_flag valid,
declared dependencies present in the registry. Bench-delta evidence
slots in later (Phase 10 GPU work); the static checks alone catch the
"dead patch" class of bugs §6.8 was designed to surface.

Three CLI surfaces (see `cli/patches.py` for the argparse wiring):

  sndr patches prove <id>              # verify one patch + write artefact
  sndr patches prove --all             # sweep all stable patches, report coverage
  sndr patches prove --dead-detect     # list patches with no proof artefact

Static checks per patch:

  P-1  patch present in PATCH_REGISTRY
  P-2  patch has `apply_module` declared OR is in KNOWN_SPEC_ONLY allowlist
  P-3  apply_module is importable (when declared)
  P-4  patch appears in legacy `_state.PATCH_REGISTRY` register OR is in
       KNOWN_SPEC_ONLY (no shadow-orphan)
  P-5  env_flag exists and matches canonical env-key registry
  P-6  every `requires_patches` id resolves to a registry entry
  P-7  every `conflicts_with` id resolves to a registry entry
  P-8  bench-delta artefact present (informational — caller decides
       whether absent bench data blocks release; tier-policy lives in §6.8)
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# v12: this file lives at sndr/proof/__init__.py — repo root is two
# levels up (was parents[3] when the package sat at vllm/sndr_core/).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROOF_DIR = REPO_ROOT / "evidence" / "patch_proof"

log = logging.getLogger("genesis.patches.prove")


__all__ = [
    "ProofCheck",
    "PatchProof",
    "static_checks_for_patch",
    "write_proof_artefact",
    "list_dead_patches",
    "load_proof_artefact",
    "find_proof_artefacts",
    "DEFAULT_PROOF_DIR",
    "classify_proof",
    "summarize_proof_status",
    "build_proof_for_patch",
    "PROOF_STATUS_BUCKETS",
]


# ─── Data structures ──────────────────────────────────────────────────


@dataclass
class ProofCheck:
    """One static check result."""
    rule: str            # "P-1" .. "P-8"
    passed: bool
    message: str


@dataclass
class PatchProof:
    """Aggregate proof artefact for one patch."""
    patch_id: str
    vllm_pin: str
    genesis_pin: str
    commit_sha: str
    host: str
    measured_at: str
    static_checks: list[ProofCheck] = field(default_factory=list)
    bench_delta: Optional[dict] = None    # populated later by GPU bench

    # ─── Convenience ──────────────────────────────────────────────

    @property
    def static_passed(self) -> bool:
        return all(c.passed for c in self.static_checks)

    @property
    def static_errors(self) -> list[ProofCheck]:
        return [c for c in self.static_checks if not c.passed]


# ─── Provenance helpers ───────────────────────────────────────────────


def _git_short_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _detect_vllm_pin() -> str:
    """Best-effort vllm version detection (works on Mac without vllm
    installed — returns 'not-installed')."""
    try:
        from importlib.metadata import version
        return version("vllm")
    except Exception:
        return "not-installed"


def _detect_genesis_pin() -> str:
    try:
        from importlib.metadata import version
        return version("vllm-sndr-core")
    except Exception:
        try:
            # Fallback to in-tree version.py if installed package not present.
            from sndr.version import __version__
            return __version__
        except Exception:
            return "unknown"


def _detect_host() -> str:
    """Phase 0 evidence-ledger host convention: local | server | <hostname>."""
    forced = os.environ.get("SNDR_HOST_LABEL")
    if forced:
        return forced
    try:
        import socket
        return socket.gethostname()
    except Exception:
        return "unknown"


# ─── Static check implementations ─────────────────────────────────────


def _check_p1_in_registry(patch_id: str, registry: dict) -> ProofCheck:
    """P-1: patch present in dispatcher.PATCH_REGISTRY."""
    if patch_id in registry:
        return ProofCheck("P-1", True, f"patch {patch_id!r} present in registry")
    return ProofCheck("P-1", False, f"patch {patch_id!r} not in PATCH_REGISTRY")


def _check_p2_apply_module_declared(
    patch_id: str, meta: dict, known_spec_only: frozenset[str],
    resolved_apply_module: Optional[str] = None,
    legacy_register_names: Optional[set[str]] = None,
) -> ProofCheck:
    """P-2: patch has a working apply mechanism.

    Three independent ways a patch satisfies P-2 (any one is enough):

      A. `resolved_apply_module` is set — either from an explicit
         `apply_module` key in registry.py OR from the integration-tree
         walk in `dispatcher.spec._build_apply_module_map`. Stage 6
         migrated patches (apply hooks under `integrations/<family>/`)
         pass through this branch.
      B. patch is in `KNOWN_SPEC_ONLY` allowlist (documentation /
         legacy stub entries that intentionally have no apply path).
      C. patch is present in the legacy `@register_patch` decorator
         register (`apply._state.PATCH_REGISTRY`) — these run via the
         legacy monolithic `_per_patch_dispatch.py` and don't need a
         V2 `apply_module` field yet. Phase 10 migrates each one
         individually into `integrations/<family>/`.

    Branch C is essential so the 127 patches still in the monolith
    aren't flagged as "dead" — they DO apply at runtime, just not via
    the V2 spec-loop yet.
    """
    apply_module = resolved_apply_module or meta.get("apply_module")
    if apply_module:
        return ProofCheck("P-2", True,
                          f"apply_module resolved: {apply_module}")
    if patch_id in known_spec_only:
        return ProofCheck("P-2", True,
                          "patch is in KNOWN_SPEC_ONLY allowlist "
                          "(documentation/legacy entry)")
    if legacy_register_names is not None:
        # Match `patch_id` against legacy register names. The legacy
        # convention uses combined entries like "P1/P2 FP8 kernel
        # dispatcher" — one register entry covers multiple PATCH_REGISTRY
        # ids. Accept "P1 ", "P1/", or exact match as a hit.
        for name in legacy_register_names:
            if (name == patch_id
                    or name.startswith(patch_id + " ")
                    or name.startswith(patch_id + "/")):
                return ProofCheck("P-2", True,
                                  "patch in legacy `@register_patch` register "
                                  "(applies via _per_patch_dispatch.py) — "
                                  "Phase 10 will migrate to integrations/<family>/")
    return ProofCheck("P-2", False,
                      "no apply_module AND not in KNOWN_SPEC_ONLY allowlist "
                      "AND not in legacy register — dead registry entry")


def _check_p3_apply_module_importable(
    meta: dict, resolved_apply_module: Optional[str] = None,
) -> Optional[ProofCheck]:
    """P-3: when apply_module is set (raw OR derived), it must be importable."""
    apply_module = resolved_apply_module or meta.get("apply_module")
    if not apply_module:
        return None     # skip — covered by P-2
    try:
        importlib.import_module(apply_module)
        return ProofCheck("P-3", True,
                          f"apply_module {apply_module!r} imports cleanly")
    except ImportError as e:
        return ProofCheck("P-3", False,
                          f"cannot import apply_module {apply_module!r}: {e}")
    except Exception as e:
        return ProofCheck("P-3", False,
                          f"apply_module {apply_module!r} raised "
                          f"{type(e).__name__} on import: {e}")


def _check_p4_no_shadow_orphan(patch_id: str, known_spec_only: frozenset[str],
                                legacy_names: set[str]) -> ProofCheck:
    """P-4: patch appears in legacy `_state.PATCH_REGISTRY` register
    OR is in KNOWN_SPEC_ONLY (so apply.shadow doesn't flag it)."""
    if patch_id in known_spec_only:
        return ProofCheck("P-4", True,
                          "patch is in KNOWN_SPEC_ONLY allowlist")
    # Legacy register stores names that often match patch_id directly.
    # Substring match handles variants like `apply_P67b` for `P67b`.
    if any(patch_id in name or name in patch_id for name in legacy_names):
        return ProofCheck("P-4", True,
                          "patch present in legacy apply register")
    return ProofCheck("P-4", False,
                      "patch absent from legacy register AND not in "
                      "KNOWN_SPEC_ONLY — apply.shadow would flag it")


def _check_p5_env_flag(patch_id: str, meta: dict,
                        canonical_keys: set[str]) -> ProofCheck:
    """P-5: env_flag exists and matches canonical env-key registry."""
    flag = meta.get("env_flag")
    if not flag:
        return ProofCheck("P-5", False,
                          "no env_flag declared (patch is not toggleable)")
    if flag not in canonical_keys:
        return ProofCheck("P-5", False,
                          f"env_flag {flag!r} not in canonical key registry "
                          f"(§6.7) — register the flag or fix the typo")
    return ProofCheck("P-5", True, f"env_flag {flag!r} is canonical")


def _check_p6_requires_resolvable(
    patch_id: str, meta: dict, registry: dict,
) -> Optional[ProofCheck]:
    """P-6: every `requires_patches` resolves."""
    reqs = meta.get("requires_patches") or []
    if not reqs:
        return None
    bad = [r for r in reqs if r not in registry]
    if bad:
        return ProofCheck("P-6", False,
                          f"requires_patches references unknown patch(es): {bad}")
    return ProofCheck("P-6", True,
                      f"all {len(reqs)} requires_patches resolve")


def _check_p7_conflicts_resolvable(
    patch_id: str, meta: dict, registry: dict,
) -> Optional[ProofCheck]:
    """P-7: every `conflicts_with` resolves (silent-typo guard)."""
    conflicts = meta.get("conflicts_with") or []
    if not conflicts:
        return None
    bad = [c for c in conflicts if c not in registry]
    if bad:
        return ProofCheck("P-7", False,
                          f"conflicts_with references unknown patch(es): {bad}")
    return ProofCheck("P-7", True,
                      f"all {len(conflicts)} conflicts_with resolve")


# ─── Public API ───────────────────────────────────────────────────────


def _resolved_apply_modules() -> dict[str, Optional[str]]:
    """Build {patch_id: resolved_apply_module} via PatchSpec layer.

    PatchSpec merges explicit registry values with the integration-tree
    walk in `dispatcher.spec._build_apply_module_map`. P-2 and P-3 must
    see this resolved value so Stage 6 migrations (apply hooks under
    `integrations/<family>/`) pass without requiring an explicit
    `apply_module` key in registry.py.
    """
    try:
        from sndr.dispatcher.spec import iter_patch_specs
    except ImportError:
        return {}
    return {s.patch_id: s.apply_module for s in iter_patch_specs()}


def static_checks_for_patch(
    patch_id: str,
    *,
    registry: Optional[dict] = None,
    canonical_keys: Optional[set[str]] = None,
    known_spec_only: Optional[frozenset[str]] = None,
    legacy_names: Optional[set[str]] = None,
    resolved_apply_modules: Optional[dict[str, Optional[str]]] = None,
) -> list[ProofCheck]:
    """Run every static check against one patch. Lazy-imports registries
    so the function works in tests with stubbed dependencies."""
    if registry is None:
        from sndr.dispatcher.registry import PATCH_REGISTRY
        registry = PATCH_REGISTRY
    if canonical_keys is None:
        from sndr.cli.legacy.config_keys import load_canonical_registry
        canonical_keys = set(load_canonical_registry().keys())
    if known_spec_only is None:
        from sndr.apply.shadow import KNOWN_SPEC_ONLY_PATCHES
        known_spec_only = KNOWN_SPEC_ONLY_PATCHES
    if legacy_names is None:
        try:
            from sndr.apply._state import PATCH_REGISTRY as LEG
            # Legacy register is a list of (name, callable) tuples.
            legacy_names = {name for name, _fn in LEG}
        except Exception:
            legacy_names = set()
    if resolved_apply_modules is None:
        resolved_apply_modules = _resolved_apply_modules()

    results: list[ProofCheck] = []

    p1 = _check_p1_in_registry(patch_id, registry)
    results.append(p1)
    if not p1.passed:
        return results       # No point checking other rules.
    meta = registry[patch_id]
    resolved_am = resolved_apply_modules.get(patch_id)

    results.append(_check_p2_apply_module_declared(
        patch_id, meta, known_spec_only, resolved_am,
        legacy_register_names=legacy_names,
    ))
    p3 = _check_p3_apply_module_importable(meta, resolved_am)
    if p3 is not None:
        results.append(p3)
    results.append(_check_p4_no_shadow_orphan(
        patch_id, known_spec_only, legacy_names,
    ))
    results.append(_check_p5_env_flag(patch_id, meta, canonical_keys))
    p6 = _check_p6_requires_resolvable(patch_id, meta, registry)
    if p6 is not None:
        results.append(p6)
    p7 = _check_p7_conflicts_resolvable(patch_id, meta, registry)
    if p7 is not None:
        results.append(p7)

    return results


def write_proof_artefact(
    proof: PatchProof,
    out_dir: Path = DEFAULT_PROOF_DIR,
) -> Path:
    """Persist proof to JSON. Filename
    `<patch_id>__<vllm_pin>.json`. The vllm_pin slot lets us keep
    historical artefacts across pin bumps for the dead-detect sweep."""
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_pin = proof.vllm_pin.replace("+", "_plus_").replace("/", "_")
    fname = f"{proof.patch_id}__{safe_pin}.json"
    target = out_dir / fname
    target.write_text(
        json.dumps({
            "patch_id": proof.patch_id,
            "vllm_pin": proof.vllm_pin,
            "genesis_pin": proof.genesis_pin,
            "commit_sha": proof.commit_sha,
            "host": proof.host,
            "measured_at": proof.measured_at,
            "static_checks": [
                {"rule": c.rule, "passed": c.passed, "message": c.message}
                for c in proof.static_checks
            ],
            "static_passed": proof.static_passed,
            "bench_delta": proof.bench_delta,
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


def load_proof_artefact(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_proof_artefacts(
    patch_id: str, out_dir: Path = DEFAULT_PROOF_DIR,
) -> list[Path]:
    """Find every proof artefact for one patch (one per vllm pin)."""
    if not out_dir.is_dir():
        return []
    return sorted(out_dir.glob(f"{patch_id}__*.json"))


def list_dead_patches(
    *,
    registry: Optional[dict] = None,
    out_dir: Path = DEFAULT_PROOF_DIR,
    require_static_pass: bool = True,
) -> list[dict]:
    """Return every PATCH_REGISTRY entry that has no proof artefact.

    When `require_static_pass=True`, artefacts with `static_passed=false`
    don't count as proof — the patch is still considered dead until the
    static failures are fixed.
    """
    if registry is None:
        from sndr.dispatcher.registry import PATCH_REGISTRY
        registry = PATCH_REGISTRY

    dead: list[dict] = []
    for patch_id, meta in registry.items():
        artefacts = find_proof_artefacts(patch_id, out_dir)
        proven = False
        for a in artefacts:
            try:
                data = load_proof_artefact(a)
            except (OSError, json.JSONDecodeError):
                continue
            if require_static_pass and not data.get("static_passed", False):
                continue
            proven = True
            break
        if not proven:
            dead.append({
                "patch_id": patch_id,
                "lifecycle": meta.get("lifecycle", "?"),
                "tier": meta.get("tier", "?"),
                "family": meta.get("family", "?"),
                "artefacts_found": [a.name for a in artefacts],
            })
    return dead


def build_proof_for_patch(patch_id: str) -> PatchProof:
    """Run static checks + collect provenance into a `PatchProof`
    without writing it (caller decides via `write_proof_artefact`)."""
    checks = static_checks_for_patch(patch_id)
    return PatchProof(
        patch_id=patch_id,
        vllm_pin=_detect_vllm_pin(),
        genesis_pin=_detect_genesis_pin(),
        commit_sha=_git_short_sha(),
        host=_detect_host(),
        measured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        static_checks=checks,
    )


# ─── Proof-status classification (§6.8 read-side reporting) ───────────


# Bucket order matters — `summarize_proof_status` uses it for stable
# table ordering and for picking the "best" bucket when one patch has
# multiple artefacts.
PROOF_STATUS_BUCKETS: tuple[str, ...] = (
    "bench_with_baseline",   # static green + bench_delta + at least one *_delta_pct
    "bench_attached",        # static green + bench_delta has a metric
    "static_only",           # static green, no bench evidence
    "static_failed",         # artefact exists but static_passed=false
    "dead",                  # no artefact at all
)

# Metric keys that, if present in `bench_delta`, classify it as
# "bench_attached" (vs. just having identifier metadata).
_BENCH_METRIC_KEYS: tuple[str, ...] = (
    "median_tps", "p95_tps", "decode_tpot_ms", "ttft_ms",
    "cv_pct", "tool_call_score",
)

# Keys that prove a baseline comparison was attached.
_BENCH_DELTA_KEYS: tuple[str, ...] = (
    "median_tps_delta_pct", "p95_tps_delta_pct",
    "decode_tpot_delta_pct", "ttft_delta_pct",
)


def classify_proof(artefact: dict) -> str:
    """Classify one loaded proof artefact into a single bucket.

    Returns one of `PROOF_STATUS_BUCKETS` (except `dead`, which is
    reserved for the "no artefact" case).
    """
    if not artefact.get("static_passed", False):
        return "static_failed"
    bench = artefact.get("bench_delta") or {}
    if not isinstance(bench, dict) or not bench:
        return "static_only"
    if any(k in bench and bench[k] is not None for k in _BENCH_DELTA_KEYS):
        return "bench_with_baseline"
    if any(k in bench and bench[k] is not None for k in _BENCH_METRIC_KEYS):
        return "bench_attached"
    return "static_only"


def _bucket_rank(bucket: str) -> int:
    """Lower rank = better. Used to pick the best bucket when a patch
    has multiple artefacts (e.g., across vllm pins)."""
    try:
        return PROOF_STATUS_BUCKETS.index(bucket)
    except ValueError:
        return len(PROOF_STATUS_BUCKETS)


def summarize_proof_status(
    *,
    registry: Optional[dict] = None,
    out_dir: Path = DEFAULT_PROOF_DIR,
) -> dict:
    """Aggregate proof-artefact state across every PATCH_REGISTRY entry.

    Returns a dict shaped:

        {
          "total":   <int>,
          "counts":  {bucket: int, ...},  # one entry per PROOF_STATUS_BUCKETS
          "patches": [
            {
              "patch_id":  "PN82",
              "bucket":    "bench_with_baseline",
              "lifecycle": "stable",
              "tier":      "...",
              "family":    "...",
              "artefacts": ["PN82__vllm_0.6.4.json", ...],
            },
            ...
          ],
        }

    A patch with multiple artefacts (different vllm pins) is reported
    with its *best* bucket — release decisions should use the freshest
    evidence available across pins.
    """
    if registry is None:
        from sndr.dispatcher.registry import PATCH_REGISTRY
        registry = PATCH_REGISTRY

    counts: dict[str, int] = {b: 0 for b in PROOF_STATUS_BUCKETS}
    patches: list[dict] = []

    for patch_id, meta in registry.items():
        artefacts = find_proof_artefacts(patch_id, out_dir)
        if not artefacts:
            bucket = "dead"
        else:
            best = "static_failed"
            for a in artefacts:
                try:
                    data = load_proof_artefact(a)
                except (OSError, json.JSONDecodeError):
                    continue
                b = classify_proof(data)
                if _bucket_rank(b) < _bucket_rank(best):
                    best = b
            bucket = best

        counts[bucket] = counts.get(bucket, 0) + 1
        patches.append({
            "patch_id": patch_id,
            "bucket": bucket,
            "lifecycle": meta.get("lifecycle", "?"),
            "tier": meta.get("tier", "?"),
            "family": meta.get("family", "?"),
            "artefacts": [a.name for a in artefacts],
        })

    return {
        "total": len(patches),
        "counts": counts,
        "patches": patches,
    }
