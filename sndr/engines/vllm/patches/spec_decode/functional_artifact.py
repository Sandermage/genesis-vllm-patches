# SPDX-License-Identifier: Apache-2.0
"""functional_artifact — recorded bench-validation receipts.

A FunctionalArtifact captures the evidence that one (model, pin,
config) combination was empirically benchmarked under a named
profile and produced quantified TPS/accept/VRAM metrics. The
safety_guard reads these artifacts to relax the
``FUNCTIONAL_UNVERIFIED`` requirement for matching configurations.

Without an artifact, a non-EXACT contract verdict still requires
both ``SNDR_ALLOW_SPEC_DECODE_KV_ADAPTER`` and
``SNDR_ALLOW_SPEC_DECODE_FUNCTIONAL_UNKNOWN`` envs. With a
matching artifact, only the ``KV_ADAPTER`` (structural opt-in)
env is required. ``GENESIS_*`` aliases still work via the
``get_sndr_env()`` resolver (with deprecation warning).

Files live as JSON next to this module:
  ``sndr/engines/vllm/patches/spec_decode/artifacts/<profile>.json``

Production policy stays conservative — an artifact is REQUIRED
but not SUFFICIENT for production-default promotion. The artifact
proves the contract is functionally non-regressive; whether
operator wants that profile on by default is a separate policy.

Provenance:
  Authored 2026-05-20 after PN271b/G4_71b/β′-A bench session
  proved that workload-conditional profiles (not global flags) are
  the right unit of MTP/TQ economics.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("genesis.spec_decode.functional_artifact")


_ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


# ----------------------- Artifact schema -----------------------

@dataclass
class FunctionalArtifact:
    """One bench-validation receipt.

    Schema is intentionally flat-JSON-friendly: every field maps to a
    JSON-encodable value.
    """
    model_id: str
    profile: str
    vllm_pin: str
    config_hash: str        # see compute_config_hash()
    patch_hash: str         # git commit / patch-tree fingerprint
    created_at: str         # ISO 8601
    prompt_suite: list[str]
    workload_classes: list[str]
    kv_plan: dict[str, Any]
    metrics: dict[str, Any]
    decision: str           # 'validated_conditional' / 'validated_global' / 'denied'
    allowed_workloads: list[str] = field(default_factory=list)
    denied_workloads: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "profile": self.profile,
            "vllm_pin": self.vllm_pin,
            "config_hash": self.config_hash,
            "patch_hash": self.patch_hash,
            "created_at": self.created_at,
            "prompt_suite": list(self.prompt_suite),
            "workload_classes": list(self.workload_classes),
            "kv_plan": dict(self.kv_plan),
            "metrics": dict(self.metrics),
            "decision": self.decision,
            "allowed_workloads": list(self.allowed_workloads),
            "denied_workloads": list(self.denied_workloads),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FunctionalArtifact":
        return cls(
            model_id=d["model_id"],
            profile=d["profile"],
            vllm_pin=d["vllm_pin"],
            config_hash=d["config_hash"],
            patch_hash=d["patch_hash"],
            created_at=d["created_at"],
            prompt_suite=list(d.get("prompt_suite") or []),
            workload_classes=list(d.get("workload_classes") or []),
            kv_plan=dict(d.get("kv_plan") or {}),
            metrics=dict(d.get("metrics") or {}),
            decision=d["decision"],
            allowed_workloads=list(d.get("allowed_workloads") or []),
            denied_workloads=list(d.get("denied_workloads") or []),
            notes=d.get("notes", ""),
        )


# ----------------------- Hashing -----------------------

def compute_config_hash(model_id: str, vllm_pin: str,
                        kv_plan: dict[str, Any], mtp_k: int | None,
                        drafter_backend: str | None = None) -> str:
    """Stable hash over (model, kv_plan, K, drafter_backend).

    NOTE: vllm_pin is intentionally NOT in the hash. Pin is stored in
    the artifact as metadata so an operator log clearly shows what
    pin was bench'd, but a pin bump should not silently invalidate a
    config that's structurally identical. If you want stricter
    matching, validate ``art.vllm_pin == live_pin`` at the caller.

    Order-stable via sort_keys. 16-hex truncated sha256.
    """
    blob = json.dumps(
        {
            "model_id": model_id,
            "kv_plan": kv_plan,
            "mtp_k": mtp_k,
            "drafter_backend": drafter_backend,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# ----------------------- IO -----------------------

def _ensure_dir() -> None:
    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def write(artifact: FunctionalArtifact, path: str | os.PathLike | None = None
          ) -> str:
    """Write artifact to JSON. If path is None, derives from profile."""
    _ensure_dir()
    if path is None:
        path = _ARTIFACTS_DIR / f"{artifact.profile}.json"
    p = Path(path)
    p.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True))
    log.info("[functional_artifact] wrote %s (config_hash=%s)",
             p, artifact.config_hash)
    return str(p)


def read(path: str | os.PathLike) -> FunctionalArtifact:
    p = Path(path)
    raw = json.loads(p.read_text())
    return FunctionalArtifact.from_dict(raw)


def find_matching(model_id: str, profile: str,
                  config_hash: str) -> FunctionalArtifact | None:
    """Look up an artifact matching (model, profile, config_hash).

    Looks in the shipped artifacts/ directory first, then in any
    operator-supplied directory via env
    ``SNDR_SPEC_DECODE_ARTIFACTS_DIR`` (alias
    ``GENESIS_SPEC_DECODE_ARTIFACTS_DIR``).
    """
    candidates: list[Path] = []
    if _ARTIFACTS_DIR.exists():
        candidates.extend(_ARTIFACTS_DIR.glob("*.json"))
    from ...env import get_sndr_env
    extra = (get_sndr_env("SPEC_DECODE_ARTIFACTS_DIR") or "").strip()
    if extra and Path(extra).exists():
        candidates.extend(Path(extra).glob("*.json"))

    for path in candidates:
        try:
            art = read(path)
        except Exception as _e:  # noqa: BLE001
            log.warning("[functional_artifact] skip %s: %s", path, _e)
            continue
        if (art.model_id == model_id
                and art.profile == profile
                and art.config_hash == config_hash):
            return art
    return None


# ----------------------- Bench ingest -----------------------

def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "cv": 0.0}
    n = len(values)
    mean = statistics.mean(values)
    if n > 1:
        std = statistics.stdev(values)
        cv = std / mean if mean else 0.0
    else:
        std = 0.0
        cv = 0.0
    return {"n": n, "mean": mean, "std": std, "cv": cv,
            "min": min(values), "max": max(values)}


def _geomean(values: list[float]) -> float:
    import math
    if not values:
        return 0.0
    return math.exp(sum(math.log(max(v, 1e-9)) for v in values) / len(values))


def from_bench(
    *,
    model_id: str,
    profile: str,
    vllm_pin: str,
    kv_plan: dict[str, Any],
    mtp_k: int | None,
    drafter_backend: str | None,
    patch_hash: str,
    created_at: str,
    baseline_bench_path: str | os.PathLike,
    profile_bench_path: str | os.PathLike,
    structured_classes: list[str] = ("structured_count", "code_gen",
                                     "tool_json"),
    free_classes: list[str] = ("free_chat", "summarization"),
    promotion_gain_threshold: float = 0.10,
    notes: str = "",
) -> FunctionalArtifact:
    """Build a FunctionalArtifact from two bench JSON files.

    Decision rule:
      structured mean Δ >= +``promotion_gain_threshold`` ->
        decision='validated_conditional' (allowed_workloads = structured)
      global geomean Δ >= +``promotion_gain_threshold`` ->
        decision='validated_global'
      else -> decision='denied'
    """
    bdata = json.loads(Path(baseline_bench_path).read_text())
    pdata = json.loads(Path(profile_bench_path).read_text())

    def _means(d: dict[str, Any]) -> dict[str, float]:
        return {p: d["results"][p]["tps_stats"]["mean"]
                for p in d["results"]}

    bmeans = _means(bdata)
    pmeans = _means(pdata)

    workload_classes = sorted(set(bmeans) | set(pmeans))
    deltas = {p: (pmeans[p] - bmeans[p]) / bmeans[p]
              for p in workload_classes if p in bmeans and p in pmeans}

    # Aggregates
    baseline_geomean = _geomean(list(bmeans.values()))
    profile_geomean = _geomean(list(pmeans.values()))
    profile_global_delta = (profile_geomean - baseline_geomean) / baseline_geomean

    structured_means_profile = [pmeans[c] for c in structured_classes
                                if c in pmeans]
    structured_means_baseline = [bmeans[c] for c in structured_classes
                                 if c in bmeans]
    profile_structured_mean = (statistics.mean(structured_means_profile)
                               if structured_means_profile else 0.0)
    baseline_structured_mean = (statistics.mean(structured_means_baseline)
                                if structured_means_baseline else 0.0)
    profile_structured_delta = (
        (profile_structured_mean - baseline_structured_mean)
        / baseline_structured_mean
        if baseline_structured_mean else 0.0
    )

    free_means_profile = [pmeans[c] for c in free_classes if c in pmeans]
    free_means_baseline = [bmeans[c] for c in free_classes if c in bmeans]
    profile_free_mean = (statistics.mean(free_means_profile)
                         if free_means_profile else 0.0)
    baseline_free_mean = (statistics.mean(free_means_baseline)
                          if free_means_baseline else 0.0)
    profile_free_delta = (
        (profile_free_mean - baseline_free_mean) / baseline_free_mean
        if baseline_free_mean else 0.0
    )

    accept = (pdata.get("accept_trace") or {})
    vram = pdata.get("vram") or []
    vram_free_min = (min((v.get("free_mib", 0) for v in vram), default=0)
                     if vram else None)

    # Decision
    if profile_global_delta >= promotion_gain_threshold:
        decision = "validated_global"
        allowed = list(workload_classes)
        denied: list[str] = []
    elif profile_structured_delta >= promotion_gain_threshold:
        decision = "validated_conditional"
        allowed = [c for c in workload_classes
                   if deltas.get(c, 0.0) >= promotion_gain_threshold]
        denied = [c for c in workload_classes if c not in allowed]
    else:
        decision = "denied"
        allowed = []
        denied = list(workload_classes)

    metrics = {
        "baseline_tps_per_class": bmeans,
        "profile_tps_per_class": pmeans,
        "delta_tps_per_class": deltas,
        "baseline_geomean_tps": baseline_geomean,
        "profile_geomean_tps": profile_geomean,
        "profile_delta_global": profile_global_delta,
        "profile_structured_mean_tps": profile_structured_mean,
        "baseline_structured_mean_tps": baseline_structured_mean,
        "profile_delta_structured": profile_structured_delta,
        "profile_free_mean_tps": profile_free_mean,
        "baseline_free_mean_tps": baseline_free_mean,
        "profile_delta_free": profile_free_delta,
        "promotion_gain_threshold": promotion_gain_threshold,
        "acceptance": accept,
        "vram_free_mib_min": vram_free_min,
        "vram_snapshot": vram,
    }

    config_hash = compute_config_hash(
        model_id=model_id, vllm_pin=vllm_pin, kv_plan=kv_plan,
        mtp_k=mtp_k, drafter_backend=drafter_backend,
    )

    return FunctionalArtifact(
        model_id=model_id,
        profile=profile,
        vllm_pin=vllm_pin,
        config_hash=config_hash,
        patch_hash=patch_hash,
        created_at=created_at,
        prompt_suite=workload_classes,
        workload_classes=workload_classes,
        kv_plan=kv_plan,
        metrics=metrics,
        decision=decision,
        allowed_workloads=allowed,
        denied_workloads=denied,
        notes=notes,
    )


__all__ = [
    "FunctionalArtifact",
    "compute_config_hash",
    "write",
    "read",
    "find_matching",
    "from_bench",
]
