#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Attach bench measurements to per-patch proof artefacts.

Why
---

Per-patch proof artefacts under ``evidence/patch_proof/<id>__<pin>.json``
carry a ``bench_delta`` field that classifies the artefact into one of:

  * ``bench_with_baseline`` — has ``*_delta_pct`` keys against a known
    baseline (this is the hardened-release target);
  * ``bench_attached``      — has at least one bench metric but no
    delta-percentage versus a baseline;
  * ``static_only``         — bench_delta empty.

``sndr patches prove --all`` only writes the static side. Real bench
data is filled in by attaching a bench-suite JSON to the patches that
were active during the bench. This script does that attachment:

  python3 scripts/attach_bench_proof.py \
      --bench tools/bench_results/27b_dflash_multiconc_2026-05-16.json \
      --preset prod-qwen3.6-27b-dflash-multiconc \
      [--baseline tests/integration/baselines/27b_dflash_multiconc_2026-05-16.json] \
      [--dry-run]

The bench JSON is the ``genesis_bench_suite.py --quick``/full output (a
single object with keys ``wall_TPS``, ``decode_TPOT_ms``, ``ttft_ms``,
``cv_pct``, …). The preset alias resolves the patch list — every
proof artefact for those patches gets the bench measurements written
into its ``bench_delta``. If ``--baseline`` is provided and contains
the same metric keys, the deltas are computed and emitted as
``*_delta_pct`` entries (which promotes the bucket to
``bench_with_baseline``).

Exit codes:
  0 — attachments succeeded (or dry-run preview emitted)
  1 — bench file unreadable / preset unresolvable / no patches updated
  2 — internal error
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Pull the canonical methodology SHA helper from the sibling audit script
# so attached bench_delta entries carry the same fingerprint the audit
# checks against. See scripts/audit_bench_methodology.py for the contract.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from audit_bench_methodology import (  # noqa: E402
    METHODOLOGY_FILE,
    _canonical_methodology_sha,
)


_BENCH_METRIC_FIELDS = {
    "median_tps":      ("wall_TPS", "wall_TPS_median", "wall_TPS_mean",
                        "median_tps"),
    "p95_tps":         ("wall_TPS_p95", "p95_tps"),
    "decode_tpot_ms":  ("decode_TPOT_ms", "decode_TPOT_ms_median",
                        "decode_tpot_ms"),
    "ttft_ms":         ("TTFT_ms", "ttft_ms_median", "TTFT_ms_median",
                        "ttft_ms"),
    "cv_pct":          ("cv_pct", "wall_TPS_cv_pct", "TPS_cv_pct"),
    "tool_call_score": ("tool_call_score", "tool_call_pass_rate"),
}

# Nested keys to recurse into. genesis_bench_suite emits a nested
# structure like ``decode_bench.wall_TPS.mean``; flatten those paths
# before applying the metric-field map above.
_BENCH_SUBSECTIONS = ("decode_bench", "aggregate", "summary", "stats")

# Map bench metric → matching delta-pct key (per
# vllm/sndr_core/proof/__init__.py:_BENCH_DELTA_KEYS).
_DELTA_KEYS = {
    "median_tps":     "median_tps_delta_pct",
    "p95_tps":        "p95_tps_delta_pct",
    "decode_tpot_ms": "decode_tpot_delta_pct",
    "ttft_ms":        "ttft_delta_pct",
}


def _extract_metric(bench: dict, candidates: tuple[str, ...]) -> Optional[float]:
    for k in candidates:
        if k in bench:
            v = bench[k]
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, dict):
                # Common pattern: {"value": 130.7, "unit": "tok/s"} or
                # {"mean": 130.7, "stdev": 4.3}.
                for inner in ("value", "median", "mean"):
                    if inner in v and isinstance(v[inner], (int, float)):
                        return float(v[inner])
    return None


def _collect_bench_metrics(bench: dict) -> dict[str, float]:
    """Flatten a bench-suite JSON into the canonical metric subset.

    Probes the top-level keys directly plus every known subsection
    (``decode_bench``, ``aggregate``, ``summary``, ``stats``) — at each
    layer the field map in :data:`_BENCH_METRIC_FIELDS` resolves
    candidate keys, and nested ``{mean, median, value}`` envelopes are
    flattened by :func:`_extract_metric`.
    """
    out: dict[str, float] = {}
    layers: list[dict] = [bench]
    for sub in _BENCH_SUBSECTIONS:
        node = bench.get(sub)
        if isinstance(node, dict):
            layers.append(node)
    for layer in layers:
        for canonical, candidates in _BENCH_METRIC_FIELDS.items():
            if canonical in out:
                continue
            v = _extract_metric(layer, candidates)
            if v is not None:
                out[canonical] = v
    return out


def _compute_deltas(
    current: dict[str, float], baseline: dict[str, float],
) -> dict[str, float]:
    """For each known delta-key target, compute percent delta against
    baseline if both sides have the metric. Positive = improvement on
    throughput metrics, regression on latency metrics — interpretation
    is done by the release-check policy, not here."""
    deltas: dict[str, float] = {}
    for metric, delta_key in _DELTA_KEYS.items():
        c = current.get(metric)
        b = baseline.get(metric)
        if c is None or b is None or b == 0:
            continue
        deltas[delta_key] = (c - b) / b * 100.0
    return deltas


def _proof_artefacts_for(patch_id: str, proof_dir: Path) -> list[Path]:
    if not proof_dir.is_dir():
        return []
    return sorted(proof_dir.glob(f"{patch_id}__*.json"))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", required=True,
                    help="Path to a bench-suite JSON (output of "
                         "tools/genesis_bench_suite.py).")
    ap.add_argument("--preset", required=True,
                    help="V2 preset alias whose patches receive the "
                         "attachment (e.g. prod-qwen3.6-27b-dflash-multiconc).")
    ap.add_argument("--baseline", default=None,
                    help="Optional baseline JSON. When provided, "
                         "delta-percent values are computed and attached "
                         "so the proof bucket promotes to "
                         "bench_with_baseline.")
    ap.add_argument("--proof-dir", default="evidence/patch_proof",
                    help="Override proof artefact directory.")
    ap.add_argument("--include-default-on", action="store_true",
                    help="Also attach the bench to every patch flagged "
                         "``default_on=True`` in the registry. Those "
                         "patches load implicitly when the preset does "
                         "not explicitly disable them, so the bench "
                         "result is empirical evidence for them too — "
                         "even when the preset's genesis_env block "
                         "doesn't name them.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the planned changes, do not write files.")
    args = ap.parse_args(argv)

    bench_path = Path(args.bench)
    if not bench_path.is_file():
        print(f"bench file not found: {bench_path}", file=sys.stderr)
        return 1
    try:
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"failed to parse {bench_path}: {e}", file=sys.stderr)
        return 1

    metrics = _collect_bench_metrics(bench)
    if not metrics:
        print(f"no recognizable bench metrics extracted from {bench_path}",
              file=sys.stderr)
        return 1

    baseline_metrics: dict[str, float] = {}
    if args.baseline:
        bpath = Path(args.baseline)
        if not bpath.is_file():
            print(f"baseline file not found: {bpath}", file=sys.stderr)
            return 1
        try:
            baseline_metrics = _collect_bench_metrics(
                json.loads(bpath.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError) as e:
            print(f"failed to parse baseline {bpath}: {e}", file=sys.stderr)
            return 1

    try:
        from sndr.model_configs import registry_v2
    except ImportError as e:
        print(f"cannot import registry_v2: {e}", file=sys.stderr)
        return 2
    try:
        composed = registry_v2.load_alias(args.preset)
    except Exception as e:  # noqa: BLE001
        print(f"failed to resolve preset {args.preset!r}: {e}", file=sys.stderr)
        return 1

    env = composed.genesis_env or {}
    enabled_env_keys = {
        k for k, v in env.items() if str(v) in ("1", "true", "True")
    }

    from sndr.dispatcher.registry import PATCH_REGISTRY
    env_to_pid = {
        meta["env_flag"]: pid
        for pid, meta in PATCH_REGISTRY.items()
        if isinstance(meta, dict) and isinstance(meta.get("env_flag"), str)
    }
    target_set: set[str] = {
        env_to_pid[k] for k in enabled_env_keys if k in env_to_pid
    }
    default_on_added: set[str] = set()
    if args.include_default_on:
        for pid, meta in PATCH_REGISTRY.items():
            if not isinstance(meta, dict):
                continue
            if meta.get("default_on") is not True:
                continue
            # Skip when the preset explicitly *disabled* the patch.
            env_flag = meta.get("env_flag")
            if isinstance(env_flag, str) and env.get(env_flag) == "0":
                continue
            if pid not in target_set:
                default_on_added.add(pid)
                target_set.add(pid)
    target_patches = sorted(target_set)
    if not target_patches:
        print(f"preset {args.preset!r} did not resolve to any registered "
              f"patches (env keys: {sorted(enabled_env_keys)[:5]}...)",
              file=sys.stderr)
        return 1

    proof_dir = REPO_ROOT / args.proof_dir
    delta_pct = _compute_deltas(metrics, baseline_metrics) if baseline_metrics else {}

    attach_payload = dict(metrics)
    attach_payload.update(delta_pct)
    attach_payload["bench_source"] = bench_path.name
    attach_payload["preset"] = args.preset
    attach_payload["attached_at"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    attach_payload["methodology_sha"] = _canonical_methodology_sha(
        METHODOLOGY_FILE
    )
    if args.baseline:
        attach_payload["baseline_source"] = Path(args.baseline).name

    bucket_label = "bench_with_baseline" if delta_pct else "bench_attached"

    print(f"attach-bench-proof: preset={args.preset!r}")
    print(f"  bench: {bench_path}")
    if args.baseline:
        print(f"  baseline: {args.baseline}")
    print(f"  metrics: {sorted(metrics.keys())}")
    print(f"  deltas:  {sorted(delta_pct.keys()) or '(none — no baseline)'}")
    print(f"  target bucket: {bucket_label}")
    print(f"  patches from preset env: "
          f"{len(target_patches) - len(default_on_added)}")
    if default_on_added:
        print(f"  patches from default_on=True (implicit): "
              f"{len(default_on_added)}")
    print(f"  total target patches: {len(target_patches)}")
    print()

    written = 0
    skipped = 0
    missing = 0
    for pid in target_patches:
        artefacts = _proof_artefacts_for(pid, proof_dir)
        if not artefacts:
            missing += 1
            continue
        for art in artefacts:
            try:
                data = json.loads(art.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                skipped += 1
                continue
            # Preserve existing bench_delta keys not overwritten by this run.
            existing = data.get("bench_delta") or {}
            merged = dict(existing) if isinstance(existing, dict) else {}
            merged.update(attach_payload)
            data["bench_delta"] = merged
            if args.dry_run:
                continue
            art.write_text(
                json.dumps(data, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            written += 1

    if args.dry_run:
        print(f"  dry-run preview: would update {len(target_patches) - missing} patches "
              f"({missing} have no proof artefact yet — run `sndr patches "
              f"prove --all` first)")
        return 0

    print(f"  ✓ wrote bench_delta into {written} proof artefact(s)")
    if missing:
        print(f"  ⚠ {missing} patch(es) had no proof artefact — run "
              f"`sndr patches prove --all` first")
    if skipped:
        print(f"  ⚠ {skipped} artefact(s) skipped (unreadable)")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
