# SPDX-License-Identifier: Apache-2.0
"""SNDR Core CLI — `sndr memory` VRAM budget tooling.

T1.3 (audit closure 2026-05-09 / production roadmap §18.3 Phase 1).

Subcommands:

  sndr memory explain <PRESET_KEY> [--gpu-vram GiB]
                                   [--ctx N] [--seqs N]
                                   [--kv-dtype fp8|fp16|...]
                                   [--json]
        — Static VRAM breakdown for a preset BEFORE booting vllm.
          Reads `cfg.model_path` (config.json + safetensors) and
          computes weights / KV / activations / CUDA-graph / Marlin
          components. CLI args override config defaults so operators
          can sweep without editing the YAML.

  sndr memory simulate --ctx 128k --sequences 2 [--model PATH]
        — Forward-projection helper: same estimator, but takes
          ctx/seqs directly and a model path (no preset). Useful for
          "what if I bumped max_model_len from 64k to 128k?".

  sndr memory doctor
        — Ad-hoc registry probe: which presets fit on the operator's
          declared GPU? Lists all builtin/community presets with
          per-preset utilization, flags >95% as RED.

  sndr memory report --live
        — Phase 2 (deferred): live VRAM probe via memory_metrics.

Phase 1 ships explain + simulate + doctor as static estimators.
Live integration arrives in Phase 2 once boot probe wiring is solid.

Author: Sandermage(Sander)-Barzov Aleksandr.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import _io


# ─── helpers ────────────────────────────────────────────────────────────


def _parse_ctx(s: str) -> int:
    """Parse '128k' / '256K' / '32768' → integer token count.

    Tolerant: accepts 'k' (1024×) and 'K' (1024×) suffix; raw integers
    pass through unchanged. Used for both --ctx and --max-model-len
    overrides.
    """
    s = s.strip().lower()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1024)
    if s.endswith("m"):
        return int(float(s[:-1]) * 1024 * 1024)
    return int(s)


def _gib_to_bytes(s: str) -> int:
    """Parse '24' or '24GiB' or '24 GiB' → integer bytes."""
    s = s.strip().lower().replace(" ", "")
    for suf in ("gib", "gb", "g"):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return int(float(s) * (1 << 30))


def _component_to_dict(c) -> dict[str, Any]:
    return {
        "name": c.name,
        "bytes": c.bytes_,
        "human": c.human,
        "confidence": c.confidence,
        "notes": c.notes,
    }


def _estimate_to_dict(estimate) -> dict[str, Any]:
    return {
        "preset": estimate.preset_key,
        "model_path": estimate.model_path,
        "gpu_count": estimate.gpu_count,
        "gpu_vram_bytes": estimate.gpu_vram_bytes,
        "total_bytes": estimate.total_bytes,
        "utilization": round(estimate.utilization, 4),
        "components": [_component_to_dict(c) for c in estimate.components],
        "warnings": list(estimate.warnings),
        "recommendations": list(estimate.recommendations),
    }


# ─── `sndr memory explain` ──────────────────────────────────────────────


def _resolve_preset_v1_or_v2(key: str):
    """Phase 4.7 MVP — try V1 preset registry first, fall back to V2 alias.

    Returns a V1 `ModelConfig`. V2 aliases compose into V1 via the V2
    composer so the existing memory estimator pipeline works unchanged.
    """
    from vllm.sndr_core.model_configs.registry import get as get_config
    from vllm.sndr_core.model_configs.schema import SchemaError

    # V1 lookup first (existing path; keeps `sndr memory explain a5000-2x-35b-prod` working).
    try:
        cfg = get_config(key)
        if cfg is not None:
            return cfg
    except SchemaError:
        pass
    # V2 alias fallback (`prod-35b`, `long-ctx-27b`, etc.).
    try:
        from vllm.sndr_core.model_configs.registry_v2 import load_alias
        return load_alias(key)
    except SchemaError as e:
        raise SchemaError(
            f"preset/alias {key!r} not found in V1 registry or V2 aliases: {e}"
        ) from e


_CRITICAL_COMPONENT_PREFIXES = ("Model weights", "KV cache")


def _has_critical_low_confidence(estimate) -> tuple[bool, list[str]]:
    """Return (flag, missing_inputs).

    A component is "critical low-confidence" when it sits in the
    Model-weights or KV-cache buckets, reports zero bytes, and was
    tagged `confidence="low"` by the estimator. In that state the
    estimator has no signal for the two largest VRAM consumers, so
    the verdict must NOT claim safety.

    Returns the list of probable missing inputs so the CLI / JSON
    consumer can surface an actionable fix.
    """
    missing: list[str] = []
    bad = False
    for c in getattr(estimate, "components", []):
        if not c.name.startswith(_CRITICAL_COMPONENT_PREFIXES):
            continue
        if c.bytes_ == 0 and c.confidence == "low":
            bad = True
            if c.name.startswith("Model weights"):
                missing.append("model safetensors not readable")
            elif c.name.startswith("KV cache"):
                missing.append("KV shape (num_hidden_layers / head_dim) not derivable")
    # Dedup while preserving order.
    seen = set()
    missing_unique = [m for m in missing if not (m in seen or seen.add(m))]
    return bad, missing_unique


def _compute_verdict(estimate, p95_factor: float = 1.15,
                     worst_factor: float = 1.35) -> dict:
    """Derive median / p95 / worst-case totals + verdict.

    Components carry confidence bands; we approximate p95 ≈ median × 1.15
    and worst-case ≈ median × 1.35 (calibration data ships in Phase 4.7
    advanced; this MVP uses conservative factors).

    Verdict thresholds:

      UNKNOWN   — Model weights or KV cache estimated as 0 bytes with
                  low confidence: no defensible capacity statement can
                  be made. Recommendations must NOT propose raising
                  context / batch in this state.
      SAFE      — p95 ≤ vram_budget
      TIGHT     — median ≤ budget < p95
      OOM_RISK  — median > budget OR worst > budget × 1.05
    """
    median_mib = int(estimate.total_bytes / (1024 * 1024))
    budget_mib = int(estimate.gpu_vram_bytes / (1024 * 1024))
    p95_mib = int(median_mib * p95_factor)
    worst_mib = int(median_mib * worst_factor)

    low_conf, missing_inputs = _has_critical_low_confidence(estimate)
    if low_conf:
        verdict = "UNKNOWN"
        actionable = False
    elif median_mib > budget_mib:
        verdict = "OOM_RISK"
        actionable = True
    elif worst_mib > int(budget_mib * 1.05):
        verdict = "OOM_RISK"
        actionable = True
    elif p95_mib > budget_mib:
        verdict = "TIGHT"
        actionable = True
    else:
        verdict = "SAFE"
        actionable = True

    return {
        "verdict": verdict,
        "actionable": actionable,
        "missing_inputs": missing_inputs,
        "total_median_mib_per_gpu": median_mib,
        "total_p95_mib_per_gpu": p95_mib,
        "total_worst_mib_per_gpu": worst_mib,
        "budget_mib_per_gpu": budget_mib,
        "p95_factor": p95_factor,
        "worst_factor": worst_factor,
    }


def _parse_ctx_sweep(s: str) -> list[int]:
    """Parse `--ctx-sweep 4k,16k,64k,128k` into a list of ints.

    Phase 4.7 MVP — sweep mode lets operators see how verdict
    transitions across context sizes without re-running explain N times.
    """
    out: list[int] = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(_parse_ctx(chunk))
    if not out:
        raise ValueError(
            "--ctx-sweep must list at least one context size "
            "(comma-separated, e.g. 4k,16k,64k)"
        )
    return out


def _run_explain(opts: argparse.Namespace) -> int:
    from vllm.sndr_core.runtime.memory_estimator import (
        estimate_for_config,
        render_waterfall,
    )
    from vllm.sndr_core.model_configs.schema import SchemaError

    # accept both V1 preset keys AND V2 aliases.
    try:
        cfg = _resolve_preset_v1_or_v2(opts.preset)
    except SchemaError as e:
        _io.fatal(f"preset {opts.preset!r}: {e}", 2)
    except Exception as e:
        _io.fatal(f"preset {opts.preset!r}: {type(e).__name__}: {e}", 2)
    if cfg is None:
        _io.fatal(f"preset {opts.preset!r} not found", 2)

    # Phase 4.7 MVP — `--ctx-sweep K1,K2,K3` short-circuits the single-run
    # branch. Walk each context size, emit a compact verdict row.
    ctx_sweep_raw = getattr(opts, "ctx_sweep", None)
    if ctx_sweep_raw:
        try:
            sweep_values = _parse_ctx_sweep(ctx_sweep_raw)
        except ValueError as e:
            _io.fatal(str(e), 2)

        rows: list[dict] = []
        for ctx_val in sweep_values:
            from dataclasses import replace as _replace
            sweep_cfg = _replace(cfg, max_model_len=ctx_val)
            if opts.seqs is not None:
                sweep_cfg = _replace(sweep_cfg, max_num_seqs=int(opts.seqs))
            if opts.kv_dtype is not None:
                sweep_cfg = _replace(sweep_cfg, kv_cache_dtype=opts.kv_dtype)
            estimate = estimate_for_config(sweep_cfg)
            if opts.gpu_vram is not None:
                estimate = _replace(
                    estimate, gpu_vram_bytes=_gib_to_bytes(opts.gpu_vram),
                )
            verdict_info = _compute_verdict(estimate)
            rows.append({
                "ctx": ctx_val,
                "verdict": verdict_info["verdict"],
                "median_mib": verdict_info["total_median_mib_per_gpu"],
                "p95_mib":    verdict_info["total_p95_mib_per_gpu"],
                "worst_mib":  verdict_info["total_worst_mib_per_gpu"],
                "budget_mib": verdict_info["budget_mib_per_gpu"],
            })

        if opts.json:
            print(json.dumps({
                "preset": opts.preset,
                "ctx_sweep": sweep_values,
                "rows": rows,
            }, indent=2, default=str))
            return 0

        print(f"sndr memory explain {opts.preset!r} --ctx-sweep")
        print("─" * 78)
        print(f"  {'ctx':>10}  {'verdict':<9}  {'median':>10}  {'p95':>10}  "
              f"{'worst':>10}  {'budget':>10}")
        for r in rows:
            sym = {"SAFE": "✓", "TIGHT": "⚠", "OOM_RISK": "✗"}.get(
                r["verdict"], "·",
            )
            print(
                f"  {r['ctx']:>10d}  {sym} {r['verdict']:<7}  "
                f"{r['median_mib']:>10d}  {r['p95_mib']:>10d}  "
                f"{r['worst_mib']:>10d}  {r['budget_mib']:>10d}"
            )
        return 0 if all(r["verdict"] != "OOM_RISK" for r in rows) else 1

    # Apply CLI overrides BEFORE estimating so the budget reflects the
    # operator's "what if" inputs rather than the config's defaults.
    if opts.ctx is not None:
        cfg.max_model_len = _parse_ctx(opts.ctx)
    if opts.seqs is not None:
        cfg.max_num_seqs = int(opts.seqs)
    if opts.kv_dtype is not None:
        cfg.kv_cache_dtype = opts.kv_dtype

    estimate = estimate_for_config(cfg)

    # GPU VRAM override — operator might run on different hardware than
    # the preset declares. Apply LAST so utilization reflects override.
    if opts.gpu_vram is not None:
        # Re-build estimate with overridden cap. Simplest: dataclass replace.
        from dataclasses import replace as _replace
        estimate = _replace(estimate, gpu_vram_bytes=_gib_to_bytes(opts.gpu_vram))
        # Recompute recommendations under new cap.
        # Keep it simple — re-call estimate_for_config would re-read disk;
        # we just re-classify here.
        util = estimate.utilization
        recs: list[str] = []
        if util > 0.95:
            recs.append(
                f"⚠ Very tight budget ({util * 100:.0f}%). "
                "Consider lowering max_model_len or enabling fp8 KV cache."
            )
        elif util > 0.85:
            recs.append(
                f"Budget at {util * 100:.0f}% — leave room for fragmentation."
            )
        elif 0 < util < 0.6:
            recs.append(
                f"Budget only {util * 100:.0f}% utilized — room to grow."
            )
        estimate = _replace(estimate, recommendations=tuple(recs))

    # Compute verdict with explicit uncertainty bands. When critical
    # components (Model weights / KV cache) come back zero with low
    # confidence the verdict becomes UNKNOWN — recommendations must be
    # demoted to "we cannot say" rather than the dangerous default
    # "you can raise max_model_len" that the utilization heuristic
    # would otherwise emit.
    verdict_info = _compute_verdict(estimate)
    if verdict_info["verdict"] == "UNKNOWN":
        from dataclasses import replace as _replace
        missing = verdict_info.get("missing_inputs") or ["model/KV inputs missing"]
        warn = (
            "Cannot make a capacity recommendation: critical components "
            f"have zero-byte low-confidence estimates ({'; '.join(missing)})."
        )
        estimate = _replace(estimate, recommendations=(warn,))

    if opts.json:
        payload = _estimate_to_dict(estimate)
        # verdict + uncertainty bands always present in JSON.
        payload.update(verdict_info)
        print(json.dumps(payload, indent=2, default=str))
        return 0

    print(render_waterfall(estimate, use_color=sys.stdout.isatty()))
    # render verdict line after the waterfall.
    v = verdict_info["verdict"]
    sym = {"SAFE": "✓", "TIGHT": "⚠", "OOM_RISK": "✗"}.get(v, "·")
    print()
    print(
        f"  {sym} verdict: {v}  "
        f"median={verdict_info['total_median_mib_per_gpu']} MiB  "
        f"p95={verdict_info['total_p95_mib_per_gpu']} MiB  "
        f"worst={verdict_info['total_worst_mib_per_gpu']} MiB  "
        f"budget={verdict_info['budget_mib_per_gpu']} MiB"
    )
    return 0


# ─── `sndr memory simulate` ─────────────────────────────────────────────


def _run_simulate(opts: argparse.Namespace) -> int:
    """Estimator without a preset — pure ctx/seqs/model knobs."""
    from vllm.sndr_core.runtime.memory_estimator import (
        read_model_shape,
        estimate_weights,
        estimate_kv_cache,
        estimate_activations,
        estimate_cuda_graph_reserve,
        estimate_marlin_scratch,
        MemoryComponent,
        MemoryEstimate,
        render_waterfall,
        lookup_gpu_vram,
    )

    if not opts.model:
        _io.fatal("--model PATH is required for `sndr memory simulate`", 2)

    shape = read_model_shape(opts.model)
    ctx = _parse_ctx(opts.ctx) if opts.ctx else 32768
    seqs = int(opts.sequences) if opts.sequences else 1
    tp = int(opts.tp_size) if opts.tp_size else 1
    kv_dtype = opts.kv_dtype or "fp16"
    gpu_vram = (_gib_to_bytes(opts.gpu_vram)
                if opts.gpu_vram else lookup_gpu_vram(opts.gpu_name))

    components: list[MemoryComponent] = [
        MemoryComponent(
            "Model weights (after TP shard)",
            estimate_weights(shape, tp_size=tp),
            notes=f"on-disk total ÷ TP={tp}",
            confidence="high",
        ),
        MemoryComponent(
            f"KV cache ({ctx // 1024}K × {seqs} seq, {kv_dtype})",
            estimate_kv_cache(
                shape,
                max_model_len=ctx,
                max_num_seqs=seqs,
                kv_dtype=kv_dtype,
                tp_size=tp,
            ),
            notes=f"per-GPU after TP={tp}",
            confidence="high",
        ),
        MemoryComponent(
            "Activations / scratch (heuristic)",
            estimate_activations(shape, max_num_batched_tokens=4096),
            notes="capped 2 GiB",
            confidence="medium",
        ),
        MemoryComponent(
            "CUDA-graph reserve",
            estimate_cuda_graph_reserve(max_num_seqs=seqs),
            notes="per-bucket capture cost",
            confidence="medium",
        ),
    ]
    marlin = estimate_marlin_scratch(shape)
    if marlin > 0:
        components.append(MemoryComponent(
            "Marlin repack scratch (peak)",
            marlin,
            notes="transient",
            confidence="medium",
        ))

    estimate = MemoryEstimate(
        preset_key="(simulate)",
        model_path=opts.model,
        gpu_count=tp,
        gpu_vram_bytes=gpu_vram,
        components=tuple(components),
        warnings=(),
        recommendations=(),
        shape=shape,
    )

    if opts.json:
        print(json.dumps(_estimate_to_dict(estimate), indent=2, default=str))
        return 0
    print(render_waterfall(estimate, use_color=sys.stdout.isatty()))
    return 0


# ─── `sndr memory doctor` ───────────────────────────────────────────────


def _run_doctor(opts: argparse.Namespace) -> int:
    """For each registered preset, compute utilization and flag risky ones."""
    from vllm.sndr_core.model_configs.registry import (
        list_keys,
        get as get_config,
    )
    from vllm.sndr_core.runtime.memory_estimator import (
        estimate_for_config,
    )

    rows: list[dict[str, Any]] = []
    for key in sorted(list_keys()):
        try:
            cfg = get_config(key)
        except Exception as e:
            rows.append({
                "preset": key,
                "error": f"{type(e).__name__}: {e}",
            })
            continue
        try:
            est = estimate_for_config(cfg)
        except Exception as e:
            rows.append({
                "preset": key,
                "error": f"{type(e).__name__}: {e}",
            })
            continue
        rows.append({
            "preset": key,
            "model_path": est.model_path,
            "gpu_count": est.gpu_count,
            "gpu_vram_bytes": est.gpu_vram_bytes,
            "total_bytes": est.total_bytes,
            "utilization": round(est.utilization, 4),
            "n_warnings": len(est.warnings),
        })

    if opts.json:
        print(json.dumps({"presets": rows}, indent=2, default=str))
        return 0

    _io.banner("memory doctor", f"{len(rows)} presets evaluated")
    cols = [
        ("Preset", "preset", 36),
        ("GPUs", "gpu_count", 5),
        ("Util%", "utilization", 7),
        ("Used / Cap", "_size", 22),
        ("Status", "_status", 12),
    ]
    header = " | ".join(name.ljust(width) for name, _, width in cols)
    print(header)
    print("-+-".join("-" * width for _, _, width in cols))
    for r in rows:
        if "error" in r:
            print(f"  {r['preset'][:36]:<36} | (error: {r['error'][:60]})")
            continue
        util_pct = r["utilization"] * 100
        status = ("OK" if util_pct < 85
                  else ("TIGHT" if util_pct < 95 else "AT-RISK"))
        size_str = (f"{r['total_bytes'] / (1 << 30):.1f}/"
                    f"{r['gpu_vram_bytes'] / (1 << 30):.0f} GiB")
        cells = [
            r["preset"][:36].ljust(36),
            str(r["gpu_count"]).ljust(5),
            f"{util_pct:5.1f}%".ljust(7),
            size_str.ljust(22),
            status.ljust(12),
        ]
        print(" | ".join(cells))
    return 0


# ─── `sndr memory report --live` ─ Phase 2 stub ─────────────────────────


def _probe_live_vram() -> dict:
    """C16 Phase 2 (UNIFIED_CONFIG plan 2026-05-09): live VRAM probe via
    nvidia-smi (no GPU runtime required in the calling Python env).

    Returns dict shape:
        {"available": bool, "n_gpus": int,
         "gpus": [{"index": i, "name": ..., "used_mib": int,
                    "free_mib": int, "total_mib": int, "util_pct": float}, ...],
         "totals": {"used_mib": int, "free_mib": int, "total_mib": int}}
    """
    import shutil
    import subprocess
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "reason": "nvidia-smi not on PATH",
                "n_gpus": 0, "gpus": [], "totals": {}}
    try:
        r = subprocess.run([
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], capture_output=True, text=True, timeout=5)
    except Exception as e:
        return {"available": False, "reason": str(e), "n_gpus": 0,
                "gpus": [], "totals": {}}
    if r.returncode != 0:
        return {"available": False, "reason": f"nvidia-smi rc={r.returncode}",
                "n_gpus": 0, "gpus": [], "totals": {}}
    gpus = []
    total_used = total_free = total_total = 0
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            idx = int(parts[0])
            name = parts[1]
            used = int(parts[2])
            free = int(parts[3])
            total = int(parts[4])
            util = float(parts[5])
        except ValueError:
            continue
        gpus.append({
            "index": idx, "name": name,
            "used_mib": used, "free_mib": free, "total_mib": total,
            "util_pct": util,
            "vram_pct": (used / total * 100) if total > 0 else 0,
        })
        total_used += used
        total_free += free
        total_total += total
    return {
        "available": True, "n_gpus": len(gpus), "gpus": gpus,
        "totals": {
            "used_mib": total_used, "free_mib": total_free,
            "total_mib": total_total,
            "vram_pct": (total_used / total_total * 100) if total_total > 0 else 0,
        },
    }


def _run_report(opts: argparse.Namespace) -> int:
    """C16 Phase 2: live VRAM probe via nvidia-smi + diff against
    estimator output if --preset given.

    `--live` flag: live probe only (no estimate diff).
    Without --live: tries to diff live vs estimator for the preset.
    """
    live = _probe_live_vram() if getattr(opts, "live", False) else None
    estimate_dict = None
    diff_lines = None

    preset = getattr(opts, "preset", None)
    if preset:
        try:
            from vllm.sndr_core.model_configs.registry import get
            from vllm.sndr_core.runtime.memory_estimator import (
                estimate_for_config,
            )
            cfg = get(preset)
            if cfg is not None:
                estimate = estimate_for_config(cfg)
                estimate_dict = {
                    "preset_key": estimate.preset_key,
                    "total_bytes": estimate.total_bytes,
                    "components": [
                        {"name": c.name, "bytes": c.bytes_,
                         "human": c.human, "confidence": c.confidence}
                        for c in estimate.components
                    ],
                }
                # Compute diff if live also available
                if live and live.get("available") and live["totals"]:
                    estimated_mib = estimate.total_bytes / (1 << 20)
                    actual_mib = live["totals"]["used_mib"]
                    diff_lines = {
                        "estimated_mib": int(estimated_mib),
                        "actual_mib": int(actual_mib),
                        "delta_mib": int(actual_mib - estimated_mib),
                        "ratio": (actual_mib / estimated_mib
                                  if estimated_mib > 0 else None),
                    }
        except Exception as e:
            _io.warn(f"estimator failed: {e}")

    out = {
        "live": live,
        "estimate": estimate_dict,
        "diff": diff_lines,
    }

    if opts.json:
        print(json.dumps(out, indent=2, default=str))
        return 0

    print("sndr memory report")
    print("─" * 60)
    if live is None:
        # Backward-compat: Phase 1 behavior (delegate to summary)
        try:
            from vllm.sndr_core.runtime.memory_metrics import (
                genesis_memory_summary,
            )
            report = genesis_memory_summary()
            print(json.dumps(report, indent=2, default=str))
            return 0
        except Exception as e:
            _io.fatal(
                f"memory_metrics unavailable: {type(e).__name__}: {e}", 1
            )
            return 1

    if not live.get("available"):
        print(f"  Live probe: UNAVAILABLE — {live.get('reason')}")
        return 1
    print(f"  Live probe ({live['n_gpus']} GPU(s)):")
    for g in live["gpus"]:
        print(f"    GPU {g['index']}: {g['name']}")
        print(f"      VRAM used: {g['used_mib']:6d} / {g['total_mib']:6d} MiB "
              f"({g['vram_pct']:.1f}%)")
        print(f"      Compute util: {g['util_pct']:.0f}%")
    if live["totals"]:
        t = live["totals"]
        print(f"    TOTAL: {t['used_mib']:6d} / {t['total_mib']:6d} MiB "
              f"({t['vram_pct']:.1f}%)")

    if estimate_dict is not None:
        print()
        print(f"  Estimate for preset {preset!r}:")
        print(f"    Total: {estimate_dict['total_bytes'] // (1 << 20)} MiB")
        for c in estimate_dict["components"]:
            print(f"      {c['name']:<32s}  {c['human']:>12s}  [{c['confidence']}]")

    if diff_lines:
        print()
        print("  Live vs Estimate diff:")
        print(f"    Estimated: {diff_lines['estimated_mib']:6d} MiB")
        print(f"    Actual:    {diff_lines['actual_mib']:6d} MiB")
        print(f"    Δ:         {diff_lines['delta_mib']:+6d} MiB")
        if diff_lines.get("ratio"):
            print(f"    Ratio:     {diff_lines['ratio']:.2f}x")
    return 0


# ─── argparse plumbing ──────────────────────────────────────────────────


def add_argparser(subparsers: Any) -> None:
    parent = subparsers.add_parser(
        "memory",
        help="VRAM budget estimator + live memory diagnostics.",
        description=(
            "`sndr memory` — preflight VRAM accounting. Phase 1 ships "
            "`explain`, `simulate`, and `doctor` (static estimators); "
            "`report --live` wraps the runtime memory_metrics collector. "
            "Phase 2 will integrate live probe-vs-estimate diff."
        ),
    )
    sub = parent.add_subparsers(dest="memory_cmd", title="Subcommands",
                                metavar="{explain,simulate,doctor,report}")

    # explain
    p_explain = sub.add_parser(
        "explain",
        help="Static VRAM breakdown for one preset.",
        description=(
            "Compute per-component VRAM budget for `<preset>` "
            "(weights + KV + activations + CUDA-graph + Marlin). "
            "CLI flags override preset defaults — useful for "
            "`what-if max_model_len=128k` analyses without editing YAML."
        ),
    )
    p_explain.add_argument("preset", help="model_config key.")
    p_explain.add_argument(
        "--gpu-vram", default=None,
        help="Override GPU capacity (e.g. 24, 48, 96). Defaults to "
             "preset's hardware spec or 24 GiB.",
    )
    p_explain.add_argument(
        "--ctx", default=None,
        help="Override max_model_len (e.g. 128k, 256K, 32768).",
    )
    p_explain.add_argument(
        "--ctx-sweep", default=None,
        help="Phase 4.7 MVP: comma-separated context sizes to sweep "
             "(e.g. `--ctx-sweep 4k,16k,64k,128k,256k`). Emits one verdict "
             "row per size; exits 1 if any row is OOM_RISK. Mutually "
             "exclusive with single-shot --ctx.",
    )
    p_explain.add_argument(
        "--seqs", type=int, default=None,
        help="Override max_num_seqs.",
    )
    p_explain.add_argument(
        "--kv-dtype", default=None,
        help="Override kv_cache_dtype (fp8_e5m2 / fp16 / bf16 / auto).",
    )
    p_explain.add_argument("--json", action="store_true")
    p_explain.set_defaults(func=_run_explain)

    # simulate
    p_sim = sub.add_parser(
        "simulate",
        help="Estimator without a preset — pure ctx/seqs/model.",
    )
    p_sim.add_argument("--model", required=False,
                       help="Model path (must contain config.json).")
    p_sim.add_argument("--ctx", default=None,
                       help="Context length (e.g. 128k).")
    p_sim.add_argument("--sequences", default=None,
                       help="Concurrent sequences.")
    p_sim.add_argument("--tp-size", default=None,
                       help="Tensor-parallel size.")
    p_sim.add_argument("--kv-dtype", default=None,
                       help="KV cache dtype (default: fp16).")
    p_sim.add_argument("--gpu-vram", default=None,
                       help="GPU capacity in GiB.")
    p_sim.add_argument("--gpu-name", default=None,
                       help="GPU model name (used for vram lookup).")
    p_sim.add_argument("--json", action="store_true")
    p_sim.set_defaults(func=_run_simulate)

    # doctor
    p_doc = sub.add_parser(
        "doctor",
        help="Probe every preset's utilization; flag at-risk configs.",
    )
    p_doc.add_argument("--json", action="store_true")
    p_doc.set_defaults(func=_run_doctor)

    # report
    p_rep = sub.add_parser(
        "report",
        help="Live VRAM probe (C16 Phase 2) + optional preset estimate diff.",
    )
    p_rep.add_argument("--live", action="store_true",
                       help="C16 Phase 2: real nvidia-smi probe instead of "
                            "the legacy genesis_memory_summary() collector.")
    p_rep.add_argument("--preset", default=None,
                       help="Optional preset key — when provided alongside "
                            "--live, prints the live-vs-estimate diff.")
    p_rep.add_argument("--json", action="store_true")
    p_rep.set_defaults(func=_run_report)

    parent.set_defaults(func=lambda _ns: parent.print_help() or 0)


__all__ = ["add_argparser"]
