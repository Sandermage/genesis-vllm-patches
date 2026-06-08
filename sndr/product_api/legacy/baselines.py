# SPDX-License-Identifier: Apache-2.0
"""Quality / benchmark baselines + regression diff for the GUI.

Saves a benchmark or eval result as a named, trusted *baseline*, then diffs a
new result against it scenario-by-scenario: per-metric delta, direction-aware
regression flags, and a CI-style exit code (non-zero when anything regressed).
Operators stop eyeballing two JSON blobs — they see "tps −10% on code: REGRESS".

Metric direction is inferred from the name (throughput/accuracy = higher-better,
latency = lower-better) so the same engine works for bench numbers and eval
scores. Results are stored operator-local under ``SNDR_HOME/gui/baselines``.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Optional

# Lower-is-better metrics (latency-like); everything else is higher-is-better.
_LOWER_BETTER = re.compile(r"(ttft|tpot|latency|_ms\b|ms$|p\d{2}|tail|error|loss)", re.IGNORECASE)


def _lower_is_better(metric: str) -> bool:
    return bool(_LOWER_BETTER.search(metric))


def _store_dir() -> Path:
    from sndr.engines.vllm.locations.project_paths import install_root

    return install_root() / "gui" / "baselines"


def _normalize(result: dict[str, Any]) -> dict[str, Any]:
    """Accept either ``{scenarios:[{name,metrics}]}`` or flat ``{metrics}``."""
    if isinstance(result.get("scenarios"), list):
        return result
    return {"label": result.get("label", ""), "scenarios": [{"name": "overall", "metrics": result.get("metrics", {})}]}


# ── store ───────────────────────────────────────────────────────────────────


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:60] or "baseline"


# Baseline ids are only ever produced by ``_slug`` (+ a numeric timestamp), so a
# well-formed id is purely lowercase-alnum + hyphen. Reject anything else before
# it reaches the filesystem — a caller-supplied id like ``../../secrets`` must
# never resolve a path outside the store (path-traversal guard).
_VALID_ID = re.compile(r"^[a-z0-9-]{1,80}$")


def _safe_id(baseline_id: str) -> bool:
    return bool(_VALID_ID.match(baseline_id or ""))


def save_baseline(result: dict[str, Any], *, label: Optional[str] = None, stamp: Optional[int] = None) -> dict[str, Any]:
    label = (label or result.get("label") or "baseline").strip()
    ts = int(stamp if stamp is not None else time.time())
    bid = f"{_slug(label)}-{ts}"
    record = {"id": bid, "label": label, "saved_at": ts, "result": _normalize(result)}
    d = _store_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{bid}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {"id": bid, "label": label, "saved_at": ts}


def list_baselines() -> list[dict[str, Any]]:
    d = _store_dir()
    if not d.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(d.glob("*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
            out.append({"id": rec["id"], "label": rec.get("label", ""), "saved_at": rec.get("saved_at", 0),
                        "scenarios": [s.get("name") for s in rec.get("result", {}).get("scenarios", [])]})
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    return sorted(out, key=lambda b: b.get("saved_at", 0), reverse=True)


def get_baseline(baseline_id: str) -> Optional[dict[str, Any]]:
    if not _safe_id(baseline_id):
        return None
    path = _store_dir() / f"{baseline_id}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def delete_baseline(baseline_id: str) -> bool:
    if not _safe_id(baseline_id):
        return False
    path = _store_dir() / f"{baseline_id}.json"
    if not path.is_file():
        return False
    try:
        path.unlink()
        return True
    except OSError:
        return False


# ── trend ────────────────────────────────────────────────────────────────────


def _metric_value(scenarios: list[dict[str, Any]], metric: str, scenario: Optional[str]) -> Optional[float]:
    """Mean of ``metric`` across matching scenarios (or one named scenario)."""
    vals: list[float] = []
    for sc in scenarios:
        if scenario and sc.get("name") != scenario:
            continue
        mv = (sc.get("metrics") or {}).get(metric)
        if isinstance(mv, (int, float)):
            vals.append(float(mv))
    return round(sum(vals) / len(vals), 4) if vals else None


def trend(metric: Optional[str] = None, *, scenario: Optional[str] = None) -> dict[str, Any]:
    """Time-ordered series of one metric across all saved baselines.

    Turns the baseline store (named reference points) into a regression trend:
    how a metric evolved run-over-run. Picks a throughput-like metric by default.
    """
    d = _store_dir()
    recs: list[dict[str, Any]] = []
    metrics_seen: set[str] = set()
    if d.is_dir():
        for path in sorted(d.glob("*.json")):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for sc in rec.get("result", {}).get("scenarios", []):
                metrics_seen.update((sc.get("metrics") or {}).keys())
            recs.append(rec)
    recs.sort(key=lambda r: r.get("saved_at", 0))

    available = sorted(metrics_seen)
    if not metric:
        metric = next((m for m in available if re.search(r"tps|throughput|tok", m, re.I)), available[0] if available else "")

    points: list[dict[str, Any]] = []
    for rec in recs:
        value = _metric_value(rec.get("result", {}).get("scenarios", []), metric, scenario)
        if value is not None:
            points.append({
                "saved_at": rec.get("saved_at", 0),
                "label": rec.get("label") or rec.get("id", ""),
                "value": value,
            })
    return {
        "metric": metric,
        "scenario": scenario,
        "points": points,
        "lower_is_better": _lower_is_better(metric) if metric else False,
        "metrics_available": available,
    }


# ── diff ─────────────────────────────────────────────────────────────────────


def diff_results(current: dict[str, Any], baseline: dict[str, Any], *, threshold_pct: float = 5.0) -> dict[str, Any]:
    """Scenario-by-scenario, metric-by-metric regression diff."""
    cur = {s["name"]: s.get("metrics", {}) for s in _normalize(current)["scenarios"]}
    base = {s["name"]: s.get("metrics", {}) for s in _normalize(baseline)["scenarios"]}

    scenarios: list[dict[str, Any]] = []
    regressed = improved = 0
    for name in sorted(set(cur) | set(base)):
        if name not in cur:
            scenarios.append({"name": name, "status": "removed", "metrics": []})
            continue
        if name not in base:
            scenarios.append({"name": name, "status": "added", "metrics": []})
            continue
        rows: list[dict[str, Any]] = []
        for metric in sorted(set(cur[name]) | set(base[name])):
            cv, bv = cur[name].get(metric), base[name].get(metric)
            if not isinstance(cv, (int, float)) or not isinstance(bv, (int, float)):
                continue
            delta = cv - bv
            pct = (delta / bv * 100.0) if bv else 0.0
            lower_better = _lower_is_better(metric)
            worse = (delta > 0) if lower_better else (delta < 0)
            is_reg = worse and abs(pct) >= threshold_pct
            is_imp = (not worse) and abs(pct) >= threshold_pct
            regressed += int(is_reg)
            improved += int(is_imp)
            rows.append({
                "metric": metric, "current": round(cv, 4), "baseline": round(bv, 4),
                "delta": round(delta, 4), "pct": round(pct, 2),
                "lower_is_better": lower_better, "regression": is_reg, "improvement": is_imp,
            })
        scenarios.append({"name": name, "status": "compared", "metrics": rows})

    has_reg = regressed > 0
    return {
        "threshold_pct": threshold_pct,
        "scenarios": scenarios,
        "regressed": regressed,
        "improved": improved,
        "has_regression": has_reg,
        "exit_code": 3 if has_reg else 0,
        "verdict": "REGRESSION" if has_reg else ("IMPROVED" if improved else "STABLE"),
    }


__all__ = [
    "delete_baseline", "diff_results", "get_baseline", "list_baselines", "save_baseline",
]
