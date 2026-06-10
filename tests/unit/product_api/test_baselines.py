# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the quality/bench baseline store + regression diff."""
from __future__ import annotations

from sndr.product_api.legacy import baselines as bl


def _result(label, tps, ttft, tool=0.95):
    return {"label": label, "scenarios": [
        {"name": "code", "metrics": {"tps": tps, "ttft_ms": ttft, "tool_call_success": tool}},
    ]}


def test_save_list_get_delete(monkeypatch, tmp_path):
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    rec = bl.save_baseline(_result("prod-27b @ dev209", 120.0, 110.0))
    assert rec["id"] and rec["label"] == "prod-27b @ dev209"
    assert any(b["id"] == rec["id"] for b in bl.list_baselines())
    assert bl.get_baseline(rec["id"])["result"]["scenarios"][0]["metrics"]["tps"] == 120.0
    assert bl.delete_baseline(rec["id"]) is True
    assert bl.get_baseline(rec["id"]) is None


def test_trend_orders_points_and_picks_default_metric(monkeypatch, tmp_path):
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    # Saved out of chronological order — trend must sort by saved_at.
    bl.save_baseline(_result("run2", 124.0, 105.0), label="run2", stamp=2000)
    bl.save_baseline(_result("run1", 118.0, 112.0), label="run1", stamp=1000)
    bl.save_baseline(_result("run3", 131.0, 101.0), label="run3", stamp=3000)

    tr = bl.trend("tps")
    assert tr["metric"] == "tps" and tr["lower_is_better"] is False
    assert [p["value"] for p in tr["points"]] == [118.0, 124.0, 131.0]
    assert [p["label"] for p in tr["points"]] == ["run1", "run2", "run3"]
    assert "tps" in tr["metrics_available"] and "ttft_ms" in tr["metrics_available"]

    # No metric given → auto-pick a throughput-like one (tps).
    assert bl.trend()["metric"] == "tps"
    # Latency metric is lower-is-better.
    assert bl.trend("ttft_ms")["lower_is_better"] is True


def test_trend_empty_when_no_store(monkeypatch, tmp_path):
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    tr = bl.trend("tps")
    assert tr["points"] == [] and tr["metrics_available"] == []


def test_diff_flags_regressions_by_metric_direction():
    base = _result("base", tps=120.0, ttft=100.0, tool=0.95)
    # tps DOWN (higher-is-better → regression), ttft UP (lower-is-better →
    # regression), tool_call_success steady.
    cur = _result("cur", tps=108.0, ttft=130.0, tool=0.95)
    d = bl.diff_results(cur, base, threshold_pct=5.0)
    metrics = {m["metric"]: m for m in d["scenarios"][0]["metrics"]}
    assert metrics["tps"]["regression"] is True and metrics["tps"]["delta"] < 0
    assert metrics["ttft_ms"]["regression"] is True and metrics["ttft_ms"]["delta"] > 0
    assert metrics["tool_call_success"]["regression"] is False
    assert d["has_regression"] is True
    assert d["exit_code"] == 3  # CI gate: non-zero on any regression


def test_diff_within_threshold_is_clean():
    base = _result("base", tps=120.0, ttft=100.0)
    cur = _result("cur", tps=118.0, ttft=103.0)  # ~1.7% / 3% — under 5%
    d = bl.diff_results(cur, base, threshold_pct=5.0)
    assert d["has_regression"] is False and d["exit_code"] == 0
    assert d["improved"] >= 0 and d["regressed"] == 0


def test_diff_handles_new_and_missing_scenarios():
    base = {"label": "b", "scenarios": [{"name": "code", "metrics": {"tps": 100.0}}]}
    cur = {"label": "c", "scenarios": [{"name": "narr", "metrics": {"tps": 90.0}}]}
    d = bl.diff_results(cur, base, threshold_pct=5.0)
    # Scenarios present on only one side are reported, not crashed on.
    names = {s["name"]: s for s in d["scenarios"]}
    assert "code" in names and "narr" in names
    assert names["code"]["status"] == "removed" and names["narr"]["status"] == "added"


def test_flat_result_is_normalized():
    flat = {"label": "x", "metrics": {"tps": 100.0}}
    d = bl.diff_results(flat, flat, threshold_pct=5.0)
    assert d["scenarios"][0]["name"] == "overall"
    assert d["has_regression"] is False


def test_baseline_id_path_traversal_is_rejected(monkeypatch, tmp_path):
    """A caller-supplied id must never resolve a path outside the store."""
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    # Plant a JSON file one level above the store that an attacker would target.
    secret = tmp_path / "gui" / "secret.json"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text('{"result": {"leaked": true}}', encoding="utf-8")
    for evil in ("../secret", "../../etc/passwd", "..%2fsecret", "a/b", "UPPER", "with space"):
        assert bl.get_baseline(evil) is None, f"traversal not blocked: {evil!r}"
        assert bl.delete_baseline(evil) is False, f"traversal delete not blocked: {evil!r}"
    # The planted file must still be there — nothing was read or unlinked.
    assert secret.is_file()


def test_valid_baseline_id_still_resolves(monkeypatch, tmp_path):
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    rec = bl.save_baseline(_result("ok", 1.0, 1.0))
    assert bl._safe_id(rec["id"])  # the ids we mint always pass the guard
    assert bl.get_baseline(rec["id"]) is not None
