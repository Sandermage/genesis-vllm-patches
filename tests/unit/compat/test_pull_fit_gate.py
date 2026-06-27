# SPDX-License-Identifier: Apache-2.0
"""B2 — `sndr pull --dry-run` fit gate.

Pre-download envelope fit verdict ("will model X fit my card before I download
N GB?"), driven by each registry entry's declared min_vram_gb_per_rank floor.
"""
from __future__ import annotations

from sndr.compat.models import pull
from sndr.compat.models.registry import get_model


# ─── _fit_verdict_for_entry (pure) ───────────────────────────────────────────

def test_fit_pass_with_headroom():
    entry = get_model("qwen3_6_27b_int4_autoround")  # {1: 24.0, 2: 14.0}
    verdict, detail = pull._fit_verdict_for_entry(entry, vram_gib=24.0, tp=2)
    assert verdict == "PASS"
    assert "14.0 GiB/rank at TP2" in detail


def test_fit_fail_below_floor():
    entry = get_model("qwen3_6_35b_a3b_fp8")  # {1: 48.0, 2: 24.0}
    verdict, detail = pull._fit_verdict_for_entry(entry, vram_gib=24.0, tp=1)
    assert verdict == "FAIL"
    assert "48.0 GiB/rank at TP1" in detail
    assert "OOM at boot" in detail


def test_fit_tight_at_floor():
    entry = get_model("qwen3_6_35b_a3b_fp8")  # TP2 floor = 24.0
    verdict, detail = pull._fit_verdict_for_entry(entry, vram_gib=24.0, tp=2)
    assert verdict == "TIGHT"
    assert "little headroom" in detail


def test_fit_falls_back_to_smallest_tp_when_requested_tp_absent():
    entry = get_model("qwen3_6_35b_a3b_fp8")  # has TP1, TP2; ask for TP4
    verdict, detail = pull._fit_verdict_for_entry(entry, vram_gib=80.0, tp=4)
    # TP4 not declared -> uses the smallest declared TP (TP1, floor 48).
    assert verdict == "PASS"
    assert "TP1" in detail


# ─── _resolve_card_vram_gib ──────────────────────────────────────────────────

def test_resolve_card_explicit():
    gib, src = pull._resolve_card_vram_gib(card="24", fake_gpus=None)
    assert gib == 24.0 and src == "card:24GB"


def test_resolve_card_invalid():
    gib, src = pull._resolve_card_vram_gib(card="not-a-number", fake_gpus=None)
    assert gib is None and "invalid" in src


def test_resolve_card_fake_gpus():
    gib, src = pull._resolve_card_vram_gib(
        card=None, fake_gpus="RTX A5000:24564:8.6")
    assert gib is not None and abs(gib - 24564 / 1024.0) < 0.01
    assert src == "fake"


# ─── dry-run integration (via main) ──────────────────────────────────────────

def test_dry_run_prints_fit_pass(capsys):
    rc = pull.main(["qwen3_6_27b_int4_autoround", "--dry-run", "--card", "24"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fit verdict:" in out
    assert "PASS" in out
    assert "run `sndr kv-calc`" in out


def test_dry_run_prints_fit_fail(capsys):
    rc = pull.main(["qwen3_6_35b_a3b_fp8", "--dry-run", "--card", "24",
                    "--tp", "1"])
    # dry-run itself returns 0 (it is a plan, not the download); the FAIL is in
    # the verdict text so the operator sees it BEFORE downloading 38 GB.
    assert rc == 0
    out = capsys.readouterr().out
    assert "✗ FAIL" in out
    assert "below the per-rank floor" in out


def test_dry_run_fit_unknown_without_card(capsys, monkeypatch):
    """No --card and no detectable rig -> UNKNOWN verdict (never crashes the
    dry-run)."""
    monkeypatch.setattr(
        pull, "_resolve_card_vram_gib",
        lambda card, fake_gpus: (None, "no card detected"))
    rc = pull.main(["qwen3_6_27b_int4_autoround", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "? UNKNOWN" in out


def test_dry_run_proceeds_despite_insufficient_disk(capsys, monkeypatch):
    """Regression: a dry-run on a small-disk host (or without a gated-repo
    token) must STILL reach the fit verdict and return 0 — the disk/token
    preflight ✗ is informational in plan mode, not an abort. Locks the CI flake
    where a runner with 19 GB free aborted the 35B dry-run at the disk check
    (return 3) before the operator ever saw the fit answer."""
    monkeypatch.setattr(
        pull, "_check_disk_space",
        lambda target_dir, needed_gb, headroom=1.2: (
            False, "insufficient disk: 19.1 GB free, need ~45.6 GB"))
    monkeypatch.setattr(
        pull, "_check_hf_token_for_gated",
        lambda entry: (False, "no HF token for gated repo"))
    rc = pull.main(["qwen3_6_35b_a3b_fp8", "--dry-run", "--card", "24",
                    "--tp", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "disk:    ✗" in out          # the warning is surfaced ...
    assert "fit verdict:" in out        # ... but the verdict is still reached
    assert "✗ FAIL" in out


def test_dry_run_fake_gpus_fit(capsys):
    rc = pull.main(["qwen3_6_27b_int4_autoround", "--dry-run",
                    "--fake-gpus", "RTX A5000:24564:8.6"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fit verdict:" in out
    assert "(fake," in out


def test_parser_accepts_card_and_fake_gpus():
    args = pull._parse_args(["qwen3_6_27b_int4_autoround", "--dry-run",
                             "--card", "24"])
    assert args.card == "24" and args.dry_run is True
    args = pull._parse_args(["qwen3_6_27b_int4_autoround", "--dry-run",
                             "--fake-gpus", "RTX A5000:24564:8.6"])
    assert args.fake_gpus == "RTX A5000:24564:8.6"
