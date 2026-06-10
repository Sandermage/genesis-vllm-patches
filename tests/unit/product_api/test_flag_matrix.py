# SPDX-License-Identifier: Apache-2.0
"""Tests for the env-flag matrix (registry catalogue + live overlay)."""
from sndr.product_api.legacy import flag_matrix as fm


def test_build_matrix_from_registry():
    out = fm.build_matrix()
    assert out["has_live"] is False
    flags = out["flags"]
    assert len(flags) > 100                       # the registry has 240+ flags
    assert all(f["env_flag"] for f in flags)
    # sorted by family then flag
    fams = [f["family"] or "" for f in flags]
    assert fams == sorted(fams)
    c = out["counts"]
    assert c["total"] == len(flags)
    assert c["default_on"] + c["default_off"] == c["total"]


def test_live_overlay_marks_drift():
    base = fm.build_matrix()
    # Take one default-on and one default-off flag from the real registry.
    on = next(f["env_flag"] for f in base["flags"] if f["default_on"])
    off = next(f["env_flag"] for f in base["flags"] if not f["default_on"])
    # Live engine has the default-off one ON, and the default-on one OFF.
    live = fm.build_matrix(live_flags={off})
    assert live["has_live"] is True
    by_flag = {f["env_flag"]: f for f in live["flags"]}
    assert by_flag[off]["live_on"] is True and by_flag[off]["drift"] == "extra"
    assert by_flag[on]["live_on"] is False and by_flag[on]["drift"] == "missing"
    assert live["counts"]["missing"] >= 1 and live["counts"]["extra"] >= 1


def test_live_flags_from_inspect():
    inspect = {"Config": {"Env": ["GENESIS_ENABLE_P82=1", "FOO=bar", "GENESIS_ENABLE_PN95=0"]}}
    flags = fm.live_flags_from_inspect(inspect)
    assert "GENESIS_ENABLE_P82" in flags          # truthy → live
    assert "GENESIS_ENABLE_PN95" not in flags      # "0" → not live
    assert "FOO" not in flags
