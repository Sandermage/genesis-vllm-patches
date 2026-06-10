# SPDX-License-Identifier: Apache-2.0
"""Y11 + Y12 (UNIFIED_CONFIG plan 2026-05-09) tests.

Y11 — UpstreamPinPolicy: per-config required/allowed/blocked vLLM pins.
Y12 — OverridesPolicy: declared safe env-var overrides + numeric ranges.
"""
from __future__ import annotations

import pytest

from sndr.model_configs.schema import (
    UpstreamPinPolicy, OverridesPolicy, ModelConfig, HardwareSpec,
    DockerConfig, SchemaError, dump_yaml, load_yaml,
)


# ─── UpstreamPinPolicy

def test_upstream_default_allows_anything():
    p = UpstreamPinPolicy()
    p.validate()
    assert p.check("any-pin") is None
    assert p.check(None) is None


def test_upstream_blocked_pin_rejected():
    p = UpstreamPinPolicy(blocked_pins=["0.20.2rc1.dev99+gbad"],
                          notes="dev99 has KV-eviction crash on hybrid GDN")
    p.validate()
    msg = p.check("0.20.2rc1.dev99+gbad")
    assert msg is not None
    assert "blocked_pins" in msg
    assert "dev99 has KV-eviction crash" in msg


def test_upstream_required_pin_must_match():
    p = UpstreamPinPolicy(required_pin="0.20.2rc1.dev93+g51f22dcfd")
    p.validate()
    assert p.check("0.20.2rc1.dev93+g51f22dcfd") is None
    msg = p.check("0.20.2rc1.dev9+g01d4d1ad3")
    assert msg is not None and "required_pin" in msg


def test_upstream_allowed_pins_membership():
    p = UpstreamPinPolicy(allowed_pins=["a", "b", "c"])
    p.validate()
    assert p.check("a") is None
    msg = p.check("d")
    assert msg is not None and "allowed_pins" in msg


def test_upstream_required_in_blocked_rejected_at_validate():
    with pytest.raises(SchemaError, match="blocked_pins"):
        UpstreamPinPolicy(required_pin="x", blocked_pins=["x"]).validate()


def test_upstream_overlap_allowed_blocked_rejected():
    with pytest.raises(SchemaError, match="both allowed_pins and blocked_pins"):
        UpstreamPinPolicy(
            allowed_pins=["a", "b"],
            blocked_pins=["b", "c"],
        ).validate()


# ─── OverridesPolicy

def test_overrides_default_rejects_everything():
    o = OverridesPolicy()
    o.validate()
    msg = o.check("ANY_KEY", "any_value")
    assert msg is not None and "allow_env is empty" in msg


def test_overrides_allow_env_membership():
    o = OverridesPolicy(allow_env=["GENESIS_P67_NUM_KV_SPLITS"])
    o.validate()
    assert o.check("GENESIS_P67_NUM_KV_SPLITS", "32") is None
    msg = o.check("OTHER_KEY", "x")
    assert msg is not None and "not in allow_env" in msg


def test_overrides_safe_range_accepts_in_range():
    o = OverridesPolicy(
        allow_env=["GENESIS_P67_NUM_KV_SPLITS"],
        safe_ranges={"GENESIS_P67_NUM_KV_SPLITS": ["16", "64"]},
    )
    o.validate()
    assert o.check("GENESIS_P67_NUM_KV_SPLITS", "32") is None
    assert o.check("GENESIS_P67_NUM_KV_SPLITS", "64") is None
    assert o.check("GENESIS_P67_NUM_KV_SPLITS", "16") is None


def test_overrides_safe_range_rejects_out_of_range():
    o = OverridesPolicy(
        allow_env=["GENESIS_PN16_TOOL_THINK_BUDGET"],
        safe_ranges={"GENESIS_PN16_TOOL_THINK_BUDGET": ["50", "500"]},
    )
    o.validate()
    msg = o.check("GENESIS_PN16_TOOL_THINK_BUDGET", "10")
    assert msg is not None and "outside safe range" in msg
    msg = o.check("GENESIS_PN16_TOOL_THINK_BUDGET", "1000")
    assert msg is not None and "outside safe range" in msg


def test_overrides_safe_range_rejects_non_numeric():
    o = OverridesPolicy(
        allow_env=["KEY"],
        safe_ranges={"KEY": ["1", "10"]},
    )
    msg = o.check("KEY", "not_a_number")
    assert msg is not None and "not numeric" in msg


def test_overrides_validate_rejects_bad_range_shape():
    with pytest.raises(SchemaError, match=r"\[min, max\] 2-list"):
        OverridesPolicy(
            allow_env=["KEY"], safe_ranges={"KEY": ["1"]}  # only one bound
        ).validate()


def test_overrides_validate_rejects_non_numeric_bounds():
    with pytest.raises(SchemaError, match="not numeric"):
        OverridesPolicy(
            allow_env=["KEY"], safe_ranges={"KEY": ["a", "b"]}
        ).validate()


# ─── YAML round-trip

def _cfg_with_blocks(upstream=None, overrides=None) -> ModelConfig:
    return ModelConfig(
        key="test-y11-y12",
        title="Y11/Y12 round-trip config",
        description="Minimal docker config exercising upstream + overrides.",
        schema_version=1,
        maintainer="sandermage",
        model_path="/models/dummy",
        hardware=HardwareSpec(
            gpu_match_keys=["rtx a5000"], n_gpus=1,
            min_vram_per_gpu_mib=1,
        ),
        docker=DockerConfig(image="i", container_name="c", port=8000),
        upstream=upstream,
        overrides=overrides,
    )


def test_upstream_overrides_yaml_roundtrip():
    cfg = _cfg_with_blocks(
        upstream=UpstreamPinPolicy(
            required_pin="0.20.2rc1.dev93+g51f22dcfd",
            allowed_pins=["0.20.2rc1.dev93+g51f22dcfd"],
            blocked_pins=["0.20.2rc1.dev99+gbroken"],
            notes="dev99 has hybrid GDN crash",
        ),
        overrides=OverridesPolicy(
            allow_env=["GENESIS_P67_NUM_KV_SPLITS",
                       "GENESIS_PN16_TOOL_THINK_BUDGET"],
            safe_ranges={
                "GENESIS_P67_NUM_KV_SPLITS": ["16", "64"],
                "GENESIS_PN16_TOOL_THINK_BUDGET": ["50", "500"],
            },
            notes="Hot-tunable knobs validated by sweeps",
        ),
    )
    yaml_str = dump_yaml(cfg)
    cfg2 = load_yaml(yaml_str)
    assert cfg2.upstream is not None
    assert cfg2.upstream.required_pin == "0.20.2rc1.dev93+g51f22dcfd"
    assert cfg2.upstream.blocked_pins == ["0.20.2rc1.dev99+gbroken"]
    assert cfg2.overrides is not None
    assert "GENESIS_P67_NUM_KV_SPLITS" in cfg2.overrides.allow_env
    assert cfg2.overrides.safe_ranges["GENESIS_P67_NUM_KV_SPLITS"] == ["16", "64"]
