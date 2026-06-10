# SPDX-License-Identifier: Apache-2.0
"""Phase B compose passthrough — ModelDef.patches_attribution must
survive the V2 → V1 collapse done by `compose()`.

If compose() drops the attribution map, the resolver layer in
`patch_plan.py` has nothing to read at runtime and every patch
falls back to role='unknown'. This test catches that drift early.
"""
from __future__ import annotations

import pytest

from sndr.model_configs.compose import compose
from sndr.model_configs.schema import (
    HardwareSpec,
    PatchAttribution,
)
from sndr.model_configs.schema_v2 import (
    HardwareDef,
    HardwareSizing,
    ModelCapabilities,
    ModelDef,
    RuntimeBlock,
    RuntimeDockerBlock,
)


def _stub_hardware() -> HardwareDef:
    return HardwareDef(
        schema_version=2, kind="hardware", id="stub-hw",
        title="t", maintainer="x",
        hardware=HardwareSpec(
            gpu_match_keys=["test-gpu"], n_gpus=1, min_vram_per_gpu_mib=8000,
        ),
        runtime=RuntimeBlock(
            default="docker", supported=["docker"],
            docker=RuntimeDockerBlock(image="vllm:stub"),
        ),
        sizing=HardwareSizing(
            max_model_len=4096, max_num_seqs=1, max_num_batched_tokens=2048,
            gpu_memory_utilization=0.9,
        ),
    )


def _stub_model(patches=None, patches_attribution=None) -> ModelDef:
    return ModelDef(
        schema_version=2, kind="model", id="stub-model",
        title="t", maintainer="x", last_validated="2026-05-16",
        license="apache-2.0", model_path="/m",
        capabilities=ModelCapabilities(attention_arch="dense"),
        patches=patches or {},
        patches_attribution=patches_attribution or {},
    )


class TestComposePassesAttribution:
    def test_empty_attribution_survives(self):
        cfg = compose(_stub_model(), _stub_hardware())
        assert cfg.patches_attribution == {}

    def test_attribution_entries_preserved(self):
        m = _stub_model(
            patches={"GENESIS_ENABLE_PN204": "0"},
            patches_attribution={
                "PN204": PatchAttribution(
                    role="optional_perf",
                    bench_evidence="dev371 35B conc=8: 689 TPS",
                ),
                "PN134": PatchAttribution(
                    role="suspected_regression",
                    note="-25% TPS bench regressor",
                ),
            },
        )
        cfg = compose(m, _stub_hardware())
        assert "PN204" in cfg.patches_attribution
        assert "PN134" in cfg.patches_attribution
        assert cfg.patches_attribution["PN204"].role == "optional_perf"
        assert cfg.patches_attribution["PN134"].role == "suspected_regression"

    def test_attribution_is_copy_not_alias(self):
        """compose() should not let downstream mutations corrupt the
        original ModelDef.patches_attribution."""
        m = _stub_model(patches_attribution={
            "PN17": PatchAttribution(role="defensive"),
        })
        cfg = compose(m, _stub_hardware())
        cfg.patches_attribution["PN999"] = PatchAttribution(role="unknown")
        assert "PN999" not in m.patches_attribution


class TestComposePreservesAttributionAcrossProfile:
    """Profile patches_delta can disable/enable env flags. The
    attribution map stays as the model authored it — profiles aren't
    expected to override attribution in Phase B (deferred to a
    later phase if real demand surfaces)."""

    def test_profile_disable_does_not_drop_attribution(self):
        from sndr.model_configs.schema_v2 import (
            PatchesDelta, ProfileDef,
        )

        m = _stub_model(
            patches={"GENESIS_ENABLE_PN204": "1"},
            patches_attribution={
                "PN204": PatchAttribution(
                    role="optional_perf",
                    bench_evidence="dev371 35B conc=8: 689 TPS",
                ),
            },
        )
        profile = ProfileDef(
            schema_version=2, kind="profile", id="stub-profile",
            parent_model="stub-model", maintainer="x",
            patches_delta=PatchesDelta(
                disable=["GENESIS_ENABLE_PN204"],
            ),
        )
        cfg = compose(m, _stub_hardware(), profile=profile)
        # PN204 disabled in genesis_env (value="0" per disable semantics)
        # — but attribution metadata survives so the resolver can still
        # surface "this patch was intentionally disabled and here's why".
        assert "PN204" in cfg.patches_attribution
        assert cfg.patches_attribution["PN204"].role == "optional_perf"
