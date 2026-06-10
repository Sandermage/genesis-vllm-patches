# SPDX-License-Identifier: Apache-2.0
"""Tests for ModelDef.override_generation_config — Phase 2026-05-30 closure
of the §2.10 Qwen-sampling-defaults gap on the V2 compose codepath.

Contract enforced:

  1. Field defaults to None (additive, backward-compatible).
  2. dict with primitive values (int/float/str/bool/list) accepted.
  3. Non-dict rejected with SchemaError.
  4. Empty key inside dict rejected.
  5. Nested dict inside the value rejected (keep flat for CLI JSON).
  6. compose() includes `--override-generation-config` flag in
     vllm_extra_args when ModelDef declares it.
  7. compose() omits the flag when ModelDef.override_generation_config
     is None.
  8. JSON value is canonicalised: keys sorted, no whitespace in separators.
"""
from __future__ import annotations

import json

import pytest

from sndr.model_configs.schema_v2 import ModelDef, SchemaError


def _make_model(override=None):
    """Construct a minimum-viable ModelDef for unit tests."""
    return ModelDef(
        schema_version=2,
        kind="model",
        id="test-model",
        title="Test",
        maintainer="test",
        model_path="/models/x",
        last_validated="2026-05-30",
        license="apache-2.0",
        override_generation_config=override,
    )


# ─── Field accepted values ────────────────────────────────────────────


class TestFieldAcceptance:
    def test_none_is_default(self):
        m = _make_model()
        assert m.override_generation_config is None

    def test_dict_with_primitives_accepted(self):
        cfg = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
        m = _make_model(cfg)
        m.validate()
        assert m.override_generation_config == cfg

    def test_string_value_accepted(self):
        m = _make_model({"foo": "bar"})
        m.validate()

    def test_bool_value_accepted(self):
        m = _make_model({"enable_thinking": False})
        m.validate()

    def test_list_value_accepted(self):
        m = _make_model({"stop_token_ids": [151645, 151643]})
        m.validate()


# ─── Field rejection ──────────────────────────────────────────────────


class TestFieldRejection:
    def test_non_dict_rejected(self):
        with pytest.raises(SchemaError, match="must be dict"):
            _make_model("not-a-dict").validate()

    def test_list_rejected(self):
        with pytest.raises(SchemaError, match="must be dict"):
            _make_model([("temperature", 0.6)]).validate()

    def test_empty_key_rejected(self):
        with pytest.raises(SchemaError, match="must be non-empty str"):
            _make_model({"": 0.6}).validate()

    def test_non_string_key_rejected(self):
        with pytest.raises(SchemaError, match="must be non-empty str"):
            # dict with int key (unusual but Python allows it)
            _make_model({1: 0.6}).validate()

    def test_nested_dict_value_rejected(self):
        """Values must be flat primitives — nested dicts complicate JSON
        serialisation downstream and don't match vllm CLI semantic."""
        with pytest.raises(SchemaError, match="must be int .* str .* list"):
            _make_model({"cfg": {"nested": 1}}).validate()

    def test_none_value_rejected(self):
        with pytest.raises(SchemaError):
            _make_model({"x": None}).validate()


# ─── compose() integration ────────────────────────────────────────────


class TestComposeIntegration:
    """Integration check — confirm compose layer emits the CLI flag.

    Construct a minimal-viable V2 model/hardware pair, run compose(),
    and verify vllm_extra_args contains the override flag with the JSON
    value canonicalised (sorted keys, compact separators).
    """

    def _build_pair(self, override=None):
        """Build a (ModelDef, HardwareDef) pair sufficient for compose().

        Mirrors the existing test_compose_patch_attribution.py fixtures
        — keeps shape stable when V2 schema evolves."""
        from sndr.model_configs.schema import HardwareSpec
        from sndr.model_configs.schema_v2 import (
            HardwareDef, HardwareSizing, ModelCapabilities,
            RuntimeBlock, RuntimeDockerBlock,
        )

        model = ModelDef(
            schema_version=2, kind="model", id="m",
            title="T", maintainer="x",
            model_path="/models/m",
            last_validated="2026-05-30", license="apache-2.0",
            capabilities=ModelCapabilities(attention_arch="dense"),
            override_generation_config=override,
        )
        hardware = HardwareDef(
            schema_version=2, kind="hardware", id="hw",
            title="HW", maintainer="x",
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
        return model, hardware

    def test_override_emitted_when_set(self):
        from sndr.model_configs.compose import compose
        model, hardware = self._build_pair(
            {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
        )
        cfg = compose(model, hardware)
        assert "--override-generation-config" in cfg.vllm_extra_args
        idx = cfg.vllm_extra_args.index("--override-generation-config")
        json_val = cfg.vllm_extra_args[idx + 1]
        # Round-trip JSON to confirm valid + values match
        parsed = json.loads(json_val)
        assert parsed == {"temperature": 0.6, "top_p": 0.95, "top_k": 20}

    def test_override_omitted_when_none(self):
        from sndr.model_configs.compose import compose
        model, hardware = self._build_pair(None)
        cfg = compose(model, hardware)
        assert "--override-generation-config" not in cfg.vllm_extra_args

    def test_json_value_sorted_compact(self):
        """JSON encoding uses sort_keys=True + compact separators so the
        compose output is deterministic across renders."""
        from sndr.model_configs.compose import compose
        # Out-of-order dict: expect sorted JSON
        model, hardware = self._build_pair(
            {"top_k": 20, "temperature": 0.6, "top_p": 0.95}
        )
        cfg = compose(model, hardware)
        idx = cfg.vllm_extra_args.index("--override-generation-config")
        json_val = cfg.vllm_extra_args[idx + 1]
        # Expect alphabetical key order + no spaces
        assert json_val == '{"temperature":0.6,"top_k":20,"top_p":0.95}'
