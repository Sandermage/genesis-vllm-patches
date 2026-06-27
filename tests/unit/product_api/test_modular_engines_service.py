# SPDX-License-Identifier: Apache-2.0
"""Honesty contract for the modular engines domain service.

Two former lies are locked out here:

  1. ``EngineSummary`` carried a ``container_count`` field hardcoded to 0
     in ``engines_service._build_summary`` with a comment claiming the
     'routes layer' would fill it — which it never did. The field is
     removed rather than reporting a false 0; this test asserts it is
     gone from both the schema and every produced summary.

  2. The engine detail surface is wired to the real adapter, so a known
     engine reports its real supported-pin set (no longer the empty
     stub). vLLM ships committed pins, so its detail is non-empty.
"""
from __future__ import annotations

import pytest

# The modular engines schemas are pydantic models; the light CI test leg has no
# pydantic. Skip cleanly there rather than failing collection (dir convention).
pytest.importorskip("pydantic")

from sndr.product_api.domain.engines_service import (  # noqa: E402
    get_engine_detail,
    list_engine_summaries,
)
from sndr.product_api.schemas.engines import EngineDetail, EngineSummary  # noqa: E402


def test_engine_summary_schema_has_no_container_count_field():
    """The misleading hardcoded-0 field must not exist on the schema."""
    assert "container_count" not in EngineSummary.model_fields
    assert "container_count" not in EngineDetail.model_fields


def test_produced_summaries_carry_no_container_count():
    """Every produced summary's serialized form omits the removed field —
    no consumer can read a fabricated 0."""
    summaries = list_engine_summaries()
    assert summaries, "at least vllm should be registered"
    for s in summaries:
        assert isinstance(s, EngineSummary)
        assert "container_count" not in s.model_dump()


def test_vllm_summary_present():
    names = {s.name for s in list_engine_summaries()}
    assert "vllm" in names


def test_vllm_detail_reports_real_supported_pins():
    """The detail surface delegates to the adapter, which now answers from
    the real pins dir — so vLLM (which ships committed pins) reports a
    non-empty supported-pin list and the matching capabilities flags."""
    detail = get_engine_detail("vllm")
    assert isinstance(detail, EngineDetail)
    assert detail.supported_pins, "vllm ships committed pins with anchors.json"
    assert "0.23.1_b4c80ec0f" in detail.supported_pins
    assert detail.capabilities["multi_pin"] is True
    assert detail.capabilities["drift_detection"] is True
