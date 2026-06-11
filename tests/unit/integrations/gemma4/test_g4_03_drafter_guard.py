# SPDX-License-Identifier: Apache-2.0
"""Unit tests for G4_03 — non-causal drafter guard binding + fail-loud.

2026-06-11 repoint (preflight residual triage §6 "binding failures"):
the current pin removed ``vllm/v1/spec_decode/eagle3.py``; Eagle3 now
runs through ``EagleProposer`` (``eagle.py``) and the shared method
``SpecDecodeBaseProposer._create_draft_vllm_config``
(``llm_base_proposer.py:1157``). These tests pin the new contract:

  * primary binding wraps ``SpecDecodeBaseProposer`` (rename-proof,
    covers both Eagle3 and DFlash — DFlash's override delegates to
    ``super()._create_draft_vllm_config()``);
  * the wrapper gates on ``method in ("eagle3", "dflash")`` so MTP /
    draft_model / ngram proposers pass through untouched;
  * per-class probes on ``eagle.EagleProposer`` / ``dflash.DFlashProposer``
    remain as fallback;
  * binding failures while the guard is ENABLED are loud: status
    "failed" (or "partial" for one-of-two fallback coverage), never a
    silent "applied"/"skipped".

All vllm modules are faked in ``sys.modules`` — no torch/CUDA needed.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

MODULE_PATH = (
    "sndr.engines.vllm.patches.model_compat.gemma4."
    "g4_03_gemma4_ampere_non_causal_drafter_guard"
)
_ENV_ENABLE = "GENESIS_ENABLE_G4_03_GEMMA4_NON_CAUSAL_DRAFTER_GUARD"
_ENV_DISABLE = "GENESIS_DISABLE_G4_03_GUARD"
_ENV_G4_10 = "GENESIS_ENABLE_G4_10_GEMMA4_AMPERE_NON_CAUSAL_BACKEND"


# ─── Fixtures / fakes ─────────────────────────────────────────────────


@pytest.fixture
def g4_03(monkeypatch):
    """Fresh guard module per test (module-level _APPLIED state)."""
    monkeypatch.delenv(_ENV_DISABLE, raising=False)
    monkeypatch.delenv(_ENV_G4_10, raising=False)
    sys.modules.pop(MODULE_PATH, None)
    mod = importlib.import_module(MODULE_PATH)
    yield mod
    sys.modules.pop(MODULE_PATH, None)


def _install_fake_spec_decode(
    monkeypatch,
    *,
    with_base: bool = True,
    with_eagle: bool = False,
    with_dflash: bool = False,
):
    """Build a minimal fake ``vllm.v1.spec_decode`` tree in sys.modules.

    Returns a dict of the installed proposer classes keyed by
    "base" / "eagle" / "dflash". Absent submodules raise ImportError on
    ``from vllm.v1.spec_decode import <name>`` because the fake package
    has no loader/__path__.
    """
    vllm = ModuleType("vllm")
    v1 = ModuleType("vllm.v1")
    spec = ModuleType("vllm.v1.spec_decode")
    vllm.v1 = v1
    v1.spec_decode = spec
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.v1", v1)
    monkeypatch.setitem(sys.modules, "vllm.v1.spec_decode", spec)
    # Ensure submodules from any REAL vllm install never leak in.
    for sub in ("llm_base_proposer", "eagle", "dflash", "eagle3"):
        monkeypatch.delitem(
            sys.modules, f"vllm.v1.spec_decode.{sub}", raising=False
        )

    installed: dict[str, type] = {}

    if with_base:
        base_mod = ModuleType("vllm.v1.spec_decode.llm_base_proposer")

        class SpecDecodeBaseProposer:
            """Mirrors pristine llm_base_proposer.py:59 surface."""

            def __init__(self, method: str, vllm_config) -> None:
                # Pristine __init__ line 71: self.method mirrors
                # speculative_config.method.
                self.method = method
                self.speculative_config = SimpleNamespace(method=method)
                self.vllm_config = vllm_config

            def _create_draft_vllm_config(self):
                return "ORIGINAL_DRAFT_CONFIG"

        base_mod.SpecDecodeBaseProposer = SpecDecodeBaseProposer
        spec.llm_base_proposer = base_mod
        monkeypatch.setitem(
            sys.modules, "vllm.v1.spec_decode.llm_base_proposer", base_mod
        )
        installed["base"] = SpecDecodeBaseProposer

    if with_eagle:
        eagle_mod = ModuleType("vllm.v1.spec_decode.eagle")

        class EagleProposer:
            def __init__(self, method: str, vllm_config) -> None:
                self.method = method
                self.speculative_config = SimpleNamespace(method=method)
                self.vllm_config = vllm_config

            def _create_draft_vllm_config(self):
                return "ORIGINAL_EAGLE_DRAFT_CONFIG"

        eagle_mod.EagleProposer = EagleProposer
        spec.eagle = eagle_mod
        monkeypatch.setitem(
            sys.modules, "vllm.v1.spec_decode.eagle", eagle_mod
        )
        installed["eagle"] = EagleProposer

    if with_dflash:
        dflash_mod = ModuleType("vllm.v1.spec_decode.dflash")

        class DFlashProposer:
            def __init__(self, method: str, vllm_config) -> None:
                self.method = method
                self.speculative_config = SimpleNamespace(method=method)
                self.vllm_config = vllm_config

            def _create_draft_vllm_config(self):
                return "ORIGINAL_DFLASH_DRAFT_CONFIG"

        dflash_mod.DFlashProposer = DFlashProposer
        spec.dflash = dflash_mod
        monkeypatch.setitem(
            sys.modules, "vllm.v1.spec_decode.dflash", dflash_mod
        )
        installed["dflash"] = DFlashProposer

    return installed


def _gemma4_vllm_config():
    hf_config = SimpleNamespace(
        architectures=["Gemma4ForConditionalGeneration"],
        model_type="gemma4",
    )
    model_config = SimpleNamespace(hf_config=hf_config)
    return SimpleNamespace(model_config=model_config)


def _qwen_vllm_config():
    hf_config = SimpleNamespace(
        architectures=["Qwen3MoeForCausalLM"],
        model_type="qwen3_moe",
    )
    model_config = SimpleNamespace(hf_config=hf_config)
    return SimpleNamespace(model_config=model_config)


# ─── Source-level regression pins ─────────────────────────────────────


def test_source_does_not_reference_removed_eagle3_module(g4_03):
    """Pin removed eagle3.py — the guard must not bind to the dead path."""
    source = Path(g4_03.__file__).read_text()
    assert "spec_decode import eagle3" not in source
    assert "spec_decode.eagle3" not in source


# ─── apply() binding behavior ─────────────────────────────────────────


def test_disabled_env_skips(g4_03, monkeypatch):
    monkeypatch.delenv(_ENV_ENABLE, raising=False)
    status, reason = g4_03.apply()
    assert status == "skipped"
    assert _ENV_ENABLE in reason


def test_primary_binding_wraps_base_proposer(g4_03, monkeypatch):
    monkeypatch.setenv(_ENV_ENABLE, "1")
    installed = _install_fake_spec_decode(monkeypatch, with_base=True)
    status, reason = g4_03.apply()
    assert status == "applied"
    assert "SpecDecodeBaseProposer" in reason
    wrapper = installed["base"]._create_draft_vllm_config
    assert getattr(wrapper, "_genesis_g4_03_wrapped", False)


def test_idempotent_second_apply(g4_03, monkeypatch):
    monkeypatch.setenv(_ENV_ENABLE, "1")
    _install_fake_spec_decode(monkeypatch, with_base=True)
    assert g4_03.apply()[0] == "applied"
    status, reason = g4_03.apply()
    assert status == "applied"
    assert "idempotent" in reason


def test_fallback_binding_covers_both_drafter_classes(g4_03, monkeypatch):
    """Base module missing — per-class probes must cover Eagle AND DFlash."""
    monkeypatch.setenv(_ENV_ENABLE, "1")
    installed = _install_fake_spec_decode(
        monkeypatch, with_base=False, with_eagle=True, with_dflash=True
    )
    status, _reason = g4_03.apply()
    assert status == "applied"
    for key in ("eagle", "dflash"):
        wrapper = installed[key]._create_draft_vllm_config
        assert getattr(wrapper, "_genesis_g4_03_wrapped", False)


def test_partial_fallback_is_loud_not_applied(g4_03, monkeypatch):
    """Only one of two drafter classes found → "partial", never "applied"."""
    monkeypatch.setenv(_ENV_ENABLE, "1")
    _install_fake_spec_decode(
        monkeypatch, with_base=False, with_eagle=True, with_dflash=False
    )
    status, reason = g4_03.apply()
    assert status == "partial"
    assert "dflash" in reason.lower()


def test_no_binding_fails_loud_when_enabled(g4_03, monkeypatch):
    """Enabled guard with zero bindable sites must report "failed".

    This is the §6 silent hazard: the pre-repoint code swallowed the
    ImportError at log.debug and could report success while Eagle3 ran
    unguarded.
    """
    monkeypatch.setenv(_ENV_ENABLE, "1")
    _install_fake_spec_decode(
        monkeypatch, with_base=False, with_eagle=False, with_dflash=False
    )
    status, _reason = g4_03.apply()
    assert status == "failed"
    assert not g4_03.is_applied()


# ─── Wrapped-method runtime gating ────────────────────────────────────


def _apply_on_fake_base(g4_03, monkeypatch, *, ampere: bool = True):
    monkeypatch.setenv(_ENV_ENABLE, "1")
    installed = _install_fake_spec_decode(monkeypatch, with_base=True)
    status, _ = g4_03.apply()
    assert status == "applied"
    monkeypatch.setattr(g4_03, "is_ampere_sm86", lambda: ampere)
    return installed["base"]


@pytest.mark.parametrize("method", ["eagle3", "dflash"])
def test_guard_refuses_non_causal_drafter_on_ampere_gemma4(
    g4_03, monkeypatch, method
):
    cls = _apply_on_fake_base(g4_03, monkeypatch, ampere=True)
    proposer = cls(method, _gemma4_vllm_config())
    with pytest.raises(RuntimeError, match=r"\[Genesis G4_03\]"):
        proposer._create_draft_vllm_config()


@pytest.mark.parametrize("method", ["mtp", "eagle", "draft_model", "ngram"])
def test_guard_passes_through_causal_methods(g4_03, monkeypatch, method):
    """MTP (incl. native Gemma4Proposer path), EAGLE-1 etc. stay untouched."""
    cls = _apply_on_fake_base(g4_03, monkeypatch, ampere=True)
    proposer = cls(method, _gemma4_vllm_config())
    assert proposer._create_draft_vllm_config() == "ORIGINAL_DRAFT_CONFIG"


def test_guard_passes_through_non_gemma4_target(g4_03, monkeypatch):
    cls = _apply_on_fake_base(g4_03, monkeypatch, ampere=True)
    proposer = cls("eagle3", _qwen_vllm_config())
    assert proposer._create_draft_vllm_config() == "ORIGINAL_DRAFT_CONFIG"


def test_guard_passes_through_off_ampere(g4_03, monkeypatch):
    cls = _apply_on_fake_base(g4_03, monkeypatch, ampere=False)
    proposer = cls("eagle3", _gemma4_vllm_config())
    assert proposer._create_draft_vllm_config() == "ORIGINAL_DRAFT_CONFIG"


def test_guard_defers_to_g4_10_when_enabled(g4_03, monkeypatch):
    cls = _apply_on_fake_base(g4_03, monkeypatch, ampere=True)
    monkeypatch.setenv(_ENV_G4_10, "1")
    proposer = cls("eagle3", _gemma4_vllm_config())
    assert proposer._create_draft_vllm_config() == "ORIGINAL_DRAFT_CONFIG"


def test_refusal_message_recommends_native_gemma4_mtp(g4_03, monkeypatch):
    """Recommendation must match the pin's native Gemma4Proposer MTP path
    (vllm/v1/spec_decode/gemma4.py:31; method "mtp")."""
    cls = _apply_on_fake_base(g4_03, monkeypatch, ampere=True)
    proposer = cls("eagle3", _gemma4_vllm_config())
    with pytest.raises(RuntimeError) as excinfo:
        proposer._create_draft_vllm_config()
    msg = str(excinfo.value)
    assert "method: mtp" in msg
    assert "Gemma4Proposer" in msg


# ─── revert() ─────────────────────────────────────────────────────────


def test_revert_restores_original_method(g4_03, monkeypatch):
    monkeypatch.setenv(_ENV_ENABLE, "1")
    installed = _install_fake_spec_decode(monkeypatch, with_base=True)
    cls = installed["base"]
    original = cls.__dict__["_create_draft_vllm_config"]
    assert g4_03.apply()[0] == "applied"
    assert cls.__dict__["_create_draft_vllm_config"] is not original
    assert g4_03.revert() is True
    assert cls.__dict__["_create_draft_vllm_config"] is original
    assert not g4_03.is_applied()
