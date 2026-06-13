# SPDX-License-Identifier: Apache-2.0
"""TDD for Patch N379 — DefaultModelLoader / LoadConfig fail-fast
validation (vendor of OPEN vllm#45196, pr-sweep-50 roadmap chunk 1
Theme D, wave 2).

Three silent-misconfig classes converted into loud construction-time
ValueErrors (PN96-style drift killers):

  1. LoadConfig typing — ``load_format: str | LoadFormats`` is
     ``str | Any`` at runtime (TYPE_CHECKING alias), so pydantic
     accepted ANY type; a typo'd ``safetensors_load_strategy`` silently
     fell back to lazy. Literal/str annotations make pydantic reject
     both at construction.
  2. DefaultModelLoader extra-config — non-dict
     ``model_loader_extra_config``, non-bool ``enable_multithread_load``,
     non-positive/bool ``num_threads`` (used to die deep inside
     ThreadPoolExecutor), and the multithread + non-lazy
     ``safetensors_load_strategy`` combination (the multi-thread
     iterator on this pin demonstrably DROPS the strategy —
     default_loader.py:245-251 vs the single-thread call at :257).
  3. Explicit-safetensors ``.pt`` fallback — ``_prepare_weights``
     appended ``*.pt`` even for ``use_safetensors=True`` formats, so a
     pt-only dir under ``load_format="safetensors"`` opened the .pt via
     safe_open (cryptic SafetensorError instead of "no weights found").

Why Genesis wants it: safety prerequisite for the multithread-load
experiment (``enable_multithread_load: true, num_threads: 8`` —
~30-60 s saved per 35B restart, dozens of restarts per bench session)
— SERVER-STAGE item, not part of this vendoring.

These tests verify anchors against the PRISTINE pin tree, replacement
hygiene, and the validation behavior END-TO-END on synthetic-but-
compilable fakes carrying the byte-exact anchors (P79d convention).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sndr.kernel import TextPatchResult
from sndr.kernel.multi_file import MultiFilePatchTransaction

from sndr.engines.vllm.patches.loader.pn379_load_config_fail_fast import (
    GENESIS_PN379_MARKER,
    PN379_LOAD_IMPORT_ANCHOR,
    PN379_LOAD_IMPORT_REPLACEMENT,
    PN379_LOAD_FORMAT_ANCHOR,
    PN379_LOAD_FORMAT_REPLACEMENT,
    PN379_LOAD_STRATEGY_ANCHOR,
    PN379_LOAD_STRATEGY_REPLACEMENT,
    PN379_DL_EXTRA_DICT_ANCHOR,
    PN379_DL_EXTRA_DICT_REPLACEMENT,
    PN379_DL_VALIDATION_ANCHOR,
    PN379_DL_VALIDATION_REPLACEMENT,
    PN379_DL_PT_FALLBACK_ANCHOR,
    PN379_DL_PT_FALLBACK_REPLACEMENT,
    _make_load_config_patcher,
    _make_default_loader_patcher,
)

PRISTINE_ROOT = Path("/private/tmp/candidate_pin_current/vllm")

ALL_REPLACEMENTS = (
    PN379_LOAD_IMPORT_REPLACEMENT,
    PN379_LOAD_FORMAT_REPLACEMENT,
    PN379_LOAD_STRATEGY_REPLACEMENT,
    PN379_DL_EXTRA_DICT_REPLACEMENT,
    PN379_DL_VALIDATION_REPLACEMENT,
    PN379_DL_PT_FALLBACK_REPLACEMENT,
)


# ─── synthetic-but-compilable fakes carrying the byte-exact anchors ──────

FAKE_LOAD_CONFIG_PY = '''\
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fake_loader_module import LoadFormats
else:
    LoadFormats = Any


class LoadConfig:
    """Configuration for loading the model weights."""

    load_format: str | LoadFormats = "auto"
    download_dir: str | None = None
    safetensors_load_strategy: str | None = None
'''

FAKE_DEFAULT_LOADER_PY = '''\
class LoadConfig:
    def __init__(self, load_format="auto", safetensors_load_strategy=None,
                 model_loader_extra_config=None):
        self.load_format = load_format
        self.safetensors_load_strategy = safetensors_load_strategy
        self.model_loader_extra_config = (
            {} if model_loader_extra_config is None else model_loader_extra_config
        )


class BaseModelLoader:
    def __init__(self, load_config):
        self.load_config = load_config


class DefaultModelLoader(BaseModelLoader):
    DEFAULT_NUM_THREADS = 8

    def __init__(self, load_config):
        super().__init__(load_config)
        self.local_expert_ids = None

        extra_config = load_config.model_loader_extra_config
        allowed_keys = {
            "enable_multithread_load",
            "num_threads",
            "enable_weights_track",
        }
        unexpected_keys = set(extra_config.keys()) - allowed_keys

        if unexpected_keys:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{unexpected_keys}"
            )

        self.enable_weights_track: bool | None = extra_config.get(
            "enable_weights_track", None
        )

    def _prepare_weights(self, load_format, fall_back_to_pt):
        use_safetensors = False
        if load_format == "hf":
            allow_patterns = ["*.safetensors", "*.bin"]
        elif (
            load_format == "safetensors"
            or load_format == "fastsafetensors"
            or load_format == "instanttensor"
        ):
            use_safetensors = True
            allow_patterns = ["*.safetensors"]
        elif load_format == "pt":
            allow_patterns = ["*.pt"]
        else:
            raise ValueError(f"Unknown load_format: {load_format}")

        if fall_back_to_pt:
            allow_patterns += ["*.pt"]

        return allow_patterns, use_safetensors
'''


@pytest.fixture
def fake_load_config_py(tmp_path):
    p = tmp_path / "load.py"
    p.write_text(FAKE_LOAD_CONFIG_PY)
    return str(p)


@pytest.fixture
def fake_default_loader_py(tmp_path):
    p = tmp_path / "default_loader.py"
    p.write_text(FAKE_DEFAULT_LOADER_PY)
    return str(p)


def _apply_both(fake_load_config_py, fake_default_loader_py):
    txn = MultiFilePatchTransaction(
        [
            _make_load_config_patcher(target_file=fake_load_config_py),
            _make_default_loader_patcher(target_file=fake_default_loader_py),
        ],
        name="PN379-test",
    )
    status, detail = txn.apply_or_skip()
    assert status == "applied", f"txn did not apply: {detail}"
    return (
        Path(fake_load_config_py).read_text(),
        Path(fake_default_loader_py).read_text(),
    )


def _exec_module(src):
    ns = {}
    # dont_inherit: this test module uses `from __future__ import
    # annotations`; compile() would otherwise propagate that flag into
    # the fake and stringify the annotations we assert on.
    exec(compile(src, "<patched>", "exec", dont_inherit=True), ns)
    return ns


# ═══ 1. Anchor contract against the PRISTINE pin tree ════════════════════


@pytest.mark.skipif(
    not (PRISTINE_ROOT / "config/load.py").is_file(),
    reason="pristine candidate pin tree not present on this machine",
)
def test_pristine_anchors_present_exactly_once():
    load = (PRISTINE_ROOT / "config/load.py").read_text()
    dl = (
        PRISTINE_ROOT / "model_executor/model_loader/default_loader.py"
    ).read_text()
    assert load.count(PN379_LOAD_IMPORT_ANCHOR) == 1
    assert load.count(PN379_LOAD_FORMAT_ANCHOR) == 1
    assert load.count(PN379_LOAD_STRATEGY_ANCHOR) == 1
    assert dl.count(PN379_DL_EXTRA_DICT_ANCHOR) == 1
    assert dl.count(PN379_DL_VALIDATION_ANCHOR) == 1
    assert dl.count(PN379_DL_PT_FALLBACK_ANCHOR) == 1


@pytest.mark.skipif(
    not (PRISTINE_ROOT / "config/load.py").is_file(),
    reason="pristine candidate pin tree not present on this machine",
)
def test_pristine_strategy_values_still_match_literal():
    """The Literal set baked into the replacement must equal the set the
    pin's weight_utils actually dispatches on — if upstream adds a
    strategy, the Literal must be re-derived (else PN379 would reject a
    valid config)."""
    wu = (
        PRISTINE_ROOT / "model_executor/model_loader/weight_utils.py"
    ).read_text()
    for value in ("lazy", "eager", "prefetch", "torchao"):
        assert (
            f'"{value}"' in PN379_LOAD_STRATEGY_REPLACEMENT
        ), f"{value} missing from the Literal replacement"
    # weight_utils dispatches on eager/prefetch/torchao explicitly
    for value in ("eager", "prefetch", "torchao"):
        assert f'safetensors_load_strategy == "{value}"' in wu or (
            f"safetensors_load_strategy == '{value}'" in wu
        ) or f'"{value}"' in wu


@pytest.mark.skipif(
    not (PRISTINE_ROOT / "config/load.py").is_file(),
    reason="pristine candidate pin tree not present on this machine",
)
def test_pristine_multithread_path_still_drops_strategy():
    """The reject-combination hunk is justified ONLY while the pin's
    multi-thread iterator ignores safetensors_load_strategy. If upstream
    threads the strategy through, this patch needs re-study."""
    dl = (
        PRISTINE_ROOT / "model_executor/model_loader/default_loader.py"
    ).read_text()
    mt_call = dl.split("multi_thread_safetensors_weights_iterator(")[1]
    mt_call = mt_call.split(")")[0]
    assert "safetensors_load_strategy" not in mt_call


# ═══ 2. Replacement hygiene ═══════════════════════════════════════════════


def test_replacements_carry_genesis_markers():
    for repl in ALL_REPLACEMENTS:
        assert "[Genesis PN379" in repl


def test_drift_markers_have_no_self_collision():
    """PN369 false-skip class (mirrors tools/lint_drift_markers.py)."""
    for patcher in (
        _make_load_config_patcher(target_file="/nonexistent"),
        _make_default_loader_patcher(target_file="/nonexistent"),
    ):
        marker_line = f"# [Genesis wiring marker: {patcher.marker}]\n"
        for dm in patcher.upstream_drift_markers:
            if dm.startswith("[Genesis"):
                continue
            assert dm not in marker_line
            for sp in patcher.sub_patches:
                assert dm not in sp.replacement


def test_pt_fallback_replacement_guards_on_use_safetensors():
    assert "fall_back_to_pt and not use_safetensors" in (
        PN379_DL_PT_FALLBACK_REPLACEMENT
    )


# ═══ 3. Validation behavior — end-to-end on patched executable fakes ═════


class TestLoaderValidation:
    def _loader_ns(self, fake_load_config_py, fake_default_loader_py):
        _, dl_src = _apply_both(fake_load_config_py, fake_default_loader_py)
        return _exec_module(dl_src)

    def test_non_dict_extra_config_rejected(
        self, fake_load_config_py, fake_default_loader_py
    ):
        ns = self._loader_ns(fake_load_config_py, fake_default_loader_py)
        cfg = ns["LoadConfig"](model_loader_extra_config="not-a-dict")
        with pytest.raises(ValueError, match="must be a dict"):
            ns["DefaultModelLoader"](cfg)

    def test_non_bool_multithread_rejected(
        self, fake_load_config_py, fake_default_loader_py
    ):
        ns = self._loader_ns(fake_load_config_py, fake_default_loader_py)
        cfg = ns["LoadConfig"](
            model_loader_extra_config={"enable_multithread_load": "yes"}
        )
        with pytest.raises(ValueError, match="enable_multithread_load"):
            ns["DefaultModelLoader"](cfg)

    @pytest.mark.parametrize("bad", [0, -4, True, "8", 2.5])
    def test_bad_num_threads_rejected(
        self, fake_load_config_py, fake_default_loader_py, bad
    ):
        """num_threads=0 used to die deep inside ThreadPoolExecutor;
        bool is an int subclass and must be rejected explicitly."""
        ns = self._loader_ns(fake_load_config_py, fake_default_loader_py)
        cfg = ns["LoadConfig"](
            model_loader_extra_config={
                "enable_multithread_load": True,
                "num_threads": bad,
            }
        )
        with pytest.raises(ValueError, match="num_threads"):
            ns["DefaultModelLoader"](cfg)

    def test_multithread_with_non_lazy_strategy_rejected(
        self, fake_load_config_py, fake_default_loader_py
    ):
        ns = self._loader_ns(fake_load_config_py, fake_default_loader_py)
        cfg = ns["LoadConfig"](
            safetensors_load_strategy="torchao",
            model_loader_extra_config={"enable_multithread_load": True},
        )
        with pytest.raises(ValueError, match="does not support"):
            ns["DefaultModelLoader"](cfg)

    @pytest.mark.parametrize("ok_strategy", [None, "lazy"])
    def test_multithread_with_lazy_or_none_ok(
        self, fake_load_config_py, fake_default_loader_py, ok_strategy
    ):
        ns = self._loader_ns(fake_load_config_py, fake_default_loader_py)
        cfg = ns["LoadConfig"](
            safetensors_load_strategy=ok_strategy,
            model_loader_extra_config={
                "enable_multithread_load": True,
                "num_threads": 8,
            },
        )
        loader = ns["DefaultModelLoader"](cfg)
        assert loader.enable_weights_track is None

    def test_valid_experiment_config_accepted(
        self, fake_load_config_py, fake_default_loader_py
    ):
        """The exact server-stage experiment shape must construct clean."""
        ns = self._loader_ns(fake_load_config_py, fake_default_loader_py)
        cfg = ns["LoadConfig"](
            model_loader_extra_config={
                "enable_multithread_load": True,
                "num_threads": 8,
            }
        )
        ns["DefaultModelLoader"](cfg)

    def test_vanilla_empty_extra_config_unchanged(
        self, fake_load_config_py, fake_default_loader_py
    ):
        ns = self._loader_ns(fake_load_config_py, fake_default_loader_py)
        ns["DefaultModelLoader"](ns["LoadConfig"]())

    def test_unexpected_keys_still_rejected(
        self, fake_load_config_py, fake_default_loader_py
    ):
        """The pristine allowed-keys check must survive the patch."""
        ns = self._loader_ns(fake_load_config_py, fake_default_loader_py)
        cfg = ns["LoadConfig"](model_loader_extra_config={"tpyo_key": 1})
        with pytest.raises(ValueError, match="Unexpected extra config"):
            ns["DefaultModelLoader"](cfg)


class TestPtFallback:
    def test_explicit_safetensors_does_not_add_pt(
        self, fake_load_config_py, fake_default_loader_py
    ):
        _, dl_src = _apply_both(fake_load_config_py, fake_default_loader_py)
        ns = _exec_module(dl_src)
        loader = ns["DefaultModelLoader"](ns["LoadConfig"]())
        patterns, use_st = loader._prepare_weights(
            "safetensors", fall_back_to_pt=True
        )
        assert use_st is True
        assert "*.pt" not in patterns

    def test_hf_still_falls_back_to_pt(
        self, fake_load_config_py, fake_default_loader_py
    ):
        _, dl_src = _apply_both(fake_load_config_py, fake_default_loader_py)
        ns = _exec_module(dl_src)
        loader = ns["DefaultModelLoader"](ns["LoadConfig"]())
        patterns, use_st = loader._prepare_weights("hf", fall_back_to_pt=True)
        assert use_st is False
        assert "*.pt" in patterns


class TestLoadConfigAnnotations:
    def test_patched_load_config_compiles_and_annotations_updated(
        self, fake_load_config_py, fake_default_loader_py
    ):
        load_src, _ = _apply_both(fake_load_config_py, fake_default_loader_py)
        ns = _exec_module(load_src)
        ann = ns["LoadConfig"].__annotations__
        # load_format dropped the runtime-Any union -> plain str
        assert ann["load_format"] is str
        # strategy annotation is now the Literal union (evaluated object)
        strategy = str(ann["safetensors_load_strategy"])
        for value in ("lazy", "eager", "prefetch", "torchao"):
            assert value in strategy
        assert "Literal" in load_src


# ═══ 4. Idempotency / atomicity ═══════════════════════════════════════════


def test_idempotent_second_apply(fake_load_config_py, fake_default_loader_py):
    load_src, dl_src = _apply_both(fake_load_config_py, fake_default_loader_py)
    assert GENESIS_PN379_MARKER in load_src
    assert GENESIS_PN379_MARKER in dl_src

    for make, target in (
        (_make_load_config_patcher, fake_load_config_py),
        (_make_default_loader_patcher, fake_default_loader_py),
    ):
        patcher = make(target_file=target)
        result, failure = patcher.apply()
        assert result == TextPatchResult.IDEMPOTENT, failure

    assert Path(fake_load_config_py).read_text() == load_src
    assert Path(fake_default_loader_py).read_text() == dl_src


def test_atomic_skip_when_one_anchor_missing(
    fake_load_config_py, fake_default_loader_py
):
    """If the loader-side anchor is gone (e.g. #45196 merged upstream),
    NEITHER file may be modified — a Literal annotation without the
    loader checks would advertise a validation contract it doesn't
    deliver."""
    broken = FAKE_DEFAULT_LOADER_PY.replace(
        "        if fall_back_to_pt:\n",
        "        if fall_back_to_pt and not use_safetensors:\n",
    )
    Path(fake_default_loader_py).write_text(broken)
    before_load = Path(fake_load_config_py).read_text()

    txn = MultiFilePatchTransaction(
        [
            _make_load_config_patcher(target_file=fake_load_config_py),
            _make_default_loader_patcher(target_file=fake_default_loader_py),
        ],
        name="PN379-test",
    )
    status, detail = txn.apply_or_skip()
    assert status == "skipped", detail
    assert Path(fake_load_config_py).read_text() == before_load
    assert Path(fake_default_loader_py).read_text() == broken
