# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N379 — DefaultModelLoader / LoadConfig fail-fast
validation (vendor of OPEN vllm#45196).

================================================================
Source PR
================================================================
https://github.com/vllm-project/vllm/pull/45196
"[Bugfix][Model] Validate DefaultModelLoader / LoadConfig and fail
with clear errors" by @Sunt-ing (Ting SUN), OPEN as of 2026-06-11.
Studied via `gh pr view/diff 45196` (pr-sweep-50 roadmap chunk 1,
Theme D, wave 2). Anchors byte-verified count==1 on the pristine pin
0.22.1rc1.dev259+g303916e93 tree, 2026-06-11.

================================================================
WHAT IT DOES (three silent-misconfig classes -> loud ValueErrors)
================================================================

1. LoadConfig typing (config/load.py). ``load_format: str |
   LoadFormats`` is ``str | Any`` at runtime (``LoadFormats`` is a
   TYPE_CHECKING alias bound to ``Any`` in the else-branch), so the
   pydantic dataclass accepted ANY type; likewise a typo'd
   ``safetensors_load_strategy`` ("eagre") silently fell back to the
   lazy path. Vendored fix: ``load_format: str`` and
   ``safetensors_load_strategy: Literal["lazy", "eager", "prefetch",
   "torchao"] | None`` — pydantic (``@config`` = pydantic dataclass,
   verified on the pin's config/utils.py) rejects both at
   construction. The Literal set was re-derived from THE PIN's
   weight_utils.py dispatch sites (909/930/985/991) — all four values
   live; default_loader.py:390 runtime-assigns "torchao", which is
   inside the set.

2. DefaultModelLoader extra-config validation (__init__):
   - non-dict ``model_loader_extra_config`` (``.keys()`` used to throw
     a bare AttributeError mid-construction),
   - non-bool ``enable_multithread_load``,
   - ``num_threads`` not a positive int — ``isinstance(True, int)`` is
     True, so bool is rejected explicitly; ``num_threads=0`` used to
     fail LATE as ``ThreadPoolExecutor: max_workers must be greater
     than 0`` deep in the load path,
   - ``enable_multithread_load`` combined with a non-lazy
     ``safetensors_load_strategy`` — BYTE-VERIFIED on this pin: the
     multi-thread branch (default_loader.py:245-251) calls
     ``multi_thread_safetensors_weights_iterator`` WITHOUT the
     strategy, while the single-thread call (:257) passes it — the
     requested eager/prefetch/torchao is silently dropped. Reject the
     combination; None/"lazy" passes.

3. Explicit-safetensors ``.pt`` fallback (_prepare_weights). The
   pristine ``if fall_back_to_pt:`` appended ``*.pt`` even when
   ``use_safetensors`` was already True, so a pt-only model dir under
   ``load_format="safetensors"`` matched the ``.pt`` and later opened
   it via safe_open -> cryptic ``SafetensorError: deserializing
   header`` instead of "Cannot find any model weights". Guarded with
   ``and not use_safetensors``; ``auto``/``hf`` still fall back.

Genesis deviation from the upstream diff (iron rule #10 — adapt, not
copy): the strategy-reject check reads the already-bool-validated
local ``enable_multithread_load`` instead of upstream's second raw
``extra_config.get(...)`` lookup; the TYPE_CHECKING ``LoadFormats``
import is left in place (unused at runtime, harmless) to keep the
anchor surface minimal.

================================================================
WHY GENESIS WANTS IT
================================================================

Zero hot-path cost (constructor-only). Safety prerequisite for the
multithread-load experiment (``enable_multithread_load: true,
num_threads: 8`` in model_loader_extra_config) — ~30-60 s saved per
35B restart, dozens of restarts per bench session. That experiment is
a SERVER-STAGE item (A/B on 35B/27B against wall-clock restart time),
NOT part of this vendoring. Static pre-deploy mirror of the same
rules: scripts/audit_config_keys.py loader-key pass.

================================================================
SAFETY MODEL
================================================================

- Default OFF (opt-in via ``GENESIS_ENABLE_PN379_LOAD_CONFIG_FAIL_FAST=1``).
- Atomic 2-file transaction (config/load.py +
  model_executor/model_loader/default_loader.py): a Literal annotation
  without the loader checks would advertise a validation contract it
  doesn't deliver — all six sub-patches or none.
- Anchors missing (e.g. #45196 merges upstream) -> Layer-5 auto-skip,
  source stays vanilla.
- Behavior change is confined to configs that were ALREADY broken
  (silently misloading or failing late); every valid config constructs
  identically.
- Worst-case regression: upstream adds a new valid
  safetensors_load_strategy value on a future pin while our Literal
  rejects it -> loud ValueError at boot, recover with
  ``GENESIS_ENABLE_PN379_LOAD_CONFIG_FAIL_FAST=0`` + restart. The
  pristine-tree test pins the Literal set to the pin's weight_utils
  dispatch so the preflight catches this BEFORE promotion.

ENV: GENESIS_ENABLE_PN379_LOAD_CONFIG_FAIL_FAST=1   (opt-in, default OFF)

RISK: LOW — six anchored edits, byte-verified count==1 each on the
pristine pin; constructor/prepare-weights only, zero steady-state work.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Vendor of: vllm#45196 (Sunt-ing, OPEN), adapted per iron rule #10.
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import (
    TextPatch,
    TextPatcher,
)
from sndr.kernel.multi_file import MultiFilePatchTransaction

log = logging.getLogger("genesis.wiring.pn379_load_config_fail_fast")

GENESIS_PN379_MARKER = (
    "Genesis PN379 LoadConfig/DefaultModelLoader fail-fast vllm#45196 2026-06-11"
)


# ─── sub-patch 1: config/load.py — Literal import ─────────────────────────

PN379_LOAD_IMPORT_ANCHOR = "from typing import TYPE_CHECKING, Any\n"

PN379_LOAD_IMPORT_REPLACEMENT = (
    "# [Genesis PN379 vllm#45196] Literal import for the fail-fast\n"
    "# safetensors_load_strategy annotation below.\n"
    "from typing import TYPE_CHECKING, Any, Literal\n"
)


# ─── sub-patch 2: config/load.py — load_format: str ───────────────────────

PN379_LOAD_FORMAT_ANCHOR = '    load_format: str | LoadFormats = "auto"\n'

PN379_LOAD_FORMAT_REPLACEMENT = (
    "    # [Genesis PN379 vllm#45196] LoadFormats is a TYPE_CHECKING alias\n"
    "    # bound to Any at runtime, so the pristine union accepted ANY type\n"
    "    # and a non-string load_format failed late with confusing internal\n"
    "    # errors. Plain `str` makes pydantic reject it at construction.\n"
    '    load_format: str = "auto"\n'
)


# ─── sub-patch 3: config/load.py — strategy Literal ───────────────────────

PN379_LOAD_STRATEGY_ANCHOR = "    safetensors_load_strategy: str | None = None\n"

PN379_LOAD_STRATEGY_REPLACEMENT = (
    "    # [Genesis PN379 vllm#45196] A typo'd strategy silently fell back\n"
    "    # to the lazy path; the Literal makes pydantic reject it loudly.\n"
    "    # Value set re-derived from THIS pin's weight_utils.py dispatch\n"
    "    # (eager/prefetch/torchao + lazy/None default), not copied blind.\n"
    "    safetensors_load_strategy: (\n"
    '        Literal["lazy", "eager", "prefetch", "torchao"] | None\n'
    "    ) = None\n"
)


# ─── sub-patch 4: default_loader.py — extra-config dict check ─────────────

PN379_DL_EXTRA_DICT_ANCHOR = (
    "        extra_config = load_config.model_loader_extra_config\n"
    "        allowed_keys = {\n"
)

PN379_DL_EXTRA_DICT_REPLACEMENT = (
    "        extra_config = load_config.model_loader_extra_config\n"
    "        # [Genesis PN379 vllm#45196] Fail fast on a non-dict extra\n"
    "        # config instead of an AttributeError on .keys() below.\n"
    "        if not isinstance(extra_config, dict):\n"
    "            raise ValueError(\n"
    '                f"model_loader_extra_config must be a dict for load format "\n'
    '                f"{load_config.load_format}, got {type(extra_config).__name__}"\n'
    "            )\n"
    "        allowed_keys = {\n"
)


# ─── sub-patch 5: default_loader.py — multithread/num_threads checks ──────
#
# Anchored on the enable_weights_track assignment (the last pristine
# statement of __init__) so the checks run AFTER the allowed-keys
# reject and the patched __init__ stays a strict superset of pristine.

PN379_DL_VALIDATION_ANCHOR = (
    "        self.enable_weights_track: bool | None = extra_config.get(\n"
    '            "enable_weights_track", None\n'
    "        )\n"
)

PN379_DL_VALIDATION_REPLACEMENT = (
    "        # ════════════════════════════════════════════════════════════\n"
    "        # [Genesis PN379 vllm#45196] fail-fast loader-config checks.\n"
    "        # num_threads=0 used to die LATE as 'ThreadPoolExecutor:\n"
    "        # max_workers must be greater than 0'; bool is an int subclass\n"
    "        # and is rejected explicitly. The multi-thread iterator on this\n"
    "        # pin drops safetensors_load_strategy (the call at ~:245 does\n"
    "        # not pass it; the single-thread call does) — reject the\n"
    "        # combination instead of silently ignoring the strategy.\n"
    "        # ════════════════════════════════════════════════════════════\n"
    '        enable_multithread_load = extra_config.get("enable_multithread_load", False)\n'
    "        if not isinstance(enable_multithread_load, bool):\n"
    "            raise ValueError(\n"
    '                f"enable_multithread_load must be a bool, got "\n'
    '                f"{type(enable_multithread_load).__name__}"\n'
    "            )\n"
    '        num_threads = extra_config.get("num_threads")\n'
    "        if num_threads is not None and (\n"
    "            not isinstance(num_threads, int)\n"
    "            or isinstance(num_threads, bool)\n"
    "            or num_threads <= 0\n"
    "        ):\n"
    "            raise ValueError(\n"
    '                f"num_threads must be a positive integer, got {num_threads!r}"\n'
    "            )\n"
    "        if enable_multithread_load and (\n"
    '            load_config.safetensors_load_strategy not in (None, "lazy")\n'
    "        ):\n"
    "            raise ValueError(\n"
    '                "enable_multithread_load does not support "\n'
    '                "safetensors_load_strategy="\n'
    '                f"{load_config.safetensors_load_strategy!r}; the "\n'
    '                "multi-thread loader only implements the default lazy "\n'
    '                "strategy."\n'
    "            )\n"
    "\n"
    "        self.enable_weights_track: bool | None = extra_config.get(\n"
    '            "enable_weights_track", None\n'
    "        )\n"
)


# ─── sub-patch 6: default_loader.py — explicit-safetensors .pt guard ──────

PN379_DL_PT_FALLBACK_ANCHOR = (
    "        if fall_back_to_pt:\n"
    '            allow_patterns += ["*.pt"]\n'
)

PN379_DL_PT_FALLBACK_REPLACEMENT = (
    "        # [Genesis PN379 vllm#45196] Don't fall back to .pt for explicit\n"
    "        # safetensors formats; otherwise a pt-only dir matches the .pt\n"
    "        # and later opens it via safe_open (cryptic SafetensorError\n"
    "        # instead of 'Cannot find any model weights'). auto/hf still\n"
    "        # fall back (use_safetensors is False there).\n"
    "        if fall_back_to_pt and not use_safetensors:\n"
    '            allow_patterns += ["*.pt"]\n'
)


def _make_load_config_patcher(target_file: str | None = None) -> TextPatcher | None:
    """Build the config/load.py patcher (sub-patches 1-3).

    ``target_file`` overrides resolution for tests; default resolves the
    installed vllm tree.
    """
    if target_file is None:
        resolved = resolve_vllm_file("config/load.py")
        if resolved is None:
            return None
        target_file = str(resolved)
    return TextPatcher(
        patch_name="PN379 config/load.py — LoadConfig fail-fast annotations",
        target_file=target_file,
        marker=GENESIS_PN379_MARKER + " :: config/load.py",
        sub_patches=[
            TextPatch(
                name="pn379_literal_import",
                anchor=PN379_LOAD_IMPORT_ANCHOR,
                replacement=PN379_LOAD_IMPORT_REPLACEMENT,
                required=True,
            ),
            TextPatch(
                name="pn379_load_format_str",
                anchor=PN379_LOAD_FORMAT_ANCHOR,
                replacement=PN379_LOAD_FORMAT_REPLACEMENT,
                required=True,
            ),
            TextPatch(
                name="pn379_strategy_literal",
                anchor=PN379_LOAD_STRATEGY_ANCHOR,
                replacement=PN379_LOAD_STRATEGY_REPLACEMENT,
                required=True,
            ),
        ],
        # Defended '[Genesis'-prefixed marker only (PN369 class) — real
        # upstream absorption is caught by the required anchors going
        # missing (the pristine annotations ARE the anchors; #45196
        # rewrites all three lines).
        upstream_drift_markers=[
            "[Genesis PN379",
        ],
    )


def _make_default_loader_patcher(
    target_file: str | None = None,
) -> TextPatcher | None:
    """Build the default_loader.py patcher (sub-patches 4-6)."""
    if target_file is None:
        resolved = resolve_vllm_file(
            "model_executor/model_loader/default_loader.py"
        )
        if resolved is None:
            return None
        target_file = str(resolved)
    return TextPatcher(
        patch_name=(
            "PN379 model_executor/model_loader/default_loader.py — "
            "loader fail-fast validation"
        ),
        target_file=target_file,
        marker=GENESIS_PN379_MARKER + " :: default_loader.py",
        sub_patches=[
            TextPatch(
                name="pn379_extra_config_dict_check",
                anchor=PN379_DL_EXTRA_DICT_ANCHOR,
                replacement=PN379_DL_EXTRA_DICT_REPLACEMENT,
                required=True,
            ),
            TextPatch(
                name="pn379_multithread_num_threads_checks",
                anchor=PN379_DL_VALIDATION_ANCHOR,
                replacement=PN379_DL_VALIDATION_REPLACEMENT,
                required=True,
            ),
            TextPatch(
                name="pn379_explicit_safetensors_pt_guard",
                anchor=PN379_DL_PT_FALLBACK_ANCHOR,
                replacement=PN379_DL_PT_FALLBACK_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN379",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN379 atomically across config/load.py + default_loader.py.

    All six sub-patches or none — a Literal annotation without the
    loader-side checks (or vice versa) would advertise a validation
    contract the running engine doesn't deliver.
    """
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN379")
    log_decision("PN379", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    load_patcher = _make_load_config_patcher()
    dl_patcher = _make_default_loader_patcher()
    if load_patcher is None:
        return "skipped", "config/load.py not resolvable"
    if dl_patcher is None:
        return "skipped", (
            "model_executor/model_loader/default_loader.py not resolvable"
        )

    if not os.path.isfile(dl_patcher.target_file):
        return "skipped", f"target disappeared: {dl_patcher.target_file}"
    try:
        with open(dl_patcher.target_file, encoding="utf-8") as f:
            dl_content = f.read()
    except OSError as e:
        return "skipped", f"cannot read {dl_patcher.target_file}: {e}"

    if dl_patcher.marker in dl_content:
        log.info("[PN379] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"

    # Upstream-absorption probe: the pristine pin's .pt fallback is
    # unguarded. If `and not use_safetensors` already appears WITHOUT
    # our marker, upstream merged #45196 (or a successor) — self-retire
    # and queue the iron-rule-#11 deep diff rather than stacking checks.
    if "fall_back_to_pt and not use_safetensors" in dl_content:
        return "skipped", (
            "fall_back_to_pt already guarded by use_safetensors in pristine "
            "default_loader.py — upstream appears to have merged #45196; "
            "deep-diff and retire PN379 (iron rule #11)"
        )

    txn = MultiFilePatchTransaction(
        [load_patcher, dl_patcher],
        name="PN379",
    )
    status, detail = txn.apply_or_skip()
    if status == "applied":
        return "applied", (
            "PN379 applied: LoadConfig rejects non-string load_format and "
            "typo'd safetensors_load_strategy at construction (pydantic "
            "Literal); DefaultModelLoader rejects non-dict extra config, "
            "non-bool enable_multithread_load, non-positive num_threads and "
            "the multithread+non-lazy-strategy combination; explicit "
            "safetensors no longer falls back to .pt. Vendor of vllm#45196 "
            "(OPEN). Safety prerequisite for the multithread-load "
            "experiment (server-stage)."
        )
    return status, detail
