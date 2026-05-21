# SPDX-License-Identifier: Apache-2.0
"""PN275 — DFlash drafter VllmConfig max_cgs alignment (dev371 compat).

Closes the Q27-DFlash / Q35-DFlash dev371 boot failure documented in
``sndr_private/planning/audits/P2_DFLASH_DEV371_INCOMPATIBILITY_DESIGN_2026-05-21_RU.md``
and root-caused in
``sndr_private/planning/audits/P2_DFLASH_CANDIDATE_A_DESIGN_REFINED_2026-05-21_RU.md``.

Root cause (3-layer chain, verified against upstream dev371 source
SHA bf610c2f56764e1b30bc6065f4ceace3d6e59036):

  1. Parent ``VllmConfig`` init at ``vllm/config/vllm.py:1722``
     forces ``compilation_config.max_cudagraph_capture_size`` to
     match ``max(cudagraph_capture_sizes)`` after auto-computing
     both — so initial state is consistent.

  2. Genesis P95 (Marlin TP cudagraph cap on Ampere) fires AFTER
     ``VllmConfig.__post_init__`` and resets
     ``max_cudagraph_capture_size=8`` for TP>1 + Marlin + Ampere,
     WITHOUT touching ``cudagraph_capture_sizes`` (which stays at
     e.g. ``[..., 6]``). State is now desynchronized.

  3. DFlash ``_create_draft_vllm_config`` calls ``replace(base,
     attention_config=...)`` (via ``vllm/config/utils.py:119``)
     which rebuilds the VllmConfig through ``cls(**dataclass_dict)``.
     The new instance re-runs the post-init cross-validator at
     ``vllm/config/vllm.py:1703-1715`` which raises:

         ValueError: customized max_cudagraph_capture_size(=8) should
         be consistent with the max value of cudagraph_capture_sizes(=6)

  This cross-validator did not exist on dev338, so the desync was
  silently tolerated there.

Fix shape (B1 from the refined design doc):

  Wrap ``vllm.config.utils.replace`` so that for ``VllmConfig``
  instances where the source's ``compilation_config`` has
  ``max != max(sizes)`` AND the caller does NOT supply
  ``compilation_config`` via kwargs, inject an aligned
  ``compilation_config`` (built via the original ``replace`` on the
  CompilationConfig with the corrected ``max_cudagraph_capture_size``)
  BEFORE delegating to the original ``replace``. The rebuilt
  VllmConfig sees consistent state and pydantic validation passes.

Non-goals:
  * Does NOT touch ``CompilationConfig`` directly outside the
    aligned rebuild.
  * Does NOT change P95's contract — P95 still caps to 8 on the
    parent; this wrapper only re-aligns sizes on the rebuilt draft.
  * Does NOT short-circuit any caller's explicit ``compilation_config``
    kwarg — if a caller knows what it wants, the wrapper defers.
  * Default-OFF. Opt-in via ``GENESIS_ENABLE_PN275_DFLASH_MAX_CGS_ALIGN=1``.
    The DFlash hold gate (audit-side) stays in place until the
    smoke validates the fix.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""

from __future__ import annotations

import importlib
import logging
import os

log = logging.getLogger("genesis.wiring.pn275_dflash_max_cgs_align")

GENESIS_MARKER = "Genesis PN275 DFlash max_cgs align dev371 compat v1"
ENV_FLAG = "GENESIS_ENABLE_PN275_DFLASH_MAX_CGS_ALIGN"

_WRAPPED_ATTR = "_genesis_dflash_align_wrapped"
_ORIGINAL_ATTR = "_genesis_dflash_align_original"

_TARGET_MODULE = "vllm.config.utils"
_TARGET_FN_NAME = "replace"

_TRUTHY = ("1", "true", "yes", "on")


def should_apply() -> bool:
    """Strict opt-in. No platform / CUDA / SM probe — the wrapper is
    pure Python and safe to install on any host that imports
    `vllm.config.utils`."""
    return os.environ.get(ENV_FLAG, "").strip().lower() in _TRUTHY


def _build_wrapper(original):
    """Construct the wrapper closure. Separated so the unit tests can
    exercise it without importing the real `vllm.config.utils`.

    The wrapper has three fast-exit paths:
      1. Source is NOT a VllmConfig → delegate untouched.
      2. Caller already supplies ``compilation_config`` kwarg → defer.
      3. Source's compilation_config has consistent (or trivially
         absent) sizes / max → delegate untouched.

    Only when ALL fast exits miss does the wrapper compute the aligned
    compilation_config and inject it into kwargs.
    """
    def _wrapped(dataclass_instance, /, **kwargs):
        # Fast exit 1: only intervene on VllmConfig (string match avoids
        # importing the class at wrap time).
        cls = type(dataclass_instance)
        if cls.__name__ != "VllmConfig":
            return original(dataclass_instance, **kwargs)

        # Fast exit 2: caller already supplies compilation_config —
        # assume intent, do not second-guess.
        if "compilation_config" in kwargs:
            return original(dataclass_instance, **kwargs)

        cc = getattr(dataclass_instance, "compilation_config", None)
        if cc is None:
            return original(dataclass_instance, **kwargs)

        sizes = getattr(cc, "cudagraph_capture_sizes", None)
        mx = getattr(cc, "max_cudagraph_capture_size", None)

        # Fast exit 3: if EITHER field is absent on the source, the
        # validator at vllm/config/vllm.py:1703-1715 cannot raise
        # (its condition requires BOTH non-None). Treat empty list
        # of sizes as equivalent to None — vllm's validator logic
        # checks `cudagraph_capture_sizes is not None`, but an empty
        # list produces valid_max_size=0 which always mismatches
        # anything non-zero. Either way, our wrapper has no useful
        # intervention to make on empty sizes.
        if mx is None or not sizes:
            return original(dataclass_instance, **kwargs)

        # Both fields are set on the source. Even when they look
        # consistent right now (e.g. parent's first init aligned them),
        # P95 (Marlin TP cudagraph cap) text-patches
        # vllm/config/vllm.py:_set_cudagraph_sizes to FORCE
        # max_cudagraph_capture_size=8 again during the rebuild
        # triggered by `original` → `cls(**dataclass_dict)`. valid_max_size
        # is then recomputed from cudagraph_capture_sizes (still 6),
        # producing a mid-rebuild mismatch. The validator's escape
        # hatch is `cudagraph_capture_sizes is None` → only warns,
        # does not raise. So clear sizes on the injected
        # compilation_config; vllm will auto-recompute them inside
        # `_set_cudagraph_sizes` (line 1648-1688) and align max at
        # line 1722.
        try:
            aligned_cc = original(
                cc, cudagraph_capture_sizes=None,
            )
        except Exception as e:  # noqa: BLE001
            # Never break the outer replace because alignment failed —
            # fall through to original and let pydantic surface the
            # real error.
            log.debug(
                "[PN275] alignment of compilation_config failed (%s: %s); "
                "passing through to original replace",
                type(e).__name__, e,
            )
            return original(dataclass_instance, **kwargs)

        kwargs["compilation_config"] = aligned_cc
        log.debug(
            "[PN275] VllmConfig replace pre-aligned: cleared "
            "cudagraph_capture_sizes (was len=%d, max=%s) to route "
            "validator into warning-only branch; vllm auto-recomputes "
            "sizes + max during _set_cudagraph_sizes",
            len(sizes), mx,
        )
        return original(dataclass_instance, **kwargs)

    setattr(_wrapped, _WRAPPED_ATTR, True)
    setattr(_wrapped, _ORIGINAL_ATTR, original)
    return _wrapped


def _genesis_pn275_install_at_import(module_globals: "dict") -> bool:
    """Install the wrapper into a module-globals dict at module-import time.

    Mirrors the P103 self-install pattern. Called from the text-patched
    block appended at the bottom of ``vllm/config/utils.py``. Survives
    ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` because workers re-import
    ``vllm.config.utils`` from disk and re-execute the appended block,
    which then re-installs the wrapper in the fresh worker's module
    namespace.

    Args:
        module_globals: ``globals()`` of ``vllm.config.utils`` (passed
            by the text-patched call site). Must contain ``replace``.

    Returns:
        True if wrapper installed (or already installed; idempotent).
        False if installation skipped (env not set, missing function,
        etc.). Never raises — failures keep ``vllm.config.utils``
        importable.
    """
    try:
        import os as _os
        if _os.environ.get(ENV_FLAG, "").strip().lower() not in _TRUTHY:
            return False
        original = module_globals.get(_TARGET_FN_NAME)
        if original is None:
            return False
        if getattr(original, _WRAPPED_ATTR, False):
            return True  # already wrapped (idempotent)
        wrapped = _build_wrapper(original)
        module_globals[_TARGET_FN_NAME] = wrapped
        return True
    except Exception:  # noqa: BLE001
        return False


# ─── Text-patch block appended at the bottom of vllm/config/utils.py ───
#
# The block runs every time the module is imported — in the parent
# APIServer, in the EngineCore subprocess, AND in every Worker_TP*
# process spawned via VLLM_WORKER_MULTIPROC_METHOD=spawn. This is the
# only mechanism that survives spawn semantics, because workers
# re-import vllm modules from disk and a runtime setattr-wrap in the
# parent does NOT propagate.

_PN275_SELF_INSTALL_BLOCK = (
    "\n\n"
    "# ============================================================\n"
    "# [Genesis PN275 self-install] — module-import-time hook\n"
    "# ============================================================\n"
    "# When GENESIS_ENABLE_PN275_DFLASH_MAX_CGS_ALIGN=1, wrap\n"
    "# `replace` so that the rebuilt VllmConfig has\n"
    "# compilation_config.max_cudagraph_capture_size aligned with\n"
    "# max(cudagraph_capture_sizes) — dev371 cross-validator compat.\n"
    "# Survives any startup mechanism (exec vllm serve, worker spawn).\n"
    "# Lazy import — if vllm.sndr_core isn't on sys.path (test env,\n"
    "# partial install), the try/except keeps this module importable.\n"
    "try:\n"
    "    import os as _genesis_pn275_os\n"
    "    if _genesis_pn275_os.environ.get(\n"
    "        \"GENESIS_ENABLE_PN275_DFLASH_MAX_CGS_ALIGN\", \"\"\n"
    "    ).strip().lower() in (\"1\", \"true\", \"yes\", \"on\"):\n"
    "        from vllm.sndr_core.integrations.spec_decode."
    "pn275_dflash_max_cgs_align "
    "import (\n"
    "            _genesis_pn275_install_at_import as "
    "_genesis_pn275_inst,\n"
    "        )\n"
    "        _genesis_pn275_inst(globals())\n"
    "except Exception:  # noqa: BLE001\n"
    "    # Never break vllm.config.utils import — Genesis is opt-in.\n"
    "    pass\n"
)

# Anchor: the entire `replace` function definition in vllm/config/utils.py
# at dev371 SHA bf610c2f56764e1b30bc6065f4ceace3d6e59036. We append the
# self-install block immediately after the function's `return` line.
_PN275_SELF_INSTALL_ANCHOR = (
    "def replace(dataclass_instance: ConfigT, /, **kwargs) -> ConfigT:\n"
    "    \"\"\"Like [`dataclasses.replace`]"
    "(https://docs.python.org/3/library/dataclasses.html#dataclasses.replace),\n"
    "    but compatible with Pydantic dataclasses which use "
    "`pydantic.fields.Field` instead\n"
    "    of `dataclasses.field`\"\"\"\n"
    "    cls = type(dataclass_instance)\n"
    "    dataclass_dict = dataclass_instance.__dict__\n"
    "    dataclass_dict = {k: v for k, v in dataclass_dict.items() "
    "if is_init_field(cls, k)}\n"
    "    dataclass_dict.update(kwargs)\n"
    "    return cls(**dataclass_dict)\n"
)

_PN275_SELF_INSTALL_REPLACEMENT = (
    _PN275_SELF_INSTALL_ANCHOR + _PN275_SELF_INSTALL_BLOCK
)

_GENESIS_PN275_SELF_INSTALL_MARKER = (
    "Genesis PN275 self-install hook (DFlash dev371 compat)"
)


def _make_self_install_text_patcher():
    """Build the TextPatcher that appends the self-install block to
    ``vllm/config/utils.py``. Returns None if vllm tree is not
    resolvable (torch-less host, partial install)."""
    from vllm.sndr_core.detection.guards import resolve_vllm_file
    from vllm.sndr_core.core import TextPatch, TextPatcher

    target = resolve_vllm_file("config/utils.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN275 config/utils.py — self-install hook (DFlash dev371 compat)"
        ),
        target_file=str(target),
        marker=_GENESIS_PN275_SELF_INSTALL_MARKER,
        sub_patches=[
            TextPatch(
                name="pn275_self_install_at_utils_py_end",
                anchor=_PN275_SELF_INSTALL_ANCHOR,
                replacement=_PN275_SELF_INSTALL_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            # Specific to our own insertion. Re-applies hit Layer 2
            # IDEMPOTENT via the marker first.
            "[Genesis PN275 self-install]",
        ],
    )


def apply() -> tuple[str, str]:
    """Install PN275 on both the disk-resident vllm/config/utils.py
    (durable; survives spawn workers) and the current process's
    in-memory module (defense-in-depth; ensures immediate effect
    in APIServer + EngineCore that already imported the module).

    Returns ``(status, reason)`` per dispatcher convention. Never
    raises — failure paths log and return ``("skipped", ...)`` so
    other patches keep applying.
    """
    if not should_apply():
        return "skipped", (
            f"{ENV_FLAG} not set — opt-in patch (DFlash dev371 compat). "
            f"See sndr_private/planning/audits/"
            f"P2_DFLASH_CANDIDATE_A_DESIGN_REFINED_2026-05-21_RU.md "
            f"for the design rationale."
        )

    # ─── Step 1: durable text-patch on vllm/config/utils.py ─────────
    # This is what survives `exec vllm serve` + spawn workers. Workers
    # re-import vllm.config.utils from disk and re-execute the appended
    # self-install block, which re-installs the wrapper in the fresh
    # worker's module namespace.
    text_patch_status = "skipped"
    text_patch_reason = "vllm tree not resolvable"
    try:
        from vllm.sndr_core.detection.guards import vllm_install_root
        from vllm.sndr_core.core import TextPatchResult

        if vllm_install_root() is not None:
            patcher = _make_self_install_text_patcher()
            if patcher is not None:
                result, failure = patcher.apply()
                if result in (
                    TextPatchResult.APPLIED,
                    TextPatchResult.IDEMPOTENT,
                ):
                    text_patch_status = (
                        "applied" if result == TextPatchResult.APPLIED
                        else "idempotent"
                    )
                    text_patch_reason = (
                        "vllm/config/utils.py self-install hook "
                        "appended (survives `exec vllm serve` + worker "
                        "spawn)"
                    )
                else:
                    text_patch_status = "skipped"
                    text_patch_reason = (
                        f"text-patch did not land: "
                        f"{failure.reason if failure else 'unknown'} — "
                        f"{failure.detail if failure and failure.detail else 'unknown'}"
                    )
    except Exception as e:  # noqa: BLE001
        log.debug("[PN275] text-patch step non-fatal failure: %s", e)
        text_patch_reason = f"text-patch raised: {e}"

    # ─── Step 2: defense-in-depth setattr-wrap in current process ───
    # Wraps the function in THIS process's vllm.config.utils module
    # dict. Useful when:
    #   * Anyone in the current process already imported vllm.config.utils
    #     BEFORE the text-patch landed (the appended block ran on a
    #     pre-patch import; subsequent imports would hit the cached
    #     module). Setattr-wrap covers that gap.
    #   * The current process is parent APIServer or EngineCore (which
    #     don't re-import vllm.config.utils on demand — they cached it
    #     long ago).
    setattr_status = "skipped"
    setattr_reason = "fallback skipped"
    try:
        mod = importlib.import_module(_TARGET_MODULE)
    except ImportError as e:
        setattr_reason = f"vllm.config.utils not importable: {e}"
    else:
        original = getattr(mod, _TARGET_FN_NAME, None)
        if original is None:
            setattr_reason = (
                f"{_TARGET_FN_NAME!r} not in {_TARGET_MODULE!r}"
            )
        elif getattr(original, _WRAPPED_ATTR, False):
            setattr_status = "applied"
            setattr_reason = "already wrapped in this process (idempotent)"
        else:
            wrapped = _build_wrapper(original)
            setattr(mod, _TARGET_FN_NAME, wrapped)
            setattr_status = "applied"
            setattr_reason = (
                f"in-process {_TARGET_FN_NAME} wrapped (defense-in-depth)"
            )

    return "applied", (
        f"PN275 applied: text-patch={text_patch_status} "
        f"({text_patch_reason}); setattr={setattr_status} "
        f"({setattr_reason})"
    )


def is_applied() -> bool:
    """True iff our marker is present on ``vllm.config.utils.replace``."""
    try:
        mod = importlib.import_module(_TARGET_MODULE)
    except ImportError:
        return False
    fn = getattr(mod, _TARGET_FN_NAME, None)
    return fn is not None and getattr(fn, _WRAPPED_ATTR, False)


def revert() -> bool:
    """Restore the original ``replace`` function. Returns True on
    success. Idempotent: if not wrapped, returns False."""
    try:
        mod = importlib.import_module(_TARGET_MODULE)
    except ImportError:
        return False
    fn = getattr(mod, _TARGET_FN_NAME, None)
    if fn is None or not getattr(fn, _WRAPPED_ATTR, False):
        return False
    original = getattr(fn, _ORIGINAL_ATTR, None)
    if original is None:
        return False
    setattr(mod, _TARGET_FN_NAME, original)
    return True
