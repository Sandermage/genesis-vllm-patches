# SPDX-License-Identifier: Apache-2.0
"""PN92 — nixl_ep trial-import guard (vllm PR #40154 / issue #42525 backport).

Problem
=======
On vllm nightly `dcacdf9a` (2026-05-13 onward) the image ships an
`nixl_ep` C++ extension compiled against CUDA 12 inside an otherwise
CUDA-13 image. On any host without `libcudart.so.12` available the
import fails:

    ImportError: libcudart.so.12: cannot open shared object file

The current `has_nixl_ep()` guard at
`vllm/utils/import_utils.py:~418` only checks for *module presence*
via `find_spec`, not actual importability. Cascading import path —
ANY model loader path that touches `fused_moe` (Qwen3.5/3.6,
DeepSeek-V3, Mixtral, etc.) traverses
`fused_moe.all2all_utils → prepare_finalize.nixl_ep → nixl_ep_cpp`
and dies with the underlying ImportError wrapped as a pydantic
ValidationError:

    Model architectures ['Qwen3_5ForConditionalGeneration']
    failed to be inspected.

This is not Qwen-specific — it breaks all hybrid-MoE models on
nightly `dcacdf9a` and later until upstream PR #40154 lands.

Fix
===
Replace `find_spec`-only check with a trial `import nixl_ep` wrapped
in `try/except (ImportError, OSError)`. Mirrors the upstream PR
verbatim. Safe and additive: on healthy hosts (with the right
libcudart available) trial import succeeds and behavior is unchanged.

Env gate: `GENESIS_ENABLE_PN92_NIXL_EP_TRIAL_IMPORT=1` (default OFF
since older nightlies don't ship the broken module). Operators that
just bumped the nightly tag turn this on.

Companion patches in the same file are also fixed (deep_ep, mori)
because they share the same find_spec-only pattern and the upstream
PR addresses all three at once.
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import TextPatch, TextPatcher

log = logging.getLogger("genesis.wiring.pn92_nixl_ep_trial_import")

GENESIS_MARKER = "Genesis PN92 nixl_ep trial-import guard (vllm PR #40154)"


def _enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN92_NIXL_EP_TRIAL_IMPORT", "0",
    ).strip().lower() in ("1", "true", "yes", "on")


# Anchor verified against vllm nightly dcacdf9a (2026-05-13). All three
# helpers follow the identical pattern in vllm/utils/import_utils.py.
NIXL_EP_OLD = (
    "def has_nixl_ep() -> bool:\n"
    "    \"\"\"Whether the optional `nixl_ep` package is available.\"\"\"\n"
    "    return _has_module(\"nixl_ep\")\n"
)
NIXL_EP_NEW = (
    "def has_nixl_ep() -> bool:\n"
    "    \"\"\"Whether the optional `nixl_ep` package is available AND importable.\n"
    "\n"
    "    [Genesis PN92] vllm PR #40154 / issue #42525 — guard against\n"
    "    CUDA-12 .so shipped in CUDA-13 images. find_spec alone is not\n"
    "    enough; the runtime libcudart linkage may still fail. Trial\n"
    "    import wrapped in (ImportError, OSError) is the proper check.\"\"\"\n"
    "    if not _has_module(\"nixl_ep\"):\n"
    "        return False\n"
    "    try:\n"
    "        import nixl_ep  # noqa: F401\n"
    "        return True\n"
    "    except (ImportError, OSError):\n"
    "        return False\n"
)

DEEP_EP_OLD = (
    "def has_deep_ep() -> bool:\n"
    "    \"\"\"Whether the optional `deep_ep` package is available.\"\"\"\n"
    "    return _has_module(\"deep_ep\")\n"
)
DEEP_EP_NEW = (
    "def has_deep_ep() -> bool:\n"
    "    \"\"\"Whether the optional `deep_ep` package is available AND importable.\n"
    "\n"
    "    [Genesis PN92] same trial-import guard as has_nixl_ep — protects\n"
    "    against C++ extensions linked against the wrong libcudart.\"\"\"\n"
    "    if not _has_module(\"deep_ep\"):\n"
    "        return False\n"
    "    try:\n"
    "        import deep_ep  # noqa: F401\n"
    "        return True\n"
    "    except (ImportError, OSError):\n"
    "        return False\n"
)

MORI_OLD = (
    "def has_mori() -> bool:\n"
    "    \"\"\"Whether the optional `mori` package is available.\"\"\"\n"
    "    return _has_module(\"mori\")\n"
)
MORI_NEW = (
    "def has_mori() -> bool:\n"
    "    \"\"\"Whether the optional `mori` package is available AND importable.\n"
    "\n"
    "    [Genesis PN92] same trial-import guard as has_nixl_ep.\"\"\"\n"
    "    if not _has_module(\"mori\"):\n"
    "        return False\n"
    "    try:\n"
    "        import mori  # noqa: F401\n"
    "        return True\n"
    "    except (ImportError, OSError):\n"
    "        return False\n"
)


def _make_patcher() -> TextPatcher | None:
    if not _enabled():
        return None
    target = resolve_vllm_file("utils/import_utils.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN92 nixl_ep / deep_ep / mori trial-import guard (vllm PR #40154)",
        target_file=str(target),
        marker=GENESIS_MARKER,
        sub_patches=[
            TextPatch(
                name="pn92_nixl_ep",
                anchor=NIXL_EP_OLD,
                replacement=NIXL_EP_NEW,
                required=True,
            ),
            TextPatch(
                name="pn92_deep_ep",
                anchor=DEEP_EP_OLD,
                replacement=DEEP_EP_NEW,
                required=False,  # may not exist on older pins
            ),
            TextPatch(
                name="pn92_mori",
                anchor=MORI_OLD,
                replacement=MORI_NEW,
                required=False,  # may not exist on older pins
            ),
        ],
        upstream_drift_markers=[
            "Genesis PN92",
            "trial-import guard",
        ],
    )


def apply() -> tuple[str, str]:
    if not _enabled():
        return "skipped", "PN92 disabled (set GENESIS_ENABLE_PN92_NIXL_EP_TRIAL_IMPORT=1)"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "target file utils/import_utils.py not resolvable"
    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as fh:
        content = fh.read()
    if patcher.marker in content:
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m in content:
            return "skipped", f"drift marker {m!r} already in file"
    result, failure = patcher.apply()
    from sndr.kernel import result_to_wiring_status
    return result_to_wiring_status(
        result, failure,
        applied_message="PN92 nixl_ep/deep_ep/mori trial-import guards installed",
        patch_name=patcher.patch_name,
    )
