# SPDX-License-Identifier: Apache-2.0
"""PN80 — Genesis backport of vllm#41845 (Or Ozeri / IBM, MERGED 2026-05-07).

Fixes OOM during LoRA tensorizer deserialization by passing `device`
explicitly to `TensorDeserializer`. Without the device parameter,
tensorizer first deserializes to host RAM (full model size, possibly
2-50 GB depending on LoRA rank), then transfers to GPU — peak host
RAM blows up.

With `device=device` parameter, tensorizer streams directly to GPU,
peak host RAM stays ~constant.

Single-line text-patch in vllm/lora/lora_model.py.

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Backport of: vllm#41845 (Or Ozeri @ IBM, MERGED 2026-05-07).
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.locations import resolve_vllm_file, vllm_install_root
from sndr.kernel import (
    TextPatch, TextPatcher, result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pn80_lora_tensorizer_device")

GENESIS_PN80_MARKER = "Genesis PN80 LoRA tensorizer device (vllm#41845)"


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN80_LORA_TENSORIZER_DEVICE", ""
    ).strip().lower() in ("1", "true", "yes", "on")


ANCHOR_OLD = (
    "            tensors = TensorDeserializer(\n"
    "                lora_tensor_path,\n"
    "                dtype=tensorizer_config.dtype,\n"
    "                **tensorizer_args.deserialization_kwargs,\n"
    "            )\n"
)
ANCHOR_NEW = (
    "            tensors = TensorDeserializer(\n"
    "                lora_tensor_path,\n"
    "                dtype=tensorizer_config.dtype,\n"
    "                device=device,\n"
    "                **tensorizer_args.deserialization_kwargs,\n"
    "            )\n"
)


def _make_patcher() -> "TextPatcher | None":
    target = resolve_vllm_file("lora/lora_model.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN80 LoRA tensorizer device kwarg (vllm#41845)",
        target_file=str(target),
        marker=GENESIS_PN80_MARKER,
        sub_patches=[
            TextPatch(
                name="lora_tensorizer_device",
                anchor=ANCHOR_OLD,
                replacement=ANCHOR_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "device=device,\n                **tensorizer_args.deserialization_kwargs"
        ],
        patch_id="PN80",
    )


def apply() -> tuple[str, str]:
    """Apply PN80. Returns (status, reason) per wiring contract."""
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN80")
    log_decision("PN80", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "lora/lora_model.py not found in vllm install"

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message="PN80 applied — LoRA tensorizer device kwarg active",
        patch_name="PN80 LoRA tensorizer device",
    )
