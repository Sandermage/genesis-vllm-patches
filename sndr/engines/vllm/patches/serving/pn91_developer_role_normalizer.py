# SPDX-License-Identifier: Apache-2.0
"""PN91 — `developer` role pre-render normalizer.

Modern OpenAI Responses API sends `role="developer"` for the equivalent
of a system message (it's the "developer-side" instructions distinct
from end-user instructions). The official Qwen chat templates do NOT
handle this role and `raise_exception('Unexpected message role.')` at
template render time → HTTP 500 to client.

froggeric/Qwen-Fixed-Chat-Templates and our bundled qwen3.6-enhanced
both map `developer` → `system` inside their jinja, but that only
helps when the operator explicitly mounts an enhanced template via
`--chat-template`. The default model-bundled template (from the model
weights tokenizer_config.json) keeps the original behavior.

PN91 normalizes at the parser layer — BEFORE template render — so the
fix holds regardless of which chat template is active. Maps both the
local `role` variable AND `message["role"]` because downstream parts
of vLLM read from both surfaces.

Anchor: `vllm/entrypoints/chat_utils.py::_parse_chat_message_content`
at the `role = message["role"]` line (stable across dev9..dev209
inspected pins).

Env gate: `GENESIS_ENABLE_PN91_DEVELOPER_ROLE=1` (default OFF — opt-in
during the rollout window, will flip to default-on after PROD soak).
"""
from __future__ import annotations

import logging
import os

from vllm.sndr_core.detection.guards import resolve_vllm_file, vllm_install_root
from vllm.sndr_core.core import TextPatch, TextPatcher

log = logging.getLogger("genesis.wiring.pn72_developer_role_normalizer")

GENESIS_MARKER = "Genesis PN91 developer-role normalizer (pre-render)"


def _enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN91_DEVELOPER_ROLE", "0",
    ).strip().lower() in ("1", "true", "yes", "on")


# Stable anchor on dev209+g5536fc0c0 inspected 2026-05-13 in
# vllm/entrypoints/chat_utils.py::_parse_chat_message_content. The
# function-body opens with these exact two lines.
PN91_OLD = (
    "    role = message[\"role\"]\n"
    "    content = message.get(\"content\")\n"
)
PN91_NEW = (
    "    role = message[\"role\"]\n"
    "    # [Genesis PN91] developer→system role normalizer. OpenAI\n"
    "    # Responses API sends role='developer' as the developer-side\n"
    "    # instructions surface. Default Qwen templates raise on it.\n"
    "    # Map to 'system' BEFORE downstream parsers/template see it.\n"
    "    if role == \"developer\":\n"
    "        role = \"system\"\n"
    "        try:\n"
    "            message[\"role\"] = \"system\"  # downstream reads message too\n"
    "        except Exception:  # TypedDict frozen / unusual subtype\n"
    "            pass\n"
    "    content = message.get(\"content\")\n"
)


def _make_patcher() -> TextPatcher | None:
    if not _enabled():
        return None
    target = resolve_vllm_file("entrypoints/chat_utils.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN91 developer-role pre-render normalizer",
        target_file=str(target),
        marker=GENESIS_MARKER,
        sub_patches=[
            TextPatch(
                name="pn72_developer_role_normalize",
                anchor=PN91_OLD,
                replacement=PN91_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "Genesis PN91",
            "developer→system role normalizer",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN91 text-patch. Returns (wiring_status, message)."""
    if not _enabled():
        return "skipped", "PN91 disabled (set GENESIS_ENABLE_PN91_DEVELOPER_ROLE=1)"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "target file chat_utils.py not resolvable"
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
    from vllm.sndr_core.core import result_to_wiring_status
    return result_to_wiring_status(
        result, failure,
        applied_message="PN91 developer→system role normalized at parser layer",
        patch_name=patcher.patch_name,
    )
