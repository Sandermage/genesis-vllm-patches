# SPDX-License-Identifier: Apache-2.0
"""PN127 — Qwen 3.5/3.6 enhanced chat-template auto-install.

Closes operator pain: before this patch, picking the correct chat
template for Qwen 3.5/3.6 hybrid_gdn_moe (interleaved-thinking +
XML tool_call, M2.5-style) required four manual steps:
  1. Know that the default template breaks on multi-turn tool-call
     (club-3090#53, club-3090#72 — 30-120 s SSE silence)
  2. Locate an enhanced version (fragmented across froggeric,
     Sandermage v7.62, club-3090 repos)
  3. Drop the .jinja file next to the checkpoint
  4. Pass `--chat-template /path/to/file.jinja` in launch args

PN127 removes steps 2-3: the enhanced template is baked into the
Genesis package as an asset and copied at apply() time into a
writable location known to the operator. Operators no longer hunt
for the file — it lives at a canonical data-image path right
after `pip install`.

Operator usage
==============

Launch:
  vllm serve <model> \
    ...
    --chat-template /tmp/genesis/chat_templates/qwen3.6_enhanced.jinja

or through the env var GENESIS_AUTO_CHAT_TEMPLATE_PATH (read by the
launch script and appended to `--chat-template`):
  GENESIS_AUTO_CHAT_TEMPLATE_PATH=/tmp/genesis/chat_templates/qwen3.6_enhanced.jinja

What's inside the template
==========================
  - Multimodal (image + video token rendering)
  - XML tool_call + tool_response wrapping (qwen3_coder parser
    compatible)
  - M2.5-style interleaved thinking:
    * historical assistant reasoning before the last user query
      hidden (no cache pollution)
    * assistant turns after the last user query keep <think>
    * generation always starts inside <think>
  - 7 fixes missing from the default Qwen template:
    1. empty `<think></think>` spam
    2. `</thinking>` hallucination (wrong close tag)
    3. unclosed think before tool call
    4. no-user-query startup crash
    5. developer role passthrough (for IDE agents)
    6. multi-turn tool-call SSE deadlock (club-3090#72)
    7. think→tool_call boundary truncation

Source
======
  - Base: Sandermage Genesis v7.62 chat_template_enhanced.jinja
  - Cross-validated: froggeric Qwen-Fixed-Chat-Templates
  - Live verify: club-3090 turbo dual config, 30/30 tool regression PASS

Safety
======
  - Opt-in: GENESIS_ENABLE_PN127_AUTO_CHAT_TEMPLATE=1
  - Idempotent: SHA256 compare — rewrites only on change
  - Never raises — a failed write only emits log.warning; the
    operator falls back to an explicit --chat-template path
  - Target: /tmp/genesis/chat_templates/qwen3.6_enhanced.jinja
    (or the GENESIS_CHAT_TEMPLATE_DIR override)
  - Source: vllm.sndr_core.assets.chat_templates.qwen3.6_enhanced.jinja

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Genesis-original 2026-05-15 — closes the club-3090#53 / club-3090#72
class of template-on-disk-dependency mishaps.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

log = logging.getLogger("genesis.wiring.pn127_chat_template_qwen36")

GENESIS_PN127_MARKER = "Genesis PN127 Qwen3.5/3.6 chat-template auto-install v1"
_ENV_ENABLE = "GENESIS_ENABLE_PN127_AUTO_CHAT_TEMPLATE"
_ENV_DISABLE = "GENESIS_DISABLE_PN127_AUTO_CHAT_TEMPLATE"
_ENV_DIR_OVERRIDE = "GENESIS_CHAT_TEMPLATE_DIR"

_DEFAULT_INSTALL_DIR = "/tmp/genesis/chat_templates"
_TEMPLATE_FILENAME = "qwen3.6_enhanced.jinja"


def _env_enabled() -> bool:
    """Default OFF until bench-validated."""
    if os.environ.get(_ENV_DISABLE, "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    val = os.environ.get(_ENV_ENABLE, "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _resolve_install_dir() -> Path:
    """Where to write the template. Operator may override via env."""
    custom = os.environ.get(_ENV_DIR_OVERRIDE, "").strip()
    if custom:
        return Path(custom)
    return Path(_DEFAULT_INSTALL_DIR)


def _read_packaged_template() -> str | None:
    """Read the template baked into the Genesis package.

    v12 (2026-06-08): canonical asset path is ``sndr.assets.chat_templates``.
    The legacy ``vllm.sndr_core.assets.chat_templates`` path is preserved as
    a fallback for the v12.x transition window (shipped only when the
    legacy compat tree is bind-mounted). After the legacy tree is deleted
    in v13.0, only the canonical path remains.
    """
    from importlib.resources import files
    for pkg in ("sndr.assets.chat_templates", "vllm.sndr_core.assets.chat_templates"):
        try:
            asset = files(pkg) / _TEMPLATE_FILENAME
            return asset.read_text(encoding="utf-8")
        except (ModuleNotFoundError, FileNotFoundError, OSError):
            continue
    log.warning(
        "[PN127] packaged template asset not readable from sndr.assets nor "
        "vllm.sndr_core.assets — install may be missing the chat_templates "
        "package data. Skip.",
    )
    return None


def _sha256_short(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


_APPLIED = False
_INSTALLED_PATH: Path | None = None


def apply() -> tuple[str, str]:
    """Bake the template into a writable location. Idempotent."""
    global _APPLIED, _INSTALLED_PATH

    if not _env_enabled():
        return "skipped", (
            f"PN127 disabled (set {_ENV_ENABLE}=1 to let Genesis "
            f"auto-install the Qwen3.5/3.6 enhanced chat-template; the "
            f"path will be logged below and available via --chat-template)"
        )

    if _APPLIED and _INSTALLED_PATH is not None and _INSTALLED_PATH.is_file():
        return "applied", (
            f"PN127 already installed at {_INSTALLED_PATH} (idempotent skip)"
        )

    template = _read_packaged_template()
    if template is None:
        return "skipped", "PN127 packaged template asset missing — see warning above"

    install_dir = _resolve_install_dir()
    target = install_dir / _TEMPLATE_FILENAME

    try:
        install_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return "skipped", f"PN127 cannot create install dir {install_dir}: {e}"

    # Check whether a write is needed (SHA comparison)
    expected_sha = _sha256_short(template)
    if target.is_file():
        try:
            current_sha = _sha256_short(target.read_text(encoding="utf-8"))
            if current_sha == expected_sha:
                _APPLIED = True
                _INSTALLED_PATH = target
                return "applied", (
                    f"PN127 template already at {target} "
                    f"(sha256:{expected_sha[:8]}, idempotent)"
                )
        except OSError:
            pass  # will overwrite below
        log.info(
            "[PN127] existing template at %s differs (sha changed) — overwriting",
            target,
        )

    try:
        target.write_text(template, encoding="utf-8")
    except OSError as e:
        return "skipped", f"PN127 cannot write to {target}: {e}"

    _APPLIED = True
    _INSTALLED_PATH = target

    log.info(
        "[PN127] installed Qwen3.5/3.6 enhanced chat-template at %s "
        "(sha256:%s). Operator: pass --chat-template %s in launch "
        "args to apply.",
        target, expected_sha[:8], target,
    )
    return "applied", (
        f"PN127 installed: enhanced chat-template copied to {target} "
        f"(sha256:{expected_sha[:8]}). Operator no longer hunts for "
        f"the file — launch vllm with --chat-template {target}. "
        f"Resolves club-3090#53 (multi-turn tool-call) + club-3090#72 "
        f"(SSE silence on narrative <tool_call>)."
    )


def is_applied() -> bool:
    return _APPLIED


def installed_path() -> Path | None:
    """Path of the installed template (or None when PN127 is not applied)."""
    return _INSTALLED_PATH


def revert() -> bool:
    """Remove the installed template. Idempotent."""
    global _APPLIED, _INSTALLED_PATH
    if not _APPLIED or _INSTALLED_PATH is None:
        return False
    try:
        if _INSTALLED_PATH.is_file():
            _INSTALLED_PATH.unlink()
    except OSError as e:
        log.warning("[PN127] revert: cannot remove %s: %s", _INSTALLED_PATH, e)
        return False
    _APPLIED = False
    _INSTALLED_PATH = None
    return True
