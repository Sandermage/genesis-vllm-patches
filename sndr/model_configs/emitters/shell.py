# SPDX-License-Identifier: Apache-2.0
"""Shell-quoting helper shared by every emitter — M.5.2.

Single source of truth so launch-script / docker-run / quadlet
emitters all quote consistently. Previously inlined in
``model_configs/schema.py`` as ``_shell_quote``.
"""
from __future__ import annotations

import shlex


def shell_quote(value: str) -> str:
    """Quote a value so generated shell commands preserve it exactly."""
    return shlex.quote(str(value))
