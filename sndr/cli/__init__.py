# SPDX-License-Identifier: Apache-2.0
"""sndr CLI — operator-facing commands.

Entry point: :func:`sndr.cli.main.main` (registered in pyproject.toml as
``sndr`` console script).

Architecture: each command is a self-contained module in ``commands/``.
The main dispatcher discovers commands at import time. Adding a new command
means dropping a file in ``commands/`` and registering it in
``commands/__init__.py``.
"""
