# SPDX-License-Identifier: Apache-2.0
"""``model_configs.emitters`` — pure-render layer over ModelConfig.

M.5.2 (2026-05-27): relocated the system-level emitters
(``to_launch_script`` + ``_build_vllm_cmd`` + ``_build_docker_cmd`` +
YAML I/O) out of the 2768-LOC ``model_configs/schema.py`` monolith.
The ModelConfig methods retain thin one-line delegations so the
~10 existing caller sites (``cli/launch.py``, ``cli/profile.py``,
``cli/compose.py``, ``cli/install.py``, ``cli/report.py``,
``runtime_command.py``, ``compat/recipes.py``, ``compat/presets.py``,
``compat/model_config_cli.py``, internal self-calls) keep working
unchanged.

Public functions:

  * :func:`render_launch_script` — full bash launch script
  * :func:`build_vllm_cmd` — vllm serve CLI parts
  * :func:`build_docker_cmd` — docker run command embedding the parts
  * :func:`shell_quote` — single source for ``shlex.quote`` usage
  * :func:`dump_yaml` / :func:`load_yaml` / :func:`to_plain_dict` /
    :func:`from_plain_dict` — YAML round-trip
"""
from __future__ import annotations

from .docker_cmd import build_docker_cmd
from .launch_script import render_launch_script
from .shell import shell_quote
from .vllm_cmd import build_vllm_cmd
from .yaml_io import (
    dump_yaml,
    from_plain_dict,
    load_yaml,
    to_plain_dict,
    validate_cfg,
)

__all__ = (
    "render_launch_script",
    "build_vllm_cmd",
    "build_docker_cmd",
    "shell_quote",
    "dump_yaml",
    "load_yaml",
    "to_plain_dict",
    "from_plain_dict",
    "validate_cfg",
)
