# SPDX-License-Identifier: Apache-2.0
"""Release gate: the runtime __version__ must match the packaging version, and
must not carry a pre-release/.dev suffix on a tagged release."""
from __future__ import annotations

import pathlib
import re


def _read(p):
    return pathlib.Path(__file__).resolve().parents[3].joinpath(p).read_text()


def test_runtime_version_matches_pyproject():
    rt = re.search(r'__version__\s*:\s*str\s*=\s*"([^"]+)"', _read("sndr/version.py")).group(1)
    pp = re.search(r'(?m)^version\s*=\s*"([^"]+)"', _read("pyproject.toml")).group(1)
    assert rt == pp, f"version.py ({rt}) != pyproject.toml ({pp}) — reconcile before release"


def test_release_version_has_no_dev_suffix():
    rt = re.search(r'__version__\s*:\s*str\s*=\s*"([^"]+)"', _read("sndr/version.py")).group(1)
    assert not re.search(r'\.(dev|rc|a|b)\d*$', rt), f"{rt} is a pre-release suffix; finalize for public release"
