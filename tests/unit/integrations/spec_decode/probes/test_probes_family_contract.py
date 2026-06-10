# SPDX-License-Identifier: Apache-2.0
"""spec_decode/probes/ family contract (Phase 3 bucket 1, 2026-05-21).

Probes are diagnostic instrumentation, not runtime behavior changes.
This contract covers the registered probes (PN262, PN262B) that now
live under spec_decode/probes/ after the Phase 3 bucket 1 relocation.

Unregistered probes (PN241, PN248, PN258, PN266, PN267, PN268, PN269,
PN270, PN272) are listed here for import-safety coverage only — they
have no registry entry and are loaded directly via __init__.py env
gates.
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

# Registered probes (have PATCH_REGISTRY entry with family=spec_decode).
REGISTERED_PROBES = [
    (
        "sndr.engines.vllm.patches.spec_decode.probes.pn262_flash_attn_drafter_trace",
        "PN262",
    ),
    (
        "sndr.engines.vllm.patches.spec_decode.probes.pn262b_kv_alloc_trace",
        "PN262B",
    ),
]

# Unregistered probes — direct-load via __init__.py env gates, no
# registry entry. Listed for import + marker discipline only.
UNREGISTERED_PROBES = [
    "sndr.engines.vllm.patches.spec_decode.probes.pn241_mtp_trace",
    "sndr.engines.vllm.patches.spec_decode.probes.pn248_acceptance_trace",
    "sndr.engines.vllm.patches.spec_decode.probes.pn258_oracle_acceptance",
    "sndr.engines.vllm.patches.spec_decode.probes.pn266_propose_trace",
    "sndr.engines.vllm.patches.spec_decode.probes.pn267_kv_bridge_trace",
    "sndr.engines.vllm.patches.spec_decode.probes.pn268_drafter_blocks_origin",
    "sndr.engines.vllm.patches.spec_decode.probes.pn269_a0_block_table_trace",
    "sndr.engines.vllm.patches.spec_decode.probes.pn270_drafter_kv_proj_audit",
    "sndr.engines.vllm.patches.spec_decode.probes.pn272_gemma4_drafter_input_probe",
]


def _get_registry_field(patch_id: str, field: str) -> str | None:
    registry_path = (
        Path(__file__).resolve().parents[5]
        / "sndr" / "dispatcher" / "registry.py"
    )
    text = registry_path.read_text()
    m = re.search(rf'"{patch_id}":\s*\{{(.*?)^    \}}', text, flags=re.M | re.S)
    if not m:
        return None
    body = m.group(1)
    f_match = re.search(rf'"{field}":\s*"([^"]+)"', body)
    return f_match.group(1) if f_match else None


@pytest.mark.parametrize("module_path,patch_id", REGISTERED_PROBES)
class TestRegisteredProbesContract:
    """Per-probe invariants for registered probes."""

    def test_module_importable(self, module_path, patch_id):
        if module_path in sys.modules:
            del sys.modules[module_path]
        try:
            importlib.import_module(module_path)
        except ImportError as e:
            if "torch" in str(e) or "triton" in str(e):
                pytest.skip(f"{patch_id} requires torch/triton: {e}")
            raise

    def test_apply_function_exists(self, module_path, patch_id):
        mod = importlib.import_module(module_path)
        has_apply = hasattr(mod, "apply") and callable(mod.apply)
        has_should_apply = (
            hasattr(mod, "should_apply") and callable(mod.should_apply)
        )
        assert has_apply or has_should_apply, (
            f"{patch_id}: no apply() or should_apply() in {module_path}"
        )

    def test_registry_family_is_spec_decode(self, module_path, patch_id):
        family = _get_registry_field(patch_id, "family")
        assert family == "spec_decode", (
            f"{patch_id}: registry family={family!r}, expected 'spec_decode'"
        )

    def test_registry_apply_module_matches_filesystem(
        self, module_path, patch_id
    ):
        registered = _get_registry_field(patch_id, "apply_module")
        assert registered == module_path, (
            f"{patch_id}: registry apply_module={registered!r}, "
            f"expected {module_path!r}"
        )

    def test_module_torch_less_import_safety(self, module_path, patch_id):
        mod = importlib.import_module(module_path)
        src_text = Path(mod.__file__).read_text()
        for line in src_text.splitlines():
            if line.startswith(("import torch", "from torch")):
                pytest.fail(
                    f"{patch_id}: top-level torch import in {module_path}"
                )


@pytest.mark.parametrize("module_path", UNREGISTERED_PROBES)
class TestUnregisteredProbesImportSafety:
    """Import + lifecycle safety for direct-load probes (no registry entry)."""

    def test_module_importable(self, module_path):
        if module_path in sys.modules:
            del sys.modules[module_path]
        try:
            importlib.import_module(module_path)
        except ImportError as e:
            if "torch" in str(e) or "triton" in str(e):
                pytest.skip(f"requires torch/triton: {e}")
            raise

    def test_apply_function_exists(self, module_path):
        try:
            mod = importlib.import_module(module_path)
        except ImportError as e:
            if "torch" in str(e) or "triton" in str(e):
                pytest.skip(f"requires torch/triton: {e}")
            raise
        has_apply = hasattr(mod, "apply") and callable(mod.apply)
        has_should_apply = (
            hasattr(mod, "should_apply") and callable(mod.should_apply)
        )
        assert has_apply or has_should_apply, (
            f"no apply() or should_apply() in {module_path}"
        )


class TestProbesFamilyFilesystemConsistency:
    """Filesystem ↔ list drift detector."""

    def test_all_probes_files_listed(self):
        probes_dir = (
            Path(__file__).resolve().parents[5]
            / "sndr" / "engines" / "vllm" / "patches" / "spec_decode" / "probes"
        )
        listed = {p.rsplit(".", 1)[-1] for p, _ in REGISTERED_PROBES}
        listed |= {p.rsplit(".", 1)[-1] for p in UNREGISTERED_PROBES}
        on_disk = {
            f.stem for f in probes_dir.glob("*.py")
            if f.name != "__init__.py"
        }
        missing_from_list = on_disk - listed
        assert not missing_from_list, (
            f"probes/ files not covered by contract: {missing_from_list}"
        )

    def test_all_listed_files_exist(self):
        probes_dir = (
            Path(__file__).resolve().parents[5]
            / "sndr" / "engines" / "vllm" / "patches" / "spec_decode" / "probes"
        )
        listed = {p.rsplit(".", 1)[-1] for p, _ in REGISTERED_PROBES}
        listed |= {p.rsplit(".", 1)[-1] for p in UNREGISTERED_PROBES}
        on_disk = {
            f.stem for f in probes_dir.glob("*.py")
            if f.name != "__init__.py"
        }
        missing_from_disk = listed - on_disk
        assert not missing_from_disk, (
            f"contract lists files that don't exist: {missing_from_disk}"
        )
