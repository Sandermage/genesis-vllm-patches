# SPDX-License-Identifier: Apache-2.0
"""Migrate registry `apply_module` fields from the v12.x back-compat shim
paths (vllm.sndr_core.*) to their canonical sndr.* targets.

Safety model — never guess a path:
  1. For each apply_module `vllm.sndr_core.A.B.C`, locate the shim file
     `vllm/sndr_core/A/B/C.py` and read its `from <canonical> import *`
     line. The shim itself authoritatively declares the canonical target.
  2. Verify the canonical module actually imports.
  3. Only then record the replacement.

Run with --apply to write the registry; default is a dry-run report.

⚠️ PREREQUISITE — coordinated change required (do NOT run --apply standalone):
   ``tests/unit/integrations/test_relocation_shims.py::test_registry_uses_new_path``
   pins apply_module to the EARLIER gemma4→spec_decode/probes relocation
   target, which is still in the ``vllm.sndr_core.*`` namespace. Migrating
   apply_module to the v12.x ``sndr.*`` canonical makes that test fail while
   fixing ``test_probes_family_contract`` — the two suites encode different
   relocation eras and currently DISAGREE. Before applying, update the
   REGISTERED_RELOCATIONS / PROBE_RELOCATIONS *new_path* entries in
   test_relocation_shims.py to the same ``sndr.*`` canonical so both agree.
   This is an operator-owned patch-domain decision (registry is boot-critical).
"""
from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
REGISTRY = REPO_ROOT / "sndr" / "dispatcher" / "registry.py"

_APPLY_RE = re.compile(r'"apply_module":\s*"(vllm\.sndr_core\.[A-Za-z0-9_.]+)"')
_SHIM_TARGET_RE = re.compile(r"^from\s+(sndr\.[A-Za-z0-9_.]+)\s+import\s+\*", re.M)


def shim_canonical(module_path: str) -> str | None:
    """Read the shim file for a vllm.sndr_core.* module and return the
    canonical sndr.* target it re-exports, or None if not resolvable."""
    rel = Path(*module_path.split(".")).with_suffix(".py")
    shim_file = REPO_ROOT / rel
    if not shim_file.is_file():
        return None
    m = _SHIM_TARGET_RE.search(shim_file.read_text())
    return m.group(1) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the registry")
    args = ap.parse_args()

    text = REGISTRY.read_text()
    olds = sorted(set(_APPLY_RE.findall(text)))
    print(f"apply_module entries on vllm.sndr_core.*: {len(olds)}")

    mapping: dict[str, str] = {}
    unresolved: list[str] = []
    import_fail: list[tuple[str, str]] = []
    for old in olds:
        canonical = shim_canonical(old)
        if not canonical:
            unresolved.append(old)
            continue
        try:
            importlib.import_module(canonical)
        except Exception as exc:  # noqa: BLE001
            # Env-gated imports (torch/triton/vllm absent on this host) are
            # NOT path problems — verify the canonical module file exists on
            # disk instead, then accept. Any other error → leave untouched.
            msg = str(exc)
            env_gated = isinstance(exc, ModuleNotFoundError) and any(
                f"No module named '{dep}'" in msg
                for dep in ("torch", "triton", "vllm.v1", "vllm._C", "flash_attn")
            )
            canonical_file = REPO_ROOT / Path(*canonical.split(".")).with_suffix(".py")
            if env_gated and canonical_file.is_file():
                mapping[old] = canonical
                continue
            import_fail.append((canonical, f"{type(exc).__name__}: {exc}"))
            continue
        mapping[old] = canonical

    print(f"resolved + import-verified: {len(mapping)}")
    if unresolved:
        print(f"\nUNRESOLVED (no shim / no target) — left untouched: {len(unresolved)}")
        for u in unresolved:
            print(f"  {u}")
    if import_fail:
        print(f"\nCANONICAL IMPORT FAILED — left untouched: {len(import_fail)}")
        for mod, err in import_fail:
            print(f"  {mod}  →  {err}")

    if not args.apply:
        print("\n(dry-run — pass --apply to write the registry)")
        return 0

    new_text = text
    for old, canonical in mapping.items():
        new_text = new_text.replace(
            f'"apply_module": "{old}"', f'"apply_module": "{canonical}"'
        )
    REGISTRY.write_text(new_text)
    print(f"\n✓ rewrote {len(mapping)} apply_module fields in {REGISTRY}")
    remaining = len(_APPLY_RE.findall(new_text))
    print(f"  remaining vllm.sndr_core.* apply_module: {remaining}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
