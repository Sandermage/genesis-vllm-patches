# SPDX-License-Identifier: Apache-2.0
"""Plugin signature verification + sandbox — Wave 4.3 / production roadmap §8.2.

Why
───
Genesis's plugin system (`vllm.sndr_core.compat.plugins`) discovers
external patches via Python entry points. Without signature checks
ANY pip-installed package declaring `vllm_sndr_core_plugins`
entry-point can register patches that monkey-patch vLLM internals.
This is a supply-chain attack vector.

Wave 4.3 adds:
  1. **SHA-256 manifest verification** — plugin authors ship a
     `genesis-plugin-manifest.json` with file hashes; we verify the
     installed plugin matches.
  2. **Optional Ed25519 signature** — manifest can be signed with
     a key whose pubkey is allowlisted in
     `~/.sndr/plugin_trust_anchors.json`.
  3. **Sandbox metadata** — declare what the plugin patches (vllm
     module paths) so we can refuse plugins that touch `core` paths
     without explicit allowlist override.

Usage
─────

```python
from sndr.compat.plugin_signature import verify_plugin

ok, reason = verify_plugin(plugin_dir="/path/to/installed/plugin")
if not ok:
    log.error(f"plugin rejected: {reason}")
```

Operator policy
───────────────
- Default mode: **warn-only** — log signature failures but proceed.
- Strict mode (env `SNDR_PLUGIN_STRICT_SIGNATURES=1`): refuse to load
  unsigned/invalid-signature plugins.
- Trust anchor mode (when `~/.sndr/plugin_trust_anchors.json` exists):
  only signed plugins from the allowlisted keys load.

Author: Sandermage(Sander)-Barzov Aleksandr.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("genesis.compat.plugin_signature")


# ─── Manifest data class ───────────────────────────────────────────────


@dataclass(frozen=True)
class PluginManifest:
    """Validated plugin manifest.

    JSON schema:

        {
          "schema_version": 1,
          "name": "my-plugin",
          "version": "0.1.0",
          "files": {
            "relative/path.py": "<sha256-hex>"
          },
          "patches_modules": ["vllm.something"],
          "signature_algo": "ed25519",
          "signature_key_id": "key-alias",
          "signature": "<base64url-bytes>",
          "manifest_hash": "<sha256-hex of canonical-JSON without signature fields>"
        }
    """
    schema_version: int
    name: str
    version: str
    files: dict[str, str]
    patches_modules: tuple[str, ...]
    signature_algo: Optional[str] = None
    signature_key_id: Optional[str] = None
    signature: Optional[str] = None
    manifest_hash: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PluginManifest":
        if not isinstance(d, dict):
            raise ValueError("manifest must be a dict")
        sv = d.get("schema_version")
        if sv != 1:
            raise ValueError(f"manifest schema_version must be 1 (got {sv!r})")
        for required in ("name", "version", "files", "patches_modules"):
            if required not in d:
                raise ValueError(f"manifest missing required field: {required!r}")
        files = d["files"]
        if not isinstance(files, dict):
            raise ValueError("manifest 'files' must be a dict")
        for path, h in files.items():
            if not isinstance(path, str) or not isinstance(h, str):
                raise ValueError(
                    f"manifest 'files' must map str→str (got {path!r}→{h!r})"
                )
            if len(h) != 64 or not all(c in "0123456789abcdef" for c in h.lower()):
                raise ValueError(
                    f"manifest 'files' hash for {path!r} not sha256 hex"
                )
        modules = d["patches_modules"]
        if not isinstance(modules, list) or not all(isinstance(m, str) for m in modules):
            raise ValueError("'patches_modules' must be list[str]")
        return cls(
            schema_version=sv,
            name=str(d["name"]),
            version=str(d["version"]),
            files=dict(files),
            patches_modules=tuple(modules),
            signature_algo=d.get("signature_algo"),
            signature_key_id=d.get("signature_key_id"),
            signature=d.get("signature"),
            manifest_hash=d.get("manifest_hash"),
        )

    def to_canonical_json_unsigned(self) -> bytes:
        """Canonical JSON of manifest WITHOUT signature fields, for hashing.

        Used during both signing and verification — both sides must
        compute the hash over the same bytes.
        """
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "files": dict(sorted(self.files.items())),
            "patches_modules": list(self.patches_modules),
        }
        return json.dumps(
            d, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")


# ─── File hashing ───────────────────────────────────────────────────────


def _file_sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file_hashes(plugin_dir: Path, manifest: PluginManifest) -> tuple[bool, list[str]]:
    """Verify each file in `manifest.files` matches the actual content.

    Returns (all_ok, list_of_violations). Violations are paths whose
    actual hash differs OR the file is missing.
    """
    violations: list[str] = []
    for rel_path, expected_hex in manifest.files.items():
        target = plugin_dir / rel_path
        if not target.is_file():
            violations.append(f"{rel_path}: missing")
            continue
        actual = _file_sha256_hex(target)
        if actual.lower() != expected_hex.lower():
            violations.append(
                f"{rel_path}: hash mismatch (expected {expected_hex[:8]}…, "
                f"got {actual[:8]}…)"
            )
    return (not violations), violations


# ─── Signature verification ─────────────────────────────────────────────


def _load_trust_anchors() -> dict[str, bytes]:
    """Load `~/.sndr/plugin_trust_anchors.json` → {key_id: pubkey_bytes}.

    File schema:
        {
          "key-alias": "<base64url ed25519 public key>"
        }

    Empty/missing file → empty dict (no signed plugins accepted).
    """
    path = Path("~/.sndr/plugin_trust_anchors.json").expanduser()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.warning("[plugin_sig] trust_anchors load failed: %s", e)
        return {}
    if not isinstance(data, dict):
        log.warning("[plugin_sig] trust_anchors must be {key_id: pubkey}")
        return {}
    out: dict[str, bytes] = {}
    for key_id, b64 in data.items():
        if not isinstance(key_id, str) or not isinstance(b64, str):
            continue
        try:
            out[key_id] = base64.urlsafe_b64decode(b64.encode("ascii") + b"==")
        except Exception:
            log.warning("[plugin_sig] failed to decode pubkey for %s", key_id)
    return out


def verify_signature(manifest: PluginManifest) -> tuple[bool, str]:
    """Verify Ed25519 signature on the manifest.

    Returns (ok, reason). Possible non-OK reasons:
      - "no_signature" — manifest unsigned
      - "no_trust_anchors" — no anchors loaded → can't verify
      - "unknown_key_id"  — key_id not in trust anchors
      - "verify_failed"   — cryptographic verification failed
      - "cryptography_unavailable" — library missing
    """
    if not manifest.signature:
        return False, "no_signature"
    if manifest.signature_algo not in ("ed25519",):
        return False, f"unsupported_algo: {manifest.signature_algo!r}"
    if not manifest.signature_key_id:
        return False, "missing_key_id"

    anchors = _load_trust_anchors()
    if not anchors:
        return False, "no_trust_anchors"
    pubkey_bytes = anchors.get(manifest.signature_key_id)
    if pubkey_bytes is None:
        return False, f"unknown_key_id: {manifest.signature_key_id!r}"

    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
        from cryptography.exceptions import InvalidSignature
    except ImportError:
        return False, "cryptography_unavailable"

    try:
        pub = Ed25519PublicKey.from_public_bytes(pubkey_bytes)
    except Exception as e:
        return False, f"pubkey_load_failed: {e}"

    sig_bytes = base64.urlsafe_b64decode(
        manifest.signature.encode("ascii") + b"=="
    )
    payload = manifest.to_canonical_json_unsigned()
    try:
        pub.verify(sig_bytes, payload)
    except InvalidSignature:
        return False, "verify_failed"
    except Exception as e:
        return False, f"verify_error: {e}"
    return True, "verified"


# ─── Sandbox classification ─────────────────────────────────────────────


# Module prefixes plugins are NOT permitted to patch unless the operator
# explicitly opts in via SNDR_PLUGIN_ALLOW_CORE=1.
_PROTECTED_MODULES = (
    "sndr.",                    # Genesis core (v12 canonical namespace)
    "vllm.sndr_core",           # Genesis core (v11 shim namespace, compat window)
    "vllm.sndr_engine",         # Genesis engine
    "vllm.v1.core",             # vllm KV cache manager
    "vllm.v1.sample.rejection_sampler",  # spec-decode verify
)


def classify_sandbox(manifest: PluginManifest) -> tuple[str, list[str]]:
    """Classify plugin's risk level based on which modules it patches.

    Returns (level, reasons). Levels:
      - "safe"      — touches only operator-friendly surfaces
      - "moderate"  — touches vllm internals but not protected core
      - "risky"     — touches protected modules → strict mode refuses
    """
    risky_hits: list[str] = []
    for module in manifest.patches_modules:
        for protected in _PROTECTED_MODULES:
            if module.startswith(protected):
                risky_hits.append(f"{module} (protected: {protected})")
                break
    if risky_hits:
        return "risky", risky_hits
    if any(m.startswith("vllm.") for m in manifest.patches_modules):
        return "moderate", []
    return "safe", []


# ─── High-level entry point ─────────────────────────────────────────────


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of `verify_plugin()`."""
    ok: bool
    level: str          # safe / moderate / risky / rejected
    file_violations: list[str]
    signature_status: str
    sandbox_reasons: list[str]
    summary: str


def _strict_mode() -> bool:
    return os.environ.get("SNDR_PLUGIN_STRICT_SIGNATURES", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _allow_core_mode() -> bool:
    return os.environ.get("SNDR_PLUGIN_ALLOW_CORE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def verify_plugin(
    plugin_dir: Path | str,
    *,
    manifest_filename: str = "genesis-plugin-manifest.json",
) -> VerificationResult:
    """End-to-end verification of an installed plugin.

    Args:
      plugin_dir: directory with the installed plugin's source.
      manifest_filename: name of the manifest JSON in plugin_dir.

    Returns VerificationResult. The caller decides whether to load
    the plugin based on `result.ok`.
    """
    plugin_dir = Path(plugin_dir)
    manifest_path = plugin_dir / manifest_filename
    if not manifest_path.is_file():
        return VerificationResult(
            ok=not _strict_mode(),
            level="rejected" if _strict_mode() else "moderate",
            file_violations=[],
            signature_status="no_manifest",
            sandbox_reasons=[],
            summary=f"manifest {manifest_filename!r} not found in {plugin_dir}",
        )

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = PluginManifest.from_dict(raw)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        return VerificationResult(
            ok=False, level="rejected",
            file_violations=[],
            signature_status="manifest_invalid",
            sandbox_reasons=[],
            summary=f"manifest invalid: {e}",
        )

    # File hashes
    files_ok, file_violations = verify_file_hashes(plugin_dir, manifest)

    # Signature
    sig_ok, sig_status = verify_signature(manifest)

    # Sandbox classification
    level, sandbox_reasons = classify_sandbox(manifest)

    # Decide
    if not files_ok:
        return VerificationResult(
            ok=False, level="rejected",
            file_violations=file_violations,
            signature_status=sig_status,
            sandbox_reasons=sandbox_reasons,
            summary=(
                f"file hash mismatches ({len(file_violations)}); "
                f"refuses load to prevent tampering"
            ),
        )

    if level == "risky" and not _allow_core_mode():
        return VerificationResult(
            ok=False, level="rejected",
            file_violations=[],
            signature_status=sig_status,
            sandbox_reasons=sandbox_reasons,
            summary=(
                "plugin patches protected modules; set "
                "SNDR_PLUGIN_ALLOW_CORE=1 to override"
            ),
        )

    if _strict_mode() and not sig_ok:
        return VerificationResult(
            ok=False, level="rejected",
            file_violations=[],
            signature_status=sig_status,
            sandbox_reasons=sandbox_reasons,
            summary=(
                f"strict mode requires valid signature "
                f"(got {sig_status}); refusing load"
            ),
        )

    return VerificationResult(
        ok=True, level=level,
        file_violations=[],
        signature_status=sig_status,
        sandbox_reasons=sandbox_reasons,
        summary=(
            f"verified: file hashes OK, signature {sig_status}, "
            f"sandbox level {level}"
        ),
    )


__all__ = [
    "PluginManifest",
    "VerificationResult",
    "verify_plugin",
    "verify_file_hashes",
    "verify_signature",
    "classify_sandbox",
]
