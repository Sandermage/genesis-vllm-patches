# SPDX-License-Identifier: Apache-2.0
"""Managed API tokens (personal access tokens) for the Product API.

Operators issue named, revocable Bearer tokens for programmatic / CI access to
the read-only Product API instead of sharing the single legacy ``SNDR_GUI_TOKEN``.

Design (GitHub-PAT style):
* A token looks like ``sndr_pat_<id>_<secret>`` where ``id`` is an 8-char hex
  record id and ``secret`` is 48 hex chars (192 bits). Both are hex so the
  ``_`` delimiter is unambiguous.
* Only a scrypt **hash of the secret** is persisted (reusing the password KDF);
  the plaintext is shown exactly once at issue time.
* Verification is O(1): the token carries its record id, so we look up one
  record and scrypt-verify the secret (no scan over all tokens).
* The store file is written atomically with ``0600`` perms, like the user store.
"""
from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .passwords import hash_password, verify_password

_PREFIX = "sndr_pat"


@dataclass(frozen=True)
class ApiToken:
    id: str
    label: str
    prefix: str
    created_at: float
    last_used: Optional[float]
    created_by: str


def _to_token(record: dict[str, Any]) -> ApiToken:
    return ApiToken(
        id=str(record.get("id", "")),
        label=str(record.get("label", "")),
        prefix=str(record.get("prefix", "")),
        created_at=float(record.get("created_at", 0.0)),
        last_used=(float(record["last_used"]) if record.get("last_used") is not None else None),
        created_by=str(record.get("created_by", "")),
    )


class TokenStore:
    def __init__(self, auth_dir: Path | str) -> None:
        self._dir = Path(auth_dir)
        self._path = self._dir / "api_tokens.json"

    def _read(self) -> list[dict[str, Any]]:
        if not self._path.is_file():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def _write(self, rows: list[dict[str, Any]]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self._dir, 0o700)
        except OSError:
            pass
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, self._path)

    def issue(self, label: str, *, created_by: str) -> tuple[str, ApiToken]:
        tid = secrets.token_hex(4)        # 8 hex chars
        secret = secrets.token_hex(24)    # 48 hex chars / 192 bits
        plaintext = f"{_PREFIX}_{tid}_{secret}"
        record = {
            "id": tid,
            "label": (label or "").strip() or "api-token",
            "prefix": f"{_PREFIX}_{tid}",
            "hash": hash_password(secret),
            "created_at": time.time(),
            "last_used": None,
            "created_by": created_by,
        }
        rows = self._read()
        rows.append(record)
        self._write(rows)
        return plaintext, _to_token(record)

    def list(self) -> list[ApiToken]:
        return [_to_token(row) for row in self._read()]

    def revoke(self, token_id: str) -> bool:
        rows = self._read()
        remaining = [row for row in rows if str(row.get("id")) != token_id]
        if len(remaining) == len(rows):
            return False
        self._write(remaining)
        return True

    def verify(self, plaintext: str) -> Optional[str]:
        """Return the ``created_by`` username for a valid token, else ``None``.
        Updates ``last_used`` on success."""
        parts = (plaintext or "").split("_")
        if len(parts) != 4 or parts[0] != "sndr" or parts[1] != "pat":
            return None
        tid, secret = parts[2], parts[3]
        rows = self._read()
        for row in rows:
            if str(row.get("id")) == tid:
                if verify_password(secret, str(row.get("hash", ""))):
                    row["last_used"] = time.time()
                    self._write(rows)
                    return str(row.get("created_by", ""))
                return None
        return None
