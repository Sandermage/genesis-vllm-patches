# SPDX-License-Identifier: Apache-2.0
"""At-rest secrets store for the GUI's remote-host features.

Holds sensitive values the operator would otherwise retype every session —
SSH passwords and engine API keys — encrypted at rest. These never live in the
plaintext ``hosts.json`` and are never returned in list payloads; callers fetch
them server-side only when establishing a connection.

Backends (resolved in order, override with ``SNDR_SECRETS_BACKEND``):

* ``keyring`` — the OS keychain (macOS Keychain / Secret Service / WinCred).
  Preferred: the key material is managed by the OS, not by us.
* ``file`` — a Fernet-encrypted JSON file under ``SNDR_HOME/gui``. The Fernet
  key is kept in the OS keychain when available, else in a ``0600`` key file
  beside the data (encryption-at-rest for a single-user trusted box).
* ``memory`` — last resort when neither ``cryptography`` nor a keychain is
  present: in-process only, not persisted. ``available()`` is still True but
  secrets do not survive a restart.

All public functions are import-safe and degrade gracefully — the daemon runs
without the optional deps; remote features that need a secret report that it is
unavailable rather than crashing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

_SERVICE = "sndr-gui"
_KEYRING_KEY_NAME = "secrets-fernet-key"


def _state_dir() -> Path:
    from vllm.sndr_core.locations.project_paths import install_root

    return install_root() / "gui"


def _secrets_path() -> Path:
    return _state_dir() / "secrets.enc"


def _keyfile_path() -> Path:
    return _state_dir() / ".secrets.key"


# ─── Optional backend probing ───────────────────────────────────────────────


def _keyring_usable() -> bool:
    """True only when a real OS keychain backend is wired (not the null/fail one)."""
    try:
        import keyring
        from keyring.backends import fail as _fail

        backend = keyring.get_keyring()
        # The "null"/"fail" backends mean no real keychain is available.
        name = type(backend).__module__
        if isinstance(backend, _fail.Keyring):
            return False
        if "null" in name.lower() or "fail" in name.lower():
            return False
        return True
    except Exception:
        return False


def _have_cryptography() -> bool:
    try:
        import cryptography.fernet  # noqa: F401
        return True
    except Exception:
        return False


_BACKEND: Optional[str] = None


def reset_backend_cache() -> None:
    """Drop the cached backend choice (used by tests after env changes)."""
    global _BACKEND, _MEMORY
    _BACKEND = None
    _MEMORY = {}


def backend_name() -> str:
    """Resolve and cache which backend is in effect."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    override = (os.environ.get("SNDR_SECRETS_BACKEND") or "").strip().lower()
    if override in ("keyring", "file", "memory"):
        _BACKEND = override
    elif _keyring_usable():
        _BACKEND = "keyring"
    elif _have_cryptography():
        _BACKEND = "file"
    else:
        _BACKEND = "memory"
    return _BACKEND


def available() -> bool:
    """Whether secrets can be stored at all (memory counts, but is volatile)."""
    return backend_name() in ("keyring", "file", "memory")


def persistent() -> bool:
    """Whether stored secrets survive a daemon restart."""
    return backend_name() in ("keyring", "file")


# ─── keyring backend ────────────────────────────────────────────────────────


def _kr_set(name: str, value: str) -> None:
    import keyring

    keyring.set_password(_SERVICE, name, value)


def _kr_get(name: str) -> Optional[str]:
    import keyring

    return keyring.get_password(_SERVICE, name)


def _kr_delete(name: str) -> bool:
    import keyring
    import keyring.errors

    try:
        keyring.delete_password(_SERVICE, name)
        return True
    except keyring.errors.PasswordDeleteError:
        return False
    except Exception:
        return False


# ─── file backend (Fernet) ──────────────────────────────────────────────────


def _fernet():
    from cryptography.fernet import Fernet

    key = _load_or_create_fernet_key()
    return Fernet(key)


def _load_or_create_fernet_key() -> bytes:
    # Prefer the OS keychain for the key material even on the file backend.
    if _keyring_usable():
        import keyring

        existing = keyring.get_password(_SERVICE, _KEYRING_KEY_NAME)
        if existing:
            return existing.encode("ascii")
        from cryptography.fernet import Fernet

        key = Fernet.generate_key()
        keyring.set_password(_SERVICE, _KEYRING_KEY_NAME, key.decode("ascii"))
        return key
    # Else a 0600 key file beside the data.
    path = _keyfile_path()
    if path.is_file():
        return path.read_bytes()
    from cryptography.fernet import Fernet

    key = Fernet.generate_key()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(key)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return key


def _file_read() -> dict[str, str]:
    path = _secrets_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text("utf-8")) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def _file_write(data: dict[str, str]) -> None:
    path = _secrets_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".enc.tmp")
    tmp.write_text(json.dumps(data), "utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(path)


def _file_set(name: str, value: str) -> None:
    fernet = _fernet()
    data = _file_read()
    data[name] = fernet.encrypt(value.encode("utf-8")).decode("ascii")
    _file_write(data)


def _file_get(name: str) -> Optional[str]:
    token = _file_read().get(name)
    if not token:
        return None
    try:
        return _fernet().decrypt(token.encode("ascii")).decode("utf-8")
    except Exception:
        return None


def _file_delete(name: str) -> bool:
    data = _file_read()
    if name not in data:
        return False
    del data[name]
    _file_write(data)
    return True


# ─── memory backend ─────────────────────────────────────────────────────────

_MEMORY: dict[str, str] = {}


# ─── public API ─────────────────────────────────────────────────────────────


def set_secret(name: str, value: str) -> None:
    """Store ``value`` under ``name``. An empty value clears the secret."""
    if not value:
        delete_secret(name)
        return
    backend = backend_name()
    if backend == "keyring":
        _kr_set(name, value)
    elif backend == "file":
        _file_set(name, value)
    else:
        _MEMORY[name] = value


def get_secret(name: str) -> Optional[str]:
    """Return the stored secret, or ``None`` if absent/undecryptable."""
    backend = backend_name()
    if backend == "keyring":
        return _kr_get(name)
    if backend == "file":
        return _file_get(name)
    return _MEMORY.get(name)


def has_secret(name: str) -> bool:
    return get_secret(name) is not None


def delete_secret(name: str) -> bool:
    """Remove a secret. Returns True if one existed."""
    backend = backend_name()
    if backend == "keyring":
        return _kr_delete(name)
    if backend == "file":
        return _file_delete(name)
    return _MEMORY.pop(name, None) is not None


__all__ = [
    "available",
    "backend_name",
    "delete_secret",
    "get_secret",
    "has_secret",
    "persistent",
    "reset_backend_cache",
    "set_secret",
]
