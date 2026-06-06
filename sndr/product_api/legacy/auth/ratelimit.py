# SPDX-License-Identifier: Apache-2.0
"""In-memory brute-force throttle for the login + 2FA endpoints.

Tracks recent failures per key (username); after ``threshold`` failures inside
``window`` seconds the key is throttled for ``lockout`` seconds. A success
clears the counter. State is in-memory (cleared on restart) — a deliberately
simple first line of defence; a permanent lock would enable a username-targeted
DoS, so the lock is temporary and auto-clearing.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


@dataclass
class LoginGuard:
    threshold: int = field(default_factory=lambda: _int_env("SNDR_AUTH_LOCK_THRESHOLD", 8))
    window: int = field(default_factory=lambda: _int_env("SNDR_AUTH_LOCK_WINDOW", 300))
    lockout: int = field(default_factory=lambda: _int_env("SNDR_AUTH_LOCK_SECONDS", 900))

    def __post_init__(self) -> None:
        # key -> (failure_timestamps, locked_until)
        self._state: dict[str, tuple[list[float], float]] = {}

    def retry_after(self, key: str, *, now: float | None = None) -> int:
        """Seconds the key must wait, or 0 if it may attempt now."""
        moment = time.time() if now is None else now
        fails, locked_until = self._state.get(key, ([], 0.0))
        if locked_until > moment:
            return int(locked_until - moment) + 1
        return 0

    def record_failure(self, key: str, *, now: float | None = None) -> int:
        """Record a failed attempt; returns the lockout seconds if now locked."""
        moment = time.time() if now is None else now
        fails, locked_until = self._state.get(key, ([], 0.0))
        fails = [t for t in fails if t > moment - self.window]
        fails.append(moment)
        if len(fails) >= self.threshold:
            locked_until = moment + self.lockout
            fails = []
        self._state[key] = (fails, locked_until)
        return int(locked_until - moment) if locked_until > moment else 0

    def record_success(self, key: str) -> None:
        self._state.pop(key, None)
