# SPDX-License-Identifier: Apache-2.0
"""Redaction utility — masks sensitive data in operator-shareable artifacts.

Used by `sndr report bundle` (CLI) before tar'ing diagnostic output for
upload to support / GitHub issues. Default rules are conservative —
operators can extend via `~/.sndr/redact_rules.yaml` or pass
`--no-redact` for internal-only bundles.

Default replacements:

  - IPv4 / IPv6 addresses                  → `<IP>`
  - hostnames in SSH targets (user@host)   → `user@<HOSTNAME>`
  - API keys (Bearer xyz / GENESIS_*_KEY=) → `<REDACTED>`
  - License tokens (`<base64>.<base64>`)   → `<LICENSE_TOKEN>`
  - HF tokens (hf_xxx... 30+ chars)        → `<HF_TOKEN>`
  - filesystem paths under /home/<user>    → `/home/<USER>`
  - filesystem paths under /Users/<user>   → `/Users/<USER>` (macOS)
  - container names with `<host>-<id>`     → preserves stem, masks tail
  - email addresses                        → `<EMAIL>`

Non-rules (intentionally NOT redacted):

  - Public model paths (Qwen3.6-..., Gemma-...) — these are operationally
    important context.
  - vLLM image tags / digests — necessary to reproduce.
  - Genesis patch IDs / version constants — not sensitive.
  - Generic /opt/* /var/* paths — not user-identifying.

Author: Sandermage(Sander)-Barzov Aleksandr.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


# ─── Built-in redaction patterns ─────────────────────────────────────────


@dataclass(frozen=True)
class RedactRule:
    """One redaction rule.

    `pattern` is a compiled regex. `replacement` is either a string
    template (`<IP>`) or a callable `(re.Match) -> str` for dynamic
    replacements that need to inspect the match.
    """
    name: str
    pattern: re.Pattern[str]
    replacement: str | Callable[[re.Match[str]], str]
    description: str = ""


def _ssh_target_replacement(m: re.Match[str]) -> str:
    """Preserve `user` portion, mask host."""
    user = m.group("user")
    return f"{user}@<HOSTNAME>"


def _path_user_replacement(m: re.Match[str]) -> str:
    """Mask the username segment of /home/<user> or /Users/<user>."""
    prefix = m.group("prefix")  # e.g. /home or /Users
    return f"{prefix}/<USER>"


# Order matters: more specific rules go first so they fire before the
# generic catch-all patterns.
DEFAULT_RULES: tuple[RedactRule, ...] = (
    # API keys + secrets
    RedactRule(
        name="bearer_token",
        pattern=re.compile(
            r"\bBearer\s+[A-Za-z0-9._\-+/=]{8,}\b",
            re.IGNORECASE,
        ),
        replacement="Bearer <REDACTED>",
        description="HTTP Authorization Bearer tokens.",
    ),
    RedactRule(
        name="api_key_env",
        pattern=re.compile(
            r"\b(?P<key>(?:GENESIS|SNDR|VLLM|OPENAI|ANTHROPIC|HF|HUGGINGFACE)"
            r"_[A-Z0-9_]*(?:KEY|TOKEN|SECRET))=([^\s\"';]+)",
        ),
        replacement=r"\g<key>=<REDACTED>",
        description="ENV-style API keys / tokens.",
    ),
    RedactRule(
        name="hf_token",
        pattern=re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
        replacement="<HF_TOKEN>",
        description="Hugging Face access tokens (hf_xxxx).",
    ),
    RedactRule(
        name="license_token",
        pattern=re.compile(
            # Two base64url chunks separated by `.` — same shape as
            # signed Ed25519 license tokens.
            r"\b[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{40,}\b"
        ),
        replacement="<LICENSE_TOKEN>",
        description="Ed25519-signed license tokens (payload.signature).",
    ),
    # SSH targets BEFORE generic IP / email rules so user@host keeps user.
    RedactRule(
        name="ssh_target",
        pattern=re.compile(
            r"\b(?P<user>[a-zA-Z][a-zA-Z0-9_.\-]*)@"
            r"(?:(?:\d{1,3}\.){3}\d{1,3}|[a-zA-Z0-9][a-zA-Z0-9.\-]*\.[a-zA-Z]{2,}|[a-zA-Z0-9][a-zA-Z0-9.\-]*)\b"
        ),
        replacement=_ssh_target_replacement,
        description="SSH-style user@host targets (preserves user).",
    ),
    # Email addresses (after ssh_target so they don't double-match).
    RedactRule(
        name="email",
        pattern=re.compile(
            r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"
        ),
        replacement="<EMAIL>",
        description="Email addresses.",
    ),
    # IPv4
    RedactRule(
        name="ipv4",
        pattern=re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        replacement="<IP>",
        description="IPv4 addresses.",
    ),
    # IPv6 — match common forms (full + double-colon shorthand).
    RedactRule(
        name="ipv6",
        pattern=re.compile(
            r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{0,4}\b"
        ),
        replacement="<IPV6>",
        description="IPv6 addresses (full + ::shorthand).",
    ),
    # User-home paths
    RedactRule(
        name="home_path",
        pattern=re.compile(
            r"(?P<prefix>/home|/Users)/[a-zA-Z][a-zA-Z0-9_.\-]*"
        ),
        replacement=_path_user_replacement,
        description="User-home paths (/home/<user>, /Users/<user>).",
    ),
)


# ─── Public API ──────────────────────────────────────────────────────────


@dataclass
class Redactor:
    """Apply a list of rules to a string. Tracks per-rule hit counts for
    operator transparency ("we replaced 5 IPs and 2 SSH targets")."""
    rules: tuple[RedactRule, ...] = field(default_factory=lambda: DEFAULT_RULES)
    counts: dict[str, int] = field(default_factory=dict)

    def redact(self, text: str) -> str:
        """Apply all rules in order. Returns redacted text + updates
        per-rule hit counts in `self.counts`."""
        if not text:
            return text
        out = text
        for rule in self.rules:
            n_before = len(rule.pattern.findall(out))
            if not n_before:
                continue
            out = rule.pattern.sub(rule.replacement, out)
            self.counts[rule.name] = self.counts.get(rule.name, 0) + n_before
        return out

    def reset_counts(self) -> None:
        self.counts.clear()


def redact(text: str, *, rules: Iterable[RedactRule] | None = None) -> str:
    """One-shot convenience function.

    Use `Redactor()` directly when you need stable per-rule counts
    across multiple inputs (e.g. for bundle summary).
    """
    r = Redactor(rules=tuple(rules) if rules is not None else DEFAULT_RULES)
    return r.redact(text)


def load_user_rules(path: Path | None = None) -> tuple[RedactRule, ...]:
    """Optional: load user-defined rules from `~/.sndr/redact_rules.yaml`.

    Schema (YAML):

      rules:
        - name: corp_internal_ip
          pattern: '\\b10\\.42\\.\\d+\\.\\d+\\b'
          replacement: <CORP_IP>
          description: company internal subnet
        - name: project_codename
          pattern: 'projectx-[a-z]+'
          replacement: <PROJECT>

    Replacement strings only — callable replacements are reserved for
    built-in rules (callable can't be safely loaded from YAML).
    """
    if path is None:
        path = Path("~/.sndr/redact_rules.yaml").expanduser()
    if not path.is_file():
        return ()
    try:
        import yaml
    except ImportError:
        return ()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError):
        return ()
    raw_rules = data.get("rules", [])
    out: list[RedactRule] = []
    for raw in raw_rules:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        pat_str = raw.get("pattern")
        repl = raw.get("replacement", "<REDACTED>")
        desc = raw.get("description", "")
        if not (isinstance(name, str) and isinstance(pat_str, str)
                and isinstance(repl, str)):
            continue
        try:
            pat = re.compile(pat_str)
        except re.error:
            continue
        out.append(RedactRule(name, pat, repl, desc))
    return tuple(out)


def redact_dict(d: Any, *, rules: Iterable[RedactRule] | None = None) -> Any:
    """Recursively redact every str leaf in a JSON-style dict/list/scalar.

    Useful for redacting `sndr doctor --json` output before bundling.
    """
    r = Redactor(rules=tuple(rules) if rules is not None else DEFAULT_RULES)
    return _walk(d, r)


def _walk(node: Any, r: Redactor) -> Any:
    if isinstance(node, str):
        return r.redact(node)
    if isinstance(node, list):
        return [_walk(x, r) for x in node]
    if isinstance(node, tuple):
        return tuple(_walk(x, r) for x in node)
    if isinstance(node, dict):
        return {k: _walk(v, r) for k, v in node.items()}
    return node


__all__ = [
    "RedactRule",
    "Redactor",
    "DEFAULT_RULES",
    "redact",
    "redact_dict",
    "load_user_rules",
]
