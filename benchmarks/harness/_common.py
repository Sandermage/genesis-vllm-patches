# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the Genesis v7.0 benchmark harness."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable


# ═══════════════════════════════════════════════════════════════════════════
#                          REPORT STRUCT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GateResult:
    """One Go/No-Go gate verdict."""
    name: str
    value: Any
    threshold: str
    passed: bool


@dataclass
class HarnessReport:
    name: str
    endpoint: str
    model: str
    started_at: str = ""
    finished_at: str = ""
    metrics: dict = field(default_factory=dict)
    gates: list[GateResult] = field(default_factory=list)
    raw: dict = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gates"] = [asdict(g) for g in self.gates]
        return d

    @property
    def all_passed(self) -> bool:
        return all(g.passed for g in self.gates)


# ═══════════════════════════════════════════════════════════════════════════
#                         STANDARD CLI
# ═══════════════════════════════════════════════════════════════════════════

def make_arg_parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog)
    p.add_argument(
        "--endpoint",
        default=os.environ.get(
            "GENESIS_BENCH_ENDPOINT",
            "http://192.168.1.10:8000/v1",
        ),
        help="OpenAI-compatible endpoint base URL (no trailing slash).",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("GENESIS_BENCH_API_KEY", "genesis-local"),
        help="API key / bearer token for auth.",
    )
    p.add_argument(
        "--model",
        default=os.environ.get(
            "GENESIS_BENCH_MODEL",
            "qwen3.6-35b-a3b-integration",
        ),
        help="served-model-name to target.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output JSON path. If omitted, write to "
             "benchmarks/results/<ISO>_<name>.json.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Emit only the final JSON to stdout (machine-readable runs).",
    )
    return p


def default_out_path(name: str) -> str:
    """Default JSON path: benchmarks/results/<ISO_UTC>_<name>.json."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(root, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return os.path.join(root, f"{ts}_{name}.json")


def write_report(report: HarnessReport, out_path: str, quiet: bool = False) -> None:
    data = report.to_dict()
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    if not quiet:
        print(json.dumps(data, indent=2, default=str))


def run_harness(
    name: str,
    body: Callable[[argparse.Namespace, HarnessReport], None],
) -> int:
    """Boilerplate wrapper for a harness script.

    `body` receives args + an initialized report; it is responsible for
    populating metrics/gates/raw. On exception we record it as `error`
    and exit 2. If `report.all_passed` is False, we exit 1. Else exit 0.
    """
    parser = make_arg_parser(name)
    args = parser.parse_args()
    out_path = args.out or default_out_path(name)

    report = HarnessReport(
        name=name,
        endpoint=args.endpoint,
        model=args.model,
        started_at=datetime.utcnow().isoformat() + "Z",
    )
    try:
        body(args, report)
    except SystemExit:
        raise
    except Exception as e:
        report.error = f"{type(e).__name__}: {e}"
        report.finished_at = datetime.utcnow().isoformat() + "Z"
        write_report(report, out_path, quiet=args.quiet)
        return 2

    report.finished_at = datetime.utcnow().isoformat() + "Z"
    write_report(report, out_path, quiet=args.quiet)
    return 0 if report.all_passed else 1


# ═══════════════════════════════════════════════════════════════════════════
#                    SIMPLE OPENAI-COMPAT CLIENT
# ═══════════════════════════════════════════════════════════════════════════

def post_chat(
    endpoint: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.0,
    seed: int | None = 42,
    timeout: float = 600.0,
    extra_body: dict | None = None,
) -> dict:
    """Single-shot chat completion. Uses stdlib only (no OpenAI SDK)."""
    import urllib.request
    import urllib.error
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if seed is not None:
        body["seed"] = seed
    if extra_body:
        body.update(extra_body)

    req = urllib.request.Request(
        url=f"{endpoint.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"HTTP {e.code} from {endpoint}: {e.read().decode('utf-8', 'replace')[:500]}"
        )


def post_completion_stream(
    endpoint: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    timeout: float = 600.0,
) -> tuple[float, int, float]:
    """Streaming completion, returns (ttft_sec, output_tokens, total_sec).

    Uses SSE streaming to separate TTFT from total generation time, which
    the TGS benchmark needs to compute decode tokens/sec.
    """
    import urllib.request
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        url=f"{endpoint.rstrip('/')}/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    t_start = time.perf_counter()
    first_token_time = None
    total_tokens = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = ev.get("usage") or {}
            if usage and "completion_tokens" in usage:
                total_tokens = int(usage["completion_tokens"])
            choices = ev.get("choices") or []
            if choices and first_token_time is None:
                # First content delta arrived
                text = (choices[0].get("text") or "") or (
                    choices[0].get("delta", {}).get("content") or ""
                )
                if text:
                    first_token_time = time.perf_counter()
    total = time.perf_counter() - t_start
    ttft = (first_token_time - t_start) if first_token_time else total
    return ttft, total_tokens, total


def make_tokenizer_calibrated_filler(
    endpoint: str,
    api_key: str,
    model: str,
    target_tokens: int,
    timeout: float = 30.0,
    max_iter: int = 6,
) -> tuple[str, int]:
    """Generate a filler prompt whose actual tokenization ≤ target_tokens.

    vLLM exposes `/v1/tokenize` (OpenAI-compat) which we use to ask the
    SERVER's own tokenizer. We iteratively generate a random filler,
    tokenize it, and trim/expand until within a small window of the
    target.

    Returns `(filler_text, measured_token_count)`.

    This is critical for long-context tests because English-word-like
    fillers tokenize at ~2-3 tokens/word, not 1:1 — we overshoot by 2×
    or undershoot by 3× if we guess. Fall-back: if /tokenize endpoint
    is unavailable, use the heuristic filler from tgs_decode._make_filler.
    """
    import json
    import random
    import string
    import urllib.error
    import urllib.request

    def _gen(n_words: int, seed: int = 42) -> str:
        random.seed(seed)
        # Use short common english words — likely 1 token each for BPE.
        common = [
            "the", "and", "of", "to", "in", "a", "is", "it", "for",
            "on", "that", "this", "with", "as", "at", "by", "be", "from",
            "was", "were", "an", "are", "or", "has", "have", "had", "not",
            "but", "can", "will", "all", "any", "some", "one", "two",
        ]
        return " ".join(common[i % len(common)] for i in range(n_words))

    def _tokenize_count(text: str) -> int | None:
        req = urllib.request.Request(
            url=f"{endpoint.rstrip('/')}/tokenize",
            data=json.dumps({"model": model, "prompt": text}).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return int(data.get("count", 0)) or len(data.get("tokens", []))
        except (urllib.error.URLError, urllib.error.HTTPError):
            return None

    # Start guess: 1 word ≈ 1 token for common-word filler.
    n_words = target_tokens
    best_text = _gen(n_words)
    best_count = _tokenize_count(best_text)

    if best_count is None:
        # Tokenize endpoint unavailable → heuristic fallback
        # (use only 0.75× target to play safe against overshoot).
        safe_words = max(1, int(target_tokens * 0.75))
        return _gen(safe_words), safe_words

    # Binary-ish search: adjust word count by ratio.
    for _ in range(max_iter):
        if abs(best_count - target_tokens) < max(32, target_tokens // 100):
            break
        if best_count == 0:
            break
        ratio = target_tokens / best_count
        n_words = max(1, int(n_words * ratio))
        best_text = _gen(n_words)
        cnt = _tokenize_count(best_text)
        if cnt is None:
            break
        best_count = cnt

    # Safety: if still above target, trim words until fits.
    while best_count is not None and best_count > target_tokens and n_words > 1:
        n_words = max(1, n_words - (best_count - target_tokens))
        best_text = _gen(n_words)
        cnt = _tokenize_count(best_text)
        if cnt is None:
            break
        best_count = cnt

    return best_text, best_count if best_count is not None else n_words


def probe_health(endpoint: str, timeout: float = 10.0) -> bool:
    """GET /health — returns True if 200."""
    import urllib.request
    import urllib.error
    base = endpoint.rstrip("/").rsplit("/v1", 1)[0] if endpoint.endswith("/v1") else endpoint.rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError):
        return False
