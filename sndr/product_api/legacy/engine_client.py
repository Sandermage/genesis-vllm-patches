# SPDX-License-Identifier: Apache-2.0
"""Live vLLM engine client — the bridge from the GUI to a *running* engine.

The Product API otherwise describes the static project (catalog, patches,
plans). This module reaches the OpenAI-compatible vLLM server that a launch
actually starts and reports its live state: health, loaded models, version,
Prometheus metrics (queue depth, KV-cache usage, throughput, TTFT/TPOT,
spec-decode acceptance), and a proxied test chat completion.

Design:
* stdlib ``urllib`` only (import-safe, no new dependency);
* the engine is addressed by a validated **host** (or operator-set env URLs) —
  never an arbitrary client-supplied URL, with fixed ports/paths (anti-SSRF);
* short timeouts; an unreachable engine yields a structured "down" payload,
  never a 500;
* metric names are matched generically with fallbacks across vLLM versions
  (e.g. ``kv_cache_usage_perc`` vs older ``gpu_cache_usage_perc``).
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Optional

# Hostname / IPv4 only — no scheme, path, port or credentials. Blocks SSRF via
# a crafted ``host`` query param; the scheme/port/path are always fixed here.
_HOST_RE = re.compile(r"^[A-Za-z0-9._-]{1,253}$")

_DEFAULT_API_PORT = 8000
_DEFAULT_METRICS_PORT = 8001
_MAX_TOKENS_CAP = 4096  # bound a GUI-initiated generation (anti-DoS)


def _clamp_tokens(value: Any) -> int:
    try:
        return max(1, min(_MAX_TOKENS_CAP, int(value)))
    except (TypeError, ValueError):
        return 256

# Per-engine last metrics scrape, for token-throughput deltas across polls.
_LAST_SCRAPE: dict[str, tuple[float, float, float]] = {}

# Per-engine rolling metrics history (for GUI sparklines / trend lines).
_HISTORY: dict[str, list[dict[str, Any]]] = {}
_HISTORY_CAP = 60


def _safe_host(host: Optional[str]) -> str:
    candidate = (host or os.environ.get("SNDR_RUNTIME_HOST") or "127.0.0.1").strip()
    if not _HOST_RE.match(candidate):
        return "127.0.0.1"
    return candidate


def _resolve_api_key(explicit: Optional[str]) -> Optional[str]:
    """Pick the engine API key: explicit (from the GUI) wins, else operator env.

    Engines launched with ``--api-key`` (e.g. the 35B PROD on :8102) reject
    ``/v1/*`` with 401 unless a ``Authorization: Bearer`` header is sent.
    """
    key = (explicit or "").strip()
    if key:
        return key
    for name in ("SNDR_ENGINE_API_KEY", "VLLM_API_KEY", "SNDR_OPENAI_API_KEY", "OPENAI_API_KEY"):
        val = (os.environ.get(name) or "").strip()
        if val:
            return val
    return None


def _auth_headers(api_key: Optional[str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def resolve_engine(host: Optional[str] = None, port: Optional[int] = None) -> dict[str, str]:
    """Resolve the engine's base/metrics URLs. An explicit ``port`` wins (so the
    GUI chat can target any engine, e.g. 8101/8102); otherwise operator env, then
    the validated host with fixed default ports."""
    safe = _safe_host(host)
    if port is not None:
        try:
            p = int(port)
        except (TypeError, ValueError):
            p = 0
        if 1 <= p <= 65535:
            base = f"http://{safe}:{p}/v1"
            return {"host": safe, "base_url": base, "root_url": base[:-3].rstrip("/"),
                    "metrics_url": f"http://{safe}:{p + 1}/metrics"}
    base = os.environ.get("SNDR_OPENAI_BASE_URL", "").strip() or f"http://{safe}:{_DEFAULT_API_PORT}/v1"
    base = base.rstrip("/")
    root = base[:-3].rstrip("/") if base.endswith("/v1") else base
    metrics = os.environ.get("SNDR_METRICS_URL", "").strip() or f"http://{safe}:{_DEFAULT_METRICS_PORT}/metrics"
    return {"host": safe, "base_url": base, "root_url": root, "metrics_url": metrics}


def _get(url: str, *, timeout: float = 3.0, api_key: Optional[str] = None) -> tuple[int, str]:
    headers = {"Accept": "application/json, text/plain", **_auth_headers(api_key)}
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed scheme/port
        return response.status, response.read().decode("utf-8", "replace")


def _post_json(url: str, payload: dict, *, timeout: float = 60.0, api_key: Optional[str] = None) -> tuple[int, str]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json", **_auth_headers(api_key)}
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed scheme/port
        return response.status, response.read().decode("utf-8", "replace")


def engine_status(host: Optional[str] = None, *, port: Optional[int] = None, timeout: float = 3.0, api_key: Optional[str] = None) -> dict[str, Any]:
    """Probe ``/health``, ``/version`` and ``/v1/models`` on the running engine."""
    eng = resolve_engine(host, port)
    key = _resolve_api_key(api_key)
    result: dict[str, Any] = {
        "reachable": False,
        "host": eng["host"],
        "base_url": eng["base_url"],
        "metrics_url": eng["metrics_url"],
        "version": None,
        "models": [],
        "error": None,
    }
    try:
        status, _ = _get(f"{eng['root_url']}/health", timeout=timeout)
        result["reachable"] = 200 <= status < 300
    except (urllib.error.URLError, OSError, ValueError) as exc:
        result["error"] = _describe(exc)
        return result
    # Best-effort enrichment — failures here don't flip "reachable".
    try:
        _, version_body = _get(f"{eng['root_url']}/version", timeout=timeout, api_key=key)
        result["version"] = (json.loads(version_body) or {}).get("version")
    except Exception:
        pass
    try:
        _, models_body = _get(f"{eng['base_url']}/models", timeout=timeout, api_key=key)
        data = json.loads(models_body).get("data", [])
        result["models"] = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]
    except Exception:
        pass
    return result


def probe_host(host: Optional[str] = None, port: int = 8000, *, timeout: float = 3.0, api_key: Optional[str] = None) -> dict[str, Any]:
    """Probe a host's vLLM OpenAI endpoint for reachability + version + models.

    Used by the Hosts fleet view to show live engine status per host. Honours
    the same SSRF host allow-list as ``engine_status`` and uses an explicit
    port (the per-host ``engine_port``) rather than the fixed default. The
    optional ``api_key`` is forwarded so a key-protected engine still lists its
    served models. No mutation; read-only HTTP GET with a short timeout."""
    safe = _safe_host(host)
    key = _resolve_api_key(api_key)
    try:
        safe_port = int(port)
    except (TypeError, ValueError):
        safe_port = 8000
    if not (1 <= safe_port <= 65535):
        safe_port = 8000
    root = f"http://{safe}:{safe_port}"
    result: dict[str, Any] = {
        "reachable": False,
        "host": safe,
        "port": safe_port,
        "base_url": f"{root}/v1",
        "version": None,
        "models": [],
        "latency_ms": None,
        "error": None,
    }
    started = time.monotonic()
    try:
        status, _ = _get(f"{root}/health", timeout=timeout)
        result["reachable"] = 200 <= status < 300
        result["latency_ms"] = round((time.monotonic() - started) * 1000, 1)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        result["error"] = _describe(exc)
        return result
    try:
        _, version_body = _get(f"{root}/version", timeout=timeout, api_key=key)
        result["version"] = (json.loads(version_body) or {}).get("version")
    except Exception:
        pass
    try:
        _, models_body = _get(f"{root}/v1/models", timeout=timeout, api_key=key)
        data = json.loads(models_body).get("data", [])
        result["models"] = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]
    except Exception:
        pass
    return result


def parse_prometheus(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    """Parse Prometheus text-exposition into ``{name: [(labels, value), ...]}``."""
    out: dict[str, list[tuple[dict[str, str], float]]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            name = line[: line.index("{")]
            close = line.index("}")
            label_str = line[line.index("{") + 1 : close]
            tail = line[close + 1 :].strip().split()
            labels = _parse_labels(label_str)
        else:
            parts = line.split()
            name, labels, tail = parts[0], {}, parts[1:]
        if not tail:
            continue
        try:
            value = float(tail[0])
        except ValueError:
            continue
        out.setdefault(name, []).append((labels, value))
    return out


def _parse_labels(label_str: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for match in re.finditer(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"', label_str):
        labels[match.group(1)] = match.group(2)
    return labels


def _sum(metrics: dict, *names: str) -> Optional[float]:
    for name in names:
        if name in metrics:
            return sum(value for _labels, value in metrics[name])
    return None


def _first(metrics: dict, *names: str) -> Optional[float]:
    for name in names:
        if metrics.get(name):
            return metrics[name][0][1]
    return None


def _avg_from_histogram(metrics: dict, base: str) -> Optional[float]:
    total = _sum(metrics, f"{base}_sum")
    count = _sum(metrics, f"{base}_count")
    if total is None or not count:
        return None
    return total / count


def engine_metrics(host: Optional[str] = None, *, port: Optional[int] = None, timeout: float = 3.0, now: Optional[float] = None) -> dict[str, Any]:
    """Scrape + distill the engine's Prometheus metrics into operator KPIs."""
    eng = resolve_engine(host, port)
    payload: dict[str, Any] = {"reachable": False, "metrics_url": eng["metrics_url"], "error": None, "kpis": {}}
    try:
        status, text = _get(eng["metrics_url"], timeout=timeout)
        if not (200 <= status < 300):
            payload["error"] = f"metrics endpoint returned {status}"
            return payload
    except (urllib.error.URLError, OSError, ValueError) as exc:
        payload["error"] = _describe(exc)
        return payload

    metrics = parse_prometheus(text)
    payload["reachable"] = True

    prompt_tokens = _sum(metrics, "vllm:prompt_tokens_total")
    gen_tokens = _sum(metrics, "vllm:generation_tokens_total")

    # Token throughput from the delta since the last scrape (cumulative counters).
    moment = time.time() if now is None else now
    gen_per_s: Optional[float] = None
    prev = _LAST_SCRAPE.get(eng["metrics_url"])
    if prev and gen_tokens is not None:
        dt = moment - prev[0]
        if dt > 0 and gen_tokens >= prev[2]:
            gen_per_s = round((gen_tokens - prev[2]) / dt, 1)
    if prompt_tokens is not None or gen_tokens is not None:
        _LAST_SCRAPE[eng["metrics_url"]] = (moment, prompt_tokens or 0.0, gen_tokens or 0.0)

    kpis = {
        "requests_running": _sum(metrics, "vllm:num_requests_running"),
        "requests_waiting": _sum(metrics, "vllm:num_requests_waiting"),
        # KV-cache fraction: new name first, older fallback. Value is 0..1.
        "kv_cache_usage": _first(metrics, "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
        "prompt_tokens_total": prompt_tokens,
        "generation_tokens_total": gen_tokens,
        "generation_toks_per_s": gen_per_s,
        "ttft_avg_s": _avg_from_histogram(metrics, "vllm:time_to_first_token_seconds"),
        "tpot_avg_s": _avg_from_histogram(metrics, "vllm:time_per_output_token_seconds"),
        "e2e_latency_avg_s": _avg_from_histogram(metrics, "vllm:e2e_request_latency_seconds"),
        "requests_success_total": _sum(metrics, "vllm:request_success_total"),
        "preemptions_total": _sum(metrics, "vllm:num_preemptions_total"),
        # Genesis stack runs MTP spec-decode — surface acceptance explicitly.
        "spec_decode_acceptance_rate": _first(metrics, "vllm:spec_decode_acceptance_rate"),
        "spec_decode_accepted_total": _sum(metrics, "vllm:spec_decode_num_accepted_tokens_total"),
        "spec_decode_draft_total": _sum(metrics, "vllm:spec_decode_num_draft_tokens_total"),
    }
    payload["kpis"] = {key: value for key, value in kpis.items() if value is not None}
    payload["metric_families"] = len(metrics)

    # Append a sample to the rolling history for trend lines / sparklines.
    sample = {
        "ts": round(moment, 1),
        "throughput": gen_per_s,
        "kv_cache": kpis.get("kv_cache_usage"),
        "running": kpis.get("requests_running"),
        "waiting": kpis.get("requests_waiting"),
    }
    history = _HISTORY.setdefault(eng["metrics_url"], [])
    history.append(sample)
    if len(history) > _HISTORY_CAP:
        del history[: len(history) - _HISTORY_CAP]
    payload["history"] = list(history)
    return payload


def _apply_sampling(body: dict[str, Any], payload: dict[str, Any]) -> None:
    """Forward optional, clamped OpenAI sampling params from the GUI."""
    for key, lo, hi in (("top_p", 0.0, 1.0), ("presence_penalty", -2.0, 2.0), ("frequency_penalty", -2.0, 2.0)):
        value = payload.get(key)
        if value is not None:
            try:
                body[key] = min(hi, max(lo, float(value)))
            except (TypeError, ValueError):
                pass
    stop = payload.get("stop")
    if isinstance(stop, list):
        cleaned = [str(item) for item in stop[:4] if str(item).strip()]
        if cleaned:
            body["stop"] = cleaned
    # Reasoning models: forward enable_thinking via chat_template_kwargs so the
    # template renders the <think> path (vLLM convention). Whitelisted keys only.
    ctk = payload.get("chat_template_kwargs")
    if isinstance(ctk, dict):
        allowed = {k: ctk[k] for k in ("enable_thinking",) if k in ctk and isinstance(ctk[k], bool)}
        if allowed:
            body["chat_template_kwargs"] = allowed


def stream_chat(payload: dict[str, Any], *, host: Optional[str] = None, port: Optional[int] = None, timeout: float = 120.0, api_key: Optional[str] = None):
    """Stream a chat completion from the engine, yielding ND-JSON lines.

    Each line is a JSON object: ``{"delta": "..."}`` for token chunks, then a
    final ``{"done": true, "ttft_ms", "latency_ms", "tokens", "usage"}``. An
    error yields ``{"error": "..."}``. Generator form so FastAPI can stream it.
    """
    eng = resolve_engine(host, port)
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    body = {
        "model": payload.get("model") or "default",
        "messages": messages,
        "max_tokens": _clamp_tokens(payload.get("max_tokens", 256)),
        "temperature": float(payload.get("temperature", 0.7)),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    _apply_sampling(body, payload)
    request = urllib.request.Request(
        f"{eng['base_url']}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream", **_auth_headers(_resolve_api_key(api_key))},
        method="POST",
    )
    started = time.time()
    ttft: Optional[float] = None
    tokens = 0
    usage: dict[str, Any] = {}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed scheme/port
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []) or []:
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        if ttft is None:
                            ttft = time.time() - started
                        tokens += 1
                        yield json.dumps({"delta": delta})
    except Exception as exc:  # noqa: BLE001 - surface as a stream error line
        yield json.dumps({"error": _describe(exc)})
        return
    yield json.dumps(
        {
            "done": True,
            "ttft_ms": round((ttft or 0.0) * 1000),
            "latency_ms": round((time.time() - started) * 1000),
            "tokens": int(usage.get("completion_tokens", tokens)),
            "usage": usage,
        }
    )


def engine_chat(
    payload: dict[str, Any], *, host: Optional[str] = None, port: Optional[int] = None, timeout: float = 60.0, api_key: Optional[str] = None
) -> dict[str, Any]:
    """Proxy a non-streaming chat completion to the engine (a GUI smoke test)."""
    eng = resolve_engine(host, port)
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    body = {
        "model": payload.get("model") or "default",
        "messages": messages,
        "max_tokens": _clamp_tokens(payload.get("max_tokens", 256)),
        "temperature": float(payload.get("temperature", 0.7)),
        "stream": False,
    }
    _apply_sampling(body, payload)
    started = time.time()
    status, text = _post_json(f"{eng['base_url']}/chat/completions", body, timeout=timeout, api_key=_resolve_api_key(api_key))
    elapsed_ms = round((time.time() - started) * 1000)
    data = json.loads(text) if text else {}
    if not (200 <= status < 300):
        detail = data.get("error", {}).get("message") if isinstance(data.get("error"), dict) else text[:300]
        raise EngineError(detail or f"engine returned {status}")
    choice = (data.get("choices") or [{}])[0]
    return {
        "reply": (choice.get("message") or {}).get("content", ""),
        "model": data.get("model", body["model"]),
        "usage": data.get("usage", {}),
        "finish_reason": choice.get("finish_reason"),
        "latency_ms": elapsed_ms,
    }


def chat_raw(
    messages: list[dict[str, Any]],
    *,
    tools: Optional[list[dict[str, Any]]] = None,
    model: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Low-level chat completion that supports OpenAI tool-calling.

    Unlike :func:`engine_chat` (which returns only the reply text), this returns
    the full assistant message — including any ``tool_calls`` — so a tool-using
    agent loop can act on them. ``tools`` is the OpenAI ``tools`` array; the
    engine must be launched with auto tool-choice + a tool-call parser
    (hermes / qwen3_xml) for tool_calls to be emitted, otherwise the model just
    answers in plain text (graceful degradation).
    """
    eng = resolve_engine(host, port)
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    body: dict[str, Any] = {
        "model": model or "default",
        "messages": messages,
        "max_tokens": _clamp_tokens(max_tokens),
        "temperature": float(temperature),
        "stream": False,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    status, text = _post_json(f"{eng['base_url']}/chat/completions", body, timeout=timeout, api_key=_resolve_api_key(api_key))
    data = json.loads(text) if text else {}
    if not (200 <= status < 300):
        detail = data.get("error", {}).get("message") if isinstance(data.get("error"), dict) else text[:300]
        raise EngineError(detail or f"engine returned {status}")
    choice = (data.get("choices") or [{}])[0]
    return {
        "message": choice.get("message") or {"role": "assistant", "content": ""},
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage", {}),
        "model": data.get("model", body["model"]),
    }


class EngineError(Exception):
    """Engine returned an error to a proxied request."""


def _describe(exc: Exception) -> str:
    reason = getattr(exc, "reason", None)
    if reason is not None:
        return f"{type(exc).__name__}: {reason}"
    return f"{type(exc).__name__}: {exc}"
