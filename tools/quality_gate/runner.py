# SPDX-License-Identifier: Apache-2.0
"""Genesis quality-gate runner CLI — thin glue between the bash drivers and the
unit-tested probe/soak/verdict core.

The bash drivers (`scripts/verify_stress.sh`, `scripts/soak_continuous.sh`) call
subcommands here to: build a request payload, send a streaming request and
capture telemetry, and render a structured verdict line. Keeping the HTTP and
the load-bearing logic on this side means the bash layer stays a thin
orchestrator and the part worth testing (payload shapes, recall, verdicts) lives
in `probes.py` / `soak.py` and is covered by tests/unit/quality_gate.

All output that the bash layer parses is a single JSON object per call on
stdout, so the contract is explicit and shell-quote-safe.

Subcommands:
  gen-niah         --model M --scale N [--secret-out F] [--max-tokens K]
  gen-probe        --kind KIND --model M  (tool_prefill|ide_agent|multiturn|lcb|reasoning)
  ladder           --n-ctx N [--start S --step T --fraction F]
  scale-for        --target-tokens N --tok-per-scale R
  send             --url U --req F --timeout S   (streaming; emits telemetry JSON)
  verdict-niah     --kind K --http C --secret S --content-file F [--prompt-tokens N]
  verdict-probe    --kind K --http C [--content-len N --tool-calls N --completion N
                                      --finish R --min-tokens N]
  verdict-400      --kind K --target-tokens N --n-ctx N
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
import time
import urllib.error
import urllib.request

# Allow running both as `python3 -m quality_gate.runner` and as a path.
try:  # pragma: no cover - import shim
    from . import probes, soak  # noqa: F401  (soak imported for parity / future use)
except ImportError:  # pragma: no cover
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from quality_gate import probes, soak  # noqa: F401


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")


def _write_req(path: str, req: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(req, f)


# ---------------------------------------------------------------------------
# Streaming send — mirrors club-3090's send_streaming_niah: stream:true +
# include_usage, measures TTFT, returns http_code, content, prompt/completion
# tokens, tool-call count, finish_reason. No GPU needed; pure HTTP.
# ---------------------------------------------------------------------------
def cmd_send(args: argparse.Namespace) -> int:  # noqa: PLR0912, PLR0915 - SSE stream parser is inherently branchy
    with open(args.req, encoding="utf-8") as f:
        req = json.load(f)
    req["stream"] = True
    req["stream_options"] = {"include_usage": True}
    body = json.dumps(req).encode("utf-8")
    http_req = urllib.request.Request(
        args.url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    result: dict = {"http_code": 0, "error": None}
    t0 = time.time()
    ttft = None
    content_parts: list[str] = []
    usage = None
    finish_reason = ""
    tool_calls: dict[int, dict] = {}
    try:
        with urllib.request.urlopen(http_req, timeout=args.timeout) as resp:
            result["http_code"] = resp.getcode() or 200
            for raw in resp:
                line = raw.decode("utf-8", errors="ignore").rstrip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {}) or {}
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if ttft is None and (
                        delta.get("content") or reasoning or delta.get("tool_calls")
                    ):
                        ttft = time.time() - t0
                    if delta.get("content"):
                        content_parts.append(delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        slot = tool_calls.setdefault(idx, {"name": ""})
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                    if choices[0].get("finish_reason"):
                        finish_reason = choices[0]["finish_reason"]
                if "usage" in chunk and chunk["usage"]:
                    usage = chunk["usage"]
        wall = time.time() - t0
        content = "".join(content_parts)
        result.update(
            {
                "content": content,
                "content_len": len(content),
                "prompt_tokens": (usage or {}).get("prompt_tokens", 0),
                "completion_tokens": (usage or {}).get("completion_tokens", 0),
                "tool_calls": len([s for s in tool_calls.values() if s.get("name")]),
                "finish_reason": finish_reason,
                "ttft_ms": round(ttft * 1000) if ttft is not None else None,
                "wall_ms": round(wall * 1000),
            }
        )
    except urllib.error.HTTPError as e:
        result["http_code"] = e.code
        result["error"] = str(e)
        with contextlib.suppress(Exception):
            result["error_body"] = e.read().decode("utf-8", errors="replace")[:500]
    except Exception as e:  # noqa: BLE001 - report any transport failure as code 0
        result["http_code"] = 0
        result["error"] = f"{type(e).__name__}: {e}"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f)
    _emit(result)
    return 0


def cmd_gen_niah(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed) if args.seed is not None else random.Random()
    secret = probes.make_niah_secret(rng)
    req = probes.make_niah_request(
        args.model, args.scale, secret=secret, max_tokens=args.max_tokens
    )
    _write_req(args.req, req)
    if args.secret_out:
        with open(args.secret_out, "w", encoding="utf-8") as f:
            f.write(secret)
    _emit({"secret": secret, "scale": args.scale, "req": args.req})
    return 0


_PROBE_BUILDERS = {
    "tool_prefill": lambda m, a: probes.make_tool_prefill_request(m, a.target_chars),
    "ide_agent": lambda m, a: probes.make_ide_agent_request(m),
    "multiturn": lambda m, a: probes.make_multiturn_request(m),
    "lcb": lambda m, a: probes.make_lcb_coding_request(m),
    "reasoning": lambda m, a: probes.make_reasoning_request(m),
}


def cmd_gen_probe(args: argparse.Namespace) -> int:
    builder = _PROBE_BUILDERS.get(args.kind)
    if builder is None:
        _emit({"error": f"unknown probe kind: {args.kind}"})
        return 2
    req = builder(args.model, args)
    _write_req(args.req, req)
    _emit({"kind": args.kind, "req": args.req})
    return 0


def cmd_ladder(args: argparse.Namespace) -> int:
    rungs = probes.ceiling_ladder_rungs(
        args.n_ctx,
        start_tokens=args.start,
        step_tokens=args.step,
        fraction=args.fraction,
    )
    _emit({"n_ctx": args.n_ctx, "rungs": rungs})
    return 0


def cmd_scale_for(args: argparse.Namespace) -> int:
    try:
        scale = probes.scale_for_target_tokens(args.target_tokens, args.tok_per_scale)
    except ValueError as e:
        _emit({"error": str(e)})
        return 2
    _emit({"scale": scale})
    return 0


def cmd_verdict_niah(args: argparse.Namespace) -> int:
    content = ""
    if args.content_file:
        with open(args.content_file, encoding="utf-8") as f:
            content = f.read()
    v = probes.verdict_longctx_rung(
        args.kind, args.http, args.secret, content, prompt_tokens=args.prompt_tokens
    )
    _emit(v.as_dict())
    return 0


def cmd_verdict_probe(args: argparse.Namespace) -> int:
    v = probes.verdict_http_probe(
        args.kind,
        args.http,
        content_len=args.content_len,
        tool_calls=args.tool_calls,
        completion_tokens=args.completion,
        finish_reason=args.finish,
        min_tokens=args.min_tokens,
    )
    _emit(v.as_dict())
    return 0


def cmd_verdict_400(args: argparse.Namespace) -> int:
    v = probes.verdict_oversize_400(args.kind, args.target_tokens, args.n_ctx)
    _emit(v.as_dict())
    return 0


# ---------------------------------------------------------------------------
# Soak subcommands (continuous Cliff-2b ramp). State is a JSON conversation
# file the bash driver carries between turns.
# ---------------------------------------------------------------------------
def cmd_soak_init(args: argparse.Namespace) -> int:
    state = soak.continuous_initial_state(args.session)
    with open(args.state, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    _emit({"session": args.session, "state": args.state})
    return 0


def cmd_soak_request(args: argparse.Namespace) -> int:
    """Append the next turn's user message to state, write the request payload."""
    with open(args.state, encoding="utf-8") as f:
        state = json.load(f)
    spec = soak.turn_spec(args.turn)
    state["messages"].append({"role": "user", "content": spec["user"]})
    req = {
        "model": args.model,
        "messages": state["messages"],
        "max_tokens": spec["max_tokens"],
        "temperature": spec["temp"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        "tools": probes._ide_tool_schemas()[:3],
        "tool_choice": "auto",
    }
    with open(args.req, "w", encoding="utf-8") as f:
        json.dump(req, f)
    with open(args.state, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    _emit({"turn": args.turn, "req": args.req})
    return 0


def cmd_soak_ingest(args: argparse.Namespace) -> int:
    """Append the assistant response + a synthetic tool result for the NEXT turn,
    so accumulated context keeps ramping to Cliff-2b territory by turn 5."""
    with open(args.state, encoding="utf-8") as f:
        state = json.load(f)
    with open(args.metrics, encoding="utf-8") as f:
        metrics = json.load(f)
    content = metrics.get("content") or "(empty response)"
    state["messages"].append({"role": "assistant", "content": content})
    next_spec = next(
        (t for t in soak.CONTINUOUS_TURNS if t["turn"] == args.turn + 1), None
    )
    if next_spec and next_spec["tool_synth"]:
        tool_name, kind, target = next_spec["tool_synth"]
        filler = soak.synth_filler(kind, target)
        synth_id = f"call_synth_t{args.turn}_s{state['session_id']}"
        # Rewrite the just-appended assistant message with a synthetic tool_call
        # so the schema stays valid, then append the tool result.
        state["messages"][-1] = {
            "role": "assistant",
            "content": content or None,
            "tool_calls": [
                {
                    "id": synth_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps({"_synthetic": True}),
                    },
                }
            ],
        }
        state["messages"].append(
            {"role": "tool", "tool_call_id": synth_id, "content": filler}
        )
        state["fallback_tool_calls_synthesized"] = (
            state.get("fallback_tool_calls_synthesized", 0) + 1
        )
    with open(args.state, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    _emit({"turn": args.turn, "ingested": True})
    return 0


def cmd_soak_verdict(args: argparse.Namespace) -> int:
    """Compute the soak verdict from a turn-telemetry CSV-like JSON-lines file.

    ``--cliff2b`` switches to club-3090's verbatim Cliff-2b PASS gate (silent_empty
    == 0, growth < 200 MiB, retention >= 98%, errors == 0, plus the 5x5 ramp-shape
    check); without it the looser general soak gate is used.
    """
    rows = []
    with open(args.rows, encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            if stripped:
                rows.append(json.loads(stripped))
    if args.cliff2b:
        v = soak.compute_cliff2b_verdict(
            rows,
            boot_vram_mib=args.boot_vram,
            expected_sessions=args.expected_sessions,
            timed_out=bool(args.timed_out),
        )
    else:
        v = soak.compute_soak_verdict(
            rows,
            boot_vram_mib=args.boot_vram,
            growth_limit_mib=args.growth_limit,
            expected_sessions=args.expected_sessions,
            timed_out=bool(args.timed_out),
        )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(v.as_dict(), f, indent=2)
    _emit(v.as_dict())
    return v.exit_code


def cmd_soak_attribute(args: argparse.Namespace) -> int:
    """Compare an overlays-ON verdict JSON to a stripped verdict JSON."""
    with open(args.on, encoding="utf-8") as f:
        on = soak.SoakVerdict(**json.load(f))
    with open(args.stripped, encoding="utf-8") as f:
        stripped = soak.SoakVerdict(**json.load(f))
    result = soak.attribution_delta(on, stripped, patch=args.patch, topology_tp=args.tp)
    _emit(result.as_dict())
    return 0


def build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915 - flat subparser registration
    ap = argparse.ArgumentParser(prog="quality_gate.runner", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("gen-niah", help="write a NIAH request payload")
    p.add_argument("--model", required=True)
    p.add_argument("--scale", type=int, required=True)
    p.add_argument("--req", required=True)
    p.add_argument("--secret-out")
    p.add_argument("--max-tokens", type=int, default=30)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_gen_niah)

    p = sub.add_parser("gen-probe", help="write a non-NIAH probe request payload")
    p.add_argument("--kind", required=True, choices=list(_PROBE_BUILDERS))
    p.add_argument("--model", required=True)
    p.add_argument("--req", required=True)
    p.add_argument("--target-chars", type=int, default=100_000)
    p.set_defaults(func=cmd_gen_probe)

    p = sub.add_parser("ladder", help="emit the ceiling-ladder rungs for n_ctx")
    p.add_argument("--n-ctx", type=int, required=True)
    p.add_argument("--start", type=int, default=95_000)
    p.add_argument("--step", type=int, default=30_000)
    p.add_argument("--fraction", type=float, default=0.92)
    p.set_defaults(func=cmd_ladder)

    p = sub.add_parser("scale-for", help="target tokens -> filler scale")
    p.add_argument("--target-tokens", type=int, required=True)
    p.add_argument("--tok-per-scale", type=float, required=True)
    p.set_defaults(func=cmd_scale_for)

    p = sub.add_parser("send", help="send a streaming request, emit telemetry JSON")
    p.add_argument("--url", required=True)
    p.add_argument("--req", required=True)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--out")
    p.set_defaults(func=cmd_send)

    p = sub.add_parser("verdict-niah", help="verdict for a NIAH rung")
    p.add_argument("--kind", required=True)
    p.add_argument("--http", type=int, required=True)
    p.add_argument("--secret", required=True)
    p.add_argument("--content-file")
    p.add_argument("--prompt-tokens", type=int, default=0)
    p.set_defaults(func=cmd_verdict_niah)

    p = sub.add_parser("verdict-probe", help="verdict for an HTTP probe")
    p.add_argument("--kind", required=True)
    p.add_argument("--http", type=int, required=True)
    p.add_argument("--content-len", type=int, default=0)
    p.add_argument("--tool-calls", type=int, default=0)
    p.add_argument("--completion", type=int, default=0)
    p.add_argument("--finish", default="")
    p.add_argument("--min-tokens", type=int, default=0)
    p.set_defaults(func=cmd_verdict_probe)

    p = sub.add_parser("verdict-400", help="disambiguate an HTTP 400 on a ceiling rung")
    p.add_argument("--kind", required=True)
    p.add_argument("--target-tokens", type=int, required=True)
    p.add_argument("--n-ctx", type=int, required=True)
    p.set_defaults(func=cmd_verdict_400)

    p = sub.add_parser("soak-init", help="write the initial continuous-session state")
    p.add_argument("--state", required=True)
    p.add_argument("--session", type=int, required=True)
    p.set_defaults(func=cmd_soak_init)

    p = sub.add_parser(
        "soak-request", help="append next turn's user msg, write request"
    )
    p.add_argument("--state", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--turn", type=int, required=True)
    p.add_argument("--req", required=True)
    p.set_defaults(func=cmd_soak_request)

    p = sub.add_parser("soak-ingest", help="append assistant + synthetic tool result")
    p.add_argument("--state", required=True)
    p.add_argument("--metrics", required=True)
    p.add_argument("--turn", type=int, required=True)
    p.set_defaults(func=cmd_soak_ingest)

    p = sub.add_parser(
        "soak-verdict", help="compute the soak verdict from turn telemetry"
    )
    p.add_argument("--rows", required=True, help="JSON-lines telemetry file")
    p.add_argument("--boot-vram", type=int, required=True)
    p.add_argument("--growth-limit", type=int, default=200)
    p.add_argument("--expected-sessions", type=int, default=5)
    p.add_argument("--timed-out", type=int, default=0)
    p.add_argument(
        "--cliff2b",
        action="store_true",
        help="use club-3090's verbatim Cliff-2b gate (silent_empty==0, "
        "growth<200MiB, retention>=98%%, errors==0, 5x5 ramp-shape check)",
    )
    p.add_argument("--out")
    p.set_defaults(func=cmd_soak_verdict)

    p = sub.add_parser(
        "soak-attribute", help="overlays-ON vs stripped attribution delta"
    )
    p.add_argument("--on", required=True, help="overlays-ON verdict JSON")
    p.add_argument("--stripped", required=True, help="overlays-stripped verdict JSON")
    p.add_argument("--patch", default="overlays")
    p.add_argument("--tp", type=int, default=1)
    p.set_defaults(func=cmd_soak_attribute)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
