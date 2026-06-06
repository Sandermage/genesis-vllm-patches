# SPDX-License-Identifier: Apache-2.0
"""Scoped ops-copilot — a tool-calling assistant over the read-only Product API.

The copilot answers operator questions about the Genesis vLLM stack by calling a
curated set of **read-only** tools (catalog, doctor, presets, patches, capacity,
deployment *planning*) and synthesising the results. It never mutates anything:

* Every tool in :data:`_TOOLS` is read-only or dry-run. There is no tool that
  deploys, applies a config, patches, or runs a remote command.
* Actions that change state are surfaced as *proposed actions* — a plan plus a
  deep-link the human follows into the existing, apply-gated UI (Installer,
  Launch Plan, …). The copilot proposes; a person clicks apply.

This keeps the copilot inside the project's security posture (read-only by
default; ``SNDR_ENABLE_APPLY`` gates real execution elsewhere) while still being
useful. The agent loop (:func:`run_copilot`) takes an injected ``chat_fn`` so it
is unit-testable without a live tool-calling engine.

Extending it is one entry in :data:`_TOOLS`: a name, a JSON-schema for the args,
and a read-only handler returning a compact dict. See ``_tool`` registrations
below — add capacity/eval/bench/log-analysis tools the same way.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional

# ── tool registry ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for the function arguments
    handler: Callable[[dict[str, Any]], dict[str, Any]]  # read-only; returns a compact dict
    category: str = "read"  # "read" | "plan" — surfaced in the UI tool list


def _schema(props: dict[str, Any], required: Optional[list[str]] = None) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required or [], "additionalProperties": False}


# ── read-only handlers (compact, token-frugal results) ───────────────────────


def _h_overview(_args: dict[str, Any]) -> dict[str, Any]:
    from . import overview

    ov = overview.collect_product_overview()
    cat, caps = ov.catalog, ov.capabilities
    return {
        "models": cat.models_count, "hardware": cat.hardware_count, "profiles": cat.profiles_count,
        "presets": cat.presets_count, "preset_status_counts": cat.status_counts,
        "preset_families": cat.family_counts, "default_presets": list(cat.default_presets),
        "engine_installed": getattr(caps, "engine_installed", None),
        "vllm_version": getattr(getattr(caps, "platform", None), "vllm_version", None),
    }


def _h_doctor(_args: dict[str, Any]) -> dict[str, Any]:
    from . import doctor

    rep = doctor.collect_doctor_report()
    # Surface only non-ok findings to keep the result compact and actionable.
    notable = [
        {"category": f.category, "title": f.title, "severity": f.severity, "action": f.action}
        for f in rep.findings if f.severity in ("warning", "blocked", "info")
    ][:20]
    return {"summary": rep.summary, "warnings": list(rep.warnings)[:10], "findings": notable}


def _h_list_presets(args: dict[str, Any]) -> dict[str, Any]:
    from . import presets

    res = presets.list_presets(
        family=args.get("family"), workload=args.get("workload"),
        hardware=args.get("hardware"), mode=args.get("mode"), status=args.get("status"),
    )
    rows = [{"id": p.id, "model": p.model, "hardware": p.hardware, "runtime": p.runtime,
             "status": (p.card or {}).get("status")} for p in res.presets[:25]]
    return {"matched": res.matched, "total": res.total, "presets": rows,
            "truncated": res.matched > len(rows)}


def _h_get_preset(args: dict[str, Any]) -> dict[str, Any]:
    from . import presets

    pid = str(args.get("preset_id") or "").strip()
    if not pid:
        raise ValueError("preset_id is required")
    rec = presets.get_preset(pid)
    return {"id": rec.id, "model": rec.model, "hardware": rec.hardware,
            "profile": rec.profile, "runtime": rec.runtime, "card": rec.card}


def _h_list_patches(args: dict[str, Any]) -> dict[str, Any]:
    from .patches import listing

    rows = listing.list_patches(
        tier=args.get("tier"), lifecycle=args.get("lifecycle"),
        family=args.get("family"), default_on=args.get("default_on"),
    )
    out = [{"id": r.patch_id, "title": getattr(r, "title", ""), "tier": getattr(r, "tier", ""),
            "lifecycle": getattr(r, "lifecycle", ""), "default_on": getattr(r, "default_on", None)}
           for r in rows[:30]]
    return {"count": len(rows), "patches": out, "truncated": len(rows) > len(out)}


def _h_estimate_vram(args: dict[str, Any]) -> dict[str, Any]:
    from . import kv_math

    models = kv_math.known_models()
    model_id = str(args.get("model_id") or "")
    arch = models.get(model_id)
    if arch is None:
        return {"error": f"unknown model_id: {model_id}", "known_models": list(models)}
    kv_name = str(args.get("kv_dtype") or "fp8")
    kv_bytes = float(kv_math.KV_DTYPE_BYTES.get(kv_name, 1.0))
    tp = int(args.get("tp") or 1)
    est = kv_math.estimate(
        arch, context=int(args.get("context") or 32768), kv_bytes=kv_bytes,
        concurrency=int(args.get("concurrency") or 1), tp=tp, gpu_count=tp,
        gpu_vram_mib=int(args.get("gpu_vram_mib") or 24564), util=float(args.get("util") or 0.9),
        overhead_mib=1500.0,
    )
    return {"model": model_id, "kv_dtype": kv_name, "fits": est["fits"],
            "headroom_mib": est["headroom_mib"], "max_context": est["max_context"],
            "total_per_gpu_mib": est["total_per_gpu_mib"], "budget_per_gpu_mib": est["budget_per_gpu_mib"]}


def _h_plan_install(args: dict[str, Any]) -> dict[str, Any]:
    """Dry-run install plan — read-only. Surfaces a proposed action the human
    follows into the apply-gated Installer wizard (the copilot never applies)."""
    from . import installer

    preset_id = str(args.get("preset_id") or "").strip()
    target = str(args.get("target") or "compose").strip()
    host_id = str(args.get("host_id") or "").strip()
    host = {"label": host_id or "the host", "host": host_id}
    if host_id:
        from .host_profiles import host_profile_payload, list_host_profiles

        prof = next((p for p in list_host_profiles() if p.id == host_id), None)
        if prof is not None:
            host = host_profile_payload(prof)
    plan = installer.build_install_plan(host=host, preset_id=preset_id, target=target)
    summary = {
        "preset_id": preset_id, "target": plan.get("target_label"),
        "steps": len(plan.get("steps", [])), "danger_count": plan.get("danger_count"),
        "provisions_infra": plan.get("provisions_infra"), "artifact": plan["artifact"]["filename"],
    }
    # The proposed action is handed to the UI, which deep-links into the gated
    # Installer wizard with these fields prefilled. The copilot does not apply it.
    summary["proposed_action"] = {
        "kind": "install", "label": f"Review install of {preset_id} → {plan.get('target_label')}",
        "section": "setup", "params": {"preset_id": preset_id, "target": target, "host_id": host_id},
    }
    return summary


# Registry — add a tool by appending one entry (the single extension point).
_TOOLS: tuple[Tool, ...] = (
    Tool("get_overview", "Catalog + capability snapshot: model/preset/profile counts, preset status "
         "breakdown, whether the engine is installed and the vLLM version.", _schema({}), _h_overview),
    Tool("run_doctor", "Run the read-only doctor and return notable (warning/blocked/info) findings "
         "with suggested actions, plus the severity summary.", _schema({}), _h_doctor),
    Tool("list_presets", "List/filter preset configs in the catalog.", _schema({
        "family": {"type": "string", "description": "e.g. qwen3.6"},
        "status": {"type": "string", "description": "e.g. production, production_candidate"},
        "hardware": {"type": "string"}, "workload": {"type": "string"}, "mode": {"type": "string"},
    }), _h_list_presets),
    Tool("get_preset", "Get one preset's full record (model, hardware, runtime, card with reference "
         "metrics).", _schema({"preset_id": {"type": "string"}}, ["preset_id"]), _h_get_preset),
    Tool("list_patches", "List/filter the Genesis patch registry.", _schema({
        "tier": {"type": "string"}, "lifecycle": {"type": "string", "description": "e.g. active, retired"},
        "family": {"type": "string"}, "default_on": {"type": "boolean"},
    }), _h_list_patches),
    Tool("estimate_vram", "Estimate per-GPU VRAM fit + max context for a known model at a given "
         "context / tensor-parallel / GPU VRAM / KV dtype.", _schema({
             "model_id": {"type": "string", "description": "a known model id (see get_overview / known_models)"},
             "context": {"type": "integer"}, "tp": {"type": "integer"}, "concurrency": {"type": "integer"},
             "gpu_vram_mib": {"type": "integer"}, "kv_dtype": {"type": "string", "description": "fp8 / fp16 / int8 / tq_k8v4"},
         }, ["model_id"]), _h_estimate_vram),
    Tool("plan_install", "Produce a DRY-RUN install plan for a preset on a target (compose, proxmox, "
         "proxmox_vm, …). Returns the plan + a proposed action for the human to review and apply in the "
         "Installer — it does not apply anything.", _schema({
             "preset_id": {"type": "string"}, "target": {"type": "string"}, "host_id": {"type": "string"},
         }, ["preset_id"]), _h_plan_install, category="plan"),
)

_TOOLS_BY_NAME = {t.name: t for t in _TOOLS}


SYSTEM_PROMPT = (
    "You are the SNDR Ops Copilot for a self-hosted Genesis vLLM stack (patches, presets, "
    "deployments, hosts, benchmarks). You help an operator understand and plan — you are READ-ONLY.\n"
    "Rules:\n"
    "1. Use the provided tools to gather real facts before answering. Never invent counts, versions, "
    "preset ids, patch ids, or metrics — call a tool.\n"
    "2. You cannot change anything. Never claim you deployed, applied, patched, or restarted anything. "
    "For changes, use plan_install (dry-run) and tell the operator to review & apply it in the UI.\n"
    "3. Be concise and operator-focused. Cite concrete numbers from tool results. If a tool errors or "
    "the data is missing, say so plainly rather than guessing.\n"
    "4. Prefer the smallest set of tool calls that answers the question."
)


# ── public surface ───────────────────────────────────────────────────────────


def tool_specs() -> list[dict[str, Any]]:
    """OpenAI ``tools`` array describing the callable functions for the engine."""
    return [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in _TOOLS]


def tool_catalog() -> list[dict[str, Any]]:
    """Human-facing tool list for the GUI ('what can the copilot do')."""
    return [{"name": t.name, "description": t.description, "category": t.category} for t in _TOOLS]


def execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Run a registered read-only tool. Unknown/failed tools return an error
    dict (fed back to the model) rather than raising — the loop must continue."""
    tool = _TOOLS_BY_NAME.get(name)
    if tool is None:
        return {"ok": False, "error": f"unknown tool: {name}"}
    try:
        result = tool.handler(args if isinstance(args, dict) else {})
    except Exception as exc:  # noqa: BLE001 - surface to the model, never crash the loop
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    out: dict[str, Any] = {"ok": True, "result": result}
    if isinstance(result, dict) and isinstance(result.get("proposed_action"), dict):
        out["proposed_action"] = result["proposed_action"]
    return out


def _accumulate_usage(total: dict[str, int], usage: Any) -> None:
    if isinstance(usage, dict):
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            v = usage.get(k)
            if isinstance(v, (int, float)):
                total[k] = total.get(k, 0) + int(v)


def run_copilot(
    messages: list[dict[str, Any]],
    *,
    chat_fn: Callable[..., dict[str, Any]],
    max_steps: int = 5,
) -> dict[str, Any]:
    """Run the tool-calling loop.

    ``chat_fn(messages, tools=...)`` must return ``{"message": {...}, "usage": {...}}``
    (see :func:`engine_client.chat_raw`). Returns the final reply, the ordered
    tool-call trace (for UI transparency), any proposed actions, token usage and
    why it stopped. Tools are executed read-only, server-side.
    """
    convo: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    convo += [m for m in messages if m.get("role") != "system"]
    specs = tool_specs()
    steps: list[dict[str, Any]] = []
    proposed: list[dict[str, Any]] = []
    usage: dict[str, int] = {}

    for _ in range(max(1, max_steps)):
        out = chat_fn(convo, tools=specs)
        msg = out.get("message") or {}
        _accumulate_usage(usage, out.get("usage"))
        tool_calls = msg.get("tool_calls") or []
        convo.append({"role": "assistant", "content": msg.get("content") or "",
                      **({"tool_calls": tool_calls} if tool_calls else {})})
        if not tool_calls:
            return {"reply": msg.get("content") or "", "steps": steps,
                    "proposed_actions": proposed, "usage": usage, "stopped": "final"}
        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except (TypeError, ValueError):
                args = {}
            result = execute_tool(name, args if isinstance(args, dict) else {})
            steps.append({"tool": name, "args": args, "ok": result.get("ok", True)})
            if result.get("proposed_action"):
                proposed.append(result["proposed_action"])
            convo.append({"role": "tool", "tool_call_id": tc.get("id"), "name": name,
                          "content": json.dumps(result.get("result", result))[:6000]})

    # Out of tool budget — ask once more without tools for a closing answer.
    out = chat_fn(convo, tools=None)
    _accumulate_usage(usage, out.get("usage"))
    return {"reply": (out.get("message") or {}).get("content") or "", "steps": steps,
            "proposed_actions": proposed, "usage": usage, "stopped": "max_steps"}


__all__ = ["Tool", "tool_specs", "tool_catalog", "execute_tool", "run_copilot", "SYSTEM_PROMPT"]
