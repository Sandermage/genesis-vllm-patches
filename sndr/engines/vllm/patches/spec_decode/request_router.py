# SPDX-License-Identifier: Apache-2.0
"""request_router — deterministic explicit-signal MTP profile router.

D1 deliverable. Pure library code. Given an incoming chat-completion
request shape (OpenAI-compatible dict) and an optional matched
``FunctionalArtifact``, returns a ``ProfileSelection`` describing which
spec-decode profile SHOULD be used for this request.

Critical design constraints (set 2026-05-20 after β′-A bench):

  1. NEVER reads prompt text. Free-form natural text classification is
     fragile; a false-positive (chat got routed to structured profile)
     costs -50% TPS. A false-negative (structured got MTP-off) costs
     zero. The router is conservative by design.

  2. Signal priority (high to low):
     a. ``response_format`` set to json_object / json_schema
     b. ``tool_choice`` requires a specific function (string 'required'
        or dict with 'type': 'function')
     c. ``extra_body.workload_class`` is set AND in the artifact's
        ``allowed_workloads`` list (explicit operator tag)
     d. No signal -> fallback profile (typically MTP OFF)

  3. Artifact gate: if a workload_class is detected but the matching
     artifact lists it in ``denied_workloads`` (or not in
     ``allowed_workloads``), the router falls back. Bench evidence
     wins; signal alone does not unlock the profile.

  4. Router is a SUGGESTION engine. The runtime gate is still the
     safety_guard at boot. If the engine was booted without MTP at
     all (guard denied), router output is informational only.

  5. NO Qwen behavior. Router does not affect non-Gemma4 model
     requests because no Gemma4-style artifact will match (provider
     unmatched).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .functional_artifact import FunctionalArtifact

log = logging.getLogger("genesis.spec_decode.request_router")


# Signal source IDs (stable strings for logging + telemetry)
SIGNAL_RESPONSE_FORMAT = "response_format"
SIGNAL_TOOL_CHOICE = "tool_choice"
SIGNAL_TOOLS_LIST = "tools_list"  # only when tool_choice REQUIRES tool use
SIGNAL_WORKLOAD_CLASS = "extra_body.workload_class"
SIGNAL_NONE = "no_signal"


@dataclass
class ProfileSelection:
    """The router's decision for one request."""
    profile: str
    """Profile name selected. Either the artifact's profile (when
    accepted) or the fallback profile."""

    signal: str
    """Which signal triggered the decision. One of the SIGNAL_*
    constants."""

    workload_class: str | None
    """Detected workload class (tool_json / structured_count / ...).
    None if no signal."""

    reason: str
    """Human-readable summary suitable for an operator log line."""

    accepted: bool
    """True if the proposed profile is the artifact's profile.
    False if the router fell back."""


# ----------------------- Signal detection -----------------------

def _is_tool_json_response_format(rf: Any) -> bool:
    """``response_format`` is one of:
       {'type': 'json_object'}
       {'type': 'json_schema', 'json_schema': {...}}
    """
    if rf is None:
        return False
    if isinstance(rf, dict):
        t = rf.get("type") or rf.get("kind") or ""
        return str(t).lower() in ("json_object", "json_schema")
    if isinstance(rf, str):
        return rf.lower() in ("json_object", "json_schema")
    return False


def _is_tool_json_tool_choice(tc: Any) -> bool:
    """``tool_choice`` requires a specific tool call iff:
       - string 'required'
       - dict {'type': 'function', ...}
    NOT triggered by 'auto' alone (model may choose to chat).
    """
    if tc is None:
        return False
    if isinstance(tc, str):
        return tc.lower() == "required"
    if isinstance(tc, dict):
        t = tc.get("type") or ""
        return str(t).lower() == "function"
    return False


def _extract_workload_class(request: dict) -> str | None:
    """Look for an explicit operator tag in either:
       - request['workload_class']  (some clients put it top-level)
       - request['extra_body']['workload_class']
    """
    if not isinstance(request, dict):
        return None
    top = request.get("workload_class")
    if isinstance(top, str) and top.strip():
        return top.strip()
    extra = request.get("extra_body")
    if isinstance(extra, dict):
        v = extra.get("workload_class")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


# ----------------------- Core router -----------------------

def select_profile(
    *,
    request: dict | None,
    artifact: FunctionalArtifact | None,
    fallback_profile: str = "gemma4-31b-tq-default",
) -> ProfileSelection:
    """Pure function: request shape + optional artifact -> decision.

    No torch. No I/O. No prompt-text reading.

    Behavior:
      - If artifact is None (no validated profile available, OR provider
        not Gemma4): always return fallback. This is the Qwen path.
      - Detect signal in priority order. First non-None wins.
      - If detected workload_class is in artifact.allowed_workloads:
        accept the artifact's profile.
      - Otherwise (no signal, OR class denied) -> fallback.
    """
    if artifact is None:
        return ProfileSelection(
            profile=fallback_profile,
            signal=SIGNAL_NONE,
            workload_class=None,
            reason=("no FunctionalArtifact available; router falls back "
                    "to default (this is the no-Gemma4/no-validation path)"),
            accepted=False,
        )

    req = request if isinstance(request, dict) else {}

    # 1) response_format -> tool_json
    if _is_tool_json_response_format(req.get("response_format")):
        return _resolve(
            artifact=artifact,
            workload_class="tool_json",
            signal=SIGNAL_RESPONSE_FORMAT,
            evidence=f"response_format={req.get('response_format')!r}",
            fallback_profile=fallback_profile,
        )

    # 2) tool_choice -> tool_json
    if _is_tool_json_tool_choice(req.get("tool_choice")):
        return _resolve(
            artifact=artifact,
            workload_class="tool_json",
            signal=SIGNAL_TOOL_CHOICE,
            evidence=f"tool_choice={req.get('tool_choice')!r}",
            fallback_profile=fallback_profile,
        )

    # 3) explicit operator tag
    wc = _extract_workload_class(req)
    if wc:
        return _resolve(
            artifact=artifact,
            workload_class=wc,
            signal=SIGNAL_WORKLOAD_CLASS,
            evidence=f"workload_class={wc!r} (explicit tag)",
            fallback_profile=fallback_profile,
        )

    # 4) no signal
    return ProfileSelection(
        profile=fallback_profile,
        signal=SIGNAL_NONE,
        workload_class=None,
        reason=(
            "no explicit workload signal (response_format, tool_choice, "
            "or extra_body.workload_class); router conservatively falls "
            "back to default (no prompt-text inference in v1)"
        ),
        accepted=False,
    )


def _resolve(*, artifact: FunctionalArtifact, workload_class: str,
             signal: str, evidence: str, fallback_profile: str
             ) -> ProfileSelection:
    """Given a detected workload_class, decide accept vs fallback by
    consulting the artifact's allowed_workloads."""
    if workload_class in artifact.allowed_workloads:
        return ProfileSelection(
            profile=artifact.profile,
            signal=signal,
            workload_class=workload_class,
            reason=(
                f"signal={signal} {evidence}; workload_class={workload_class!r} "
                f"is in artifact.allowed_workloads={artifact.allowed_workloads}; "
                f"router proposes profile={artifact.profile!r}"
            ),
            accepted=True,
        )
    # Detected but denied
    return ProfileSelection(
        profile=fallback_profile,
        signal=signal,
        workload_class=workload_class,
        reason=(
            f"signal={signal} {evidence}; workload_class={workload_class!r} "
            f"NOT in artifact.allowed_workloads={artifact.allowed_workloads} "
            f"(denied: {artifact.denied_workloads}); router falls back"
        ),
        accepted=False,
    )


__all__ = [
    "ProfileSelection",
    "select_profile",
    "SIGNAL_RESPONSE_FORMAT",
    "SIGNAL_TOOL_CHOICE",
    "SIGNAL_TOOLS_LIST",
    "SIGNAL_WORKLOAD_CLASS",
    "SIGNAL_NONE",
]
