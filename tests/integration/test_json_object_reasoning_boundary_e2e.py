# SPDX-License-Identifier: Apache-2.0
"""50-trial E2E json_object reproducer — port of vllm#44993's test plan.

Upstream context: under async scheduling + spec decode (MTP) + a
Qwen-style reasoning parser, ``response_format={"type":"json_object"}``
silently bypasses the grammar when ``</think>`` lands inside a
spec-decode window (vllm#43388). Upstream's reproducer measured a ~96%
failure rate on the pristine tree (Markdown fences, prose wrapping,
duplicate ``{{`` openers) and 50/50 ok with #44297 + #44993 applied.

Genesis P62 (vendor of vllm#36138, sibling approach) covers the same
bug family. This port is the FIRST end-to-end proof that
json_object/regex/choice structured output works on OUR stack
(Qwen3.6 PROD, TP=2, TQ k8v4, MTP K=3, async scheduling) — per the
2026-06-11 roadmap row for #44993 ("port the 50-trial E2E json_object
reproducer"). The unit-level half of the port lives in
tests/unit/integrations/serving/test_p62_grammar_advance_44993.py.

Gating (matches tests/integration/test_patch_regression_bounds.py):
skip-marked for the rig — the whole E2E class skips unless
``GENESIS_INTEGRATION_ENDPOINT`` is set. The rig stage engages it:

    GENESIS_INTEGRATION_ENDPOINT=http://127.0.0.1:8103/v1 \\
    GENESIS_INTEGRATION_API_KEY=genesis-local \\
    GENESIS_INTEGRATION_MODEL=qwen3.6-35b-a3b \\
    GENESIS_E2E_JSON_OBJECT_TRIALS=50 \\
    python3 -m pytest \\
        tests/integration/test_json_object_reasoning_boundary_e2e.py -v

P62 note: run once with the PROD env as-is to establish the baseline;
if GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING is being validated, run
the same 50 trials per arm (OFF/ON) and compare the breakdowns.

The ``classify`` function is a faithful port of upstream's classifier
and is unit-tested offline below (no server needed) so the
classification contract cannot rot between rig runs.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections import Counter

import pytest


REQUIRED_KEYS = ("city", "country", "population")

USER_PROMPT = (
    "Give me a JSON object describing Seoul, with keys city, country, "
    "and population."
)


def classify(content: str) -> tuple[str, str]:
    """Classify one json_object response body — port of the vllm#44993
    reproducer's classifier (kind, detail)."""
    if not content or not content.strip():
        return "empty", ""
    raw = content
    s = content.strip()
    if s.startswith("```"):
        return "markdown_fence", s[:120]
    if raw[0] != "{":
        return "leading_garbage", repr(raw[:40])
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return "no_json", repr(s[:80])
    if m.group(0) != s:
        return "prose_wrapped", s[:120]
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        return "bad_json", str(e)
    for k in REQUIRED_KEYS:
        if k not in obj:
            return "bad_json", "missing " + repr(k)
    return "ok", json.dumps(obj, ensure_ascii=False)[:120]


# ── Offline contract tests for the classifier (always run) ────────────


class TestClassifyContract:
    def test_ok_strict_json(self):
        body = '{"city": "Seoul", "country": "South Korea", "population": 9}'
        kind, _ = classify(body)
        assert kind == "ok"

    def test_markdown_fence_failure_mode(self):
        """The dominant pristine failure mode upstream observed: the
        grammar never engages and the model emits a fenced block."""
        kind, _ = classify('```json\n{"city": "Seoul"}\n```')
        assert kind == "markdown_fence"

    def test_duplicate_opening_token_is_bad_json(self):
        """Bug 2 signature: '{{...}' from the un-advanced FSM re-emitting
        the opening token (HTTP 200, body fails to parse)."""
        kind, _ = classify('{{"city": "Seoul", "country": "KR", '
                           '"population": 9}')
        assert kind == "bad_json"

    def test_prose_wrapped(self):
        kind, _ = classify('{"a": 1} — here is your object.')
        # leading char IS '{' but the JSON does not span the whole body.
        assert kind == "prose_wrapped"

    def test_leading_garbage(self):
        kind, _ = classify('Sure! {"city": "Seoul"}')
        assert kind == "leading_garbage"

    def test_missing_required_key_is_bad_json(self):
        kind, detail = classify('{"city": "Seoul", "country": "KR"}')
        assert kind == "bad_json"
        assert "population" in detail

    def test_empty(self):
        assert classify("")[0] == "empty"
        assert classify("   \n")[0] == "empty"


# ── E2E gating (rig only) ─────────────────────────────────────────────


def _endpoint() -> str | None:
    return os.environ.get("GENESIS_INTEGRATION_ENDPOINT")


def _api_key() -> str:
    return os.environ.get("GENESIS_INTEGRATION_API_KEY", "genesis-local")


def _model() -> str | None:
    return os.environ.get("GENESIS_INTEGRATION_MODEL")


def _trials() -> int:
    return int(os.environ.get("GENESIS_E2E_JSON_OBJECT_TRIALS", "50"))


def _chat_json_object(base_url: str, model: str, timeout: float = 120.0) -> str:
    """One json_object chat completion via stdlib urllib (no openai
    dependency on the rig). Returns the message content; raises
    urllib.error.HTTPError on non-2xx (counted as http_<code>)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": USER_PROMPT}],
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": True},
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 4096,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_api_key()}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"] or ""


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not _endpoint(),
    reason="GENESIS_INTEGRATION_ENDPOINT not set — rig-stage E2E only",
)
class TestJsonObjectReasoningBoundaryE2E:
    """The 50-trial loop. Pristine-upstream baseline: ~96% fail.
    Expected on Genesis PROD (P62 family active): 50/50 ok."""

    def test_50_trials_json_object_all_ok(self):
        model = _model()
        assert model, (
            "GENESIS_INTEGRATION_MODEL must be set together with "
            "GENESIS_INTEGRATION_ENDPOINT"
        )
        trials = _trials()
        counts: Counter[str] = Counter()
        details: list[str] = []
        for i in range(1, trials + 1):
            try:
                content = _chat_json_object(_endpoint(), model)
                kind, detail = classify(content)
            except urllib.error.HTTPError as e:
                # vllm#44006-class strict-mode failures surface as 500s.
                kind, detail = f"http_{e.code}", str(e)
            counts[kind] += 1
            if kind != "ok":
                details.append(f"trial {i}: {kind}: {detail}")
        breakdown = ", ".join(
            f"{k}={v}" for k, v in sorted(counts.items())
        )
        assert counts["ok"] == trials, (
            f"json_object E2E: {counts['ok']}/{trials} ok ({breakdown})\n"
            + "\n".join(details[:10])
        )
