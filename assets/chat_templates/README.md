# Genesis bundled chat templates

Optional Jinja chat templates for vLLM `--chat-template` flag. **NOT used by default** — Genesis start scripts let vLLM use the bundled tokenizer's `chat_template` field. Apply these only if your client's behavior matches the documented scenarios.

## qwen3.6-enhanced.jinja

**Source:** [allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix)
**Author:** Cheuk-Yiu Chan ([@allanchan339](https://github.com/allanchan339))
**Blog:** [qwen3.6-enhanced.jinja: CoT leakage into tool turns and why preserve_thinking works now](https://allanchan339.github.io/bug-fixes/2026/05/02/Qwen36-27B-updated-jinja.html) (2026-05-02)
**License:** see upstream repo

### What it fixes

Multi-turn CoT leakage when running `qwen3.5-enhanced.jinja` on a Qwen3.6 model:
- Multimodal paths preserved (image/video tokens)
- Interleaved thinking aligned to actual 3.6 behavior
- **Self-healing for missing `</think>` tag before `<tool_call>` block** — the original template left malformed assistant text in the prompt, so causal models still conditioned on broken structure
- `preserve_thinking` supported in BOTH true and false modes (3.5-enhanced template forced false as a workaround)

### When to use

✅ Use if any of these apply:
- You're already passing `--chat-template qwen3.5-enhanced.jinja` to vLLM with a 3.6 model
- You see CoT/reasoning leaking into `tool_response` content fields on multi-turn agent runs
- You see tool instructions silently ignored after a few turns of agent traffic
- Your downstream parser is `qwen3_coder` (relevant — same parser lane the author validated against)

❌ Skip if:
- You don't pass `--chat-template` (default — vLLM uses tokenizer's bundled template)
- Your client is single-turn or short-context
- You serve a non-Qwen3.6 model

### How to use

```bash
# In your start script, add:
exec vllm serve /models/Qwen3.6-27B-int4-AutoRound \
  --chat-template "$HOME/genesis-vllm-patches/assets/chat_templates/qwen3.6-enhanced.jinja" \
  ...
```

### Relationship to Genesis runtime patches

This is an **operator-supplied chat template**, not a runtime patch. It complements (does NOT replace) Genesis runtime patches that fix vLLM internals:
- **PN51** (`enable_thinking=false` streaming routing — vllm#40816) — backend-side
- **PN56** (Qwen3Coder XML parse fallback — vllm#41466) — parser-side
- **P107** (MTP truncation detector at reasoning→tool boundary — vllm#41467) — backend-side
- **P62** (Structured-output spec-decode reasoning-end timing — vllm#36138) — scheduler-side

Together: bundled jinja prevents the bug from reaching vLLM in the first place; Genesis patches catch what slips through.

## qwen3.5-enhanced.jinja

**Same source.** Bundled for completeness — the author's earlier template with `preserve_thinking=false` workaround. Ship if you serve a Qwen3.5 model and want enhanced multimodal/tool support.

## Bundling rationale + credit

Genesis bundles these files unmodified (with attribution above) so operators don't have to clone a separate repo + manage a third path. Per project policy, no generated edits to the templates themselves; they remain Cheuk-Yiu Chan's authored work. License terms inherited from upstream.

If the upstream author ships updates, refresh via:
```bash
curl -L https://raw.githubusercontent.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix/main/chat-template/qwen3.6-enhanced.jinja \
  -o assets/chat_templates/qwen3.6-enhanced.jinja
```
