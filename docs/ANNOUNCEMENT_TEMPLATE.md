# Announcement template

A reusable skeleton for a **"we shipped X" post** — a new model graduating to
PROD, a meaningful capability (a new spec-decode path, a quant tier, a cross-rig
result), or a release. Fill the skeleton, drop in the measured numbers, post.
The goal is one credible, copy-pasteable post that tells a reader what shipped,
what it measures, and how to run it.

## When to use it

- A model/preset graduates (⚙️ → ✅) and you want people to know it exists.
- A capability or finding ships that is worth a standalone post.
- A tagged release with user-visible changes.

For a *reply* to a specific question, lead with the one-line answer, then drop
the relevant section underneath.

## Section order

| # | Section | Purpose | Skip when |
|---|---|---|---|
| 1 | **Intro** — one sentence | What shipped: the preset/model, the engine, the topology. Credit upstream/model/quant authors inline. | Never. |
| 2 | **Headline** | The single most interesting number, one line — the reason to keep reading. | Never. |
| 3 | **Results panel** | The measured table (model · quant · KV · spec-decode · TPS · tool-call · VRAM · ctx). | Never — numbers are the point. |
| 4 | **Why** | The workload it serves and the trade it makes. | A trivially obvious ship. |
| 5 | **Run it** | The install one-liner + the exact `sndr` command + port + served model name. Copy-pasteable. | — |
| 6 | **Credits** | Upstream vLLM PR authors, model author, quant/drafter author — each `@handle` + link. | No third-party work involved. |
| 7 | **What'd help** | The concrete cross-rig ask — link the [`numbers-from-your-rig`](https://github.com/Sandermage/sndr_core_engine/issues/new?template=numbers-from-your-rig.yml) issue form. | — |

## Skeleton

```markdown
**SNDR Core Engine (Genesis) — <what shipped, one sentence>.**
<Model> on <engine + version>, <topology> (e.g. 2× A5000, TP=2). All model
credit to [@<author>](<hf-link>); <quant/drafter> by [@<author2>](<link>).

**Headline:** <the single most interesting result — e.g. "Qwen3.6-35B-A3B
sustains 239.7 tok/s single-stream on 2× A5000, +53% over stock vLLM.">

### Results — <rig>, <engine + version>, <pin>
| Model | Quant / KV | Spec-decode | Decode TPS | Tool-call | Ctx | VRAM |
|---|---|---|--:|:--:|--:|--:|
| <model> | <FP8 · TQ k8v4> | <MTP K=5> | <239.7> | <7/7> | <256K> | <GB> |

<one-line read of the result — within CV? lossless spec-decode? new ceiling?>

**Why:** <the workload this serves and the trade it makes.>

**Run it:**
\`\`\`bash
curl -sSL https://raw.githubusercontent.com/Sandermage/sndr_core_engine/main/install.sh | bash
sndr run <preset>                 # → chat;   sndr up <preset> for the GUI
\`\`\`
Serves an OpenAI-compatible API on http://127.0.0.1:8000/v1 (model: <name>).

**Credits:** upstream [vLLM](https://github.com/vllm-project/vllm) ·
model [@<author>](<link>) · quant/drafter [@<author2>](<link>).

**What'd help:** numbers from other rigs (3090 / 4090 / 5090 / A6000) —
drop them via the *Numbers from your rig* issue form. Cross-rig results land
in BENCHMARKS with credit.
```

## Platform variants

Same facts, three shapes. Lead with the measured numbers; never overclaim.

### r/LocalLLaMA (the primary venue)

- **Title:** concrete and number-led — e.g. *"Qwen3.6-35B-A3B at 239.7 tok/s on
  2× A5000 (consumer Ampere) — runtime vLLM patch overlay, Apache-2.0"*.
- **Body:** the full skeleton above. Lead with the results table; this audience
  reads numbers first. Be explicit about the rig and that it's reproducible.
- Include the repo link once, near the top, and once in a "Run it" block.
- Reply to every "numbers on my <card>?" with the issue-form link.

### Hacker News — "Show HN"

- **Title:** *"Show HN: SNDR Core Engine – run Qwen3.6/Gemma4 on consumer
  NVIDIA via runtime vLLM patches"*. No emoji, no hype words.
- **First comment** (post it yourself): the "why" — what problem it solves
  (vLLM targets datacenter SKUs; this makes consumer Ampere/Ada/Blackwell a
  first-class target), the honest limitations, and that patches retire when
  upstream merges. Link the [`README`](../README.md) and [`BENCHMARKS.md`](BENCHMARKS.md).
- HN rewards candor about trade-offs over marketing. State what doesn't work yet.

### X / Twitter thread

1. Hook + headline number + the OG card image (`assets/og-card.png`).
2. The results table as an image or a tight code block.
3. The "run it" one-liner.
4. Credits to upstream + model/quant authors (tag handles).
5. The repo link + the cross-rig ask.

Keep each post under the limit; thread rather than truncate.

## Don'ts

- Don't quote a number you didn't measure on a named rig. Mark anything
  illustrative as illustrative.
- Don't claim a model "works" without the tool-call / soak evidence behind it.
- Don't omit upstream/model/quant credit — capability that isn't ours gets
  attributed up front.
