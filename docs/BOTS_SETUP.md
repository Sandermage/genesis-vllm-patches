# GitHub Bots Setup

Quick guide for the AI / security bots installed on this repo (and the few that need a manual click).

## Installed automatically (via `.github/` configs)

### Dependabot
- **Config**: `.github/dependabot.yml`
- **What it does**: Weekly check of Python deps (genesis_vllm_plugin, benchmarks, tests), GitHub Actions versions, Docker base images. Opens PRs grouped by minor/patch update level to reduce noise.
- **Limit**: 5 open Python PRs / 3 GitHub Actions / 2 Docker — caps the inbox.
- **Labels applied**: `dependencies` / `ci` / `docker` + `automated`

### CodeQL
- **Config**: `.github/workflows/codeql.yml`
- **What it does**: Static analysis on Python code (vllm/_genesis/, genesis_*.py, external_probe/) on every push to `main`, every PR touching those paths, and weekly Monday at 06:00 UTC.
- **Query suite**: `security-and-quality` (strictest — catches both security issues and code quality smells)
- **Where to see results**: GitHub repo → Security → Code scanning alerts

### Repo-level features (enabled via API)

These were enabled programmatically — no setup needed:

- ✅ **Secret scanning** + push protection (catches accidentally committed API keys, tokens)
- ✅ **Dependabot alerts** (notifies on known vulnerable dependencies)
- ✅ **Dependabot security updates** (auto-PR with security patches)
- ✅ **Discussions** (enabled — for cross-rig collab + Q&A in addition to Issues)

## Manual install needed (5 minutes via UI)

### gemini-code-assist (Google AI code review)

This is the bot that reviewed our PR vllm#40914 and caught the buffer-reuse bug. **Free for public repos.**

1. Visit https://github.com/apps/gemini-code-assist
2. Click **Install**
3. Choose "Only select repositories" → tick `genesis-vllm-patches`
4. Confirm
5. Done — every new PR automatically gets a code review comment within ~1 minute of opening

### CodeRabbit (alternative AI reviewer for cross-validation)

Optional second-opinion bot. Sometimes catches what gemini misses, and vice versa.

1. Visit https://github.com/apps/coderabbitai
2. Install on `genesis-vllm-patches`
3. Free for open-source

Trade-off: 2 AI reviewers on the same PR can be noisy. Worth it for high-stakes PRs (kernels, security-sensitive changes); overkill for typo fixes.

## Verifying setup

```bash
# Check repo security features
gh api repos/Sandermage/genesis-vllm-patches --jq '.security_and_analysis'

# Check Dependabot alerts active
gh api repos/Sandermage/genesis-vllm-patches/vulnerability-alerts && echo "ENABLED"

# Check installed apps (requires admin scope)
gh api repos/Sandermage/genesis-vllm-patches/installation 2>/dev/null
```

## What each bot will do for us

| Bot | When it fires | What it catches |
|---|---|---|
| **gemini-code-assist** | Every new PR | Code review (logic bugs, perf issues, missing buffers like our v7.45 fix story) |
| **Dependabot** | Weekly + on advisory | Outdated/vulnerable deps with auto-PR for upgrades |
| **CodeQL** | Every push to main + every PR + weekly | Security smells (SQL injection, path traversal, unsafe deserialization in Python) |
| **Secret scanning** | Every push | API keys / tokens accidentally committed |

## Why these specific choices

- **gemini-code-assist**: proven on our PR vllm#40914 — caught a real bug we missed. Free for public.
- **Dependabot + CodeQL**: native GitHub, zero-config beyond YAML, free for public.
- **Not chosen**: CodeRabbit (good but redundant with gemini), Copilot Code Review (paid), GitGuardian (better for private secrets-heavy repos).
