# Security policy

## Reporting a vulnerability

Genesis vLLM Patches is a runtime-patch package — security
disclosures must reach the maintainer privately before going
public.

- **Do NOT** open a GitHub issue for a suspected vulnerability.
- Email **`sander.odessa@gmail.com`** with the details. A
  disclosure is acknowledged within 72 hours and coordinated on
  standard responsible-disclosure timelines.

What to include:

- Reproducer (preset key, vLLM pin, GPU, environment, payload).
- Impact assessment (information disclosure, code execution,
  denial of service, ...).
- Suggested mitigation if you have one.

## Scope

Reports are in scope when they affect the public-core package
(`sndr` (the operator product)), the shipped CLI (`sndr`), or any artefact
under [`tools/`](tools/), [`scripts/`](scripts/), or
[`docs/`](docs/) in this repository.

Out of scope:

- Upstream vLLM bugs reproducible without Genesis — report those
  to <https://github.com/vllm-project/vllm/issues>.
- Hardware-specific kernel issues already documented in
  [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) (the named
  cliffs catalogue).
- Misconfigurations of the operator's own preset that don't
  affect anyone else's installation.

## Supported versions

Genesis pins a specific vLLM commit; security fixes track the
`KNOWN_GOOD_VLLM_PINS` allowlist in
`sndr/engines/vllm/detection/guards.py`. Operators running an
unsupported pin should bump to the current pin per
[`docs/RELEASE_POLICY.md`](docs/RELEASE_POLICY.md) before
escalating.

## Public-core stays public

Genesis public core is Apache-2.0 and ships without any license
gate that restricts functionality. Any change that would gate a
public-core feature behind a license check is itself a security-
adjacent issue and falls under this policy.

## Transitive dependency CVE remediation

The runtime core declares only `pyyaml` + `packaging`. Everything
else lives behind opt-in extras (`dev`, `http`, `gui-api`, `k8s`,
`gui-remote`, `gui-auth-pam`, `telemetry`, `license`) and the
documented overlay install (`pip install --no-deps -e .`) pulls
none of them — so transitive CVEs in those extras never touch the
vLLM runtime tree.

Dependabot pip alerts are triaged against the *resolved* extras
tree (clean venv, full extras, `pipdeptree`). Critical/high alerts
with an available patch and a genuine path through one of our
declared extras get a minimal `>=` floor in the **owning** extra:

- **h11** `>=0.16.0` — CVE-2025-43859 / GHSA-vqfr-h8mv-ghfj.
  Owned by `gui-api`: `uvicorn[standard]` → `h11`.
- **urllib3** `>=2.7.0` — CVE-2026-44431 / GHSA-qccp-gfcp-xxvc.
  Owned by `http` and `telemetry` (`requests` → `urllib3`) and by
  `k8s` (`kubernetes` → `urllib3`, a direct kubernetes requirement).

### Alerts that are NOT part of our dependency graph

Some Dependabot pip alerts fire on packages that are **not** in the
resolved extras tree at all. They are documented here so the alert
is understood, not silently dismissed — and deliberately **not**
pinned (pinning a package we do not pull would be meaningless and
misleading):

- **poetry** (`>=2.3.3`) — a build tool we do not declare or vendor;
  it never enters the install tree.
- **dulwich** (`>=1.2.5`) and **msgpack** (`>=1.2.1`) — these are
  *poetry's own* dependencies. With poetry absent, neither is
  installed.
- **pillow** (`>=12.2.0`, e.g. GHSA-5xmw-vc9v-4wf2 / CVE-2026-42309) —
  no declared package of ours pulls Pillow; it is not in the
  resolved tree.

Re-verify by resolving the full extras tree in a clean venv:

```bash
python3.11 -m venv /tmp/v && . /tmp/v/bin/activate
pip install -e '.[dev,http,gui-api,k8s,gui-remote,gui-auth-pam,telemetry,license]'
pip install pipdeptree
pipdeptree --reverse --packages h11,urllib3   # present → owned by our extras
pipdeptree --reverse --packages pillow,dulwich,poetry,msgpack  # "No packages matched"
```
