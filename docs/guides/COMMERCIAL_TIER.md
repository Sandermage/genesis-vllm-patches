# Commercial Tier — Author and Operator Guide

**Status**: Reference documentation
**Audience**: Commercial customers, sndr-engine wheel maintainers, operators

## Overview

sndr-platform has two tiers of patches:

| Tier | License | Audience | Distribution |
|---|---|---|---|
| **Community** | Apache 2.0 (public) | Any operator | PyPI: `pip install sndr` |
| **Engine** | Proprietary | Paid customers only | Private PyPI / direct: `pip install sndr-engine` |

Engine-tier patches typically deliver specialized kernels, KV-cache schemes,
or speculative-decoding algorithms that require ongoing maintenance and
performance optimization.

## How the commercial tier is enforced

### 1. License token (cryptographic)

Each commercial customer receives a signed Ed25519 token. The token payload
contains:

```json
{
  "customer_id": "acme-corp",
  "issued_at": 1733424000,
  "expires_at": 1764960000,
  "engine_major": 12
}
```

The token is signed with an offline private key; the public verification key
is embedded in `sndr/license.py`. Customers install the token via either:

- Environment: `export SNDR_ENGINE_LICENSE_KEY="<token>"`
- File: write to `~/.sndr/license.json`

### 2. Wheel discovery via entry points

The commercial wheel `sndr-engine` publishes patches via setuptools entry
points:

```toml
# sndr-engine/pyproject.toml
[project.entry-points."sndr.engines.vllm.patches"]
p67 = "sndr_engine.vllm.patches.p67:patch"
pn21 = "sndr_engine.vllm.patches.pn21:patch"
```

The community `sndr` wheel discovers these via `sndr.plugins.loader`. It
never imports `sndr_engine` directly. Customers without the wheel get
community-only mode automatically.

### 3. Dispatcher tier gate

When the dispatcher evaluates an engine-tier patch, it calls
`sndr.license.check_engine_tier_eligible()` which verifies:

1. License token present and parsable
2. Ed25519 signature valid
3. Token not expired
4. `engine_major` matches `sndr.__version_major__`
5. `sndr-engine` wheel installed and importable

Only when all checks pass does the patch apply.

## Boot flow

```
sndr.init()
    ↓
license.check_engine_tier_eligible()
    ├─ LICENSED → dispatcher allows engine-tier patches
    │              ↓
    │           plugins.discover_engine_patches("vllm")
    │              ↓
    │           [loaded patches] merged into PATCH_REGISTRY
    │
    ├─ EXPIRED → log warning, community-only mode
    ├─ NO_PACKAGE → log info, community-only mode
    └─ BAD_SIGNATURE → log error, community-only mode
```

## Author guide: shipping a commercial wheel

### Project layout

```
sndr-engine/
├── pyproject.toml
├── sndr_engine/
│   ├── __init__.py
│   └── vllm/
│       ├── __init__.py
│       └── patches/
│           ├── p67.py
│           └── pn21.py
├── tests/
└── README.md
```

### Patch module conventions

Each patch module exposes a top-level `patch` symbol that the entry point
references:

```python
# sndr_engine/vllm/patches/p67.py
from sndr.dispatcher.spec import PatchSpec

patch = PatchSpec(
    id="P67",
    tier="engine",
    family="attention.turboquant",
    title="TurboQuant multi-query Triton kernel",
    apply_module="sndr_engine.vllm.patches.p67",
    # ... applies_to, env_flag, etc.
)
```

### Entry-point registration

```toml
# sndr-engine/pyproject.toml
[project]
name = "sndr-engine"
version = "12.0.0"
requires-python = ">=3.10"
dependencies = ["sndr>=12.0,<13.0"]

[project.entry-points."sndr.engines.vllm.patches"]
p67 = "sndr_engine.vllm.patches.p67:patch"
pn21 = "sndr_engine.vllm.patches.pn21:patch"
# ... add one entry per patch
```

### Version compatibility

The `engine_major` field of license tokens MUST match `sndr.__version_major__`.
If a customer upgrades sndr (e.g. 12 → 13) without rotating their license, the
tier gate returns `LICENSED_LEGACY` (warning) or `VERSION_MISMATCH` (blocked).

Maintain at least one release cycle of overlapping support.

## Operator guide: installing a commercial license

### Step 1: Install both wheels

```bash
pip install sndr sndr-engine
```

### Step 2: Install the license

Option A — environment variable (preferred for containers):

```bash
export SNDR_ENGINE_LICENSE_KEY="<token-from-vendor>"
```

Option B — file (preferred for long-running hosts):

```bash
mkdir -p ~/.sndr
cat > ~/.sndr/license.json <<EOF
{"token": "<token-from-vendor>"}
EOF
chmod 600 ~/.sndr/license.json
```

### Step 3: Verify

```bash
sndr engines.info vllm
```

The output includes `patch_count_engine`. A non-zero value confirms engine-tier
patches loaded successfully.

## Security model

Token bearer credentials:

- **NEVER** logged. The code in `sndr.license` is reviewed quarterly for
  accidental token logging.
- **NEVER** sent to telemetry. Metrics carry `customer_id_hash` only.
- **NEVER** persisted by sndr itself. Operators choose env vs file storage.
- **NEVER** verified per-request. Verification happens once at boot; result
  cached for process lifetime.

If a token leaks, the customer rotates it via the vendor portal. Old tokens
remain valid until their natural expiration. We do not maintain revocation
lists in v12 — the short expiration window (typically 12 months) is the
mitigation.

## License lifecycle events

| Status | Customer-visible effect | Log event |
|---|---|---|
| `LICENSED` | engine patches active | `license.verified` |
| `LICENSED_LEGACY` | engine patches active + warning | `license.legacy_signature` |
| `EXPIRED` | community-only mode + warning | `license.expired` |
| `BAD_SIGNATURE` | community-only mode + error | `license.invalid_signature` |
| `VERSION_MISMATCH` | community-only mode + error | `license.version_mismatch` |
| `NO_KEY` | community-only mode | `license.no_token` |
| `NO_PACKAGE` | community-only mode | `license.no_package` |

The GUI surfaces the current status in the licensing panel. Expiry warnings
appear 30 days before expiration.

## References

- License code: `sndr/license.py`
- Plugin loader: `sndr/plugins/loader.py`
- Dispatcher tier gate: `sndr/dispatcher/decision.py:_check_tier_gate`
- Token generation tool: `tools/license_keygen.py`
