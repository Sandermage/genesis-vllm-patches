# SNDR GUI Security Model

The GUI daemon (`sndr gui-api`) is read-only and local-first. This document
states the boundaries it enforces and how to operate it safely.

## Default localhost bind

The daemon binds to `127.0.0.1` by default. It is not exposed on the LAN/WAN.

```bash
python3 -m sndr.cli gui-api --host 127.0.0.1 --port 8765
```

Binding to a non-loopback address is possible (`--host 0.0.0.0`) but should only
be done behind a trusted network boundary and with a token set. The recommended
remote-access path is an SSH tunnel, not a public bind.

## Token auth

Set `SNDR_GUI_TOKEN` to require a bearer token. When set, every `/api/v1/*`
route except `/api/v1/health` and `/api/v1/auth/status` requires:

```text
Authorization: Bearer <token>
# or
X-SNDR-Token: <token>
```

Requests without a valid token receive `401`. With no token set (default), the
local daemon is open — appropriate for `127.0.0.1` only.

```bash
SNDR_GUI_TOKEN=$(openssl rand -hex 16) python3 -m sndr.cli gui-api
```

In the GUI, set the token in Advanced → API & Schema. It is stored in
`localStorage` (`sndr.gui.token`) and attached to API calls. Note: browser
`EventSource` cannot send headers, so with a token set the live event feed falls
back to authenticated polling.

## Remote access via SSH tunnel

Keep the daemon on `127.0.0.1` on the host and forward a port:

```bash
ssh -L 8765:127.0.0.1:8765 user@gpu-host
```

The daemon stays unreachable from the network; only your forwarded loopback port
can reach it.

## Plan-before-apply and the apply opt-in

By default the daemon is dry-run: `apply` endpoints record the exact commands an
operator would run but do not execute. Real execution is opt-in:

```bash
# default: dry-run only
python3 -m sndr.cli gui-api

# opt in to real service-action execution
python3 -m sndr.cli gui-api --enable-apply
# or: SNDR_ENABLE_APPLY=1 python3 -m sndr.cli gui-api
```

When apply is enabled (`GET /api/v1/auth/status` → `apply_enabled: true`):

- **read-only** actions (`status` / `logs`) execute and return real output;
- **mutating** actions (`start` / `stop` / `restart`) require an explicit
  `confirm: true` in the request — otherwise the daemon returns `409` and runs
  nothing;
- follow flags (`-f`) are stripped so a command cannot block the daemon;
- remote targets run over a single SSH invocation
  (`ssh <ssh_target> <command>`); empty SSH target runs locally.

Config/patch writes remain operator-local only. The launch plan still reports
`actionable=false` while blocking gates exist; launch apply is not wired to real
execution. Verify any mutating action against a disposable container before
pointing it at a production service.

## Operator-local writes only

The only writes happen under your install dir (`$SNDR_HOME`, default `~/.sndr`):

| Data | Location |
| --- | --- |
| Host profiles | `$SNDR_HOME/gui/hosts.json` |
| GUI settings | browser `localStorage` (client side) |
| Operator-local presets | `$SNDR_HOME/model_configs/...` (atomic write + `.bak` + `.lock`) |
| Report bundles | `$SNDR_HOME/reports/<id>/` |

Nothing is written to the repository, the builtin V2 catalog, or a remote host.

## Redaction

Report bundles are redacted by default: the operator home path is replaced with
`~` and the GUI token with `***` before the snapshot is written. Generate a raw
(unredacted) bundle only for local inspection.

## CORS

The daemon allows local frontend origins via an origin regex
(`http://localhost|127.0.0.1[:port]`) for `GET`, `POST`, and `DELETE`. It does
not allow arbitrary cross-origin access.

## Summary

| Property | Value |
| --- | --- |
| Default bind | `127.0.0.1:8765` |
| Auth | optional bearer token via `SNDR_GUI_TOKEN` |
| Mutations | dry-run only; real execution gated and not enabled here |
| External writes | none (operator-local `$SNDR_HOME` only) |
| Recommended remote mode | SSH tunnel to loopback |

## User authentication (accounts, 2FA, OAuth)

Beyond the shared `SNDR_GUI_TOKEN`, the daemon supports full user authentication
that adapts to the deployment context (host vs container). Backends compose:

- **Local accounts** — usernames + `scrypt`-hashed passwords (stdlib, no deps).
- **System login (PAM)** — optional; authenticate against host OS accounts.
- **Google / Apple** — optional OIDC sign-in when client credentials are set.
- **TOTP 2FA** — RFC 6238 one-time codes (Google Authenticator, Authy, 1Password).

Sessions are stateless HMAC-signed tokens delivered as an httpOnly cookie
(same-origin) and also returned in the login response for cross-origin/API use.

### Enablement (`SNDR_AUTH`)

| Value | Behaviour |
| --- | --- |
| `on` | Always require login; manage accounts. |
| `off` | No auth (open localhost). |
| `auto` (default) | Require auth when bound beyond loopback, when accounts already exist, or when `SNDR_GUI_TOKEN` is set. |

On first start with auth enabled and no accounts, an **admin is bootstrapped**
from the system user running the daemon. Its password is taken from
`SNDR_ADMIN_PASSWORD` if set, otherwise generated and printed **once** to the
daemon log — store it and change it after first login.

### Environment variables

```bash
SNDR_AUTH=on                       # on | off | auto (default)
SNDR_ADMIN_PASSWORD=...            # initial admin password (else auto-generated once)
SNDR_AUTH_SESSION_TTL=86400        # session lifetime, seconds
SNDR_PUBLIC_URL=https://gui.host   # public base URL (OAuth redirect + secure cookie)

# Optional system-account login (host deployments; needs python-pam)
SNDR_AUTH_PAM=1

# Optional OAuth (inert until set). Register the redirect URI shown below.
SNDR_OAUTH_GOOGLE_CLIENT_ID=...
SNDR_OAUTH_GOOGLE_CLIENT_SECRET=...
SNDR_OAUTH_APPLE_CLIENT_ID=...
SNDR_OAUTH_APPLE_CLIENT_SECRET=...   # Apple's signed client-secret JWT
```

OAuth redirect URI to register with the provider:
`<SNDR_PUBLIC_URL>/api/v1/auth/oauth/<google|apple>/callback`

### Persistence (container-safe)

All auth state lives under `$SNDR_HOME/auth` (default `~/.sndr/auth`):

- `users.json` — accounts, roles, 2FA enrolments (file mode `0600`).
- `session.key` — the session signing key (`0600`).

Mount `$SNDR_HOME` as a volume so accounts, 2FA and sessions survive container
restarts:

```bash
docker run -d \
  -e SNDR_HOME=/data/sndr \
  -e SNDR_AUTH=on \
  -e SNDR_ADMIN_PASSWORD="change-me" \
  -v sndr-data:/data/sndr \         # <-- persistent volume
  -p 8765:8765 \
  your-image \
  python3 -m sndr.cli gui-api --host 0.0.0.0 --port 8765
```

Without the volume, the user store is recreated empty on each restart and a new
admin is bootstrapped.

### PAM vs container caveat

PAM authenticates against the OS accounts **visible to the process**. Inside a
container that is the *container's* user database, not the host's, unless host
auth files are bind-mounted and the container is privileged (an anti-pattern).
For container deployments prefer local accounts (and OAuth); reserve PAM for the
daemon running directly on a host.

### Roles & endpoints

Roles: `admin` (manage users), `operator`, `viewer`. Auth endpoints:

```text
POST /api/v1/auth/login              {username,password} -> session (or needs_2fa)
POST /api/v1/auth/login/2fa          {username,code}     -> session
POST /api/v1/auth/logout
GET  /api/v1/auth/me
POST /api/v1/auth/password           {current,new}
GET  /api/v1/auth/users              (admin) list
POST /api/v1/auth/users              (admin) {username,password,role}
DELETE /api/v1/auth/users/{username} (admin)
POST /api/v1/auth/2fa/enroll | activate {code} | disable
GET  /api/v1/auth/oauth/{provider}/login | callback
```

The legacy `SNDR_GUI_TOKEN` bearer continues to authorize API/service clients
regardless of user accounts.

## Hardening (brute-force, sessions, CSRF, 2FA recovery, audit)

The auth subsystem ships defence-in-depth beyond passwords + 2FA:

- **Brute-force throttle.** Repeated failed logins (and 2FA codes) lock the
  account temporarily (HTTP `429`). Tunable: `SNDR_AUTH_LOCK_THRESHOLD` (8),
  `SNDR_AUTH_LOCK_WINDOW` (300s), `SNDR_AUTH_LOCK_SECONDS` (900s). The lock is
  temporary/auto-clearing (a permanent lock would enable a targeted DoS).
- **Session revocation.** Session tokens carry an epoch bound to the account; a
  **password change** or **"sign out everywhere"** (`POST /auth/sessions/revoke`)
  bumps it, instantly invalidating every previously-issued token.
- **CSRF.** Cookie-authenticated mutating requests must be same-origin
  (`Sec-Fetch-Site` / `Origin` checked). Bearer-token clients are exempt (the
  attacker cannot read the token); OAuth callbacks are exempted by path.
- **2FA recovery codes.** Enabling 2FA issues 10 single-use recovery codes
  (scrypt-hashed, shown once). Login accepts a recovery code in place of a TOTP;
  `POST /auth/2fa/recovery` regenerates the set.
- **OAuth privilege.** OAuth sign-ups are always `operator` — never auto-admin
  (no race-to-be-first escalation); an admin promotes them explicitly.
- **Audit log.** Login success/failure/lockout, logout, user create/delete,
  2FA enable/disable, password change and session revocation are recorded to the
  events feed (`GET /api/v1/events/recent`, kind `auth`).
- **Engine DoS guard.** GUI-initiated generations cap `max_tokens` (4096) and the
  micro-benchmark caps requests/concurrency.

Residual notes: the throttle state is in-memory (cleared on restart); set
`SNDR_PUBLIC_URL=https://…` in production so the session cookie gets the
`secure` flag; put the daemon behind a TLS reverse proxy for transport security.
