<!-- SPDX-License-Identifier: Apache-2.0 -->
# SNDR Control Center — Web GUI

The operator dashboard for the SNDR / Genesis vLLM control plane. A single-page
React app that talks to the daemon's read-only-by-default Product API and gives
one pane over the whole stack: fleet, containers, GPU/hardware telemetry,
Kubernetes & Proxmox virtualization, the model/preset/config catalog, the launch
planner, spec-decode routing, patches & flags, a live engine chat + ops copilot,
and a built-in SSH terminal.

It is built as a static bundle and **served by the daemon itself** (FastAPI,
`sndr.product_api`) from `web_static/` — there is no separate web server in
production.

---

## Tech stack

| Concern | Choice | Notes |
|---|---|---|
| Framework | **React 18 + TypeScript** | function components + hooks only |
| Build | **Vite** | `tsc -b` typecheck + `vite build` bundle |
| Icons | **lucide-react** | per-icon named imports (tree-shaken) |
| Terminal | **@xterm/xterm** + addon-fit | lazy-loaded into its own chunk |
| Styling | one hand-written **`src/styles.css`** | CSS custom properties, dark/light themes; fonts are `Inter` / system mono (no web-font dependency) |
| i18n | **custom flat dictionary** (`src/i18n.ts`) | EN + RU, no runtime framework |
| Server cache | **TanStack Query v5** | shared/deduped read cache + mutations (`hooks/useApiQuery.ts`, `hooks/useLibrary.ts`) with explicit query keys |
| UI state | plain React hooks | component-local; no Redux/Zustand, no router library |

No global UI-state or router library is used: the app routes by URL hash and keeps
view state in component hooks. Server reads go through a thin `src/api.ts` client
wrapped in **TanStack Query**, so one endpoint shares a single deduped, retrying
cache across screens and a mutation invalidates it in one place — e.g. the
prompt/tool library and the chat prompt selector share the `["prompts"]` key, so
editing a prompt updates both with no manual refetch. This keeps the bundle small
and the dependency surface tiny while removing hand-rolled refetch/`catch {}` gaps.

---

## Layout

```
gui/web/
  index.html              # Vite entry
  vite.config.ts          # build config
  tsconfig.json           # tsc -b project (include: ["src"])
  eslint.config.js        # flat ESLint config (react-hooks + jsx-a11y)
  package.json            # dev/build/preview/typecheck/lint only
  src/
    main.tsx              # React root + QueryClientProvider
    App.tsx               # shell only: sidebar nav, topbar, routing, modals, lazy boundary
    api.ts                # typed Product API client (fetch + auth headers)
    i18n.ts               # EN/RU dictionary + t()/tr() + useLang()
    styles.css            # the entire stylesheet
    nav.ts                # single-source section registry (ids + nav metadata)
    lazy-panels.ts        # the React.lazy() panel registry (shared by shell + renderer)
    hooks/                # useViewport, useApiQuery, useLibrary, useLiveEvents
    lib/                  # pure helpers (coerce, format, readiness-gates, section-spec, …)
    components/           # shared primitives (dialogs, code-block, toast, tables, …)
    sections/             # one module per panel; section-workspace.tsx dispatches to
                          #   the per-section *-section.tsx components
    <Panel>.tsx           # large panels at the root (Containers, Engine, Routing, …)
```

~110 source modules. The shell (`App.tsx`) owns only state/routing/layout/modals;
`sections/section-workspace.tsx` is a thin dispatcher that renders the active
`<XSection/>` component. Heavy panels are `React.lazy`-loaded (from
`lazy-panels.ts`) behind one `Suspense` boundary, so the initial chunk carries
only the shell + the Overview path; the xterm terminal and each section load on
demand.

> **History:** `App.tsx` and `section-workspace.tsx` were each broken out of a
> ~3.3k / ~1.4k-line god file into a shell + focused modules (see the GUI roadmap
> in `sndr_private/planning/gui/GUI_ROADMAP.md`). Behaviour-neutral, verified by
> `tsc -b` + `vite build` + `eslint` at every step.

---

## Sections

The sidebar groups every surface:

- **Overview** — at-a-glance KPIs + readiness.
- **Infrastructure** — Fleet (per-host engine probe over SSH), Containers
  (Docker control + logs + live KV/throughput KPIs), Kubernetes (read-only
  cluster view), Virtualization (Proxmox VE hosts/guests + KubeVirt), Hardware
  (per-GPU telemetry over nvidia-smi).
- **Setup** — guided first-run checklist, install-onto-host, deploy-a-model.
- **Models & Config** — Models workbench, Presets catalog + recommender, Configs
  V2 composer (model + hardware + profile → runtime config).
- **Planner / Launch Plan / Services** — KV-cache & VRAM fit calculator,
  recommend → compose → gate → launch, service lifecycle.
- **Engine** — multi-turn streaming Chat against a running vLLM model, and an
  Ops Copilot (read-only tool-calling assistant over the Product API).
- **Routing** — the deterministic spec-decode workload router + live classifier.
- **Validate** — Doctor diagnostics, Patches inventory/matrix, env-flag matrix.
- **Benchmarks / Evidence / Reports / Tools** — bench A/B, proof bundles, report
  export, command palette, settings, audit log.

---

## Live-wired data — nothing hardcoded

Every project-state value the GUI shows is **fetched at runtime from the live
`sndr` package** via the Product API; there are no baked-in project facts. This
is by design, so the console never drifts from the engine:

| Surface | Source (runtime) | Never hardcoded |
|---|---|---|
| Patch count | `sndr.dispatcher.spec.iter_patch_specs()` → `patches.total` (backend `patches/listing.py`); rendered from `patches?.total` | no `TOTAL_PATCHES` / `321` literal in `src/` |
| vLLM pin(s) | `updater.supported_pins()` regex-scans `vllm_pin_required:` out of the live model YAMLs; `canonical_pin = pins[0]` | no `devNNN` pin literal in the bundle |
| Models | V2 catalog via `sndr.model_configs.registry_v2` (`overview.collect_catalog_summary()`) | the `qwen3.6-*` strings in `src/` are fallback **seeds** / i18n examples, replaced by `loadAll()` on mount |
| SNDR Core version | `overview.capabilities.platform.sndr_core_version` | no `12.0.0` literal in `src/` |
| Deploy artifact | `deployment.py` composes from `registry_v2` + `runtime_command.build_runtime_command()` (embeds `genesis_pin` from live config) | — |

The served bundle is built by `make gui-build` and committed to
`sndr/product_api/legacy/web_static/`; the daemon serves it same-origin. Because
all state is API-sourced, the bundle automatically reflects the current pin
(dev424), patch count (321), and model catalog without a rebuild — a rebuild is
only needed for UI/code changes, not data changes.

---

## Adaptive layout

The GUI targets everything from a laptop to a 3440-wide ultrawide workstation.
A `useViewport` hook ([`src/hooks/useViewport.ts`](src/hooks/useViewport.ts))
resolves the window width into one of four tiers and stamps it on the shell as a
`data-viewport` attribute, so CSS can branch without prop-drilling:

| Tier | Width | Target |
|---|---|---|
| `compact` | `< 1280` | laptops |
| `standard` | `1280–1919` | standard desktops |
| `wide` | `1920–2879` | wide monitors |
| `ultra` | `≥ 2880` | 3440 ultrawide |

The design principle on wide screens is **fill the width, earn it** — tiles stay
full-width but are enriched (larger value + sub-label + trailing context /
sparkline) rather than centered behind a max-width gutter. Height-bound panels
(chat, terminal, log viewers) size off `vh` so they fit the viewport on any
screen.

---

## Internationalization (EN / RU)

i18n is a dependency-free flat dictionary in
[`src/i18n.ts`](src/i18n.ts):

- `t(lang, "nav.key")` — semantic keys for the navigation/structural strings.
- `tr("English source")` — source-string lookup used by every screen: it returns
  the Russian from the `RU_BY_EN` map when the UI is in Russian, and falls back
  to the English source otherwise. New strings are localized by wrapping the
  literal in `tr("…")` — no key invention, no `lang` threading.
- `useLang()` persists the choice and broadcasts a change event so the whole tree
  re-renders on toggle.

The Russian map carries ~2.5k professionally-reviewed strings (single voice:
«вы» + infinitive on controls, consistent IT terminology). To add a language,
add a second source-keyed map and branch in `tr()`.

---

## Develop

```bash
cd gui/web
npm install
npm run dev          # Vite dev server on http://127.0.0.1:5173
```

Point the dev GUI at a running daemon with `VITE_SNDR_API_BASE`
(default same-origin):

```bash
VITE_SNDR_API_BASE=http://127.0.0.1:8765 npm run dev
```

Other scripts:

```bash
npm run typecheck     # tsc -b, no emit
npm run lint          # eslint .
npm run build         # tsc -b && vite build  → dist/
npm run preview       # serve the built dist/ for a production-parity check
```

---

## Build & deploy

The daemon serves the bundle from `web_static/` next to the Product API. The
release flow is: build, then copy `dist/` into the served directory.

```bash
cd gui/web
npm run build
rm -rf ../../sndr/product_api/legacy/web_static
cp -R dist ../../sndr/product_api/legacy/web_static
```

`SNDR_GUI_STATIC` can override the served directory; otherwise the daemon
resolves the packaged `web_static/` automatically. After deploying, hard-refresh
the browser (`Cmd/Ctrl+Shift+R`) so the new hashed bundle loads.

The daemon gzips the bundle and every API response, so the ~190 KB-gzipped JS +
~50 KB-gzipped CSS is what actually crosses the wire on a cold load.

---

## Security

Browser-side hardening is enforced by the daemon on every response (see
`sndr.product_api.legacy.http_app`): a strict Content-Security-Policy
(`script-src 'self'`, `frame-ancestors 'none'`, `object-src 'none'`),
`X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`,
`Referrer-Policy: no-referrer`, `Cross-Origin-Opener-Policy`, a `Permissions-Policy`
lockdown, and `Strict-Transport-Security` when served over TLS. The GUI never
persists the engine API key (memory only), and CORS defaults to localhost with
`allow_credentials=false`. The daemon is read-only by default; mutating actions
require `SNDR_ENABLE_APPLY` and explicit confirmation.

---

## Notes

- Tests (vitest unit + Playwright e2e) and dev-only tooling are kept out of the
  public release bundle; they live in the private archive alongside the parked
  Carbon-based GUI rewrite experiment.
- The shipped GUI is `src/App.tsx`. There is no second active frontend.
