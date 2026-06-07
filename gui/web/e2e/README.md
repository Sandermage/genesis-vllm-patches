# End-to-end tests

Two tiers, by design — they answer different questions and run in different places.

## 1. Hermetic tier (CI) — `hermetic.spec.ts`

No backend. Every `/api/**` call is mocked from `fixtures.ts` (shapes shared with
the jsdom shell test via `../src/test/fixtures-data.ts`), and the run is against
`vite preview` of the production bundle.

```bash
npm run test:e2e:ci      # playwright.hermetic.config.ts (auto-starts preview)
```

It proves, with **no daemon**:
- the production bundle boots without a runtime/JS crash;
- every sidebar entry routes to a real panel (no dead routes, no blank/crashed
  section on empty data);
- the real, CSS-composed DOM is free of structural accessibility violations
  (axe: roles, names, landmarks, ARIA) on every section.

This is the tier wired into `.github/workflows/gui_web.yml` (the `e2e` job).

## 2. Integration tier (dev box) — `smoke`, `host_wiring`, `server_switch`, `chat_rag`

These drive real flows against a **live read-only daemon** (`sndr gui-api`, :8765)
plus the Vite dev server. They are NOT hermetic and NOT run in CI — they validate
the GUI against an actual backend on the dev box / staging.

```bash
# Terminal 1: the daemon          Terminal 2: the GUI
sndr gui-api                      npm run dev:carbon
# Terminal 3:
npm run test:e2e                 # playwright.config.ts (ignores hermetic.spec.ts)
# override the origin if Vite picked a fallback port:
PLAYWRIGHT_BASE_URL=http://127.0.0.1:5174 npm run test:e2e
```

`playwright.config.ts` ignores `hermetic.spec.ts` so the two tiers never overlap.

## Accessibility

`hermetic.spec.ts` gates the structural axe rules on every section. `color-contrast`
is improved to WCAG AA on the default (light) theme plus dark/carbon (see
`src/styles.css`) but is not yet gated across the stylised `lime` palette and the
full set of stat-label classes — a tracked design-token initiative.
