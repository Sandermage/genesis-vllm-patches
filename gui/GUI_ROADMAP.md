# SNDR GUI — modernization roadmap & progress

Derived from a code audit + a survey of leading admin panels (React-Admin, Refine,
shadcn/ui, Tremor, Grafana, Portainer, Proxmox, Coolify, OpenWebUI). Goal: a GUI
that is structurally + adaptively compatible with the SNDR backend and bug-free.

## What's already excellent (keep)

Clean `lib/`/`hooks/`/`components/`/`sections/` layering · aggressive code-splitting
(~25 `lazy()`) · strict TS with **zero** `as any`/`@ts-ignore`/TODO · real ultrawide
adaptivity (`useViewport` + `[data-viewport]`) · unified overlay/toast/a11y system ·
daemon-down self-recovery.

## Progress

- [x] **Iteration 1 — quick win + systematize** (`213df92b`)
  - macro tool: 4 rapid Yahoo calls → 429; now ONE batched `/finance/spark` request.
  - section registry: the section list was hand-synced in 3 places (`SectionId`,
    `SECTION_IDS`, `navGroups`); now a single `NAV_SECTIONS` + `ROUTABLE_ONLY` in
    `nav.ts` from which all three are derived. Drift class removed.
- [x] **Iteration 2 — data layer (TanStack Query)**
  - `QueryClientProvider` in `main.tsx` (operator defaults: 10s stale, retry 1, no
    refetch-on-focus). `hooks/useLibrary.ts` = typed query/mutation hooks with a
    shared cache. Reference migration: the prompts/tools library + the chat prompt
    selector now share `["prompts"]` — a mutation invalidates one key and every
    consumer updates, no manual refetch. This is the pattern to extend.

- [x] **Iteration 3 — section data layer on Query**
  - `hooks/useApiQuery.ts`: a `useFetch`-compatible read hook backed by TanStack
    Query (shared/deduped cache, retry, AbortSignal) but with an EXPLICIT queryKey
    (backing `useFetch` transparently was unsafe — two fetchers with the same deps
    would collide). Migrated all 7 `useFetch` call sites (models-workbench,
    diagnostics, catalog-cards) and retired the now-dead `useFetch.ts`.
- [x] **Iteration 4 — break up the App.tsx god file (3365 → 1497 lines, −56%)**
  - Done as 5 behaviour-neutral, independently verified relocations (each:
    `tsc -b` + `vite build` + `eslint` + serve-smoke 200, then a commit), per the
    one-piece-at-a-time method:
    1. `lib/readiness-gates.ts` (buildReadinessGates/targetStatus/countGates) +
       `hooks/useLiveEvents.ts` (the SSE/polling event feed).
    2. settings persistence (`GUI_SETTINGS_STORAGE_KEY`, `defaultGuiSettings`,
       `loadGuiSettings`, `isAccent`) co-located into `settings.tsx`.
    3. `lib/overview-presenters.ts` (buildEvents/buildCliMirror/runtimeHost) +
       dropped App's duplicate `LoadState` (consolidated on `FetchState`).
    4. `lib/section-spec.ts` (per-section header metadata) + `WorkflowSteps` into
       `components/shell-bits.tsx`.
    5. **Capstone:** `sections/section-workspace.tsx` (the ~1320-line, 50-prop
       section renderer) + `lazy-panels.ts` (the 24 code-split panels). Static
       analysis confirmed SectionWorkspace references **zero** App-local helpers,
       so the move was a clean relocation; App dropped 85 now-unused imports.
  - App.tsx is now a pure shell (state / effects / routing / layout / modals).
    Bundle output unchanged (index 319.70 kB) — confirms behaviour preserved.
- [x] **Iteration 5 — split section-workspace.tsx into per-section components
  (1400 → 493 lines, −65%)**
  - The standalone section renderer (extracted in Iteration 4) was itself a god
    file: one 50-prop component rendering a `{sectionId === "X" && …}` block per
    section. Split the 11 substantial blocks into self-contained section
    components under `sections/*-section.tsx`, each with a typed minimal-prop
    interface (8 separate verified commits, same tsc/build/eslint/smoke gate):
    Advanced, Presets, Overview, Clients, Patches, Benchmarks, Evidence, Setup,
    Services, Doctor, Reports.
  - **Cohesion win:** UI state moved into the section that owns it — `presetTab`
    into PresetsSection, `setupTab` + the install-intent effect into SetupSection.
    The dispatcher now holds no section-local state, only pure derivations it
    fans out (card / composed / patchRows). Sections derive their own small
    rollups (bench-proven / family / workload counts) from props.
  - section-workspace.tsx is now a 493-line dispatcher (header + the small
    single-panel sections, which weren't worth extracting). Largest section file
    is 240 lines (Advanced). Bundle output unchanged. The 50-prop dispatcher
    signature remains — a context/props-bundle is the optional next polish.

## Next (prioritized)

1. **Extend Query to the App-level loads** — replace the hand-rolled `usePoll` +
   the ~45-`useState` orchestration in App.tsx. Higher value but entangled with the
   god file, so best done WITH breaking App.tsx up (step 3). Kills the
   swallowed-`catch{}` gaps via Query's built-in error/retry.
2. **Typed contract codegen — BLOCKED on the backend.** Survey + measurement: the
   daemon's OpenAPI is **0% typed** (185 routes, 0 with a real 200 response schema;
   they return `dict[str, Any]`). `openapi-typescript` would emit `unknown`
   everywhere. **Prereq:** give the routes `response_model`s built from the frozen
   dataclasses (a backend track), *then* generate `src/api/schema.gen.ts` (already
   referenced in `eslint.config.js`) and drop the 41 `Record<string, any>` holes.
   Until then the hand-written types stay (they're good — ~80 thorough types).
3. **Break up the god files** — App.tsx ✅ (Iteration 4: 3365 → 1497, shell only)
   and section-workspace.tsx ✅ (Iteration 5: 1400 → 493, dispatcher + 11 section
   components). Remaining: Containers.tsx (32 `useState`, 7 inline tabs). Optional
   polish: replace the section-workspace dispatcher's 50-prop signature with a
   context / props-bundle.
4. **Component system** — a `SelectableCard` primitive (unify fleet/launch/catalog
   card variants), promote `PanelHeader`/`EmptyState`, a `cmdk` command palette.
5. **Workspace** — extend the prompt/tool library to the full OpenWebUI idiom
   (Clone / Export-JSON / Import, + Models presets).
6. **i18n tooling** — a `tr()` extractor + CI untranslated-key audit (the
   ~2,500-entry source-keyed `RU_BY_EN` map silently un-translates on a copy edit).
7. **Build integration** — make `dist → web_static` a build step of `sndr` (today
   it's a manual sync), and a11y lint `warn` → `error` for new code.
