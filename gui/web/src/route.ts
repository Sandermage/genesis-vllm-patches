// Shared hash-routing helpers, used by both App (section + preset routing) and
// ContainersPanel (container deep-linking). Lives in its own module so the two
// can import the same logic without a circular dependency (App imports
// ContainersPanel, so ContainersPanel must not import back from App).
//
// Hash grammar:  #<section>[?<k>=<v>&…]
//   #presets?id=prod-35b-multiconc
//   #containers?c=vllm-pn95-2xa5000&src=host-a5000   (src=local for the daemon socket)

// Every routable section id. Kept here (not derived from the nav) so a
// hand-typed/bookmarked deep-link can be validated before we trust it.
export const SECTION_IDS = new Set<string>([
  "overview", "setup", "fleet", "hosts", "models", "configs", "presets",
  "planner", "copilot", "launch-plan", "services", "containers", "doctor",
  "patches", "benchmarks", "evidence", "clients", "chat", "reports",
  "operations", "advanced",
]);

// The section segment of the current hash, validated against SECTION_IDS.
// Tolerates a leading `#`/`#/` and an optional `?query` suffix.
export function sectionFromHash(): string | null {
  if (typeof window === "undefined") return null;
  const raw = window.location.hash.replace(/^#\/?/, "").split("?")[0];
  return SECTION_IDS.has(raw) ? raw : null;
}

// Parsed query params from the current hash (the part after `?`).
function hashParams(): URLSearchParams {
  if (typeof window === "undefined") return new URLSearchParams();
  const h = window.location.hash;
  const q = h.indexOf("?");
  return new URLSearchParams(q === -1 ? "" : h.slice(q + 1));
}

// Read a single decoded query param from the hash (null when absent/empty).
export function hashParam(key: string): string | null {
  const v = hashParams().get(key);
  return v ? decodeURIComponent(v) : null;
}

// The `id` record selector — the preset deep-link param.
export function recordIdFromHash(): string | null {
  return hashParam("id");
}

// Compose a canonical hash for a section plus optional query params. Params with
// null/undefined/"" values are dropped, so callers can pass a record verbatim.
export function buildHash(section: string, params?: Record<string, string | null | undefined>): string {
  const qs = params
    ? Object.entries(params)
        .filter(([, v]) => v != null && v !== "")
        .map(([k, v]) => `${k}=${encodeURIComponent(v as string)}`)
        .join("&")
    : "";
  return qs ? `${section}?${qs}` : section;
}

// Replace the hash without adding a history entry (used for same-section record
// changes, so clicking through records doesn't flood the Back stack).
export function replaceHash(hash: string): void {
  if (typeof window === "undefined") return;
  const base = `${window.location.pathname}${window.location.search}`;
  window.history.replaceState(null, "", `${base}#${hash}`);
}
