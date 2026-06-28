// SPDX-License-Identifier: Apache-2.0
// Shared GUI-settings vocabulary + theme helpers. Lives outside App.tsx so that
// extracted panels (CommandPalette, OperationalConsole, …) can reference the
// settings types and theme cycle without importing back into the app shell.
import { type ReactNode } from "react";
import { Sun, Moon, Sparkles, Leaf } from "lucide-react";
import { tr } from "./i18n";
import { type RuntimeMode } from "./nav";

/** Tabs on the operational console (jobs / events / logs / cli). */
export type ConsoleTab = "jobs" | "events" | "logs" | "cli";

export type ThemeMode = "light" | "dark" | "carbon" | "lime";
export type DensityMode = "comfortable" | "compact";
export type AccentMode = "teal" | "blue" | "emerald" | "amber";
/** Simple = consumer surface (~5 nav items, the choice-first funnel);
 *  Expert = the full operator workbench (~21 sections). Stored values from the
 *  prior "operator"/"engineer" naming are parsed back-compat (see
 *  loadGuiSettings): old "operator" → "simple", old "engineer" → "expert". */
export type DetailMode = "simple" | "expert";

/** Persisted operator UI preferences. */
export type GuiSettings = {
  theme: ThemeMode;
  density: DensityMode;
  accent: AccentMode;
  detailMode: DetailMode;
  showConnectionMap: boolean;
  autoRefresh: boolean;
  sidebarCollapsed: boolean;
  /** Where the GPU engine runs (launch/SSH/chat host hints). A FRESH user with
   *  no configured remote host lands in "local" — the beginner-first path that
   *  drives this machine's local daemon at 127.0.0.1:8765 — instead of the
   *  remote-SSH fleet flow. The choice persists, so a returning user who picked
   *  "remote" (configuring a GPU node) keeps it. */
  runtimeMode: RuntimeMode;
  /** Engine host used in "Remote GPU" runtime mode (reachability probes,
   *  lifecycle planner, endpoint rows). Operator-editable so it can point at a
   *  real GPU node instead of an unresolvable placeholder. */
  remoteHost: string;
};

/** Fallback remote-host label when the operator hasn't set a real one yet. */
export const DEFAULT_REMOTE_HOST = "gpu-build-01";

export const THEME_CYCLE: ThemeMode[] = ["light", "dark", "carbon", "lime"];
export const VALID_THEMES = new Set<ThemeMode>(THEME_CYCLE);

/** Next theme in the cycle (wraps around). */
export function nextTheme(current: ThemeMode): ThemeMode {
  const index = THEME_CYCLE.indexOf(current);
  return THEME_CYCLE[(index + 1) % THEME_CYCLE.length] ?? "dark";
}

export function themeLabel(theme: ThemeMode): string {
  return theme === "light" ? tr("Light") : theme === "dark" ? tr("Dark") : theme === "carbon" ? tr("Carbon") : tr("Lime");
}

export function themeIcon(theme: ThemeMode): ReactNode {
  return theme === "light" ? <Sun size={16} /> : theme === "carbon" ? <Sparkles size={16} /> : theme === "lime" ? <Leaf size={16} /> : <Moon size={16} />;
}

/** localStorage key for persisted operator UI preferences. */
export const GUI_SETTINGS_STORAGE_KEY = "sndr.gui.settings";

export const defaultGuiSettings: GuiSettings = {
  theme: "light",
  density: "comfortable",
  accent: "teal",
  // A FRESH user (no stored setting) lands in Simple mode — the choice-first
  // consumer surface. Returning users keep whatever they last chose (parsed
  // back-compat from the old "operator"/"engineer" values in loadGuiSettings).
  detailMode: "simple",
  showConnectionMap: true,
  autoRefresh: false,
  sidebarCollapsed: false,
  // A FRESH user (no configured remote host) drives the LOCAL daemon — the
  // beginner-first "pick a model → launch → chat" path, not the remote-SSH
  // fleet flow. Returning users keep whatever runtime mode they last chose.
  runtimeMode: "local",
  remoteHost: DEFAULT_REMOTE_HOST
};

function isAccent(value: unknown): value is AccentMode {
  return value === "teal" || value === "blue" || value === "emerald" || value === "amber";
}

/** Parse a stored detailMode, accepting both the current ("simple"/"expert")
 *  and the legacy ("operator"/"engineer") vocabularies so a returning user
 *  keeps their pref across the rename. Legacy "operator" → simple,
 *  "engineer" → expert. Anything else (incl. a missing value) → the default. */
function parseDetailMode(value: unknown): DetailMode {
  if (value === "simple" || value === "operator") return "simple";
  if (value === "expert" || value === "engineer") return "expert";
  return defaultGuiSettings.detailMode;
}

/** Read + validate persisted operator settings, repairing any missing/corrupt
 *  field from defaults (defensive against localStorage schema drift). */
export function loadGuiSettings(): GuiSettings {
  try {
    const raw = window.localStorage.getItem(GUI_SETTINGS_STORAGE_KEY);
    if (!raw) return defaultGuiSettings;
    const parsed = JSON.parse(raw) as Partial<GuiSettings>;
    return {
      ...defaultGuiSettings,
      ...parsed,
      theme: parsed.theme && VALID_THEMES.has(parsed.theme) ? parsed.theme : "light",
      density: parsed.density === "compact" ? "compact" : "comfortable",
      accent: isAccent(parsed.accent) ? parsed.accent : defaultGuiSettings.accent,
      detailMode: parseDetailMode(parsed.detailMode),
      showConnectionMap:
        typeof parsed.showConnectionMap === "boolean"
          ? parsed.showConnectionMap
          : defaultGuiSettings.showConnectionMap,
      autoRefresh:
        typeof parsed.autoRefresh === "boolean"
          ? parsed.autoRefresh
          : defaultGuiSettings.autoRefresh,
      sidebarCollapsed:
        typeof parsed.sidebarCollapsed === "boolean"
          ? parsed.sidebarCollapsed
          : defaultGuiSettings.sidebarCollapsed,
      // Honor a returning user's persisted runtime mode; a fresh user (no stored
      // value) gets the local-first default.
      runtimeMode:
        parsed.runtimeMode === "remote" || parsed.runtimeMode === "local"
          ? parsed.runtimeMode
          : defaultGuiSettings.runtimeMode,
      remoteHost:
        typeof parsed.remoteHost === "string" && parsed.remoteHost.trim()
          ? parsed.remoteHost.trim()
          : defaultGuiSettings.remoteHost
    };
  } catch {
    return defaultGuiSettings;
  }
}
