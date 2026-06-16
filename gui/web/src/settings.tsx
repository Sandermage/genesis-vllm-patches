// SPDX-License-Identifier: Apache-2.0
// Shared GUI-settings vocabulary + theme helpers. Lives outside App.tsx so that
// extracted panels (CommandPalette, OperationalConsole, …) can reference the
// settings types and theme cycle without importing back into the app shell.
import { type ReactNode } from "react";
import { Sun, Moon, Sparkles, Leaf } from "lucide-react";
import { tr } from "./i18n";

/** Tabs on the operational console (jobs / events / logs / cli). */
export type ConsoleTab = "jobs" | "events" | "logs" | "cli";

export type ThemeMode = "light" | "dark" | "carbon" | "lime";
export type DensityMode = "comfortable" | "compact";
export type AccentMode = "teal" | "blue" | "emerald" | "amber";
export type DetailMode = "operator" | "engineer";

/** Persisted operator UI preferences. */
export type GuiSettings = {
  theme: ThemeMode;
  density: DensityMode;
  accent: AccentMode;
  detailMode: DetailMode;
  showConnectionMap: boolean;
  autoRefresh: boolean;
  sidebarCollapsed: boolean;
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
  detailMode: "engineer",
  showConnectionMap: true,
  autoRefresh: false,
  sidebarCollapsed: false,
  remoteHost: DEFAULT_REMOTE_HOST
};

function isAccent(value: unknown): value is AccentMode {
  return value === "teal" || value === "blue" || value === "emerald" || value === "amber";
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
      detailMode: parsed.detailMode === "operator" ? "operator" : "engineer",
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
      remoteHost:
        typeof parsed.remoteHost === "string" && parsed.remoteHost.trim()
          ? parsed.remoteHost.trim()
          : defaultGuiSettings.remoteHost
    };
  } catch {
    return defaultGuiSettings;
  }
}
