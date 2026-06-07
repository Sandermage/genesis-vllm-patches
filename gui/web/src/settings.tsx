// SPDX-License-Identifier: Apache-2.0
// Shared GUI-settings vocabulary + theme helpers. Lives outside App.tsx so that
// extracted panels (CommandPalette, OperationalConsole, …) can reference the
// settings types and theme cycle without importing back into the app shell.
import { type ReactNode } from "react";
import { Sun, Moon, Sparkles, Leaf } from "lucide-react";

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
  return theme === "light" ? "Light" : theme === "dark" ? "Dark" : theme === "carbon" ? "Carbon" : "Lime";
}

export function themeIcon(theme: ThemeMode): ReactNode {
  return theme === "light" ? <Sun size={16} /> : theme === "carbon" ? <Sparkles size={16} /> : theme === "lime" ? <Leaf size={16} /> : <Moon size={16} />;
}
