// SPDX-License-Identifier: Apache-2.0
// Shared navigation + readiness-gate vocabulary. Lives outside App.tsx so that
// extracted section panels (GateRow, SetupWizard, …) can reference SectionId /
// RuntimeMode / Gate without importing back into the app shell.
import { type GateStatus } from "./components/primitives";
import { type LucideIcon, BarChart3, Box, Boxes, Cpu, Database, FileText, Gauge, Home, LayoutGrid, Link2, MessageSquare, Network, Rocket, Route, Server, Settings, ShieldCheck, SlidersHorizontal, Table2, Wrench } from "lucide-react";

/** A top-level nav section: id + sidebar placement (group header + icon). */
export type SectionDescriptor = { id: string; label: string; group: string; icon: LucideIcon };

/**
 * SINGLE SOURCE OF TRUTH for the sidebar, in render order. `group: ""` = the
 * ungrouped lead item. The grouped nav (App.tsx), `SectionId` and `SECTION_IDS`
 * are all DERIVED from this + `ROUTABLE_ONLY` — add a section in one place.
 */
export const NAV_SECTIONS = [
  { id: "overview", label: "Overview", group: "", icon: Home },
  { id: "hosts", label: "Fleet", group: "Infrastructure", icon: LayoutGrid },
  { id: "containers", label: "Containers", group: "Infrastructure", icon: Boxes },
  { id: "virtualization", label: "Virtualization", group: "Infrastructure", icon: Server },
  { id: "hardware", label: "Hardware", group: "Infrastructure", icon: Cpu },
  { id: "setup", label: "Setup", group: "Infrastructure", icon: Settings },
  { id: "models", label: "Models", group: "Models & Config", icon: Box },
  { id: "presets", label: "Presets", group: "Models & Config", icon: Database },
  { id: "configs", label: "Configs", group: "Models & Config", icon: SlidersHorizontal },
  { id: "planner", label: "Planner", group: "Models & Config", icon: Gauge },
  { id: "launch-plan", label: "Launch Plan", group: "Deploy", icon: Rocket },
  { id: "services", label: "Services", group: "Deploy", icon: Network },
  { id: "chat", label: "Chat & Copilot", group: "Engine", icon: MessageSquare },
  { id: "routing", label: "Routing", group: "Engine", icon: Route },
  { id: "clients", label: "Clients", group: "Engine", icon: Link2 },
  { id: "doctor", label: "Doctor", group: "Validate", icon: ShieldCheck },
  { id: "patches", label: "Patches", group: "Validate", icon: Wrench },
  { id: "benchmarks", label: "Benchmarks", group: "Validate", icon: BarChart3 },
  { id: "evidence", label: "Evidence", group: "Validate", icon: FileText },
  { id: "reports", label: "Reports", group: "Validate", icon: Table2 },
  { id: "advanced", label: "Advanced", group: "Tools", icon: SlidersHorizontal },
] as const satisfies readonly SectionDescriptor[];

/** Routable, but not shown as a top-level nav item (deep-links / sub-tabs). */
export const ROUTABLE_ONLY = ["fleet", "copilot", "kubernetes", "flags", "operations"] as const;

/** Every routable section id (derived from NAV_SECTIONS + ROUTABLE_ONLY). */
export type SectionId = (typeof NAV_SECTIONS)[number]["id"] | (typeof ROUTABLE_ONLY)[number];

/** Routable section ids as a Set, for deep-link/hash validation (derived). */
export const SECTION_IDS: ReadonlySet<string> = new Set<string>([
  ...NAV_SECTIONS.map((s) => s.id), ...ROUTABLE_ONLY,
]);

/** Whether the GUI talks to a local daemon or a remote host over an SSH tunnel. */
export type RuntimeMode = "local" | "remote";

/** A single readiness gate surfaced on the overview / launch path. */
export type Gate = {
  id: string;
  label: string;
  detail: string;
  status: GateStatus;
  action: string;
};

/**
 * Maps a readiness-gate id to the section an operator should open to resolve it.
 * Keys include both dash and underscore spellings because gate ids arrive from
 * several backend producers; keep both in sync when adding a gate.
 */
export const GATE_TARGET: Record<string, { section: SectionId; label: string }> = {
  catalog: { section: "configs", label: "Open Configs" },
  "preset-card": { section: "presets", label: "Open Presets" },
  preset_card: { section: "presets", label: "Open Presets" },
  runtime: { section: "hosts", label: "Open Hosts" },
  engine: { section: "doctor", label: "Open Doctor" },
  engine_package: { section: "doctor", label: "Open Doctor" },
  patch_doctor: { section: "patches", label: "Open Patch Doctor" },
  "service-api": { section: "services", label: "Open Services" },
  service_lifecycle: { section: "services", label: "Open Services" },
  evidence: { section: "evidence", label: "Open Evidence" },
  evidence_orchestration: { section: "evidence", label: "Open Evidence" },
  "release-proof": { section: "reports", label: "Open Reports" },
  release_proof: { section: "reports", label: "Open Reports" }
};
