// SPDX-License-Identifier: Apache-2.0
// Shared navigation + readiness-gate vocabulary. Lives outside App.tsx so that
// extracted section panels (GateRow, SetupWizard, …) can reference SectionId /
// RuntimeMode / Gate without importing back into the app shell.
import { type GateStatus } from "./components/primitives";

/** Every routable section id in the Control Center shell. */
export type SectionId =
  | "overview"
  | "setup"
  | "fleet"
  | "hosts"
  | "hardware"
  | "models"
  | "configs"
  | "presets"
  | "planner"
  | "copilot"
  | "launch-plan"
  | "services"
  | "containers"
  | "kubernetes"
  | "virtualization"
  | "routing"
  | "doctor"
  | "patches"
  | "flags"
  | "benchmarks"
  | "evidence"
  | "clients"
  | "chat"
  | "reports"
  | "operations"
  | "advanced";

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
