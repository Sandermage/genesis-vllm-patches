// SPDX-License-Identifier: Apache-2.0
// The Advanced (developer surface) section: a tabbed workbench for operations,
// config-keys/traces references, appearance, license, notifications, API &
// schema, audit log, updates, admin and developer tools. Extracted from
// section-workspace.tsx so each section is a self-contained, testable unit.
import { Suspense } from "react";
import {
  Activity, BadgeCheck, Bell, Code2, Cpu, DownloadCloud, FileText, KeyRound,
  Palette, Settings, ShieldCheck, SlidersHorizontal, SquareTerminal, Terminal
} from "lucide-react";
import { tr } from "../i18n";
import type { AuthUser, EnvironmentReport, PatchDoctorReport, ProductCapability } from "../api";
import type { GuiSettings } from "../settings";
import { TabbedSection } from "../components/tabbed-section";
import { ModuleCard, ModuleGrid } from "../components/layout";
import { InfoRows } from "../components/primitives";
import { CodeBlock } from "../components/code-block";
import { CapabilityTable } from "../components/capability-table";
import { AppearanceSettings, LicensePanel, NotificationSettings, ApiTokenField, ApiTokenManager } from "../lazy-panels";
import { SkeletonCards } from "../Skeleton";
import { SecurityPanel, UserAdminPanel } from "../Auth";
import { UpdatesPanel } from "../Updates";
import { OperationsConsole } from "./operations";
import { ConfigKeysPanel, TracesPanel } from "./diagnostics";
import { EndpointExplorer } from "./api-explorer";
import { AuditLogPanel } from "./audit-log";
import { AdminSurfaceMatrix } from "./patch-doctor";
import { EnvironmentPanel } from "./environment";
import { ConfigDraftEditor } from "./config-draft-editor";
import { EventLog } from "./operational-console";

export function AdvancedSection({
  apiBase,
  settings,
  environment,
  authUser,
  featureRows,
  patchDoctor,
  selectedPreset,
  runtimeTarget,
  patchPolicy,
  composed,
  cliLines,
  events,
  onMonitorJob,
  onSettings,
  onAuthRefresh
}: {
  apiBase: string;
  settings: GuiSettings;
  environment: EnvironmentReport | null;
  authUser: AuthUser | null;
  featureRows: ProductCapability[];
  patchDoctor: PatchDoctorReport | null;
  selectedPreset: string;
  runtimeTarget: string;
  patchPolicy: string;
  composed: Record<string, unknown>;
  cliLines: string[];
  events: Array<[string, string, string]>;
  onMonitorJob: (id: string) => void;
  onSettings: (patch: Partial<GuiSettings>) => void;
  onAuthRefresh: () => void;
}) {
  return (
    <TabbedSection
      id="advanced"
      tabs={[
        {
          id: "operations",
          label: tr("Operations"),
          icon: <Terminal size={15} />,
          render: () => <OperationsConsole onMonitor={onMonitorJob} />
        },
        {
          id: "config-keys",
          label: tr("Config keys"),
          icon: <SlidersHorizontal size={15} />,
          render: () => (
            <ModuleGrid>
              <ModuleCard title={tr("Config-key glossary")} icon={<SlidersHorizontal size={18} />} desc={tr("Every GENESIS_ENABLE_* flag, V1/V2 config key and policy key with provenance — searchable operator reference (mirrors `sndr config-keys`).")} wide>
                <ConfigKeysPanel />
              </ModuleCard>
            </ModuleGrid>
          )
        },
        {
          id: "traces",
          label: tr("Traces"),
          icon: <Activity size={15} />,
          render: () => (
            <ModuleGrid>
              <ModuleCard title={tr("Diagnostic trace catalog")} icon={<Activity size={18} />} desc={tr("Per-patch debug traces — the container path each lands at and the env var that enables it. Operator reference (mirrors `sndr trace list`).")} wide>
                <TracesPanel />
              </ModuleCard>
            </ModuleGrid>
          )
        },
        {
          id: "appearance",
          label: tr("Appearance"),
          icon: <Palette size={15} />,
          render: () => (
            <ModuleGrid>
              <ModuleCard title={tr("Appearance and Operator Mode")} icon={<Palette size={18} />} wide>
                <AppearanceSettings settings={settings} onSettings={onSettings} />
              </ModuleCard>
            </ModuleGrid>
          )
        },
        {
          id: "license",
          label: tr("License & modules"),
          icon: <BadgeCheck size={15} />,
          render: () => (
            <ModuleGrid>
              <ModuleCard title={tr("License & SNDR Engine")} icon={<BadgeCheck size={18} />} desc={tr("Active tier (community vs commercial SNDR Engine), the signed license token (subject / expiry / signature), whether the vllm.sndr_engine overlay is installed, and how many engine-tier patches it unlocks.")} wide>
                <Suspense fallback={<SkeletonCards count={1} />}>
                  <LicensePanel />
                </Suspense>
              </ModuleCard>
            </ModuleGrid>
          )
        },
        {
          id: "notifications",
          label: tr("Notifications"),
          icon: <Bell size={15} />,
          render: () => (
            <ModuleGrid>
              <ModuleCard title={tr("Alerts & notifications")} icon={<Bell size={18} />} desc={tr("Get a Telegram push when a managed engine container goes DOWN (crash / OOM / stop) or recovers. The daemon watches over the docker socket; gated behind apply.")} wide>
                <NotificationSettings />
              </ModuleCard>
            </ModuleGrid>
          )
        },
        {
          id: "api",
          label: tr("API & Schema"),
          icon: <Settings size={15} />,
          render: () => (
            <ModuleGrid>
              <ModuleCard title={tr("Daemon & Access")} icon={<Settings size={18} />} desc={tr("Daemon endpoint, OpenAPI and the optional access token for remote/tunnel use.")}>
                <InfoRows rows={[
                  [tr("API Base"), apiBase],
                  ["OpenAPI", `${apiBase}/openapi.json`],
                  [tr("Mode"), tr("Read-only Product API")],
                  ["SNDR Core", environment?.sndr_core_version ?? "-"],
                  [tr("Frontend"), tr("Vite/React, served by daemon")]
                ]} />
                <ApiTokenField />
              </ModuleCard>
              <ModuleCard title={tr("API Tokens")} icon={<KeyRound size={18} />} desc={tr("Named, revocable Bearer tokens for programmatic / CI access (auth required). Plaintext shown once.")} wide>
                <ApiTokenManager enabled={!!authUser} />
              </ModuleCard>
              <ModuleCard title={tr("Endpoint Explorer")} icon={<Code2 size={18} />} desc={tr("Send a live GET to any read-only Product API endpoint and inspect the JSON.")} wide>
                <EndpointExplorer />
              </ModuleCard>
              <ModuleCard title={tr("MCP server (AI agents)")} icon={<SquareTerminal size={18} />} desc={tr("Expose this read-only control plane to external AI agents (Claude Desktop, Cursor) over the Model Context Protocol — the same catalog/doctor/preset/patch tools the Ops Copilot uses. Add this to the client's MCP config; it runs as a stdio subprocess (no port, read-only).")} wide>
                <CodeBlock lines={[
                  "{",
                  '  "mcpServers": {',
                  '    "sndr": {',
                  '      "command": "python3",',
                  '      "args": ["-m", "sndr.product_api.legacy.mcp_server"]',
                  "    }",
                  "  }",
                  "}"
                ]} />
              </ModuleCard>
            </ModuleGrid>
          )
        },
        {
          id: "audit",
          label: tr("Audit log"),
          icon: <FileText size={15} />,
          render: () => (
            <ModuleGrid>
              <ModuleCard title={tr("Audit log")} icon={<FileText size={18} />} desc={tr("Tamper-evident record of daemon events — auth, jobs, operations and system actions. Live.")} wide>
                <AuditLogPanel />
              </ModuleCard>
            </ModuleGrid>
          )
        },
        {
          id: "updates",
          label: tr("Updates"),
          icon: <DownloadCloud size={15} />,
          render: () => (
            <ModuleGrid>
              <ModuleCard title={tr("Updates")} icon={<DownloadCloud size={18} />} desc={tr("Pin-gated self-update for the GUI + sndr_core patcher. The vLLM pin only moves to a patcher-supported value; the server docker step stays manual. Apply is gated + confirmed.")} wide>
                <UpdatesPanel />
              </ModuleCard>
            </ModuleGrid>
          )
        },
        {
          id: "admin",
          label: tr("Admin"),
          icon: <KeyRound size={15} />,
          render: () => (
            <ModuleGrid>
              <ModuleCard title={tr("Admin Surface Matrix")} icon={<KeyRound size={18} />} desc={tr("Product API write/read surfaces and their status.")} wide>
                <AdminSurfaceMatrix featureRows={featureRows} patchDoctor={patchDoctor} />
              </ModuleCard>
              <ModuleCard title={tr("Engine & Dependencies")} icon={<Cpu size={18} />} desc={tr("Versions and runtime tools on the daemon host.")}>
                <EnvironmentPanel env={environment} />
              </ModuleCard>
              <ModuleCard title={tr("Feature Contracts")} icon={<ShieldCheck size={18} />} desc={tr("Capability inventory with live statuses.")}>
                <CapabilityTable rows={featureRows} />
              </ModuleCard>
              {authUser && (
                <ModuleCard title={tr("Account & Security")} icon={<ShieldCheck size={18} />} desc={tr("Your password and two-factor settings.")} wide>
                  <SecurityPanel user={authUser} onChanged={onAuthRefresh} />
                </ModuleCard>
              )}
              {authUser?.role === "admin" && (
                <ModuleCard title={tr("User Management")} icon={<KeyRound size={18} />} desc={tr("Create, list and remove accounts (admin).")} wide>
                  <UserAdminPanel currentUser={authUser} />
                </ModuleCard>
              )}
            </ModuleGrid>
          )
        },
        {
          id: "developer",
          label: tr("Developer"),
          icon: <SlidersHorizontal size={15} />,
          render: () => (
            <ModuleGrid>
              <ModuleCard title={tr("Config Draft and Diff")} icon={<SlidersHorizontal size={18} />} desc={`${tr("Local runtime draft for")} ${selectedPreset}.`} wide>
                <ConfigDraftEditor
                  selectedPreset={selectedPreset}
                  composed={composed}
                  runtimeTarget={runtimeTarget}
                  patchPolicy={patchPolicy}
                />
              </ModuleCard>
              <ModuleCard title={tr("CLI Mirror")} icon={<SquareTerminal size={18} />} desc={tr("Equivalent CLI for the current operator context.")}>
                <CodeBlock lines={cliLines} />
              </ModuleCard>
              <ModuleCard title={tr("Live Events")} icon={<Activity size={18} />} desc={tr("Daemon event feed (jobs, lifecycle, reports).")}>
                <EventLog events={events} />
              </ModuleCard>
            </ModuleGrid>
          )
        }
      ]}
    />
  );
}
