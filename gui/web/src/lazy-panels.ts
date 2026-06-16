// SPDX-License-Identifier: Apache-2.0
// Code-split section panels. Each heavy section renders only when its nav entry
// is active, so it is lazy()-imported here and excluded from the initial bundle;
// a single Suspense boundary (around SectionWorkspace) covers them all. Shared
// by the app shell (App.tsx, which renders ContainersPanel directly) and the
// section renderer (sections/section-workspace.tsx).
import { lazy } from "react";

export const ChatConsole = lazy(() => import("./Engine").then((m) => ({ default: m.ChatConsole })));
export const EngineBenchPanel = lazy(() => import("./Engine").then((m) => ({ default: m.EngineBenchPanel })));
export const EngineMetricsPanel = lazy(() => import("./Engine").then((m) => ({ default: m.EngineMetricsPanel })));
export const EnginePlayground = lazy(() => import("./Engine").then((m) => ({ default: m.EnginePlayground })));
export const EngineStatusCard = lazy(() => import("./Engine").then((m) => ({ default: m.EngineStatusCard })));

export const ConfigsSection = lazy(() => import("./sections/configs-workbench").then((m) => ({ default: m.ConfigsSection })));
export const HostsSection = lazy(() => import("./sections/hosts-section").then((m) => ({ default: m.HostsSection })));
export const DeploymentConsole = lazy(() => import("./sections/deployment").then((m) => ({ default: m.DeploymentConsole })));
export const ServiceLifecyclePlanner = lazy(() => import("./sections/services").then((m) => ({ default: m.ServiceLifecyclePlanner })));
export const PatchInventoryControl = lazy(() => import("./sections/patch-inventory").then((m) => ({ default: m.PatchInventoryControl })));
export const ApiTokenManager = lazy(() => import("./sections/settings-panels").then((m) => ({ default: m.ApiTokenManager })));
export const NotificationSettings = lazy(() => import("./sections/settings-panels").then((m) => ({ default: m.NotificationSettings })));
export const AppearanceSettings = lazy(() => import("./sections/settings-panels").then((m) => ({ default: m.AppearanceSettings })));
export const ApiTokenField = lazy(() => import("./sections/settings-panels").then((m) => ({ default: m.ApiTokenField })));
export const KvCalcPanel = lazy(() => import("./Planner").then((m) => ({ default: m.KvCalcPanel })));
export const BaselinePanel = lazy(() => import("./Planner").then((m) => ({ default: m.BaselinePanel })));
export const InstallWizard = lazy(() => import("./Installer").then((m) => ({ default: m.InstallWizard })));
export const CopilotPanel = lazy(() => import("./Copilot").then((m) => ({ default: m.CopilotPanel })));
export const ContainersPanel = lazy(() => import("./Containers").then((m) => ({ default: m.ContainersPanel })));
export const VirtualizationPanel = lazy(() => import("./sections/virtualization").then((m) => ({ default: m.VirtualizationPanel })));
export const HardwarePanel = lazy(() => import("./Hardware").then((m) => ({ default: m.HardwarePanel })));
export const RoutingPanel = lazy(() => import("./Routing").then((m) => ({ default: m.RoutingPanel })));
export const FlagsPanel = lazy(() => import("./Flags").then((m) => ({ default: m.FlagsPanel })));
export const LicensePanel = lazy(() => import("./License").then((m) => ({ default: m.LicensePanel })));
