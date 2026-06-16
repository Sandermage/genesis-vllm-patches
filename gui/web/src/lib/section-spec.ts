// SPDX-License-Identifier: Apache-2.0
// Per-section header metadata (kicker / title / description) rendered by the
// workspace header. Pure lookup keyed by SectionId, kept out of App.tsx so the
// shell only consumes it. Strings flow through tr() for i18n coverage.
import { tr } from "../i18n";
import type { SectionId } from "../nav";

export function sectionSpec(sectionId: SectionId) {
  const specs: Record<
    SectionId,
    { kicker: string; title: string; description: string }
  > = {
    overview: {
      kicker: tr("System map"),
      title: tr("Overview"),
      description: tr("One screen summary of Product API health, catalog coverage, runtime targets and workload readiness."),
    },
    setup: {
      kicker: tr("First-run workflow"),
      title: tr("Setup"),
      description: tr("Local server and remote desktop setup path with explicit daemon, tunnel and gate stages."),
    },
    fleet: {
      kicker: tr("Multi-server overview"),
      title: tr("Fleet"),
      description: tr("Every registered GPU/engine host at a glance — one concurrent SSH sweep shows status, running model, vLLM version, GPUs and live patch count per server."),
    },
    hosts: {
      kicker: tr("Runtime inventory"),
      title: tr("Hosts"),
      description: tr("Local and remote host inventory, transport state, runtime tools and SSH tunnel commands."),
    },
    models: {
      kicker: tr("Model catalog"),
      title: tr("Models"),
      description: tr("Model families, hardware envelopes and composed runtime details from the V2 registry."),
    },
    configs: {
      kicker: tr("V2 config editor"),
      title: tr("Configs"),
      description: tr("Graphical editor for V2 model, hardware, profile and preset composition with safe draft preview."),
    },
    presets: {
      kicker: tr("Preset catalog"),
      title: tr("Presets"),
      description: tr("Full preset table with cards, workload policy, evidence visibility and selected explain payload."),
    },
    planner: {
      kicker: tr("Capacity & regression"),
      title: tr("Planner"),
      description: tr("KV-cache / VRAM fit calculator (GQA, MoE and tensor-parallel aware, calibratable) and quality-baseline regression diff."),
    },
    copilot: {
      kicker: tr("Read-only assistant"),
      title: tr("Ops Copilot"),
      description: tr("Tool-calling assistant over the read-only Product API — answers from real catalog/doctor/preset/patch/capacity data and proposes changes you review & apply."),
    },
    "launch-plan": {
      kicker: tr("Operator workbench"),
      title: tr("Launch Plan"),
      description: tr("Recommendation builder, plan composer, readiness gates, artifacts and CLI mirror."),
    },
    services: {
      kicker: tr("Lifecycle"),
      title: tr("Services"),
      description: tr("Service lifecycle, rendered launch artifacts, status, logs and safe write API boundary."),
    },
    containers: {
      kicker: tr("Docker control"),
      title: tr("Containers"),
      description: tr("Manage the vLLM/engine containers on a server — live CPU/memory, logs, start/stop/restart, and gated exec — over the local docker socket or a registered host via SSH."),
    },
    kubernetes: {
      kicker: tr("Cluster"),
      title: tr("Kubernetes"),
      description: tr("Read-only Kubernetes view — cluster status and nodes with GPU capacity/allocatable/requested, conditions, taints and labels. Honours your kubeconfig + RBAC."),
    },
    virtualization: {
      kicker: tr("Compute"),
      title: tr("Virtualization"),
      description: tr("Proxmox VE hosts & guests, KubeVirt VMs and Kubernetes nodes in one pane — each linked back to the SNDR preset it runs."),
    },
    hardware: {
      kicker: tr("GPU telemetry"),
      title: tr("GPU & Hardware"),
      description: tr("Live per-GPU utilisation, VRAM, temperature, power, clocks, fan, PCIe, pstate and ECC over nvidia-smi — for the daemon host or a registered host via SSH."),
    },
    routing: {
      kicker: tr("Spec-decode routing"),
      title: tr("Workload routing"),
      description: tr("Per bench-validated profile: which workloads are allowed/denied and their measured TPS delta — plus a classifier that predicts how a request's signals resolve to a profile. One source of truth, shared with the gateway."),
    },
    flags: {
      kicker: tr("Patch flags"),
      title: tr("Env-flag matrix"),
      description: tr("Every GENESIS_ENABLE_* flag with its default, searchable and filterable — overlay a running engine's live ON/OFF state and flag drift."),
    },
    doctor: {
      kicker: tr("Diagnostics"),
      title: tr("Doctor"),
      description: tr("Readiness gates, blockers, warnings and release-proof preflight diagnostics."),
    },
    patches: {
      kicker: tr("Patch control"),
      title: tr("Patches"),
      description: tr("Patch simulation, policy matrix, enabled patch count and safe/minimal/compact policy preview."),
    },
    benchmarks: {
      kicker: tr("Performance"),
      title: tr("Benchmarks"),
      description: tr("Benchmark baselines, expected TPS/TTFT context, run plan and evidence orchestration state."),
    },
    evidence: {
      kicker: tr("Proof bundle"),
      title: tr("Evidence"),
      description: tr("Evidence references, visibility, benchmark baseline status and future release report bundles."),
    },
    clients: {
      kicker: tr("Integrations"),
      title: tr("Clients"),
      description: tr("OpenAI-compatible endpoints, health/metrics URLs, client snippets and GUI/CLI integration modes."),
    },
    chat: {
      kicker: tr("Local model chat"),
      title: tr("Chat"),
      description: tr("Multi-turn streaming chat with any running local vLLM model — pick the engine host/port and model, tune the system prompt and sampling."),
    },
    reports: {
      kicker: tr("Operator reports"),
      title: tr("Reports"),
      description: tr("Launch, benchmark, patch and release-proof report types planned for GUI export."),
    },
    operations: {
      kicker: tr("Project workbench"),
      title: tr("Operations"),
      description: tr("Run sndr_core's canonical maintenance, audit and proof workflows as live-monitored jobs — the CLI surface, integrated."),
    },
    advanced: {
      kicker: tr("Developer surface"),
      title: tr("Advanced"),
      description: tr("API base, OpenAPI/schema, feature contracts, CLI mirror and future desktop settings."),
    }
  };
  return specs[sectionId];
}
