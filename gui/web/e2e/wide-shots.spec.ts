// SPDX-License-Identifier: Apache-2.0
// Visual-review screenshot sweep — renders the production bundle with the
// hermetic API mock at an ultra-wide (3440) dark-theme viewport and captures
// each navigable section. NOT a CI gate: a developer aid for the wide-screen
// layout work (see e2e/fixtures.ts for the mocked data shapes).
//   Run: npx playwright test --config playwright.shots.config.ts
import { test } from "@playwright/test";
import { mockApi } from "./fixtures";
import { RESPONSES, URL_TABLE } from "../tests/test/fixtures-data";

const OUT = process.env.SHOT_OUT || "/tmp/sndr-wide";

// The card-heavy sections under review plus a few known-good references.
const SECTIONS = [
  "Overview", "Models", "Presets", "Configs", "Planner",
  "Fleet", "Hosts", "Containers", "Virtualization", "Hardware",
  "Launch Plan", "Doctor", "Patches", "Benchmarks", "Routing", "Advanced",
];

async function bootDark(page: import("@playwright/test").Page) {
  await page.addInitScript(() => {
    window.localStorage.setItem(
      "sndr.gui.settings",
      JSON.stringify({ theme: "dark", density: "comfortable", accent: "teal" }),
    );
  });
}

test("wide dark-theme section sweep (3440)", async ({ page }) => {
  test.setTimeout(180_000);
  await mockApi(page);
  await bootDark(page);
  // Force the true ultra-wide viewport: the project's `devices["Desktop Chrome"]`
  // pins 1280x720, which would otherwise override the config viewport.
  await page.setViewportSize({ width: 3440, height: 1440 });
  await page.goto("/");
  await page.locator('.side-nav button:has-text("Overview")').first().waitFor();
  await page.waitForTimeout(500);

  for (const label of SECTIONS) {
    const button = page.locator(".side-nav button", { hasText: label }).first();
    if ((await button.count()) === 0) {
      console.log(`skip (no nav entry): ${label}`);
      continue;
    }
    await button.click();
    await page.waitForTimeout(700);
    const slug = label.toLowerCase().replace(/\s+/g, "-");
    await page.screenshot({ path: `${OUT}/${slug}.png`, fullPage: false });
    console.log(`shot: ${slug}`);
  }
});

// Overview with realistic catalog data (mirrors the live daemon numbers) so the
// enriched layout AND any data duplication across cards can be judged for real,
// not against the empty hermetic fixtures.
test("overview with rich data", async ({ page }) => {
  const base = RESPONSES as Record<string, any>;
  const overview = {
    capabilities: {
      ...base.overview.capabilities,
      platform: {
        ...base.overview.capabilities.platform,
        public_brand: "Genesis", package_name: "SNDR Core",
        sndr_core_version: "12.0.0.dev0", os_name: "Darwin", machine: "arm64",
        python_version: "3.11.9", engine_installed: true,
      },
    },
    catalog: {
      ...base.overview.catalog,
      models_count: 10, hardware_count: 4, profiles_count: 12,
      presets_count: 23, preset_cards_count: 23, unannotated_presets_count: 0,
      workload_counts: { free_chat: 5, code_gen: 4, tool_calls: 6, structured_json: 3, summarization: 5 },
      family_counts: { qwen3: 6, llama: 2, deepseek: 2 },
    },
  };
  const presets = {
    ...base.presets, total: 23, matched: 23,
    presets: Array.from({ length: 23 }, (_, i) => ({
      id: `preset-${i}`, model: "qwen3.6-35b", hardware: "a5000x2", profile: "fp8",
      has_card: true, card: { primary_metric: { value: i < 6 ? 120 : 0 }, title: `Preset ${i}` },
    })),
  };
  const patches = {
    ...base.patches, total: 252, matched: 252,
    patches: Array.from({ length: 252 }, (_, i) => ({ id: `patch-${i}`, default_on: i < 52 })),
  };
  const doctor = {
    ...base.doctor,
    findings: Array.from({ length: 16 }, (_, i) => ({
      id: `f-${i}`, severity: i < 2 ? "blocked" : i < 7 ? "warning" : "info",
      title: `Finding ${i}`, category: "runtime", detail: "",
    })),
  };
  const environment = {
    ...base.environment, sndr_core_version: "12.0.0.dev0", engine_name: "vLLM",
    engine_version: "0.20.2", engine_installed: true, os_name: "Darwin", machine: "arm64", python_version: "3.11.9",
  };
  const hosts = { hosts: [{ id: "h1", host: "127.0.0.1", label: "This host", port: 8765, engine_port: 8000 }] };
  const rich: Record<string, any> = { ...base, overview, presets, patches, doctor, environment, hosts };

  await page.route("**/api/**", (route) => {
    const pathname = new URL(route.request().url()).pathname;
    const hit = URL_TABLE.find(([suffix]) => pathname.includes(suffix));
    const body = hit ? rich[hit[1]] : {};
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body ?? {}) });
  });
  await bootDark(page);
  for (const width of [1920, 3440]) {
    await page.setViewportSize({ width, height: width >= 2880 ? 1440 : 1000 });
    await page.goto("/");
    await page.locator('.side-nav button:has-text("Overview")').first().click();
    await page.waitForTimeout(700);
    await page.screenshot({ path: `${OUT}/rich-overview-${width}.png`, fullPage: false });
    console.log(`shot: rich-overview-${width}`);
  }
});

// Confirm the Containers > Logs tab is a bounded scroll box (was stretching the
// page across 3-4 screens). Mocks one container with 400 log lines.
test("containers logs scroll box", async ({ page }) => {
  const logLines = Array.from({ length: 400 }, (_, i) =>
    `2026-06-08T18:${String(i % 60).padStart(2, "0")}:00 [INFO] worker ${i} processed request id=${1000 + i} latency=${(i % 50) + 5}ms tokens=${i * 3}`,
  ).join("\n");
  const containers = {
    containers: [{ name: "vllm-prod", id: "abc123def456", image: "vllm/vllm-openai:nightly", state: "running", status: "Up 2 hours", ports: "8000/tcp", created: "2h" }],
    source: "local",
  };
  await page.route("**/api/**", (route) => {
    const p = new URL(route.request().url()).pathname;
    let body: any = {};
    if (p.endsWith("/logs")) body = { container: "vllm-prod", logs: logLines };
    else if (p.endsWith("/update-plan")) body = { update_available: false, mode: "manual" };
    else if (p.endsWith("/sndr-state")) body = { ok: false };
    else if (p.endsWith("/containers/stats")) body = { stats: {} };
    else if (p.endsWith("/stats")) body = { container: "vllm-prod", stats: {} };
    else if (p.endsWith("/source")) body = { container: "vllm-prod", preset_id: null, drift: [], drift_count: 0, live_patches: [], live_patch_count: 0 };
    else if (p.endsWith("/engine")) body = {};
    else if (p.endsWith("/system/df")) body = { types: [], total_size: 0 };
    else if (p.endsWith("/api/v1/containers")) body = containers;
    else if (p.includes("/containers/vllm-prod")) body = { Name: "vllm-prod", Config: { Image: "vllm/vllm-openai:nightly" }, State: { Status: "running" }, NetworkSettings: { Ports: {} }, Mounts: [] };
    else if (p.includes("/auth/status")) body = (RESPONSES as Record<string, any>).authStatus;
    else { const hit = URL_TABLE.find(([s]) => p.includes(s)); body = hit ? (RESPONSES as Record<string, any>)[hit[1]] : {}; }
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body ?? {}) });
  });
  await bootDark(page);
  await page.addInitScript(() => window.localStorage.setItem("sndr.containers.view", "table"));
  await page.setViewportSize({ width: 1920, height: 1000 });
  await page.goto("/");
  await page.locator('.side-nav button:has-text("Containers")').first().click();
  await page.waitForTimeout(700);
  await page.locator('.crow-name [role="button"]').first().click();
  await page.waitForTimeout(500);
  await page.locator('.cpage-rail button:has-text("Logs")').first().click();
  await page.waitForTimeout(600);
  await page.screenshot({ path: `${OUT}/containers-logs.png`, fullPage: false });
  // Assert the log viewer is bounded (not taller than the viewport).
  const box = await page.locator(".logs-tab .container-logs").first().boundingBox();
  console.log(`logs box height: ${box?.height} (viewport 1000)`);
});

// Capture the three Setup tabs (Guided / Install onto host / Deploy) with a
// realistic install mock, so the display + the install wizard can be reviewed.
test("setup tabs", async ({ page }) => {
  const base = RESPONSES as Record<string, any>;
  const installTargets = {
    apply_enabled: false,
    hosts: [
      { id: "gpu-build-01", label: "gpu-build-01", host: "192.168.1.10", port: 8765, engine_port: 8000, gpu_arch: "A5000", gpus: 2 },
      { id: "gpu-build-02", label: "gpu-build-02", host: "192.168.1.11", port: 8765, engine_port: 8000, gpu_arch: "A6000", gpus: 1 },
    ],
    targets: [
      { id: "compose", label: "Docker Compose", filename: "docker-compose.yml", kind: "compose", needs: "docker", summary: "Run the engine as a compose service" },
      { id: "systemd", label: "systemd unit", filename: "sndr-engine.service", kind: "systemd", needs: "systemd", summary: "Native systemd service on the host" },
      { id: "quadlet", label: "Podman Quadlet", filename: "sndr.container", kind: "quadlet", needs: "podman", summary: "Rootless Podman via Quadlet" },
      { id: "kubernetes", label: "Kubernetes", filename: "sndr.yaml", kind: "k8s", needs: "kubectl", summary: "Deployment + Service manifest" },
      { id: "proxmox_vm", label: "Proxmox VM", filename: "provision.sh", kind: "proxmox", needs: "proxmox", summary: "Provision a GPU-passthrough VM" },
    ],
  };
  const plan = {
    host: { label: "gpu-build-01", host: "192.168.1.10" }, preset_id: "preset-0", target: "compose",
    target_label: "Docker Compose", artifact: { kind: "compose", filename: "docker-compose.yml", content: "services:\n  vllm:\n    image: vllm/vllm-openai:nightly\n    runtime: nvidia\n    ports:\n      - \"8000:8000\"\n    environment:\n      - GENESIS_ENABLE_PN90=1\n    command: --model qwen3.6-35b --tensor-parallel-size 2\n" },
    image_override: null, with_daemon: false,
    steps: [
      { order: 1, kind: "upload", title: "Upload docker-compose.yml to /opt/sndr", danger: false, cmd: "scp docker-compose.yml gpu-build-01:/opt/sndr/" },
      { order: 2, kind: "remote-exec", title: "Pull the engine image", danger: false, cmd: "docker compose pull" },
      { order: 3, kind: "remote-exec", title: "Start the engine", danger: false, cmd: "docker compose up -d" },
    ],
    danger_count: 0, provisions_infra: false, dry_run: true, can_apply: false, notes: "Review the plan, then run over SSH.",
  };
  const presets = { ...base.presets, total: 6, matched: 6, presets: Array.from({ length: 6 }, (_, i) => ({ id: `preset-${i}`, model: "qwen3.6-35b", hardware: "a5000x2", profile: "fp8", has_card: true, card: { primary_metric: { value: 120 }, title: `Preset ${i}` } })) };
  const environment = { ...base.environment, engine_installed: true, engine_version: "0.20.2", tools: [{ name: "docker", present: true }, { name: "nvidia-smi", present: true }] };
  const doctor = { ...base.doctor, summary: { ok: 12, warning: 2, blocked: 0 }, findings: Array.from({ length: 14 }, (_, i) => ({ id: `f-${i}`, severity: i < 2 ? "warning" : "info", title: `Check ${i}` })) };
  await page.route("**/api/**", (route) => {
    const p = new URL(route.request().url()).pathname;
    let body: any = {};
    if (p.includes("/install/targets")) body = installTargets;
    else if (p.includes("/install/plan")) body = plan;
    else if (p.includes("/presets/recommend")) body = base.recommendPresets;
    else if (p.endsWith("/presets")) body = presets;
    else if (p.includes("/environment")) body = environment;
    else if (p.includes("/doctor")) body = doctor;
    else if (p.includes("/auth/status")) body = base.authStatus;
    else { const hit = URL_TABLE.find(([s]) => p.includes(s)); body = hit ? base[hit[1]] : {}; }
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body ?? {}) });
  });
  await bootDark(page);
  await page.setViewportSize({ width: 1920, height: 1100 });
  await page.goto("/");
  await page.locator('.side-nav button:has-text("Setup")').first().click();
  await page.waitForTimeout(600);
  await page.screenshot({ path: `${OUT}/setup-guided.png`, fullPage: false });
  await page.locator('.section-tabs button:has-text("Install onto host")').first().click();
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${OUT}/setup-install.png`, fullPage: false });
  await page.locator('.installer button:has-text("Build install plan")').first().click();
  await page.waitForTimeout(600);
  await page.screenshot({ path: `${OUT}/setup-install-plan.png`, fullPage: true });
  await page.locator('.section-tabs button:has-text("Deploy a model")').first().click();
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${OUT}/setup-deploy.png`, fullPage: false });
  console.log("shot: setup-guided / setup-install / setup-install-plan / setup-deploy");
});

// Capture the container detail Config tab (visual live-settings editor) + Overview.
test("containers config + overview tabs", async ({ page }) => {
  const inspect = {
    Id: "abc123def456789", Name: "/vllm-prod", Image: "sha256:deadbeef",
    Created: "2026-06-08T10:00:00Z",
    Config: {
      Image: "vllm/vllm-openai:nightly", Entrypoint: ["python", "-m", "vllm.entrypoints.openai.api_server"],
      Cmd: ["--model", "qwen3.6-35b", "--tensor-parallel-size", "2"], WorkingDir: "/app",
      Env: ["PATH=/usr/bin", "GENESIS_ENABLE_PN90=1", "SNDR_PIN=nightly-abc", "HF_TOKEN=secrethidden", "CUDA_VISIBLE_DEVICES=0,1", "VLLM_USE_V1=1"],
      Labels: { "sndr.preset": "prod-35b-multiconc", "sndr.role": "engine" }, ExposedPorts: { "8000/tcp": {} },
    },
    State: { Running: true, Status: "running", StartedAt: "2026-06-08T10:01:00Z", Health: { Status: "healthy" }, Pid: 4242 },
    HostConfig: { RestartPolicy: { Name: "unless-stopped" }, NanoCpus: 4_000_000_000, Memory: 17_179_869_184, NetworkMode: "bridge", Privileged: false },
    NetworkSettings: { Ports: { "8000/tcp": [{ HostPort: "8000", HostIp: "0.0.0.0" }] }, Networks: { bridge: { IPAddress: "172.17.0.2" } }, IPAddress: "172.17.0.2" },
    Mounts: [{ Source: "/data/models", Destination: "/models", RW: false, Type: "bind" }],
    RestartCount: 0,
  };
  const containers = { containers: [{ name: "vllm-prod", id: "abc123def456", image: "vllm/vllm-openai:nightly", state: "running", status: "Up 2 hours", ports: "8000/tcp", created: "2h" }], source: "local" };
  await page.route("**/api/**", (route) => {
    const p = new URL(route.request().url()).pathname;
    let body: any = {};
    if (p.endsWith("/update-plan")) body = { update_available: false, mode: "manual" };
    else if (p.endsWith("/sndr-state")) body = { ok: true, vllm_version: "0.20.2", sndr_version: "12.0.0", patches: 18, configs: 5 };
    else if (p.endsWith("/containers/stats")) body = { stats: { "vllm-prod": { cpu_pct: 34, mem_pct: 58, mem_usage: 9_000_000_000, mem_limit: 17_179_869_184, net_rx: 1234, net_tx: 5678, blk_read: 1000, blk_write: 2000, pids: 24 } } };
    else if (p.endsWith("/stats")) body = { container: "vllm-prod", stats: { cpu_pct: 34, mem_pct: 58, mem_usage: 9_000_000_000, mem_limit: 17_179_869_184, net_rx: 1234, net_tx: 5678, blk_read: 1000, blk_write: 2000, pids: 24 } };
    else if (p.endsWith("/source")) body = { container: "vllm-prod", preset_id: "prod-35b-multiconc", preset_title: "Prod 35B", linked_by: "label", drift: [], drift_count: 0, live_patches: [], live_patch_count: 0, served_model: "qwen3.6-35b" };
    else if (p.endsWith("/engine")) body = { reachable: true, port: 8000 };
    else if (p.endsWith("/networks")) body = { networks: [{ name: "bridge", driver: "bridge", scope: "local" }, { name: "sndr-net", driver: "bridge", scope: "local" }] };
    else if (p.endsWith("/system/df")) body = { types: [], total_size: 0 };
    else if (p.endsWith("/api/v1/containers")) body = containers;
    else if (p.includes("/containers/vllm-prod")) body = inspect;
    else if (p.includes("/auth/status")) body = (RESPONSES as Record<string, any>).authStatus;
    else { const hit = URL_TABLE.find(([s]) => p.includes(s)); body = hit ? (RESPONSES as Record<string, any>)[hit[1]] : {}; }
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body ?? {}) });
  });
  await bootDark(page);
  await page.addInitScript(() => window.localStorage.setItem("sndr.containers.view", "table"));
  await page.setViewportSize({ width: 1920, height: 1100 });
  await page.goto("/");
  await page.locator('.side-nav button:has-text("Containers")').first().click();
  await page.waitForTimeout(700);
  await page.locator('.crow-name [role="button"]').first().click();
  await page.waitForTimeout(600);
  await page.screenshot({ path: `${OUT}/container-overview.png`, fullPage: false });
  await page.locator('.cpage-rail button:has-text("Config")').first().click();
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${OUT}/container-config.png`, fullPage: true });
  // The Stats tab must be gone (merged into Overview).
  const statsCount = await page.locator('.cpage-rail button:has-text("Stats")').count();
  console.log(`stats tab count (expect 0): ${statsCount}`);
});

// High-fidelity element clip of the Overview hero tiles, for inspecting tile
// internals (value alignment, click affordance, text clamp) up close.
test("overview hero tile clip", async ({ page }) => {
  await mockApi(page);
  await bootDark(page);
  for (const width of [1920, 3440]) {
    await page.setViewportSize({ width, height: width >= 2880 ? 1440 : 900 });
    await page.goto("/");
    // The app boots on launch-plan; navigate to Overview so `.ov-hero` exists.
    await page.locator('.side-nav button:has-text("Overview")').first().click();
    await page.waitForTimeout(500);
    const hero = page.locator(".ov-hero").first();
    await hero.screenshot({ path: `${OUT}/hero-${width}.png` });
    console.log(`shot: hero-${width}`);
  }
});

// Verify the Overview sub-tabs (Summary / Environment / Coverage) all route to
// real content and capture each for review.
test("overview sub-tabs", async ({ page }) => {
  await mockApi(page);
  await bootDark(page);
  await page.setViewportSize({ width: 1920, height: 1080 });
  await page.goto("/");
  await page.locator('.side-nav button:has-text("Overview")').first().click();
  await page.waitForTimeout(400);
  for (const tab of ["Summary", "Environment", "Coverage"]) {
    await page.locator('.section-tabs button', { hasText: tab }).first().click();
    await page.waitForTimeout(300);
    await page.screenshot({ path: `${OUT}/ov-tab-${tab.toLowerCase()}.png`, fullPage: false });
    console.log(`shot: ov-tab-${tab.toLowerCase()}`);
  }
});

// Focused responsive check: capture the sections under active work at three
// real widths so the adaptive (viewport-tier) layout can be reviewed side by side.
test("responsive widths for sections under work", async ({ page }) => {
  test.setTimeout(180_000);
  await mockApi(page);
  await bootDark(page);
  const WIDTHS = [1280, 1920, 3440];
  const FOCUS = ["Overview", "Containers"];
  for (const width of WIDTHS) {
    await page.setViewportSize({ width, height: width >= 2880 ? 1440 : 900 });
    await page.goto("/");
    await page.locator('.side-nav button:has-text("Overview")').first().waitFor();
    await page.waitForTimeout(400);
    for (const label of FOCUS) {
      const button = page.locator(".side-nav button", { hasText: label }).first();
      if ((await button.count()) === 0) continue;
      await button.click();
      await page.waitForTimeout(600);
      const slug = label.toLowerCase().replace(/\s+/g, "-");
      await page.screenshot({ path: `${OUT}/${slug}-${width}.png`, fullPage: false });
      console.log(`shot: ${slug}-${width}`);
    }
  }
});
