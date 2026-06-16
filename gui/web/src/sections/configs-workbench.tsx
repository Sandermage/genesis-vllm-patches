// SPDX-License-Identifier: Apache-2.0
// Configs workbench — the Configs tab: element editor, V2 config workbench
// (compose/preview/plan/apply), composition chain, resolved config, and the
// model/hardware/profile selectors + inspectors.
import { Fragment, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  AlertCircle, AlertTriangle, Box, CheckCircle2, ChevronRight, Code2, Cpu, Database, FileText,
  Gauge, GitBranch, HardDrive, Layers, Layers3, ListChecks, Maximize2, PackageCheck, RefreshCw,
  Rocket, Search, Server, SlidersHorizontal, SquareTerminal, X
} from "lucide-react";
import {
  api, type V2ConfigCatalog, type V2ConfigItem, type V2ConfigApplyResult, type V2ConfigPlan,
  type V2ConfigPreview, type V2LayerApplyResult, type V2LayerDefinition, type UserPresetList
} from "../api";
import { tr } from "../i18n";
import { asNumber, asText } from "../lib/coerce";
import { formatTokens, formatVram } from "../lib/format";
import { type RuntimeConfigDraft, buildRuntimeDraft, runtimeDraftDiff } from "../lib/runtime-draft";
import { getIn, setIn, objToYaml } from "../lib/config-utils";
import { CodeEditor } from "../components/code-editor";
import { useDialogFocus, closeOnBackdrop } from "../dialog";
import { SkeletonLines } from "../Skeleton";
import { CompactList, InfoRows, RailCheck, StatusBadge } from "../components/primitives";
import { ModuleCard, ModuleGrid } from "../components/layout";
import { CodeBlock, CopyButton } from "../components/code-block";
import { TabbedSection } from "../components/tabbed-section";
import { EmptyState } from "../components/empty-state";
import { CatalogCard, ModelFitCard } from "./catalog-cards";
import { ConfigApplyPanel, ConfigComparePanel, ConfigPlanPanel } from "./config";
import { type ElementKind, ELEMENT_FIELDS_FOR, discoverExtraFields, groupFields, ElementField } from "./element-fields";
import { ParamFields } from "./config-draft-editor";
import { ProfileDeltaPanel, UserPresetsPanel } from "./presets";
import { itemBadges } from "./models-workbench";

function ConfigElementEditor({ catalog }: { catalog: V2ConfigCatalog | null }) {
  const itemsByKind: Record<ElementKind, V2ConfigItem[]> = {
    model: catalog?.models ?? [],
    hardware: catalog?.hardware ?? [],
    profile: catalog?.profiles ?? [],
    preset: catalog?.presets ?? []
  };
  const [kind, setKind] = useState<ElementKind>("model");
  const [itemId, setItemId] = useState("");
  const [filter, setFilter] = useState("");
  const [edited, setEdited] = useState<Record<string, any> | null>(null);
  const [source, setSource] = useState("");
  const [state, setState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [applyResult, setApplyResult] = useState<V2LayerApplyResult | null>(null);
  const [applying, setApplying] = useState(false);

  const items = itemsByKind[kind];
  const activeId = itemId || items[0]?.id || "";
  const visible = items.filter((item) => {
    const needle = filter.trim().toLowerCase();
    return !needle || item.id.toLowerCase().includes(needle) || (item.title ?? "").toLowerCase().includes(needle);
  });

  useEffect(() => {
    if (!activeId) {
      setEdited(null);
      return;
    }
    const controller = new AbortController();
    setState("loading");
    setError(null);
    setApplyResult(null);
    api.v2Layer(kind, activeId, controller.signal)
      .then((layer) => {
        if (controller.signal.aborted) return;
        setEdited(layer.definition as Record<string, any>);
        setSource(layer.source);
        setState("ready");
      })
      .catch((err) => {
        if (controller.signal.aborted) return;
        setEdited(null);
        setState("error");
        setError(err instanceof Error ? err.message : String(err));
      });
    return () => controller.abort();
  }, [kind, activeId]);

  // Adaptive: curated fields first, then any extra scalar leaves the loaded
  // definition actually contains (so new/unknown fields still render + edit).
  const fields = useMemo(() => {
    const curated = ELEMENT_FIELDS_FOR(kind);
    if (!edited) return curated;
    const known = new Set(curated.map((spec) => spec.path));
    return [...curated, ...discoverExtraFields(edited, known)];
  }, [kind, edited]);
  const yaml = edited ? objToYaml(edited) : ["# Select an element to edit"];

  async function runSave() {
    if (!edited) return;
    setApplying(true);
    try {
      const result = await api.v2LayerApply({ kind, layer_id: activeId, yaml_text: yaml.join("\n") + "\n" });
      setApplyResult(result);
    } catch (err) {
      setApplyResult({
        kind, layer_id: activeId, target_path: "", action: "create", written: false,
        bytes_written: 0, status: "blocked", message: err instanceof Error ? err.message : String(err), blocked_reasons: []
      });
    } finally {
      setApplying(false);
    }
  }

  return (
    <section className="element-editor">
      <aside className="element-rail">
        <div className="config-panel-title">
          <Layers3 size={16} />
          <strong>{tr("Element")}</strong>
        </div>
        <p className="config-panel-desc">{tr("Choose a config layer and item to inspect and edit.")}</p>
        <div className="element-kind-tiles">
          {([
            { id: "model", icon: <Box size={15} /> },
            { id: "hardware", icon: <Cpu size={15} /> },
            { id: "profile", icon: <SlidersHorizontal size={15} /> },
            { id: "preset", icon: <Database size={15} /> }
          ] as Array<{ id: ElementKind; icon: ReactNode }>).map((option) => (
            <button
              key={option.id}
              type="button"
              className={`element-kind-tile${kind === option.id ? " active" : ""}`}
              onClick={() => { setKind(option.id); setItemId(""); }}
            >
              {option.icon}
              <span>{option.id}</span>
              <small>{itemsByKind[option.id].length}</small>
            </button>
          ))}
        </div>
        <label className="search-box">
          <Search size={15} />
          <input aria-label={`${tr("Search")} ${kind}`} value={filter} onChange={(event) => setFilter(event.target.value)} placeholder={`${tr("Search")} ${kind}`} />
        </label>
        <div className="catalog-list">
          {visible.map((item) => (
            <CatalogCard
              key={item.id}
              icon={
                item.kind === "hardware" ? <Cpu size={16} />
                : item.kind === "profile" ? <SlidersHorizontal size={16} />
                : item.kind === "preset" ? <Database size={16} />
                : <Box size={16} />
              }
              id={item.id}
              title={item.title && item.title !== item.id ? item.title : undefined}
              badges={itemBadges(item)}
              active={item.id === activeId}
              onClick={() => setItemId(item.id)}
            />
          ))}
          {visible.length === 0 && (
            <EmptyState
              icon={<Layers size={22} />}
              title={`${tr("No")} ${kind} ${tr("matches")}`}
              message={filter ? <>{tr("Nothing in")} {kind} {tr("matches")} “{filter}”.</> : `${tr("No")} ${kind} ${tr("elements in the catalog.")}`}
              action={filter ? { label: tr("Clear search"), icon: <X size={14} />, onClick: () => setFilter("") } : undefined}
            />
          )}
        </div>
      </aside>

      <section className="element-form-panel">
        <div className="config-panel-title">
          <SlidersHorizontal size={16} />
          <strong>{activeId || tr("No selection")}</strong>
          <StatusBadge status={state === "loading" ? "partial" : state === "error" ? "missing" : "available"} />
        </div>
        <p className="element-source">{source}</p>
        {error && <div className="config-plan-error"><AlertCircle size={15} /><span>{error}</span></div>}
        {edited ? (
          <div className="element-groups">
            {groupFields(fields).map(([group, groupFieldsList]) => (
              <div className="element-group" key={group}>
                {group && <div className="element-group-head">{group}</div>}
                <div className="element-fields">
                  {groupFieldsList.map((spec) => (
                    <ElementField
                      key={spec.path}
                      spec={spec}
                      value={getIn(edited, spec.path)}
                      onChange={(value) => setEdited((current) => (current ? setIn(current, spec.path, value) : current))}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <SkeletonLines count={6} />
        )}
        {applyResult && (
          <div className={`element-apply ${applyResult.status}`}>
            <span className="finding-icon">
              {applyResult.status === "applied" ? <CheckCircle2 size={14} /> : <AlertCircle size={14} />}
            </span>
            <div>
              <strong>{applyResult.status === "applied" ? tr("Saved to user dir") : applyResult.status}</strong>
              <small>{applyResult.target_path || applyResult.message}</small>
              {applyResult.blocked_reasons.length > 0 && <small>{applyResult.blocked_reasons.join("; ")}</small>}
            </div>
          </div>
        )}
        <div className="config-actions">
          <span className="config-actions-note">{tr("Edits update the YAML draft on the right")}</span>
          <button className="ghost-button" onClick={() => { setItemId(activeId); setState("idle"); }}>
            <RefreshCw size={14} /> {tr("Reload")}
          </button>
          <button className="primary-action" onClick={() => void runSave()} disabled={!edited || applying}>
            <PackageCheck size={14} /> {applying ? tr("Saving…") : tr("Save to user dir")}
          </button>
        </div>
      </section>

      <section className="element-yaml-panel">
        <div className="config-panel-title">
          <Code2 size={16} />
          <strong>{kind}.yaml</strong>
          <span>{yaml.length} {tr("lines")}</span>
        </div>
        <CodeBlock lines={yaml} />
        <p className="element-note">
          {tr("Full definition with your edits. Copy to export, or use the Compose tab → Apply Plan to persist an operator-local preset. Direct model/hardware/profile writes need a layer-apply API.")}
        </p>
      </section>
    </section>
  );
}

export function ConfigsSection(props: {
  catalog: V2ConfigCatalog | null;
  preview: V2ConfigPreview | null;
  selectedPreset: string;
  userPresets: UserPresetList | null;
  onPreview: (preview: V2ConfigPreview) => void;
  onUserPresetsRefresh: () => void;
}) {
  const [tab, setTab] = useState<"compose" | "edit">("compose");
  return (
    <div className="configs-section">
      <div className="configs-top-tabs">
        <button className={tab === "compose" ? "active" : ""} onClick={() => setTab("compose")}>
          <Layers3 size={15} /> {tr("Compose")}
        </button>
        <button className={tab === "edit" ? "active" : ""} onClick={() => setTab("edit")}>
          <SlidersHorizontal size={15} /> {tr("Edit element")}
        </button>
      </div>
      {tab === "compose" ? <V2ConfigWorkbench {...props} /> : <ConfigElementEditor catalog={props.catalog} />}
    </div>
  );
}

// Visual composition chain — a preset = model + hardware + profile, resolved to
// the runtime config.
function CompositionChain({ model, hardware, profile, composed }: {
  model: V2ConfigItem | null; hardware: V2ConfigItem | null; profile: V2ConfigItem | null; composed: Record<string, any>;
}) {
  const layers = [
    { kind: tr("Model"), icon: <Box size={13} />, item: model, optional: false,
      fact: model ? [model.fields?.quantization || model.fields?.dtype, model.fields?.attention_arch].filter(Boolean).join(" · ") : "" },
    { kind: tr("Hardware"), icon: <Server size={13} />, item: hardware, optional: false,
      fact: hardware ? `${hardware.fields?.n_gpus ?? "?"}× GPU${hardware.fields?.min_vram_per_gpu_mib ? ` · ${Math.round(hardware.fields.min_vram_per_gpu_mib / 1024)}GB` : ""}` : "" },
    { kind: tr("Profile"), icon: <SlidersHorizontal size={13} />, item: profile, optional: true,
      fact: profile ? String(profile.fields?.role || profile.status || tr("override")) : "" }
  ];
  const resultFacts = [
    composed?.max_model_len && `${(Number(composed.max_model_len) / 1024).toFixed(0)}K ctx`,
    composed?.max_num_seqs && `${composed.max_num_seqs} seq`,
    composed?.kv_cache_dtype && `KV ${composed.kv_cache_dtype}`
  ].filter(Boolean) as string[];
  return (
    <div className="comp-chain">
      {layers.map((l) => (
        <Fragment key={l.kind}>
          <div className={`comp-layer ${l.item ? "" : l.optional ? "none" : "empty"}`}>
            <div className="comp-layer-head">{l.icon} {l.kind}{l.optional && !l.item ? ` (${tr("none")})` : ""}</div>
            <code className="comp-layer-id">{l.item?.id ?? (l.optional ? "—" : tr("pick one"))}</code>
            {l.fact && <div className="comp-layer-fact">{l.fact}</div>}
          </div>
          <ChevronRight className="comp-arrow" size={16} />
        </Fragment>
      ))}
      <div className="comp-layer result">
        <div className="comp-layer-head"><Rocket size={13} /> {tr("Composed")}</div>
        <code className="comp-layer-id">{tr("runtime config")}</code>
        <div className="comp-layer-fact">{resultFacts.length ? resultFacts.join(" · ") : tr("preview to resolve")}</div>
      </div>
    </div>
  );
}

// The full resolved runtime config (every scalar the layers compose to).
function ResolvedConfig({ composed }: { composed: Record<string, any> }) {
  const entries = Object.entries(composed || {})
    .filter(([, v]) => v !== null && v !== undefined && v !== "" && typeof v !== "object")
    .sort((a, b) => a[0].localeCompare(b[0]));
  const envCount = Object.keys((composed?.genesis_env as Record<string, unknown>) ?? {}).length;
  if (!entries.length) return null;
  return (
    <details className="resolved-config" open>
      <summary>{tr("Resolved runtime config")} — {entries.length} {tr("parameters")}{envCount ? ` · ${envCount} ${tr("patch flags")}` : ""}</summary>
      <div className="resolved-grid">
        {entries.map(([k, v]) => (
          <div key={k} className="resolved-row"><code>{k}</code><span>{typeof v === "boolean" ? (v ? tr("yes") : tr("no")) : String(v)}</span></div>
        ))}
      </div>
    </details>
  );
}

function V2ConfigWorkbench({
  catalog,
  preview,
  selectedPreset,
  userPresets,
  onPreview,
  onUserPresetsRefresh
}: {
  catalog: V2ConfigCatalog | null;
  preview: V2ConfigPreview | null;
  selectedPreset: string;
  userPresets: UserPresetList | null;
  onPreview: (preview: V2ConfigPreview) => void;
  onUserPresetsRefresh: () => void;
}) {
  const initialPreset = catalog?.presets.find((preset) => preset.id === selectedPreset) ?? catalog?.presets[0];
  const [modelId, setModelId] = useState(initialPreset?.model ?? "qwen3.6-35b-a3b-fp8");
  const [hardwareId, setHardwareId] = useState(initialPreset?.hardware ?? "a5000-2x-24gbvram-16cpu-128gbram");
  const [profileId, setProfileId] = useState(initialPreset?.profile ?? "");
  const [runtime, setRuntime] = useState(initialPreset?.runtime ?? "docker");
  const [filter, setFilter] = useState("");
  const [draftPresetId, setDraftPresetId] = useState(`gui-draft-${selectedPreset}`);
  const [configPlan, setConfigPlan] = useState<V2ConfigPlan | null>(null);
  const [planState, setPlanState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [planError, setPlanError] = useState<string | null>(null);
  const [applyResult, setApplyResult] = useState<V2ConfigApplyResult | null>(null);
  const [applyState, setApplyState] = useState<"idle" | "loading" | "done">("idle");
  const [applyError, setApplyError] = useState<string | null>(null);
  const [profileDef, setProfileDef] = useState<V2LayerDefinition | null>(null);
  const [artifactKind, setArtifactKind] = useState<"compose" | "run" | "systemd" | "env">("compose");
  // composed is derived from preview, so depending on `preview` covers it; the
  // `?? {}` is inlined to avoid an unstable new-object dependency each render.
  const baseDraft = useMemo(
    () => buildRuntimeDraft(preview?.composed ?? {}, runtime, "safe"),
    [preview, runtime]
  );
  const [draft, setDraft] = useState<RuntimeConfigDraft>(baseDraft);
  const [yamlText, setYamlText] = useState<string | null>(null);
  useEffect(() => {
    setDraft(baseDraft);
    setYamlText(null);
  }, [baseDraft]);
  const setParam = (patch: Partial<RuntimeConfigDraft>) => setDraft((current) => ({ ...current, ...patch }));

  useEffect(() => {
    if (!profileId) {
      setProfileDef(null);
      return;
    }
    let cancelled = false;
    api.v2Layer("profile", profileId)
      .then((detail) => {
        if (!cancelled) setProfileDef(detail);
      })
      .catch(() => {
        if (!cancelled) setProfileDef(null);
      });
    return () => {
      cancelled = true;
    };
  }, [profileId]);

  useEffect(() => {
    if (!initialPreset) return;
    setModelId(initialPreset.model ?? modelId);
    setHardwareId(initialPreset.hardware ?? hardwareId);
    setProfileId(initialPreset.profile ?? "");
    setRuntime(initialPreset.runtime ?? "docker");
    setDraftPresetId(`gui-draft-${initialPreset.id}`);
    setConfigPlan(null);
    setPlanState("idle");
    setPlanError(null);
    setApplyResult(null);
    setApplyState("idle");
    setApplyError(null);
    // Only resync when catalog/selected preset changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialPreset?.id]);

  const compatibleProfiles = (catalog?.profiles ?? []).filter(
    (profile) => !profile.parent_model || profile.parent_model === modelId
  );
  const visiblePresets = (catalog?.presets ?? []).filter((preset) => {
    const needle = filter.trim().toLowerCase();
    if (!needle) return true;
    return [preset.id, preset.title, preset.model, preset.hardware, preset.profile]
      .filter(Boolean)
      .some((value) => String(value).toLowerCase().includes(needle));
  });
  const selectedModel = catalog?.models.find((item) => item.id === modelId) ?? null;
  const selectedHardware = catalog?.hardware.find((item) => item.id === hardwareId) ?? null;
  const selectedProfile = catalog?.profiles.find((item) => item.id === profileId) ?? null;
  const configCtx: ConfigContext = { presetId: draftPresetId, modelId, hardwareId, profileId, runtime, draft };
  const configCli = buildConfigCli(configCtx);
  const configArtifacts = buildConfigArtifacts(configCtx);
  const configDiffs = runtimeDraftDiff(baseDraft, draft);
  const configYaml = yamlText ?? buildFullConfigYaml(configCtx).join("\n");

  async function runPreview(next?: {
    model?: string | null;
    hardware?: string | null;
    profile?: string | null;
    runtime?: string | null;
  }) {
    const model = next?.model ?? modelId;
    const hardware = next?.hardware ?? hardwareId;
    const profile = next?.profile ?? profileId;
    const runtimeValue = next?.runtime ?? runtime;
    if (!model || !hardware) return;
    setConfigPlan(null);
    setPlanState("idle");
    setPlanError(null);
    onPreview(
      await api.v2ConfigPreview({
        model_id: model,
        hardware_id: hardware,
        profile_id: profile || undefined,
        runtime: runtimeValue || undefined
      })
    );
  }

  function applyPresetItem(preset: V2ConfigItem) {
    const nextModel = preset.model ?? modelId;
    const nextHardware = preset.hardware ?? hardwareId;
    const nextProfile = preset.profile ?? "";
    const nextRuntime = preset.runtime ?? "docker";
    setModelId(nextModel);
    setHardwareId(nextHardware);
    setProfileId(nextProfile);
    setRuntime(nextRuntime);
    setDraftPresetId(`gui-draft-${preset.id}`);
    void runPreview({
      model: nextModel,
      hardware: nextHardware,
      profile: nextProfile,
      runtime: nextRuntime
    });
  }

  async function runConfigPlan() {
    setPlanState("loading");
    setPlanError(null);
    try {
      const nextPlan = await api.v2ConfigPlan({
        preset_id: draftPresetId,
        model_id: modelId,
        hardware_id: hardwareId,
        profile_id: profileId || undefined,
        runtime: runtime || undefined
      });
      setConfigPlan(nextPlan);
      setPlanState("ready");
      setApplyResult(null);
      setApplyState("idle");
      setApplyError(null);
    } catch (err) {
      setPlanState("error");
      setPlanError(err instanceof Error ? err.message : String(err));
    }
  }

  async function runConfigApply() {
    if (!configPlan) return;
    setApplyState("loading");
    setApplyError(null);
    try {
      const result = await api.v2ConfigApply({
        preset_id: draftPresetId,
        model_id: modelId,
        hardware_id: hardwareId,
        profile_id: profileId || undefined,
        runtime: runtime || undefined,
        expected_plan_id: configPlan.plan_id
      });
      setApplyResult(result);
      setApplyState("done");
      if (result.status === "applied") {
        onUserPresetsRefresh();
      }
    } catch (err) {
      setApplyState("done");
      setApplyError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <section className="config-workbench">
      <div className="config-control-bar">
        <div className="config-control-selects">
          <ConfigSelect
            label={tr("Model")}
            icon={<Box size={16} />}
            value={modelId}
            items={catalog?.models ?? []}
            onChange={(value) => {
              setModelId(value);
              const firstProfile = catalog?.profiles.find((profile) => profile.parent_model === value)?.id ?? "";
              setProfileId(firstProfile);
              void runPreview({ model: value, profile: firstProfile });
            }}
          />
          <ConfigSelect
            label={tr("Hardware")}
            icon={<HardDrive size={16} />}
            value={hardwareId}
            items={catalog?.hardware ?? []}
            onChange={(value) => { setHardwareId(value); void runPreview({ hardware: value }); }}
          />
          <ConfigSelect
            label={tr("Profile")}
            icon={<Gauge size={16} />}
            value={profileId}
            items={compatibleProfiles}
            allowEmpty
            onChange={(value) => { setProfileId(value); void runPreview({ profile: value }); }}
          />
          <label className="config-select-card">
            <span><Server size={16} /> {tr("Runtime")}</span>
            <select value={runtime} onChange={(event) => { setRuntime(event.target.value); void runPreview({ runtime: event.target.value }); }}>
              {["docker", "podman", "kubernetes", "systemd", "bare"].map((item) => (<option key={item} value={item}>{item}</option>))}
            </select>
            <small>{tr("Container / orchestration target")}</small>
          </label>
          <label className="config-select-card">
            <span><PackageCheck size={16} /> {tr("Draft id")}</span>
            <input
              value={draftPresetId}
              onChange={(event) => { setDraftPresetId(event.target.value); setConfigPlan(null); setPlanState("idle"); setPlanError(null); }}
            />
            <small>{tr("Operator-local config name")}</small>
          </label>
        </div>
        <div className="config-control-actions">
          <button className="ghost-button" onClick={() => void runPreview()}><RefreshCw size={15} /> {tr("Preview")}</button>
          <button className="ghost-button" onClick={() => void runConfigPlan()} disabled={planState === "loading"}>
            <ListChecks size={15} /> {planState === "loading" ? tr("Planning") : tr("Plan")}
          </button>
          <button
            className={configPlan?.valid ? "primary-action" : "disabled-launch"}
            disabled={!configPlan?.valid || applyState === "loading"}
            title={configPlan?.valid ? tr("Write this draft to your operator-local config dir") : tr("Run Plan first")}
            onClick={() => void runConfigApply()}
          >
            <PackageCheck size={15} /> {applyState === "loading" ? tr("Applying") : tr("Apply Plan")}
          </button>
        </div>
      </div>

      <div className="config-status-strip">
        <RailCheck label={tr("Compatible")} value={preview?.compatible ? tr("yes") : tr("no")} status={preview?.compatible ? "pass" : "warning"} />
        <RailCheck label={tr("Status")} value={preview?.status ?? "—"} status={preview?.status ? "pass" : "warning"} />
        <RailCheck label={tr("Context")} value={formatTokens(asNumber(preview?.composed.max_model_len))} status="pass" />
        <RailCheck label={tr("Sequences")} value={String(asNumber(preview?.composed.max_num_seqs) || "—")} status="pass" />
        <RailCheck label={tr("KV cache")} value={asText(preview?.composed.kv_cache_dtype, "—")} status="pass" />
        <RailCheck label={tr("Spec decode")} value={asText(preview?.composed.spec_decode_method, "—")} status="pass" />
        <RailCheck label={tr("Patches")} value={String(asNumber(preview?.composed.enabled_patches_count) || "—")} status="pass" />
      </div>

      <TabbedSection
        id={`config-${modelId}-${hardwareId}-${profileId}`}
        tabs={[
          {
            id: "composition", label: tr("Composition"), icon: <SlidersHorizontal size={15} />,
            render: () => (
              <>
              <ModuleGrid>
                <ModuleCard title={tr("Layer Inspector")} icon={<SlidersHorizontal size={18} />} desc={tr("A preset is composed from three layers — model + hardware + profile — resolved to one runtime config.")} wide>
                  <CompositionChain model={selectedModel} hardware={selectedHardware} profile={selectedProfile} composed={preview?.composed ?? {}} />
                  {selectedProfile?.parent_model && modelId && selectedProfile.parent_model !== modelId && (
                    <div className="comp-conflict">
                      <AlertTriangle size={13} /> {tr("Profile")} <code>{selectedProfile.id}</code> {tr("targets model")} <code>{selectedProfile.parent_model}</code>, {tr("but the composer has")} <code>{modelId}</code> — {tr("sizing/patches from this profile may not apply cleanly.")}
                    </div>
                  )}
                  <ResolvedConfig composed={preview?.composed ?? {}} />
                  <div className="config-layers-row">
                    <ConfigItemInspector title="Model" titleLabel={tr("Model")} item={selectedModel} />
                    <ConfigItemInspector title="Hardware" titleLabel={tr("Hardware")} item={selectedHardware} />
                    <ConfigItemInspector title="Profile" titleLabel={tr("Profile")} item={selectedProfile} />
                  </div>
                  {profileDef && <ProfileDeltaPanel def={profileDef.definition} />}
                </ModuleCard>
                <ModuleCard title={tr("Compatibility & Fit")} icon={<Gauge size={18} />} desc={`${tr("Does")} ${modelId} ${tr("fit on")} ${hardwareId}?`} wide>
                  <ModelFitCard
                    modelId={modelId}
                    hardwareOptions={(catalog?.hardware ?? []).map((item) => item.id)}
                    defaultHardware={hardwareId}
                  />
                </ModuleCard>
              </ModuleGrid>
              <ModuleGrid className="config-aux-grid">
                <ModuleCard title={tr("Compose Messages")} icon={<FileText size={18} />} desc={tr("Notes emitted while composing this configuration.")}>
                  {(preview?.messages ?? []).length
                    ? <CompactList rows={(preview?.messages ?? []).map((message, index) => [`${tr("Message")} ${index + 1}`, message])} />
                    : <p className="muted">{tr("Composition produced no messages.")}</p>}
                </ModuleCard>
                <ModuleCard title={tr("Preset Templates")} icon={<Database size={18} />} desc={tr("Load a builtin preset's layer stack into the composer.")}>
                  <label className="search-box">
                    <Search size={15} />
                    <input aria-label={tr("Search presets, models and profiles")} value={filter} onChange={(event) => setFilter(event.target.value)} placeholder={tr("Search preset/model/profile")} />
                  </label>
                  <div className="preset-template-list">
                    {visiblePresets.map((preset) => (
                      <button
                        className={preset.model === modelId && preset.hardware === hardwareId && preset.profile === profileId ? "active" : ""}
                        key={preset.id}
                        onClick={() => applyPresetItem(preset)}
                      >
                        <strong>{preset.id}</strong>
                        <span>{preset.model}</span>
                        <small>{preset.profile ?? tr("no profile")} / {preset.status || tr("unannotated")}</small>
                      </button>
                    ))}
                  </div>
                </ModuleCard>
                <ModuleCard title={tr("User Presets")} icon={<PackageCheck size={18} />} desc={tr("Operator-local presets written by Apply Plan.")}>
                  <UserPresetsPanel presets={userPresets} />
                </ModuleCard>
              </ModuleGrid>
              </>
            )
          },
          {
            id: "draft", label: tr("Draft"), icon: <Code2 size={15} />,
            render: () => (
              <ModuleGrid>
                <ModuleCard title={tr("Runtime Parameters")} icon={<SlidersHorizontal size={18} />} desc={`${tr("Local draft — diff vs composed baseline")} (${configDiffs.length} ${tr("changes")}).`} wide>
                  <div className="param-editor-controls">
                    <ParamFields baseDraft={baseDraft} draft={draft} set={setParam} />
                  </div>
                </ModuleCard>
                <ModuleCard title={tr("Draft YAML")} icon={<Code2 size={18} />} desc={tr("Editable preview — persist via Apply Plan.")} wide>
                  <div className="yaml-editor">
                    <CodeEditorField value={configYaml} onChange={setYamlText} label={tr("config YAML")} />
                    <div className="config-actions">
                      <span className="config-actions-note">{tr("Editable preview — persist via Apply Plan")}</span>
                      <button className="ghost-button" onClick={() => setYamlText(null)}>{tr("Reset to generated")}</button>
                    </div>
                  </div>
                </ModuleCard>
                <ModuleCard title={tr("Draft Diff")} icon={<GitBranch size={18} />} desc={tr("Draft vs composed baseline + planned file diff.")} wide>
                  <div className="diff-panel tall">
                    <strong>{tr("Draft vs composed baseline")} ({configDiffs.length})</strong>
                    {configDiffs.length ? configDiffs.map((line, index) => <span key={index}>{line}</span>) : <span>{tr("No parameter changes")}</span>}
                    {configPlan && configPlan.diff_lines.length > 0 && (
                      <>
                        <strong className="diff-section">{tr("Planned file diff")}</strong>
                        {configPlan.diff_lines.map((line, index) => (
                          <span key={`plan-${index}`} className={line.startsWith("+") ? "add" : line.startsWith("-") ? "del" : ""}>{line}</span>
                        ))}
                      </>
                    )}
                  </div>
                </ModuleCard>
              </ModuleGrid>
            )
          },
          {
            id: "outputs", label: tr("Outputs"), icon: <SquareTerminal size={15} />,
            render: () => (
              <ModuleGrid>
                <ModuleCard title={tr("Generated Artifacts")} icon={<Layers3 size={18} />} desc={tr("Compose / docker run / systemd / env rendered from this draft.")} wide>
                  <div className="artifact-mode">
                    <div className="artifact-tabs">
                      {(["compose", "run", "systemd", "env"] as const).map((kind) => (
                        <button key={kind} className={artifactKind === kind ? "active" : ""} onClick={() => setArtifactKind(kind)}>
                          {kind === "run" ? "docker run" : kind}
                        </button>
                      ))}
                    </div>
                    <CodeBlock lines={configArtifacts[artifactKind]} />
                  </div>
                </ModuleCard>
                <ModuleCard title={tr("CLI Mirror")} icon={<SquareTerminal size={18} />} desc={tr("Equivalent sndr CLI to reproduce this composition.")} wide>
                  <CodeBlock lines={configCli} />
                </ModuleCard>
              </ModuleGrid>
            )
          },
          {
            id: "compare", label: tr("Compare"), icon: <GitBranch size={15} />,
            render: () => (
              <ModuleGrid>
                <ModuleCard title={tr("Compare presets")} icon={<GitBranch size={18} />} desc={tr("Diff two presets' composed runtime configuration — context, concurrency, KV, patches and more.")} wide>
                  <ConfigComparePanel presets={catalog?.presets ?? []} />
                </ModuleCard>
              </ModuleGrid>
            )
          },
          {
            id: "plan", label: tr("Plan & Apply"), icon: <PackageCheck size={15} />,
            render: () => (
              <ModuleGrid>
                <ModuleCard title={tr("Plan & Apply")} icon={<PackageCheck size={18} />} desc={tr("Write-safe plan, then Apply Plan to persist an operator-local preset.")} wide>
                  {planError && <div className="config-plan-error"><AlertCircle size={15} /><span>{planError}</span></div>}
                  {applyError && <div className="config-plan-error"><AlertCircle size={15} /><span>{applyError}</span></div>}
                  {configPlan ? <ConfigPlanPanel plan={configPlan} /> : <p className="muted">{tr("Run “Plan” to preview the write (diff + target path), then “Apply Plan”.")}</p>}
                  {applyResult && <ConfigApplyPanel result={applyResult} />}
                </ModuleCard>
              </ModuleGrid>
            )
          }
        ]}
      />
    </section>
  );
}

function ConfigSelect({
  label,
  icon,
  value,
  items,
  allowEmpty = false,
  onChange
}: {
  label: string;
  icon: ReactNode;
  value: string;
  items: V2ConfigItem[];
  allowEmpty?: boolean;
  onChange: (value: string) => void;
}) {
  const selected = items.find((item) => item.id === value) ?? null;
  return (
    <label className="config-select-card">
      <span>{icon} {label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {allowEmpty && <option value="">{tr("No profile")}</option>}
        {items.map((item) => (
          <option key={item.id} value={item.id}>{item.id}</option>
        ))}
      </select>
      <small>{selected?.title ?? tr("No layer selected")}</small>
    </label>
  );
}

const inspectorRows = (): Record<string, Array<[string, string]>> => ({
  Model: [
    ["served_model_name", tr("Served")], ["quantization", tr("Quant")], ["dtype", tr("Dtype")],
    ["attention_arch", tr("Attention")], ["kv_cache_dtype", tr("KV cache")],
    ["min_gpu_count", tr("Min GPUs")], ["min_total_vram_mib", tr("Min VRAM")], ["patch_count", tr("Patches")]
  ],
  Hardware: [
    ["n_gpus", tr("GPUs")], ["min_vram_per_gpu_mib", tr("VRAM/GPU")], ["max_model_len", tr("Max ctx")],
    ["max_num_seqs", tr("Max seqs")], ["gpu_memory_utilization", tr("GPU util")], ["runtime_default", tr("Runtime")]
  ],
  Profile: [
    ["max_model_len", tr("Max ctx")], ["max_num_seqs", tr("Max seqs")],
    ["gpu_memory_utilization", tr("GPU util")], ["enable_delta", tr("Patch +")], ["disable_delta", tr("Patch −")]
  ]
});

function ConfigItemInspector({ title, titleLabel, item }: { title: string; titleLabel?: string; item: V2ConfigItem | null }) {
  const label = titleLabel ?? title;
  const icon = title === "Hardware" ? <Cpu size={16} /> : title === "Profile" ? <SlidersHorizontal size={16} /> : <Box size={16} />;
  if (!item) {
    return (
      <div className="config-item-inspector empty">
        <span className="catalog-card-ico">{icon}</span>
        <div><strong>{label}</strong><small>{tr("No selection")}</small></div>
      </div>
    );
  }
  const f = item.fields ?? {};
  const fmt = (key: string, value: any): string => {
    if (value === null || value === undefined || value === "") return "-";
    if (key.includes("vram")) return formatVram(Number(value));
    if (key === "max_model_len") return formatTokens(Number(value));
    if (Array.isArray(value)) return value.length ? value.map(String).join(", ") : "-";
    return String(value);
  };
  const rows = (inspectorRows()[title] ?? [])
    .filter(([key]) => f[key] !== undefined && f[key] !== null && f[key] !== "")
    .map(([key, label]) => [label, fmt(key, f[key])] as [string, string]);
  return (
    <div className="config-item-inspector card">
      <div className="config-inspector-head">
        <span className="catalog-card-ico">{icon}</span>
        <div>
          <strong>{label}: {item.id}</strong>
          {item.title && item.title !== item.id && <small>{item.title}</small>}
        </div>
      </div>
      {itemBadges(item).length > 0 && (
        <span className="catalog-badges">
          {itemBadges(item).map((badge, index) => (
            <span key={index} className={`catalog-badge tone-${badge.tone ?? "neutral"}`}>{badge.label}</span>
          ))}
        </span>
      )}
      {rows.length > 0 && <InfoRows rows={rows} />}
      <small className="config-inspector-src">{item.source}</small>
    </div>
  );
}


function CodeEditorField({ value, onChange, label }: { value: string; onChange: (next: string) => void; label?: string }) {
  const [expanded, setExpanded] = useState(false);
  const dialogRef = useRef<HTMLElement>(null);
  useDialogFocus(dialogRef, expanded);
  useEffect(() => {
    if (!expanded) return;
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") setExpanded(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded]);
  return (
    <div className="code-wrap code-editor-field">
      <div className="code-actions">
        <button className="icon-only" title={tr("Expand editor")} aria-label={tr("Expand editor to fullscreen")} onClick={() => setExpanded(true)}><Maximize2 size={13} /></button>
        <CopyButton value={value} label={label ?? tr("text")} />
      </div>
      <CodeEditor value={value} onChange={onChange} ariaLabel={label ?? tr("YAML editor")} />
      {expanded && (
        <div className="dialog-backdrop" role="presentation" onClick={closeOnBackdrop(() => setExpanded(false))}>
          <section ref={dialogRef} className="code-expand" role="dialog" aria-modal="true" aria-label={`${label ?? tr("Editor")} — ${tr("fullscreen editor")}`}>
            <header className="code-expand-head">
              <Code2 size={15} />
              <strong>{label ?? tr("Editor")}</strong>
              <span className="muted">{value.split("\n").length} {tr("lines")}</span>
              <CopyButton value={value} label={label ?? tr("text")} />
              <button className="icon-only" onClick={() => setExpanded(false)} aria-label={tr("Close")}><X size={16} /></button>
            </header>
            <CodeEditor value={value} onChange={onChange} expanded autoFocus ariaLabel={label ?? tr("YAML editor")} />
          </section>
        </div>
      )}
    </div>
  );
}

type ConfigContext = {
  presetId: string;
  modelId: string;
  hardwareId: string;
  profileId: string;
  runtime: string;
  draft: RuntimeConfigDraft;
};

function buildFullConfigYaml(ctx: ConfigContext): string[] {
  const { presetId, modelId, hardwareId, profileId, runtime, draft } = ctx;
  return [
    "schema_version: 2",
    "kind: preset",
    `id: ${presetId}`,
    `model: ${modelId}`,
    `hardware: ${hardwareId}`,
    ...(profileId ? [`profile: ${profileId}`] : []),
    `runtime: ${runtime}`,
    "card:",
    "  title: GUI draft preset",
    "  status: experimental",
    "  audience: operator",
    `patch_policy: ${draft.patch_policy}`,
    "sizing_override:",
    `  max_model_len: ${draft.max_model_len}`,
    `  max_num_seqs: ${draft.max_num_seqs}`,
    `  max_num_batched_tokens: ${draft.max_num_batched_tokens}`,
    `  gpu_memory_utilization: ${draft.gpu_memory_utilization.toFixed(2)}`,
    `  enable_chunked_prefill: ${draft.enable_chunked_prefill}`,
    `  enforce_eager: ${draft.enforce_eager}`,
    `  disable_custom_all_reduce: ${draft.disable_custom_all_reduce}`,
    "capabilities:",
    `  kv_cache_dtype: ${draft.kv_cache_dtype}`,
    "  spec_decode:",
    `    method: ${draft.spec_decode_method || "none"}`,
    `    num_speculative_tokens: ${draft.spec_decode_K}`
  ];
}

function buildConfigCli(ctx: ConfigContext): string[] {
  const { presetId, modelId, hardwareId, profileId, runtime, draft } = ctx;
  const profileFlag = profileId ? ` --profile ${profileId}` : "";
  return [
    "# Compose and inspect the layered config",
    `sndr config compose --model ${modelId} --hardware ${hardwareId}${profileFlag} --runtime ${runtime}`,
    `sndr config show ${presetId} --format yaml`,
    "",
    "# Plan a write-safe operator-local draft (~/.sndr/model_configs)",
    `sndr config plan --preset ${presetId} \\`,
    `  --model ${modelId} --hardware ${hardwareId}${profileFlag} --runtime ${runtime}`,
    "sndr config apply --plan <plan_id>",
    "",
    "# Dry-run a launch from the draft",
    `sndr launch plan --preset ${presetId} --runtime-target ${runtime} --patch-policy ${draft.patch_policy} --dry-run`
  ];
}

function buildConfigArtifacts(ctx: ConfigContext): Record<"compose" | "run" | "systemd" | "env", string[]> {
  const { presetId, draft } = ctx;
  const container = `vllm-${presetId}`;
  // Only env the vLLM container actually reads. The preset/model/policy
  // identity is carried by the launch plan + container labels, not by env
  // vars the engine never consults — emitting SNDR_PRESET/SNDR_MODEL/etc. here
  // would set no-op vars an operator might assume are load-bearing.
  const env = [
    `VLLM_MAX_MODEL_LEN=${draft.max_model_len}`,
    `VLLM_MAX_NUM_SEQS=${draft.max_num_seqs}`,
    `VLLM_GPU_MEMORY_UTILIZATION=${draft.gpu_memory_utilization.toFixed(2)}`,
    `VLLM_KV_CACHE_DTYPE=${draft.kv_cache_dtype}`
  ];
  return {
    env,
    run: [
      "docker run --rm --gpus all \\",
      `  --name ${container} \\`,
      "  -p 8000:8000 -p 8001:8001 \\",
      ...env.map((entry) => `  -e ${entry} \\`),
      "  ghcr.io/sndr/vllm-runtime:catalog \\",
      `  --served-model-name ${presetId} \\`,
      `  --max-model-len ${draft.max_model_len} \\`,
      `  --max-num-seqs ${draft.max_num_seqs} \\`,
      `  --gpu-memory-utilization ${draft.gpu_memory_utilization.toFixed(2)} \\`,
      `  --kv-cache-dtype ${draft.kv_cache_dtype}`
    ],
    compose: [
      'version: "3.8"',
      "services:",
      "  sndr-vllm:",
      "    image: ghcr.io/sndr/vllm-runtime:catalog",
      `    container_name: ${container}`,
      '    ports: ["8000:8000", "8001:8001"]',
      "    environment:",
      ...env.map((entry) => {
        const [key, ...rest] = entry.split("=");
        return `      ${key}: ${rest.join("=")}`;
      }),
      "    deploy:",
      "      resources:",
      "        reservations:",
      "          devices: [{ capabilities: [gpu] }]"
    ],
    systemd: [
      "[Unit]",
      `Description=SNDR vLLM runtime (${presetId})`,
      "After=network-online.target",
      "",
      "[Service]",
      ...env.map((entry) => `Environment=${entry}`),
      `ExecStart=/usr/bin/docker run --rm --gpus all --name ${container} \\`,
      "  -p 8000:8000 ghcr.io/sndr/vllm-runtime:catalog",
      "Restart=on-failure",
      "RestartSec=5s",
      "",
      "[Install]",
      "WantedBy=multi-user.target"
    ]
  };
}
