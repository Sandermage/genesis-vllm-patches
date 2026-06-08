// SPDX-License-Identifier: Apache-2.0
// Models workbench — the Models tab: catalog browser with per-model key facts,
// fit cards/matrix, KV envelope, patch matrix, runtime envelope and the layer /
// draft / management editors. Extracted from App.tsx (modularization) with no
// behavior change. itemBadges is exported because the catalog views reuse it.
 
import { lazy, useMemo, useState } from "react";
import {
  Box, Boxes, Code2, Cpu, Database, Download, FileText, Gauge, GitBranch, HardDrive,
  Layers3, MemoryStick, PackageCheck, Search, ShieldCheck, SlidersHorizontal, Table2, Wrench, X
} from "lucide-react";
import { api, type V2ConfigItem, type V2ConfigCatalog, type PresetRecord } from "../api";
import { useFetch } from "../hooks/useFetch";
import { asRecord } from "../lib/coerce";
import { formatVram } from "../lib/format";
import { InfoRows, StatusBadge } from "../components/primitives";
import { ModuleCard, ModuleGrid } from "../components/layout";
import { TabbedSection } from "../components/tabbed-section";
import { EmptyState } from "../components/empty-state";
import { CatalogCard, type CatalogBadge, ModelFitCard, ModelFitMatrix, KvEnvelopeCard } from "./catalog-cards";
import { PatchMatrixViewer } from "./patch-matrix";
import { RuntimeEnvelopePanel } from "./preset-insight";
import { LayerEditor } from "./layer-editor";
import { ConfigDraftEditor } from "./config-draft-editor";

const ModelManagementPanel = lazy(() => import("../Engine").then((m) => ({ default: m.ModelManagementPanel })));

function modelFamily(id: string): string {
  if (id.startsWith("qwen")) return "Qwen 3.6";
  if (id.startsWith("gemma")) return "Gemma 4";
  const head = id.split("-")[0];
  return head.charAt(0).toUpperCase() + head.slice(1);
}

export function itemBadges(item: V2ConfigItem): CatalogBadge[] {
  const f = item.fields ?? {};
  const out: CatalogBadge[] = [];
  const push = (label: unknown, tone: CatalogBadge["tone"] = "neutral") => {
    if (label === null || label === undefined || label === "") return;
    out.push({ label: String(label), tone });
  };
  if (item.kind === "model") {
    push(f.quantization ?? f.dtype, "accent");
    if (f.kv_cache_dtype && f.kv_cache_dtype !== "auto") push(`KV ${f.kv_cache_dtype}`);
    if (Number(f.patch_count) > 0) push(`${f.patch_count} patches`, "ok");
  } else if (item.kind === "hardware") {
    push(f.n_gpus ? `${f.n_gpus}× GPU` : null, "accent");
    if (f.max_model_len) push(`${Math.round(Number(f.max_model_len) / 1024)}k ctx`);
    if (f.runtime_default) push(String(f.runtime_default));
  } else if (item.kind === "profile") {
    push(item.parent_model, "accent");
    if (f.max_num_seqs) push(`seqs ${f.max_num_seqs}`);
    if (Number(f.enable_delta) > 0) push(`+${f.enable_delta} patch`, "ok");
  } else if (item.kind === "preset") {
    push(f.model ?? item.model, "accent");
    push(f.hardware ?? item.hardware);
  }
  return out.slice(0, 3);
}

// ModelFitCard + ModelFitMatrix extracted to ./sections/catalog-cards.

function ModelSummaryStrip({ models, activeId }: { models: V2ConfigItem[]; activeId: string }) {
  const fieldsOf = (m: V2ConfigItem) => (m.fields ?? {}) as Record<string, any>;
  const families = new Set(models.map((m) => modelFamily(m.id))).size;
  const moe = models.filter((m) => String(fieldsOf(m).attention_arch ?? "").includes("moe")).length;
  const toolReady = models.filter((m) => fieldsOf(m).tool_call_parser).length;
  const vrams = models.map((m) => Number(fieldsOf(m).min_total_vram_mib) || 0).filter(Boolean);
  const vramRange = vrams.length ? `${Math.round(Math.min(...vrams) / 1024)}–${Math.round(Math.max(...vrams) / 1024)} GB` : "—";
  const tiles: Array<{ label: string; value: string }> = [
    { label: "Models", value: String(models.length) },
    { label: "Families", value: String(families) },
    { label: "MoE", value: String(moe) },
    { label: "Dense", value: String(models.length - moe) },
    { label: "Tool-ready", value: String(toolReady) },
    { label: "Min VRAM", value: vramRange }
  ];
  return (
    <div className="preset-summary-strip model-summary-strip">
      {tiles.map((tile) => (
        <div className="preset-stat" key={tile.label}>
          <span className="preset-stat-value">{tile.value}</span>
          <span className="preset-stat-label">{tile.label}</span>
        </div>
      ))}
      <div className="preset-stat"><span className="preset-stat-value">{activeId || "—"}</span><span className="preset-stat-label">Selected</span></div>
    </div>
  );
}

function ModelKeyFacts({ fields, def }: { fields: Record<string, any>; def: Record<string, any> }) {
  const caps = (def.capabilities ?? {}) as Record<string, any>;
  const reqs = (def.requires ?? {}) as Record<string, any>;
  const vram = Number(fields.min_total_vram_mib ?? reqs.min_total_vram_mib) || 0;
  const facts: Array<[string, string]> = [
    ["Arch", String(fields.attention_arch ?? caps.attention_arch ?? "—")],
    ["Quant", String(def.quantization ?? fields.quantization ?? "baked/none")],
    ["Dtype", String(def.dtype ?? fields.dtype ?? "—")],
    ["KV cache", String(caps.kv_cache_dtype ?? fields.kv_cache_dtype ?? "—")],
    ["Min VRAM", vram ? `${Math.round(vram / 1024)} GB` : "—"],
    ["Min GPUs", String(fields.min_gpu_count ?? reqs.min_gpu_count ?? "—")],
    ["Patches", String(fields.patch_count ?? Object.keys(def.patches ?? {}).length ?? "—")]
  ];
  return (
    <div className="model-keyfacts">
      {facts.map(([label, value]) => (
        <div className="model-keyfact" key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </div>
  );
}

// KvEnvelopeCard extracted to ./sections/catalog-cards.

export function ModelsWorkbench({
  catalog,
  presets,
  selectedPreset,
  selectedModel,
  composed,
  card,
  patchCount,
  runtimeTarget,
  patchPolicy,
  onPreset
}: {
  catalog: V2ConfigCatalog | null;
  presets: PresetRecord[];
  selectedPreset: string;
  selectedModel: string | null;
  composed: Record<string, unknown>;
  card: Record<string, unknown>;
  patchCount: number;
  runtimeTarget: string;
  patchPolicy: string;
  onPreset: (id: string) => void;
}) {
  const models = catalog?.models ?? [];
  const [filter, setFilter] = useState("");
  const [picked, setPicked] = useState<string | null>(null);
  const activeId = picked ?? selectedModel ?? models[0]?.id ?? "";
  const active = models.find((model) => model.id === activeId) ?? null;
  const visible = models.filter((model) => {
    const needle = filter.trim().toLowerCase();
    if (!needle) return true;
    return [model.id, model.title, model.summary]
      .filter(Boolean)
      .some((value) => String(value).toLowerCase().includes(needle));
  });
  const groupedModels = useMemo(() => {
    const map = new Map<string, V2ConfigItem[]>();
    visible.forEach((model) => {
      const family = modelFamily(model.id);
      if (!map.has(family)) map.set(family, []);
      map.get(family)!.push(model);
    });
    return Array.from(map.entries());
  }, [visible]);
  const modelPresets = presets.filter((preset) => preset.model === activeId);
  const { data: cacheReport } = useFetch((signal) => api.modelsCache(signal), []);
  // Real transformer dims (layers / KV heads / head_dim / params / quant) — used
  // for KV+VRAM math; surfaced here so the catalog isn't just curated metadata.
  // calcModels keys by family ("qwen3.6-27b-int4"), the catalog by full variant;
  // match on the "<family>-<size>" base (e.g. qwen3.6-27b, gemma-4-31b).
  const { data: archMeta } = useFetch(() => api.calcModels(), []);
  const archEntry = (() => {
    if (!archMeta || !activeId) return null;
    if (archMeta.models[activeId]) return { key: activeId, arch: archMeta.models[activeId] };
    const base = (id: string) => (id.toLowerCase().match(/^(.+?-\d+b)/)?.[1] ?? id.toLowerCase());
    const t = base(activeId);
    for (const [k, v] of Object.entries(archMeta.models)) if (base(k) === t) return { key: k, arch: v };
    return null;
  })();
  const arch = archEntry?.arch ?? null;
  const archKey = archEntry?.key ?? null;
  const { data: fullDef, state: defState } = useFetch(
    (signal) => api.v2Layer("model", activeId, signal),
    [activeId],
    { enabled: Boolean(activeId) }
  );

  const def = (fullDef?.definition ?? {}) as Record<string, any>;
  const caps = (def.capabilities ?? {}) as Record<string, any>;
  const reqs = (def.requires ?? {}) as Record<string, any>;
  const vers = (def.versions ?? {}) as Record<string, any>;
  const spec = (caps.spec_decode ?? {}) as Record<string, any>;
  const patchMatrix = (def.patches ?? {}) as Record<string, string>;
  const attribution = (def.patches_attribution ?? {}) as Record<string, any>;
  const notes: string[] = Array.isArray(def.notes) ? def.notes : [];
  // Representative rig for the fit envelope: prefer a hardware a preset actually
  // uses with this model, else the model's stated minimums, else a 2×24GB A5000.
  const envHw = (() => {
    const hw = catalog?.hardware.find((h) => h.id === modelPresets[0]?.hardware);
    if (hw) return { tp: Number(hw.fields?.n_gpus) || 2, vram: Number(hw.fields?.min_vram_per_gpu_mib) || 24564, label: hw.id };
    return { tp: Number(reqs.min_gpu_count) || 2, vram: 24564, label: "2×24GB (default)" };
  })();
  const dval = (value: unknown): string => {
    if (Array.isArray(value)) return value.length ? value.map(String).join(", ") : "-";
    if (value === null || value === undefined || value === "") return "-";
    if (typeof value === "boolean") return value ? "yes" : "no";
    return String(value);
  };

  return (
    <div className="models-view">
      <ModelSummaryStrip models={models} activeId={activeId} />
    <section className="models-workbench">
      <section className="model-list-panel">
        <div className="config-panel-title">
          <Box size={16} />
          <strong>Model Catalog</strong>
          <span>{models.length}</span>
        </div>
        <label className="search-box">
          <Search size={15} />
          <input aria-label="Search models" value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="Search model" />
        </label>
        <div className="catalog-list">
          {groupedModels.map(([family, items]) => (
            <div className="catalog-group" key={family}>
              <div className="catalog-group-head">
                <span>{family}</span>
                <small>{items.length}</small>
              </div>
              {items.map((model) => (
                <CatalogCard
                  key={model.id}
                  icon={<Box size={16} />}
                  id={model.id}
                  title={[model.fields?.attention_arch, model.fields?.dtype].filter(Boolean).map(String).join(" · ") || model.title}
                  badges={itemBadges(model)}
                  active={model.id === activeId}
                  onClick={() => setPicked(model.id)}
                />
              ))}
            </div>
          ))}
          {visible.length === 0 && (
            <EmptyState
              icon={<Boxes size={22} />}
              title="No models match"
              message={filter ? <>Nothing in the catalog matches “{filter}”.</> : "The model catalog is empty."}
              action={filter ? { label: "Clear search", icon: <X size={14} />, onClick: () => setFilter("") } : undefined}
            />
          )}
        </div>
        <div className="catalog-foot">
          {visible.length} of {models.length} models · {groupedModels.length} famil{groupedModels.length === 1 ? "y" : "ies"}
        </div>
      </section>

      <section className="model-detail-tabbed">
        <div className="model-detail-head">
          <Cpu size={18} />
          <div>
            <strong>{def.title ?? active?.title ?? activeId ?? "No model"}</strong>
            <span>{def.model_path ?? active?.summary ?? "model definition"}</span>
          </div>
          <StatusBadge status={defState === "loading" ? "partial" : active ? "available" : "missing"} />
        </div>
        {active && <ModelKeyFacts fields={(active.fields ?? {}) as Record<string, any>} def={def} />}
        <TabbedSection
          id={`model-${activeId}`}
          tabs={[
            {
              id: "overview",
              label: "Overview",
              icon: <Box size={15} />,
              render: () => (
                <>
                <ModuleGrid>
                  <ModuleCard title="Fit envelope" icon={<Gauge size={18} />} desc="Does it fit? Context × concurrency headroom on a representative rig." wide>
                    <KvEnvelopeCard modelKey={archKey} tp={envHw.tp} vram={envHw.vram} rigLabel={envHw.label} />
                  </ModuleCard>
                </ModuleGrid>
                <ModuleGrid className="models-row">
                  <ModuleCard title="Identity" icon={<Box size={18} />} desc="Serving name, precision, provenance and trust settings.">
                    <InfoRows
                      rows={[
                        ["Model id", activeId || "-"],
                        ["Served name", dval(def.served_model_name)],
                        ["Maintainer", dval(def.maintainer)],
                        ["Quantization", dval(def.quantization)],
                        ["Dtype", dval(def.dtype)],
                        ["Trust remote code", dval(def.trust_remote_code)],
                        ["License", dval(def.license)],
                        ["Validated", dval(def.last_validated)]
                      ]}
                    />
                  </ModuleCard>
                  <ModuleCard title="Architecture" icon={<Cpu size={18} />} desc="Real transformer dimensions — the basis for KV-cache and VRAM sizing.">
                    {arch ? (() => {
                      // KV bytes/token = 2 (K+V) × layers × kv_heads × head_dim × elem_bytes.
                      const kvTok = (elem: number) => 2 * arch.num_layers * arch.num_kv_heads * arch.head_dim * elem;
                      const mbPerK = (elem: number) => (kvTok(elem) * 1024) / (1024 * 1024);
                      const gbAt = (ctx: number, elem: number) => (kvTok(elem) * ctx) / (1024 ** 3);
                      return (
                        <>
                          <InfoRows rows={[
                            ["Parameters", `${arch.params_b}B${arch.is_moe && arch.active_params_b ? ` · ${arch.active_params_b}B active (MoE)` : ""}`],
                            ["Type", arch.is_moe ? "Mixture-of-Experts" : "Dense"],
                            ["Layers", String(arch.num_layers)],
                            ["KV heads", String(arch.num_kv_heads)],
                            ["Head dim", String(arch.head_dim)],
                            ["Weight precision", `${arch.weight_bits}-bit`]
                          ]} />
                          <div className="kv-footprint">
                            <div className="kv-footprint-head"><MemoryStick size={13} /> KV-cache footprint (1 request)</div>
                            <div className="kv-footprint-rows">
                              <div><span>fp8</span><b>{mbPerK(1).toFixed(1)} MB</b> / 1K · <b>{gbAt(32768, 1).toFixed(1)} GB</b> @ 32K</div>
                              <div><span>fp16</span><b>{mbPerK(2).toFixed(1)} MB</b> / 1K · <b>{gbAt(32768, 2).toFixed(1)} GB</b> @ 32K</div>
                            </div>
                          </div>
                        </>
                      );
                    })() : <p className="muted">No architecture metadata for this model id ({activeId || "—"}).</p>}
                  </ModuleCard>
                  <ModuleCard
                    title="Local Cache"
                    icon={<HardDrive size={18} />}
                    desc={`Checkpoint presence on the API daemon host${cacheReport ? ` (${cacheReport.host})` : ""}.`}
                  >
                    {(() => {
                      const entry = cacheReport?.models.find((m) => m.model_id === activeId);
                      if (!cacheReport) return <p className="muted">Checking cache…</p>;
                      if (!entry) return <p className="muted">No cache entry for this model.</p>;
                      return (
                        <>
                          <InfoRows
                            rows={[
                              ["Checkpoint path", entry.model_path || "-"],
                              ["Present on host", entry.present ? "yes" : "no"],
                              ["Size", entry.size_mib != null ? formatVram(entry.size_mib) : entry.present ? "unknown" : "-"]
                            ]}
                          />
                          <p className="fit-note">
                            {entry.present
                              ? "Checkpoint directory is present on the daemon host."
                              : "Absent on the daemon host — expected when controlling a remote GPU host from a laptop."}
                          </p>
                        </>
                      );
                    })()}
                  </ModuleCard>
                  <ModuleCard title={`Presets using ${activeId || "model"}`} icon={<Database size={18} />} desc="Catalog presets that reference this model — click to load.">
                    {modelPresets.length ? (
                      <div className="model-preset-chips">
                        {modelPresets.map((preset) => (
                          <button key={preset.id} className={preset.id === selectedPreset ? "active" : ""} onClick={() => onPreset(preset.id)}>
                            <strong>{preset.id}</strong>
                            <small>{preset.profile ?? "no profile"}</small>
                          </button>
                        ))}
                      </div>
                    ) : (
                      <p className="muted">No presets reference this model yet.</p>
                    )}
                  </ModuleCard>
                </ModuleGrid>
                <ModuleGrid>
                  <ModuleCard title="Notes" icon={<FileText size={18} />} desc="Maintainer notes and migration history." wide>
                    {notes.length ? (
                      <ul className="model-notes">{notes.map((note, index) => (<li key={index}>{note}</li>))}</ul>
                    ) : (
                      <p className="muted">No maintainer notes.</p>
                    )}
                  </ModuleCard>
                </ModuleGrid>
                </>
              )
            },
            {
              id: "capabilities",
              label: "Capabilities",
              icon: <Layers3 size={15} />,
              render: () => (
                <ModuleGrid className="models-row3">
                  <ModuleCard title="Capabilities" icon={<Layers3 size={18} />} desc="Attention arch, parsers and speculative decode.">
                    <InfoRows
                      rows={[
                        ["Attention arch", dval(caps.attention_arch)],
                        ["Tool parser", dval(caps.tool_call_parser)],
                        ["Reasoning parser", dval(caps.reasoning_parser)],
                        ["Auto tool choice", dval(caps.enable_auto_tool_choice)],
                        ["KV cache dtype", dval(caps.kv_cache_dtype)],
                        ["Spec decode", `${dval(spec.method)} / K=${dval(spec.num_speculative_tokens)}`]
                      ]}
                    />
                  </ModuleCard>
                  <ModuleCard title="Version Pins" icon={<GitBranch size={18} />} desc="Genesis and vLLM pins this model was validated on.">
                    <InfoRows
                      rows={[
                        ["Genesis pin min", dval(vers.genesis_pin_min)],
                        ["vLLM pin required", dval(vers.vllm_pin_required)],
                        ["Reference metrics", dval(vers.reference_metrics_ref)],
                        ["Pin hold", dval(vers.pin_hold)]
                      ]}
                    />
                  </ModuleCard>
                  <ModuleCard title="Generation & Serving" icon={<SlidersHorizontal size={18} />} desc="Chat template, tool/reasoning parsing and sampling defaults.">
                    <InfoRows
                      rows={[
                        ["Served name", dval(def.served_model_name)],
                        ["Chat template", def.chat_template ? "custom template" : "model default"],
                        ["Tool parser", dval(caps.tool_call_parser)],
                        ["Reasoning parser", dval(caps.reasoning_parser)],
                        ["Auto tool choice", dval(caps.enable_auto_tool_choice)]
                      ]}
                    />
                    {(() => {
                      const gen = asRecord(def.override_generation_config);
                      const keys = Object.keys(gen);
                      return keys.length ? (
                        <>
                          <h5 className="model-subhead">Sampling overrides</h5>
                          <InfoRows rows={keys.map((key) => [key.replace(/_/g, " "), dval(gen[key])] as [string, string])} />
                        </>
                      ) : (
                        <p className="muted model-gen-none">No sampling overrides — uses the model's generation defaults.</p>
                      );
                    })()}
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "fit",
              label: "Hardware fit",
              icon: <Gauge size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Requirements" icon={<ShieldCheck size={18} />} desc="Minimum GPUs, VRAM and CUDA capability.">
                    <InfoRows
                      rows={[
                        ["Min GPUs", dval(reqs.min_gpu_count)],
                        ["Min VRAM", formatVram(reqs.min_total_vram_mib)],
                        ["Min CUDA cap", Array.isArray(reqs.min_cuda_capability) ? reqs.min_cuda_capability.join(".") : "-"],
                        ["Arch blocklist", dval(reqs.rig_arch_blocklist)]
                      ]}
                    />
                  </ModuleCard>
                  <ModuleCard title="Hardware Fit" icon={<Gauge size={18} />} desc="Check this model against a rig: GPU count, CUDA capability and VRAM context." wide>
                    <ModelFitCard
                      modelId={activeId}
                      hardwareOptions={(catalog?.hardware ?? []).map((item) => item.id)}
                      defaultHardware={modelPresets[0]?.hardware ?? ""}
                    />
                  </ModuleCard>
                  <ModuleCard title="Fit Matrix" icon={<Table2 size={18} />} desc="Where this model can run across every catalogued rig — fits, blockers and VRAM headroom." wide>
                    <ModelFitMatrix modelId={activeId} hardwareIds={(catalog?.hardware ?? []).map((item) => item.id)} />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "runtime",
              label: "Runtime & patches",
              icon: <SlidersHorizontal size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Runtime Envelope" icon={<SlidersHorizontal size={18} />} desc={`Composed runtime for ${selectedPreset}.`} wide>
                    <RuntimeEnvelopePanel card={card} composed={composed} patchCount={patchCount} />
                  </ModuleCard>
                  <ModuleCard title="Patch Matrix" icon={<Wrench size={18} />} desc="Canonical env-flag overrides shipped with this model — the patches baked into its composed runtime." wide>
                    <PatchMatrixViewer patches={patchMatrix} attribution={attribution} loading={defState === "loading"} />
                  </ModuleCard>
                  <ModuleCard title="Config Draft" icon={<Code2 size={18} />} desc="Local runtime draft (diff + YAML preview)." wide>
                    <ConfigDraftEditor
                      selectedPreset={selectedPreset}
                      composed={composed}
                      runtimeTarget={runtimeTarget}
                      patchPolicy={patchPolicy}
                    />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "cache",
              label: "Cache & download",
              icon: <Download size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title="Checkpoint Cache" icon={<Download size={18} />} desc="Model weights present on the daemon host. Queue a pull for absent checkpoints." wide>
                    <ModelManagementPanel />
                  </ModuleCard>
                </ModuleGrid>
              )
            },
            {
              id: "edit",
              label: "Edit",
              icon: <PackageCheck size={15} />,
              render: () => (
                <ModuleGrid>
                  <ModuleCard title={`Visual Editor — ${activeId}`} icon={<Wrench size={18} />} desc="Full model definition editing (adaptive). Saves an operator-local copy." wide>
                    <LayerEditor kind="model" layerId={activeId} />
                  </ModuleCard>
                </ModuleGrid>
              )
            }
          ]}
        />
      </section>
    </section>
    </div>
  );
}
