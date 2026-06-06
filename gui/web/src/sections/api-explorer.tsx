// SPDX-License-Identifier: Apache-2.0
// API explorer section: a read-only GET endpoint prober + a redacted report
// bundle generator. Extracted from App.tsx (modularization) with no behavior
// change.
import { useState } from "react";
import { Play, AlertCircle, CheckCircle2 } from "lucide-react";
import { api, type ReportBundleResult } from "../api";
import { InfoRows } from "../components/primitives";
import { CodeBlock } from "../components/code-block";

const EXPLORER_ENDPOINTS = [
  "/api/v1/health", "/api/v1/auth/status", "/api/v1/capabilities", "/api/v1/overview",
  "/api/v1/catalog/summary", "/api/v1/environment", "/api/v1/doctor",
  "/api/v1/presets", "/api/v1/configs/v2/catalog", "/api/v1/patches/doctor",
  "/api/v1/proof/status", "/api/v1/models/cache", "/api/v1/memory/fit?model_id=qwen3.6-35b-a3b-fp8&hardware_id=a5000-2x-24gbvram-16cpu-128gbram",
  "/api/v1/jobs", "/api/v1/events/recent", "/api/v1/hosts"
];

export function EndpointExplorer() {
  const [path, setPath] = useState(EXPLORER_ENDPOINTS[2]);
  const [result, setResult] = useState<string[] | null>(null);
  const [meta, setMeta] = useState<{ ok: boolean; ms: number } | null>(null);
  const [busy, setBusy] = useState(false);

  async function send() {
    setBusy(true);
    setMeta(null);
    const started = performance.now();
    try {
      const data = await api.raw(path);
      const ms = Math.round(performance.now() - started);
      setResult(JSON.stringify(data, null, 2).split("\n").slice(0, 400));
      setMeta({ ok: true, ms });
    } catch (err) {
      const ms = Math.round(performance.now() - started);
      setResult([err instanceof Error ? err.message : String(err)]);
      setMeta({ ok: false, ms });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="endpoint-explorer">
      <div className="endpoint-explorer-bar">
        <span className="method-pill">GET</span>
        <select value={path} onChange={(event) => { setPath(event.target.value); setResult(null); setMeta(null); }}>
          {EXPLORER_ENDPOINTS.map((endpoint) => (
            <option key={endpoint} value={endpoint}>{endpoint.replace(/\?.*$/, "")}</option>
          ))}
        </select>
        <button className="primary-action" onClick={() => void send()} disabled={busy}>
          <Play size={15} /> {busy ? "Sending…" : "Send"}
        </button>
        {meta && (
          <span className={`endpoint-meta ${meta.ok ? "ok" : "bad"}`}>
            {meta.ok ? "200 OK" : "error"} · {meta.ms}ms
          </span>
        )}
      </div>
      {result && <CodeBlock lines={result} />}
    </div>
  );
}

export function ReportGenerator({ selectedPreset }: { selectedPreset: string }) {
  const types: Array<[string, string, string]> = [
    ["catalog", "Catalog snapshot", "Overview, catalog, environment, doctor and patch coverage"],
    ["launch", "Launch report", "Plan, gates, runtime artifact and preset explain"],
    ["patch", "Patch report", "Registry coverage, lifecycle and policy"],
    ["doctor", "Doctor report", "Aggregated diagnostics snapshot"]
  ];
  const [busy, setBusy] = useState<string | null>(null);
  const [result, setResult] = useState<ReportBundleResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function generate(reportType: string) {
    setBusy(reportType);
    setError(null);
    try {
      const out = await api.reportBundle({ report_type: reportType, preset_id: selectedPreset, redact: true });
      setResult(out);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate bundle");
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="report-generator">
      <div className="action-rows">
        {types.map(([id, title, detail]) => (
          <div key={id}>
            <div>
              <strong>{title}</strong>
              <small>{detail}</small>
            </div>
            <button onClick={() => void generate(id)} disabled={busy !== null}>
              {busy === id ? "Generating…" : "Generate"}
            </button>
          </div>
        ))}
      </div>
      {error && <div className="config-plan-error"><AlertCircle size={14} /><span>{error}</span></div>}
      {result && (
        <div className="report-result">
          <div className="report-result-head">
            <CheckCircle2 size={15} />
            <strong>{result.bundle_id}</strong>
            <span className={`status-badge ${result.redacted ? "applied" : "partial"}`}>
              {result.redacted ? "redacted" : "raw"}
            </span>
          </div>
          <InfoRows
            rows={[
              ["Type", result.report_type],
              ["Files", result.files.join(", ")],
              ["Written to", result.bundle_dir]
            ]}
          />
          <p className="fit-note">{result.note}</p>
        </div>
      )}
    </div>
  );
}
