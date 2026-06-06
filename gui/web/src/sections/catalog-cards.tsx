// SPDX-License-Identifier: Apache-2.0
// Catalog cards: selectable catalog entry, single-rig fit report, all-rigs fit
// matrix, and the KV fit-envelope heatmap. Extracted from App.tsx
// (modularization) with no behavior change.
import { useState, useEffect, Fragment, type ReactNode } from "react";
import { CheckCircle2, CircleAlert, RefreshCw } from "lucide-react";
import { api, type FitCheck, type MemoryFitReport } from "../api";
import { useFetch } from "../hooks/useFetch";
import { PercentBar } from "../components/charts";
import { formatVram } from "../lib/format";

/** Small badge shown on catalog cards; shared with App.tsx's itemBadges(). */
export type CatalogBadge = { label: string; tone?: "neutral" | "accent" | "ok" | "warn" };

export function CatalogCard({
  icon,
  id,
  title,
  badges = [],
  active,
  onClick
}: {
  icon: ReactNode;
  id: string;
  title?: string;
  badges?: CatalogBadge[];
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button type="button" className={`catalog-card${active ? " active" : ""}`} onClick={onClick}>
      <span className="catalog-card-ico">{icon}</span>
      <span className="catalog-card-body">
        <strong>{id}</strong>
        {title && <small>{title}</small>}
        {badges.length > 0 && (
          <span className="catalog-badges">
            {badges.map((badge, index) => (
              <span key={`${badge.label}-${index}`} className={`catalog-badge tone-${badge.tone ?? "neutral"}`}>
                {badge.label}
              </span>
            ))}
          </span>
        )}
      </span>
    </button>
  );
}

export function ModelFitCard({
  modelId,
  hardwareOptions,
  defaultHardware
}: {
  modelId: string;
  hardwareOptions: string[];
  defaultHardware: string;
}) {
  const initial =
    hardwareOptions.find((id) => id === defaultHardware) ?? hardwareOptions[0] ?? "";
  const [hardware, setHardware] = useState(initial);

  // Re-pin the hardware select when the default changes (e.g. a preset loads).
  useEffect(() => {
    if (hardwareOptions.length && !hardwareOptions.includes(hardware)) {
      setHardware(initial);
    }
  }, [hardwareOptions, initial, hardware]);

  const { data: report, state, reload } = useFetch(
    (signal) => api.memoryFit({ model_id: modelId, hardware_id: hardware }, signal),
    [modelId, hardware],
    { enabled: Boolean(modelId && hardware) }
  );

  const sevIcon = (severity: FitCheck["severity"]) =>
    severity === "ok" ? (
      <CheckCircle2 size={15} className="fit-ico ok" />
    ) : (
      <CircleAlert size={15} className={`fit-ico ${severity}`} />
    );

  const vram = report?.vram;
  return (
    <div className="model-fit">
      <div className="model-fit-bar">
        <label className="model-fit-pick">
          <span>Target hardware</span>
          <select value={hardware} onChange={(event) => setHardware(event.target.value)}>
            {hardwareOptions.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
        </label>
        {state === "ready" && report && (
          <span className={`fit-verdict ${report.compatible ? "ok" : "blocked"}`}>
            {report.compatible ? (
              <>
                <CheckCircle2 size={15} /> Compatible
              </>
            ) : (
              <>
                <CircleAlert size={15} /> Blocked
              </>
            )}
          </span>
        )}
      </div>

      {state === "loading" && <p className="muted">Checking fit…</p>}
      {state === "error" && (
        <p className="muted fit-error">
          Could not build a fit report for this pairing.
          <button type="button" className="link-button" onClick={reload}>
            <RefreshCw size={13} /> Retry
          </button>
        </p>
      )}

      {state === "ready" && report && (
        <>
          <ul className="fit-checks">
            {report.checks.map((check) => (
              <li key={check.id} className={`fit-check ${check.severity}`}>
                {sevIcon(check.severity)}
                <div>
                  <strong>{check.title}</strong>
                  <small>{check.detail}</small>
                </div>
              </li>
            ))}
          </ul>
          {vram && (
            <PercentBar
              value={Math.min(vram.model_min_mib, vram.rig_floor_mib)}
              max={Math.max(vram.model_min_mib, vram.rig_floor_mib, 1)}
              label="VRAM (informational)"
              caption={`model needs ${formatVram(vram.model_min_mib)} · rig floor ${formatVram(
                vram.rig_floor_mib
              )} (${vram.n_gpus}×${formatVram(vram.vram_per_gpu_mib)} × ${Math.round(
                vram.gpu_memory_utilization * 100
              )}%) · KV ${vram.kv_cache_dtype}`}
              tone={vram.headroom_mib >= 0 ? "ok" : "warn"}
            />
          )}
          <p className="fit-note">
            Verdict uses the deterministic requirements (GPU count, CUDA capability, blocklist).
            VRAM is shown for context — the rig floor is a conservative match threshold, not the
            card's real capacity, so it never flips the verdict.
          </p>
        </>
      )}
    </div>
  );
}

// Enterprise fit matrix: probe the model against EVERY catalogued rig so an
// operator sees at a glance where it can run and where it is blocked.
export function ModelFitMatrix({ modelId, hardwareIds }: { modelId: string; hardwareIds: string[] }) {
  const [rows, setRows] = useState<MemoryFitReport[] | null>(null);
  const [loading, setLoading] = useState(false);
  const key = hardwareIds.join(",");
  useEffect(() => {
    if (!modelId || hardwareIds.length === 0) { setRows([]); return; }
    let cancelled = false;
    setLoading(true);
    Promise.all(hardwareIds.map((hw) => api.memoryFit({ model_id: modelId, hardware_id: hw }).catch(() => null)))
      .then((reports) => { if (!cancelled) setRows(reports.filter((report): report is MemoryFitReport => Boolean(report))); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelId, key]);

  if (loading && !rows) return <p className="muted">Checking {hardwareIds.length} rigs…</p>;
  if (!rows || rows.length === 0) return <p className="muted">No hardware in the catalog to match against.</p>;
  const fits = rows.filter((report) => report.compatible).length;
  return (
    <div className="fit-matrix">
      <div className="fit-matrix-summary">
        <span className="fleet-status ok"><span className="fleet-dot" />fits on {fits}</span>
        <span className="fleet-status danger"><span className="fleet-dot" />blocked on {rows.length - fits}</span>
      </div>
      <div className="patch-table-scroll">
        <table className="module-table fit-matrix-table">
          <thead><tr><th>Hardware</th><th>Verdict</th><th>GPUs</th><th>VRAM headroom</th><th>KV</th><th>Blockers</th></tr></thead>
          <tbody>
            {rows.map((report) => {
              const blockers = report.checks.filter((check) => !check.ok).map((check) => check.title);
              return (
                <tr key={report.hardware_id}>
                  <td><strong>{report.hardware_title || report.hardware_id}</strong></td>
                  <td><span className={`fit-pill ${report.compatible ? "ok" : "blocked"}`}>{report.compatible ? "fits" : "blocked"}</span></td>
                  <td>{report.vram.n_gpus}× {formatVram(report.vram.vram_per_gpu_mib)}</td>
                  <td className={report.vram.headroom_mib >= 0 ? "fit-pos" : "fit-neg"}>{report.vram.headroom_mib >= 0 ? "+" : "−"}{Math.abs(report.vram.headroom_mib / 1024).toFixed(1)} GB</td>
                  <td>{report.vram.kv_cache_dtype}</td>
                  <td>{blockers.length ? blockers.join(", ") : <span className="muted">—</span>}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="fit-note">Verdict uses deterministic requirements (GPU count, CUDA capability, arch blocklist). VRAM headroom is informational context against a conservative rig floor.</p>
    </div>
  );
}

// Interactive fit envelope for a model on a representative rig — context ×
// concurrency, cell colour = headroom. Reuses the KV calculator backend so the
// catalog answers "what can this model actually run?" in place.
export function KvEnvelopeCard({ modelKey, tp, vram, rigLabel }: { modelKey: string | null; tp: number; vram: number; rigLabel: string }) {
  const [env, setEnv] = useState<{ contexts: number[]; concurrencies: number[]; grid: Array<Array<{ context: number; headroom_mib: number; fits: boolean }>> } | null>(null);
  const [err, setErr] = useState(false);
  useEffect(() => {
    if (!modelKey) return;
    setEnv(null); setErr(false);
    let alive = true;
    api.calcKv({ model_id: modelKey, context: 32768, concurrency: 1, tp, gpu_count: tp, gpu_vram_mib: vram, util: 0.9, kv_dtype: "fp8" })
      .then((r) => { if (alive) setEnv(r.envelope); }).catch(() => { if (alive) setErr(true); });
    return () => { alive = false; };
  }, [modelKey, tp, vram]);
  if (!modelKey) return <p className="muted">No KV sizing metadata for this model id.</p>;
  if (err) return <p className="muted">Couldn't compute the fit envelope.</p>;
  if (!env) return <p className="muted">Computing fit envelope…</p>;
  const fmtC = (c: number) => (c >= 1000 ? `${Math.round(c / 1000)}K` : String(c));
  const color = (h: number) => (h < 0 ? "over" : h < 2048 ? "tight" : h < 6144 ? "ok" : "good");
  return (
    <>
      <p className="fit-note">On <code>{rigLabel}</code> · {tp}× {Math.round(vram / 1024)}GB · fp8 KV · 90% util.</p>
      <div className="heatmap">
        <div className="heatmap-grid" style={{ gridTemplateColumns: `40px repeat(${env.contexts.length}, 1fr)` }}>
          <span className="heatmap-corner" />
          {env.contexts.map((c) => <span key={c} className="heatmap-xlabel">{fmtC(c)}</span>)}
          {[...env.grid].reverse().map((row, ri) => {
            const k = [...env.concurrencies].reverse()[ri];
            return (
              <Fragment key={k}>
                <span className="heatmap-ylabel">{k}×</span>
                {row.map((cell) => <div key={cell.context} className={`heatmap-cell ${color(cell.headroom_mib)}`} title={`${fmtC(cell.context)} ctx · ${k} conc → ${cell.fits ? "fits" : "over budget"}`} />)}
              </Fragment>
            );
          })}
        </div>
        <div className="heatmap-legend"><span className="good" /> roomy<span className="ok" /> fits<span className="tight" /> tight<span className="over" /> over</div>
      </div>
    </>
  );
}
