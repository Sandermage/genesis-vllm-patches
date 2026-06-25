import { useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, Copy, DownloadCloud, GitBranch, Loader2, RefreshCw, ShieldCheck } from "lucide-react";
import { api, type UpdateApplyResult, type UpdateCheck, type UpdatePlan, type UpdateStatus } from "./api";
import { tr } from "./i18n";

// Pin-gated self-updater panel: read-only status + plan by default; the apply
// button is gated (daemon apply flag + confirm) and the vLLM pin only ever moves
// to a patcher-supported value. The server docker-pin step stays manual.
export function UpdatesPanel() {
  const [status, setStatus] = useState<UpdateStatus | null>(null);
  const [check, setCheck] = useState<UpdateCheck | null>(null);
  const [plan, setPlan] = useState<UpdatePlan | null>(null);
  const [targetPin, setTargetPin] = useState<string>("");
  const [busy, setBusy] = useState<string | null>(null);
  const [applyResult, setApplyResult] = useState<UpdateApplyResult | null>(null);
  const [confirm, setConfirm] = useState(false);

  async function loadStatus() {
    setBusy("status");
    try { const s = await api.updateStatus(); setStatus(s); setTargetPin(s.canonical_pin ?? ""); }
    catch { /* ignore */ } finally { setBusy(null); }
  }
  useEffect(() => { void loadStatus(); }, []);

  async function runCheck() { setBusy("check"); try { setCheck(await api.updateCheck()); } catch { /* ignore */ } finally { setBusy(null); } }
  async function buildPlan() { setBusy("plan"); setApplyResult(null); try { setPlan(await api.updatePlan(targetPin || undefined)); } catch { /* ignore */ } finally { setBusy(null); } }
  async function applyUpdate() {
    if (!confirm) return;
    setBusy("apply");
    try { setApplyResult(await api.updateApply(true, targetPin || undefined)); await loadStatus(); }
    catch (e) { setApplyResult({ applied: false, status: "error", message: e instanceof Error ? e.message : String(e) }); }
    finally { setBusy(null); setConfirm(false); }
  }

  const g = status?.git;
  return (
    <div className="updates-panel">
      <div className="updates-grid">
        <div className="updates-kv"><span>{tr("Patcher (sndr_core)")}</span><strong>v{status?.sndr_core_version ?? "—"}</strong></div>
        <div className="updates-kv"><span>{tr("Git")}</span><strong>{g?.is_repo ? `${g.branch} @ ${g.commit}${g.dirty ? " · dirty" : ""}` : tr("not a checkout")}</strong></div>
        <div className="updates-kv"><span>{tr("GUI build")}</span><strong>{status?.gui_build?.bundle ?? (status?.gui_build?.published ? tr("published") : "—")}</strong></div>
        <div className="updates-kv"><span>{tr("Apply")}</span><strong className={status?.apply_enabled ? "ok" : "muted"}>{status?.apply_enabled ? tr("enabled") : tr("read-only (SNDR_ENABLE_APPLY=0)")}</strong></div>
      </div>

      <div className="updates-pins">
        <span className="updates-label"><ShieldCheck size={13} /> {tr("Patcher-supported vLLM pins (the only allowed targets)")}</span>
        <div className="updates-pin-list">
          {(status?.supported_pins ?? []).map((p) => (
            <button key={p} className={`updates-pin ${p === targetPin ? "active" : ""}`} onClick={() => setTargetPin(p)} title={tr("Use as the update target pin")}>
              {p === status?.canonical_pin && <CheckCircle2 size={12} />}<code>{p}</code>
            </button>
          ))}
        </div>
      </div>

      <div className="updates-actions">
        <button className="ghost-button" onClick={() => void runCheck()} disabled={busy === "check"}>
          {busy === "check" ? <Loader2 size={14} className="spin" /> : <RefreshCw size={14} />} {tr("Check remote")}
        </button>
        <button className="ghost-button" onClick={() => void buildPlan()} disabled={busy === "plan"}>
          {busy === "plan" ? <Loader2 size={14} className="spin" /> : <GitBranch size={14} />} {tr("Build plan")}
        </button>
        {check && (
          <span className={`updates-check ${check.update_available ? "warn" : "ok"}`}>
            {check.error ? `· ${check.error}` : check.update_available ? `· ${tr("update available (remote")} ${check.remote_commit})` : `· ${tr("up to date")}`}
          </span>
        )}
      </div>

      {plan && (
        <div className="updates-plan">
          <div className={`updates-plan-head ${plan.valid ? "ok" : "blocked"}`}>
            {plan.valid ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
            <strong>{tr("Plan → pin")} {plan.target_pin ?? "—"}</strong>
            {!plan.pin_gate.ok && <span className="updates-gate-fail">{tr("pin gate failed")}</span>}
          </div>
          {!plan.pin_gate.ok && plan.pin_gate.reason && (
            <p className="updates-gate-reason muted">{plan.pin_gate.reason}</p>
          )}
          {plan.blocked_reasons.length > 0 && (
            <ul className="updates-blocked">{plan.blocked_reasons.map((r, i) => <li key={i}><AlertTriangle size={12} /> {r}</li>)}</ul>
          )}
          <ol className="updates-steps">
            {plan.steps.map((s) => (
              <li key={s.order} className={s.kind === "local" ? "step-local" : "step-manual"}>
                <span className="step-kind">{s.kind === "local" ? tr("auto") : tr("manual")}</span>
                <span className="step-title">{s.title}</span>
                <code className="step-cmd">{s.cmd}</code>
                <button className="icon-only" title={tr("Copy command")} onClick={() => void navigator.clipboard?.writeText(s.cmd)}><Copy size={12} /></button>
              </li>
            ))}
          </ol>
          <div className="updates-apply">
            <label className="updates-confirm">
              <input type="checkbox" checked={confirm} onChange={(e) => setConfirm(e.target.checked)} disabled={!plan.valid || !status?.apply_enabled} />
              {tr("I confirm running the local update steps now")}
            </label>
            <button className="primary-action" onClick={() => void applyUpdate()} disabled={!plan.valid || !status?.apply_enabled || !confirm || busy === "apply"}>
              {busy === "apply" ? <Loader2 size={14} className="spin" /> : <DownloadCloud size={14} />} {tr("Apply local update")}
            </button>
          </div>
          {!status?.apply_enabled && <p className="updates-hint">{tr("Apply is disabled — start the daemon with")} <code>SNDR_ENABLE_APPLY=1</code>. {tr("The server vLLM-pin step is always manual (pin policy).")}</p>}
        </div>
      )}

      {applyResult && (
        <div className={`updates-result ${applyResult.applied ? "ok" : "blocked"}`}>
          <strong>{applyResult.status}</strong>{applyResult.message ? ` — ${applyResult.message}` : ""}
          {applyResult.results && (
            <ol className="updates-steps">
              {applyResult.results.map((r) => (
                <li key={r.order} className={r.status === "ok" ? "step-local" : "step-manual"}>
                  <span className="step-kind">{r.status}</span><span className="step-title">{r.title}</span>
                  <code className="step-cmd">exit {r.exit_code}</code>
                </li>
              ))}
            </ol>
          )}
        </div>
      )}
    </div>
  );
}
