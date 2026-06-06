import { useEffect, useState } from "react";
import { BadgeCheck, Box, KeyRound, Lock, Package, RefreshCw, ShieldCheck, ShieldX } from "lucide-react";
import { api, type LicenseStatus } from "./api";

const STATUS_HELP: Record<string, string> = {
  no_package: "The commercial vllm.sndr_engine overlay is not installed.",
  no_key: "sndr_engine is installed but no license key was found (set SNDR_ENGINE_LICENSE_KEY or drop the key file).",
  bad_signature: "The license token's Ed25519 signature did not verify.",
  bad_payload: "The license signature is valid but its payload failed the contract check.",
  expired: "The license token has expired.",
  version_mismatch: "The sndr_engine version is incompatible with this core.",
  licensed: "A valid signed license entitles the engine tier.",
  licensed_legacy: "A plain (unsigned) key is present — legacy entitlement.",
  override: "Engine tier is force-enabled via an operator override.",
};

type ExpiryState = "none" | "valid" | "soon" | "expired";
function expiryInfo(expires: string | null | undefined): { state: ExpiryState; days: number | null; label: string } {
  if (!expires) return { state: "none", days: null, label: "no expiry set" };
  const t = Date.parse(expires);
  if (Number.isNaN(t)) return { state: "none", days: null, label: String(expires) };
  const days = Math.floor((t - Date.now()) / 86_400_000);
  if (days < 0) return { state: "expired", days, label: `expired ${-days}d ago` };
  if (days <= 30) return { state: "soon", days, label: `${days}d remaining` };
  return { state: "valid", days, label: `${days}d remaining` };
}

export function LicensePanel() {
  const [data, setData] = useState<LicenseStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const load = () => { setLoading(true); api.license().then(setData).catch(() => setData(null)).finally(() => setLoading(false)); };
  useEffect(() => { load(); }, []);

  if (data && !data.available) {
    return <div className="lic-empty"><Lock size={20} /><strong>License layer unavailable</strong><span>{data.reason}</span></div>;
  }

  const tier = data?.tier ?? "—";
  const isEngine = data?.eligible === true;
  const eng = data?.engine;
  const lic = data?.license;
  const sig = lic?.signature_valid;

  return (
    <div className="lic">
      <div className="lic-head">
        <div className={`lic-tier ${isEngine ? "engine" : "community"}`}>
          {isEngine ? <BadgeCheck size={18} /> : <Box size={18} />}
          <div className="lic-tier-id">
            <strong>{isEngine ? "SNDR Engine" : "Community"} tier</strong>
            <span>{data?.core ?? "public (unlicensed)"}</span>
          </div>
          <span className={`lic-pill ${isEngine ? "ok" : ""}`}>{tier}</span>
        </div>
        <button className="ghost-button" onClick={load} disabled={loading}>
          {loading ? <RefreshCw size={14} className="spin" /> : <RefreshCw size={14} />} Recheck
        </button>
      </div>

      <div className="lic-grid">
        <div className="lic-card">
          <div className="lic-card-t"><Package size={13} /> sndr_engine overlay</div>
          <div className={`lic-row ${eng?.installed ? "ok" : "off"}`}>
            <span>Installed</span><strong>{eng?.installed ? `yes${eng.version ? ` · v${eng.version}` : ""}` : "no"}</strong>
          </div>
          <div className="lic-row"><span>Module</span><strong className="mono">{eng?.module ?? "—"}</strong></div>
          {!eng?.installed && <div className="lic-hint">Install the commercial <code>vllm-sndr-engine</code> package to unlock engine-tier kernels/patches.</div>}
        </div>

        <div className="lic-card">
          <div className="lic-card-t"><KeyRound size={13} /> License token</div>
          <div className="lic-row"><span>Subject</span><strong className="mono">{lic?.subject ?? "—"}</strong></div>
          <div className="lic-row"><span>Expires</span><strong className="mono">{lic?.expires ?? "—"}</strong></div>
          {(() => {
            const ei = expiryInfo(lic?.expires);
            if (ei.state === "none") return null;
            const pct = ei.days === null ? 0 : Math.max(2, Math.min(100, Math.round((ei.days / 365) * 100)));
            return (
              <div className={`lic-expiry ${ei.state}`}>
                <div className="lic-expiry-bar"><span style={{ width: `${ei.state === "expired" ? 100 : pct}%` }} /></div>
                <span className="lic-expiry-label">{ei.label}</span>
                {ei.state === "soon" && <span className="lic-hint">Renew the license token soon to keep the engine tier entitled.</span>}
                {ei.state === "expired" && <span className="lic-hint">Token expired — engine-tier patches are gated off until renewed.</span>}
              </div>
            );
          })()}
          <div className={`lic-row ${sig === true ? "ok" : sig === false ? "bad" : ""}`}>
            <span>Signature</span>
            <strong>{sig === true ? <><ShieldCheck size={12} /> valid</> : sig === false ? <><ShieldX size={12} /> invalid</> : "—"}</strong>
          </div>
        </div>

        <div className="lic-card">
          <div className="lic-card-t"><ShieldCheck size={13} /> Entitlement</div>
          <div className={`lic-row ${isEngine ? "ok" : "off"}`}><span>Engine tier</span><strong>{isEngine ? "enabled" : "locked"}</strong></div>
          <div className="lic-row"><span>Premium patches on</span><strong className="mono">{data?.premium_patches_enabled ?? 0}</strong></div>
          <div className="lic-row"><span>Engine-tier patches</span><strong className="mono">{data?.engine_tier_patches ?? 0}</strong></div>
        </div>
      </div>

      {data?.status && (
        <div className={`lic-status ${isEngine ? "ok" : "locked"}`}>
          <span className="lic-status-code">{data.status}</span>
          <span className="lic-status-msg">{STATUS_HELP[data.status] ?? data.reason}</span>
        </div>
      )}
    </div>
  );
}
