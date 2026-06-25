// SPDX-License-Identifier: Apache-2.0
// Apply-order shadow panel — surfaces sndr.apply.shadow.compare_apply_orders: the
// static diff of the legacy per-patch apply loop vs the spec-driven loop. The
// operator-critical signal is spec_boot_unsafe — patches the legacy loop applies
// that would silently DROP under SNDR_APPLY_VIA_SPECS=1 (a healthy-looking boot
// quietly missing patches). Read-only; self-fetches /api/v1/patches/shadow.
import { CircleAlert, CircleCheck, Loader2, RefreshCw, ShieldAlert } from "lucide-react";
import { tr } from "../i18n";
import { useApiQuery } from "../hooks/useApiQuery";
import { api } from "../api";

function ChipList({ ids, tone }: { ids: string[]; tone: "danger" | "warn" | "neutral" }) {
  return (
    <div className="chip-row">
      {ids.map((id) => <span key={id} className={`chip ${tone === "neutral" ? "" : tone}`}>{id}</span>)}
    </div>
  );
}

export function ApplyShadowPanel() {
  const { data, state, error, reload } = useApiQuery(
    ["patch-shadow"],
    (signal) => api.patchShadow(signal),
    { staleTime: 30_000 },
  );

  if (state === "loading" || state === "idle") {
    return <div className="muted"><Loader2 size={14} /> {tr("Loading apply-order shadow…")}</div>;
  }
  if (state === "error") {
    return <div className="chat-advisory"><CircleAlert size={13} /> <span>{error ?? tr("Failed to load apply-order shadow.")}</span></div>;
  }

  const unsafe = data?.spec_boot_unsafe ?? [];
  const unexpected = data?.spec_only_unexpected ?? [];
  const unparseable = data?.legacy_unparseable ?? [];
  const legacy = data?.legacy_count ?? 0;
  const spec = data?.spec_count ?? 0;

  return (
    <div className="retire-impact">
      <div className="retire-impact-head">
        <span className={unsafe.length > 0 ? "retire-summary danger" : "retire-summary ok"}>
          {unsafe.length > 0
            ? <><ShieldAlert size={13} /> {unsafe.length} {tr("spec-boot-unsafe")}</>
            : <><CircleCheck size={13} /> {tr("no spec-boot-unsafe drops")}</>}
          <span className="muted"> · {tr("legacy")} {legacy} / {tr("spec")} {spec}</span>
        </span>
        <button className="ghost-button" onClick={reload}><RefreshCw size={12} /> {tr("Refresh")}</button>
      </div>
      {data?.error && <div className="muted">{tr("Shadow unavailable")} ({data.error})</div>}

      {unsafe.length > 0 && (
        <div className="shadow-block">
          <span className="guidance-label danger"><ShieldAlert size={13} /> {tr("Would silently drop under spec-driven apply")}</span>
          <ChipList ids={unsafe} tone="danger" />
        </div>
      )}
      {unexpected.length > 0 && (
        <div className="shadow-block">
          <span className="guidance-label warn">{tr("Spec-only & not on the known allowlist")}</span>
          <ChipList ids={unexpected} tone="warn" />
        </div>
      )}
      {unparseable.length > 0 && (
        <div className="shadow-block">
          <span className="guidance-label warn">{tr("Legacy entries the spec loop can't parse")}</span>
          <ChipList ids={unparseable} tone="warn" />
        </div>
      )}
      {unsafe.length === 0 && unexpected.length === 0 && unparseable.length === 0 && !data?.error && (
        <div className="muted">{tr("Legacy and spec-driven apply orders agree — no silent drops.")}</div>
      )}
    </div>
  );
}
