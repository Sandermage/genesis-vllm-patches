// SPDX-License-Identifier: Apache-2.0
// Transient toast notifications: a fire-and-forget `toast()` dispatcher and the
// `ToastHost` that renders + auto-dismisses them. Extracted from App.tsx
// (modularization) with no behavior change.
import { useEffect, useState } from "react";
import { CheckCircle2, AlertCircle, Activity, X } from "lucide-react";
import { tr } from "../i18n";

export type ToastTone = "info" | "success" | "error";

export function toast(message: string, tone: ToastTone = "info") {
  window.dispatchEvent(new CustomEvent("sndr-toast", { detail: { message, tone, id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}` } }));
}

export function ToastHost() {
  const [items, setItems] = useState<Array<{ id: string; message: string; tone: ToastTone }>>([]);
  useEffect(() => {
    const onToast = (event: Event) => {
      const detail = (event as CustomEvent).detail as { id: string; message: string; tone: ToastTone };
      setItems((prev) => [...prev.slice(-3), detail]);
      // Errors linger long enough to actually read a failure; transient
      // success/info notices clear quickly.
      const ttl = detail.tone === "error" ? 8000 : 4200;
      window.setTimeout(() => setItems((prev) => prev.filter((item) => item.id !== detail.id)), ttl);
    };
    window.addEventListener("sndr-toast", onToast);
    return () => window.removeEventListener("sndr-toast", onToast);
  }, []);
  if (items.length === 0) return null;
  return (
    <div className="toast-host" role="region" aria-label={tr("Notifications")}>
      {items.map((item) => (
        // Errors announce assertively (role=alert); success/info politely
        // (role=status). The per-toast role carries the right aria-live, so the
        // container itself stays a plain labelled region.
        <div key={item.id} className={`toast toast-${item.tone}`} role={item.tone === "error" ? "alert" : "status"} aria-atomic="true">
          {item.tone === "success" ? <CheckCircle2 size={15} /> : item.tone === "error" ? <AlertCircle size={15} /> : <Activity size={15} />}
          <span>{item.message}</span>
          <button className="icon-only" onClick={() => setItems((prev) => prev.filter((x) => x.id !== item.id))} aria-label={tr("Dismiss")}><X size={13} /></button>
        </div>
      ))}
    </div>
  );
}
