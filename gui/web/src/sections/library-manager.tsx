// SPDX-License-Identifier: Apache-2.0
// Prompts & Tools manager — OpenWebUI-style: add/edit/delete named prompt
// templates and declarative (SSRF-safe) HTTP tools that the copilot can call.
import { useEffect, useState } from "react";
import { X, Plus, Trash2, Pencil, BookText, Wrench, Save } from "lucide-react";
import { api, type Prompt, type ManagedTool, type ManagedToolParam } from "../api";
import { tr } from "../i18n";

type PromptForm = { id?: string; name: string; title: string; content: string };
type ToolForm = { id?: string; existing?: boolean; name: string; title: string; description: string; method: "GET" | "POST"; url: string; params: ManagedToolParam[]; enabled: boolean };

export function LibraryManager({ onClose, onChanged }: { onClose: () => void; onChanged?: () => void }) {
  const [tab, setTab] = useState<"prompts" | "tools">("prompts");
  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [onClose]);
  return (
    <div className="dialog-backdrop" role="presentation" onClick={onClose}>
      {/* eslint-disable-next-line jsx-a11y/click-events-have-key-events, jsx-a11y/no-noninteractive-element-interactions -- stops backdrop-close on inside clicks; Escape closes via the keydown listener */}
      <section className="library-dialog" role="dialog" aria-modal="true" aria-label={tr("Prompts & Tools")} onClick={(e) => e.stopPropagation()}>
        <div className="library-head">
          <div className="library-tabs">
            <button className={tab === "prompts" ? "active" : ""} onClick={() => setTab("prompts")}><BookText size={14} /> {tr("Prompts")}</button>
            <button className={tab === "tools" ? "active" : ""} onClick={() => setTab("tools")}><Wrench size={14} /> {tr("Tools")}</button>
          </div>
          <button className="icon-only" onClick={onClose} title={tr("Close")}><X size={16} /></button>
        </div>
        {tab === "prompts" ? <PromptManager onChanged={onChanged} /> : <ToolManager />}
      </section>
    </div>
  );
}

function PromptManager({ onChanged }: { onChanged?: () => void }) {
  const [items, setItems] = useState<Prompt[]>([]);
  const [edit, setEdit] = useState<PromptForm | null>(null);
  const [err, setErr] = useState("");
  const load = () => api.listPrompts().then((r) => setItems(r.prompts)).catch(() => {});
  useEffect(() => { load(); }, []);
  async function save() {
    if (!edit) return;
    try {
      if (edit.id) await api.updatePrompt(edit.id, { name: edit.name, title: edit.title, content: edit.content });
      else await api.createPrompt({ name: edit.name, title: edit.title, content: edit.content });
      setEdit(null); setErr(""); load(); onChanged?.();
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
  }
  async function del(id: string) { await api.deletePrompt(id); load(); onChanged?.(); }
  return (
    <div className="library-body">
      <div className="library-list">
        <button className="primary-action library-add" onClick={() => setEdit({ name: "", title: "", content: "" })}><Plus size={14} /> {tr("New prompt")}</button>
        {items.map((p) => (
          <div key={p.id} className="library-row">
            <div className="library-row-main"><strong>{p.name}</strong><span className="muted">{p.title || p.id}</span></div>
            {!p.builtin && <button className="icon-only" title={tr("Edit")} onClick={() => setEdit({ id: p.id, name: p.name, title: p.title, content: p.content })}><Pencil size={13} /></button>}
            {p.builtin ? <span className="library-badge">{tr("built-in")}</span> : <button className="icon-only" title={tr("Delete")} onClick={() => void del(p.id)}><Trash2 size={13} /></button>}
          </div>
        ))}
      </div>
      {edit && (
        <div className="library-form">
          <label className="library-form-wide"><span>{tr("Name")}</span><input value={edit.name} onChange={(e) => setEdit({ ...edit, name: e.target.value })} placeholder="Crypto market analyst" /></label>
          <label className="library-form-wide"><span>{tr("Title")}</span><input value={edit.title} onChange={(e) => setEdit({ ...edit, title: e.target.value })} /></label>
          <label className="library-form-wide"><span>{tr("Content (system prompt)")}</span><textarea rows={10} value={edit.content} onChange={(e) => setEdit({ ...edit, content: e.target.value })} /></label>
          {err && <div className="library-err">{err}</div>}
          <div className="library-form-actions">
            <button className="ghost-button" onClick={() => { setEdit(null); setErr(""); }}>{tr("Cancel")}</button>
            <button className="primary-action" onClick={() => void save()}><Save size={14} /> {tr("Save")}</button>
          </div>
        </div>
      )}
    </div>
  );
}

function ToolManager() {
  const [items, setItems] = useState<ManagedTool[]>([]);
  const [edit, setEdit] = useState<ToolForm | null>(null);
  const [err, setErr] = useState("");
  const load = () => api.listManagedTools().then((r) => setItems(r.tools)).catch(() => {});
  useEffect(() => { load(); }, []);
  async function save() {
    if (!edit) return;
    const body = { name: edit.name, title: edit.title, description: edit.description, method: edit.method, url: edit.url, params: edit.params, enabled: edit.enabled };
    try {
      if (edit.existing && edit.id) await api.updateManagedTool(edit.id, body);
      else await api.createManagedTool(body);
      setEdit(null); setErr(""); load();
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
  }
  async function toggle(t: ManagedTool) { await api.updateManagedTool(t.id, { enabled: !t.enabled }); load(); }
  async function del(id: string) { await api.deleteManagedTool(id); load(); }
  function setParam(i: number, patch: Partial<ManagedToolParam>) {
    if (!edit) return;
    const ps = edit.params.slice();
    const cur = ps[i];
    if (cur) { ps[i] = { ...cur, ...patch }; setEdit({ ...edit, params: ps }); }
  }
  return (
    <div className="library-body">
      <div className="library-list">
        <button className="primary-action library-add" onClick={() => setEdit({ name: "", title: "", description: "", method: "GET", url: "", params: [], enabled: true })}><Plus size={14} /> {tr("New tool")}</button>
        {items.map((t) => (
          <div key={t.id} className="library-row">
            <div className="library-row-main"><strong>{t.name}</strong><span className="muted">{t.method} {t.url}</span></div>
            <input type="checkbox" className="library-toggle" title={tr("Enabled")} aria-label={tr("Enabled")} checked={t.enabled} onChange={() => void toggle(t)} />
            <button className="icon-only" title={tr("Edit")} onClick={() => setEdit({ id: t.id, existing: true, name: t.name, title: t.title, description: t.description, method: t.method, url: t.url, params: t.params.map((p) => ({ ...p })), enabled: t.enabled })}><Pencil size={13} /></button>
            <button className="icon-only" title={tr("Delete")} onClick={() => void del(t.id)}><Trash2 size={13} /></button>
          </div>
        ))}
      </div>
      {edit && (
        <div className="library-form">
          <label><span>{tr("Name (a-z0-9_)")}</span><input value={edit.name} onChange={(e) => setEdit({ ...edit, name: e.target.value })} placeholder="coingecko_price" disabled={edit.existing} /></label>
          <label><span>{tr("Method")}</span><select value={edit.method} onChange={(e) => setEdit({ ...edit, method: e.target.value as "GET" | "POST" })}><option>GET</option><option>POST</option></select></label>
          <label className="library-form-wide"><span>{tr("Title")}</span><input value={edit.title} onChange={(e) => setEdit({ ...edit, title: e.target.value })} /></label>
          <label className="library-form-wide"><span>{tr("Description (what the model sees)")}</span><input value={edit.description} onChange={(e) => setEdit({ ...edit, description: e.target.value })} /></label>
          <label className="library-form-wide"><span>{tr("URL template — {param} placeholders, host is fixed")}</span><input value={edit.url} onChange={(e) => setEdit({ ...edit, url: e.target.value })} placeholder="https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd" spellCheck={false} /></label>
          <div className="library-params library-form-wide">
            <div className="library-params-head"><span>{tr("Parameters")}</span><button className="ghost-button" onClick={() => setEdit({ ...edit, params: [...edit.params, { name: "", type: "string" }] })}><Plus size={12} /> {tr("Add")}</button></div>
            {edit.params.map((p, i) => (
              <div key={i} className="library-param">
                <input placeholder={tr("name")} value={p.name} onChange={(e) => setParam(i, { name: e.target.value })} />
                <select value={p.type} onChange={(e) => setParam(i, { type: e.target.value as ManagedToolParam["type"] })}><option>string</option><option>integer</option><option>number</option><option>boolean</option></select>
                <label className="library-req"><input type="checkbox" checked={!!p.required} onChange={(e) => setParam(i, { required: e.target.checked })} /> {tr("req")}</label>
                <input placeholder={tr("description")} value={p.description ?? ""} onChange={(e) => setParam(i, { description: e.target.value })} />
                <button className="icon-only" onClick={() => setEdit({ ...edit, params: edit.params.filter((_, j) => j !== i) })}><Trash2 size={12} /></button>
              </div>
            ))}
          </div>
          {err && <div className="library-err">{err}</div>}
          <div className="library-form-actions">
            <button className="ghost-button" onClick={() => { setEdit(null); setErr(""); }}>{tr("Cancel")}</button>
            <button className="primary-action" onClick={() => void save()}><Save size={14} /> {tr("Save")}</button>
          </div>
        </div>
      )}
    </div>
  );
}
