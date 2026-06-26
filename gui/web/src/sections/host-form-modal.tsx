// SPDX-License-Identifier: Apache-2.0
// Add/edit modal for a remote host profile (label, host/SSH, ports, engine key,
// SSH auth, hardware, tags). The dialog carries an aria-label so assistive tech
// announces add-vs-edit context.
import { useRef, useState } from "react";
import { Server, AlertCircle } from "lucide-react";
import { api, type HostProfile } from "../api";
import { useDialogFocus, useEscapeKey, closeOnBackdrop } from "../dialog";
import { toast } from "../components/toast";
import { tr } from "../i18n";

const HOST_ROLES = ["production", "staging", "dev", "experiment"] as const;

export function HostFormModal({
  initial,
  onClose,
  onSaved
}: {
  initial: HostProfile | null;
  onClose: () => void;
  onSaved: () => void;
}) {
  const blank = { label: "", host: "", ssh_target: "", port: 8765, engine_port: 8000, api_key: "", ssh_user: "", ssh_auth: "agent", ssh_key_path: "", ssh_port: 22, ssh_password: "", role: "", hardware: "", gpus: 0, notes: "", tags: "" };
  const [form, setForm] = useState(initial
    ? { label: initial.label, host: initial.host, ssh_target: initial.ssh_target, port: initial.port, engine_port: initial.engine_port, api_key: "", ssh_user: initial.ssh_user, ssh_auth: initial.ssh_auth || "agent", ssh_key_path: initial.ssh_key_path, ssh_port: initial.ssh_port || 22, ssh_password: "", role: initial.role, hardware: initial.hardware, gpus: initial.gpus, notes: initial.notes, tags: initial.tags.join(", ") }
    : blank);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dialogRef = useRef<HTMLElement>(null);
  useDialogFocus(dialogRef);
  useEscapeKey(onClose);
  const set = (patch: Partial<typeof form>) => setForm((prev) => ({ ...prev, ...patch }));
  async function save() {
    setBusy(true);
    setError(null);
    try {
      const saved = await api.hostUpsert({
        ...(initial ? { id: initial.id } : {}),
        label: form.label,
        host: form.host,
        ssh_target: form.ssh_target,
        transport: form.ssh_target || form.ssh_user ? "ssh" : "local",
        port: Number(form.port),
        engine_port: Number(form.engine_port),
        // Only send the engine key when the operator actually typed one — a
        // blank field means "keep the stored key" (it's never pre-filled, since
        // the daemon never returns it), so editing a host can't silently wipe it.
        ...(form.api_key ? { api_key: form.api_key } : {}),
        ssh_user: form.ssh_user,
        ssh_auth: form.ssh_auth,
        ssh_key_path: form.ssh_key_path,
        ssh_port: Number(form.ssh_port) || 22,
        role: form.role,
        hardware: form.hardware,
        gpus: Number(form.gpus),
        notes: form.notes,
        tags: form.tags.split(",").map((tag) => tag.trim()).filter(Boolean)
      });
      // A typed SSH password is persisted (encrypted) via the check endpoint,
      // never through the plaintext profile.
      if (form.ssh_auth === "password" && form.ssh_password) {
        try { await api.sshCheck({ host: saved.host, host_id: saved.id, user: form.ssh_user, auth_method: "password", password: form.ssh_password, ssh_port: Number(form.ssh_port) || 22 }); } catch { /* surfaced on the card's SSH check */ }
      }
      onSaved();
      onClose();
      toast(initial ? `${tr("Host updated:")} ${form.label}` : `${tr("Host added:")} ${form.label}`, "success");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      toast(tr("Failed to save host"), "error");
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="dialog-backdrop" role="presentation" onClick={closeOnBackdrop(onClose)}>
      <section ref={dialogRef} className="host-modal" role="dialog" aria-modal="true" aria-label={initial ? `${tr("Edit host profile")} ${initial.label}` : tr("Add host profile")}>
        <div className="module-card-title">
          <Server size={18} />
          <h2>{initial ? `${tr("Edit")} ${initial.label}` : tr("Add host profile")}</h2>
        </div>
        <div className="host-form-grid">
          <label className="param-field"><span>{tr("Label")}</span><input value={form.label} onChange={(e) => set({ label: e.target.value })} placeholder="Prod A5000" /></label>
          <label className="param-field"><span>{tr("Host / IP")}</span><input value={form.host} onChange={(e) => set({ host: e.target.value })} placeholder="192.0.2.10" /></label>
          <label className="param-field"><span>{tr("SSH target")}</span><input value={form.ssh_target} onChange={(e) => set({ ssh_target: e.target.value })} placeholder="user@192.0.2.10" /></label>
          <label className="param-field"><span>{tr("Role")}</span>
            <select value={form.role} onChange={(e) => set({ role: e.target.value })}>
              <option value="">{tr("— none —")}</option>
              {HOST_ROLES.map((role) => <option key={role} value={role}>{role}</option>)}
            </select>
          </label>
          <label className="param-field"><span>{tr("GUI port")}</span><input type="number" value={form.port} onChange={(e) => set({ port: Number(e.target.value) })} /></label>
          <label className="param-field"><span>{tr("Engine port")}</span><input type="number" value={form.engine_port} onChange={(e) => set({ engine_port: Number(e.target.value) })} /></label>
          <label className="param-field"><span>{tr("Hardware")}</span><input value={form.hardware} onChange={(e) => set({ hardware: e.target.value })} placeholder="2× A5000 24GB" /></label>
          <label className="param-field"><span>GPUs</span><input type="number" value={form.gpus} onChange={(e) => set({ gpus: Number(e.target.value) })} /></label>
        </div>
        <label className="param-field"><span>{tr("Engine API key")}{initial?.has_api_key ? tr(" (stored — leave blank to keep)") : tr(" (optional)")}</span><input type="password" value={form.api_key} onChange={(e) => set({ api_key: e.target.value })} placeholder={initial?.has_api_key ? tr("•••••• stored — type to replace") : tr("if the engine needs one — e.g. genesis-local")} autoComplete="off" spellCheck={false} /></label>
        <div className="host-form-grid host-ssh-grid">
          <label className="param-field"><span>{tr("SSH user")}</span><input value={form.ssh_user} onChange={(e) => set({ ssh_user: e.target.value })} placeholder="user" /></label>
          <label className="param-field"><span>{tr("SSH port")}</span><input type="number" value={form.ssh_port} onChange={(e) => set({ ssh_port: Number(e.target.value) || 22 })} /></label>
          <label className="param-field"><span>{tr("SSH auth")}</span>
            <select value={form.ssh_auth} onChange={(e) => set({ ssh_auth: e.target.value })}>
              <option value="agent">{tr("ssh-agent / default keys")}</option>
              <option value="key">{tr("private key file")}</option>
              <option value="password">{tr("password")}</option>
            </select>
          </label>
          {form.ssh_auth === "key" && <label className="param-field"><span>{tr("Private key path")}</span><input value={form.ssh_key_path} onChange={(e) => set({ ssh_key_path: e.target.value })} placeholder="~/.ssh/id_ed25519" spellCheck={false} /></label>}
          {form.ssh_auth === "password" && <label className="param-field"><span>{tr("SSH password")} {initial ? tr("(stored, blank = keep)") : ""}</span><input type="password" value={form.ssh_password} onChange={(e) => set({ ssh_password: e.target.value })} placeholder={tr("encrypted at rest")} autoComplete="off" spellCheck={false} /></label>}
        </div>
        <label className="param-field"><span>{tr("Tags (comma-separated)")}</span><input value={form.tags} onChange={(e) => set({ tags: e.target.value })} placeholder="27b, tq-k8v4" /></label>
        <label className="param-field"><span>{tr("Notes")}</span><input value={form.notes} onChange={(e) => set({ notes: e.target.value })} placeholder="MTP K=3 / Wave 8" /></label>
        {error && <div className="inline-error"><AlertCircle size={15} /> {error}</div>}
        <div className="host-modal-actions">
          <button className="ghost-button" onClick={onClose}>{tr("Cancel")}</button>
          <button className="primary-action" onClick={() => void save()} disabled={busy || !(form.label || form.host)}>
            <Server size={15} /> {busy ? tr("Saving…") : initial ? tr("Save changes") : tr("Add profile")}
          </button>
        </div>
      </section>
    </div>
  );
}
