import { useEffect, useRef, useState } from "react";
import { AlertTriangle, Loader2, SendHorizontal, Server, ShieldAlert, Sparkles, Trash2, Wrench } from "lucide-react";
import { api, type CopilotProposedAction, type CopilotStep, type CopilotTool, type HostProfile } from "./api";
import { tr } from "./i18n";

type Msg = { role: "user" | "assistant"; content: string; steps?: CopilotStep[]; proposed?: CopilotProposedAction[]; stopped?: string };

// Cap retained + sent conversation so a long session can't grow memory or the
// per-request payload without bound (the ChatConsole caps its API payload too).
const MAX_HISTORY = 40;

const SUGGESTIONS = [
  "How many presets are there, and which are production?",
  "Run the doctor — is anything wrong with this host?",
  "Will qwen3.6-27b-int4 fit 32K context on 2× A5000 with fp8 KV?",
  "List the active patches in the spec-decode family.",
];

// Scoped ops-copilot: a read-only tool-calling assistant over the Product API.
// It calls read/dry-run tools server-side and cites real numbers; for changes it
// surfaces a *proposed action* the operator reviews & applies in the gated UI —
// the copilot never mutates anything itself.
export function CopilotPanel({ onNavigate }: { onNavigate: (section: string, params?: Record<string, unknown>) => void }) {
  const [tools, setTools] = useState<CopilotTool[]>([]);
  const [hosts, setHosts] = useState<HostProfile[]>([]);
  const [target, setTarget] = useState<{ hostId: string; host?: string; port?: number }>({ hostId: "" });
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [showTools, setShowTools] = useState(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    api.copilotTools().then((t) => setTools(t.tools)).catch(() => {});
    api.hosts().then((h) => setHosts(h.hosts)).catch(() => {});
  }, []);
  useEffect(() => { const el = scrollRef.current; if (el) el.scrollTop = el.scrollHeight; }, [msgs, busy]);

  function pickHost(id: string) {
    const h = hosts.find((x) => x.id === id);
    setTarget(h ? { hostId: id, host: h.host, port: h.engine_port || 8000 } : { hostId: "" });
  }

  async function send(text?: string) {
    const q = (text ?? input).trim();
    if (!q || busy) return;
    const history: Msg[] = [...msgs, { role: "user" as const, content: q }].slice(-MAX_HISTORY);
    setMsgs(history); setInput(""); setBusy(true); setErr(null);
    try {
      const res = await api.copilotChat(
        history.map((m) => ({ role: m.role, content: m.content })),
        { host: target.host, port: target.port, host_id: target.hostId || undefined },
      );
      setMsgs((m) => [...m, { role: "assistant" as const, content: res.reply, steps: res.steps, proposed: res.proposed_actions, stopped: res.stopped }].slice(-MAX_HISTORY));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally { setBusy(false); }
  }

  return (
    <div className="copilot">
      <div className="copilot-bar">
        <label className="param-field"><span><Server size={11} /> {tr("Engine")}</span>
          <select value={target.hostId} onChange={(e) => pickHost(e.target.value)}>
            <option value="">{tr("Daemon-local engine")}</option>
            {hosts.map((h) => <option key={h.id} value={h.id}>{h.label} · {h.host}:{h.engine_port || 8000}</option>)}
          </select>
        </label>
        <button className="ghost-button" onClick={() => setShowTools((v) => !v)} title={tr("What the copilot can call (all read-only / dry-run)")}>
          <Wrench size={13} /> {tools.length} {tr("tools")}
        </button>
        {msgs.length > 0 && <button className="ghost-button" onClick={() => { setMsgs([]); setErr(null); }}><Trash2 size={13} /> {tr("Clear")}</button>}
        <span className="copilot-readonly"><ShieldAlert size={12} /> {tr("read-only")}</span>
      </div>

      {showTools && (
        <div className="copilot-tools">
          {tools.map((t) => (
            <div key={t.name} className="copilot-tool">
              <div className="copilot-tool-head"><code>{t.name}</code><span className={`copilot-tool-cat ${t.category}`}>{t.category === "plan" ? tr("dry-run") : tr("read")}</span></div>
              <p>{t.description}</p>
            </div>
          ))}
        </div>
      )}

      <div className="copilot-scroll" ref={scrollRef}>
        {msgs.length === 0 && (
          <div className="copilot-empty">
            <Sparkles size={24} />
            <strong>{tr("Ops Copilot")}</strong>
            <span>{tr("Read-only assistant for this Genesis stack. Ask about presets, patches, capacity or health — it calls the Product API and cites real numbers. It")} <b>{tr("proposes")}</b> {tr("changes; you apply them.")}</span>
            <div className="copilot-suggest">{SUGGESTIONS.map((s) => <button key={s} onClick={() => void send(s)}>{tr(s)}</button>)}</div>
          </div>
        )}
        {msgs.map((m, i) => (
          <div key={i} className={`copilot-msg ${m.role}`}>
            {m.content && <div className="copilot-msg-body">{m.content}</div>}
            {m.steps && m.steps.length > 0 && (
              <div className="copilot-trace">
                <span className="copilot-trace-lbl"><Wrench size={11} /> {tr("called")} {m.steps.length} {m.steps.length > 1 ? tr("tools") : tr("tool")}</span>
                {m.steps.map((s, j) => <span key={j} className={`copilot-step ${s.ok ? "" : "err"}`} title={JSON.stringify(s.args)}>{s.tool}</span>)}
              </div>
            )}
            {m.proposed && m.proposed.map((a, j) => (
              <div key={j} className="copilot-action">
                <ShieldAlert size={14} />
                <span className="copilot-action-label">{a.label}</span>
                <button className="primary-action" onClick={() => onNavigate(a.section, a.params)}>{tr("Review & apply")} →</button>
              </div>
            ))}
            {m.stopped === "max_steps" && <span className="copilot-note">{tr("stopped at the tool-call limit — ask a narrower question for more depth.")}</span>}
          </div>
        ))}
        {busy && <div className="copilot-msg assistant"><div className="copilot-msg-body copilot-thinking"><Loader2 size={14} className="spin" /> {tr("thinking…")}</div></div>}
      </div>

      {err && <div className="copilot-err"><AlertTriangle size={13} /> {err}</div>}
      <div className="copilot-input">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void send(); } }}
          placeholder={tr("Ask the ops copilot…  (Enter to send, Shift+Enter for newline)")}
          rows={2} spellCheck={false}
        />
        <button className="primary-action" onClick={() => void send()} disabled={!input.trim() || busy}>
          {busy ? <Loader2 size={14} className="spin" /> : <SendHorizontal size={14} />} {tr("Send")}
        </button>
      </div>
    </div>
  );
}
