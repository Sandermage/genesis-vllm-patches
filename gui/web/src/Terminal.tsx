import { useEffect, useRef, useState } from "react";
import { Terminal as XTerm } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
import "@xterm/xterm/css/xterm.css";
import { AlertTriangle, Maximize2, Minus, SquareTerminal, X } from "lucide-react";
import { getApiBase, getApiToken, type HostProfile } from "./api";

function terminalWsUrl(hostId: string): string {
  let u: URL;
  try { u = new URL(getApiBase(), window.location.href); } catch { u = new URL(window.location.href); }
  const proto = u.protocol === "https:" ? "wss:" : "ws:";
  const token = getApiToken();
  const q = token ? `?token=${encodeURIComponent(token)}` : "";
  return `${proto}//${u.host}/api/v1/hosts/${encodeURIComponent(hostId)}/terminal${q}`;
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

type WinBox = { x: number; y: number; w: number; h: number };

// Enterprise-style floating SSH terminal: draggable by the title bar, resizable
// from the edges/corner, maximize/restore, real xterm.js over a paramiko PTY
// websocket. Resizing reflows the PTY (FitAddon → server resize). The remote
// shell is apply-gated server-side; the gate message surfaces as a banner.
export function TerminalModal({ host, onClose }: { host: HostProfile; onClose: () => void }) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const termRef = useRef<XTerm | null>(null);
  const fitRef = useRef<FitAddon | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [state, setState] = useState<"connecting" | "open" | "closed" | "error">("connecting");
  const [banner, setBanner] = useState<string | null>(null);
  const [dims, setDims] = useState<{ cols: number; rows: number } | null>(null);

  const initial = (): WinBox => {
    const w = clamp(960, 480, window.innerWidth - 40);
    const h = clamp(600, 280, window.innerHeight - 80);
    return { x: Math.max(20, (window.innerWidth - w) / 2), y: Math.max(20, (window.innerHeight - h) / 3), w, h };
  };
  const [box, setBox] = useState<WinBox>(initial);
  const [maximized, setMaximized] = useState(false);
  const restoreRef = useRef<WinBox | null>(null);

  function sendResize() {
    const fit = fitRef.current, term = termRef.current, ws = wsRef.current;
    if (!fit || !term) return;
    try { fit.fit(); } catch { /* not laid out */ }
    setDims({ cols: term.cols, rows: term.rows });
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "resize", cols: term.cols, rows: term.rows }));
  }

  // Connect once.
  useEffect(() => {
    const el = mountRef.current;
    if (!el) return;
    const term = new XTerm({
      // A concrete font stack — xterm's renderer measures glyphs directly and
      // can't resolve a CSS var(), so passing one falls back to an ugly generic.
      fontFamily: '"JetBrains Mono", ui-monospace, "SF Mono", "SFMono-Regular", "Cascadia Code", Menlo, Consolas, "Liberation Mono", monospace',
      fontSize: 13, fontWeight: 400, fontWeightBold: 600, lineHeight: 1.35, letterSpacing: 0,
      cursorBlink: true, cursorStyle: "bar", cursorWidth: 2, scrollback: 10000,
      allowProposedApi: true, drawBoldTextInBrightColors: true, macOptionIsMeta: true,
      // Curated 16-colour palette (GitHub-dark-dimmed feel) so command output is
      // crisp and professional instead of xterm's saturated defaults.
      theme: {
        background: "#0c0f16", foreground: "#d7dce5",
        cursor: "#5ad6c0", cursorAccent: "#0c0f16",
        selectionBackground: "#2a4a63", selectionForeground: "#ffffff",
        black: "#3b4252", red: "#f87171", green: "#7dd09a", yellow: "#e6c07b",
        blue: "#79b8ff", magenta: "#c792ea", cyan: "#5ad6c0", white: "#d7dce5",
        brightBlack: "#5c6370", brightRed: "#ff8a8a", brightGreen: "#9ee6b4", brightYellow: "#f2d98d",
        brightBlue: "#9cd0ff", brightMagenta: "#dab8f4", brightCyan: "#7fe6d4", brightWhite: "#ffffff",
      },
    });
    const fit = new FitAddon();
    term.loadAddon(fit);
    term.open(el);
    termRef.current = term; fitRef.current = fit;
    setTimeout(() => { try { fit.fit(); } catch { /* ignore */ } }, 50);

    const ws = new WebSocket(terminalWsUrl(host.id));
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;
    ws.onopen = () => { setState("open"); term.focus(); sendResize(); };
    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        try {
          const m = JSON.parse(ev.data);
          if (m.type === "error") { term.write(`\r\n\x1b[31m✗ ${m.data}\x1b[0m\r\n`); setState("error"); setBanner(m.data); }
          else if (m.type === "ready") term.write(`\x1b[38;5;43m• connected to ${m.data}\x1b[0m\r\n`);
        } catch { /* ignore */ }
      } else { term.write(new Uint8Array(ev.data)); }
    };
    ws.onclose = () => { setState((s) => (s === "error" ? s : "closed")); term.write("\r\n\x1b[90m• session closed\x1b[0m\r\n"); };
    ws.onerror = () => { setState("error"); };
    const dataSub = term.onData((d) => { if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "input", data: d })); });

    return () => { dataSub.dispose(); try { ws.close(); } catch { /* ignore */ } term.dispose(); termRef.current = null; fitRef.current = null; wsRef.current = null; };
  }, [host.id]);

  // Refit the PTY whenever the body changes size (drag-resize, maximize, window).
  useEffect(() => {
    const el = mountRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => sendResize());
    ro.observe(el);
    const onWin = () => sendResize();
    window.addEventListener("resize", onWin);
    return () => { ro.disconnect(); window.removeEventListener("resize", onWin); };
  }, []);

  function beginDrag(e: React.PointerEvent, mode: "move" | "resize" | "resize-e" | "resize-s") {
    if (maximized) return;
    e.preventDefault();
    const sx = e.clientX, sy = e.clientY;
    const o = box;
    const onMove = (ev: PointerEvent) => {
      const dx = ev.clientX - sx, dy = ev.clientY - sy;
      if (mode === "move") {
        setBox({ ...o, x: clamp(o.x + dx, -o.w + 120, window.innerWidth - 120), y: clamp(o.y + dy, 0, window.innerHeight - 44) });
      } else {
        const w = mode === "resize-s" ? o.w : Math.max(480, Math.min(o.w + dx, window.innerWidth - o.x));
        const h = mode === "resize-e" ? o.h : Math.max(280, Math.min(o.h + dy, window.innerHeight - o.y));
        setBox({ ...o, w, h });
      }
    };
    const onUp = () => { window.removeEventListener("pointermove", onMove); window.removeEventListener("pointerup", onUp); };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }

  function toggleMax() {
    if (maximized) { setBox(restoreRef.current ?? initial()); setMaximized(false); }
    else { restoreRef.current = box; setBox({ x: 12, y: 12, w: window.innerWidth - 24, h: window.innerHeight - 24 }); setMaximized(true); }
  }

  const dot = state === "open" ? "ok" : state === "error" ? "down" : state === "closed" ? "muted" : "warn";
  const style = maximized
    ? { left: 12, top: 12, width: window.innerWidth - 24, height: window.innerHeight - 24 }
    : { left: box.x, top: box.y, width: box.w, height: box.h };

  return (
    <div className="terminal-window" style={style} role="dialog" aria-modal="false" aria-label={`Terminal ${host.label}`}>
      <header className="terminal-titlebar" onPointerDown={(e) => beginDrag(e, "move")} onDoubleClick={toggleMax}>
        <span className="terminal-traffic"><i /><i /><i /></span>
        <SquareTerminal size={15} />
        <strong>{host.label}</strong>
        <span className="terminal-target">{(host.ssh_user || "ssh") + "@" + host.host}{host.ssh_port && host.ssh_port !== 22 ? `:${host.ssh_port}` : ""}</span>
        <span className={`terminal-state ${dot}`}><span className="terminal-dot" />{state}</span>
        <span className="terminal-winbtns">
          <button className="terminal-winbtn" onPointerDown={(e) => e.stopPropagation()} onClick={() => sendResize()} title="Refit"><Minus size={14} /></button>
          <button className="terminal-winbtn" onPointerDown={(e) => e.stopPropagation()} onClick={toggleMax} title={maximized ? "Restore" : "Maximize"}><Maximize2 size={14} /></button>
          <button className="terminal-winbtn close" onPointerDown={(e) => e.stopPropagation()} onClick={onClose} title="Close"><X size={15} /></button>
        </span>
      </header>
      {banner && <div className="terminal-banner"><AlertTriangle size={14} /> {banner}</div>}
      <div className="terminal-body" ref={mountRef} onPointerDown={() => termRef.current?.focus()} />
      <footer className="terminal-footer">
        <span className={`terminal-foot-state ${dot}`}><span className="terminal-dot" />{state === "open" ? "connected" : state}</span>
        <span className="terminal-foot-host">{(host.ssh_user || "ssh") + "@" + host.host}{host.ssh_port && host.ssh_port !== 22 ? `:${host.ssh_port}` : ""}</span>
        <span className="terminal-foot-sp" />
        <span className="terminal-foot-meta">SSH · PTY</span>
        {dims && <span className="terminal-foot-dims">{dims.cols}×{dims.rows}</span>}
      </footer>
      {!maximized && (
        <>
          <span className="terminal-resize edge-e" onPointerDown={(e) => beginDrag(e, "resize-e")} />
          <span className="terminal-resize edge-s" onPointerDown={(e) => beginDrag(e, "resize-s")} />
          <span className="terminal-resize corner" onPointerDown={(e) => beginDrag(e, "resize")} />
        </>
      )}
    </div>
  );
}
