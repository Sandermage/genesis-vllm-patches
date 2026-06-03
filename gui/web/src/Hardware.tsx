import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import {
  Activity, AlertTriangle, ArrowDown, ArrowUp, CircuitBoard, Cpu, Fan, Gauge,
  HardDrive, Network, RefreshCw, Server, Thermometer, Zap,
} from "lucide-react";
import { api, type GpuInfo, type HardwareSystem, type HardwareTelemetry } from "./api";

type HostOption = { id: string; label: string };
type Source = { kind: "local" } | { kind: "host"; hostId: string };
type NetRate = { rx: number; tx: number };

const num = (v: number | null | undefined, d = 0) => (typeof v === "number" && !Number.isNaN(v) ? v : d);
const gib1 = (mib: number | null | undefined) => (num(mib) / 1024).toFixed(1);
const clamp = (v: number, lo = 0, hi = 100) => Math.max(lo, Math.min(hi, v));

// Human-readable byte-rate (bytes/sec → KB/s, MB/s, GB/s).
function rate(bps: number): string {
  if (bps >= 1e9) return `${(bps / 1e9).toFixed(1)} GB/s`;
  if (bps >= 1e6) return `${(bps / 1e6).toFixed(1)} MB/s`;
  if (bps >= 1e3) return `${(bps / 1e3).toFixed(0)} KB/s`;
  return `${Math.round(bps)} B/s`;
}

// Tone thresholds shared by the ring gauges + bars (util / temp / power).
function tone(pct: number): "ok" | "warn" | "hot" {
  if (pct >= 90) return "hot";
  if (pct >= 70) return "warn";
  return "ok";
}

// A large circular gauge — the hero metric for a GPU (util / temp / power).
const RING_R = 34;
const RING_C = 2 * Math.PI * RING_R;
function Ring({ label, icon, value, pct, sub }: {
  label: string; icon: ReactNode; value: string; pct: number; sub?: string;
}) {
  const t = tone(pct);
  const off = RING_C * (1 - clamp(pct) / 100);
  return (
    <div className="hw-ring">
      <div className="hw-ring-dial">
        <svg viewBox="0 0 80 80" className={`hw-ring-svg ${t}`}>
          <circle cx="40" cy="40" r={RING_R} className="hw-ring-bg" />
          <circle cx="40" cy="40" r={RING_R} className="hw-ring-fg"
            strokeDasharray={RING_C} strokeDashoffset={off} />
        </svg>
        <strong className={`hw-ring-v ${t}`}>{value}</strong>
      </div>
      <span className="hw-ring-label">{icon} {label}</span>
      {sub && <span className="hw-ring-sub">{sub}</span>}
    </div>
  );
}

// A labelled progress bar (memory, bandwidth, fan).
function Bar({ label, value, pct, sub, tint }: {
  label: string; value: string; pct: number; sub?: string; tint?: boolean;
}) {
  const t = tint ? tone(pct) : "ok";
  return (
    <div className="hw-bar">
      <div className="hw-bar-head">
        <span className="hw-bar-label">{label}</span>
        <strong className={`hw-bar-value ${tint ? t : ""}`}>{value}</strong>
      </div>
      <div className="hw-bar-track"><div className={`hw-bar-fill ${t}`} style={{ width: `${clamp(pct)}%` }} /></div>
      {sub && <span className="hw-bar-sub">{sub}</span>}
    </div>
  );
}

function Row({ k, v, unit }: { k: string; v: ReactNode; unit?: string }) {
  return (
    <div className="hw-row">
      <span className="hw-row-k">{k}</span>
      <span className="hw-row-v">{v}{unit ? <em className="hw-row-u"> {unit}</em> : null}</span>
    </div>
  );
}

function GpuCard({ gpu, index }: { gpu: GpuInfo; index: number }) {
  const util = num(gpu.gpu_util);
  const temp = num(gpu.temp_gpu);
  const power = num(gpu.power);
  const powerMax = num(gpu.power_max_limit) || num(gpu.power_default_limit);
  const powerHeadroom = powerMax ? Math.max(0, powerMax - power) : 0;
  const memUsed = num(gpu.mem_used);
  const memTotal = num(gpu.mem_total) || 1;
  const memFree = gpu.mem_free != null ? num(gpu.mem_free) : Math.max(0, memTotal - memUsed);
  const memPct = clamp((memUsed / memTotal) * 100);
  const memBw = num(gpu.mem_util);
  const tempPct = clamp((temp / 90) * 100); // ~90°C as the visual ceiling
  const powerPct = powerMax ? clamp((power / powerMax) * 100) : 0;
  const fan = gpu.fan_speed;
  const eccUnc = num(Number(gpu.ecc_uncorrected));
  const pcieFull = !!gpu.pcie_gen && gpu.pcie_gen === gpu.pcie_gen_max && gpu.pcie_width === gpu.pcie_width_max;

  // A coarse health verdict surfaced as a status dot in the header.
  const status: { t: "ok" | "warn" | "hot"; label: string } =
    eccUnc > 0 ? { t: "hot", label: "ECC fault" }
    : temp >= 84 ? { t: "hot", label: "Hot" }
    : util >= 70 || temp >= 70 ? { t: "warn", label: "Under load" }
    : util < 5 ? { t: "ok", label: "Idle" }
    : { t: "ok", label: "Active" };

  return (
    <div className="hw-gpu">
      <div className="hw-gpu-head">
        <span className="hw-gpu-idx">{index}</span>
        <div className="hw-gpu-id">
          <strong className="hw-gpu-name" title={gpu.name ?? ""}>{(gpu.name ?? "GPU").replace(/^NVIDIA\s+/i, "")}</strong>
          <span className="hw-gpu-meta">
            {gpu.driver_version ? `driver ${gpu.driver_version}` : "driver —"}
            {gpu.compute_mode ? ` · ${gpu.compute_mode}` : ""}
          </span>
        </div>
        {gpu.pstate && <span className={`hw-pstate ${gpu.pstate === "P0" ? "ok" : ""}`} title="performance state">{gpu.pstate}</span>}
        <span className={`hw-status-dot ${status.t}`} title={status.label} />
      </div>

      <div className="hw-rings">
        <Ring label="Util" icon={<Gauge size={10} />} value={`${util}%`} pct={util} sub="utilization" />
        <Ring label="Temp" icon={<Thermometer size={10} />} value={`${temp}°`} pct={tempPct} sub={`${temp} °C`} />
        <Ring label="Power" icon={<Zap size={10} />} value={`${Math.round(powerPct)}%`} pct={powerPct}
          sub={powerMax ? `${Math.round(power)} / ${Math.round(powerMax)} W` : `${Math.round(power)} W`} />
      </div>

      <div className="hw-block">
        <Bar label="VRAM" value={`${gib1(memUsed)} / ${gib1(memTotal)} GB`} pct={memPct} tint
          sub={`${gib1(memFree)} GB free · ${Math.round(memPct)}% used`} />
        <Bar label="Memory bandwidth" value={`${memBw}%`} pct={memBw} sub="memory controller load" />
      </div>

      <div className="hw-grid">
        <div className="hw-sect">
          <div className="hw-sect-t"><Gauge size={11} /> Clocks</div>
          <Row k="Graphics" v={`${num(gpu.clock_gpu)} / ${num(gpu.clock_gpu_max)}`} unit="MHz" />
          <Row k="Memory" v={`${num(gpu.clock_mem)} / ${num(gpu.clock_mem_max)}`} unit="MHz" />
          <Row k="SM" v={`${num(gpu.clock_sm)}`} unit="MHz" />
        </div>
        <div className="hw-sect">
          <div className="hw-sect-t"><Fan size={11} /> Thermal &amp; power</div>
          <Row k="Fan" v={fan != null ? `${fan}` : "—"} unit={fan != null ? "%" : ""} />
          {gpu.temp_mem ? <Row k="Mem temp" v={`${gpu.temp_mem}`} unit="°C" /> : <Row k="Headroom" v={`${Math.round(powerHeadroom)}`} unit="W" />}
          <Row k="Limit" v={powerMax ? `${Math.round(powerMax)}` : "—"} unit={powerMax ? "W" : ""} />
        </div>
        <div className="hw-sect">
          <div className="hw-sect-t"><CircuitBoard size={11} /> Bus &amp; link</div>
          <Row k="PCIe" v={gpu.pcie_gen ? `gen${gpu.pcie_gen}×${num(gpu.pcie_width)}` : "—"} />
          <Row k="Max" v={gpu.pcie_gen_max ? `gen${num(gpu.pcie_gen_max)}×${num(gpu.pcie_width_max)}` : "—"} />
          <Row k="State" v={<span className={gpu.pcie_gen && !pcieFull ? "hw-warn-t" : ""}>{gpu.pcie_gen ? (pcieFull ? "full" : "degraded") : "—"}</span>} />
        </div>
        <div className="hw-sect">
          <div className="hw-sect-t"><Activity size={11} /> Reliability</div>
          <Row k="P-state" v={gpu.pstate ?? "—"} />
          <Row k="ECC corr" v={gpu.ecc_corrected ?? "—"} />
          <Row k="ECC uncorr" v={<span className={eccUnc > 0 ? "hw-bad" : ""}>{gpu.ecc_uncorrected ?? "—"}</span>} />
        </div>
      </div>

      <div className="hw-foot">
        {gpu.vbios_version && <span className="hw-foot-i"><CircuitBoard size={10} /> vBIOS {gpu.vbios_version}</span>}
        {gpu.uuid && <span className="hw-foot-i hw-uuid" title="GPU UUID"><Server size={10} /> {gpu.uuid}</span>}
      </div>
    </div>
  );
}

// Host-level facts: CPU, RAM, disk free, and live network throughput.
function HostPanel({ system, gpuCount, netRate }: { system: HardwareSystem; gpuCount: number; netRate: NetRate | null }) {
  const ramTotal = num(system.ram_total_gb);
  const ramUsed = num(system.ram_used_gb);
  const ramFree = Math.max(0, ramTotal - ramUsed);
  const ramPct = ramTotal ? clamp((ramUsed / ramTotal) * 100) : 0;
  const disk = system.disk;
  const diskUsed = num(disk?.used_gb);
  const diskTotal = num(disk?.total_gb);
  const diskPct = disk?.used_pct != null ? clamp(disk.used_pct) : diskTotal ? clamp((diskUsed / diskTotal) * 100) : 0;
  const iface = system.net?.[0]?.name;
  const hasNet = (system.net?.length ?? 0) > 0;

  return (
    <div className="hw-host">
      <div className="hw-host-head">
        <span className="hw-host-name"><HardDrive size={15} /> {system.hostname ?? "host"}</span>
        {system.cpu && <span className="hw-host-cpu"><Cpu size={13} /> {system.cpu}{system.cpu_count ? ` · ${system.cpu_count} cores` : ""}</span>}
        {system.primary_ip && <span className="hw-host-ip"><Network size={12} /> {system.primary_ip}</span>}
        <span className="hw-host-gpus"><Activity size={12} /> {gpuCount} GPU{gpuCount === 1 ? "" : "s"}</span>
      </div>
      <div className="hw-host-grid">
        {ramTotal > 0 && (
          <Bar label="System memory" value={`${ramUsed.toFixed(1)} / ${ramTotal.toFixed(1)} GB`} pct={ramPct} tint
            sub={`${ramFree.toFixed(1)} GB available · ${Math.round(ramPct)}% used`} />
        )}
        {disk && diskTotal > 0 && (
          <Bar label={`Disk · ${disk.mount}`} value={`${diskUsed.toFixed(0)} / ${diskTotal.toFixed(0)} GB`} pct={diskPct} tint
            sub={`${num(disk.free_gb).toFixed(0)} GB free · ${Math.round(diskPct)}% used`} />
        )}
        {hasNet && (
          <div className="hw-net">
            <span className="hw-bar-label">Network{iface ? ` · ${iface}` : ""}</span>
            <div className="hw-net-rates">
              <span className="hw-net-r down"><ArrowDown size={14} /> {netRate ? rate(netRate.rx) : "measuring…"}</span>
              <span className="hw-net-r up"><ArrowUp size={14} /> {netRate ? rate(netRate.tx) : "measuring…"}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export function HardwarePanel({ hosts, initialHostId }: { hosts: HostOption[]; initialHostId?: string }) {
  const [source, setSource] = useState<Source>(
    initialHostId && hosts.some((h) => h.id === initialHostId) ? { kind: "host", hostId: initialHostId } : { kind: "local" });
  const [data, setData] = useState<HardwareTelemetry | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [netRate, setNetRate] = useState<NetRate | null>(null);
  const loadingRef = useRef(false);
  loadingRef.current = loading;
  // Previous cumulative RX/TX sample + timestamp, to derive live throughput.
  const prevNet = useRef<{ t: number; rx: number; tx: number } | null>(null);

  const load = useCallback(async () => {
    setLoading(true); setErr(null);
    try {
      const r = source.kind === "host" ? await api.hostGpuRemote(source.hostId) : await api.hostGpu();
      setData(r);
      const ifs = r.system?.net ?? [];
      if (ifs.length) {
        const rx = ifs.reduce((s, i) => s + num(i.rx_bytes), 0);
        const tx = ifs.reduce((s, i) => s + num(i.tx_bytes), 0);
        const now = Date.now();
        const prev = prevNet.current;
        if (prev && now > prev.t) {
          const dt = (now - prev.t) / 1000;
          setNetRate({ rx: Math.max(0, (rx - prev.rx) / dt), tx: Math.max(0, (tx - prev.tx) / dt) });
        }
        prevNet.current = { t: now, rx, tx };
      } else {
        prevNet.current = null;
      }
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e)); setData(null);
    } finally { setLoading(false); }
  }, [source]);

  // New source → drop stale data + network baseline.
  useEffect(() => { setData(null); setNetRate(null); prevNet.current = null; void load(); }, [load]);
  // Live refresh every 4s while the tab is visible and no request is in flight.
  useEffect(() => {
    const t = window.setInterval(() => { if (!loadingRef.current && !document.hidden) void load(); }, 4000);
    return () => window.clearInterval(t);
  }, [load]);

  const sys = data?.system;
  const gpus = data?.gpus ?? [];

  return (
    <div className="hw">
      <div className="hw-bar-top">
        <label className="hw-source">
          <Server size={14} />
          <select value={source.kind === "local" ? "__local__" : source.hostId}
            onChange={(e) => { const v = e.target.value; setSource(v === "__local__" ? { kind: "local" } : { kind: "host", hostId: v }); }}>
            <option value="__local__">This daemon host</option>
            {hosts.map((h) => <option key={h.id} value={h.id}>{h.label} · SSH</option>)}
          </select>
        </label>
        <span className="hw-auto">live · 4s</span>
        <button className="ghost-button" onClick={() => void load()} disabled={loading}>
          {loading ? <RefreshCw size={14} className="spin" /> : <RefreshCw size={14} />} Refresh
        </button>
      </div>

      {sys && (sys.cpu || sys.ram_total_gb || sys.disk || sys.net) && (
        <HostPanel system={sys} gpuCount={gpus.length} netRate={netRate} />
      )}

      {err && <div className="hw-empty err"><AlertTriangle size={22} /><strong>Telemetry unavailable</strong><span>{err}</span></div>}

      {!err && data && gpus.length === 0 && (
        <div className="hw-empty"><Activity size={22} /><strong>No GPU telemetry</strong>
          <span>{data.error ?? "nvidia-smi reported no devices on this host."}</span></div>
      )}

      {gpus.length > 0 && (
        <div className="hw-gpus">
          {gpus.map((g, i) => <GpuCard key={g.uuid ?? i} gpu={g} index={i} />)}
        </div>
      )}

      {!data && loading && <div className="hw-empty"><RefreshCw size={20} className="spin" /><span>Reading hardware telemetry…</span></div>}
    </div>
  );
}
