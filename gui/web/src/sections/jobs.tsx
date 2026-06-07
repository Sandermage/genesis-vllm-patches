// SPDX-License-Identifier: Apache-2.0
// Jobs domain: the recent-jobs table, a small progress bar, and the live job
// monitor modal that polls a job to completion. Extracted from App.tsx
// (modularization).
//
// Enterprise hardening over the inline originals (markup classes unchanged):
//   * table headers carry scope="col" and the action column an aria-label, so
//     screen-reader table navigation announces real column relationships;
//   * Progress is a real role="progressbar" with aria-valuenow/min/max.
import { useEffect, useRef, useState, Fragment } from "react";
import { Activity, ChevronRight, X, CheckCircle2, AlertCircle, Circle, Terminal } from "lucide-react";
import { api, type Job } from "../api";
import { SkeletonTable } from "../Skeleton";
import { CodeBlock, CopyButton } from "../components/code-block";
import { StatusPill } from "../components/primitives";
import { useDialogFocus, useEscapeKey, closeOnBackdrop } from "../dialog";

/** Tone for a job's executor outcome: queued dry-run, succeeded, or failed. */
export function jobTone(job: Job): "neutral" | "success" | "danger" {
  return job.dry_run ? "neutral" : job.status === "succeeded" ? "success" : "danger";
}

// Shared executor-job result card. Used by the Launch Plan, Configs apply queue
// and the Services lifecycle planner so the three apply paths render identically.
export function JobResultBlock({ job, showNote = false }: { job: Job; showNote?: boolean }) {
  return (
    <div className="service-job">
      <div className="service-job-head">
        <div>
          <strong>{job.job_id}</strong>
          <span>{job.kind}</span>
        </div>
        <StatusPill tone={jobTone(job)}>
          {job.dry_run ? "dry-run recorded" : `executed: ${job.status}`}
        </StatusPill>
      </div>
      <CodeBlock lines={job.log} />
      {showNote && job.note && <p className="service-reason">{job.note}</p>}
    </div>
  );
}

export function JobsTable({ onMonitor }: { onMonitor?: (id: string) => void }) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [openId, setOpenId] = useState<string | null>(null);
  const [state, setState] = useState<"loading" | "ready">("loading");

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const result = await api.jobs();
        if (!cancelled) { setJobs(result.jobs); setState("ready"); }
      } catch {
        if (!cancelled) setState("ready");
      }
    };
    void load();
    const timer = setInterval(load, 5000);
    return () => { cancelled = true; clearInterval(timer); };
  }, []);

  if (state === "loading") return <SkeletonTable rows={5} cols={6} />;
  if (jobs.length === 0) {
    return (
      <p className="muted">
        No jobs yet. Run <strong>Apply Launch</strong>, a service action, or queue a
        bench/evidence job — real dry-run and executed jobs appear here.
      </p>
    );
  }
  const cls = (job: Job) => job.dry_run ? "queued" : job.status === "succeeded" ? "done" : job.status === "failed" ? "failed" : "running";
  const stamp = (ts: number) => new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  return (
    <section className="jobs-block">
      <table className="jobs-table">
        <thead>
          <tr>
            <th scope="col">Job</th>
            <th scope="col">Kind</th>
            <th scope="col">Status</th>
            <th scope="col">Steps</th>
            <th scope="col">Time</th>
            <th scope="col" aria-label="Actions" />
          </tr>
        </thead>
        <tbody>
          {jobs.map((job) => (
            <Fragment key={job.job_id}>
              <tr className={openId === job.job_id ? "active job-row" : "job-row"} onClick={() => setOpenId(openId === job.job_id ? null : job.job_id)}>
                <td><code>{job.job_id}</code></td>
                <td>{job.kind}</td>
                <td><span className={`job-status ${cls(job)}`}>{job.dry_run ? "dry-run" : job.status}</span></td>
                <td>{job.steps.length}</td>
                <td>{stamp(job.created_at)}</td>
                <td className="job-row-actions">
                  {onMonitor && (
                    <button className="icon-only" title="Live monitor" aria-label={`Monitor ${job.job_id}`} onClick={(event) => { event.stopPropagation(); onMonitor(job.job_id); }}>
                      <Activity size={14} />
                    </button>
                  )}
                  <ChevronRight size={14} className={openId === job.job_id ? "job-caret open" : "job-caret"} />
                </td>
              </tr>
              {openId === job.job_id && (
                <tr className="job-detail-row">
                  <td colSpan={6}>
                    {job.note && <p className="fit-note">{job.note}</p>}
                    <CodeBlock lines={job.log} />
                  </td>
                </tr>
              )}
            </Fragment>
          ))}
        </tbody>
      </table>
    </section>
  );
}

export function Progress({ value }: { value: number }) {
  const clamped = Math.max(0, Math.min(100, value));
  return (
    <span className="progress-track" role="progressbar" aria-valuenow={Math.round(clamped)} aria-valuemin={0} aria-valuemax={100}>
      <span style={{ width: `${clamped}%` }} />
    </span>
  );
}

const JOB_TERMINAL = new Set(["succeeded", "failed", "done", "error", "cancelled"]);

// Live job monitor modal — polls a job until it reaches a terminal state,
// streaming status, steps and log. Shared by launch / benchmark / evidence.
export function JobMonitorModal({ jobId, onClose }: { jobId: string; onClose: () => void }) {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const dialogRef = useRef<HTMLElement>(null);
  useDialogFocus(dialogRef);
  useEscapeKey(onClose);

  useEffect(() => {
    let cancelled = false;
    let timer = 0;
    const poll = async () => {
      try {
        const next = await api.job(jobId);
        if (cancelled) return;
        setJob(next);
        const terminal = next.dry_run || JOB_TERMINAL.has(next.status);
        if (!terminal) timer = window.setTimeout(poll, 1500);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      }
    };
    void poll();
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [jobId]);

  const running = !!job && !job.dry_run && !JOB_TERMINAL.has(job.status);
  const tone = !job ? "running" : job.dry_run ? "queued" : job.status === "succeeded" || job.status === "done" ? "done" : job.status === "failed" || job.status === "error" ? "failed" : "running";
  const statusText = !job ? "polling…" : job.dry_run ? "dry-run (recorded)" : job.status;
  const progress = typeof job?.progress === "number" ? job.progress : running ? -1 : 100;

  return (
    <div className="dialog-backdrop" role="presentation" onClick={closeOnBackdrop(onClose)}>
      <section ref={dialogRef} className="job-monitor" role="dialog" aria-modal="true">
        <header className="job-monitor-head">
          <div className="job-monitor-title">
            <Activity size={18} className={running ? "spin" : ""} />
            <div>
              <h2>{job?.title ?? "Job"}</h2>
              <code>{jobId}{job ? ` · ${job.kind}` : ""}</code>
            </div>
          </div>
          <span className={`job-status ${tone}`}>{statusText}</span>
          <button className="icon-only" onClick={onClose} aria-label="Close"><X size={16} /></button>
        </header>

        {progress >= 0 ? <Progress value={progress} /> : <div className="job-monitor-indeterminate"><span /></div>}
        {error && <div className="inline-error"><AlertCircle size={15} /> {error}</div>}

        {job && job.steps.length > 0 && (
          <div className="job-monitor-steps">
            {job.steps.map((step) => (
              <div className={`job-step ${step.status}`} key={step.order}>
                <span className="job-step-icon">
                  {step.status === "succeeded" || step.status === "done" ? <CheckCircle2 size={14} /> : step.status === "failed" ? <AlertCircle size={14} /> : step.status === "running" ? <Activity size={14} className="spin" /> : <Circle size={14} />}
                </span>
                <strong>{step.title}</strong>
                <code>{step.command}</code>
              </div>
            ))}
          </div>
        )}

        {job && job.note && <p className="fit-note">{job.note}</p>}

        <div className="job-monitor-log">
          <div className="job-monitor-log-head"><Terminal size={13} /> Log{running ? " · live" : ""}<CopyButton value={(job?.log ?? []).join("\n")} label="job log" /></div>
          <pre className="code-block">{(job?.log ?? ["(waiting for output…)"]).join("\n")}</pre>
        </div>

        <div className="job-monitor-foot">
          <span className="muted">{running ? "Polling every 1.5s until the job finishes." : job?.dry_run ? "Dry-run — start the daemon with --enable-apply to execute." : "Job finished."}</span>
          <button className="primary-action" onClick={onClose}>Close</button>
        </div>
      </section>
    </div>
  );
}
