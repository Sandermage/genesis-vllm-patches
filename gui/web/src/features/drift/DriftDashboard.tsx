// SPDX-License-Identifier: Apache-2.0
/**
 * DriftDashboard — overview of drift status for the current engine and pin.
 *
 * Drift severity buckets are surfaced as KPI tiles, and any affected patches
 * are listed so the operator can decide whether to refresh anchors.
 */
import { useEffect, useState } from 'react';
import {
  Tile, Tag, InlineNotification, SkeletonText, ClickableTile,
} from '@carbon/react';
import { useEngineStore } from '@/stores/engine';
import { getDriftSummary, type DriftSummary } from './api';

const SEVERITY_TONE: Record<string, 'green' | 'cyan' | 'red' | 'magenta'> = {
  ok: 'green',
  benign: 'cyan',
  drift: 'red',
  blocked: 'magenta',
};

export function DriftDashboard(): JSX.Element {
  const engine = useEngineStore((s) => s.selected);
  const pin = useEngineStore((s) => s.pin);
  const [summary, setSummary] = useState<DriftSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!pin) return;
    let cancelled = false;
    setSummary(null);
    setError(null);
    getDriftSummary(engine, pin)
      .then((s) => { if (!cancelled) setSummary(s); })
      .catch((e) => { if (!cancelled) setError(String(e)); });
    return () => { cancelled = true; };
  }, [engine, pin]);

  if (!pin) {
    return (
      <InlineNotification
        kind="info"
        title="Select a pin to view drift status"
        subtitle="Pick a pin from the top bar to see its drift report."
        hideCloseButton
      />
    );
  }

  if (error) {
    return (
      <InlineNotification
        kind="error"
        title="Drift report unavailable"
        subtitle={error}
        hideCloseButton
      />
    );
  }

  if (!summary) {
    return <SkeletonText paragraph lineCount={5} />;
  }

  return (
    <div className="sndr-drift-dashboard">
      <Tile>
        <h3>Drift summary — {engine} @ {pin}</h3>
        <p>Last checked: {summary.checked_at}</p>
        <Tag type={SEVERITY_TONE[summary.overall_severity] ?? 'gray'}>
          {summary.overall_severity}
        </Tag>
      </Tile>

      <div className="sndr-kpi-grid">
        <ClickableTile>
          <h4>OK</h4>
          <p>{summary.files_ok}</p>
        </ClickableTile>
        <ClickableTile>
          <h4>Benign</h4>
          <p>{summary.files_benign}</p>
        </ClickableTile>
        <ClickableTile>
          <h4>Drift</h4>
          <p>{summary.files_drift}</p>
        </ClickableTile>
        <ClickableTile>
          <h4>Blocked</h4>
          <p>{summary.files_blocked}</p>
        </ClickableTile>
      </div>

      {summary.affected_patches.length > 0 && (
        <Tile>
          <h4>Affected patches ({summary.affected_patches.length})</h4>
          <ul>
            {summary.affected_patches.slice(0, 20).map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
        </Tile>
      )}
    </div>
  );
}
