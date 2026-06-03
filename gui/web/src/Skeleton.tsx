// Shared skeleton-loader primitives. Content-shaped shimmering placeholders that
// hold layout while data is in flight, so panels fade in instead of popping in
// after a blank/spinner. Used across App, ContainersPanel, Engine and Fleet, so
// they live in their own module (no circular imports between those files).

// Base block. `count` repeats it; `variant` picks the size (line / metric / card).
export function Skeleton({ variant = "line", count = 1, className = "" }: {
  variant?: "line" | "metric" | "card";
  count?: number;
  className?: string;
}) {
  const cls = variant === "metric" ? "skel-metric" : variant === "card" ? "skel-card" : "skel-line";
  return (
    <>
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className={`skeleton ${cls} ${className}`.trim()} aria-hidden="true" />
      ))}
    </>
  );
}

// Responsive grid of metric tiles — the canonical "catalog/stats loading" shape.
export function SkeletonMetrics({ count = 4 }: { count?: number }) {
  return (
    <div className="skel-grid metrics" role="status" aria-label="Loading…">
      <Skeleton variant="metric" count={count} />
    </div>
  );
}

// Stacked lines — for key/value detail panels, logs and text blocks.
export function SkeletonLines({ count = 5 }: { count?: number }) {
  return (
    <div className="skel-grid" role="status" aria-label="Loading…">
      <Skeleton variant="line" count={count} />
    </div>
  );
}

// Responsive grid of cards — for lists rendered as cards (containers, hosts).
export function SkeletonCards({ count = 6 }: { count?: number }) {
  return (
    <div className="skel-grid cards" role="status" aria-label="Loading…">
      <Skeleton variant="card" count={count} />
    </div>
  );
}

// Table-shaped skeleton — a faint header row plus `rows` body rows of `cols`
// cells, so a loading table keeps its column rhythm instead of collapsing.
export function SkeletonTable({ rows = 5, cols = 4 }: { rows?: number; cols?: number }) {
  const grid = { gridTemplateColumns: `repeat(${cols}, 1fr)` };
  return (
    <div className="skel-table" role="status" aria-label="Loading…">
      <div className="skel-table-row skel-table-head" style={grid}>
        {Array.from({ length: cols }).map((_, i) => <Skeleton key={i} variant="line" />)}
      </div>
      {Array.from({ length: rows }).map((_, r) => (
        <div className="skel-table-row" key={r} style={grid}>
          {Array.from({ length: cols }).map((_, c) => <Skeleton key={c} variant="line" />)}
        </div>
      ))}
    </div>
  );
}
