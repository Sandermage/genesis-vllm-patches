// SPDX-License-Identifier: Apache-2.0
/**
 * DataView — generic data-loading wrapper for feature modules.
 *
 * Handles the three states every feature module faces:
 *   1. loading → DataTableSkeleton
 *   2. error → InlineNotification
 *   3. empty → InlineNotification info
 *   4. populated → caller-provided render(data)
 *
 * Reduces boilerplate from ~50 LOC per feature to ~5 LOC.
 */
import { useEffect, useState, type ReactNode } from 'react';
import { DataTableSkeleton, InlineNotification } from '@carbon/react';

interface DataViewProps<T> {
  load: () => Promise<T>;
  deps?: ReadonlyArray<unknown>;
  isEmpty?: (data: T) => boolean;
  emptyTitle?: string;
  errorTitle?: string;
  skeletonRows?: number;
  skeletonHeaders?: Array<{ key: string; header: string }>;
  children: (data: T) => ReactNode;
}

export function DataView<T>({
  load,
  deps = [],
  isEmpty,
  emptyTitle = 'No data',
  errorTitle = 'Failed to load',
  skeletonRows = 4,
  skeletonHeaders,
  children,
}: DataViewProps<T>): JSX.Element {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError(null);
    load()
      .then((d) => { if (!cancelled) setData(d); })
      .catch((e) => { if (!cancelled) setError(String(e)); });
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  if (error) {
    return <InlineNotification kind="error" title={errorTitle} subtitle={error} hideCloseButton />;
  }
  if (data === null) {
    return skeletonHeaders
      ? <DataTableSkeleton headers={skeletonHeaders} rowCount={skeletonRows} />
      : <p>Loading…</p>;
  }
  if (isEmpty && isEmpty(data)) {
    return <InlineNotification kind="info" title={emptyTitle} hideCloseButton />;
  }
  return <>{children(data)}</>;
}
