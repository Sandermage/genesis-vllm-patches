// SPDX-License-Identifier: Apache-2.0
/**
 * Minimal API client for sndr-platform Control Center.
 *
 * This module wraps the global ``fetch`` with:
 *   - Standardized base URL (configurable)
 *   - Request id propagation
 *   - Error normalization (RFC 7807 → ApiError instances)
 *   - Auth header injection (when a session cookie is present)
 *
 * Each feature module imports this client and builds typed wrappers on top.
 */

/** Base URL for the API (defaults to same-origin). */
const API_BASE = import.meta.env.VITE_SNDR_API_BASE ?? '';

/** Typed error matching the RFC 7807 Problem Details shape. */
export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly code: string,
    message: string,
    public readonly extensions: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

/** Envelope shape returned by every endpoint. */
export interface Envelope<T> {
  data: T;
  meta: {
    api_version: string;
    request_id: string;
    engine?: string | null;
    pin?: string | null;
    timestamp: string;
  };
}

interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  body?: unknown;
  headers?: Record<string, string>;
  signal?: AbortSignal;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<Envelope<T>> {
  const headers: Record<string, string> = {
    Accept: 'application/json',
    ...(options.headers ?? {}),
  };

  if (options.body !== undefined && !(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }

  const response = await fetch(`${API_BASE}${path}`, {
    method: options.method ?? 'GET',
    headers,
    body: options.body instanceof FormData
      ? options.body
      : options.body !== undefined
        ? JSON.stringify(options.body)
        : undefined,
    signal: options.signal,
    credentials: 'same-origin',
  });

  if (!response.ok) {
    let problem: { type?: string; title?: string; status?: number; detail?: string; extensions?: Record<string, unknown> } = {};
    try {
      problem = await response.json();
    } catch {
      // not JSON; fall through with status-based message
    }
    throw new ApiError(
      response.status,
      problem.type ?? 'sndr.api.unknown',
      problem.detail ?? problem.title ?? response.statusText,
      problem.extensions ?? {},
    );
  }

  return response.json() as Promise<Envelope<T>>;
}

export const apiClient = {
  get: <T>(path: string, opts?: Omit<RequestOptions, 'method'>) =>
    request<T>(path, { ...opts, method: 'GET' }),
  post: <T>(path: string, body?: unknown, opts?: Omit<RequestOptions, 'method' | 'body'>) =>
    request<T>(path, { ...opts, method: 'POST', body }),
  put: <T>(path: string, body?: unknown, opts?: Omit<RequestOptions, 'method' | 'body'>) =>
    request<T>(path, { ...opts, method: 'PUT', body }),
  delete: <T>(path: string, opts?: Omit<RequestOptions, 'method'>) =>
    request<T>(path, { ...opts, method: 'DELETE' }),
};
