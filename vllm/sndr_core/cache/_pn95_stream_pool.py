# SPDX-License-Identifier: Apache-2.0
"""PN95 — CUDA stream + event pool for non-blocking demote/promote.

Inspired by vllm upstream PR #40020's `cpu/gpu_worker.py` Transfer dataclass.
Replaces the single `_PN95_CUDA_STREAM` + `stream.synchronize()` host-blocking
wait with a pool of streams paired with start/end CUDA events.

The host thread NEVER blocks on a transfer; instead `get_finished()` is
non-blocking — it pops only those transfers whose `end_event.query()`
returns True. Pending transfers stay in a deque; the host thread can do
other work (e.g. queue more transfers, run Python bookkeeping) while
PCIe DMA progresses on its own.

This is the foundation for layer-by-layer overlap (SGLang HiCache style) —
the scheduler can submit prefetch transfers for layer N+1 while layer N is
mid-compute on the default stream, then call `get_finished()` periodically
to drain completed copies into the L1 pinned pool.

API:
  acquire_stream() -> torch.cuda.Stream         — pop stream from pool or new
  release_stream(stream)                        — return stream to pool
  acquire_event()  -> torch.Event               — pop event or new
  release_event(event)
  submit(fn, ..., callback=None) -> JobId       — queue transfer; non-blocking
  get_finished() -> list[JobId]                 — non-blocking poll
  drain_completed() -> int                      — drain ALL completed jobs (sync-ish)

Caller (PN95 demote/promote) wraps existing copy code with submit() so it
runs async. The Transfer entry records start_event (recorded before fn)
and end_event (recorded after fn) on its own stream.

All operations are torch-only — no extra C++ binding required. Works on
the same CUDA driver as our existing _pn95_stream code.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

log = logging.getLogger("genesis.pn95.stream_pool")

JobId = int


@dataclass
class Transfer:
    """One in-flight async PCIe transfer.

    end_event.query() returns True when DMA is complete. start_event is kept
    for potential future profiling / overlap measurement.
    """
    job_id: JobId
    stream: Any  # torch.cuda.Stream
    start_event: Any  # torch.Event
    end_event: Any  # torch.Event
    direction: str  # "h2d" | "d2h"
    n_bytes: int
    callback: Optional[Callable] = None
    submit_time: float = 0.0
    extra: dict = field(default_factory=dict)


class _PoolState:
    """Per-process singleton owning stream/event pools and the in-flight queue.

    Thread-safe — PN95 scheduler tick + worker forward can both submit/poll.
    """
    def __init__(self):
        self.stream_pool: list = []
        self.event_pool: list = []
        self.in_flight: deque = deque()
        self.next_job_id: JobId = 1
        self.lock = threading.Lock()
        # Soft caps so a buggy caller can't blow process RSS on stream churn.
        self.max_streams = 16
        self.max_events = 64

    def acquire_stream(self):
        import torch
        with self.lock:
            if self.stream_pool:
                return self.stream_pool.pop()
        return torch.cuda.Stream()

    def release_stream(self, stream) -> None:
        with self.lock:
            if len(self.stream_pool) < self.max_streams:
                self.stream_pool.append(stream)

    def acquire_event(self):
        import torch
        with self.lock:
            if self.event_pool:
                return self.event_pool.pop()
        # blocking=False by default; we'll use .query()
        return torch.cuda.Event(enable_timing=False, blocking=False)

    def release_event(self, event) -> None:
        with self.lock:
            if len(self.event_pool) < self.max_events:
                self.event_pool.append(event)


_STATE: Optional[_PoolState] = None
_STATE_LOCK = threading.Lock()


def _state() -> _PoolState:
    global _STATE
    if _STATE is None:
        with _STATE_LOCK:
            if _STATE is None:
                _STATE = _PoolState()
    return _STATE


def submit(
    fn: Callable[[], None],
    *,
    direction: str = "d2h",
    n_bytes: int = 0,
    callback: Optional[Callable] = None,
    extra: Optional[dict] = None,
) -> Transfer:
    """Queue an async transfer.

    `fn` is called with the chosen stream as the current CUDA stream context
    (use `with torch.cuda.stream(t.stream):` inside if you allocate tensors).
    start_event is recorded BEFORE fn() returns, end_event AFTER. fn must
    enqueue its work onto the current stream non-blockingly (use
    `.copy_(..., non_blocking=True)` / `.to(..., non_blocking=True)`).

    Returns a Transfer record. Caller may discard the return — the Transfer
    is also kept in the in-flight deque for drain_completed() pickup.
    """
    import torch
    import time
    st = _state()
    stream = st.acquire_stream()
    start_evt = st.acquire_event()
    end_evt = st.acquire_event()

    with torch.cuda.stream(stream):
        start_evt.record(stream)
        try:
            fn()
        except Exception as e:
            log.warning("[PN95-stream] fn raised; releasing stream/events: %s", e)
            st.release_stream(stream)
            st.release_event(start_evt)
            st.release_event(end_evt)
            raise
        end_evt.record(stream)

    with st.lock:
        job_id = st.next_job_id
        st.next_job_id += 1
    t = Transfer(
        job_id=job_id, stream=stream,
        start_event=start_evt, end_event=end_evt,
        direction=direction, n_bytes=n_bytes,
        callback=callback, submit_time=time.time(),
        extra=extra or {},
    )
    with st.lock:
        st.in_flight.append(t)
    return t


def poll_finished(release_to_pool: bool = True) -> list[Transfer]:
    """Non-blocking poll. Returns completed transfers (in order of submission).

    Implementation drains the head of the deque while `end_event.query()` is
    True. As soon as a head transfer is NOT complete, polling stops (FIFO).
    Completed transfers have their stream/events returned to pools and
    callbacks fired.
    """
    st = _state()
    done: list[Transfer] = []
    with st.lock:
        while st.in_flight and st.in_flight[0].end_event.query():
            t = st.in_flight.popleft()
            done.append(t)
    if release_to_pool:
        for t in done:
            try:
                if t.callback:
                    t.callback(t)
            except Exception:
                pass
            st.release_stream(t.stream)
            st.release_event(t.start_event)
            st.release_event(t.end_event)
    return done


def drain_all_completed(spin_us: int = 0) -> int:
    """Hot loop draining completed transfers until queue is empty OR head not done.

    Returns count of transfers drained. With spin_us=0 it is a single non-blocking
    pass (poll_finished). With spin_us>0 it busy-polls (small) for that microsecond
    budget — useful when caller wants to flush as much as possible before
    dispatching downstream compute.
    """
    import time
    if spin_us <= 0:
        return len(poll_finished())
    deadline = time.time() + (spin_us / 1_000_000.0)
    total = 0
    while time.time() < deadline:
        done = poll_finished()
        if not done:
            break
        total += len(done)
    return total


def in_flight_count() -> int:
    st = _state()
    with st.lock:
        return len(st.in_flight)


def pool_stats() -> dict:
    st = _state()
    with st.lock:
        return {
            "streams_pooled": len(st.stream_pool),
            "events_pooled": len(st.event_pool),
            "in_flight": len(st.in_flight),
            "next_job_id": st.next_job_id - 1,
        }


def reset_pools_for_tests() -> None:
    """Test-only — drop pool state."""
    global _STATE
    with _STATE_LOCK:
        _STATE = None
