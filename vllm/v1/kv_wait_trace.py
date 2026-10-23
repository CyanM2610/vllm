# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in passive KV wait events. No decisions or CUDA synchronization.

One append-only file per process; timestamps use this host's MONOTONIC clock.
The synchronous observer cost is recorded, never subtracted from latency.
"""

import atexit
import json
import os
import threading
import time
from contextlib import contextmanager
from functools import wraps

from vllm.v1 import host_diagnostic as _hd

_ROOT = os.environ.get("KV_WAIT_TRACE_DIR")
_LOCK = threading.Lock()
_FD = None
_PID = None
_COUNT = 0
_COST = 0
_CPU_COST = 0
_ONCE = set()
_BATCH_ENABLED = os.environ.get("KV_WAIT_TRACE_BATCH_BLOCKS") == "1"
_MAX_EVENTS = 128
_MAX_BYTES = 262144
_DEPTH = 0
_BUFFER = []
_BUFFER_BYTES = 0
_WRITES = 0
_WRITE_ERRORS = 0
_WRITE_WALL = 0
_WRITE_CPU = 0
_PEAK_EVENTS = 0
_PEAK_BYTES = 0


def _write(data):
    """Write under the observer lock, preserving byte order on short writes."""
    global _WRITES, _WRITE_ERRORS, _WRITE_WALL, _WRITE_CPU
    wall, cpu = time.monotonic_ns(), time.thread_time_ns()
    try:
        remaining = memoryview(data)
        while remaining:
            size = os.write(_FD, remaining)
            _WRITES += 1
            if size <= 0:
                raise OSError("incomplete KV wait trace write")
            remaining = remaining[size:]
    except OSError:
        _WRITE_ERRORS += 1
        raise
    finally:
        _WRITE_WALL += time.monotonic_ns() - wall
        _WRITE_CPU += time.thread_time_ns() - cpu


def _flush():
    """Flush occurrence-stamped lines; caller holds the observer lock."""
    global _BUFFER_BYTES
    if _BUFFER:
        data = b"".join(_BUFFER)
        # An I/O failure invalidates collection; do not retry a partial stream.
        _BUFFER.clear()
        _BUFFER_BYTES = 0
        _write(data)


@contextmanager
def batch_events():
    """Bounded synchronous batching within one native block-pool call.

    Other threads append through the same lock, so file order is unchanged.
    No lock is held across resource operations. Every scope exit flushes,
    including exceptions; large batches also flush at the size/count bound.
    """
    if not _ROOT or not _BATCH_ENABLED:
        yield
        return
    global _DEPTH, _COST, _CPU_COST
    with _LOCK:
        _DEPTH += 1
    try:
        yield
    finally:
        start, cpu = time.monotonic_ns(), time.thread_time_ns()
        with _LOCK:
            _DEPTH -= 1
            _flush()
            _COST += time.monotonic_ns() - start
            _CPU_COST += time.thread_time_ns() - cpu


def batch_call(function):
    """Decorate a resource call without changing its result or arguments."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        with batch_events():
            return function(*args, **kwargs)

    return wrapped


def emit(name: str, request_id: str | None = None, **fields) -> None:
    """Record a host observation; this does not imply GPU execution."""
    if not _ROOT:
        return
    global _FD, _PID, _COUNT, _COST, _CPU_COST, _BUFFER_BYTES
    global _PEAK_EVENTS, _PEAK_BYTES
    _d0 = _hd.mark()
    begin = time.monotonic_ns()
    cpu_begin = time.thread_time_ns()
    with _LOCK:
        _d1 = _hd.mark()
        pid = os.getpid()
        if _FD is None or pid != _PID:
            if _PID is not None and pid != _PID:
                # Inherited buffered events remain the parent's obligation.
                if _FD is not None:
                    os.close(_FD)
                _BUFFER.clear()
                _BUFFER_BYTES = 0
            os.makedirs(_ROOT, exist_ok=True)
            _FD = os.open(
                os.path.join(_ROOT, f"probe-{pid}.jsonl"),
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            _PID = pid
        row = dict(
            event=name,
            ts_ns=begin,
            pid=pid,
            tid=threading.get_native_id(),
            request_id=request_id,
            **fields,
        )
        _d2 = _hd.mark()
        data = (json.dumps(row, separators=(",", ":"), default=str) + "\n").encode()
        _d3 = _hd.mark()
        if _DEPTH and _BATCH_ENABLED and len(data) <= _MAX_BYTES:
            if len(_BUFFER) >= _MAX_EVENTS or _BUFFER_BYTES + len(data) > _MAX_BYTES:
                _flush()
            _BUFFER.append(data)
            _BUFFER_BYTES += len(data)
            _PEAK_EVENTS = max(_PEAK_EVENTS, len(_BUFFER))
            _PEAK_BYTES = max(_PEAK_BYTES, _BUFFER_BYTES)
        else:
            _flush()
            _write(data)
        _d4 = _hd.mark()
        _COUNT += 1
        _COST += time.monotonic_ns() - begin
        _CPU_COST += time.thread_time_ns() - cpu_begin
    _hd.record("emit.lock", _d0, _d1)
    _hd.record("emit.serialize", _d2, _d3)
    _hd.record("emit.write", _d3, _d4)


def once(name: str, request_id: str, **fields) -> None:
    """Record the first occurrence for a request without changing its state."""
    if not _ROOT:
        return
    key = (os.getpid(), name, request_id)
    if key not in _ONCE:
        _ONCE.add(key)
        emit(name, request_id, **fields)


@contextmanager
def span(name: str, request_id: str | None = None, **fields):
    """Emit host boundaries and an optional CUPTI-correlatable NVTX range."""
    if not _ROOT:
        yield
        return
    begin = time.monotonic_ns()
    nvtx = None
    if os.environ.get("KV_WAIT_NVTX") == "1":
        import torch

        nvtx = torch.cuda.nvtx
        nvtx.range_push("kvprobe:" + name + ":" + str(request_id))
        # An instant CPU anchor lets the analyzer align CUPTI's clock to
        # CLOCK_MONOTONIC. The surrounding bounds quantify anchor uncertainty.
        anchor_before = time.monotonic_ns()
        nvtx.mark("kvprobe_anchor:" + str(anchor_before))
        emit(
            "gpu_clock_anchor",
            request_id,
            anchor_before_ns=anchor_before,
            anchor_after_ns=time.monotonic_ns(),
        )
    emit(name + ".begin", request_id, **fields)
    try:
        yield
    finally:
        emit(name + ".end", request_id, begin_ns=begin, **fields)
        if nvtx is not None:
            nvtx.range_pop()


def close() -> None:
    """Flush the small observer cost audit at normal process shutdown."""
    global _FD
    if _FD is not None and os.getpid() == _PID:
        with _LOCK:
            _flush()
        emit(
            "observer_summary",
            events=_COUNT,
            synchronous_cost_ns=_COST,
            thread_cpu_ns=_CPU_COST,
            block_call_batching=_BATCH_ENABLED,
            write_calls=_WRITES,
            write_errors=_WRITE_ERRORS,
            dropped_events=0,
            write_wall_ns=_WRITE_WALL,
            write_thread_cpu_ns=_WRITE_CPU,
            buffer_events=0,
            buffer_bytes=0,
            peak_buffer_events=_PEAK_EVENTS,
            peak_buffer_bytes=_PEAK_BYTES,
            max_buffer_events=_MAX_EVENTS,
            max_buffer_bytes=_MAX_BYTES,
        )
        with _LOCK:
            _flush()
        _hd.dump()
        os.close(_FD)
        _FD = None


atexit.register(close)
