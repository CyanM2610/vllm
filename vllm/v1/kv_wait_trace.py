# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in passive KV wait events. No decisions or CUDA synchronization.

One append-only file per process; timestamps use this host's MONOTONIC clock.
The synchronous observer cost is recorded, never subtracted from latency.
"""

import atexit
from vllm.v1 import host_diagnostic as _hd
import json
import os
import threading
import time
from contextlib import contextmanager

_ROOT = os.environ.get("KV_WAIT_TRACE_DIR")
_LOCK = threading.Lock()
_FD = None
_PID = None
_COUNT = 0
_COST = 0
_ONCE = set()


def emit(name: str, request_id: str | None = None, **fields) -> None:
    """Record a host observation; this does not imply GPU execution."""
    if not _ROOT:
        return
    global _FD, _PID, _COUNT, _COST
    _d0 = _hd.mark()
    begin = time.monotonic_ns()
    with _LOCK:
        _d1 = _hd.mark()
        pid = os.getpid()
        if _FD is None or _PID != pid:
            os.makedirs(_ROOT, exist_ok=True)
            _FD = os.open(
                os.path.join(_ROOT, f"probe-{pid}.jsonl"),
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            _PID = pid
        row = dict(event=name, ts_ns=begin, pid=pid,
                   tid=threading.get_native_id(), request_id=request_id, **fields)
        _d2 = _hd.mark()
        data = (json.dumps(row, separators=(",", ":"), default=str) + "\n").encode()
        _d3 = _hd.mark()
        if os.write(_FD, data) != len(data):
            raise OSError("incomplete KV wait trace write")
        _d4 = _hd.mark()
        _COUNT += 1
        _COST += time.monotonic_ns() - begin
    _hd.record('emit.lock', _d0, _d1)
    _hd.record('emit.serialize', _d2, _d3)
    _hd.record('emit.write', _d3, _d4)


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
        emit("gpu_clock_anchor", request_id, anchor_before_ns=anchor_before,
             anchor_after_ns=time.monotonic_ns())
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
    if _FD is not None and _PID == os.getpid():
        emit("observer_summary", events=_COUNT, synchronous_cost_ns=_COST)
        _hd.dump()
        os.close(_FD)
        _FD = None


atexit.register(close)
