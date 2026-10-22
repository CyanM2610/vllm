"""Passive, memory-only wall/thread/process CPU measurements, dumped at close.

No CUDA calls, no timing-derived decisions, and no per-observation disk writes.
Wall minus thread CPU includes GIL/locks/off-CPU and is not pure OS runqueue wait.
"""
import atexit
import json
import os
import threading
import time
from contextlib import contextmanager

_rows = {}
_details = []
_dropped = 0


def mark():
    return time.monotonic_ns(), time.thread_time_ns(), time.process_time_ns()


def record(name, start, end, request_id=None):
    global _dropped
    tid = threading.get_native_id()
    wall, cpu, process = (end[i] - start[i] for i in range(3))
    key = (start[0] // 1_000_000_000, tid, name)
    row = _rows.setdefault(key, [0, 0, 0, 0, 0])
    row[0] += 1
    row[1] += wall
    row[2] += cpu
    row[3] += process
    row[4] = max(row[4], wall)
    if not name.startswith('emit.') or wall >= 5_000_000:
        if len(_details) < 200000:
            _details.append((name, request_id, tid, *start, *end))
        else:
            _dropped += 1


@contextmanager
def span(name, request_id=None):
    start = mark()
    try:
        yield
    finally:
        record(name, start, mark(), request_id)


def dump():
    root = os.environ.get('KV_WAIT_TRACE_DIR')
    if not root:
        return
    # Called during normal collection shutdown, after the measured trajectory.
    path = os.path.join(root, 'host-diagnostic-%d.json' % os.getpid())
    os.makedirs(root, exist_ok=True)
    data = dict(schema='passive-host-diagnostic-v1', pid=os.getpid(),
                summary_columns=['second', 'tid', 'name', 'count', 'wall_ns',
                                 'thread_cpu_ns', 'process_cpu_ns', 'max_wall_ns'],
                summary=[[*key, *row] for key, row in list(_rows.items())],
                detail_columns=['name', 'request_id', 'tid', 'start_wall_ns',
                                'start_thread_cpu_ns', 'start_process_cpu_ns',
                                'end_wall_ns', 'end_thread_cpu_ns', 'end_process_cpu_ns'],
                details=list(_details), dropped_details=_dropped,
                boundary='Overlapping process CPU must not be added; wall minus '
                         'thread CPU cannot distinguish GIL, locks, IO and off-CPU.')
    with open(path, 'w') as f:
        json.dump(data, f, separators=(',', ':'))


atexit.register(dump)
