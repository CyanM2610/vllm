# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import threading
from typing import Any

import pytest

from vllm.v1.kv_offload.tiering.cxl_numa.copy_engine import (
    CopyDirection,
    CopyOperation,
    NumaCopyEngine,
)


def _buffer(contents: bytes) -> ctypes.Array[Any]:
    return ctypes.create_string_buffer(contents, len(contents))


def _operation(
    source: ctypes.Array[Any], destination: ctypes.Array[Any]
) -> CopyOperation:
    return CopyOperation(
        ctypes.addressof(source), ctypes.addressof(destination), len(source)
    )


def test_store_job_copies_all_operations_and_reports_one_result() -> None:
    sources = [_buffer(b"first"), _buffer(b"second")]
    destinations = [_buffer(bytes(5)), _buffer(bytes(6))]
    engine = NumaCopyEngine(1, 1)
    try:
        engine.submit(
            11,
            CopyDirection.STORE,
            [_operation(src, dst) for src, dst in zip(sources, destinations)],
        )
        engine.drain()

        assert [dst.raw for dst in destinations] == [b"first", b"second"]
        [result] = engine.get_finished()
        assert (result.job_id, result.success, result.direction) == (
            11,
            True,
            CopyDirection.STORE,
        )
        assert result.num_bytes == 11
        assert engine.get_finished() == []
        assert engine.inflight_jobs == 0
    finally:
        engine.shutdown()


def test_load_job_records_direction_bytes_and_positive_elapsed_time() -> None:
    source, destination = _buffer(b"load data"), _buffer(bytes(9))
    engine = NumaCopyEngine(1, 1)
    try:
        engine.submit(12, CopyDirection.LOAD, [_operation(source, destination)])
        engine.drain()

        [result] = engine.get_finished()
        assert destination.raw == b"load data"
        assert result.direction is CopyDirection.LOAD
        assert result.num_bytes == 9
        assert result.elapsed_seconds > 0
    finally:
        engine.shutdown()


def test_zero_operation_job_completes_immediately() -> None:
    engine = NumaCopyEngine(1, 1)
    try:
        engine.submit(13, CopyDirection.STORE, [])

        results = engine.get_finished()
        assert isinstance(results, list)
        assert len(results) == 1
        assert (results[0].success, results[0].num_bytes) == (True, 0)
        assert engine.inflight_jobs == 0
    finally:
        engine.shutdown()


def test_duplicate_job_id_is_rejected() -> None:
    engine = NumaCopyEngine(1, 1)
    try:
        engine.submit(14, CopyDirection.LOAD, [])

        with pytest.raises(ValueError, match="duplicate job_id"):
            engine.submit(14, CopyDirection.STORE, [])
    finally:
        engine.shutdown()


def test_job_id_can_be_reused_after_result_is_consumed() -> None:
    engine = NumaCopyEngine(1, 1)
    try:
        engine.submit(14, CopyDirection.LOAD, [])
        assert len(engine.get_finished()) == 1

        engine.submit(14, CopyDirection.STORE, [])
        assert len(engine.get_finished()) == 1
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    ("counts", "error_type"),
    [
        ((0, 1), ValueError),
        ((-1, 1), ValueError),
        ((True, 1), TypeError),
        ((1.5, 1), TypeError),
        ((1, 0), ValueError),
        ((1, -1), ValueError),
        ((1, False), TypeError),
        ((1, 1.5), TypeError),
    ],
)
def test_constructor_rejects_invalid_thread_counts(
    counts: tuple[Any, Any], error_type: type[Exception]
) -> None:
    with pytest.raises(error_type):
        NumaCopyEngine(*counts)


def test_constructor_stops_workers_if_later_thread_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_worker = threading.Event()
    timer = threading.Timer(0.1, release_worker.set)
    timer.start()
    real_start = threading.Thread.start
    started_threads: list[threading.Thread] = []

    def worker(engine: NumaCopyEngine, load_priority: bool) -> None:
        release_worker.wait(5)

    def fail_second_start(thread: threading.Thread) -> None:
        if started_threads:
            raise RuntimeError("injected thread start failure")
        real_start(thread)
        started_threads.append(thread)

    monkeypatch.setattr(NumaCopyEngine, "_worker", worker)
    monkeypatch.setattr(threading.Thread, "start", fail_second_start)
    try:
        with pytest.raises(RuntimeError, match="thread start failure"):
            NumaCopyEngine(1, 1)
        assert len(started_threads) == 1
        assert not started_threads[0].is_alive()
    finally:
        release_worker.set()
        for thread in started_threads:
            thread.join(5)
        timer.join(5)


@pytest.mark.parametrize(
    ("index", "value", "error_type"),
    [
        (0, 0, ValueError),
        (0, True, TypeError),
        (0, 1.5, TypeError),
        (1, -1, ValueError),
        (1, False, TypeError),
        (1, 1.5, TypeError),
        (2, 0, ValueError),
        (2, True, TypeError),
        (2, 1.5, TypeError),
    ],
)
def test_copy_operation_requires_positive_integer_fields(
    index: int, value: Any, error_type: type[Exception]
) -> None:
    arguments = [1, 2, 3]
    arguments[index] = value
    with pytest.raises(error_type):
        CopyOperation(*arguments)


def test_drain_waits_until_all_jobs_finish() -> None:
    source, destination = _buffer(b"wait"), _buffer(bytes(4))
    copy_started = threading.Event()
    release_copy = threading.Event()
    drain_started = threading.Event()
    drain_returned = threading.Event()

    def copy(dst: int, src: int, size: int) -> None:
        copy_started.set()
        if not release_copy.wait(5):
            raise RuntimeError("copy release timed out")
        ctypes.memmove(dst, src, size)

    engine = NumaCopyEngine(1, 1, copy_fn=copy)
    drain_thread: threading.Thread | None = None
    try:
        engine.submit(21, CopyDirection.LOAD, [_operation(source, destination)])
        assert copy_started.wait(5)

        def drain() -> None:
            drain_started.set()
            engine.drain()
            drain_returned.set()

        drain_thread = threading.Thread(target=drain)
        drain_thread.start()
        assert drain_started.wait(5)
        assert not drain_returned.wait(0.05)
        release_copy.set()
        assert drain_returned.wait(5)
        assert destination.raw == b"wait"
    finally:
        release_copy.set()
        if drain_thread is not None:
            drain_thread.join(5)
        engine.shutdown()


def test_shutdown_waits_for_active_copy_before_returning() -> None:
    sources = [_buffer(value) for value in (b"one", b"two", b"tri")]
    destinations = [_buffer(bytes(3)) for _ in sources]
    calls: list[int] = []
    calls_lock = threading.Lock()
    both_workers_started = threading.Event()
    release_copies = threading.Event()
    shutdown_returned = threading.Event()

    def copy(dst: int, src: int, size: int) -> None:
        with calls_lock:
            calls.append(dst)
            if len(calls) == 2:
                both_workers_started.set()
        if not release_copies.wait(5):
            raise RuntimeError("copy release timed out")
        ctypes.memmove(dst, src, size)

    engine = NumaCopyEngine(1, 1, copy_fn=copy)
    shutdown_thread: threading.Thread | None = None
    try:
        engine.submit(
            22,
            CopyDirection.STORE,
            [_operation(src, dst) for src, dst in zip(sources, destinations)],
        )
        assert both_workers_started.wait(5)

        def shutdown() -> None:
            engine.shutdown()
            shutdown_returned.set()

        shutdown_thread = threading.Thread(target=shutdown)
        shutdown_thread.start()
        assert not shutdown_returned.wait(0.05)
        release_copies.set()
        assert shutdown_returned.wait(5)

        assert [dst.raw for dst in destinations] == [b"one", b"two", b"tri"]
        assert len(calls) == 3
        assert all(not thread.is_alive() for thread in engine._threads)
    finally:
        release_copies.set()
        if shutdown_thread is not None:
            shutdown_thread.join(5)
        engine.shutdown()


def test_submit_after_shutdown_raises() -> None:
    engine = NumaCopyEngine(1, 1)
    engine.shutdown()
    engine.shutdown()

    with pytest.raises(RuntimeError, match="shut down"):
        engine.submit(23, CopyDirection.LOAD, [])


def test_idle_load_worker_can_help_store_queue() -> None:
    sources = [_buffer(b"aaa"), _buffer(b"bbb")]
    destinations = [_buffer(bytes(3)), _buffer(bytes(3))]
    store_source, store_destination = _buffer(b"ccc"), _buffer(bytes(3))
    blocker_addresses = {ctypes.addressof(dst) for dst in destinations}
    store_address = ctypes.addressof(store_destination)
    load_started, store_started, helped = (threading.Event() for _ in range(3))
    release_load, release_store = (threading.Event() for _ in range(2))
    helper_names: list[str] = []

    def copy(dst: int, src: int, size: int) -> None:
        name = threading.current_thread().name
        if dst in blocker_addresses:
            started, release = (
                (load_started, release_load)
                if "_load_" in name
                else (store_started, release_store)
            )
            started.set()
            if not release.wait(5):
                raise RuntimeError("worker release timed out")
        elif dst == store_address:
            helper_names.append(name)
            helped.set()
        ctypes.memmove(dst, src, size)

    engine = NumaCopyEngine(1, 1, copy_fn=copy)
    try:
        engine.submit(
            24,
            CopyDirection.LOAD,
            [_operation(src, dst) for src, dst in zip(sources, destinations)],
        )
        assert load_started.wait(5) and store_started.wait(5)
        engine.submit(
            25,
            CopyDirection.STORE,
            [_operation(store_source, store_destination)],
        )

        release_load.set()
        assert helped.wait(5)
        assert helper_names == ["cxl_numa_copy_load_0"]
        assert store_destination.raw == b"ccc"
    finally:
        release_load.set()
        release_store.set()
        engine.shutdown()


def test_copy_exception_drains_operations_and_reports_one_failure() -> None:
    sources = [_buffer(value) for value in (b"bad", b"two", b"tri")]
    destinations = [_buffer(bytes(3)) for _ in sources]
    failing_address = ctypes.addressof(destinations[0])
    calls: list[int] = []

    def copy(dst: int, src: int, size: int) -> None:
        calls.append(dst)
        if dst == failing_address:
            raise RuntimeError("injected failure")
        ctypes.memmove(dst, src, size)

    engine = NumaCopyEngine(1, 1, copy_fn=copy)
    try:
        engine.submit(
            26,
            CopyDirection.STORE,
            [_operation(src, dst) for src, dst in zip(sources, destinations)],
        )
        engine.drain()

        assert len(calls) == 3
        assert [dst.raw for dst in destinations] == [bytes(3), b"two", b"tri"]
        [result] = engine.get_finished()
        assert not result.success
        assert result.num_bytes == 9
    finally:
        engine.shutdown()
