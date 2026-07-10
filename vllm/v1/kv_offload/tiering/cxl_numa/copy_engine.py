# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from functools import cache
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


class CopyDirection(Enum):
    LOAD = "load"
    STORE = "store"


@dataclass(frozen=True)
class CopyOperation:
    src_address: int
    dst_address: int
    size_bytes: int

    def __post_init__(self) -> None:
        for name, value in (
            ("src_address", self.src_address),
            ("dst_address", self.dst_address),
            ("size_bytes", self.size_bytes),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int")
            if value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class CopyJobResult:
    job_id: int
    success: bool
    direction: CopyDirection
    num_bytes: int
    elapsed_seconds: float


@dataclass
class _JobState:
    job_id: int
    direction: CopyDirection
    remaining: int
    num_bytes: int
    started_at: float
    success: bool = True
    lock: threading.Lock = field(default_factory=threading.Lock)

    def task_done(self, success: bool) -> CopyJobResult | None:
        with self.lock:
            self.success = self.success and success
            self.remaining -= 1
            if self.remaining != 0:
                return None
            return CopyJobResult(
                job_id=self.job_id,
                success=self.success,
                direction=self.direction,
                num_bytes=self.num_bytes,
                elapsed_seconds=time.perf_counter() - self.started_at,
            )


@cache
def _get_libc() -> Any:
    libc = ctypes.CDLL(None)
    libc.memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    libc.memcpy.restype = ctypes.c_void_p
    return libc


class NumaCopyEngine:
    def __init__(
        self,
        n_load_threads: int,
        n_store_threads: int,
        *,
        copy_fn: Callable[[int, int, int], object] | None = None,
    ) -> None:
        self._validate_thread_count("n_load_threads", n_load_threads)
        self._validate_thread_count("n_store_threads", n_store_threads)

        self._copy_fn = copy_fn or _get_libc().memcpy
        self._load_queue: deque[tuple[CopyOperation, _JobState]] = deque()
        self._store_queue: deque[tuple[CopyOperation, _JobState]] = deque()
        self._finished: deque[CopyJobResult] = deque()
        self._known_job_ids: set[int] = set()
        self._condition = threading.Condition()
        self._shutdown_lock = threading.Lock()
        self._inflight_jobs = 0
        self._accepting = True
        self._stop = False
        self._shutdown_complete = False
        self._threads: list[threading.Thread] = []

        try:
            for index in range(n_load_threads):
                self._start_worker(load_priority=True, index=index)
            for index in range(n_store_threads):
                self._start_worker(load_priority=False, index=index)
        except BaseException:
            with self._condition:
                self._accepting = False
                self._stop = True
                self._condition.notify_all()
            for thread in self._threads:
                thread.join()
            raise

    @staticmethod
    def _validate_thread_count(name: str, value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an int")
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    def _start_worker(self, load_priority: bool, index: int) -> None:
        priority = "load" if load_priority else "store"
        thread = threading.Thread(
            target=self._worker,
            args=(load_priority,),
            name=f"cxl_numa_copy_{priority}_{index}",
            daemon=True,
        )
        thread.start()
        self._threads.append(thread)

    @property
    def inflight_jobs(self) -> int:
        with self._condition:
            return self._inflight_jobs

    def submit(
        self,
        job_id: int,
        direction: CopyDirection,
        operations: Iterable[CopyOperation],
    ) -> None:
        if not isinstance(job_id, int) or isinstance(job_id, bool):
            raise TypeError("job_id must be an int")
        if not isinstance(direction, CopyDirection):
            raise TypeError("direction must be a CopyDirection")
        operation_list = tuple(operations)
        if not all(
            isinstance(operation, CopyOperation) for operation in operation_list
        ):
            raise TypeError("operations must contain CopyOperation instances")

        started_at = time.perf_counter()
        state = _JobState(
            job_id=job_id,
            direction=direction,
            remaining=len(operation_list),
            num_bytes=sum(operation.size_bytes for operation in operation_list),
            started_at=started_at,
        )
        with self._condition:
            if not self._accepting:
                raise RuntimeError("NUMA copy engine is shut down")
            if job_id in self._known_job_ids:
                raise ValueError(f"duplicate job_id: {job_id}")
            self._known_job_ids.add(job_id)
            if not operation_list:
                self._finished.append(
                    CopyJobResult(
                        job_id=job_id,
                        success=True,
                        direction=direction,
                        num_bytes=0,
                        elapsed_seconds=time.perf_counter() - started_at,
                    )
                )
                return

            queue = (
                self._load_queue
                if direction is CopyDirection.LOAD
                else self._store_queue
            )
            self._inflight_jobs += 1
            queue.extend((operation, state) for operation in operation_list)
            self._condition.notify_all()

    def get_finished(self) -> list[CopyJobResult]:
        with self._condition:
            results = list(self._finished)
            self._finished.clear()
            self._known_job_ids.difference_update(result.job_id for result in results)
            return results

    def drain(self) -> None:
        with self._condition:
            self._condition.wait_for(lambda: self._inflight_jobs == 0)

    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self._shutdown_complete:
                return
            with self._condition:
                self._accepting = False
            self.drain()
            with self._condition:
                self._stop = True
                self._condition.notify_all()
            for thread in self._threads:
                thread.join()
            self._shutdown_complete = True

    def _worker(self, load_priority: bool) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stop or self._load_queue or self._store_queue
                )
                if self._stop:
                    return
                primary = self._load_queue if load_priority else self._store_queue
                secondary = self._store_queue if load_priority else self._load_queue
                operation, state = primary.popleft() if primary else secondary.popleft()

            success = True
            try:
                self._copy_fn(
                    operation.dst_address,
                    operation.src_address,
                    operation.size_bytes,
                )
            except BaseException as exc:
                success = False
                logger.error(
                    "NUMA %s job %d block copy failed: %s",
                    state.direction.value,
                    state.job_id,
                    exc,
                )

            result = state.task_done(success)
            if result is not None:
                with self._condition:
                    self._finished.append(result)
                    self._inflight_jobs -= 1
                    self._condition.notify_all()
