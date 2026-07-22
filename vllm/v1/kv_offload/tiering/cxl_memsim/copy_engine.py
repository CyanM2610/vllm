# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.cxl_memsim.client import (
    CxlMemSimTransferResult,
)

logger = init_logger(__name__)


class _CxlMemSimClientProtocol(Protocol):
    def read(
        self, offset: int, host_address: int, size_bytes: int
    ) -> CxlMemSimTransferResult: ...

    def write(
        self, offset: int, host_address: int, size_bytes: int
    ) -> CxlMemSimTransferResult: ...


class CxlMemSimCopyDirection(Enum):
    LOAD = "load"
    STORE = "store"


@dataclass(frozen=True)
class CxlMemSimCopyOperation:
    host_address: int
    cxl_offset: int
    size_bytes: int

    def __post_init__(self) -> None:
        for name, value in (
            ("host_address", self.host_address),
            ("cxl_offset", self.cxl_offset),
            ("size_bytes", self.size_bytes),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int")
        if self.host_address <= 0:
            raise ValueError("host_address must be positive")
        if self.cxl_offset < 0:
            raise ValueError("cxl_offset must be non-negative")
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")


@dataclass(frozen=True)
class CxlMemSimCopyJobResult:
    job_id: int
    success: bool
    direction: CxlMemSimCopyDirection
    num_bytes: int
    elapsed_seconds: float
    host_copy_seconds: float
    model_seconds: float
    serialization_seconds: float
    cacheline_count: int


@dataclass
class _JobState:
    job_id: int
    direction: CxlMemSimCopyDirection
    remaining: int
    started_at: float
    num_bytes: int = 0
    success: bool = True
    host_copy_seconds: float = 0.0
    model_seconds: float = 0.0
    serialization_seconds: float = 0.0
    cacheline_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def task_done(
        self, transfer: CxlMemSimTransferResult | None
    ) -> CxlMemSimCopyJobResult | None:
        with self.lock:
            self.success = self.success and transfer is not None
            if transfer is not None:
                self.num_bytes += transfer.num_bytes
                self.host_copy_seconds += transfer.host_copy_seconds
                self.model_seconds += transfer.model_seconds
                self.serialization_seconds += transfer.serialization_seconds
                self.cacheline_count += transfer.cacheline_count
            self.remaining -= 1
            if self.remaining != 0:
                return None
            return CxlMemSimCopyJobResult(
                job_id=self.job_id,
                success=self.success,
                direction=self.direction,
                num_bytes=self.num_bytes,
                elapsed_seconds=time.perf_counter() - self.started_at,
                host_copy_seconds=self.host_copy_seconds,
                model_seconds=self.model_seconds,
                serialization_seconds=self.serialization_seconds,
                cacheline_count=self.cacheline_count,
            )


class CxlMemSimCopyEngine:
    def __init__(
        self,
        client: _CxlMemSimClientProtocol,
        n_load_threads: int,
        n_store_threads: int,
    ) -> None:
        self._validate_thread_count("n_load_threads", n_load_threads)
        self._validate_thread_count("n_store_threads", n_store_threads)
        self._client = client
        self._load_queue: deque[tuple[CxlMemSimCopyOperation, _JobState]] = deque()
        self._store_queue: deque[tuple[CxlMemSimCopyOperation, _JobState]] = deque()
        self._finished: deque[CxlMemSimCopyJobResult] = deque()
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
            name=f"cxl_memsim_copy_{priority}_{index}",
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
        direction: CxlMemSimCopyDirection,
        operations: Iterable[CxlMemSimCopyOperation],
    ) -> None:
        if not isinstance(job_id, int) or isinstance(job_id, bool):
            raise TypeError("job_id must be an int")
        if not isinstance(direction, CxlMemSimCopyDirection):
            raise TypeError("direction must be a CxlMemSimCopyDirection")
        operation_list = tuple(operations)
        if not all(
            isinstance(operation, CxlMemSimCopyOperation)
            for operation in operation_list
        ):
            raise TypeError("operations must contain CxlMemSimCopyOperation instances")

        started_at = time.perf_counter()
        state = _JobState(
            job_id=job_id,
            direction=direction,
            remaining=len(operation_list),
            started_at=started_at,
        )
        with self._condition:
            if not self._accepting:
                raise RuntimeError("CXLMemSim copy engine is shut down")
            if job_id in self._known_job_ids:
                raise ValueError(f"duplicate job_id: {job_id}")
            self._known_job_ids.add(job_id)
            if not operation_list:
                self._finished.append(
                    CxlMemSimCopyJobResult(
                        job_id=job_id,
                        success=True,
                        direction=direction,
                        num_bytes=0,
                        elapsed_seconds=time.perf_counter() - started_at,
                        host_copy_seconds=0.0,
                        model_seconds=0.0,
                        serialization_seconds=0.0,
                        cacheline_count=0,
                    )
                )
                return

            queue = (
                self._load_queue
                if direction is CxlMemSimCopyDirection.LOAD
                else self._store_queue
            )
            self._inflight_jobs += 1
            queue.extend((operation, state) for operation in operation_list)
            self._condition.notify_all()

    def get_finished(self) -> list[CxlMemSimCopyJobResult]:
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

            transfer = None
            try:
                transfer = (
                    self._client.read(
                        operation.cxl_offset,
                        operation.host_address,
                        operation.size_bytes,
                    )
                    if state.direction is CxlMemSimCopyDirection.LOAD
                    else self._client.write(
                        operation.cxl_offset,
                        operation.host_address,
                        operation.size_bytes,
                    )
                )
                if transfer.num_bytes != operation.size_bytes:
                    raise RuntimeError(
                        f"CXLMemSim copied {transfer.num_bytes} of "
                        f"{operation.size_bytes} bytes"
                    )
            except BaseException as exc:
                transfer = None
                logger.error(
                    "CXLMemSim %s job %d block copy failed: %s",
                    state.direction.value,
                    state.job_id,
                    exc,
                )

            result = state.task_done(transfer)
            if result is not None:
                with self._condition:
                    self._finished.append(result)
                    self._inflight_jobs -= 1
                    self._condition.notify_all()
