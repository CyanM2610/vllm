# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional observation points; callbacks never own native synchronization."""

import itertools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any

Observer = Callable[..., None]
_factory: Callable[[], Observer] | None = None
_ids = itertools.count()
_errors: list[str] = []


def set_phase_observer(factory: Callable[[], Observer] | None) -> None:
    """Install a process-local snapshot factory, or disable observation with None."""
    global _factory
    _factory = factory
    if factory is not None:
        _errors.clear()


def capture_phase_observer() -> Observer | None:
    """Capture the producing step now, including for later async consumption."""
    try:
        return _factory() if _factory is not None else None
    except Exception as error:
        _errors.append(repr(error))
        return None


def mark_phase(observer: Observer | None, name: str, **fields: Any) -> None:
    """Notify an optional observer; it must record errors without changing work."""
    if observer is not None:
        try:
            observer(name, **fields)
        except Exception as error:
            _errors.append(repr(error))


@contextmanager
def phase_span(name: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Bracket a host call without waiting for asynchronous work it submits."""
    observer = capture_phase_observer()
    span_id = next(_ids) if observer else None
    mark_phase(observer, name + ".begin", span_id=span_id, **fields)
    try:
        yield fields
    finally:
        mark_phase(observer, name + ".end", span_id=span_id, **fields)


def observed_host_call(name: str) -> Callable:
    """Decorate a host method, preserving its return value and exception."""

    def decorate(function: Callable) -> Callable:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if _factory is None:
                return function(*args, **kwargs)
            fields: dict[str, Any] = {}
            if name == "uniproc_submit":
                method = args[1] if len(args) > 1 else kwargs.get("method")
                fields = dict(
                    method=str(method),
                    worker_step_id=None,
                    signature=None,
                    identity_scope="executor host call",
                )
            if len(args) > 1 and hasattr(args[1], "request_id"):
                fields["request_id"] = args[1].request_id
            if (
                name.startswith("worker_")
                and len(args) > 1
                and isinstance(args[1], str)
            ):
                fields["request_id"] = args[1]
                if len(args) > 2:
                    fields["token_start"] = getattr(args[2], "start", None)
                    fields["token_end"] = getattr(args[2], "end", None)
            with phase_span(name, **fields) as metadata:
                result = function(*args, **kwargs)
                if name == "worker_get_finished":
                    metadata["finished_sending"] = sorted(result[0] or ())
                    metadata["finished_receiving"] = sorted(result[1] or ())
                return result

        return wrapped

    return decorate


def phase_observation_errors() -> list[str]:
    """Return callback errors so a trace can fail validation without changing work."""
    return list(_errors)
