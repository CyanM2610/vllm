# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional retirement observations; callbacks never own native work or waits."""

import itertools
from collections.abc import Callable
from functools import wraps
from typing import Any

_observer: Callable[..., None] | None = None
_errors: list[str] = []
_ids = itertools.count()


def set_retirement_observer(observer: Callable[..., None] | None) -> None:
    """Install a process-local witness callback; None disables observation."""
    global _observer
    _observer = observer
    if observer is not None:
        _errors.clear()


def retirement_errors() -> list[str]:
    """Return observation errors for evidence rejection outside native execution."""
    return list(_errors)


def notify_retirement(name: str, owner: Any, **fields: Any) -> None:
    """Emit a read-only observation without replacing a native result/exception."""
    if _observer is not None:
        try:
            _observer(name, owner, **fields)
        except Exception as error:
            _errors.append(repr(error))


def retirement_call(name: str) -> Callable:
    """Bracket one native call; preserve its arguments, result and exception."""

    def decorate(function: Callable) -> Callable:
        @wraps(function)
        def wrapped(owner: Any, *args: Any, **kwargs: Any) -> Any:
            if _observer is None:
                return function(owner, *args, **kwargs)
            call_id = next(_ids)
            fields = dict(call_id=call_id, args=args, kwargs=kwargs)
            notify_retirement(name, owner, phase="begin", **fields)
            result = None
            error = None
            try:
                result = function(owner, *args, **kwargs)
                return result
            except BaseException as exc:
                error = repr(exc)
                raise
            finally:
                notify_retirement(
                    name, owner, phase="end", result=result, error=error, **fields
                )

        if name in ("pool_allocate", "pool_evict"):
            from vllm.v1.kv_wait_trace import batch_call

            return batch_call(wrapped)
        return wrapped

    return decorate
