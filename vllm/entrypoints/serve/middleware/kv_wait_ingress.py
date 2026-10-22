# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Passive ASGI ingress boundaries for the opt-in KV wait probe."""

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vllm.v1.kv_wait_trace import emit as wait_emit


class KVWaitIngressMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != "/v1/completions"
        ):
            await self.app(scope, receive, send)
            return

        request_id = next(
            (
                value.decode("latin-1")
                for key, value in scope.get("headers", ())
                if key.lower() == b"x-request-id"
            ),
            None,
        )
        wait_emit("api_request_received", request_id)
        body_bytes = 0
        body_complete = False

        async def receive_traced() -> Message:
            nonlocal body_bytes, body_complete
            message = await receive()
            if message["type"] == "http.request" and not body_complete:
                body_bytes += len(message.get("body", b""))
                if not message.get("more_body", False):
                    body_complete = True
                    wait_emit("api_body_received", request_id, body_bytes=body_bytes)
            return message

        await self.app(scope, receive_traced, send)
