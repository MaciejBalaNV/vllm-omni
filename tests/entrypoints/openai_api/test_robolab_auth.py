# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from vllm_omni.entrypoints.openpi.auth import (
    RoboLabAuthenticationMiddleware,
    configured_api_tokens,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _websocket_scope(path: str, *, authorization: str | None = None) -> dict:
    headers = []
    if authorization is not None:
        headers.append((b"authorization", authorization.encode()))
    return {
        "type": "websocket",
        "path": path,
        "root_path": "",
        "headers": headers,
    }


def test_robolab_auth_uses_cli_token_precedence(monkeypatch):
    monkeypatch.setattr("vllm_omni.entrypoints.openpi.auth.envs.VLLM_API_KEY", "environment-token")

    assert configured_api_tokens(SimpleNamespace(api_key=["cli-token"])) == ["cli-token"]
    assert configured_api_tokens(SimpleNamespace(api_key=None)) == ["environment-token"]


def test_robolab_root_websocket_rejects_missing_bearer_token():
    downstream_calls = []
    sent_messages = []

    async def downstream(scope, receive, send):
        downstream_calls.append(scope)

    async def receive():
        return {"type": "websocket.disconnect"}

    async def send(message):
        sent_messages.append(message)

    middleware = RoboLabAuthenticationMiddleware(downstream, tokens=["secret-token"])
    asyncio.run(middleware(_websocket_scope("/"), receive, send))

    assert downstream_calls == []
    assert sent_messages[0]["status"] == 401


def test_robolab_root_websocket_accepts_valid_bearer_token():
    downstream_calls = []

    async def downstream(scope, receive, send):
        downstream_calls.append(scope)

    async def receive():
        return {"type": "websocket.disconnect"}

    async def send(message):
        raise AssertionError(f"unexpected response: {message}")

    middleware = RoboLabAuthenticationMiddleware(downstream, tokens=["secret-token"])
    asyncio.run(
        middleware(
            _websocket_scope("/", authorization="Bearer secret-token"),
            receive,
            send,
        )
    )

    assert len(downstream_calls) == 1


def test_robolab_auth_does_not_change_non_root_routes():
    downstream_calls = []

    async def downstream(scope, receive, send):
        downstream_calls.append(scope)

    async def receive():
        return {"type": "websocket.disconnect"}

    async def send(message):
        raise AssertionError(f"unexpected response: {message}")

    middleware = RoboLabAuthenticationMiddleware(downstream, tokens=["secret-token"])
    asyncio.run(middleware(_websocket_scope("/health"), receive, send))

    assert len(downstream_calls) == 1
