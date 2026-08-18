# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Authentication coverage for the root RoboLab websocket route."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Awaitable
from typing import Any

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import Receive, Scope, Send
from vllm import envs
from vllm.entrypoints.serve.utils.server_utils import AuthenticationMiddleware


def configured_api_tokens(args: Namespace | Any) -> list[str]:
    """Resolve CLI/environment API tokens with upstream vLLM precedence."""
    configured = getattr(args, "api_key", None)
    if configured:
        candidates = [configured] if isinstance(configured, str) else configured
    else:
        candidates = [envs.VLLM_API_KEY]
    return [str(token) for token in candidates if token]


class RoboLabAuthenticationMiddleware(AuthenticationMiddleware):
    """Apply vLLM bearer-token authentication to the exact root websocket.

    Upstream intentionally authenticates only API prefixes such as ``/v1`` so
    health endpoints stay public. RoboLab requires the otherwise unguarded ``/``
    path, so protect just that websocket without changing health behavior.
    """

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if scope["type"] != "websocket":
            return self.app(scope, receive, send)

        root_path = scope.get("root_path", "")
        url_path = scope["path"].removeprefix(root_path)
        if url_path == "/" and not self.verify_token(Headers(scope=scope)):
            response = JSONResponse(content={"error": "Unauthorized"}, status_code=401)
            return response(scope, receive, send)
        return self.app(scope, receive, send)


def add_robolab_authentication_middleware(app: Any, args: Namespace | Any) -> None:
    """Protect ``/`` whenever vLLM API authentication is configured."""
    if tokens := configured_api_tokens(args):
        app.add_middleware(RoboLabAuthenticationMiddleware, tokens=tokens)
