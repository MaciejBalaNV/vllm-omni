from __future__ import annotations

from http import HTTPStatus


class OmniClientError(ValueError):
    """Request-scoped error that should be surfaced as a 4xx response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = HTTPStatus.BAD_REQUEST.value,
        error_type: str = "BadRequestError",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = int(status_code)
        self.error_type = error_type


class GuardrailViolationError(OmniClientError):
    """Raised when a model guardrail rejects request content."""


def client_error_metadata(exc: BaseException) -> tuple[int | None, str | None]:
    if isinstance(exc, OmniClientError):
        return exc.status_code, exc.error_type
    return None, None


def client_error_from_metadata(
    message: str,
    *,
    status_code: int | None,
    error_type: str | None,
) -> OmniClientError:
    return OmniClientError(
        message,
        status_code=status_code or HTTPStatus.BAD_REQUEST.value,
        error_type=error_type or "BadRequestError",
    )


def is_client_error_status(status_code: int | None) -> bool:
    return status_code is not None and 400 <= int(status_code) < 500
