"""helix_agent.api.error — API error hierarchy and retry classification."""

from __future__ import annotations

from typing import Optional
import httpx


class ApiError(Exception):
    """Base class for all API errors."""
    pass


class MissingApiKey(ApiError):
    """Raised when no API key or auth token is available."""
    pass


class ExpiredOAuthToken(ApiError):
    """Raised when the OAuth token has expired and refresh failed."""
    pass


class AuthError(ApiError):
    """Raised for authentication/authorization failures (401, 403)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidApiKeyEnv(ApiError):
    """Raised when the API key environment variable is malformed."""

    def __init__(self, var_error: Exception) -> None:
        super().__init__(str(var_error))
        self.var_error = var_error


class HttpError(ApiError):
    """Wraps httpx transport-level errors (connection refused, timeout, etc.)."""

    def __init__(self, cause: Exception) -> None:
        super().__init__(str(cause))
        self.cause = cause

    def is_retryable(self) -> bool:
        """Returns True for connection/timeout/request errors."""
        return isinstance(
            self.cause,
            (
                httpx.ConnectError,
                httpx.TimeoutException,
                httpx.RemoteProtocolError,
                httpx.ReadError,
                httpx.WriteError,
            ),
        )


class IoError(ApiError):
    """Wraps OS-level I/O errors."""

    def __init__(self, cause: OSError) -> None:
        super().__init__(str(cause))
        self.cause = cause


class JsonError(ApiError):
    """Raised when JSON serialization or deserialization fails."""

    def __init__(self, cause: Exception) -> None:
        super().__init__(str(cause))
        self.cause = cause


class ApiResponseError(ApiError):
    """Raised when the API returns a non-2xx status code."""

    def __init__(
        self,
        status: int,
        error_type: Optional[str],
        message: Optional[str],
        body: str,
        retryable: bool,
    ) -> None:
        super().__init__(f"HTTP {status}: {message or body}")
        self.status = status
        self.error_type = error_type
        self.message = message
        self.body = body
        self.retryable = retryable

    def is_retryable(self) -> bool:
        return self.retryable


class RetriesExhausted(ApiError):
    """Raised after all retry attempts have been consumed."""

    def __init__(self, attempts: int, last_error: ApiError) -> None:
        super().__init__(f"Retries exhausted after {attempts} attempts: {last_error}")
        self.attempts = attempts
        self.last_error = last_error


class InvalidSseFrame(ApiError):
    """Raised when an SSE frame cannot be parsed."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class BackoffOverflow(ApiError):
    """Raised when the computed backoff delay would overflow numeric limits."""

    def __init__(self, attempt: int, base_delay_ms: float) -> None:
        super().__init__(
            f"Backoff overflow at attempt {attempt} with base_delay_ms={base_delay_ms}"
        )
        self.attempt = attempt
        self.base_delay_ms = base_delay_ms


def is_retryable_status(status_code: int) -> bool:
    """Returns True for HTTP status codes that indicate a transient error worth retrying."""
    return status_code in {408, 409, 429, 500, 502, 503, 504}
