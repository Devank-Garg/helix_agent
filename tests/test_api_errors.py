"""Tests for helix_agent.api.error — error hierarchy and retry classification."""

import pytest
import httpx

from helix_agent.api.error import (
    ApiError,
    ApiResponseError,
    AuthError,
    BackoffOverflow,
    HttpError,
    InvalidApiKeyEnv,
    InvalidSseFrame,
    IoError,
    JsonError,
    MissingApiKey,
    RetriesExhausted,
    is_retryable_status,
)


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------


def test_all_errors_are_api_error() -> None:
    assert issubclass(MissingApiKey, ApiError)
    assert issubclass(AuthError, ApiError)
    assert issubclass(InvalidApiKeyEnv, ApiError)
    assert issubclass(HttpError, ApiError)
    assert issubclass(IoError, ApiError)
    assert issubclass(JsonError, ApiError)
    assert issubclass(ApiResponseError, ApiError)
    assert issubclass(RetriesExhausted, ApiError)
    assert issubclass(InvalidSseFrame, ApiError)
    assert issubclass(BackoffOverflow, ApiError)


def test_all_errors_are_exceptions() -> None:
    assert issubclass(ApiError, Exception)


# ---------------------------------------------------------------------------
# Error chaining / attributes
# ---------------------------------------------------------------------------


def test_http_error_stores_cause() -> None:
    cause = httpx.ConnectError("refused")
    err = HttpError(cause)
    assert err.cause is cause
    assert "refused" in str(err)


def test_io_error_stores_cause() -> None:
    cause = OSError("disk full")
    err = IoError(cause)
    assert err.cause is cause


def test_json_error_stores_cause() -> None:
    cause = ValueError("bad json")
    err = JsonError(cause)
    assert err.cause is cause


def test_api_response_error_attributes() -> None:
    err = ApiResponseError(
        status=429,
        error_type="rate_limit_error",
        message="Too many requests",
        body='{"error": {"type": "rate_limit_error"}}',
        retryable=True,
    )
    assert err.status == 429
    assert err.error_type == "rate_limit_error"
    assert err.message == "Too many requests"
    assert err.retryable is True
    assert "429" in str(err)


def test_retries_exhausted_chains_last_error() -> None:
    last = ApiResponseError(500, None, "oops", "oops", True)
    err = RetriesExhausted(attempts=3, last_error=last)
    assert err.attempts == 3
    assert err.last_error is last
    assert "3" in str(err)


def test_invalid_sse_frame_stores_reason() -> None:
    err = InvalidSseFrame("bad frame")
    assert err.reason == "bad frame"


def test_backoff_overflow_stores_fields() -> None:
    err = BackoffOverflow(attempt=10, base_delay_ms=200.0)
    assert err.attempt == 10
    assert err.base_delay_ms == 200.0


# ---------------------------------------------------------------------------
# HttpError.is_retryable
# ---------------------------------------------------------------------------


def test_http_error_retryable_for_connect_error() -> None:
    err = HttpError(httpx.ConnectError("refused"))
    assert err.is_retryable() is True


def test_http_error_retryable_for_timeout() -> None:
    err = HttpError(httpx.TimeoutException("timeout"))
    assert err.is_retryable() is True


def test_http_error_not_retryable_for_generic() -> None:
    err = HttpError(ValueError("not a transport error"))
    assert err.is_retryable() is False


# ---------------------------------------------------------------------------
# ApiResponseError.is_retryable
# ---------------------------------------------------------------------------


def test_api_response_error_retryable_flag_true() -> None:
    err = ApiResponseError(500, None, None, "", retryable=True)
    assert err.is_retryable() is True


def test_api_response_error_retryable_flag_false() -> None:
    err = ApiResponseError(400, None, None, "", retryable=False)
    assert err.is_retryable() is False


# ---------------------------------------------------------------------------
# is_retryable_status
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504])
def test_is_retryable_status_true(status: int) -> None:
    assert is_retryable_status(status) is True


@pytest.mark.parametrize("status", [200, 201, 400, 401, 403, 404, 422])
def test_is_retryable_status_false(status: int) -> None:
    assert is_retryable_status(status) is False


# ---------------------------------------------------------------------------
# raise / catch
# ---------------------------------------------------------------------------


def test_can_catch_specific_subclass_as_api_error() -> None:
    with pytest.raises(ApiError):
        raise AuthError("bad key")


def test_can_catch_specific_subclass_directly() -> None:
    with pytest.raises(AuthError):
        raise AuthError("bad key")
