"""Tests for helix_agent.api.client — retry logic, auth, and streaming."""

from __future__ import annotations

import json
import os
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from helix_agent.api.client import AnthropicClient, AuthSource, OAuthTokenSet
from helix_agent.api.error import (
    AuthError,
    HttpError,
    MissingApiKey,
    RetriesExhausted,
)
from helix_agent.api.types import (
    InputMessage,
    MessageRequest,
    MessageResponse,
    TextOutputBlock,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RESPONSE_BODY = json.dumps({
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello!"}],
    "model": "claude-3-5-sonnet-20241022",
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 10,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 5,
    },
})


def _make_request() -> MessageRequest:
    return MessageRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[InputMessage.user_text("hi")],
    )


def _mock_response(status: int, body: str, headers: dict | None = None) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.text = body
    resp.headers = httpx.Headers(headers or {})
    return resp


# ---------------------------------------------------------------------------
# AuthSource
# ---------------------------------------------------------------------------


def test_auth_source_from_env_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    auth = AuthSource.from_env()
    headers: dict = {}
    auth.apply(headers)
    assert headers.get("x-api-key") == "sk-test"


def test_auth_source_from_env_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "bearer-tok")
    auth = AuthSource.from_env()
    headers: dict = {}
    auth.apply(headers)
    assert headers.get("Authorization") == "Bearer bearer-tok"


def test_auth_source_from_env_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(MissingApiKey):
        AuthSource.from_env()


def test_auth_source_both_keys_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "bearer-tok")
    auth = AuthSource.from_env()
    headers: dict = {}
    auth.apply(headers)
    assert "x-api-key" in headers
    assert "Authorization" in headers


# ---------------------------------------------------------------------------
# AnthropicClient construction
# ---------------------------------------------------------------------------


def test_client_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-abc")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    client = AnthropicClient.from_env()
    assert client._auth.api_key() == "sk-abc"


def test_client_builder_chaining() -> None:
    client = (
        AnthropicClient("sk-test")
        .with_base_url("https://custom.example.com/")
        .with_retry_policy(5, 100.0, 1000.0)
    )
    assert client._base_url == "https://custom.example.com"
    assert client._max_retries == 5
    assert client._initial_backoff_ms == 100.0


# ---------------------------------------------------------------------------
# send_message — success
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_send_message_success() -> None:
    client = AnthropicClient("sk-test")
    mock_resp = _mock_response(200, SAMPLE_RESPONSE_BODY, {"request-id": "req_1"})

    with patch.object(client._http, "post", new=AsyncMock(return_value=mock_resp)):
        response = await client.send_message(_make_request())

    assert isinstance(response, MessageResponse)
    assert response.id == "msg_01"
    assert response.request_id == "req_1"
    assert isinstance(response.content[0], TextOutputBlock)


# ---------------------------------------------------------------------------
# send_message — auth errors (not retried)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_send_message_401_raises_auth_error() -> None:
    client = AnthropicClient("sk-bad").with_retry_policy(2, 0.0, 0.0)
    mock_resp = _mock_response(401, '{"error": {"type": "authentication_error"}}')

    with patch.object(client._http, "post", new=AsyncMock(return_value=mock_resp)):
        with pytest.raises(AuthError):
            await client.send_message(_make_request())


@pytest.mark.anyio
async def test_send_message_403_raises_auth_error() -> None:
    client = AnthropicClient("sk-bad").with_retry_policy(2, 0.0, 0.0)
    mock_resp = _mock_response(403, '{"error": {"type": "permission_error"}}')

    with patch.object(client._http, "post", new=AsyncMock(return_value=mock_resp)):
        with pytest.raises(AuthError):
            await client.send_message(_make_request())


# ---------------------------------------------------------------------------
# send_message — retry logic
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_send_message_retries_on_500_then_succeeds() -> None:
    client = AnthropicClient("sk-test").with_retry_policy(2, 0.0, 0.0)
    fail_resp = _mock_response(500, '{"error": {"type": "api_error", "message": "Server Error"}}')
    ok_resp = _mock_response(200, SAMPLE_RESPONSE_BODY)

    post_mock = AsyncMock(side_effect=[fail_resp, ok_resp])
    with patch.object(client._http, "post", new=post_mock):
        response = await client.send_message(_make_request())

    assert post_mock.call_count == 2
    assert response.id == "msg_01"


@pytest.mark.anyio
async def test_send_message_exhausts_retries_on_persistent_500() -> None:
    client = AnthropicClient("sk-test").with_retry_policy(2, 0.0, 0.0)
    fail_resp = _mock_response(500, '{"error": {"type": "api_error", "message": "boom"}}')

    post_mock = AsyncMock(return_value=fail_resp)
    with patch.object(client._http, "post", new=post_mock):
        with pytest.raises(RetriesExhausted) as exc_info:
            await client.send_message(_make_request())

    # max_retries=2 → 3 total attempts
    assert post_mock.call_count == 3
    assert exc_info.value.attempts == 3


@pytest.mark.anyio
async def test_send_message_does_not_retry_on_400() -> None:
    client = AnthropicClient("sk-test").with_retry_policy(2, 0.0, 0.0)
    fail_resp = _mock_response(400, '{"error": {"type": "invalid_request_error"}}')

    post_mock = AsyncMock(return_value=fail_resp)
    with patch.object(client._http, "post", new=post_mock):
        from helix_agent.api.error import ApiResponseError
        with pytest.raises(ApiResponseError) as exc_info:
            await client.send_message(_make_request())

    assert post_mock.call_count == 1
    assert exc_info.value.status == 400


@pytest.mark.anyio
async def test_send_message_retries_on_connect_error_then_succeeds() -> None:
    client = AnthropicClient("sk-test").with_retry_policy(2, 0.0, 0.0)
    ok_resp = _mock_response(200, SAMPLE_RESPONSE_BODY)

    post_mock = AsyncMock(side_effect=[httpx.ConnectError("refused"), ok_resp])
    with patch.object(client._http, "post", new=post_mock):
        response = await client.send_message(_make_request())

    assert post_mock.call_count == 2
    assert response.id == "msg_01"


@pytest.mark.anyio
async def test_send_message_exhausts_retries_on_persistent_connect_error() -> None:
    client = AnthropicClient("sk-test").with_retry_policy(1, 0.0, 0.0)

    post_mock = AsyncMock(side_effect=httpx.ConnectError("refused"))
    with patch.object(client._http, "post", new=post_mock):
        with pytest.raises(RetriesExhausted):
            await client.send_message(_make_request())


# ---------------------------------------------------------------------------
# Backoff
# ---------------------------------------------------------------------------


def test_backoff_for_attempt_zero_within_bounds() -> None:
    client = AnthropicClient("sk-test").with_retry_policy(3, 200.0, 2000.0)
    delay = client._backoff_for_attempt(0)
    # full jitter: [0, min(200, 2000)] = [0, 200ms] → [0, 0.2s]
    assert 0.0 <= delay <= 0.2


def test_backoff_capped_at_max() -> None:
    client = AnthropicClient("sk-test").with_retry_policy(3, 200.0, 500.0)
    # attempt 10 → 200 * 2^10 = 204800 ms >> 500 ms cap
    delay = client._backoff_for_attempt(10)
    assert delay <= 0.5


# ---------------------------------------------------------------------------
# OAuthTokenSet
# ---------------------------------------------------------------------------


def test_oauth_token_set_fields() -> None:
    tok = OAuthTokenSet(
        access_token="access_abc",
        refresh_token="refresh_xyz",
        expires_at=9999999,
        scopes=["read", "write"],
    )
    assert tok.access_token == "access_abc"
    assert tok.refresh_token == "refresh_xyz"
    assert tok.scopes == ["read", "write"]
