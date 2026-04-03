"""helix_agent.api.client — Anthropic API client with retry logic and streaming."""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

import httpx

from .error import (
    ApiError,
    ApiResponseError,
    AuthError,
    BackoffOverflow,
    HttpError,
    InvalidApiKeyEnv,
    JsonError,
    MissingApiKey,
    RetriesExhausted,
    is_retryable_status,
)
from .sse import SseParser
from .types import MessageRequest, MessageResponse, StreamEvent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
REQUEST_ID_HEADER = "request-id"
ALT_REQUEST_ID_HEADER = "x-request-id"
DEFAULT_INITIAL_BACKOFF_MS = 200
DEFAULT_MAX_BACKOFF_MS = 2000
DEFAULT_MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# OAuth token set
# ---------------------------------------------------------------------------


@dataclass
class OAuthTokenSet:
    """An OAuth 2.0 token bundle (access + optional refresh)."""
    access_token: str
    refresh_token: Optional[str] = None
    expires_at: Optional[int] = None  # Unix timestamp (seconds)
    scopes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AuthSource — tagged union for credential variants
# ---------------------------------------------------------------------------


class AuthSource:
    """Represents the authentication credentials used for API requests.

    Variants:
    - :class:`_NoAuth`         — no credentials (will fail at request time)
    - :class:`_ApiKey`         — ``x-api-key`` header
    - :class:`_BearerToken`    — ``Authorization: Bearer …`` header
    - :class:`_ApiKeyAndBearer`— both headers

    Use the factory methods to construct instances:
    - :meth:`from_env`
    - :meth:`_api_key` / :meth:`_bearer` / :meth:`_api_key_and_bearer`
    """

    # --- factory helpers ---------------------------------------------------

    @classmethod
    def from_env(cls) -> "AuthSource":
        """Read credentials from environment variables.

        Checks ``ANTHROPIC_API_KEY`` and ``ANTHROPIC_AUTH_TOKEN``.
        Raises :exc:`MissingApiKey` when neither is set.
        Raises :exc:`InvalidApiKeyEnv` if the variable exists but cannot be read.
        """
        try:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip() or None
            bearer = os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip() or None
        except Exception as exc:
            raise InvalidApiKeyEnv(exc) from exc

        if api_key and bearer:
            return cls._api_key_and_bearer(api_key, bearer)
        if api_key:
            return cls._api_key(api_key)
        if bearer:
            return cls._bearer(bearer)
        raise MissingApiKey(
            "No API credentials found. Set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN."
        )

    @classmethod
    def _api_key(cls, key: str) -> "_ApiKey":
        return _ApiKey(key=key)

    @classmethod
    def _bearer(cls, token: str) -> "_BearerToken":
        return _BearerToken(token=token)

    @classmethod
    def _api_key_and_bearer(cls, api_key: str, bearer_token: str) -> "_ApiKeyAndBearer":
        return _ApiKeyAndBearer(api_key=api_key, bearer_token=bearer_token)

    # --- abstract interface ------------------------------------------------

    def api_key(self) -> Optional[str]:
        """Return the API key if present, else None."""
        return None

    def bearer_token(self) -> Optional[str]:
        """Return the bearer token if present, else None."""
        return None

    def apply(self, headers: dict) -> dict:
        """Inject credential headers into *headers* and return the modified dict."""
        return headers


@dataclass
class _NoAuth(AuthSource):
    """No credentials."""
    pass


@dataclass
class _ApiKey(AuthSource):
    """API-key-only auth."""
    key: str

    def api_key(self) -> Optional[str]:
        return self.key

    def apply(self, headers: dict) -> dict:
        headers["x-api-key"] = self.key
        return headers


@dataclass
class _BearerToken(AuthSource):
    """Bearer-token-only auth."""
    token: str

    def bearer_token(self) -> Optional[str]:
        return self.token

    def apply(self, headers: dict) -> dict:
        headers["Authorization"] = f"Bearer {self.token}"
        return headers


@dataclass
class _ApiKeyAndBearer(AuthSource):
    """Both API key and bearer token."""
    api_key_val: str = field(init=False)
    bearer_token_val: str = field(init=False)

    # Re-declare with proper names to avoid name clash with base methods
    def __init__(self, api_key: str, bearer_token: str) -> None:
        self.api_key_val = api_key
        self.bearer_token_val = bearer_token

    def api_key(self) -> Optional[str]:
        return self.api_key_val

    def bearer_token(self) -> Optional[str]:
        return self.bearer_token_val

    def apply(self, headers: dict) -> dict:
        headers["x-api-key"] = self.api_key_val
        headers["Authorization"] = f"Bearer {self.bearer_token_val}"
        return headers


# ---------------------------------------------------------------------------
# MessageStream — async iterator over SSE events
# ---------------------------------------------------------------------------


class MessageStream:
    """Async iterator over :class:`StreamEvent` objects from a streaming response.

    Usage::

        stream = await client.stream_message(request)
        async for event in stream:
            ...

    Attributes:
        request_id: The server-assigned request ID from the response headers.
    """

    def __init__(self, response: httpx.Response, request_id: Optional[str]) -> None:
        self._response = response
        self._request_id = request_id
        self._parser = SseParser()
        self._pending: list[StreamEvent] = []
        self._done = False

    @property
    def request_id(self) -> Optional[str]:
        """Server-assigned request ID (from ``request-id`` response header)."""
        return self._request_id

    async def next_event(self) -> Optional[StreamEvent]:
        """Return the next available :class:`StreamEvent`, or ``None`` if exhausted."""
        if self._pending:
            return self._pending.pop(0)
        if self._done:
            remaining = self._parser.finish()
            self._pending.extend(remaining)
            return self._pending.pop(0) if self._pending else None
        async for chunk in self._response.aiter_bytes():
            events = self._parser.push(chunk)
            self._pending.extend(events)
            if self._pending:
                return self._pending.pop(0)
        self._done = True
        return await self.next_event()

    def __aiter__(self) -> "MessageStream":
        return self

    async def __anext__(self) -> StreamEvent:
        event = await self.next_event()
        if event is None:
            raise StopAsyncIteration
        return event


# ---------------------------------------------------------------------------
# AnthropicClient
# ---------------------------------------------------------------------------


class AnthropicClient:
    """Async HTTP client for the Anthropic Messages API.

    Constructor alternatives::

        AnthropicClient(api_key)          # from a known key string
        AnthropicClient.from_auth(auth)   # from an AuthSource instance
        AnthropicClient.from_env()        # read credentials from environment

    Builder methods (all return *self* for chaining)::

        client.with_auth_source(auth)
        client.with_base_url(url)
        client.with_retry_policy(max_retries, initial_backoff_ms, max_backoff_ms)

    API methods (all async)::

        await client.send_message(request)      -> MessageResponse
        await client.stream_message(request)    -> MessageStream
        await client.exchange_oauth_code(...)   -> OAuthTokenSet
        await client.refresh_oauth_token(...)   -> OAuthTokenSet
    """

    def __init__(self, api_key: str) -> None:
        self._auth: AuthSource = AuthSource._api_key(api_key)
        self._base_url: str = os.environ.get(
            "ANTHROPIC_BASE_URL", DEFAULT_BASE_URL
        ).rstrip("/")
        self._max_retries: int = DEFAULT_MAX_RETRIES
        self._initial_backoff_ms: float = float(DEFAULT_INITIAL_BACKOFF_MS)
        self._max_backoff_ms: float = float(DEFAULT_MAX_BACKOFF_MS)
        self._http: httpx.AsyncClient = httpx.AsyncClient(timeout=60.0)

    # --- alternative constructors ------------------------------------------

    @classmethod
    def from_auth(cls, auth: AuthSource) -> "AnthropicClient":
        """Create a client from a pre-built :class:`AuthSource`."""
        instance = cls.__new__(cls)
        instance._auth = auth
        instance._base_url = os.environ.get(
            "ANTHROPIC_BASE_URL", DEFAULT_BASE_URL
        ).rstrip("/")
        instance._max_retries = DEFAULT_MAX_RETRIES
        instance._initial_backoff_ms = float(DEFAULT_INITIAL_BACKOFF_MS)
        instance._max_backoff_ms = float(DEFAULT_MAX_BACKOFF_MS)
        instance._http = httpx.AsyncClient(timeout=60.0)
        return instance

    @classmethod
    def from_env(cls) -> "AnthropicClient":
        """Create a client by reading credentials from environment variables."""
        auth = AuthSource.from_env()
        return cls.from_auth(auth)

    # --- builder methods ---------------------------------------------------

    def with_auth_source(self, auth: AuthSource) -> "AnthropicClient":
        """Replace the current auth source."""
        self._auth = auth
        return self

    def with_base_url(self, url: str) -> "AnthropicClient":
        """Override the base URL (useful for proxies or testing)."""
        self._base_url = url.rstrip("/")
        return self

    def with_retry_policy(
        self,
        max_retries: int,
        initial_backoff_ms: float,
        max_backoff_ms: float,
    ) -> "AnthropicClient":
        """Configure the retry/backoff policy."""
        self._max_retries = max_retries
        self._initial_backoff_ms = initial_backoff_ms
        self._max_backoff_ms = max_backoff_ms
        return self

    # --- internal helpers --------------------------------------------------

    def _build_headers(self, streaming: bool = False) -> dict:
        headers: dict[str, str] = {
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        if streaming:
            headers["accept"] = "text/event-stream"
        self._auth.apply(headers)
        return headers

    def _backoff_for_attempt(self, attempt: int) -> float:
        """Return sleep duration (seconds) for the given attempt index (0-based).

        Uses exponential backoff with full jitter.
        Raises :exc:`BackoffOverflow` if the computed delay is unreasonably large.
        """
        try:
            delay_ms = self._initial_backoff_ms * (2 ** attempt)
        except OverflowError:
            raise BackoffOverflow(attempt, self._initial_backoff_ms)
        capped_ms = min(delay_ms, self._max_backoff_ms)
        # full jitter: uniform random in [0, capped]
        jittered_ms = random.uniform(0, capped_ms)
        return jittered_ms / 1000.0

    def _extract_request_id(self, headers: httpx.Headers) -> Optional[str]:
        return headers.get(REQUEST_ID_HEADER) or headers.get(ALT_REQUEST_ID_HEADER)

    def _parse_error_response(
        self, status: int, body: str, request_id: Optional[str]
    ) -> ApiResponseError:
        """Build an :class:`ApiResponseError` from a non-2xx response body."""
        try:
            data = json.loads(body)
            error_obj = data.get("error", {})
            error_type: Optional[str] = error_obj.get("type")
            message: Optional[str] = error_obj.get("message")
        except Exception:
            error_type = None
            message = None

        if status in {401, 403}:
            # Surface as AuthError for 401/403 — not retryable
            pass

        retryable = is_retryable_status(status)
        return ApiResponseError(
            status=status,
            error_type=error_type,
            message=message,
            body=body,
            retryable=retryable,
        )

    # --- public API methods ------------------------------------------------

    async def send_message(self, request: MessageRequest) -> MessageResponse:
        """Send a non-streaming message request with automatic retries.

        Retries up to :attr:`_max_retries` times on retryable errors using
        exponential backoff with full jitter.
        """
        url = f"{self._base_url}/v1/messages"
        headers = self._build_headers(streaming=False)
        payload = request.to_dict()

        last_error: Optional[ApiError] = None
        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                sleep_s = self._backoff_for_attempt(attempt - 1)
                await asyncio.sleep(sleep_s)
            try:
                response = await self._http.post(
                    url, headers=headers, json=payload
                )
            except httpx.HTTPError as exc:
                last_error = HttpError(exc)
                if not last_error.is_retryable():
                    raise last_error
                continue
            except OSError as exc:
                from .error import IoError
                raise IoError(exc) from exc

            request_id = self._extract_request_id(response.headers)
            body = response.text

            if response.status_code == 401:
                raise AuthError(f"Unauthorized (401): {body}")
            if response.status_code == 403:
                raise AuthError(f"Forbidden (403): {body}")

            if response.status_code >= 400:
                err = self._parse_error_response(response.status_code, body, request_id)
                last_error = err
                if err.is_retryable():
                    continue
                raise err

            # Success
            try:
                data = json.loads(body)
            except Exception as exc:
                raise JsonError(exc) from exc

            return MessageResponse.from_dict(data, request_id=request_id)

        raise RetriesExhausted(
            attempts=self._max_retries + 1,
            last_error=last_error or ApiError("Unknown error"),
        )

    async def stream_message(self, request: MessageRequest) -> MessageStream:
        """Open a streaming SSE connection.

        Returns a :class:`MessageStream` that can be iterated asynchronously.
        Does NOT retry at the stream level — retries must be managed externally
        or by reconstructing the request.
        """
        url = f"{self._base_url}/v1/messages"
        headers = self._build_headers(streaming=True)
        payload = request.with_streaming().to_dict()

        try:
            req = self._http.build_request("POST", url, headers=headers, json=payload)
            response = await self._http.send(req, stream=True)
        except httpx.HTTPError as exc:
            raise HttpError(exc) from exc

        request_id = self._extract_request_id(response.headers)

        if response.status_code == 401:
            body = await response.aread()
            raise AuthError(f"Unauthorized (401): {body.decode()}")
        if response.status_code >= 400:
            body = await response.aread()
            raise self._parse_error_response(
                response.status_code, body.decode(), request_id
            )

        return MessageStream(response=response, request_id=request_id)

    async def exchange_oauth_code(
        self,
        token_url: str,
        code: str,
        redirect_uri: str,
        client_id: str,
        code_verifier: str,
    ) -> OAuthTokenSet:
        """Exchange an authorization code for tokens (OAuth 2.0 PKCE flow)."""
        payload = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "code_verifier": code_verifier,
        }
        try:
            response = await self._http.post(
                token_url,
                data=payload,
                headers={"content-type": "application/x-www-form-urlencoded"},
            )
        except httpx.HTTPError as exc:
            raise HttpError(exc) from exc

        if response.status_code >= 400:
            raise ApiResponseError(
                status=response.status_code,
                error_type=None,
                message=None,
                body=response.text,
                retryable=False,
            )

        try:
            data = json.loads(response.text)
        except Exception as exc:
            raise JsonError(exc) from exc

        return OAuthTokenSet(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token"),
            expires_at=data.get("expires_at"),
            scopes=data.get("scope", "").split() if data.get("scope") else [],
        )

    async def refresh_oauth_token(
        self,
        token_url: str,
        refresh_token: str,
        client_id: str,
    ) -> OAuthTokenSet:
        """Refresh an OAuth 2.0 access token using a refresh token."""
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        }
        try:
            response = await self._http.post(
                token_url,
                data=payload,
                headers={"content-type": "application/x-www-form-urlencoded"},
            )
        except httpx.HTTPError as exc:
            raise HttpError(exc) from exc

        if response.status_code >= 400:
            raise ApiResponseError(
                status=response.status_code,
                error_type=None,
                message=None,
                body=response.text,
                retryable=False,
            )

        try:
            data = json.loads(response.text)
        except Exception as exc:
            raise JsonError(exc) from exc

        return OAuthTokenSet(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", refresh_token),
            expires_at=data.get("expires_at"),
            scopes=data.get("scope", "").split() if data.get("scope") else [],
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()

    async def __aenter__(self) -> "AnthropicClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()
