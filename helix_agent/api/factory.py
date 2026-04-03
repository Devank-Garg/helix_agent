"""helix_agent.api.factory — Provider-agnostic client factory.

Usage::

    from helix_agent.api.factory import create_client, Provider

    client = create_client("anthropic", api_key="sk-ant-...")
    client = create_client("openai",    api_key="sk-...")
    client = create_client("gemini",    api_key="AIza...")
    client = create_client("ollama")                        # no key needed
"""

from __future__ import annotations

import os
from typing import Literal, Optional

from .anthropic import AnthropicClient
from .base import BaseClient
from .openai_compat import OpenAICompatClient

Provider = Literal["anthropic", "openai", "gemini", "ollama"]

# Default base URLs per provider
_BASE_URLS: dict[str, str] = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "ollama": "http://localhost:11434/v1",
}

# Default env-var names for API keys per provider
_ENV_KEYS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "ollama": "",  # Ollama needs no key
}


def create_client(
    provider: Provider,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 60.0,
) -> BaseClient:
    """Return a :class:`BaseClient` configured for *provider*.

    Args:
        provider:  One of ``"anthropic"``, ``"openai"``, ``"gemini"``, ``"ollama"``.
        api_key:   API key string.  If omitted, read from the provider's default
                   environment variable (e.g. ``GEMINI_API_KEY`` for Gemini).
        base_url:  Override the endpoint URL (useful for proxies / local servers).
        timeout:   HTTP timeout in seconds (default 60).

    Returns:
        A :class:`BaseClient` instance ready to use.

    Raises:
        ValueError: If *provider* is unknown or the key is missing for a provider
                    that requires one.
    """
    provider = provider.lower()  # type: ignore[assignment]
    if provider not in _BASE_URLS:
        raise ValueError(
            f"Unknown provider {provider!r}. "
            f"Choose from: {', '.join(_BASE_URLS)}"
        )

    # Resolve API key
    resolved_key = api_key
    if resolved_key is None:
        env_var = _ENV_KEYS.get(provider, "")
        if env_var:
            resolved_key = os.environ.get(env_var, "").strip() or None

    effective_url = (base_url or _BASE_URLS[provider]).rstrip("/")

    if provider == "anthropic":
        if not resolved_key:
            raise ValueError(
                "Anthropic requires an API key. Pass api_key= or set ANTHROPIC_API_KEY."
            )
        client = AnthropicClient(resolved_key)
        client.with_base_url(effective_url)
        return client

    # OpenAI-compatible providers (openai, gemini, ollama)
    return OpenAICompatClient(
        api_key=resolved_key or "",
        base_url=effective_url,
        timeout=timeout,
    )
