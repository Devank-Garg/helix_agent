"""helix_agent.api.base — Abstract base client interface.

All provider-specific clients implement this interface so the rest of the
codebase can work with any LLM provider interchangeably.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from .types import MessageRequest, MessageResponse, StreamEvent


class BaseClient(ABC):
    """Abstract async LLM client.

    Concrete implementations:
    - :class:`helix_agent.api.anthropic.AnthropicClient`
    - :class:`helix_agent.api.openai_compat.OpenAICompatClient`
      (covers OpenAI, Gemini via OpenAI-compat endpoint, and Ollama)
    """

    @abstractmethod
    async def send_message(self, request: MessageRequest) -> MessageResponse:
        """Send a non-streaming request and return the complete response."""
        ...

    @abstractmethod
    def stream_message(self, request: MessageRequest) -> AsyncIterator[StreamEvent]:
        """Stream a request, yielding :class:`StreamEvent` objects as they arrive."""
        ...

    @abstractmethod
    async def aclose(self) -> None:
        """Release underlying HTTP resources."""
        ...

    async def __aenter__(self) -> "BaseClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()
