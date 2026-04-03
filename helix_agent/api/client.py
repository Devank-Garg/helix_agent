"""helix_agent.api.client — backward-compat re-export.

Import from helix_agent.api.anthropic or helix_agent.api.factory instead.
"""

from .anthropic import AnthropicClient, AuthSource, MessageStream, OAuthTokenSet

__all__ = ["AnthropicClient", "AuthSource", "MessageStream", "OAuthTokenSet"]
