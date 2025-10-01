"""LiteLLM-compatible query function for Claude SDK.

This module provides a drop-in replacement for the standard query() function
that uses LiteLLM instead of Claude Code CLI.
"""

from collections.abc import AsyncIterator

from ._internal.query import Query
from ._internal.transport.litellm import LiteLLMTransport
from .types import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    UserMessage,
)


async def litellm_query(
    *,
    prompt: str,
    options: ClaudeAgentOptions | None = None,
    litellm_base_url: str = "http://localhost:4000",
    litellm_api_key: str = "sk-1234",
    model: str = "gpt-oss-20b-local",
) -> AsyncIterator[UserMessage | AssistantMessage | SystemMessage | ResultMessage | StreamEvent]:
    """Query LiteLLM proxy instead of Claude Code CLI.

    This is a drop-in replacement for the standard query() function that routes
    through LiteLLM instead of requiring Claude Code CLI and Anthropic API key.

    Args:
        prompt: The prompt to send
        options: Optional Claude agent options (some may be ignored)
        litellm_base_url: Base URL of LiteLLM proxy
        litellm_api_key: API key for LiteLLM proxy
        model: Model to use via LiteLLM

    Yields:
        Messages from the LLM via LiteLLM

    Example:
        ```python
        from claude_agent_sdk import litellm_query, ClaudeAgentOptions

        async for message in litellm_query(
            prompt="What is 2+2?",
            model="mistral-7b-local",  # Use local Mistral
            litellm_base_url="http://localhost:4000"
        ):
            print(message)
        ```
    """
    if options is None:
        options = ClaudeAgentOptions()

    # Create LiteLLM transport
    transport = LiteLLMTransport(
        prompt=prompt,
        options=options,
        litellm_base_url=litellm_base_url,
        litellm_api_key=litellm_api_key,
        model=model,
    )

    # Use the standard Query class with our custom transport
    query_instance = Query(
        prompt=prompt,
        options=options,
        transport=transport,
    )

    try:
        async for message in query_instance.stream():
            yield message
    finally:
        await query_instance.close()


__all__ = ["litellm_query"]
