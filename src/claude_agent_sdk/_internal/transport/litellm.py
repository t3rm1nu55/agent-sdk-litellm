"""LiteLLM transport implementation for Claude SDK.

This transport replaces the Claude Code CLI with direct LiteLLM proxy communication,
enabling 100% local operation without Anthropic API requirements.
"""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import anyio
import httpx

from ..._errors import CLIConnectionError
from ...types import ClaudeAgentOptions
from . import Transport

logger = logging.getLogger(__name__)


class LiteLLMTransport(Transport):
    """LiteLLM proxy transport for Claude SDK.

    This transport implementation routes requests through a LiteLLM proxy instead
    of the Claude Code CLI, enabling:
    - 100% local model support (Ollama, etc.)
    - No Anthropic API key required
    - Multi-provider routing (OpenAI, Google, local, etc.)
    - Budget and rate limiting via LiteLLM
    """

    def __init__(
        self,
        prompt: str,
        options: ClaudeAgentOptions,
        litellm_base_url: str = "http://localhost:4000",
        litellm_api_key: str = "sk-1234",
        model: str = "gpt-oss-20b-local",  # Default to local model
    ):
        """Initialize LiteLLM transport.

        Args:
            prompt: User prompt to send
            options: Claude agent options (mostly ignored for LiteLLM)
            litellm_base_url: Base URL of LiteLLM proxy
            litellm_api_key: API key for LiteLLM proxy
            model: Model to use via LiteLLM
        """
        self._prompt = prompt
        self._options = options
        self._litellm_base_url = litellm_base_url.rstrip("/")
        self._litellm_api_key = litellm_api_key
        self._model = model
        self._client: httpx.AsyncClient | None = None
        self._ready = False
        self._response_queue: anyio.MemoryObjectSendStream | None = None
        self._response_receive: anyio.MemoryObjectReceiveStream | None = None

    async def connect(self) -> None:
        """Connect to LiteLLM proxy."""
        if self._client:
            return

        try:
            self._client = httpx.AsyncClient(
                base_url=self._litellm_base_url,
                headers={
                    "Authorization": f"Bearer {self._litellm_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            )

            # Test connection
            response = await self._client.get("/health")
            response.raise_for_status()

            # Create message queue for streaming responses
            self._response_queue, self._response_receive = anyio.create_memory_object_stream(
                max_buffer_size=100
            )

            self._ready = True
            logger.info(f"Connected to LiteLLM proxy at {self._litellm_base_url}")

        except Exception as e:
            raise CLIConnectionError(f"Failed to connect to LiteLLM: {e}") from e

    async def write(self, data: str) -> None:
        """Write data to transport (send request to LiteLLM).

        For LiteLLM, we interpret the data as a prompt and make a chat completion request.
        """
        if not self._ready or not self._client:
            raise CLIConnectionError("LiteLLMTransport is not ready")

        try:
            # Parse the incoming data (expect JSON with prompt)
            try:
                message_data = json.loads(data)
                # Extract prompt from various possible formats
                if isinstance(message_data, dict):
                    if "message" in message_data:
                        prompt = message_data["message"].get("content", str(message_data))
                    elif "content" in message_data:
                        prompt = message_data["content"]
                    else:
                        prompt = str(message_data)
                else:
                    prompt = str(message_data)
            except json.JSONDecodeError:
                # If not JSON, use as-is
                prompt = data

            # Make request to LiteLLM
            request_payload = {
                "model": self._model,
                "messages": [
                    {
                        "role": "system",
                        "content": self._options.system_prompt or "You are a helpful research assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,  # For now, use non-streaming
                "max_tokens": 4096,
            }

            response = await self._client.post(
                "/chat/completions",
                json=request_payload,
            )
            response.raise_for_status()
            response_data = response.json()

            # Extract content from response
            content = response_data["choices"][0]["message"]["content"]

            # Format as Claude SDK message
            message = {
                "type": "assistant",
                "content": [{"type": "text", "text": content}],
            }

            # Queue the message
            if self._response_queue:
                await self._response_queue.send(message)

        except Exception as e:
            self._ready = False
            raise CLIConnectionError(f"Failed to communicate with LiteLLM: {e}") from e

    def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        """Read messages from the transport."""
        return self._read_messages_impl()

    async def _read_messages_impl(self) -> AsyncIterator[dict[str, Any]]:
        """Internal implementation of read_messages."""
        if not self._response_receive:
            raise CLIConnectionError("Transport not connected")

        try:
            async for message in self._response_receive:
                yield message
        except anyio.EndOfStream:
            pass

    async def close(self) -> None:
        """Close the transport connection."""
        self._ready = False

        if self._response_queue:
            await self._response_queue.aclose()
            self._response_queue = None

        if self._response_receive:
            await self._response_receive.aclose()
            self._response_receive = None

        if self._client:
            await self._client.aclose()
            self._client = None

    def is_ready(self) -> bool:
        """Check if transport is ready."""
        return self._ready

    async def end_input(self) -> None:
        """End the input stream.

        For LiteLLM, this signals no more messages will be sent.
        """
        if self._response_queue:
            await self._response_queue.aclose()
            self._response_queue = None


__all__ = ["LiteLLMTransport"]
