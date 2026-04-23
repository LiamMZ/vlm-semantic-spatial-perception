"""
OpenAI LLM Client

Concrete LLMClient implementation wrapping the openai SDK.
Supports text-only and multimodal (vision) prompts via base64-encoded images.

Usage:
    from src.llm_interface import OpenAIClient, GenerateConfig, ImagePart

    llm = OpenAIClient(model="gpt-4o-mini", api_key="...")

    # Text only
    response = llm.generate("What is 2+2?")

    # Multimodal
    with open("photo.png", "rb") as f:
        img = ImagePart(data=f.read())
    cfg = GenerateConfig(system_instruction="You are a helpful assistant.")
    response = llm.generate([img, "Describe this image."], config=cfg)

    # JSON mode
    cfg = GenerateConfig(response_mime_type="application/json", temperature=0.0)
    response = llm.generate("Return {\"answer\": 42}", config=cfg)
"""

from __future__ import annotations

import base64
import logging
from typing import Any, AsyncIterator, List, Optional, Union

from src.llm_interface.base import (
    ContentPart,
    GenerateConfig,
    ImagePart,
    LLMClient,
    LLMResponse,
)

try:
    import openai as _openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    _openai = None  # type: ignore

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    """
    LLMClient backed by the openai SDK.

    Args:
        model:   Model identifier (e.g. "gpt-4o-mini", "gpt-4o").
        api_key: OpenAI API key. If omitted, the SDK uses OPENAI_API_KEY env var.
        base_url: Optional custom base URL (e.g. for Azure or local proxies).
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        if not _OPENAI_AVAILABLE:
            raise ImportError("openai is required. Install with: pip install openai")
        self._model = model
        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = _openai.OpenAI(**kwargs)
        self._async_client: Optional[Any] = None  # created lazily

    # ------------------------------------------------------------------
    # LLMClient implementation
    # ------------------------------------------------------------------

    def generate(
        self,
        contents: Union[str, List[ContentPart]],
        config: Optional[GenerateConfig] = None,
    ) -> LLMResponse:
        messages = self._build_messages(contents, config)
        kwargs = self._build_kwargs(config)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **kwargs,
        )
        text = response.choices[0].message.content or ""
        return LLMResponse(
            text=text,
            model=self._model,
            usage_metadata=self._extract_usage(response),
        )

    async def generate_async(
        self,
        contents: Union[str, List[ContentPart]],
        config: Optional[GenerateConfig] = None,
    ) -> LLMResponse:
        client = self._get_async_client()
        messages = self._build_messages(contents, config)
        kwargs = self._build_kwargs(config)
        response = await client.chat.completions.create(
            model=self._model,
            messages=messages,
            **kwargs,
        )
        text = response.choices[0].message.content or ""
        return LLMResponse(
            text=text,
            model=self._model,
            usage_metadata=self._extract_usage(response),
        )

    async def generate_stream(
        self,
        contents: Union[str, List[ContentPart]],
        config: Optional[GenerateConfig] = None,
    ) -> AsyncIterator[str]:
        client = self._get_async_client()
        messages = self._build_messages(contents, config)
        kwargs = self._build_kwargs(config)
        kwargs.pop("stream", None)  # managed here
        async with client.chat.completions.stream(
            model=self._model,
            messages=messages,
            **kwargs,
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta

    async def aclose(self) -> None:
        if self._async_client is not None:
            await self._async_client.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_async_client(self) -> Any:
        if self._async_client is None:
            kwargs: dict = {}
            sync = self._client
            if sync.api_key:
                kwargs["api_key"] = sync.api_key
            if sync.base_url:
                kwargs["base_url"] = str(sync.base_url)
            self._async_client = _openai.AsyncOpenAI(**kwargs)
        return self._async_client

    def _build_messages(
        self,
        contents: Union[str, List[ContentPart]],
        config: Optional[GenerateConfig],
    ) -> List[dict]:
        messages: List[dict] = []

        # System message
        system = config.system_instruction if config else None
        if system:
            messages.append({"role": "system", "content": system})

        # User message — build content parts list
        if isinstance(contents, str):
            messages.append({"role": "user", "content": contents})
            return messages

        parts: List[dict] = []
        for item in contents:
            if isinstance(item, ImagePart):
                b64 = base64.b64encode(item.data).decode("utf-8")
                parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{item.mime_type};base64,{b64}",
                        "detail": "high",
                    },
                })
            else:
                parts.append({"type": "text", "text": str(item)})

        messages.append({"role": "user", "content": parts})
        return messages

    def _build_kwargs(self, config: Optional[GenerateConfig]) -> dict:
        kwargs: dict = {}
        if config is None:
            return kwargs
        kwargs["temperature"] = config.temperature
        kwargs["max_tokens"] = config.max_output_tokens
        if config.top_p != 0.95:  # only pass if non-default to avoid overriding model defaults
            kwargs["top_p"] = config.top_p
        if config.response_mime_type == "application/json" or config.response_json_schema:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    @staticmethod
    def _extract_usage(response: Any) -> Optional[dict]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
