import hashlib
import json
import math
import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx

from .domain import ChatMessage, LLMProvider, LLMSelection
from .settings import Settings

MODEL_UNAVAILABLE = "I couldn't reach the configured language model. Please try again in a moment."
HUGGINGFACE_MODEL_IDS = {
    "Qwen3.8-27B": "Qwen/Qwen3.8-27B",
    "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
    "DeepSeek-V4.1-Flash": "deepseek-ai/DeepSeek-V4.1-Flash",
    "gemma4-31B": "google/gemma-4-31B-it",
}


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0


def _usage_value(value: object) -> int:
    return max(0, int(value)) if isinstance(value, (int, float)) else 0


class ChatClient(Protocol):
    usage: TokenUsage

    async def chat(self, messages: list[ChatMessage]) -> str: ...

    def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]: ...


class OllamaClient:
    def __init__(self, settings: Settings, chat_model: str | None = None):
        self.settings = settings
        self.chat_model = chat_model
        self.usage = TokenUsage()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout) as client:
                response = await client.post(
                    f"{self.settings.ollama_url}/api/embed",
                    json={"model": self.settings.embedding_model, "input": texts},
                )
                response.raise_for_status()
                return response.json()["embeddings"]
        except (httpx.HTTPError, KeyError, TypeError):
            return [self._local_embedding(text) for text in texts]

    def _local_embedding(self, text: str) -> list[float]:
        vector = [0.0] * self.settings.embedding_size
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % len(vector)
            vector[index] += -1.0 if digest[0] & 1 else 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    async def chat(self, messages: list[ChatMessage]) -> str:
        self.usage = TokenUsage()
        if not self.chat_model:
            return MODEL_UNAVAILABLE
        payload = {
            "model": self.chat_model,
            "stream": False,
            "messages": [message.model_dump() for message in messages],
            "options": {"temperature": 0.2},
        }
        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout) as client:
                response = await client.post(f"{self.settings.ollama_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                self.usage = TokenUsage(
                    input_tokens=_usage_value(data.get("prompt_eval_count")),
                    output_tokens=_usage_value(data.get("eval_count")),
                )
                return data["message"]["content"]
        except (httpx.HTTPError, KeyError, TypeError):
            return MODEL_UNAVAILABLE

    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        self.usage = TokenUsage()
        if not self.chat_model:
            yield MODEL_UNAVAILABLE
            return
        payload = {
            "model": self.chat_model,
            "stream": True,
            "messages": [message.model_dump() for message in messages],
            "options": {"temperature": 0.2},
        }
        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout) as client:
                async with client.stream(
                    "POST", f"{self.settings.ollama_url}/api/chat", json=payload
                ) as res:
                    res.raise_for_status()
                    async for line in res.aiter_lines():
                        if line:
                            data = json.loads(line)
                            if data.get("done"):
                                self.usage = TokenUsage(
                                    input_tokens=_usage_value(data.get("prompt_eval_count")),
                                    output_tokens=_usage_value(data.get("eval_count")),
                                )
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
        except (httpx.HTTPError, ValueError, TypeError):
            yield MODEL_UNAVAILABLE


class RemoteChatClient:
    def __init__(self, settings: Settings, model: str, api_key: str | None, api_url: str):
        self.settings = settings
        self.model = model
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.usage = TokenUsage()

    @staticmethod
    def _messages(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
        return [message.model_dump() for message in messages]

    async def _post(self, path: str, *, headers: dict[str, str], payload: dict) -> dict:
        if not self.api_key:
            raise httpx.HTTPError("Provider API key is not configured")
        async with httpx.AsyncClient(timeout=self.settings.request_timeout) as client:
            response = await client.post(f"{self.api_url}{path}", headers=headers, json=payload)
            response.raise_for_status()
            return response.json()

    async def _stream(
        self, path: str, *, headers: dict[str, str], payload: dict
    ) -> AsyncIterator[dict]:
        if not self.api_key:
            raise httpx.HTTPError("Provider API key is not configured")
        async with httpx.AsyncClient(timeout=self.settings.request_timeout) as client:
            async with client.stream(
                "POST", f"{self.api_url}{path}", headers=headers, json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    yield json.loads(raw)


class OpenAIClient(RemoteChatClient):
    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    @staticmethod
    def _text(data: dict) -> str:
        if isinstance(data.get("output_text"), str):
            return data["output_text"]
        parts: list[str] = []
        for output in data.get("output", []):
            for content in output.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    parts.append(content["text"])
        return "".join(parts)

    async def chat(self, messages: list[ChatMessage]) -> str:
        self.usage = TokenUsage()
        try:
            data = await self._post(
                "/responses",
                headers=self.headers,
                payload={"model": self.model, "input": self._messages(messages)},
            )
            usage = data.get("usage", {})
            self.usage = TokenUsage(
                input_tokens=_usage_value(usage.get("input_tokens")),
                output_tokens=_usage_value(usage.get("output_tokens")),
            )
            return self._text(data) or MODEL_UNAVAILABLE
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            return MODEL_UNAVAILABLE

    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        self.usage = TokenUsage()
        try:
            async for event in self._stream(
                "/responses",
                headers=self.headers,
                payload={"model": self.model, "input": self._messages(messages), "stream": True},
            ):
                if event.get("type") == "response.completed":
                    usage = event.get("response", {}).get("usage", {})
                    self.usage = TokenUsage(
                        input_tokens=_usage_value(usage.get("input_tokens")),
                        output_tokens=_usage_value(usage.get("output_tokens")),
                    )
                if event.get("type") == "response.output_text.delta" and event.get("delta"):
                    yield event["delta"]
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            yield MODEL_UNAVAILABLE


class AnthropicClient(RemoteChatClient):
    @property
    def headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _payload(messages: list[ChatMessage], model: str, *, stream: bool = False) -> dict:
        system = "\n\n".join(message.content for message in messages if message.role == "system")
        payload = {
            "model": model,
            "max_tokens": 2048,
            "messages": [
                message.model_dump() for message in messages if message.role != "system"
            ],
            "stream": stream,
        }
        if system:
            payload["system"] = system
        return payload

    async def chat(self, messages: list[ChatMessage]) -> str:
        self.usage = TokenUsage()
        try:
            data = await self._post(
                "/messages", headers=self.headers, payload=self._payload(messages, self.model)
            )
            usage = data.get("usage", {})
            self.usage = TokenUsage(
                input_tokens=_usage_value(usage.get("input_tokens")),
                output_tokens=_usage_value(usage.get("output_tokens")),
            )
            return "".join(
                block.get("text", "")
                for block in data.get("content", [])
                if block.get("type") == "text"
            ) or MODEL_UNAVAILABLE
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            return MODEL_UNAVAILABLE

    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        self.usage = TokenUsage()
        try:
            async for event in self._stream(
                "/messages",
                headers=self.headers,
                payload=self._payload(messages, self.model, stream=True),
            ):
                if event.get("type") == "message_start":
                    usage = event.get("message", {}).get("usage", {})
                    self.usage.input_tokens = _usage_value(usage.get("input_tokens"))
                elif event.get("type") == "message_delta":
                    usage = event.get("usage", {})
                    self.usage.output_tokens = _usage_value(usage.get("output_tokens"))
                delta = event.get("delta", {})
                if event.get("type") == "content_block_delta" and delta.get("text"):
                    yield delta["text"]
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            yield MODEL_UNAVAILABLE


class HuggingFaceClient(RemoteChatClient):
    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _payload(self, messages: list[ChatMessage], *, stream: bool = False) -> dict:
        payload = {"model": self.model, "messages": self._messages(messages), "stream": stream}
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def chat(self, messages: list[ChatMessage]) -> str:
        self.usage = TokenUsage()
        try:
            data = await self._post(
                "/chat/completions", headers=self.headers, payload=self._payload(messages)
            )
            usage = data.get("usage", {})
            self.usage = TokenUsage(
                input_tokens=_usage_value(usage.get("prompt_tokens")),
                output_tokens=_usage_value(usage.get("completion_tokens")),
            )
            return data["choices"][0]["message"]["content"] or MODEL_UNAVAILABLE
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
            return MODEL_UNAVAILABLE

    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        self.usage = TokenUsage()
        try:
            async for event in self._stream(
                "/chat/completions",
                headers=self.headers,
                payload=self._payload(messages, stream=True),
            ):
                usage = event.get("usage") or {}
                if usage:
                    self.usage = TokenUsage(
                        input_tokens=_usage_value(usage.get("prompt_tokens")),
                        output_tokens=_usage_value(usage.get("completion_tokens")),
                    )
                content = (event.get("choices") or [{}])[0].get("delta", {}).get("content", "")
                if content:
                    yield content
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
            yield MODEL_UNAVAILABLE


class LLMClientFactory:
    def __init__(self, settings: Settings):
        self.settings = settings

    def create(self, selection: LLMSelection, api_key: str | None = None) -> ChatClient:
        if selection.provider == LLMProvider.OPENAI:
            return OpenAIClient(
                self.settings,
                selection.model,
                api_key,
                self.settings.openai_api_url,
            )
        if selection.provider == LLMProvider.ANTHROPIC:
            return AnthropicClient(
                self.settings,
                selection.model,
                api_key,
                self.settings.anthropic_api_url,
            )
        if selection.provider == LLMProvider.HUGGINGFACE:
            return HuggingFaceClient(
                self.settings,
                HUGGINGFACE_MODEL_IDS.get(selection.model, selection.model),
                api_key,
                self.settings.huggingface_api_url,
            )
        return OllamaClient(self.settings, selection.model)
