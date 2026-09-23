import asyncio
import json

import httpx

from packages.omni_rag_core.ai import (
    MODEL_UNAVAILABLE,
    AnthropicClient,
    HuggingFaceClient,
    LLMClientFactory,
    OllamaClient,
    OpenAIClient,
)
from packages.omni_rag_core.domain import ChatMessage, LLMSelection
from packages.omni_rag_core.settings import Settings


class Response:
    def __init__(self, data=None, lines=None, fail=False):
        self.data = data
        self.lines = lines or []
        self.fail = fail

    def raise_for_status(self):
        if self.fail:
            raise httpx.HTTPError("nope")

    def json(self):
        return self.data

    async def aiter_lines(self):
        for line in self.lines:
            yield line


class StreamContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *args):
        return False


class Client:
    response = Response()

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, *args, **kwargs):
        return self.response

    def stream(self, *args, **kwargs):
        return StreamContext(self.response)


def settings():
    return Settings(
        _env_file=None,
        embedding_size=8,
        cors_origins=" http://one, ,http://two ",
    )


def test_settings_origins_and_local_embedding():
    config = settings()
    assert config.origins == ["http://one", "http://two"]
    assert config.llm_models["ollama"] == ("gemma4:12b",)
    vector = OllamaClient(config)._local_embedding("Alpha beta alpha")
    assert len(vector) == 8
    assert abs(sum(x * x for x in vector) - 1) < 0.0001


def test_settings_normalizes_environment_model_catalogs():
    config = Settings(
        _env_file=None,
        openai_models="custom-a, custom-b,custom-a",
        anthropic_models="anthropic-custom",
        huggingface_models="org/model",
        ollama_models="local-model:latest",
    )
    assert config.llm_models == {
        "openai": ("custom-a", "custom-b"),
        "anthropic": ("anthropic-custom",),
        "huggingface": ("org/model",),
        "ollama": ("local-model:latest",),
    }


def test_ollama_embed_success_empty_and_fallback(monkeypatch):
    monkeypatch.setattr("packages.omni_rag_core.ai.httpx.AsyncClient", Client)
    ai = OllamaClient(settings(), "gemma4:12b")
    Client.response = Response({"embeddings": [[1, 2]]})
    assert asyncio.run(ai.embed(["hello"])) == [[1, 2]]
    assert asyncio.run(ai.embed([])) == []
    Client.response = Response(fail=True)
    assert len(asyncio.run(ai.embed(["fallback"]))[0]) == 8


def test_ollama_chat_and_stream_success_and_fallback(monkeypatch):
    monkeypatch.setattr("packages.omni_rag_core.ai.httpx.AsyncClient", Client)
    ai = OllamaClient(settings(), "gemma4:12b")
    messages = [ChatMessage(role="user", content="Hi")]
    Client.response = Response(
        {"message": {"content": "Hello"}, "prompt_eval_count": 8, "eval_count": 3}
    )
    assert asyncio.run(ai.chat(messages)) == "Hello"
    assert (ai.usage.input_tokens, ai.usage.output_tokens) == (8, 3)
    Client.response = Response(
        lines=[
            json.dumps({"message": {"content": "Hi"}}),
            "",
            json.dumps(
                {
                    "message": {},
                    "done": True,
                    "prompt_eval_count": 9,
                    "eval_count": 4,
                }
            ),
        ]
    )

    async def collect():
        return "".join([part async for part in ai.stream_chat(messages)])

    assert asyncio.run(collect()) == "Hi"
    assert (ai.usage.input_tokens, ai.usage.output_tokens) == (9, 4)
    Client.response = Response(fail=True)
    assert "couldn't reach" in asyncio.run(ai.chat(messages))
    assert "couldn't reach" in asyncio.run(collect())


def collect_stream(client, messages):
    async def collect():
        return "".join([part async for part in client.stream_chat(messages)])

    return asyncio.run(collect())


def test_openai_responses_chat_stream_and_fallback(monkeypatch):
    monkeypatch.setattr("packages.omni_rag_core.ai.httpx.AsyncClient", Client)
    messages = [ChatMessage(role="system", content="Be helpful"), ChatMessage(role="user", content="Hi")]
    ai = OpenAIClient(settings(), "gpt-5.6-terra", "key", "https://openai.example/v1/")
    Client.response = Response(
        {"output_text": "Direct answer", "usage": {"input_tokens": 11, "output_tokens": 5}}
    )
    assert asyncio.run(ai.chat(messages)) == "Direct answer"
    assert (ai.usage.input_tokens, ai.usage.output_tokens) == (11, 5)
    Client.response = Response(
        {"output": [{"content": [{"type": "output_text", "text": "Nested answer"}]}]}
    )
    assert asyncio.run(ai.chat(messages)) == "Nested answer"
    Client.response = Response(
        lines=[
            "event: response.output_text.delta",
            'data: {"type":"response.output_text.delta","delta":"Hello "}',
            'data: {"type":"response.output_text.delta","delta":"world"}',
            'data: {"type":"response.completed","response":{"usage":{"input_tokens":12,"output_tokens":6}}}',
            "data: [DONE]",
        ]
    )
    assert collect_stream(ai, messages) == "Hello world"
    assert (ai.usage.input_tokens, ai.usage.output_tokens) == (12, 6)
    Client.response = Response(fail=True)
    assert asyncio.run(ai.chat(messages)) == MODEL_UNAVAILABLE
    assert collect_stream(ai, messages) == MODEL_UNAVAILABLE


def test_anthropic_messages_chat_stream_and_missing_key(monkeypatch):
    monkeypatch.setattr("packages.omni_rag_core.ai.httpx.AsyncClient", Client)
    messages = [ChatMessage(role="system", content="System"), ChatMessage(role="user", content="Hi")]
    ai = AnthropicClient(settings(), "claude-sonnet-5", "key", "https://anthropic.example/v1")
    Client.response = Response(
        {
            "content": [{"type": "text", "text": "Hello"}, {"type": "tool_use"}],
            "usage": {"input_tokens": 13, "output_tokens": 7},
        }
    )
    assert asyncio.run(ai.chat(messages)) == "Hello"
    assert (ai.usage.input_tokens, ai.usage.output_tokens) == (13, 7)
    Client.response = Response(
        lines=[
            'data: {"type":"message_start","message":{"usage":{"input_tokens":14}}}',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}',
            'data: {"type":"message_delta","usage":{"output_tokens":8}}',
            'data: {"type":"message_stop"}',
        ]
    )
    assert collect_stream(ai, messages) == "Hi"
    assert (ai.usage.input_tokens, ai.usage.output_tokens) == (14, 8)
    missing = AnthropicClient(settings(), "claude-sonnet-5", None, "https://example.test")
    assert asyncio.run(missing.chat(messages)) == MODEL_UNAVAILABLE
    assert collect_stream(missing, messages) == MODEL_UNAVAILABLE


def test_huggingface_chat_stream_and_factory(monkeypatch):
    monkeypatch.setattr("packages.omni_rag_core.ai.httpx.AsyncClient", Client)
    messages = [ChatMessage(role="user", content="Hi")]
    ai = HuggingFaceClient(settings(), "DeepSeek-R1", "key", "https://hf.example/v1")
    Client.response = Response(
        {
            "choices": [{"message": {"content": "Reasoned answer"}}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 9},
        }
    )
    assert asyncio.run(ai.chat(messages)) == "Reasoned answer"
    assert (ai.usage.input_tokens, ai.usage.output_tokens) == (15, 9)
    Client.response = Response(
        lines=[
            'data: {"choices":[{"delta":{"content":"Token "}}]}',
            'data: {"choices":[{"delta":{"content":"stream"}}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":16,"completion_tokens":10}}',
        ]
    )
    assert collect_stream(ai, messages) == "Token stream"
    assert (ai.usage.input_tokens, ai.usage.output_tokens) == (16, 10)

    factory = LLMClientFactory(settings())
    assert isinstance(
        factory.create(LLMSelection(provider="ollama", model="gemma4:12b")), OllamaClient
    )
    assert isinstance(
        factory.create(LLMSelection(provider="openai", model="gpt-5.6-luna"), "openai-key"), OpenAIClient
    )
    assert isinstance(
        factory.create(LLMSelection(provider="anthropic", model="claude-opus-5"), "anthropic-key"),
        AnthropicClient,
    )
    huggingface = factory.create(
        LLMSelection(provider="huggingface", model="gemma4-31B"), "hf-key"
    )
    assert isinstance(huggingface, HuggingFaceClient)
    assert huggingface.model == "google/gemma-4-31B-it"
