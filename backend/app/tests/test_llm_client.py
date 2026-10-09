import json

import pytest

from app.services.llm.client import LLMClient


class _FakeStreamResponse:
    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        chunks = [
            {"choices": [{"delta": {"content": "STREAM"}}]},
            {"choices": [{"delta": {"content": "_OK"}}]},
            {"choices": [], "usage": {"total_tokens": 4}},
        ]
        for chunk in chunks:
            yield f"data: {json.dumps(chunk)}"
        yield "data: [DONE]"


class _FakeStreamContext:
    async def __aenter__(self):
        return _FakeStreamResponse()

    async def __aexit__(self, *_args) -> None:
        return None


class _FakeAsyncClient:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args) -> None:
        return None

    def stream(self, *_args, **_kwargs):
        return _FakeStreamContext()


class _FakeJsonResponse:
    def __init__(self, content: str) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


class _RetryingAsyncClient(_FakeAsyncClient):
    calls = 0

    async def post(self, *_args, **_kwargs):
        type(self).calls += 1
        if type(self).calls == 1:
            return _FakeJsonResponse("not-json")
        return _FakeJsonResponse("```json\n{\"ok\": true}\n```")


@pytest.mark.asyncio
async def test_chat_text_stream_ignores_usage_chunk_without_choices(monkeypatch) -> None:
    monkeypatch.setattr("app.services.llm.client.httpx.AsyncClient", _FakeAsyncClient)
    client = LLMClient()
    client.api_key = "test-key"

    chunks = [
        chunk
        async for chunk in client.chat_text_stream(
            "Reply briefly.",
            [{"role": "user", "content": "test"}],
        )
    ]

    assert "".join(chunks) == "STREAM_OK"


@pytest.mark.asyncio
async def test_chat_json_retries_and_parses_fenced_object(monkeypatch) -> None:
    _RetryingAsyncClient.calls = 0
    monkeypatch.setattr("app.services.llm.client.httpx.AsyncClient", _RetryingAsyncClient)
    client = LLMClient()
    client.api_key = "test-key"

    result = await client.chat_json("system", {"test": True})

    assert result == {"ok": True}
    assert _RetryingAsyncClient.calls == 2
