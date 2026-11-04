"""SSE boundary tests with real ChatOpenAI and controlled OpenAI wire data."""

import asyncio
import json
from contextlib import asynccontextmanager, nullcontext

import httpx2
import pytest
from fastapi.testclient import TestClient
from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings

SECRET = "test-upstream-secret-must-not-escape"


def settings(**values):
    return Settings(
        _env_file=None,
        openai_base_url="https://upstream.invalid/v1",
        openai_model="customer-model",
        openai_api_key=SECRET,
        **values,
    )


def chunk(delta=None, finish=None):
    return ("data: " + json.dumps({
        "id": "chatcmpl-test", "object": "chat.completion.chunk",
        "created": 1, "model": "customer-model",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
    }, ensure_ascii=False) + "\n\n").encode()


class WireStream(httpx2.AsyncByteStream):
    def __init__(self, parts, gate=None):
        self.parts = parts
        self.gate = gate
        self.closed = False
        self.waiting = asyncio.Event()

    async def __aiter__(self):
        for index, part in enumerate(self.parts):
            if self.gate is not None and index == 1:
                self.waiting.set()
                await self.gate.wait()
            if isinstance(part, Exception):
                raise part
            yield part

    async def aclose(self):
        self.closed = True


@asynccontextmanager
async def controlled_model(parts, *, status=200, gate=None):
    stream = WireStream(parts, gate)
    requests = []

    async def handler(request):
        requests.append(request)
        if status != 200:
            return httpx2.Response(status, json={"error": {"message": SECRET}})
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        model = ChatOpenAI(
            base_url="https://upstream.invalid/v1", model="customer-model", api_key=SECRET,
            max_tokens=512, timeout=60, max_retries=0, temperature=None,
            stream_usage=False, use_responses_api=False, http_async_client=client,
            http_socket_options=(),
        )
        yield model, stream, requests


def parse_sse(text):
    events = []
    for frame in text.split("\n\n"):
        if frame:
            lines = frame.splitlines()
            assert len(lines) == 2
            assert lines[0].startswith("event: ")
            assert lines[1].startswith("data: ")
            events.append((lines[0][7:], json.loads(lines[1][6:])))
    return events


async def request_chat(parts, payload=None, *, status=200, model_settings=None):
    from app.main import app
    from app.model import get_model

    async with controlled_model(parts, status=status) as (model, stream, requests):
        app.dependency_overrides[get_model] = lambda: model
        app.dependency_overrides[get_settings] = lambda: model_settings or settings()
        try:
            async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/api/chat/stream", json=payload or {"message": "你好"})
            return response, stream, requests
        finally:
            app.dependency_overrides.clear()


def test_first_round_sse_and_chat_completions_wire_contract():
    response, stream, requests = asyncio.run(request_chat([
        chunk({"role": "assistant", "content": ""}),
        chunk({"content": "你好\n{订单}"}), chunk(finish="stop"), b"data: [DONE]\n\n",
    ]))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    events = parse_sse(response.text)
    assert events[0][0] == "meta"
    assert events[0][1]["dropped_messages"] == 0
    assert events[0][1]["estimated_input_tokens"] > 0
    assert events[1:] == [("token", {"text": "你好\n{订单}"}), ("done", {"finish_reason": "stop"})]
    assert "你好" in response.text
    assert stream.closed
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/chat/completions"
    wire = json.loads(requests[0].content)
    assert wire["model"] == "customer-model"
    assert wire["stream"] is True
    assert wire["max_completion_tokens"] == 512
    assert "temperature" not in wire
    assert "stream_options" not in wire
    assert [m["role"] for m in wire["messages"]] == ["system", "user"]
    assert wire["messages"][-1]["content"] == "你好"


def test_second_round_history_and_independent_empty_history():
    async def run():
        response, _, requests = await request_chat([chunk({"content": "请保留包装"}), chunk(finish="stop")], {
            "message": "那运费呢？", "history": [
                {"role": "user", "content": "杯子碎了"},
                {"role": "assistant", "content": "请提供订单号"},
            ],
        })
        assert response.status_code == 200
        assert json.loads(requests[0].content)["messages"][1:] == [
            {"role": "user", "content": "杯子碎了"},
            {"role": "assistant", "content": "请提供订单号"},
            {"role": "user", "content": "那运费呢？"},
        ]
        _, _, next_requests = await request_chat([chunk({"content": "新对话"}), chunk(finish="stop")])
        assert [m["role"] for m in json.loads(next_requests[0].content)["messages"]] == ["system", "user"]
    asyncio.run(run())


@pytest.mark.parametrize("payload", [{"message": " "}, {"message": "hi", "history": [{"role": "system", "content": "x"}]}])
def test_invalid_request_is_422_before_sse(payload):
    from app.main import app
    app.dependency_overrides[get_settings] = lambda: settings()
    try:
        with TestClient(app) as client:
            response = client.post("/api/chat/stream", json=payload)
        assert response.status_code == 422
        assert "text/event-stream" not in response.headers.get("content-type", "")
    finally:
        app.dependency_overrides.clear()


def test_missing_configuration_is_503_before_sse():
    from app.main import app
    with TestClient(app) as client:
        response = client.post("/api/chat/stream", json={"message": "你好"})
    assert response.status_code == 503
    assert response.json()["code"] == "model_not_configured"
    assert "text/event-stream" not in response.headers.get("content-type", "")


def test_context_budget_is_413_before_upstream():
    response, _, requests = asyncio.run(request_chat([], model_settings=settings(
        context_window_tokens=1025, max_output_tokens=512, token_safety_margin=512,
    )))
    assert response.status_code == 413
    assert response.json()["code"] == "context_too_large"
    assert requests == []


def test_model_factory_configuration():
    from app.model import create_model
    model = create_model(settings(request_timeout_seconds=7, max_output_tokens=123))
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "customer-model"
    assert model.openai_api_base == "https://upstream.invalid/v1"
    assert model.openai_api_key.get_secret_value() == SECRET
    assert model.max_tokens == 123
    assert model.request_timeout == 7
    assert model.max_retries == 0
    assert model.temperature is None
    assert model.stream_usage is False
    assert model.use_responses_api is False
    assert isinstance(model.http_async_client, httpx2.AsyncClient)
    asyncio.run(model.root_async_client.close())


class ASGIChat:
    def __init__(self, app, spec_version="2.0"):
        self.app = app
        self.spec_version = spec_version
        self.messages = asyncio.Queue()
        self.incoming = asyncio.Queue()

    async def start(self):
        await self.incoming.put({"type": "http.request", "body": b'{"message":"hello"}', "more_body": False})
        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": self.spec_version},
            "method": "POST", "path": "/api/chat/stream", "raw_path": b"/api/chat/stream",
            "query_string": b"", "headers": [(b"content-type", b"application/json")],
            "scheme": "http", "server": ("test", 80), "client": ("client", 1), "root_path": "",
        }
        self.task = asyncio.create_task(self.app(scope, self.incoming.get, self.messages.put))

    async def next(self):
        return await asyncio.wait_for(self.messages.get(), timeout=2)


@pytest.mark.parametrize("disconnect", [False, True])
def test_incremental_delivery_and_client_disconnect_release_upstream(disconnect):
    async def run():
        from app.main import app
        from app.model import get_model
        gate = asyncio.Event()
        async with controlled_model([
            chunk({"content": "首段\n{中文}"}), chunk({"content": "第二段"}), chunk(finish="stop"),
        ], gate=gate) as (model, stream, _):
            app.dependency_overrides[get_model] = lambda: model
            app.dependency_overrides[get_settings] = lambda: settings()
            session = ASGIChat(app)
            try:
                await session.start()
                assert (await session.next())["status"] == 200
                assert parse_sse((await session.next())["body"].decode())[0][0] == "meta"
                assert parse_sse((await session.next())["body"].decode()) == [("token", {"text": "首段\n{中文}"})]
                await asyncio.wait_for(stream.waiting.wait(), 2)
                assert not gate.is_set()
                assert session.messages.empty()  # No done before upstream completion.
                if disconnect:
                    await session.incoming.put({"type": "http.disconnect"})
                    await asyncio.wait_for(session.task, 2)
                    assert stream.closed
                    assert session.messages.empty()  # No success/error sent after disconnect.
                else:
                    gate.set()
                    assert parse_sse((await session.next())["body"].decode()) == [("token", {"text": "第二段"})]
                    assert parse_sse((await session.next())["body"].decode()) == [("done", {"finish_reason": "stop"})]
                    assert (await session.next())["more_body"] is False
                    await asyncio.wait_for(session.task, 2)
                    assert stream.closed
            finally:
                app.dependency_overrides.clear()
                if not session.task.done():
                    session.task.cancel()
                    await asyncio.gather(session.task, return_exceptions=True)
    asyncio.run(run())


@pytest.mark.parametrize(("status", "code"), [(401, "upstream_authentication"), (429, "upstream_rate_limited"), (500, "upstream_error")])
def test_upstream_http_error_is_safe_sse_error_without_done(status, code):
    response, _, requests = asyncio.run(request_chat([], status=status))
    events = parse_sse(response.text)
    assert response.status_code == 200
    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == code
    assert not any(event == "done" for event, _ in events)
    assert SECRET not in response.text
    assert len(requests) == 1  # No paid automatic retry.


@pytest.mark.parametrize(("failure", "code"), [
    (httpx2.ReadTimeout(SECRET), "upstream_timeout"),
    (httpx2.ReadError(SECRET), "upstream_connection"),
    (ValueError(SECRET), "upstream_error"),
])
def test_midstream_failure_closes_stream_without_done(failure, code):
    response, stream, _ = asyncio.run(request_chat([chunk({"content": "先到的正文"}), failure]))
    events = parse_sse(response.text)
    assert events[1] == ("token", {"text": "先到的正文"})
    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == code
    assert not any(event == "done" for event, _ in events)
    assert SECRET not in response.text
    assert stream.closed


@pytest.mark.parametrize(("parts", "code"), [
    ([chunk(finish="stop")], "upstream_empty_response"),
    ([chunk({"reasoning_content": SECRET}), chunk(finish="stop")], "upstream_empty_response"),
    ([chunk({"content": [{"type": "reasoning", "reasoning": SECRET}]}), chunk(finish="stop")], "upstream_empty_response"),
    ([chunk({"content": "部分正文"}), b"data: [DONE]\n\n"], "upstream_stream_incomplete"),
    ([chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "refund", "arguments": SECRET}}]}), chunk(finish="tool_calls")], "upstream_unsupported_response"),
    ([chunk({"content": "正文"}), chunk(finish="content_filter")], "upstream_unsupported_response"),
])
def test_invalid_completion_never_becomes_success(parts, code):
    # Array-valued delta.content is deliberately outside the Chat Completions
    # string schema. The SDK warns while retaining it for block filtering.
    expected_warning = any(b'"content": [' in part for part in parts)
    with pytest.warns(UserWarning, match="Pydantic serializer warnings") if expected_warning else nullcontext():
        response, stream, _ = asyncio.run(request_chat(parts))
    events = parse_sse(response.text)
    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == code
    assert SECRET not in response.text
    assert not any(event == "done" for event, _ in events)
    assert stream.closed


def test_reasoning_blocks_are_filtered_but_text_and_length_finish_survive():
    with pytest.warns(UserWarning, match="Pydantic serializer warnings"):
        response, stream, _ = asyncio.run(request_chat([
            chunk({"content": [{"type": "reasoning", "reasoning": SECRET}, {"type": "text", "text": "公开正文"}]}),
            chunk(finish="length"),
        ]))
    assert parse_sse(response.text)[1:] == [("token", {"text": "公开正文"}), ("done", {"finish_reason": "length"})]
    assert SECRET not in response.text
    assert stream.closed


def test_closing_service_after_first_token_releases_suspended_upstream():
    async def run():
        from app.context import build_chat_messages
        from app.schemas import ChatRequest
        from app.service import stream_chat
        gate = asyncio.Event()
        async with controlled_model([
            chunk({"content": "首段"}), chunk({"content": "不应继续"}), chunk(finish="stop"),
        ], gate=gate) as (model, wire, _):
            stream = stream_chat(build_chat_messages(ChatRequest(message="hello"), settings()), model)
            assert (await anext(stream))[0] == "meta"
            assert await anext(stream) == ("token", {"text": "首段"})
            await stream.aclose()
            assert wire.closed
    asyncio.run(run())


@pytest.mark.parametrize("send_failure", [False, True])
def test_disconnect_while_sse_send_is_backpressured_releases_response(monkeypatch, send_failure):
    async def run():
        from app.main import app
        from app import model as model_module
        gate = asyncio.Event()
        send_blocked = asyncio.Event()
        async with controlled_model([
            chunk({"content": "首段"}), chunk({"content": "后续"}), chunk(finish="stop"),
        ], gate=gate) as (model, wire, _):
            monkeypatch.setattr(model_module, "create_model", lambda settings: model)
            app.dependency_overrides[get_settings] = lambda: settings()
            session = ASGIChat(app, spec_version="2.4" if send_failure else "2.0")
            real_send = session.messages.put

            async def blocked_send(message):
                await real_send(message)
                if message["type"] == "http.response.body" and b"event: token" in message.get("body", b""):
                    send_blocked.set()
                    if send_failure:
                        raise OSError("client disconnected")
                    await asyncio.Event().wait()

            session.messages.put = blocked_send
            try:
                await session.start()
                assert (await session.next())["status"] == 200
                await session.next()  # meta
                assert parse_sse((await session.next())["body"].decode()) == [("token", {"text": "首段"})]
                await asyncio.wait_for(send_blocked.wait(), 2)
                if send_failure:
                    from starlette.requests import ClientDisconnect
                    with pytest.raises(ClientDisconnect):
                        await asyncio.wait_for(session.task, 2)
                else:
                    await session.incoming.put({"type": "http.disconnect"})
                    await asyncio.wait_for(session.task, 2)
                assert wire.closed
                assert model.http_async_client.is_closed
                assert model.http_async_client.event_hooks["response"] == []
                assert session.messages.empty()
            finally:
                app.dependency_overrides.clear()
                if not session.task.done():
                    session.task.cancel()
                    await asyncio.gather(session.task, return_exceptions=True)
    asyncio.run(run())
