"""SSE boundary tests with real ChatDeepSeek and controlled OpenAI wire data."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from contextlib import asynccontextmanager, nullcontext

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_deepseek import ChatDeepSeek

from app.config import Settings, get_settings

SECRET = "test-upstream-secret-must-not-escape"


def settings(**values):
    return Settings(
        _env_file=None,
        deepseek_base_url="https://upstream.invalid/v1",
        deepseek_model="customer-model",
        deepseek_api_key=SECRET,
        **values,
    )


def chunk(delta=None, finish=None):
    return ("data: " + json.dumps({
        "id": "chatcmpl-test", "object": "chat.completion.chunk",
        "created": 1, "model": "customer-model",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
    }, ensure_ascii=False) + "\n\n").encode()


class WireStream(httpx.AsyncByteStream):
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
            return httpx.Response(status, json={"error": {"message": SECRET}})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        # Keep every factory choice and real LangChain serializer; replace only
        # the HTTP transport using the SDK's public with_options interface.
        from app.model import create_model
        model = create_model(settings())
        owned_sdk = model.root_async_client
        model.root_async_client = owned_sdk.with_options(http_client=client)
        model.async_client = model.root_async_client.chat.completions
        model.http_async_client = client
        await owned_sdk.close()
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
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
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
    assert wire["max_tokens"] == 512
    assert "max_completion_tokens" not in wire
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
    assert isinstance(model, ChatDeepSeek)
    assert model.model_name == "customer-model"
    assert model.api_base == "https://upstream.invalid/v1"
    assert model.openai_api_key.get_secret_value() == SECRET
    assert model.request_timeout == 7
    assert model.max_retries == 0
    assert model.temperature is None
    assert model.stream_usage is False
    assert model.use_responses_api is False
    assert isinstance(model.http_async_client, httpx.AsyncClient)
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
    (httpx.ReadTimeout(SECRET), "upstream_timeout"),
    (httpx.ReadError(SECRET), "upstream_connection"),
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


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
def test_real_factory_sends_only_the_selected_token_limit_field(field):
    async def run():
        from app.model import create_model
        from app.context import build_chat_messages
        from app.schemas import ChatRequest
        from app.service import stream_chat
        requests = []

        async def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=WireStream([
                chunk({"content": "answer"}), chunk(finish="stop"),
            ]))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            config = settings(deepseek_token_limit_field=field, max_output_tokens=137)
            model = create_model(config)
            owned_sdk = model.root_async_client
            model.root_async_client = owned_sdk.with_options(http_client=client)
            model.async_client = model.root_async_client.chat.completions
            model.http_async_client = client
            await owned_sdk.close()
            events = [event async for event in stream_chat(
                build_chat_messages(ChatRequest(message="hello"), config), model,
            )]
            assert events[-1] == ("done", {"finish_reason": "stop"})
        assert len(requests) == 1
        assert {name: requests[0][name] for name in ("max_tokens", "max_completion_tokens") if name in requests[0]} == {field: 137}
        assert "extra_body" not in requests[0]
    asyncio.run(run())


@pytest.mark.parametrize("diagnostic_mode", ["stderr", "logging"])
def test_malformed_response_diagnostics_are_safe_in_real_concurrent_runtime(diagnostic_mode):
    # A fresh interpreter bypasses pytest's warning recorder and exercises the
    # actual default stderr / logging.captureWarnings rendering paths.
    program = r'''
import asyncio
import logging
import sys
import warnings
from pathlib import Path
root = Path(sys.argv[1])
sys.path[:0] = [str(root), str(root / "tests")]
from test_chat import controlled_model, chunk, settings
from app.context import build_chat_messages
from app.schemas import ChatRequest
from app.service import stream_chat
from app.diagnostics import install_safe_diagnostics
initial_formatter = warnings.formatwarning
install_safe_diagnostics()
install_safe_diagnostics()
assert warnings.formatwarning is initial_formatter
if sys.argv[2] == "logging":
    logging.basicConfig(level=logging.WARNING)
    logging.captureWarnings(True)
warnings.simplefilter("always")
filters_before = list(warnings.filters)
formatter_before = warnings.formatwarning
SECRET_A = "LEAK_A_PRIVATE"
SECRET_B = "LEAK_B_PRIVATE"
async def run():
    gate = asyncio.Event()
    async with controlled_model([
        chunk({"content": [{"type": "text", "text": "answer-a"}, {"type": "reasoning", "reasoning": SECRET_A}]}),
        chunk(finish="stop"),
    ], gate=gate) as (first, first_wire, _), controlled_model([
        chunk({"content": [{"type": "text", "text": "answer-b"}, {"type": "reasoning", "reasoning": SECRET_B}]}),
        chunk(finish="stop"),
    ], gate=gate) as (second, second_wire, _):
        prepared = build_chat_messages(ChatRequest(message="hello"), settings())
        async def collect(model):
            return [event async for event in stream_chat(prepared, model)]
        tasks = [asyncio.create_task(collect(first)), asyncio.create_task(collect(second))]
        try:
            await asyncio.wait_for(asyncio.gather(first_wire.waiting.wait(), second_wire.waiting.wait()), 3)
            warnings.warn("unrelated-warning-still-visible", UserWarning)
            warnings.warn("Pydantic serializer warnings: unrelated-module-kept-visible", UserWarning)
            gate.set()
            results = await asyncio.gather(*tasks)
            assert all(events[-1] == ("done", {"finish_reason": "stop"}) for events in results)
            assert all(SECRET_A not in str(events) and SECRET_B not in str(events) for events in results)
            assert first_wire.closed and second_wire.closed
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    assert warnings.filters == filters_before
    assert warnings.formatwarning is formatter_before
asyncio.run(run())
print("concurrent-runtime-ok")
'''
    result = subprocess.run(
        [sys.executable, "-c", program, str(Path(__file__).resolve().parents[1]), diagnostic_mode],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "LEAK_A_PRIVATE" not in output
    assert "LEAK_B_PRIVATE" not in output
    assert "unrelated-warning-still-visible" in result.stderr
    assert "Pydantic serializer warnings: unrelated-module-kept-visible" in result.stderr
    assert "Model response serialization warning; details withheld." in result.stderr
    assert "concurrent-runtime-ok" in result.stdout
