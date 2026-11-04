"""Exercise real factory, LangChain, SDK and Pydantic over controlled HTTP."""
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings

import httpx2
import pytest

from app.config import Settings, get_settings

ROOT = Path(__file__).resolve().parents[1]
SECRET = "private-upstream-payload"
VALID = {"order_id": "001234", "request_type": "exchange", "expected_solution": "换一个新的"}


def settings(**overrides):
    values = dict(_env_file=None, openai_base_url="https://upstream.invalid/v1", openai_model="customer-model", openai_api_key=SECRET)
    values.update(overrides)
    return Settings(**values)


@asynccontextmanager
async def controlled_model(config, content, *, status=200, finish="stop", extra=None, gate=None):
    from app.model import create_model
    requests = []
    responses = []
    async def handler(request):
        requests.append(request)
        if gate is not None:
            gate[0].set()
            await gate[1].wait()
        body = {"error": {"message": SECRET, "param": "response_format", "code": "unsupported_parameter"}} if status != 200 else {
            "id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": config.openai_model,
            "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": content, **(extra or {})}}],
        }
        response = httpx2.Response(status, json=body)
        responses.append(response)
        return response
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        model = create_model(config)
        owned = model.root_async_client
        model.root_async_client = owned.with_options(http_client=client)
        model.async_client = model.root_async_client.chat.completions
        model.http_async_client = client
        await owned.close()
        try:
            yield model, requests, responses
        finally:
            await model.root_async_client.close()


async def request_extract(content, *, config=None, payload=None, **kwargs):
    from app.main import app
    from app.model import get_model
    config = config or settings()
    async with controlled_model(config, content, **kwargs) as (model, requests, responses):
        app.dependency_overrides[get_model] = lambda: model
        app.dependency_overrides[get_settings] = lambda: config
        try:
            async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/api/after-sales/extract", json=payload or {"description": "订单001234，我想换一个新的"})
            return response, requests, responses
        finally:
            app.dependency_overrides.clear()


@pytest.mark.parametrize("method", ["json_mode", "json_schema"])
@pytest.mark.parametrize("result", [VALID, {"order_id": None, "request_type": "unknown", "expected_solution": None}])
def test_three_required_fields_and_explicit_no_tools_wire(method, result):
    response, requests, responses = asyncio.run(request_extract(json.dumps(result), config=settings(structured_output_method=method)))
    assert response.status_code == 200
    assert response.json() == result
    assert len(requests) == 1
    wire = json.loads(requests[0].content)
    assert requests[0].url.path == "/v1/chat/completions"
    assert "tools" not in wire and "tool_choice" not in wire
    assert wire["max_tokens"] == 512 and "max_completion_tokens" not in wire
    assert wire["response_format"]["type"] == ("json_object" if method == "json_mode" else "json_schema")
    if method == "json_schema":
        schema = wire["response_format"]["json_schema"]
        assert schema["strict"] is True
        assert set(schema["schema"]["required"]) == set(VALID)
        assert schema["schema"]["additionalProperties"] is False
    assert all(r.is_closed for r in responses)


@pytest.mark.parametrize("method", ["json_mode", "json_schema"])
@pytest.mark.parametrize("content,code", [
    ("not json " + SECRET, "structured_invalid_json"),
    (json.dumps(VALID)[:-1], "structured_invalid_json"),
    (json.dumps({**VALID, "request_type": "FAKE"}), "structured_invalid_schema"),
    (json.dumps({"order_id": None, "request_type": "unknown"}), "structured_invalid_schema"),
    (json.dumps({**VALID, "refunded": True}), "structured_invalid_schema"),
    (json.dumps({**VALID, "order_id": 1234}), "structured_invalid_schema"),
])
def test_malformed_output_has_fixed_safe_error(method, content, code):
    response, requests, responses = asyncio.run(request_extract(content, config=settings(structured_output_method=method)))
    assert response.status_code == 502
    assert response.json()["code"] == code
    assert SECRET not in response.text and content not in response.text
    assert len(requests) == 1
    assert all(r.is_closed for r in responses)


@pytest.mark.parametrize("method", ["json_mode", "json_schema"])
def test_unsupported_response_format_is_safe_and_never_retried(method):
    response, requests, _ = asyncio.run(request_extract(None, status=400, config=settings(structured_output_method=method)))
    assert response.status_code == 502
    assert response.json()["code"] == "structured_mode_unsupported"
    assert SECRET not in response.text
    assert len(requests) == 1
    assert "tools" not in json.loads(requests[0].content)


@pytest.mark.parametrize("model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-0613"])
def test_library_known_tool_fallback_is_rejected_before_wire_without_warning(model):
    before = list(warnings.filters)
    with warnings.catch_warnings(record=True) as recorded:
        response, requests, _ = asyncio.run(request_extract(json.dumps(VALID), config=settings(openai_model=model, structured_output_method="json_schema")))
    assert response.status_code == 502
    assert response.json()["code"] == "structured_mode_unsupported"
    assert requests == []
    assert not recorded
    assert warnings.filters == before


@pytest.mark.parametrize("description", ["", "   ", "x" * 8001, 123])
def test_input_validation_before_upstream(description):
    response, requests, _ = asyncio.run(request_extract("", payload={"description": description}))
    assert response.status_code == 422
    assert requests == []


def test_budget_rejects_before_upstream():
    response, requests, _ = asyncio.run(request_extract("", config=settings(context_window_tokens=1025)))
    assert response.status_code == 413
    assert response.json()["code"] == "context_too_large"
    assert requests == []


def test_missing_config_before_upstream():
    from app.main import app
    from fastapi.testclient import TestClient
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)
    try:
        with TestClient(app) as client:
            response = client.post("/api/after-sales/extract", json={"description": "退款"})
        assert response.status_code == 503
        assert response.json()["code"] == "model_not_configured"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("method", ["json_mode", "json_schema"])
def test_tool_response_refusal_and_truncation_are_rejected(method):
    for extra, finish in [({"tool_calls": [{"id": "call1", "type": "function", "function": {"name": "fake", "arguments": "{}"}}]}, "tool_calls"), ({"refusal": "refused " + SECRET}, "stop"), ({}, "length")]:
        response, _, _ = asyncio.run(request_extract(json.dumps(VALID), extra=extra, finish=finish, config=settings(structured_output_method=method)))
        assert response.status_code == 502
        assert response.json()["code"] == "structured_unsupported_response"
        assert SECRET not in response.text


def test_cancel_closes_real_response_scope():
    async def run():
        from app.service import extract_after_sales
        gate = (asyncio.Event(), asyncio.Event())
        async with controlled_model(settings(), json.dumps(VALID), gate=gate) as (model, requests, _):
            task = asyncio.create_task(extract_after_sales("退款", settings(), model))
            await gate[0].wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(requests) == 1
            assert model.http_async_client.event_hooks["response"] == []
    asyncio.run(run())


def test_evaluation_cli_from_project_root_without_configuration(tmp_path):
    output = tmp_path / "eval.json"
    env = dict(os.environ, OPENAI_BASE_URL="", OPENAI_MODEL="", OPENAI_API_KEY="")
    result = subprocess.run([sys.executable, "scripts/evaluate_prompts.py", "--output", str(output)], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "ModuleNotFoundError" not in result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "not_executed"
    assert report["reason"] == "model_not_configured"
    assert len(report["extraction"]) == 9
    assert len(report["customer_service"]) == 2


def test_final_wire_guard_blocks_future_or_injected_tool_payload():
    async def run():
        from app.errors import AppError
        from app.service import extract_after_sales
        async with controlled_model(settings(), json.dumps(VALID)) as (model, requests, _):
            model.extra_body["tools"] = [{"type": "function", "function": {"name": "fake", "parameters": {"type": "object"}}}]
            with pytest.raises(AppError) as error:
                await extract_after_sales("退款", settings(), model)
            assert error.value.code == "structured_mode_unsupported"
            assert requests == []
            assert model.http_async_client.event_hooks["request"] == []
    asyncio.run(run())


def test_http_disconnect_cancels_pending_extraction():
    async def run():
        from app.main import app
        from app.model import get_model
        gate = (asyncio.Event(), asyncio.Event())
        async with controlled_model(settings(), json.dumps(VALID), gate=gate) as (model, requests, _):
            app.dependency_overrides[get_model] = lambda: model
            app.dependency_overrides[get_settings] = lambda: settings()
            incoming = asyncio.Queue()
            await incoming.put({"type": "http.request", "body": b'{"description":"refund"}', "more_body": False})
            sent = []
            async def send(message):
                sent.append(message)
            scope = {"type": "http", "asgi": {"version": "3.0"}, "method": "POST", "path": "/api/after-sales/extract", "raw_path": b"/api/after-sales/extract", "query_string": b"", "headers": [(b"content-type", b"application/json")], "scheme": "http", "server": ("test", 80), "client": ("client", 1), "root_path": ""}
            task = asyncio.create_task(app(scope, incoming.get, send))
            try:
                await asyncio.wait_for(gate[0].wait(), timeout=2)
                await incoming.put({"type": "http.disconnect"})
                done, _ = await asyncio.wait({task}, timeout=0.2)
                assert done, "HTTP disconnect must cancel the pending upstream call"
                assert not any(m.get("status") == 200 for m in sent)
                assert model.http_async_client.event_hooks["response"] == []
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                app.dependency_overrides.clear()
    asyncio.run(run())
