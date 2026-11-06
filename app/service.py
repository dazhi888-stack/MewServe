"""Forward only public answer text and finish exactly once."""

import json
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager

import httpx
from openai import BadRequestError, ContentFilterFinishReasonError, LengthFinishReasonError
from pydantic import ValidationError

from langchain_deepseek import ChatDeepSeek

from app.config import Settings
from app.context import PreparedChat, build_extraction_messages
from app.errors import AppError, public_upstream_error, structured_error
from app.schemas import AfterSalesResult
from app.model import model_response_scope


async def stream_chat(
    prepared: PreparedChat, model: ChatDeepSeek,
) -> AsyncIterator[tuple[str, dict]]:
    yield "meta", {
        "estimated_input_tokens": prepared.estimated_input_tokens,
        "dropped_messages": prepared.dropped_messages,
    }
    finish_reason = None
    has_text = False
    unsupported = False
    try:
        async with model_response_scope(model), aclosing(model.astream(prepared.messages)) as upstream:
            async for chunk in upstream:
                blocks = chunk.content_blocks
                if chunk.tool_call_chunks or chunk.additional_kwargs.get("function_call"):
                    unsupported = True
                if any(block.get("type") in {"tool_call", "tool_call_chunk"} for block in blocks):
                    unsupported = True
                for block in blocks:
                    if block.get("type") == "text" and (text := block.get("text")):
                        has_text = True
                        if not unsupported:
                            yield "token", {"text": text}
                reason = chunk.response_metadata.get("finish_reason")
                if reason is not None:
                    finish_reason = reason
        if unsupported or finish_reason not in {None, "stop", "length"}:
            raise AppError("upstream_unsupported_response", "模型返回了不支持的响应类型，请重试。", 502)
        if finish_reason is None:
            raise AppError("upstream_stream_incomplete", "模型回复中断，请重试。", 502)
        if not has_text:
            raise AppError("upstream_empty_response", "模型没有返回可显示的回复，请重试。", 502)
    except Exception as exc:
        error = exc if isinstance(exc, AppError) else public_upstream_error(exc)
        yield "error", {"code": error.code, "message": error.message}
        return
    yield "done", {"finish_reason": finish_reason}


async def extract_after_sales(
    description: str, settings: Settings, model: ChatDeepSeek,
) -> AfterSalesResult:
    """Request-owned JSON extraction; never downgrade to tools or repair JSON.

    二次开发：结构化输出收敛为 json_mode（DeepSeek API 仅支持 json_object，
    不支持 OpenAI 的 json_schema 线协议），也不再按模型名做 gpt 系列拦截。
    """
    messages = build_extraction_messages(description, settings)

    async def require_json_wire(request: httpx.Request) -> None:
        # Final serialized SDK payload, before transport. Also fail closed if a
        # future dependency adds another implicit tool fallback.
        payload = json.loads(request.content)
        if ("tools" in payload or "tool_choice" in payload
                or payload.get("response_format", {}).get("type") != "json_object"):
            raise structured_error("structured_mode_unsupported")

    @asynccontextmanager
    async def wire_scope() -> AsyncIterator[None]:
        hooks = model.http_async_client.event_hooks["request"]
        hooks.append(require_json_wire)
        try:
            yield
        finally:
            hooks.remove(require_json_wire)

    try:
        structured = model.with_structured_output(
            AfterSalesResult, method="json_mode", include_raw=True,
        )
        async with wire_scope(), model_response_scope(model):
            envelope = await structured.ainvoke(messages)
        raw = envelope["raw"]
        if (raw.tool_calls or raw.invalid_tool_calls
                or raw.additional_kwargs.get("function_call")
                or raw.additional_kwargs.get("refusal")
                or raw.response_metadata.get("finish_reason") != "stop"):
            raise structured_error("structured_unsupported_response")
        if not isinstance(raw.content, str):
            raise structured_error("structured_unsupported_response")
        # LangChain's json_mode parser accepts partial JSON. Require the exact
        # raw text to be a complete JSON document before trusting its parser.
        decoded = json.loads(raw.content)
        validated = AfterSalesResult.model_validate(decoded)
        if envelope["parsing_error"] is not None or envelope["parsed"] is None:
            raise structured_error("structured_invalid_schema")
        return validated
    except AppError:
        raise
    except json.JSONDecodeError:
        raise structured_error("structured_invalid_json") from None
    except ValidationError as exc:
        code = "structured_invalid_json" if any(
            item["type"] == "json_invalid" for item in exc.errors(include_input=False)
        ) else "structured_invalid_schema"
        raise structured_error(code) from None
    except (LengthFinishReasonError, ContentFilterFinishReasonError):
        raise structured_error("structured_unsupported_response") from None
    except BadRequestError as exc:
        # A structured request rejected for mode/schema is distinct from an
        # authentication/network/timeout error; never expose server messages.
        body = exc.body if isinstance(exc.body, dict) else {}
        detail = body.get("error", body)
        param = detail.get("param") if isinstance(detail, dict) else None
        if param == "response_format" or any(
            marker in str(exc).lower() for marker in ("response_format", "json_schema", "json_object", "structured output")
        ):
            raise structured_error("structured_mode_unsupported") from None
        raise public_upstream_error(exc) from None
    except Exception as exc:
        raise public_upstream_error(exc) from None
