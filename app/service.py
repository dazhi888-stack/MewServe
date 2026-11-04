"""Forward only public answer text and finish exactly once."""

from collections.abc import AsyncIterator
from contextlib import aclosing

from langchain_openai import ChatOpenAI

from app.context import PreparedChat
from app.errors import AppError, public_upstream_error
from app.model import model_response_scope


async def stream_chat(
    prepared: PreparedChat, model: ChatOpenAI,
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
