"""The single DeepSeek Chat model boundary.

二次开发改动：由 langchain-openai.ChatOpenAI 替换为 langchain-deepseek.ChatDeepSeek，
模型走 DeepSeek 官方 API；客户端生命周期（请求级创建/关闭）与流式契约保持不变。
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from anyio import CancelScope
from fastapi import Depends
from langchain_deepseek import ChatDeepSeek

from app.config import Settings, get_settings
from app.errors import AppError
from app.diagnostics import install_safe_diagnostics

# Stable process initialization, shared by chat and later extraction callers.
install_safe_diagnostics()


def create_model(settings: Settings) -> ChatDeepSeek:
    """Create a request-owned model; its async HTTP client must be closed."""
    if not settings.model_configured:
        raise AppError("model_not_configured", "模型尚未配置，请联系服务维护者。", 503)
    return ChatDeepSeek(
        api_base=settings.deepseek_base_url,
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        # ChatDeepSeek renames max_tokens; extra_body is merged by the SDK
        # after that conversion, preserving the explicitly selected wire name.
        max_tokens=None,
        extra_body={settings.deepseek_token_limit_field: settings.max_output_tokens},
        timeout=settings.request_timeout_seconds,
        max_retries=0,
        temperature=None,
        stream_usage=False,
        use_responses_api=False,
        http_async_client=httpx.AsyncClient(timeout=settings.request_timeout_seconds),
    )


@asynccontextmanager
async def model_response_scope(model: ChatDeepSeek) -> AsyncIterator[None]:
    """Own raw response handles without reading or retaining their bodies.

    Closing LangChain's public astream generator while it is suspended at a
    yield does not synchronously close its nested SDK stream in the locked
    version. The public HTTP response hook supplies a separate close handle.
    Models are request-owned; never share a client across simultaneous scopes.
    """
    client = model.http_async_client
    responses: list[httpx.Response] = []

    async def capture(response: httpx.Response) -> None:
        responses.append(response)

    client.event_hooks["response"].append(capture)
    try:
        yield
    finally:
        client.event_hooks["response"].remove(capture)
        with CancelScope(shield=True):
            for response in responses:
                await response.aclose()


async def get_model(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncIterator[ChatDeepSeek]:
    model = create_model(settings)
    try:
        yield model
    finally:
        with CancelScope(shield=True):
            await model.root_async_client.close()
