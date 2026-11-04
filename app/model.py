"""The single OpenAI-compatible Chat Completions model boundary."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import httpx2
from anyio import CancelScope
from fastapi import Depends
from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings
from app.errors import AppError


def create_model(settings: Settings) -> ChatOpenAI:
    """Create a request-owned model; its async HTTP client must be closed."""
    if not settings.model_configured:
        raise AppError("model_not_configured", "模型尚未配置，请联系服务维护者。", 503)
    return ChatOpenAI(
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        max_tokens=settings.max_output_tokens,
        timeout=settings.request_timeout_seconds,
        max_retries=0,
        temperature=None,
        stream_usage=False,
        use_responses_api=False,
        http_socket_options=(),
        http_async_client=httpx2.AsyncClient(timeout=settings.request_timeout_seconds),
    )


@asynccontextmanager
async def model_response_scope(model: ChatOpenAI) -> AsyncIterator[None]:
    """Own raw response handles without reading or retaining their bodies.

    Closing LangChain's public astream generator while it is suspended at a
    yield does not synchronously close its nested SDK stream in the locked
    version. The public HTTP response hook supplies a separate close handle.
    Models are request-owned; never share a client across simultaneous scopes.
    """
    client = model.http_async_client
    responses: list[httpx2.Response] = []

    async def capture(response: httpx2.Response) -> None:
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
) -> AsyncIterator[ChatOpenAI]:
    model = create_model(settings)
    try:
        yield model
    finally:
        with CancelScope(shield=True):
            await model.root_async_client.close()
