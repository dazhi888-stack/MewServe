"""HTTP entry point; health remains independent of model connectivity."""

import json
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager
from typing import Annotated

from anyio import CancelScope
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings
from app.context import build_chat_messages
from app.errors import AppError
from app.model import get_model
from app.schemas import ChatRequest
from app.service import stream_chat


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Missing model credentials are allowed; malformed settings fail at startup.
    get_settings()
    yield


app = FastAPI(title="MewHelp", lifespan=lifespan)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, error: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={"code": error.code, "message": error.message},
    )


@app.get("/health")
def health(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, str | bool]:
    return {"status": "ok", "model_configured": settings.model_configured}


async def get_chat_events(
    request: ChatRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    model: Annotated[ChatOpenAI, Depends(get_model)],
) -> AsyncIterator[AsyncIterator[str]]:
    prepared = build_chat_messages(request, settings)

    async def events() -> AsyncIterator[str]:
        async with aclosing(stream_chat(prepared, model)) as stream:
            async for event, data in stream:
                yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    iterator = events()
    try:
        yield iterator
    finally:
        # StreamingResponse may be cancelled during send(), while the iterator
        # is suspended at yield. Its dependency owns explicit iterator cleanup.
        with CancelScope(shield=True):
            await iterator.aclose()


@app.post("/api/chat/stream")
async def chat_stream(
    events: Annotated[AsyncIterator[str], Depends(get_chat_events)],
) -> StreamingResponse:
    return StreamingResponse(
        events, media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
