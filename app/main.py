"""HTTP entry point; health remains independent of model connectivity.

二次开发改动：
- 对话管线前置意图识别分支（answer / refuse 兜底式两档）。
- answer 分支接入 RAG 混合检索（BM25+向量 RRF 融合 + bge-reranker 重排），
  多轮场景先做查询改写再检索，FAQ 上下文注入 system prompt。
- 中间档置信度回答带对冲（hedge），低置信度/无关问题拒答并引导人工渠道。
- SSE 保留 intent / rag 事件；confirm 事件为历史兼容，当前路由不再产生。
"""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager
from typing import Annotated
from pathlib import Path

from anyio import CancelScope, create_task_group, get_cancelled_exc_class
from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.context import build_chat_messages
from app.errors import AppError
from app.intent import (
    CATEGORY_LABELS, IntentResult, classify_intent, needs_hedge, route_action,
)
from app.rag import get_retriever, rewrite_query
from langchain_deepseek import ChatDeepSeek
from app.model import get_model
from app.rag import FaqRetriever
from app.schemas import AfterSalesRequest, AfterSalesResult, ChatRequest
from app.service import extract_after_sales, stream_chat

REFUSAL_MESSAGE = (
    "抱歉，我是喵帮电商客服，只能为您解答商品规格、运费、退换货、发货时效等"
    "购物相关问题。您咨询的内容超出了我的服务范围，请换个购物相关的问题试试；"
    "如需进一步帮助，也可以通过店铺页面留言联系人工客服。"
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Missing model credentials are allowed; malformed settings fail at startup.
    settings = get_settings()
    # 二次开发新增：启动时初始化 Chroma FAQ 知识库（为空自动导入）。
    if settings.rag_enabled:
        get_retriever(settings).ensure_collection()
    yield


app = FastAPI(title="MewHelp", lifespan=lifespan)
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def chat_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.exception_handler(AppError)
async def app_error_handler(request: Request, error: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={"code": error.code, "message": error.message},
    )


@app.get("/health")
def health(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model_configured": settings.model_configured,
        "rag_enabled": settings.rag_enabled,
    }


def get_rag(
    settings: Annotated[Settings, Depends(get_settings)],
) -> FaqRetriever:
    return get_retriever(settings)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def get_chat_events(
    request: ChatRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    model: Annotated[ChatDeepSeek, Depends(get_model)],
    rag: Annotated[FaqRetriever, Depends(get_rag)],
) -> AsyncIterator[AsyncIterator[str]]:
    # 二次开发新增：意图识别分支（兜底式两档）。确认过的消息（confirm_intent=true）直接回答。
    intent: IntentResult | None = None
    action = "answer"
    if not request.confirm_intent and settings.intent_enabled:
        intent = await classify_intent(request.message, settings, model)
        action = route_action(intent, settings)

    # RAG 检索与消息构建放在响应开始前完成：预算超限(413)等错误保持基线的
    # "先校验后开流"契约，不会在 SSE 已启动后才报错。
    faq_context = ""
    prepared = None
    hedged = intent is not None and action == "answer" and needs_hedge(intent, settings)
    if action == "answer":
        if settings.rag_enabled and rag.available:
            # 多轮场景先把追问改写成独立检索问题，再走混合检索（在线程池中执行，
            # 避免 BM25/重排的阻塞调用拖住事件循环）。
            search_query = request.message
            if settings.query_rewrite_enabled and request.history:
                search_query = await rewrite_query(request, settings, model)
            faq_context = "\n\n".join(
                await asyncio.to_thread(rag.retrieve, search_query)
            )
        prepared = build_chat_messages(request, settings, faq_context, hedge=hedged)

    async def events() -> AsyncIterator[str]:
        if action == "answer":
            if intent is not None:
                yield _sse("intent", {
                    "category": intent.category,
                    "category_label": CATEGORY_LABELS.get(intent.category, intent.category),
                    "confidence": intent.confidence,
                    "action": action,
                    "hedged": hedged,
                })
            # 仅在真正命中 FAQ 时通知客户端；meta 由 stream_chat 统一发出，避免重复。
            if faq_context:
                yield _sse("rag", {"faq_used": True})
            async with aclosing(stream_chat(prepared, model)) as stream:
                async for event, data in stream:
                    yield _sse(event, data)
        else:
            # 置信度过低或无关问题：直接拒绝，不调用模型。
            assert intent is not None
            yield _sse("intent", {
                "category": intent.category,
                "category_label": CATEGORY_LABELS.get(intent.category, intent.category),
                "confidence": intent.confidence,
                "action": action,
            })
            yield _sse("token", {"text": REFUSAL_MESSAGE})
            yield _sse("done", {"finish_reason": "refused"})

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


@app.post("/api/after-sales/extract", response_model=AfterSalesResult)
async def after_sales_extract(
    connection: Request,
    request: AfterSalesRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    model: Annotated[ChatDeepSeek, Depends(get_model)],
) -> AfterSalesResult:
    result: AfterSalesResult | None = None
    error: Exception | None = None
    async with create_task_group() as group:
        async def disconnect() -> None:
            while True:
                if (await connection.receive())["type"] == "http.disconnect":
                    group.cancel_scope.cancel()
                    return

        group.start_soon(disconnect)
        try:
            result = await extract_after_sales(request.description, settings, model)
        except Exception as exc:
            # Raise outside the task group to preserve the AppError boundary.
            error = exc
        else:
            error = None
        finally:
            group.cancel_scope.cancel()
    if result is None:
        if error is not None:
            raise error
        raise get_cancelled_exc_class()()
    return result
