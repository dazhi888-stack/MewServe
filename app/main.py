"""HTTP entry point; health remains independent of model connectivity.

二次开发改动：
- 对话管线前置意图识别分支（answer / refuse 兜底式两档）。
- answer 分支接入 RAG 混合检索（BM25+向量 RRF 融合 + bge-reranker 重排），
  多轮场景先做查询改写再检索，FAQ 上下文注入 system prompt。
- 中间档置信度回答带对冲（hedge），低置信度/无关问题拒答并引导人工渠道。
- SSE 保留 intent / rag 事件；confirm 事件为历史兼容，当前路由不再产生。

二次开发第四批：
- 新增 Mock 订单查询工具：消息命中订单号或意图为订单/物流类时执行查询，
  订单上下文注入 system prompt，SSE 新增 tool 事件。
- 新增语义缓存：首轮提问按向量相似度命中历史回答时直接复用（跳过 RAG 与
  模型调用），SSE 新增 cache 事件；回答完成后写回缓存。
"""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager
from typing import Annotated
from pathlib import Path

from anyio import CancelScope, create_task_group, get_cancelled_exc_class
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.cache import get_semantic_cache
from app.context import build_chat_messages
from app.errors import AppError
from app.intent import (
    CATEGORY_LABELS, IntentResult, classify_intent, needs_hedge, route_action,
)
from app.rag import get_retriever, rewrite_query
from app.tools import query_order, resolve_order_tool
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


app = FastAPI(title="MewServe", lifespan=lifespan)
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
        "order_tool_enabled": settings.order_tool_enabled,
        "semantic_cache_enabled": settings.semantic_cache_enabled,
    }


@app.get("/api/orders/{order_id}")
def order_lookup(order_id: str) -> dict:
    """Mock 订单直接查询端点（演示/联调用）；对话管线走的是注入式工具。"""
    order = query_order(order_id)
    if order is None:
        raise HTTPException(status_code=404, detail=f"未找到订单 {order_id}，请核对订单号。")
    return order


@app.post("/api/cache/clear")
def cache_clear(settings: Annotated[Settings, Depends(get_settings)]) -> dict:
    """清空语义缓存（运维接口）：缓存异常/污染时无需删 chroma_db 目录。"""
    if not settings.semantic_cache_enabled:
        return {"enabled": False, "cleared": 0}
    cleared = get_semantic_cache(settings).clear()
    if cleared < 0:
        raise HTTPException(status_code=503, detail="语义缓存不可用，无法清理。")
    return {"enabled": True, "cleared": cleared}


def get_rag(
    settings: Annotated[Settings, Depends(get_settings)],
) -> FaqRetriever:
    return get_retriever(settings)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


_CACHE_CHUNK_SIZE = 40


def _chunks(text: str, size: int = _CACHE_CHUNK_SIZE) -> list[str]:
    """把缓存命中的整段回答切成小块，复用前端逐字/逐段渲染节奏。"""
    return [text[i:i + size] for i in range(0, len(text), size)]


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

    # 二次开发第四批：语义缓存（仅首轮提问参与）与 Mock 订单工具。
    # 检索/工具/缓存都放在响应开始前完成：预算超限(413)等错误保持基线的
    # "先校验后开流"契约，不会在 SSE 已启动后才报错。
    cache = None
    cached: dict | None = None
    tool_result = None
    faq_context = ""
    order_context = ""
    prepared = None
    hedged = intent is not None and action == "answer" and needs_hedge(intent, settings)
    if action == "answer":
        # 语义缓存命中则跳过订单工具、RAG 与模型调用（多轮回答依赖上下文，不参与）。
        if settings.semantic_cache_enabled and not request.history:
            cache = get_semantic_cache(settings)
            cached = await asyncio.to_thread(cache.lookup, request.message)
        if cached is None and settings.order_tool_enabled:
            # 订单工具：消息含订单号直接查询（零模型开销）；订单/物流意图时
            # 再用一次 json_mode 抽取兜底。任何失败都不阻塞主回答链路。
            tool_result = await resolve_order_tool(request.message, intent, settings, model)
            order_context = tool_result.context
        if cached is None and settings.rag_enabled and rag.available:
            # 多轮场景先把追问改写成独立检索问题，再走混合检索（在线程池中执行，
            # 避免 BM25/重排的阻塞调用拖住事件循环）。
            search_query = request.message
            if settings.query_rewrite_enabled and request.history:
                search_query = await rewrite_query(request, settings, model)
            faq_context = "\n\n".join(
                await asyncio.to_thread(rag.retrieve, search_query)
            )
        if cached is None:
            prepared = build_chat_messages(
                request, settings, faq_context, order_context=order_context, hedge=hedged,
            )

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
            if cached is not None:
                # 语义缓存命中：直接回放缓存回答，不调用模型。
                yield _sse("cache", {"hit": True, "score": cached["score"]})
                for chunk in _chunks(cached["answer"]):
                    yield _sse("token", {"text": chunk})
                yield _sse("done", {"finish_reason": "stop"})
                return
            # 订单工具命中/未命中都通知客户端；仅在注入了订单上下文时发送。
            # status 区分三种结果：hit（查到）/ not_found（给了订单号但查无此单）/
            # need_order_id（没给订单号，已注入引导话术），前端据此展示不同标注。
            if tool_result is not None and tool_result.context:
                if tool_result.found:
                    status = "hit"
                elif tool_result.order_id:
                    status = "not_found"
                else:
                    status = "need_order_id"
                yield _sse("tool", {
                    "tool": "order_query",
                    "order_id": tool_result.order_id,
                    "found": tool_result.found,
                    "status": status,
                })
            # 仅在真正命中 FAQ 时通知客户端；meta 由 stream_chat 统一发出，避免重复。
            if faq_context:
                yield _sse("rag", {"faq_used": True})
            answer_text = ""
            async with aclosing(stream_chat(prepared, model)) as stream:
                async for event, data in stream:
                    if event == "token":
                        answer_text += data.get("text", "")
                    elif (
                        event == "done" and cache is not None
                        and data.get("finish_reason") == "stop"
                        and answer_text.strip()
                    ):
                        # 回答完成后写回语义缓存；失败只记日志，不影响本次流。
                        await asyncio.to_thread(cache.store, request.message, answer_text)
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
