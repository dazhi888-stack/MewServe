"""Prepare bounded text messages without storing sessions or calling a model."""

from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.config import Settings
from app.errors import AppError
from app.prompts import CHAT_PROMPT, EXTRACTION_PROMPT, EXTRACTION_SCHEMA_DESCRIPTION
from app.schemas import ChatRequest


@dataclass(frozen=True)
class PreparedChat:
    messages: list[BaseMessage]
    estimated_input_tokens: int
    dropped_messages: int


def estimate_tokens(messages: Sequence[BaseMessage]) -> int:
    """UTF-8 bytes + 16/message, a conservative text-only budget heuristic.

    This is not an exact tokenizer for arbitrary upstream providers. Server
    templates and all text messages count; output and safety reserves are
    applied separately before any future model invocation.
    """
    return sum(len(message.content.encode("utf-8")) + 16 for message in messages)


def _input_budget(settings: Settings) -> int:
    return (
        settings.context_window_tokens
        - settings.max_output_tokens
        - settings.token_safety_margin
    )


def _reject_oversized() -> None:
    raise AppError(
        "context_too_large", "输入内容超过上下文预算，请缩短本次输入后重试。", 413
    )


# 二次开发新增：中间档置信度回答时追加的对冲指令（兜底式路由）。
HEDGE_INSTRUCTION = (
    "\n【回答提示】系统对本次问题的意图把握有限：请先给出最可能的解答，"
    "但需在开头用一句话确认理解（如“您是想咨询……吗”），并避免给出绝对的承诺；"
    "若理解有误，请引导用户补充说明。"
)


def build_chat_messages(
    request: ChatRequest, settings: Settings, faq_context: str = "",
    order_context: str = "", hedge: bool = False,
) -> PreparedChat:
    # 二次开发：faq_context 来自 RAG 检索结果，注入 system prompt；空串时模板原样跳过。
    # 二次开发（第四批）：order_context 来自 Mock 订单工具，同样注入 system prompt。
    # hedge=True 时追加对冲指令（意图置信度处于中间档，回答需带确认与缓冲）。
    system_text = CHAT_PROMPT.format(shop_name=settings.shop_name, faq_context=faq_context)
    if order_context:
        system_text += "\n\n" + order_context
    if hedge:
        system_text += HEDGE_INSTRUCTION
    system = SystemMessage(system_text)
    latest = HumanMessage(request.message)
    protected = [system, latest]
    budget = _input_budget(settings)
    if estimate_tokens(protected) > budget:
        _reject_oversized()

    # An odd configured cap rounds down to complete pairs; zero disables history.
    limit = settings.history_max_messages // 2 * 2
    retained = request.history[-limit:] if limit else []
    messages: list[BaseMessage] = [system]
    messages.extend(
        HumanMessage(item.content) if item.role == "user" else AIMessage(item.content)
        for item in retained
    )
    messages.append(latest)
    dropped = len(request.history) - len(retained)
    while estimate_tokens(messages) > budget:
        del messages[1:3]  # Discard one oldest, complete user/assistant turn.
        dropped += 2
    return PreparedChat(messages, estimate_tokens(messages), dropped)


def build_extraction_messages(description: str, settings: Settings) -> list[BaseMessage]:
    # The extraction endpoint will own its request model; share the same text
    # boundary now so direct callers cannot bypass length/blank validation.
    validated = ChatRequest(message=description)
    messages: list[BaseMessage] = [
        SystemMessage(EXTRACTION_PROMPT.format(schema=EXTRACTION_SCHEMA_DESCRIPTION)),
        HumanMessage(validated.message),
    ]
    if estimate_tokens(messages) > _input_budget(settings):
        _reject_oversized()
    return messages
