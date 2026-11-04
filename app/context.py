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


def build_chat_messages(request: ChatRequest, settings: Settings) -> PreparedChat:
    system = SystemMessage(CHAT_PROMPT.format(shop_name=settings.shop_name))
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
