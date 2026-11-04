"""Bounded, stateless chat input; client roles cannot supply system messages."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("Content must not be blank")
    return value  # Preserve whitespace, braces and Unicode exactly as supplied.


class HistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)

    @field_validator("content")
    @classmethod
    def nonblank_content(cls, value: str) -> str:
        return _nonblank(value)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    message: str = Field(min_length=1, max_length=8000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=100)

    @field_validator("message")
    @classmethod
    def nonblank_message(cls, value: str) -> str:
        return _nonblank(value)

    @model_validator(mode="after")
    def complete_turns(self) -> Self:
        if len(self.history) % 2 or any(
            item.role != ("user" if index % 2 == 0 else "assistant")
            for index, item in enumerate(self.history)
        ):
            raise ValueError("History must contain complete alternating user/assistant pairs")
        return self


class AfterSalesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    description: str = Field(min_length=1, max_length=8000)

    @field_validator("description")
    @classmethod
    def nonblank_description(cls, value: str) -> str:
        return _nonblank(value)


class AfterSalesResult(BaseModel):
    """Only facts explicitly provided by the user; nullable keys stay required."""

    model_config = ConfigDict(extra="forbid", strict=True)

    order_id: str | None
    request_type: Literal["refund", "return_refund", "exchange", "repair", "logistics", "other", "unknown"]
    expected_solution: str | None
