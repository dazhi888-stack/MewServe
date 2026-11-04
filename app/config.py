"""Validated process/.env configuration, without creating any model client."""

from functools import lru_cache
from typing import Literal, Self

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    openai_base_url: str = ""
    openai_model: str = ""
    openai_api_key: SecretStr = Field(default=SecretStr(""), repr=False)
    structured_output_method: Literal["json_mode", "json_schema"] = "json_mode"
    context_window_tokens: int = Field(default=8192, gt=0)
    max_output_tokens: int = Field(default=512, gt=0)
    token_safety_margin: int = Field(default=512, ge=0)
    history_max_messages: int = Field(default=20, ge=0)
    request_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    shop_name: str = "MewHelp"

    @field_validator("openai_base_url", "openai_model", "shop_name")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("openai_base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        if value:
            AnyHttpUrl(value)
        return value

    @field_validator("shop_name")
    @classmethod
    def validate_shop_name(cls, value: str) -> str:
        if not value:
            raise ValueError("SHOP_NAME must not be blank")
        return value

    @model_validator(mode="after")
    def validate_budget(self) -> Self:
        if self.max_output_tokens + self.token_safety_margin >= self.context_window_tokens:
            raise ValueError(
                "MAX_OUTPUT_TOKENS + TOKEN_SAFETY_MARGIN must be less than "
                "CONTEXT_WINDOW_TOKENS"
            )
        return self

    @property
    def model_configured(self) -> bool:
        """Configuration completeness only; this never probes the upstream."""
        return bool(
            self.openai_base_url
            and self.openai_model
            and self.openai_api_key.get_secret_value().strip()
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
