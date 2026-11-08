"""Validated process/.env configuration, without creating any model client.

二次开发改动：模型接入从 OpenAI 兼容配置改为 DeepSeek 官方 API，
并新增 RAG（Chroma 向量库）、意图识别阈值、订单查询工具与语义缓存配置。
"""

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

    # ---- DeepSeek 模型配置（原 openai_* 字段已移除） ----
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"
    deepseek_api_key: SecretStr = Field(default=SecretStr(""), repr=False)
    deepseek_token_limit_field: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    # DeepSeek API 仅支持 json_object 结构化输出；本 fork 收敛为 json_mode，
    # 不提供 OpenAI 风格的 json_schema 线协议选项。
    structured_output_method: Literal["json_mode"] = "json_mode"
    context_window_tokens: int = Field(default=8192, gt=0)
    max_output_tokens: int = Field(default=512, gt=0)
    token_safety_margin: int = Field(default=512, ge=0)
    history_max_messages: int = Field(default=20, ge=0)
    request_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    shop_name: str = "MewHelp"

    # ---- RAG / Chroma 配置（新增） ----
    rag_enabled: bool = True
    chroma_dir: str = "./chroma_db"
    faq_file: str = "./knowledge/faqs.md"
    rag_top_k: int = Field(default=4, gt=0)
    # 混合检索：向量与 BM25 各自召回的候选数（RRF 融合前的每路 top-k）。
    rag_candidate_k: int = Field(default=12, gt=0)
    # 向量库需要 embedding 模型：默认走硅基流动 OpenAI 兼容 embeddings 接口；
    # 未配置 key 时回退到 Chroma 自带的本地默认 embedding。
    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_model: str = "BAAI/bge-m3"
    embedding_api_key: SecretStr = Field(default=SecretStr(""), repr=False)
    # 重排：RRF 融合后的候选送 bge-reranker 精排（复用 embedding 的硅基流动配置）。
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    # 多轮对话时是否先用模型把追问改写成独立检索问题。
    query_rewrite_enabled: bool = True

    # ---- 意图识别阈值配置（新增） ----
    intent_enabled: bool = True
    intent_high_threshold: float = Field(default=0.85, ge=0, le=1)
    intent_low_threshold: float = Field(default=0.60, ge=0, le=1)

    # ---- 订单查询工具配置（第四批新增） ----
    order_tool_enabled: bool = True

    # ---- 语义缓存配置（第四批新增）：与 FAQ 共用 chroma_dir ----
    semantic_cache_enabled: bool = True
    # 命中阈值：问题向量与缓存问题的 cosine 相似度 >= 该值时复用历史回答。
    cache_similarity_threshold: float = Field(default=0.92, ge=0.5, le=1)
    # 缓存最大条数，超出按写入时间 FIFO 淘汰。
    cache_max_entries: int = Field(default=1000, gt=0)

    @field_validator("deepseek_base_url", "deepseek_model", "shop_name")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("deepseek_base_url")
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

    @field_validator("chroma_dir", "faq_file")
    @classmethod
    def strip_paths(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_budget(self) -> Self:
        if self.max_output_tokens + self.token_safety_margin >= self.context_window_tokens:
            raise ValueError(
                "MAX_OUTPUT_TOKENS + TOKEN_SAFETY_MARGIN must be less than "
                "CONTEXT_WINDOW_TOKENS"
            )
        return self

    @model_validator(mode="after")
    def validate_intent_thresholds(self) -> Self:
        if self.intent_low_threshold > self.intent_high_threshold:
            raise ValueError(
                "INTENT_LOW_THRESHOLD must not exceed INTENT_HIGH_THRESHOLD"
            )
        return self

    @property
    def model_configured(self) -> bool:
        """Configuration completeness only; this never probes the upstream."""
        return bool(
            self.deepseek_base_url
            and self.deepseek_model
            and self.deepseek_api_key.get_secret_value().strip()
        )

    @property
    def embedding_configured(self) -> bool:
        return bool(self.embedding_api_key.get_secret_value().strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
