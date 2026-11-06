"""Shared test isolation; never load the developer's real .env."""

import sys

import pytest

ENV_NAMES = (
    "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL", "DEEPSEEK_API_KEY", "DEEPSEEK_TOKEN_LIMIT_FIELD",
    "STRUCTURED_OUTPUT_METHOD", "CONTEXT_WINDOW_TOKENS", "MAX_OUTPUT_TOKENS",
    "TOKEN_SAFETY_MARGIN", "HISTORY_MAX_MESSAGES", "REQUEST_TIMEOUT_SECONDS",
    "SHOP_NAME",
    # 二次开发新增：RAG 与意图识别相关环境变量同样需要测试隔离。
    "RAG_ENABLED", "CHROMA_DIR", "FAQ_FILE", "RAG_TOP_K",
    "EMBEDDING_BASE_URL", "EMBEDDING_MODEL", "EMBEDDING_API_KEY",
    "INTENT_ENABLED", "INTENT_HIGH_THRESHOLD", "INTENT_LOW_THRESHOLD",
)


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    # 测试不访问真实向量库与意图识别上游；需要时在具体测试里显式打开。
    monkeypatch.setenv("RAG_ENABLED", "false")
    monkeypatch.setenv("INTENT_ENABLED", "false")
    monkeypatch.chdir(tmp_path)
    config = sys.modules.get("app.config")
    if config is not None:
        config.get_settings.cache_clear()
    yield
    config = sys.modules.get("app.config")
    if config is not None:
        config.get_settings.cache_clear()
