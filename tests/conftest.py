"""Shared test isolation; never load the developer's real .env."""

import sys

import pytest

ENV_NAMES = (
    "OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_API_KEY", "OPENAI_TOKEN_LIMIT_FIELD",
    "STRUCTURED_OUTPUT_METHOD", "CONTEXT_WINDOW_TOKENS", "MAX_OUTPUT_TOKENS",
    "TOKEN_SAFETY_MARGIN", "HISTORY_MAX_MESSAGES", "REQUEST_TIMEOUT_SECONDS",
    "SHOP_NAME",
)


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    config = sys.modules.get("app.config")
    if config is not None:
        config.get_settings.cache_clear()
    yield
    config = sys.modules.get("app.config")
    if config is not None:
        config.get_settings.cache_clear()
