"""Configuration boundary tests isolated from the developer's environment."""

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError


def test_health_without_env():
    from app.main import app

    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok", "model_configured": False, "rag_enabled": False,
        "order_tool_enabled": False, "semantic_cache_enabled": False,
    }


def test_order_lookup_endpoint():
    from app.main import app

    with TestClient(app) as client:
        found = client.get("/api/orders/MH20261101002")
        missing = client.get("/api/orders/MH00000000000")
    assert found.status_code == 200
    assert found.json()["status"] == "运输中"
    assert missing.status_code == 404


def test_cache_clear_endpoint_disabled():
    from app.main import app

    # 测试隔离下语义缓存关闭：清理接口应明确返回未启用，而不是报错。
    with TestClient(app) as client:
        response = client.post("/api/cache/clear")
    assert response.status_code == 200
    assert response.json() == {"enabled": False, "cleared": 0}


def test_settings_load_dotenv(tmp_path):
    from app.config import Settings

    env_file = tmp_path / ".env"
    env_file.write_text(
        "DEEPSEEK_BASE_URL=https://example.invalid/v1\n"
        "DEEPSEEK_MODEL=test-model\nDEEPSEEK_API_KEY=test-only-key\nSHOP_NAME=测试店铺\n",
        encoding="utf-8",
    )
    settings = Settings(_env_file=env_file)
    assert settings.deepseek_model == "test-model"
    assert settings.deepseek_base_url == "https://example.invalid/v1"
    assert isinstance(settings.deepseek_api_key, SecretStr)
    assert settings.deepseek_api_key.get_secret_value() == "test-only-key"
    assert settings.shop_name == "测试店铺"
    assert settings.model_configured is True


@pytest.mark.parametrize("values", [
    {"context_window_tokens": 1024, "max_output_tokens": 512, "token_safety_margin": 512},
    {"context_window_tokens": 1000, "max_output_tokens": 512, "token_safety_margin": 512},
    {"max_output_tokens": 0},
    {"token_safety_margin": -1},
    {"context_window_tokens": 0},
])
def test_settings_budget_validation(values):
    from app.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, **values)


def test_secret_not_in_repr():
    from app.config import Settings

    key = "test-secret-that-must-stay-hidden"
    settings = Settings(_env_file=None, deepseek_api_key=key)
    assert key not in repr(settings)
    assert key not in str(settings)
    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None, deepseek_api_key=key, context_window_tokens=1)
    assert key not in str(error.value)
    assert key not in repr(error.value)


def test_environment_overrides_dotenv(monkeypatch, tmp_path):
    from app.config import get_settings

    (tmp_path / ".env").write_text("DEEPSEEK_MODEL=dotenv-model\n", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_MODEL", "process-model")
    assert get_settings().deepseek_model == "process-model"


# 二次开发：DEEPSEEK_BASE_URL / DEEPSEEK_MODEL 有内置默认值，
# 完整性判定实际取决于 DEEPSEEK_API_KEY 是否配置。
@pytest.mark.parametrize("missing", [None, "DEEPSEEK_API_KEY"])
def test_health_reports_only_complete_configuration(monkeypatch, missing):
    from app.main import app

    values = {
        "DEEPSEEK_BASE_URL": "https://example.invalid/v1",
        "DEEPSEEK_MODEL": "test-model",
        "DEEPSEEK_API_KEY": "test-only-key",
    }
    for name, value in values.items():
        if name != missing:
            monkeypatch.setenv(name, value)
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    # 二次开发第四批：health 新增订单工具与语义缓存开关（测试隔离下均为关闭）。
    assert response.json() == {
        "status": "ok", "model_configured": missing is None, "rag_enabled": False,
        "order_tool_enabled": False, "semantic_cache_enabled": False,
    }
    assert "test-only-key" not in response.text


@pytest.mark.parametrize("values", [
    {"deepseek_base_url": "not-a-url"},
    {"deepseek_base_url": "ftp://example.invalid"},
    {"structured_output_method": "function_calling"},
    {"request_timeout_seconds": 0},
    {"request_timeout_seconds": float("inf")},
    {"history_max_messages": -1},
    {"shop_name": "   "},
])
def test_settings_reject_invalid_values(values):
    from app.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, **values)


def test_invalid_configuration_fails_at_startup(monkeypatch):
    from app.main import app

    monkeypatch.setenv("CONTEXT_WINDOW_TOKENS", "not-an-integer")
    with pytest.raises(ValidationError):
        with TestClient(app):
            pass


def test_token_limit_field_defaults_to_compatible_parameter():
    from app.config import Settings
    assert Settings(_env_file=None).deepseek_token_limit_field == "max_tokens"


@pytest.mark.parametrize("value", ["max_tokens", "max_completion_tokens"])
def test_token_limit_field_loads_from_process_and_dotenv(monkeypatch, tmp_path, value):
    from app.config import Settings
    path = tmp_path / ".env"
    path.write_text(f"DEEPSEEK_TOKEN_LIMIT_FIELD={value}\n", encoding="utf-8")
    assert Settings(_env_file=path).deepseek_token_limit_field == value
    monkeypatch.setenv("DEEPSEEK_TOKEN_LIMIT_FIELD", value)
    assert Settings(_env_file=None).deepseek_token_limit_field == value


@pytest.mark.parametrize("value", ["", "other", "MAX_TOKENS"])
def test_token_limit_field_rejects_invalid_values(value):
    from app.config import Settings
    with pytest.raises(ValidationError):
        Settings(_env_file=None, deepseek_token_limit_field=value)
