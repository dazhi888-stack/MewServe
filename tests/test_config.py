"""Configuration boundary tests isolated from the developer's environment."""

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError


def test_health_without_env():
    from app.main import app

    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_configured": False}


def test_settings_load_dotenv(tmp_path):
    from app.config import Settings

    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_BASE_URL=https://example.invalid/v1\n"
        "OPENAI_MODEL=test-model\nOPENAI_API_KEY=test-only-key\nSHOP_NAME=测试店铺\n",
        encoding="utf-8",
    )
    settings = Settings(_env_file=env_file)
    assert settings.openai_model == "test-model"
    assert settings.openai_base_url == "https://example.invalid/v1"
    assert isinstance(settings.openai_api_key, SecretStr)
    assert settings.openai_api_key.get_secret_value() == "test-only-key"
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
    settings = Settings(_env_file=None, openai_api_key=key)
    assert key not in repr(settings)
    assert key not in str(settings)
    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None, openai_api_key=key, context_window_tokens=1)
    assert key not in str(error.value)
    assert key not in repr(error.value)


def test_environment_overrides_dotenv(monkeypatch, tmp_path):
    from app.config import get_settings

    (tmp_path / ".env").write_text("OPENAI_MODEL=dotenv-model\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_MODEL", "process-model")
    assert get_settings().openai_model == "process-model"


@pytest.mark.parametrize("missing", [None, "OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_API_KEY"])
def test_health_reports_only_complete_configuration(monkeypatch, missing):
    from app.main import app

    values = {
        "OPENAI_BASE_URL": "https://example.invalid/v1",
        "OPENAI_MODEL": "test-model",
        "OPENAI_API_KEY": "test-only-key",
    }
    for name, value in values.items():
        if name != missing:
            monkeypatch.setenv(name, value)
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_configured": missing is None}
    assert "test-only-key" not in response.text


@pytest.mark.parametrize("values", [
    {"openai_base_url": "not-a-url"},
    {"openai_base_url": "ftp://example.invalid"},
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
