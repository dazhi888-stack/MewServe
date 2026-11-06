"""Application errors with explicit, safe public messages."""

import httpx
from openai import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError


class AppError(Exception):
    def __init__(self, code: str, message: str, status_code: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def public_upstream_error(error: Exception) -> AppError:
    # Some compatible transports raise raw HTTP errors while consuming a body.
    if isinstance(error, (APITimeoutError, httpx.TimeoutException)):
        return AppError("upstream_timeout", "模型响应超时，请稍后重试。", 504)
    if isinstance(error, AuthenticationError):
        return AppError("upstream_authentication", "模型服务鉴权失败，请联系服务维护者。", 502)
    if isinstance(error, RateLimitError):
        return AppError("upstream_rate_limited", "模型服务暂时繁忙，请稍后重试。", 503)
    if isinstance(error, (APIConnectionError, httpx.RequestError)):
        return AppError("upstream_connection", "暂时无法连接模型服务，请稍后重试。", 502)
    return AppError("upstream_error", "模型服务发生错误，请稍后重试。", 502)


def structured_error(code: str) -> AppError:
    messages = {
        "structured_mode_unsupported": "模型服务不支持所选结构化输出模式，请联系服务维护者。",
        "structured_invalid_json": "模型未返回有效 JSON，请重试。",
        "structured_invalid_schema": "模型返回的数据不符合售后字段规范，请重试。",
        "structured_unsupported_response": "模型未返回可用的售后结构化结果，请重试。",
    }
    return AppError(code, messages[code], 502)
