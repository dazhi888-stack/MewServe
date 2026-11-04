"""Application errors with explicit, safe public messages."""

import httpx2
from openai import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError


class AppError(Exception):
    def __init__(self, code: str, message: str, status_code: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def public_upstream_error(error: Exception) -> AppError:
    # Some compatible transports raise raw HTTP errors while consuming a body.
    if isinstance(error, (APITimeoutError, httpx2.TimeoutException)):
        return AppError("upstream_timeout", "模型响应超时，请稍后重试。", 504)
    if isinstance(error, AuthenticationError):
        return AppError("upstream_authentication", "模型服务鉴权失败，请联系服务维护者。", 502)
    if isinstance(error, RateLimitError):
        return AppError("upstream_rate_limited", "模型服务暂时繁忙，请稍后重试。", 503)
    if isinstance(error, (APIConnectionError, httpx2.RequestError)):
        return AppError("upstream_connection", "暂时无法连接模型服务，请稍后重试。", 502)
    return AppError("upstream_error", "模型服务发生错误，请稍后重试。", 502)
