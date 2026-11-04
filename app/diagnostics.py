"""Keep known payload-bearing serialization diagnostics out of service logs."""

import warnings

_SAFE_SERIALIZATION_WARNING = "Model response serialization warning; details withheld."
_original_formatwarning = warnings.formatwarning
_installed = False


def _safe_formatwarning(message, category, filename, lineno, line=None) -> str:
    # Pydantic includes input_value fragments in these warnings before the
    # model returns its normalized blocks. Match that exact diagnostic source,
    # rather than muting UserWarning or arbitrary model/vendor warnings.
    if (
        issubclass(category, UserWarning)
        and filename.replace("\\", "/").lower().endswith("/pydantic/main.py")
        and str(message).startswith("Pydantic serializer warnings:")
        and "PydanticSerializationUnexpectedValue" in str(message)
    ):
        return _original_formatwarning(
            _SAFE_SERIALIZATION_WARNING, category, filename, lineno, line="",
        )
    return _original_formatwarning(message, category, filename, lineno, line)


def install_safe_diagnostics() -> None:
    """Install once at model-boundary import, never inside a request.

    Both standard stderr warnings and logging.captureWarnings use this public
    formatter. No warning filters or request-specific global state are changed.
    """
    global _installed
    if not _installed:
        warnings.formatwarning = _safe_formatwarning
        _installed = True
