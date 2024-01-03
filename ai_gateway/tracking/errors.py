import traceback

import structlog
from asgi_correlation_id.context import correlation_id

__all__ = [
    "log_exception",
]

log = structlog.stdlib.get_logger("exceptions")


def log_exception(ex: Exception, extra: dict = {}) -> None:
    """Log the exception with the correlation ID.

    Args:
    ex (``Exception``):
        Raised exception during application runtime.
    extra (``dict``, `optional`):
        Additional metadata for the exception.
    """
    status_code = getattr(ex, "code", None)

    log.error(
        str(ex),
        status_code=status_code,
        backtrace=traceback.format_exc(),
        correlation_id=correlation_id.get(),
        extra=extra,
    )
