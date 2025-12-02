"""Enhanced logging utilities for document-parser."""

import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any, Optional

# Context variable to store request ID across async calls
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


class ContextLogger:
    """Logger wrapper that adds structured data to log records."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def _format_extra_data(self, extra_data: Optional[dict[str, Any]]) -> str:
        """Format extra data as key=value pairs."""
        if not extra_data:
            return ""
        parts = [f"{k}={v}" for k, v in extra_data.items()]
        return " [" + ", ".join(parts) + "]"

    def _log(self, level: int, msg: str, extra_data: Optional[dict[str, Any]] = None, **kwargs):
        """Log with extra data formatted as plain text."""
        # Add request ID if available
        request_id = request_id_var.get()
        if request_id:
            if extra_data is None:
                extra_data = {}
            extra_data["request_id"] = request_id

        # Format message with extra data
        formatted_msg = msg + self._format_extra_data(extra_data)
        self.logger.log(level, formatted_msg, **kwargs)

    def debug(self, msg: str, extra_data: Optional[dict[str, Any]] = None, **kwargs):
        """Log debug message with extra data."""
        self._log(logging.DEBUG, msg, extra_data, **kwargs)

    def info(self, msg: str, extra_data: Optional[dict[str, Any]] = None, **kwargs):
        """Log info message with extra data."""
        self._log(logging.INFO, msg, extra_data, **kwargs)

    def warning(self, msg: str, extra_data: Optional[dict[str, Any]] = None, **kwargs):
        """Log warning message with extra data."""
        self._log(logging.WARNING, msg, extra_data, **kwargs)

    def error(self, msg: str, extra_data: Optional[dict[str, Any]] = None, **kwargs):
        """Log error message with extra data."""
        self._log(logging.ERROR, msg, extra_data, **kwargs)

    def critical(self, msg: str, extra_data: Optional[dict[str, Any]] = None, **kwargs):
        """Log critical message with extra data."""
        self._log(logging.CRITICAL, msg, extra_data, **kwargs)


def setup_logging(log_level: str = "INFO"):
    """Configure application logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Use plain text formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


def get_logger(name: str) -> ContextLogger:
    """Get a context-aware logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        ContextLogger instance
    """
    return ContextLogger(logging.getLogger(name))


def set_request_id(request_id: Optional[str] = None) -> str:
    """Set request ID for the current context.

    Args:
        request_id: Optional request ID. If not provided, a new UUID will be generated.

    Returns:
        The request ID that was set
    """
    if request_id is None:
        request_id = str(uuid.uuid4())
    request_id_var.set(request_id)
    return request_id


def get_request_id() -> Optional[str]:
    """Get the current request ID from context."""
    return request_id_var.get()


class Timer:
    """Context manager for timing operations."""

    def __init__(self, name: str):
        self.name = name
        self.start_time: Optional[float] = None
        self.elapsed_ms: Optional[int] = None

    def __enter__(self) -> "Timer":
        """Start timer."""
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        """Stop timer and calculate elapsed time."""
        if self.start_time:
            self.elapsed_ms = int((time.time() - self.start_time) * 1000)

    def get_elapsed_ms(self) -> int:
        """Get elapsed time in milliseconds."""
        if self.elapsed_ms is not None:
            return self.elapsed_ms
        if self.start_time is not None:
            return int((time.time() - self.start_time) * 1000)
        return 0
