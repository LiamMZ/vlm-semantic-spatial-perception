"""
Logging utilities built on the standard library logging package.

Provides a callback handler so UI layers can receive log messages in real time
without redirecting stdout.
"""

import logging
import sys
from pathlib import Path
from typing import Callable, Optional


class CallbackLogHandler(logging.Handler):
    """Logging handler that forwards formatted messages to a callback."""

    def __init__(
        self,
        callback: Callable[..., None],
        level: int = logging.NOTSET,
        *,
        pass_record: bool = False,
    ):
        super().__init__(level)
        self.callback = callback
        self.pass_record = pass_record

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if self.pass_record:
                self.callback(msg, record)
            else:
                self.callback(msg)
        except Exception:
            self.handleError(record)


def configure_logging(
    level: int = logging.INFO,
    callback: Optional[Callable[[str], None]] = None,
    log_file: Optional[Path] = None,
    include_console: bool = True,
) -> None:
    """
    Configure root logging with optional callback and file handlers.

    Args:
        level: Minimum log level to emit.
        callback: Optional callable to receive formatted log lines immediately.
        log_file: Optional path to append log output.
        include_console: Whether to emit to stdout as well.
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # Include filename and line number to make debugging easier.
    formatter = logging.Formatter("[%(levelname)s:%(name)s:%(filename)s:%(lineno)d] %(message)s")

    if include_console and not any(
        isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout
        for h in logger.handlers
    ):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if callback and not any(isinstance(h, CallbackLogHandler) for h in logger.handlers):
        callback_handler = CallbackLogHandler(callback)
        callback_handler.setLevel(level)
        callback_handler.setFormatter(formatter)
        logger.addHandler(callback_handler)

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file for h in logger.handlers):
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s:%(name)s:%(filename)s:%(lineno)d] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(file_handler)


def get_structured_logger(name: str) -> logging.Logger:
    """
    Get a logger configured to propagate to the root handlers.

    Args:
        name: Logger name.

    Returns:
        logging.Logger: Logger instance.
    """
    logger = logging.getLogger(name)
    logger.propagate = True
    return logger
