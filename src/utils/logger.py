"""
Minimal logger implementation compatible with standard Python logging interface.

This provides a simple logger that prints to stdout while maintaining 
compatibility with the standard logging module interface.
"""

import sys
from typing import Any


class MinimalLogger:
    """
    Minimal logger that prints to stdout with level prefixes.
    
    Compatible with standard logging.Logger interface for basic operations.
    """
    
    def __init__(self, name: str = ""):
        self.name = name
    
    def _format_message(self, level: str, msg: str, *args: Any) -> str:
        """Format a log message with level prefix."""
        if args:
            msg = msg % args
        prefix = f"[{level}]" if not self.name else f"[{level}:{self.name}]"
        return f"{prefix} {msg}"
    
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log debug message."""
        print(self._format_message("DEBUG", msg, *args), file=sys.stdout)
    
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log info message."""
        print(self._format_message("INFO", msg, *args), file=sys.stdout)
    
    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log warning message."""
        print(self._format_message("WARNING", msg, *args), file=sys.stdout)
    
    def warn(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Alias for warning."""
        self.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log error message."""
        print(self._format_message("ERROR", msg, *args), file=sys.stderr)
    
    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log critical message."""
        print(self._format_message("CRITICAL", msg, *args), file=sys.stderr)
    
    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log exception with traceback."""
        import traceback
        print(self._format_message("ERROR", msg, *args), file=sys.stderr)
        traceback.print_exc(file=sys.stderr)


def get_logger(name: str = "") -> MinimalLogger:
    """
    Get a minimal logger instance.
    
    Args:
        name: Logger name (optional)
    
    Returns:
        MinimalLogger: A minimal logger instance
    """
    return MinimalLogger(name)
