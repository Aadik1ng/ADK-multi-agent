"""
Logging configuration for the multi-agent system.
Provides structured logging with JSON and console formatters.
"""
import logging
import os
import sys
import time
from functools import wraps
from typing import Any, Callable, Optional
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, "agent_name"):
            log_data["agent_name"] = record.agent_name
        if hasattr(record, "session_id"):
            log_data["session_id"] = record.session_id
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, "cache_hit"):
            log_data["cache_hit"] = record.cache_hit
        if hasattr(record, "entity_count"):
            log_data["entity_count"] = record.entity_count
        if hasattr(record, "error"):
            log_data["error"] = record.error
            
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter with colors."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            colored_levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        else:
            colored_levelname = levelname
            
        # Build message
        parts = [
            f"[{datetime.now().strftime('%H:%M:%S')}]",
            colored_levelname,
            f"[{record.name}]",
        ]
        
        # Add agent name if present
        if hasattr(record, "agent_name"):
            parts.append(f"[{record.agent_name}]")
            
        # Add duration if present
        if hasattr(record, "duration_ms"):
            parts.append(f"({record.duration_ms:.2f}ms)")
            
        # Add cache hit indicator
        if hasattr(record, "cache_hit"):
            cache_indicator = "💾" if record.cache_hit else "🔍"
            parts.append(cache_indicator)
            
        parts.append(record.getMessage())
        
        msg = " ".join(parts)
        
        # Add exception if present
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
            
        return msg


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "console",
    log_file: Optional[str] = None
) -> None:
    """
    Setup logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ("json" or "console")
        log_file: Optional file path for log output
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Choose formatter
    if log_format.lower() == "json":
        formatter = JSONFormatter()
    else:
        formatter = ConsoleFormatter()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific component.
    
    Args:
        name: Logger name (typically module or agent name)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_performance(logger: logging.Logger, operation: str):
    """
    Decorator to log performance metrics for a function.
    
    Args:
        logger: Logger instance to use
        operation: Name of the operation being timed
        
    Example:
        @log_performance(logger, "entity_extraction")
        async def extract_entities(text):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.info(
                    f"{operation} completed",
                    extra={"duration_ms": duration_ms, "operation": operation}
                )
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"{operation} failed",
                    extra={"duration_ms": duration_ms, "operation": operation, "error": str(e)},
                    exc_info=True
                )
                raise
                
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.info(
                    f"{operation} completed",
                    extra={"duration_ms": duration_ms, "operation": operation}
                )
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"{operation} failed",
                    extra={"duration_ms": duration_ms, "operation": operation, "error": str(e)},
                    exc_info=True
                )
                raise
        
        # Return appropriate wrapper based on function type
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


class AgentLogger:
    """
    Wrapper class for agent-specific logging with automatic context.
    """
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = get_logger(f"agent.{agent_name}")
        
    def _add_context(self, extra: Optional[dict] = None) -> dict:
        """Add agent context to extra fields."""
        context = {"agent_name": self.agent_name}
        if extra:
            context.update(extra)
        return context
        
    def debug(self, msg: str, **kwargs):
        """Log debug message with agent context."""
        extra = self._add_context(kwargs.pop("extra", None))
        self.logger.debug(msg, extra=extra, **kwargs)
        
    def info(self, msg: str, **kwargs):
        """Log info message with agent context."""
        extra = self._add_context(kwargs.pop("extra", None))
        self.logger.info(msg, extra=extra, **kwargs)
        
    def warning(self, msg: str, **kwargs):
        """Log warning message with agent context."""
        extra = self._add_context(kwargs.pop("extra", None))
        self.logger.warning(msg, extra=extra, **kwargs)
        
    def error(self, msg: str, **kwargs):
        """Log error message with agent context."""
        extra = self._add_context(kwargs.pop("extra", None))
        self.logger.error(msg, extra=extra, **kwargs)
        
    def critical(self, msg: str, **kwargs):
        """Log critical message with agent context."""
        extra = self._add_context(kwargs.pop("extra", None))
        self.logger.critical(msg, extra=extra, **kwargs)


# Initialize logging on module import
from context_agent_app.config import LOG_LEVEL, LOG_FORMAT, LOG_FILE
setup_logging(log_level=LOG_LEVEL, log_format=LOG_FORMAT, log_file=LOG_FILE)
