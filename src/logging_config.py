"""
Production-level logging configuration module.

This module provides comprehensive logging setup with:
- Structured logging (JSON format)
- File rotation and retention
- Environment-aware configuration
- Performance optimization
- Error tracking
- Security considerations
"""

import logging
import logging.handlers
import json
import sys
from pathlib import Path
from datetime import datetime
import os
from typing import Optional, Dict, Any


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    
    Outputs logs in JSON format for easy parsing and analysis
    by logging aggregation tools (ELK, Splunk, CloudWatch, etc.)
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format
        
        Returns:
            JSON-formatted log string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "process_id": record.process,
            "thread_id": record.thread,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info) if os.getenv("LOG_INCLUDE_TRACEBACK", "true").lower() == "true" else None
            }
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_data["user_id"] = record.user_id
        if hasattr(record, 'session_id'):
            log_data["session_id"] = record.session_id
        if hasattr(record, 'request_id'):
            log_data["request_id"] = record.request_id
        
        return json.dumps(log_data, default=str)


class TextFormatter(logging.Formatter):
    """
    Readable text formatter for console output.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as readable text.
        
        Args:
            record: Log record to format
        
        Returns:
            Formatted log string
        """
        if os.getenv("LOG_INCLUDE_TIMESTAMP", "true").lower() == "true":
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            prefix = f"[{timestamp}] [{record.levelname:8s}]"
        else:
            prefix = f"[{record.levelname:8s}]"
        
        message = f"{prefix} {record.name}:{record.funcName}:{record.lineno} - {record.getMessage()}"
        
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"
        
        return message


def setup_logging() -> logging.Logger:
    """
    Set up production-level logging with all handlers.
    
    Configuration is read from environment variables:
    - LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - LOG_FORMAT: json or text
    - LOG_OUTPUT: console, file, or both
    - LOG_DIR: Directory for log files
    - LOG_FILE_NAME: Log filename pattern
    - LOG_MAX_SIZE_MB: Max file size before rotation
    - LOG_BACKUP_COUNT: Number of backup files
    - LOG_RETENTION_DAYS: Days to retain logs
    
    Returns:
        Configured logger instance
    """
    
    # Read environment configuration
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "json").lower()
    log_output = os.getenv("LOG_OUTPUT", "both").lower()
    log_dir = os.getenv("LOG_DIR", "./logs")
    log_file_name = os.getenv("LOG_FILE_NAME", "smart_notes_{date}.log")
    max_bytes = int(os.getenv("LOG_MAX_SIZE_MB", "100")) * 1024 * 1024
    backup_count = int(os.getenv("LOG_BACKUP_COUNT", "10"))
    
    # Create log directory if it doesn't exist
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers = []
    
    # Choose formatter
    if log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter()
    
    # Console handler
    if log_output in ("console", "both"):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level))
        
        # Only errors to console in production
        if os.getenv("ENVIRONMENT", "development").lower() != "development":
            if os.getenv("LOG_CONSOLE_ONLY_ERRORS", "false").lower() == "true":
                console_handler.setLevel(logging.ERROR)
        
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_output in ("file", "both"):
        # Replace {date} in filename
        log_filename = log_file_name.replace("{date}", datetime.now().strftime("%Y-%m-%d"))
        log_filepath = log_dir / log_filename
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_filepath,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Set up cleanup of old log files (retention)
        retention_days = int(os.getenv("LOG_RETENTION_DAYS", "30"))
        _cleanup_old_logs(log_dir, retention_days)
    
    # Log configuration summary
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={log_level}, format={log_format}, output={log_output}")
    logger.info(f"Log directory: {log_dir.absolute()}")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Module name (typically __name__)
    
    Returns:
        Logger instance
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    return logging.getLogger(name)


def _cleanup_old_logs(log_dir: Path, retention_days: int) -> None:
    """
    Clean up log files older than retention period.
    
    Args:
        log_dir: Directory containing log files
        retention_days: Days to retain
    """
    try:
        current_time = datetime.now().timestamp()
        for log_file in log_dir.glob("*.log*"):
            file_age_days = (current_time - log_file.stat().st_mtime) / 86400
            if file_age_days > retention_days:
                log_file.unlink()
                logger = logging.getLogger(__name__)
                logger.debug(f"Deleted old log file: {log_file}")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error cleaning up old logs: {e}")


def log_with_context(
    logger: logging.Logger,
    level: str,
    message: str,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """
    Log a message with additional context.
    
    Args:
        logger: Logger instance
        level: Log level (debug, info, warning, error, critical)
        message: Log message
        context: Dictionary of context data
        **kwargs: Additional fields to log
    
    Example:
        >>> logger.info("Processing complete", extra={
        ...     "session_id": session_id,
        ...     "duration_seconds": 5.2,
        ...     "topics_extracted": 10
        ... })
    """
    extra = context or {}
    extra.update(kwargs)
    
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(message, extra=extra)


def configure_third_party_logging() -> None:
    """
    Configure logging for third-party libraries.
    
    Suppresses verbose logs from libraries like:
    - urllib3
    - requests
    - openai
    - streamlit
    """
    
    # Suppress verbose library logging
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("streamlit").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.debug("Third-party library logging configured")


# Initialize logging on module import
setup_logging()
configure_third_party_logging()
