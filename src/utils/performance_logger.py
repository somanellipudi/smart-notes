"""
Performance logging with session tracking and rolling file rotation.

This module provides comprehensive performance tracking for analyzing system bottlenecks:
- Rolling log files (10MB per file, keeps 10 backups)
- Session/Request ID tracking across all operations
- Timestamp and duration for each operation
- Automatic rotation to prevent disk overflow
- Easy analysis of which processes take longest
"""

import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from contextvars import ContextVar
import json

# Context variable for tracking current session ID across async calls
current_session_id: ContextVar[Optional[str]] = ContextVar('session_id', default=None)


class PerformanceLogFormatter(logging.Formatter):
    """
    Custom formatter that includes session ID and performance metrics.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with session ID and timing information."""
        # Get session ID from context
        session_id = current_session_id.get()
        if session_id:
            record.session_id = session_id
        else:
            record.session_id = "no-session"
        
        # Add timestamp in ISO format
        record.timestamp = datetime.utcnow().isoformat()
        
        # Format the message
        return super().format(record)


def setup_performance_logging(
    log_dir: str = "logs/performance",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 10,
    log_level: int = logging.INFO
) -> logging.Logger:
    """
    Setup rolling performance log file with session tracking.
    
    Args:
        log_dir: Directory for log files
        max_bytes: Maximum size per log file (default: 10MB)
        backup_count: Number of backup files to keep (default: 10)
        log_level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create performance logger
    perf_logger = logging.getLogger("performance")
    perf_logger.setLevel(log_level)
    perf_logger.propagate = False  # Don't propagate to root logger
    
    # Remove existing handlers
    perf_logger.handlers.clear()
    
    # Create rotating file handler
    log_file = log_path / "performance.log"
    handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    
    # Set formatter with session ID
    formatter = PerformanceLogFormatter(
        fmt='%(timestamp)s | SESSION=%(session_id)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    perf_logger.addHandler(handler)
    
    return perf_logger


class PerformanceTimer:
    """
    Context manager for timing operations and logging performance.
    
    Usage:
        with PerformanceTimer("ocr_extraction", session_id="session_123"):
            # ... perform OCR ...
            pass
        # Automatically logs duration
    """
    
    def __init__(
        self,
        operation_name: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize performance timer.
        
        Args:
            operation_name: Name of the operation being timed
            session_id: Session/Request ID for tracking
            metadata: Additional metadata to log (e.g., file size, claim count)
            logger: Logger to use (defaults to performance logger)
        """
        self.operation_name = operation_name
        self.session_id = session_id
        self.metadata = metadata or {}
        self.logger = logger or logging.getLogger("performance")
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        """Start timing."""
        # Set session ID in context if provided
        if self.session_id:
            current_session_id.set(self.session_id)
        
        self.start_time = time.perf_counter()
        self.logger.info(
            f"START | operation={self.operation_name} | metadata={json.dumps(self.metadata)}"
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log duration."""
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        
        if exc_type is not None:
            # Operation failed
            self.logger.error(
                f"FAILED | operation={self.operation_name} | "
                f"duration={self.duration:.3f}s | error={exc_type.__name__}: {exc_val}"
            )
        else:
            # Operation succeeded
            self.logger.info(
                f"COMPLETE | operation={self.operation_name} | "
                f"duration={self.duration:.3f}s | metadata={json.dumps(self.metadata)}"
            )
        
        return False  # Don't suppress exceptions
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata during operation."""
        self.metadata[key] = value


class OperationLogger:
    """
    Helper class for logging performance without context manager.
    
    Usage:
        logger = OperationLogger("embedding_generation", session_id="session_123")
        logger.start()
        # ... perform operation ...
        logger.complete(metadata={"num_embeddings": 50})
    """
    
    def __init__(
        self,
        operation_name: str,
        session_id: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize operation logger.
        
        Args:
            operation_name: Name of the operation
            session_id: Session/Request ID
            logger: Logger to use (defaults to performance logger)
        """
        self.operation_name = operation_name
        self.session_id = session_id
        self.logger = logger or logging.getLogger("performance")
        self.start_time = None
        self.metadata: Dict[str, Any] = {}
        
        # Set session ID in context
        if self.session_id:
            current_session_id.set(self.session_id)
    
    def start(self, metadata: Optional[Dict[str, Any]] = None):
        """Start timing the operation."""
        if metadata:
            self.metadata.update(metadata)
        
        self.start_time = time.perf_counter()
        self.logger.info(
            f"START | operation={self.operation_name} | metadata={json.dumps(self.metadata)}"
        )
    
    def complete(self, metadata: Optional[Dict[str, Any]] = None):
        """Mark operation as complete and log duration."""
        if not self.start_time:
            self.logger.warning(f"complete() called without start() for {self.operation_name}")
            return
        
        if metadata:
            self.metadata.update(metadata)
        
        duration = time.perf_counter() - self.start_time
        self.logger.info(
            f"COMPLETE | operation={self.operation_name} | "
            f"duration={duration:.3f}s | metadata={json.dumps(self.metadata)}"
        )
    
    def fail(self, error_msg: str, metadata: Optional[Dict[str, Any]] = None):
        """Mark operation as failed and log error."""
        if not self.start_time:
            self.logger.warning(f"fail() called without start() for {self.operation_name}")
            return
        
        if metadata:
            self.metadata.update(metadata)
        
        duration = time.perf_counter() - self.start_time
        self.logger.error(
            f"FAILED | operation={self.operation_name} | "
            f"duration={duration:.3f}s | error={error_msg} | metadata={json.dumps(self.metadata)}"
        )


def set_session_id(session_id: str):
    """Set the current session ID for all subsequent logs."""
    current_session_id.set(session_id)


def get_session_id() -> Optional[str]:
    """Get the current session ID."""
    return current_session_id.get()


# Initialize performance logger on module import
performance_logger = setup_performance_logging()
