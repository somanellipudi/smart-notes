"""
Production-level configuration module for Smart Notes application.

This module loads all configuration from environment variables (.env file)
following industry standards for secrets management and environment-specific settings.

Configuration is organized into logical sections:
- Environment & Debug
- API Keys & Authentication
- LLM Models
- Audio Processing
- Text Processing
- Reasoning Pipeline
- Evaluation
- Logging
- Caching
- Database
- Security
- Feature Flags
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Literal

# Load environment variables from .env file
load_dotenv()


# ==================== ENVIRONMENT & DEBUG ====================

ENVIRONMENT: Literal["development", "staging", "production"] = os.getenv(
    "ENVIRONMENT", "development"
).lower()  # type: ignore

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
ENABLE_DEBUG = DEBUG

# Base directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SESSIONS_DIR = OUTPUT_DIR / "sessions"
LOGS_DIR = PROJECT_ROOT / os.getenv("LOG_DIR", "logs")

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


# ==================== API KEYS & AUTHENTICATION ====================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID", "")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT_ID = os.getenv("LANGCHAIN_PROJECT_ID", "")

# Validation for required API keys
if ENVIRONMENT == "production" and not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is required in production environment")


# ==================== LLM MODELS ====================

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4000"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "1.0"))
LLM_FREQUENCY_PENALTY = float(os.getenv("LLM_FREQUENCY_PENALTY", "0.0"))
LLM_PRESENCE_PENALTY = float(os.getenv("LLM_PRESENCE_PENALTY", "0.0"))

# Timeout configuration for API calls
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))


# ==================== AUDIO PROCESSING ====================

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
MAX_AUDIO_DURATION = int(os.getenv("MAX_AUDIO_DURATION", "7200"))  # 2 hours
MIN_AUDIO_DURATION = int(os.getenv("MIN_AUDIO_DURATION", "3"))  # 3 seconds
AUDIO_CACHE_DIR = PROJECT_ROOT / os.getenv("AUDIO_CACHE_DIR", "data/audio_cache")
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ==================== TEXT PROCESSING ====================

MIN_SEGMENT_LENGTH = int(os.getenv("MIN_SEGMENT_LENGTH", "50"))
MAX_SEGMENT_LENGTH = int(os.getenv("MAX_SEGMENT_LENGTH", "1000"))
TOPIC_BOUNDARY_THRESHOLD = float(os.getenv("TOPIC_BOUNDARY_THRESHOLD", "0.6"))


# ==================== REASONING PIPELINE ====================

MAX_TOPICS = int(os.getenv("MAX_TOPICS", "10"))
MAX_CONCEPTS_PER_TOPIC = int(os.getenv("MAX_CONCEPTS_PER_TOPIC", "8"))
MAX_FAQS = int(os.getenv("MAX_FAQS", "15"))
MAX_WORKED_EXAMPLES = int(os.getenv("MAX_WORKED_EXAMPLES", "5"))
MAX_MISCONCEPTIONS = int(os.getenv("MAX_MISCONCEPTIONS", "5"))


# ==================== EVALUATION THRESHOLDS ====================

MIN_REASONING_CORRECTNESS = float(os.getenv("MIN_REASONING_CORRECTNESS", "0.7"))
MIN_STRUCTURAL_ACCURACY = float(os.getenv("MIN_STRUCTURAL_ACCURACY", "0.8"))
MAX_HALLUCINATION_RATE = float(os.getenv("MAX_HALLUCINATION_RATE", "0.15"))
MIN_EDUCATIONAL_USEFULNESS = float(os.getenv("MIN_EDUCATIONAL_USEFULNESS", "3.5"))


# ==================== LOGGING ====================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "json")  # json or text
LOG_OUTPUT = os.getenv("LOG_OUTPUT", "both")  # console, file, or both
LOG_FILE_NAME = os.getenv("LOG_FILE_NAME", "smart_notes_{date}.log")
LOG_MAX_SIZE_MB = int(os.getenv("LOG_MAX_SIZE_MB", "100"))
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "10"))
LOG_RETENTION_DAYS = int(os.getenv("LOG_RETENTION_DAYS", "30"))
LOG_INCLUDE_TIMESTAMP = os.getenv("LOG_INCLUDE_TIMESTAMP", "true").lower() == "true"
LOG_CONSOLE_ONLY_ERRORS = os.getenv("LOG_CONSOLE_ONLY_ERRORS", "false").lower() == "true"


# ==================== CACHING ====================

USE_CACHE = os.getenv("USE_CACHE", "true").lower() == "true"
CACHE_TYPE = os.getenv("CACHE_TYPE", "local")  # local, redis, or memcached
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_DIR = PROJECT_ROOT / os.getenv("CACHE_DIR", "data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ==================== DATABASE ====================

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/smart_notes.db")
DATABASE_POOL_SIZE = int(os.getenv("DATABASE_POOL_SIZE", "5"))
DATABASE_ECHO = os.getenv("DATABASE_ECHO", "false").lower() == "true"


# ==================== SECURITY ====================

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8501").split(",")
SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "")  # Should be set in production


# ==================== STREAMLIT UI ====================

STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
STREAMLIT_HOST = os.getenv("STREAMLIT_HOST", "localhost")
STREAMLIT_UPLOAD_SIZE_MB = int(os.getenv("STREAMLIT_UPLOAD_SIZE_MB", "200"))


# ==================== FEATURE FLAGS ====================

ENABLE_LANGCHAIN_TRACING = os.getenv("ENABLE_LANGCHAIN_TRACING", "false").lower() == "true"
ENABLE_MULTI_LANGUAGE = os.getenv("ENABLE_MULTI_LANGUAGE", "false").lower() == "true"
ENABLE_REAL_TIME_STREAMING = os.getenv("ENABLE_REAL_TIME_STREAMING", "false").lower() == "true"


# ==================== MONITORING & TELEMETRY ====================

SENTRY_DSN = os.getenv("SENTRY_DSN", "")
ENABLE_TELEMETRY = os.getenv("ENABLE_TELEMETRY", "false").lower() == "true"
TELEMETRY_SAMPLE_RATE = float(os.getenv("TELEMETRY_SAMPLE_RATE", "0.1"))


# ==================== HELPER FUNCTIONS ====================


def get_config(key: str, default: str = "") -> str:
    """
    Get configuration value from environment.
    
    Args:
        key: Configuration key
        default: Default value if not found
    
    Returns:
        Configuration value
    """
    return os.getenv(key, default)


def is_production() -> bool:
    """Check if running in production environment."""
    return ENVIRONMENT == "production"


def is_development() -> bool:
    """Check if running in development environment."""
    return ENVIRONMENT == "development"


def is_staging() -> bool:
    """Check if running in staging environment."""
    return ENVIRONMENT == "staging"
