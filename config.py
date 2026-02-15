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

# Ollama configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

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


# ==================== VERIFIABLE MODE ====================

# Enable verifiable mode for research-oriented, evidence-grounded generation
ENABLE_VERIFIABLE_MODE = os.getenv("ENABLE_VERIFIABLE_MODE", "true").lower() == "true"

# Claim validation thresholds
VERIFIABLE_VERIFIED_THRESHOLD = float(os.getenv("VERIFIABLE_VERIFIED_THRESHOLD", "0.5"))
VERIFIABLE_REJECTED_THRESHOLD = float(os.getenv("VERIFIABLE_REJECTED_THRESHOLD", "0.2"))
VERIFIABLE_MIN_EVIDENCE = int(os.getenv("VERIFIABLE_MIN_EVIDENCE", "1"))
VERIFIABLE_MAX_EVIDENCE_PER_CLAIM = int(os.getenv("VERIFIABLE_MAX_EVIDENCE_PER_CLAIM", "5"))
VERIFIABLE_MIN_INDEPENDENT_SOURCES = int(os.getenv("VERIFIABLE_MIN_INDEPENDENT_SOURCES", "1"))

# Evidence retrieval settings
VERIFIABLE_MIN_EVIDENCE_LENGTH = int(os.getenv("VERIFIABLE_MIN_EVIDENCE_LENGTH", "15"))
VERIFIABLE_RELEVANCE_THRESHOLD = float(os.getenv("VERIFIABLE_RELEVANCE_THRESHOLD", "0.2"))

# Evidence consistency scoring (hybrid: rule-based + model-based)
VERIFIABLE_CONSISTENCY_ENABLED = os.getenv("VERIFIABLE_CONSISTENCY_ENABLED", "true").lower() == "true"
VERIFIABLE_CONSISTENCY_PROVIDER = os.getenv("VERIFIABLE_CONSISTENCY_PROVIDER", "ollama")
VERIFIABLE_CONSISTENCY_THRESHOLD = float(os.getenv("VERIFIABLE_CONSISTENCY_THRESHOLD", "0.5"))

# Agent configuration
VERIFIABLE_AGENT_MIN_CONFIDENCE = float(os.getenv("VERIFIABLE_AGENT_MIN_CONFIDENCE", "0.4"))
VERIFIABLE_STRICT_MODE = os.getenv("VERIFIABLE_STRICT_MODE", "true").lower() == "true"


# ==================== INTERACTIVE VERIFIABILITY ASSESSMENT (RESEARCH) ====================

# Input sufficiency thresholds (for pre-generation warnings)
VERIFIABLE_MIN_INPUT_TOKENS = int(os.getenv("VERIFIABLE_MIN_INPUT_TOKENS", "100"))
VERIFIABLE_MIN_INPUT_CHUNKS = int(os.getenv("VERIFIABLE_MIN_INPUT_CHUNKS", "2"))

# Negative control detection (all evidence = 0 scenario)
VERIFIABLE_NEGATIVE_CONTROL_THRESHOLD = float(os.getenv("VERIFIABLE_NEGATIVE_CONTROL_THRESHOLD", "0.0"))
"""Evidence nodes count below which we mark as negative control"""

# Graph export settings
VERIFIABLE_GRAPH_EXPORT_FORMATS = ["graphml", "json", "png"]
VERIFIABLE_GRAPH_DPI = int(os.getenv("VERIFIABLE_GRAPH_DPI", "150"))
VERIFIABLE_GRAPH_FIGSIZE_WIDTH = int(os.getenv("VERIFIABLE_GRAPH_FIGSIZE_WIDTH", "14"))
VERIFIABLE_GRAPH_FIGSIZE_HEIGHT = int(os.getenv("VERIFIABLE_GRAPH_FIGSIZE_HEIGHT", "10"))

# High rejection rate warning
VERIFIABLE_HIGH_REJECTION_THRESHOLD = float(os.getenv("VERIFIABLE_HIGH_REJECTION_THRESHOLD", "0.7"))
"""Rejection rate above which we show info about correct abstention"""

# Traceability metrics
VERIFIABLE_MIN_TRACEABILITY_FOR_GOOD_QUALITY = float(
    os.getenv("VERIFIABLE_MIN_TRACEABILITY_FOR_GOOD_QUALITY", "0.7")
)
"""Expected minimum traceability rate for research-grade output"""


# ==================== ABLATION FLAGS (For Research Experiments) ====================

# Evidence-first generation (System 2)
ENABLE_EVIDENCE_FIRST = os.getenv("ENABLE_EVIDENCE_FIRST", "true").lower() == "true"
"""If False, generate claim_text immediately without waiting for evidence"""

# Conflict detection in evidence
ENABLE_CONFLICT_DETECTION = os.getenv("ENABLE_CONFLICT_DETECTION", "true").lower() == "true"
"""If False, skip contradiction detection between evidence sources"""

# Graph-based confidence propagation
ENABLE_GRAPH_CONFIDENCE = os.getenv("ENABLE_GRAPH_CONFIDENCE", "true").lower() == "true"
"""If False, don't propagate confidence through claim-evidence graph"""

# Dependency blocking (reject claims with undefined dependencies)
ENABLE_DEPENDENCY_BLOCKING = os.getenv("ENABLE_DEPENDENCY_BLOCKING", "false").lower() == "true"
"""If True, reject claims that reference undefined concepts"""

# Multi-source requirement
ENABLE_MULTI_SOURCE_REQUIREMENT = os.getenv("ENABLE_MULTI_SOURCE_REQUIREMENT", "false").lower() == "true"
"""If True, require k>=2 independent sources for verification"""

# Consistency scoring
ENABLE_CONSISTENCY_SCORING = os.getenv("ENABLE_CONSISTENCY_SCORING", "true").lower() == "true"
"""If False, skip semantic consistency checks between evidence"""


# ==================== TEXT QUALITY & ROBUSTNESS ====================

# Text quality assessment thresholds
MIN_ALPHABETIC_RATIO = float(os.getenv("MIN_ALPHABETIC_RATIO", "0.2"))
"""Minimum alphabetic character ratio (letters / total chars)"""

MAX_CID_RATIO = float(os.getenv("MAX_CID_RATIO", "0.001"))
"""Maximum CID glyph ratio (corrupted PDF indicators)"""

MIN_PRINTABLE_RATIO = float(os.getenv("MIN_PRINTABLE_RATIO", "0.9"))
"""Minimum printable character ratio"""

MIN_INPUT_CHARS_ABSOLUTE = int(os.getenv("MIN_INPUT_CHARS_ABSOLUTE", "100"))
"""Absolute minimum input characters before marking unverifiable"""

MIN_INPUT_CHARS_FOR_VERIFICATION = int(os.getenv("MIN_INPUT_CHARS_FOR_VERIFICATION", "500"))
"""Recommended minimum input characters for robust verification"""

# PDF OCR fallback
ENABLE_OCR_FALLBACK = os.getenv("ENABLE_OCR_FALLBACK", "true").lower() == "true"
"""Enable OCR fallback for PDFs when text extraction fails quality checks"""

OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "5"))
"""Maximum number of PDF pages to OCR"""

# Multi-source policy
STRICT_MULTI_SOURCE = os.getenv("STRICT_MULTI_SOURCE", "false").lower() == "true"
"""If False, allow MIN_SUPPORTING_SOURCES=1 when only one source available"""


# ==================== DOMAIN PROFILES (RESEARCH RIGOR) ====================

# Domain profile selection (for research-grade verifiability)
DEFAULT_DOMAIN_PROFILE = os.getenv("DEFAULT_DOMAIN_PROFILE", "physics")

# Claim granularity policy
MAX_PROPOSITIONS_PER_CLAIM = int(os.getenv("MAX_PROPOSITIONS_PER_CLAIM", "1"))
"""Maximum propositions per claim (1 = atomic claims)"""

# Evidence sufficiency policy
MIN_ENTAILMENT_PROB = float(os.getenv("MIN_ENTAILMENT_PROB", "0.60"))
"""Minimum entailment probability for verification"""

MIN_SUPPORTING_SOURCES = int(os.getenv("MIN_SUPPORTING_SOURCES", "2"))
"""Minimum independent supporting sources required"""

MAX_CONTRADICTION_PROB = float(os.getenv("MAX_CONTRADICTION_PROB", "0.30"))
"""Maximum contradiction probability allowed"""

# Cross-claim dependency checking
ENABLE_DEPENDENCY_WARNINGS = os.getenv("ENABLE_DEPENDENCY_WARNINGS", "true").lower() == "true"
"""Enable warnings for undefined term references in claims"""

STRICT_DEPENDENCY_ENFORCEMENT = os.getenv("STRICT_DEPENDENCY_ENFORCEMENT", "false").lower() == "true"
"""If True, downgrade claims with undefined dependencies to LOW_CONFIDENCE"""


# ==================== DEBUG & DIAGNOSTICS ====================

DEBUG_VERIFICATION = os.getenv("DEBUG_VERIFICATION", "false").lower() == "true"
"""Enable detailed claim-level debug logging during verification"""

RELAXED_VERIFICATION_MODE = os.getenv("RELAXED_VERIFICATION_MODE", "false").lower() == "true"
"""If True, use relaxed thresholds for testing (MIN_ENTAILMENT=0.50, MIN_SOURCES=1)"""

DEBUG_RETRIEVAL_HEALTH = os.getenv("DEBUG_RETRIEVAL_HEALTH", "false").lower() == "true"
"""Enable retrieval health check diagnostics"""

DEBUG_NLI_DISTRIBUTION = os.getenv("DEBUG_NLI_DISTRIBUTION", "false").lower() == "true"
"""Enable NLI output distribution analysis"""

DEBUG_CHUNKING = os.getenv("DEBUG_CHUNKING", "false").lower() == "true"
"""Enable source chunking validation diagnostics"""

MAX_CLAIMS_TO_DEBUG = int(os.getenv("MAX_CLAIMS_TO_DEBUG", "50"))
"""Maximum number of claims to print in debug output (to avoid log overflow)"""

SAVE_DEBUG_REPORT = os.getenv("SAVE_DEBUG_REPORT", "true").lower() == "true"
"""Save JSON debug report to file after verification"""

DEBUG_REPORT_PATH = os.getenv("DEBUG_REPORT_PATH", "outputs/debug_session_report.json")
"""Path to save JSON debug report"""

# Default verification thresholds (strict mode)
MIN_ENTAILMENT_PROB_DEFAULT = float(os.getenv("MIN_ENTAILMENT_PROB_DEFAULT", "0.60"))
MIN_SUPPORTING_SOURCES_DEFAULT = int(os.getenv("MIN_SUPPORTING_SOURCES_DEFAULT", "2"))
MAX_CONTRADICTION_PROB_DEFAULT = float(os.getenv("MAX_CONTRADICTION_PROB_DEFAULT", "0.30"))

# Relaxed mode thresholds (only active if RELAXED_VERIFICATION_MODE=True)
MIN_ENTAILMENT_PROB_RELAXED = float(os.getenv("MIN_ENTAILMENT_PROB_RELAXED", "0.50"))
MIN_SUPPORTING_SOURCES_RELAXED = int(os.getenv("MIN_SUPPORTING_SOURCES_RELAXED", "1"))
MAX_CONTRADICTION_PROB_RELAXED = float(os.getenv("MAX_CONTRADICTION_PROB_RELAXED", "0.50"))

# Legacy names for backward compatibility
RELAXED_MIN_ENTAILMENT_PROB = MIN_ENTAILMENT_PROB_RELAXED
RELAXED_MIN_SUPPORTING_SOURCES = MIN_SUPPORTING_SOURCES_RELAXED
RELAXED_MAX_CONTRADICTION_PROB = MAX_CONTRADICTION_PROB_RELAXED

# URL ingestion feature flag
ENABLE_URL_SOURCES = os.getenv("ENABLE_URL_SOURCES", "true").lower() == "true"
"""Enable ingestion of YouTube videos and web articles as evidence sources"""

# Evidence store validation
MIN_INPUT_CHARS_FOR_VERIFICATION = int(os.getenv("MIN_INPUT_CHARS_FOR_VERIFICATION", "500"))
"""Minimum input text length required for verification (warns if below, errors if <100)"""


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


# ==================== DOMAIN PROFILES ====================

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class DomainProfile:
    """
    Domain-specific validation profile for research-grade verifiability.
    
    Each domain has specific claim types, evidence expectations, and validation rules.
    Enables domain-specific rigor while maintaining general framework.
    
    Attributes:
        name: Domain identifier (physics, discrete_math, algorithms)
        display_name: Human-readable name
        description: Domain description
        allowed_claim_types: Claim types relevant to this domain
        evidence_type_expectations: Expected evidence types per claim type
        require_units: Whether unit checking is required (physics)
        require_proof_steps: Whether proof-step strictness is enforced (discrete_math)
        require_pseudocode: Whether pseudocode checks are required (algorithms)
        require_equations: Whether equations must be present (physics, algorithms)
        strict_dependencies: Whether to enforce strict dependency checking
    """
    name: str
    display_name: str
    description: str
    allowed_claim_types: List[str]
    evidence_type_expectations: Dict[str, List[str]]
    require_units: bool = False
    require_proof_steps: bool = False
    require_pseudocode: bool = False
    require_equations: bool = False
    strict_dependencies: bool = False


# Define domain profiles
DOMAIN_PROFILES: Dict[str, DomainProfile] = {
    "physics": DomainProfile(
        name="physics",
        display_name="Physics",
        description="Physics domain with equations, units, and physical laws",
        allowed_claim_types=["definition", "equation", "example", "misconception"],
        evidence_type_expectations={
            "definition": ["transcript", "notes", "external"],
            "equation": ["transcript", "notes", "external", "equation"],
            "example": ["transcript", "notes", "external"],
            "misconception": ["transcript", "notes"]
        },
        require_units=True,
        require_proof_steps=False,
        require_pseudocode=False,
        require_equations=True,
        strict_dependencies=False
    ),
    "discrete_math": DomainProfile(
        name="discrete_math",
        display_name="Discrete Mathematics",
        description="Discrete math domain with definitions, proofs, and formal logic",
        allowed_claim_types=["definition", "example", "misconception"],
        evidence_type_expectations={
            "definition": ["transcript", "notes", "external"],
            "example": ["transcript", "notes", "external"],
            "misconception": ["transcript", "notes"]
        },
        require_units=False,
        require_proof_steps=True,
        require_pseudocode=False,
        require_equations=False,
        strict_dependencies=True
    ),
    "algorithms": DomainProfile(
        name="algorithms",
        display_name="Algorithms & Data Structures",
        description="Algorithms domain with pseudocode, complexity analysis, and implementations",
        allowed_claim_types=["definition", "equation", "example", "misconception"],
        evidence_type_expectations={
            "definition": ["transcript", "notes", "external"],
            "equation": ["transcript", "notes", "external", "equation"],
            "example": ["transcript", "notes", "external"],
            "misconception": ["transcript", "notes"]
        },
        require_units=False,
        require_proof_steps=False,
        require_pseudocode=True,
        require_equations=False,
        strict_dependencies=False
    )
}


def get_domain_profile(domain_name: str = None) -> DomainProfile:
    """
    Get domain profile by name.
    
    Args:
        domain_name: Domain name (physics, discrete_math, algorithms). 
                     If None, returns default domain.
    
    Returns:
        DomainProfile instance
    
    Raises:
        ValueError: If domain name is invalid
    """
    if domain_name is None:
        domain_name = DEFAULT_DOMAIN_PROFILE
    
    if domain_name not in DOMAIN_PROFILES:
        raise ValueError(
            f"Invalid domain: {domain_name}. "
            f"Valid domains: {', '.join(DOMAIN_PROFILES.keys())}"
        )
    
    return DOMAIN_PROFILES[domain_name]
