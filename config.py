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
ARTIFACTS_DIR = PROJECT_ROOT / os.getenv("ARTIFACTS_DIR", "artifacts")

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)
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

# Text cleaning (boilerplate removal)
CLEANING_ENABLED = os.getenv("CLEANING_ENABLED", "true").lower() == "true"
REPEAT_FRAC = float(os.getenv("REPEAT_FRAC", "0.30"))
MIN_LINE_LEN = int(os.getenv("MIN_LINE_LEN", "4"))
MAX_TITLE_LEN = int(os.getenv("MAX_TITLE_LEN", "40"))

BOILERPLATE_REGEX_RULES = [
    {"name": "unit", "pattern": r"^\s*(unit|chapter|module|lesson|week)\s*[:\-]?\s*\d+\b"},
    {"name": "course_meta", "pattern": r"\b(b\.?tech|btech|m\.?tech|semester|dept\.?|department|university|college|institute)\b"},
    {"name": "scan_watermark", "pattern": r"\b(scanned\s+with|camscanner|adobe\s+scan|genius\s+scan|microsoft\s+lens)\b"},
    {"name": "file_marker", "pattern": r"^\s*---\s*(from|page)\b"},
    {"name": "separator", "pattern": r"^\s*[-_=]{3,}\s*$"}
]

CODE_LINE_PROTECT_REGEXES = [
    r"\bO\s*\([^)]*\)",
    r"\bTheta\s*\([^)]*\)",
    r"[{}\[\]();]",
    r"\bfor\b|\bwhile\b|\bif\b|\belse\b|\breturn\b"
]

CODE_LINE_PROTECT_TOKENS = [
    "push",
    "pop",
    "enqueue",
    "dequeue",
    "stack",
    "queue",
    "heap",
    "graph",
    "dfs",
    "bfs"
]


# ==================== PDF OCR FALLBACK ====================

ENABLE_OCR_FALLBACK = os.getenv("ENABLE_OCR_FALLBACK", "true").lower() == "true"
OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "10"))
OCR_DPI = int(os.getenv("OCR_DPI", "200"))


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

# Dense retrieval settings (sentence-transformers)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/e5-base-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
EMBEDDING_NORMALIZE = os.getenv("EMBEDDING_NORMALIZE", "true").lower() == "true"
DENSE_RETRIEVAL_TOP_K = int(os.getenv("DENSE_RETRIEVAL_TOP_K", "20"))
DENSE_RETRIEVAL_MIN_SIMILARITY = float(os.getenv("DENSE_RETRIEVAL_MIN_SIMILARITY", "0.2"))

# Adaptive evidence sufficiency settings
MAX_EVIDENCE_PER_CLAIM = int(os.getenv("MAX_EVIDENCE_PER_CLAIM", "6"))
SUFFICIENCY_TAU = float(os.getenv("SUFFICIENCY_TAU", "0.8"))
EVIDENCE_DIVERSITY_MIN_SOURCES = int(os.getenv("EVIDENCE_DIVERSITY_MIN_SOURCES", "2"))

# Optional reranker settings
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "false").lower() == "true"
RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "20"))
RERANKER_KEEP_K = int(os.getenv("RERANKER_KEEP_K", "5"))

# Evidence consistency scoring (hybrid: rule-based + model-based)
VERIFIABLE_CONSISTENCY_ENABLED = os.getenv("VERIFIABLE_CONSISTENCY_ENABLED", "true").lower() == "true"
VERIFIABLE_CONSISTENCY_PROVIDER = os.getenv("VERIFIABLE_CONSISTENCY_PROVIDER", "ollama")
VERIFIABLE_CONSISTENCY_THRESHOLD = float(os.getenv("VERIFIABLE_CONSISTENCY_THRESHOLD", "0.5"))

# Agent configuration
VERIFIABLE_AGENT_MIN_CONFIDENCE = float(os.getenv("VERIFIABLE_AGENT_MIN_CONFIDENCE", "0.4"))
VERIFIABLE_STRICT_MODE = os.getenv("VERIFIABLE_STRICT_MODE", "true").lower() == "true"


# ==================== VERIFIABLE MODE PERFORMANCE OPTIMIZATION ====================

# Selective verification: limit number of claims to verify (for speed)
VERIFY_TOP_N_CLAIMS = int(os.getenv("VERIFY_TOP_N_CLAIMS", "250"))
"""Verify only top N claims by topical relevance; 0 means verify all"""

# High-risk-only verification: verify only claims with specific characteristics
VERIFY_HIGH_RISK_ONLY = os.getenv("VERIFY_HIGH_RISK_ONLY", "false").lower() == "true"
"""If True, verify only NUMERIC, COMPLEXITY, DEFINITION, CODE claim types and claims with negations"""

# Batch NLI processing
NLI_BATCH_SIZE = int(os.getenv("NLI_BATCH_SIZE", "32"))
"""Batch size for NLI inference; larger = faster but more memory"""

# Cache configuration for performance
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
"""Batch size for embedding computation"""

EMBEDDING_CACHE_DISK = os.getenv("EMBEDDING_CACHE_DISK", "true").lower() == "true"
"""Cache embeddings to disk (Artifact Store) for reuse across runs"""

NLI_CACHE_DISK = os.getenv("NLI_CACHE_DISK", "true").lower() == "true"
"""Cache NLI results to disk keyed by (claim_hash, span_id, model_id)"""

# Profiling
ENABLE_PROFILING = os.getenv("ENABLE_PROFILING", "false").lower() == "true"
"""Profile verifiable mode performance and output to logs"""

PROFILING_ARTIFACTS_DIR = PROJECT_ROOT / os.getenv("PROFILING_ARTIFACTS_DIR", "profiling")
"""Directory for profiling artifacts"""
PROFILING_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ==================== ARTIFACT PERSISTENCE & CACHING ====================

# Artifact store (deterministic evidence persistence)
ENABLE_ARTIFACT_PERSISTENCE = os.getenv("ENABLE_ARTIFACT_PERSISTENCE", "true").lower() == "true"
ARTIFACTS_DIR = PROJECT_ROOT / os.getenv("ARTIFACTS_DIR", "artifacts")

# Cache controls (for individual pipeline stages)
EMBEDDING_CACHE_ENABLED = os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower() == "true"
RETRIEVAL_CACHE_ENABLED = os.getenv("RETRIEVAL_CACHE_ENABLED", "false").lower() == "true"
NLI_CACHE_ENABLED = os.getenv("NLI_CACHE_ENABLED", "false").lower() == "true"

# Random seed for reproducibility
GLOBAL_RANDOM_SEED = int(os.getenv("GLOBAL_RANDOM_SEED", "42"))


# ==================== ONLINE EVIDENCE AUGMENTATION ====================

# Enable online authority verification for evidence augmentation
ENABLE_ONLINE_VERIFICATION = os.getenv("ENABLE_ONLINE_VERIFICATION", "false").lower() == "true"

# Online verification cache settings
ONLINE_CACHE_ENABLED = os.getenv("ONLINE_CACHE_ENABLED", "true").lower() == "true"
ONLINE_RATE_LIMIT = float(os.getenv("ONLINE_RATE_LIMIT", "2.0"))  # Requests per second
ONLINE_TIMEOUT_SECONDS = int(os.getenv("ONLINE_TIMEOUT_SECONDS", "10"))  # Request timeout

# Minimum authority tier for solo verification (TIER_1=1, TIER_2=2, TIER_3=3)
# Tier 1/2 can verify claims alone. Tier 3 requires >=2 corroborating sources.
ONLINE_MIN_TIER_FOR_SOLO_VERIFICATION = int(os.getenv("ONLINE_MIN_TIER_FOR_SOLO_VERIFICATION", "2"))

# Maximum online sources to fetch per claim
ONLINE_MAX_SOURCES_PER_CLAIM = int(os.getenv("ONLINE_MAX_SOURCES_PER_CLAIM", "5"))


# ==================== CS-AWARE VERIFICATION SIGNALS ====================

# Enable CS-specific verification signals (numeric, complexity, code, negation)
ENABLE_CS_VERIFICATION_SIGNALS = os.getenv("ENABLE_CS_VERIFICATION_SIGNALS", "true").lower() == "true"

# CS verification signal weights (sum should ≈ 1.0 for balanced scoring)
WEIGHT_NUMERIC = float(os.getenv("WEIGHT_NUMERIC", "0.25"))  # Numeric consistency score weight
WEIGHT_COMPLEXITY = float(os.getenv("WEIGHT_COMPLEXITY", "0.25"))  # Complexity notation consistency weight
WEIGHT_CODE = float(os.getenv("WEIGHT_CODE", "0.30"))  # Code pattern anchoring weight
WEIGHT_NEGATION = float(os.getenv("WEIGHT_NEGATION", "0.20"))  # Negation mismatch penalty weight (penalty only)

# Evidence sufficiency for CS claims
REQUIRE_ANCHOR_TERMS_COMPLEXITY = os.getenv("REQUIRE_ANCHOR_TERMS_COMPLEXITY", "true").lower() == "true"
"""Require complexity-related anchor terms in evidence for COMPLEXITY_CLAIM"""

REQUIRE_ANCHOR_TERMS_DEFINITION = os.getenv("REQUIRE_ANCHOR_TERMS_DEFINITION", "true").lower() == "true"
"""Require definition-related anchor terms in evidence for DEFINITION_CLAIM"""

REQUIRE_ANCHOR_TERMS_CODE = os.getenv("REQUIRE_ANCHOR_TERMS_CODE", "true").lower() == "true"
"""Require code-related anchor terms in evidence for CODE_BEHAVIOR_CLAIM"""

MIN_ANCHOR_SCORE_FOR_EVIDENCE = float(os.getenv("MIN_ANCHOR_SCORE_FOR_EVIDENCE", "0.5"))
"""Minimum anchor term score to accept evidence for CS claims"""


# ==================== CONTRADICTION DETECTION ====================

# Enable hard contradiction gate (prevents VERIFIED status when contradiction detected)
ENABLE_CONTRADICTION_GATE = os.getenv("ENABLE_CONTRADICTION_GATE", "true").lower() == "true"
"""If enabled, claims with contradiction_prob > threshold cannot be VERIFIED"""

CONTRADICTION_GATE_THRESHOLD = float(os.getenv("CONTRADICTION_GATE_THRESHOLD", "0.6"))
"""Threshold for contradiction probability to trigger gate (default: 0.6)"""

# Enable CS-specific operation semantics rules (push/pop/enqueue/dequeue)
ENABLE_CS_OPERATION_RULES = os.getenv("ENABLE_CS_OPERATION_RULES", "false").lower() == "true"
"""If enabled, applies CS-specific semantic rules for data structure operations"""

CS_OPERATION_RULES = {
    # Stack operations
    "push": {"adds": True, "removes": False, "structure": "stack", "end": "top"},
    "pop": {"adds": False, "removes": True, "structure": "stack", "end": "top"},
    
    # Queue operations
    "enqueue": {"adds": True, "removes": False, "structure": "queue", "end": "rear"},
    "dequeue": {"adds": False, "removes": True, "structure": "queue", "end": "front"},
    
    # Heap operations
    "insert_heap": {"adds": True, "removes": False, "structure": "heap"},
    "extract_min": {"adds": False, "removes": True, "structure": "heap"},
    "extract_max": {"adds": False, "removes": True, "structure": "heap"},
}
"""CS operation semantic rules for contradiction detection"""


# ==================== CITATION RENDERING & DISPLAY ====================

# Enable citation features (embed citations in output)
ENABLE_CITATIONS = os.getenv("ENABLE_CITATIONS", "true").lower() == "true"
"""Toggle citation features globally"""

# Unverified claim handling
SHOW_UNVERIFIED_WITH_LABEL = os.getenv("SHOW_UNVERIFIED_WITH_LABEL", "true").lower() == "true"
"""Add '(needs evidence)' label to claims without supporting citations"""

SHOW_UNVERIFIED_OMIT = os.getenv("SHOW_UNVERIFIED_OMIT", "false").lower() == "true"
"""Omit claims entirely if they lack supporting citations (mutually exclusive with SHOW_UNVERIFIED_WITH_LABEL)"""

# Citation display settings
CITATION_MAX_PER_CLAIM = int(os.getenv("CITATION_MAX_PER_CLAIM", "3"))
"""Maximum number of citations to display per claim"""

SHOW_CITATION_CONFIDENCE = os.getenv("SHOW_CITATION_CONFIDENCE", "false").lower() == "true"
"""Display confidence scores alongside citations"""

CITATION_AUTHORITY_LABELS = os.getenv("CITATION_AUTHORITY_LABELS", "true").lower() == "true"
"""Show authority tier labels (TIER_1, TIER_2, TIER_3) with citations"""

CITATION_SNIPPET_MAX_CHARS = int(os.getenv("CITATION_SNIPPET_MAX_CHARS", "100"))
"""Maximum characters to display in citation snippets (truncate if longer)"""

# CS claim citation requirements
REQUIRE_CITATIONS_FOR_CS_CLAIMS = os.getenv("REQUIRE_CITATIONS_FOR_CS_CLAIMS", "true").lower() == "true"
"""Require citations for CS-specific claim types (COMPLEXITY_CLAIM, CODE_BEHAVIOR_CLAIM, DEFINITION_CLAIM, NUMERIC_CLAIM)"""

# Citation rendering formats
ENABLE_CITATION_HTML = os.getenv("ENABLE_CITATION_HTML", "false").lower() == "true"
"""Enable HTML citation rendering with collapsible panels (for web display)"""

CITATION_HTML_COLLAPSIBLE = os.getenv("CITATION_HTML_COLLAPSIBLE", "true").lower() == "true"
"""Use collapsible citation panels in HTML rendering"""


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

from src.policies.domain_profiles import (
    DomainProfile,
    DOMAIN_PROFILES,
    get_domain_profile as _get_domain_profile
)


def get_domain_profile(domain_name: str = None) -> DomainProfile:
    """
    Get domain profile by name.

    Args:
        domain_name: Domain name (physics, discrete_math, algorithms, cs).
                     If None, returns default domain.

    Returns:
        DomainProfile instance

    Raises:
        ValueError: If domain name is invalid
    """
    return _get_domain_profile(domain_name, DEFAULT_DOMAIN_PROFILE)
