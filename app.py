# CONTEXT FOR COPILOT:
# This repository implements Smart Notes, a Streamlit-based educational AI app
# with multi-modal ingestion and a multi-stage reasoning pipeline.
# The goal is to EXTEND the system with a research-oriented
# "Verifiable Mode" that enforces claim-based, evidence-grounded generation,
# without breaking existing functionality.

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
import tempfile
import hashlib
import textwrap
from typing import Tuple, Optional, Dict, Any, Callable, List
import requests

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '1'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import streamlit as st

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Set global random seed for reproducibility (before any model imports)
try:
    import config
    from src.utils.seed_control import set_global_seed
    if hasattr(config, 'GLOBAL_RANDOM_SEED'):
        set_global_seed(config.GLOBAL_RANDOM_SEED)
        logger.info(f"Global random seed set to {config.GLOBAL_RANDOM_SEED}")
except Exception as e:
    logger.warning(f"Could not set global random seed: {e}")

try:
    from src.audio.transcription import transcribe_audio
    from src.audio.image_ocr import ImageOCR, process_images
    from src.preprocessing.text_processing import preprocess_classroom_content
    from src.preprocessing.pdf_ingest import extract_pdf_text
    from src.preprocessing.url_ingest import fetch_url_text
    from src.ingestion.document_ingestor import ingest_document, IngestionDiagnostics
    from src.reasoning.verifiable_pipeline import VerifiablePipelineWrapper
    from src.evaluation.metrics import evaluate_session_output
    from src.study_book.session_manager import SessionManager
    from src.llm_provider import LLMProviderFactory
    from src.output_formatter import StreamingOutputDisplay
    from src.reasoning.fallback_handler import FallbackGenerator, PipelineEnhancer
    from src.streamlit_display import StreamlitProgressDisplay, QuickExportButtons
    from src.ui.input_validation import (
        has_any_input,
        get_input_status_message,
        validate_urls_for_processing,
        is_youtube_url,
    )
    from src.exporters.report_exporter import export_report_json, export_report_pdf
    from src.reporting.research_report import (
        build_report,
        save_reports,
        SessionMetadata,
        IngestionReport,
        VerificationSummary,
        ClaimEntry,
    )
    from src.persistence.run_history import (
        load_run_history,
        append_run,
        save_run_history,
    )
    from src.retrieval.artifact_store import generate_run_id
    from src.schema.output_schema import (
        ClassSessionOutput,
        Topic,
        Concept,
        EquationExplanation,
        WorkedExample,
        Misconception,
        FAQ,
        RealWorldConnection
    )
    import config
    logger.info("✅ All imports successful")
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    st.error(f"Failed to import required modules: {e}")
    st.stop()


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Smart Notes",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    :root {
        --apple-bg: #f5f5f7;
        --apple-card: #ffffff;
        --apple-text: #1d1d1f;
        --apple-subtle: #86868b;
        --apple-accent: #0a84ff;
        --apple-accent-secondary: #34c759;
        --apple-border: #e5e5ea;
        --apple-shadow: 0 8px 24px rgba(0,0,0,0.06);
        --apple-shadow-sm: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    html, body, [class*="stApp"] {
        background: var(--apple-bg);
        color: var(--apple-text);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    }
    
    .block-container {
        padding-top: 1.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    h1 {
        font-weight: 700;
        font-size: 2.2rem;
        letter-spacing: -0.04em;
        margin-bottom: 0.5rem;
        color: var(--apple-text);
    }
    
    h2 {
        font-weight: 600;
        font-size: 1.6rem;
        letter-spacing: -0.02em;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    h3 {
        font-weight: 600;
        font-size: 1.1rem;
        letter-spacing: -0.01em;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    p, label, span, div {
        color: var(--apple-text);
        line-height: 1.5;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 12px;
        padding: 0.65rem 1.2rem;
        font-weight: 500;
        font-size: 0.95rem;
        background: linear-gradient(135deg, var(--apple-accent) 0%, #0a7aff 100%);
        color: white;
        border: none;
        box-shadow: var(--apple-shadow-sm);
        transition: all 0.3s ease;
        letter-spacing: -0.01em;
    }
    
    .stButton>button:hover {
        opacity: 0.92;
        box-shadow: var(--apple-shadow);
        transform: translateY(-1px);
    }
    
    .stButton>button:active {
        transform: translateY(0px);
        opacity: 0.85;
    }
    
    /* Input elements */
    .stTextArea textarea, .stTextInput input, .stFileUploader {
        border-radius: 12px;
        border: 1px solid var(--apple-border) !important;
        box-shadow: var(--apple-shadow-sm);
        background: var(--apple-card) !important;
        font-size: 0.95rem;
        line-height: 1.6;
        transition: all 0.2s ease;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--apple-accent) !important;
        box-shadow: 0 0 0 3px rgba(10, 132, 255, 0.1), var(--apple-shadow-sm);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem;
        color: var(--apple-subtle);
        font-weight: 500;
        padding: 0.75rem 1rem;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--apple-accent);
        border-bottom: 2px solid var(--apple-accent);
    }
    
    /* Metrics */
    .stMetric {
        background: var(--apple-card);
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: var(--apple-shadow-sm);
        border: 1px solid var(--apple-border);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        box-shadow: var(--apple-shadow);
        transform: translateY(-2px);
    }
    
    /* Expanders */
    .stExpander {
        border: 1px solid var(--apple-border);
        border-radius: 16px;
        background: var(--apple-card);
        box-shadow: var(--apple-shadow-sm);
    }
    
    .stExpander [data-testid="stExpanderDetails"] {
        padding: 1.5rem;
    }
    
    /* Checkboxes */
    .stCheckbox {
        padding: 0.5rem 0;
    }
    
    .stCheckbox label {
        font-weight: 500;
        cursor: pointer;
        user-select: none;
    }
    
    /* Dividers */
    hr {
        border: none;
        height: 1px;
        background: var(--apple-border);
        margin: 1.5rem 0;
    }
    
    /* Messages */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 12px;
        border: none;
        background-color: rgba(10, 132, 255, 0.08);
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }
    
    .stSuccess {
        background-color: rgba(52, 199, 89, 0.08);
        color: #1d1d1f;
    }
    
    .stWarning {
        background-color: rgba(255, 159, 64, 0.08);
    }
    
    .stError {
        background-color: rgba(255, 59, 48, 0.08);
    }
    
    /* Dataframe */
    [data-testid="dataframe"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--apple-border);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--apple-card) 0%, #fafafa 100%);
        border-right: 1px solid var(--apple-border);
    }
    
    .sidebar-content {
        padding: 1rem;
    }
    
    /* Radio buttons */
    .stRadio label {
        font-weight: 500;
        margin: 0.75rem 0;
    }
    
    /* Select slider */
    .stSlider {
        margin: 1rem 0;
    }
    
    /* Captions */
    .stCaption {
        color: var(--apple-subtle);
        font-size: 0.85rem;
        line-height: 1.4;
        margin-top: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================================
# CACHED INITIALIZATION
# ============================================================================

@st.cache_resource
def initialize_ocr():
    """Initialize OCR once and cache it."""
    try:
        logger.info("Initializing OCR system...")
        ocr = ImageOCR()
        logger.info("✓ OCR initialized successfully")
        return ocr
    except Exception as e:
        logger.error(f"❌ OCR initialization failed: {e}")
        st.error(f"⚠️ OCR system unavailable: {e}\n\nImage processing will be disabled.")
        return None

@st.cache_resource
def initialize_pipeline():
    """Initialize reasoning pipeline once."""
    try:
        logger.info("Initializing reasoning pipeline...")
        pipeline = ReasoningPipeline()
        logger.info("✓ Pipeline initialized")
        return pipeline
    except Exception as e:
        logger.error(f"❌ Pipeline initialization failed: {e}")
        raise

@st.cache_resource
def initialize_session_manager():
    """Initialize session manager once."""
    try:
        logger.info("Initializing session manager...")
        manager = SessionManager()
        logger.info("✓ Session manager initialized")
        return manager
    except Exception as e:
        logger.error(f"❌ Session manager initialization failed: {e}")
        raise


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_example_session(example_name: str) -> dict:
    """
    Load example session data from examples/inputs/.
    
    Args:
        example_name: Name of example (e.g., "example1")
    
    Returns:
        Input data dictionary
    """
    example_path = Path("examples/inputs") / f"{example_name}.json"
    
    if not example_path.exists():
        st.error(f"Example file not found: {example_path}")
        return None
    
    with open(example_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def process_session(
    audio_file,
    notes_text: str,
    equations: str,
    external_context: str,
    session_id: str,
    verifiable_mode: bool = False,
    domain_profile: str = "algorithms",
    llm_provider_type: str = "openai",
    progress_callback: Optional[Callable[[str, str], None]] = None,
    debug_mode: bool = False,
    urls: Optional[List[str]] = None
) -> Tuple[dict, Optional[dict]]:
    """
    Process classroom session through the complete pipeline.
    
    Args:
        audio_file: Uploaded audio file object (or None)
        notes_text: Handwritten notes text
        equations: Equations string (one per line)
        external_context: Reference material text
        session_id: Unique session identifier
        verifiable_mode: Whether to use verifiable mode
        domain_profile: Domain profile (algorithms)
        llm_provider_type: LLM provider to use
        progress_callback: Optional progress callback for verifiable stages
        debug_mode: Whether to show debug details
        urls: List of URLs to fetch as evidence sources (optional)
    
    Returns:
        Tuple of (result_dict, verifiable_metadata)
    """
    # Step 1: Audio Transcription
    with st.spinner("🎤 Transcribing audio..."):
        if audio_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_file.read())
                tmp_path = tmp_file.name
            
            try:
                transcription = transcribe_audio(tmp_path)
            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)
        else:
            # No audio provided; skip transcription
            transcription = {"transcript": ""}
        
        transcript = transcription["transcript"]
        if transcript:
            st.success(f"✓ Transcription complete: {len(transcript)} characters")
    
    # Step 1.5: URL Content Extraction
    url_extracted_text = ""
    if urls:
        with st.spinner("🌐 Fetching URL content..."):
            try:
                for url in urls:
                    try:
                        content, metadata = fetch_url_text(url)
                        if content:
                            source_type = metadata.get("source_type", "unknown")
                            st.success(f"✓ {source_type.title()}: {metadata.get('words', 0)} words extracted from {url[:50]}...")
                            url_extracted_text += content + "\n\n"
                        else:
                            error_msg = metadata.get("error", "No content extracted")
                            st.warning(f"⚠️ Could not extract content from {url}: {error_msg}")
                    except Exception as e:
                        st.warning(f"⚠️ Error fetching {url}: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error during URL ingestion: {str(e)}")
                logger.error(f"URL ingestion error: {e}")
    
    # Combine notes with URL-extracted content
    if url_extracted_text:
        if notes_text:
            notes_text = notes_text + "\n\n---\n\nContent from URLs:\n\n" + url_extracted_text
        else:
            notes_text = url_extracted_text
    
    # Step 2: Preprocessing
    with st.spinner("🔄 Preprocessing content..."):
        equations_list = [eq.strip() for eq in equations.split('\n') if eq.strip()]
        
        preprocessed = preprocess_classroom_content(
            handwritten_notes=notes_text,
            transcript=transcript,
            equations=equations_list
        )
        
        combined_text = preprocessed["combined_text"]
        st.success(f"✓ Preprocessing complete: {preprocessed['metadata']['num_segments']} segments")
    
    # Step 3: Multi-Stage Reasoning
    with st.spinner("🧠 Running multi-stage reasoning pipeline..."):
        def _build_fallback_output(error_text: str):
            fallback = FallbackGenerator()
            base_topics = fallback.generate_default_topics(combined_text)
            fallback_output = {
                "summary": (combined_text[:500] + "...") if combined_text else "Summary not available.",
                "topics": base_topics,
                "concepts": fallback.generate_default_concepts(base_topics),
                "equations": [],
                "misconceptions": [
                    {
                        "misconception": "The content is just memorization",
                        "explanation": "Understanding relationships and applications is more important",
                        "correct_understanding": "Focus on understanding concepts, not rote memorization",
                        "related_concepts": []
                    }
                ],
                "faqs": fallback.generate_default_faqs(combined_text),
                "worked_examples": fallback.generate_default_examples(combined_text)
            }

            fallback_output = PipelineEnhancer.ensure_minimum_output(
                fallback_output,
                combined_text,
                fallback
            )

            # Ensure summary meets minimum length requirement (50 chars)
            summary = fallback_output["summary"]
            if len(summary.strip()) < 50:
                summary = "Study guide generated from available content. " + summary

            output = ClassSessionOutput(
                session_id=session_id,
                class_summary=summary,
                topics=[Topic(name=t.get("title", "Topic"), summary=t.get("description", "")) for t in fallback_output["topics"]],
                key_concepts=[Concept(name=c.get("name", "Concept"), definition=c.get("definition", "")) for c in fallback_output["concepts"]],
                equation_explanations=[EquationExplanation(equation=e.get("equation", ""), explanation=e.get("explanation", "")) for e in fallback_output.get("equations", [])],
                worked_examples=[WorkedExample(problem=e.get("problem", ""), solution=e.get("solution", "")) for e in fallback_output.get("worked_examples", [])],
                common_mistakes=[Misconception(misconception=m.get("misconception", ""), explanation=m.get("explanation", ""), correct_understanding=m.get("correct_understanding", "")) for m in fallback_output.get("misconceptions", [])],
                faqs=[FAQ(question=f.get("question", ""), answer=f.get("answer", ""), difficulty=f.get("difficulty", "medium")) for f in fallback_output.get("faqs", [])],
                real_world_connections=[RealWorldConnection(connection=r.get("connection", ""), relevance=r.get("relevance", "")) for r in fallback_output.get("connections", [])],
                metadata={"fallback": True, "error": error_text}
            )
            return output, None

        if llm_provider_type == "openai" and not config.OPENAI_API_KEY:
            st.error("❌ OPENAI_API_KEY is not set. Please add it to your .env file.")
            st.warning("Continuing with fallback output so the app remains responsive.")
            llm_provider_type = "fallback"

        if llm_provider_type == "ollama":
            try:
                requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=2)
            except Exception:
                st.warning("Ollama is not running. Start it or switch to OpenAI. Using fallback output.")
                output, verifiable_metadata = _build_fallback_output("ollama_unavailable")
                llm_provider_type = "fallback"

        selected_model = config.OLLAMA_MODEL if llm_provider_type == "ollama" else config.LLM_MODEL
        pipeline = VerifiablePipelineWrapper(
            provider_type=llm_provider_type,
            api_key=config.OPENAI_API_KEY,
            ollama_url=config.OLLAMA_URL,
            model=selected_model,
            domain_profile=domain_profile if verifiable_mode else None
        )
        
        if debug_mode:
            st.info(f"Filters being sent to pipeline: {st.session_state.output_filters}")
        
        # Parse URLs from text area
        urls = []
        if 'urls_text' in locals() and urls_text and urls_text.strip():
            urls = [url.strip() for url in urls_text.strip().split('\n') if url.strip()]
            if urls and debug_mode:
                st.info(f"Will ingest {len(urls)} URL(s) as evidence sources")
        
        if llm_provider_type != "fallback":
            try:
                output, verifiable_metadata = pipeline.process(
                    combined_content=combined_text,
                    equations=equations_list,
                    external_context=external_context,
                    session_id=session_id,
                    verifiable_mode=verifiable_mode,
                    output_filters=st.session_state.output_filters,
                    urls=urls,
                    progress_callback=progress_callback
                )
            except Exception as e:
                error_text = str(e)
                
                # Handle evidence store validation failures
                if isinstance(e, RuntimeError) and "Evidence store validation failed" in error_text:
                    st.error(f"❌ Cannot verify content: {error_text[:150]}...")
                    output = ClassSessionOutput(
                        session_id=session_id,
                        class_summary="Content provided was insufficient for analysis. Please provide at least 500 characters of substantive content or valid lecture notes.",
                        topics=[],
                        key_concepts=[],
                        equation_explanations=[],
                        worked_examples=[],
                        common_mistakes=[],
                        faqs=[],
                        real_world_connections=[],
                        metadata={"fallback": True, "error": "insufficient_input"}
                    )
                    verifiable_metadata = None
                # Handle quota errors
                elif "insufficient_quota" in error_text or "RateLimitError" in error_text:
                    st.error(
                        "❌ OpenAI quota exceeded. Please check your plan/billing, "
                        "or switch to Local LLM (Ollama) in the sidebar."
                    )
                    output, verifiable_metadata = _build_fallback_output(error_text)
                # Handle Ollama connection errors
                elif "WinError 10061" in error_text or "localhost" in error_text:
                    st.warning("Ollama is not running. Start it or switch to OpenAI. Using fallback output.")
                    output, verifiable_metadata = _build_fallback_output(error_text)
                # Generic error handling
                else:
                    st.error(f"❌ LLM call failed: {e}")
                    output, verifiable_metadata = _build_fallback_output(error_text)
        
        if verifiable_mode and verifiable_metadata:
            st.success("Verifiability report ready.")
            if debug_mode:
                timings = verifiable_metadata.get("timings", {})
                total_time = verifiable_metadata.get("total_time_seconds")
                if timings or total_time is not None:
                    with st.expander("Processing Time Breakdown", expanded=False):
                        if total_time is not None:
                            st.metric("Total Time (s)", f"{total_time:.2f}")
                        if timings:
                            for step, seconds in timings.items():
                                st.write(f"- {step.replace('_', ' ')}: {seconds:.2f}s")
        else:
            st.success(
                f"✓ Reasoning complete: {len(output.topics)} topics, "
                f"{len(output.key_concepts)} concepts extracted"
            )
    
    # Step 4: Evaluation
    with st.spinner("📊 Evaluating quality..."):
        evaluation = evaluate_session_output(
            output=output,
            source_content=combined_text,
            external_context=external_context
        )
        
        st.success(f"✓ Evaluation complete")
    
    # Step 5: Save Session
    with st.spinner("💾 Saving session..."):
        session_manager = SessionManager()
        session_path = session_manager.save_session(output, overwrite=True)
        st.success(f"✓ Session saved")
    
    result = {
        "output": output,
        "evaluation": evaluation,
        "session_path": session_path,
        "transcript_length": len(transcript),
        "verifiable_mode": verifiable_mode
    }
    
    return result, verifiable_metadata


def _ocr_cache_path() -> Path:
    cache_dir = Path("cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "ocr_cache.json"


def _load_ocr_cache() -> dict:
    cache_path = _ocr_cache_path()
    if not cache_path.exists():
        return {"order": [], "items": {}}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"order": [], "items": {}}


def _save_ocr_cache(cache: dict) -> None:
    cache_path = _ocr_cache_path()
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _image_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def _extract_text_from_pdf(pdf_file, ocr=None) -> Tuple[str, Dict[str, Any]]:
    """
    Extract text from a PDF file with multi-strategy fallback.
    
    Uses new robust extraction module with OCR fallback:
    1. pdfplumber extraction
    2. OCR (PyMuPDF rendering + Tesseract/EasyOCR fallback) for scanned PDFs
    
    Args:
        pdf_file: Streamlit uploaded file object
        ocr: Optional ImageOCR instance for OCR fallback
        
    Returns:
        Tuple of (extracted_text, metadata_dict)
        
    Raises:
        EvidenceIngestError: If extraction fails and OCR is unavailable
    """
    from src.exceptions import EvidenceIngestError
    
    try:
        # Use new robust extraction module with OCR fallback
        text, metadata = extract_pdf_text(pdf_file, ocr=ocr)
        
        if not text.strip():
            logger.warning(f"PDF {pdf_file.name} extracted but no quality text found")
            return "", metadata
        
        # Format with page information if available
        pages = metadata.get("pages", metadata.get("num_pages", 1))
        extraction_method = metadata.get("extraction_method") or metadata.get("extraction_method_used", "unknown")
        ocr_pages = metadata.get("ocr_pages", 0)
        logger.info(
            "PDF ingestion diagnostics: file=%s method=%s chars=%s ocr_pages=%s",
            pdf_file.name,
            extraction_method,
            metadata.get("chars_extracted", 0),
            ocr_pages
        )
        formatted_text = (
            f"--- PDF: {pdf_file.name} ({pages} pages, method: {extraction_method}, "
            f"ocr_pages: {ocr_pages}) ---\n{text}\n"
        )
        
        return formatted_text, metadata
    except EvidenceIngestError as e:
        # Re-raise EvidenceIngestError for proper handling upstream
        logger.error(f"Evidence ingestion failed for {pdf_file.name}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error extracting PDF {pdf_file.name}: {str(e)}")
        return "", {"error": str(e), "extraction_method": "error"}


def display_ingestion_diagnostics(diagnostics: Dict[str, Any], show_minimal: bool = False):
    """
    Display ingestion diagnostics in UI for debugging and clarity.
    
    Args:
        diagnostics: Diagnostics dictionary from get_ingestion_diagnostics()
        show_minimal: If True, show only critical issues; else show full details
    """
    if diagnostics.get("is_valid"):
        st.success(
            f"✅ **Extraction Successful**: {diagnostics['extracted_text_length']:,} characters from "
            f"{diagnostics['sources_count']} source(s)"
        )
        return
    
    # Display detailed ingestion failure
    error_msg = diagnostics.get("error_message", "Unknown ingestion error")
    
    st.error(f"❌ {error_msg}")
    
    # Show diagnostics details
    with st.expander("📋 Ingestion Diagnostics", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Text Extracted",
                f"{diagnostics['extracted_text_length']:,} chars",
                help="Total characters successfully extracted"
            )
        
        with col2:
            st.metric(
                "Minimum Required",
                f"{diagnostics['minimum_required']:,} chars",
                help=f"Required for verification mode"
            )
        
        with col3:
            shortfall = diagnostics['minimum_required'] - diagnostics['extracted_text_length']
            if shortfall > 0:
                st.metric(
                    "Shortfall",
                    f"{shortfall:,} chars",
                    delta=f"-{shortfall:,}",
                    delta_color="inverse",
                    help="Additional chars needed"
                )
            else:
                st.metric(
                    "Status",
                    "✓ Ready",
                    delta="sufficient",
                    help="Meets requirements"
                )
        
        st.write("**Source Breakdown:**")
        if diagnostics['source_breakdown']:
            sources_str = ", ".join(
                f"{count} {source_type}" 
                for source_type, count in diagnostics['source_breakdown'].items()
            )
            st.info(sources_str)
        else:
            st.warning("No sources detected")
        
        st.write("**Suggested Actions:**")
        for i, suggestion in enumerate(diagnostics['suggestions'], 1):
            st.write(f"{i}. {suggestion}")
    
    # Offering alternatives
    st.markdown("""
    #### 💡 Try These Next:
    - **OCR Mode**: If you have a scanned PDF or image, ensure OCR is available
    - **Text Input**: Paste the content directly using "Type/Paste Text" mode  
    - **Multiple Files**: Upload additional materials to meet the character threshold
    - **Fast Mode**: Use the "Generate Study Guide (Fast)" for quick analysis without verification
    """)


def _create_verifiability_progress_ui():
    stages = [
        ("claim_extraction", "Claim extraction"),
        ("retrieval", "Evidence retrieval"),
        ("nli", "NLI consistency"),
        ("decision_policy", "Decision policy")
    ]
    stage_labels = {key: label for key, label in stages}
    stage_state = {key: "pending" for key, _ in stages}

    progress_bar = st.progress(0, text="Preparing verifiability report...")
    container = st.container()
    placeholders = {}

    with container:
        for key, label in stages:
            placeholders[key] = st.empty()
            placeholders[key].markdown(f"○ {label}")

    def update_progress(stage_key: str, status: str) -> None:
        if stage_key not in placeholders:
            return

        stage_state[stage_key] = status
        icon = "○"
        if status == "running":
            icon = "⏳"
        elif status == "complete":
            icon = "✅"

        placeholders[stage_key].markdown(f"{icon} {stage_labels[stage_key]}")
        completed = sum(1 for value in stage_state.values() if value == "complete")
        progress_bar.progress(completed / len(stages))

    return update_progress


def _build_verifiability_report(verifiable_metadata: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    claim_collection = verifiable_metadata.get("claim_collection")
    claims = claim_collection.claims if claim_collection else []
    metrics = verifiable_metadata.get("metrics", {})

    report_claims = []
    for claim in claims:
        status_value = getattr(claim.status, "value", str(claim.status))
        claim_type = getattr(claim.claim_type, "value", str(claim.claim_type))
        rejection_reason = getattr(claim.rejection_reason, "value", str(claim.rejection_reason))

        display_text = (
            claim.metadata.get("ui_display", "") or
            claim.claim_text or
            claim.metadata.get("draft_text", "")
        )

        evidence_items = []
        for ev in getattr(claim, "evidence_objects", []) or []:
            span_meta = getattr(ev, "span_metadata", {}) or {}
            location = span_meta.get("location") or span_meta.get("line") or span_meta.get("page")
            evidence_items.append({
                "source_id": getattr(ev, "source_id", ""),
                "source_type": getattr(ev, "source_type", ""),
                "snippet": (getattr(ev, "snippet", "") or getattr(ev, "text", ""))[:500],
                "similarity": getattr(ev, "similarity", None),
                "location": location
            })

        report_claims.append({
            "claim_id": claim.claim_id,
            "claim_type": claim_type,
            "status": status_value,
            "confidence": claim.confidence,
            "evidence_ids": list(getattr(claim, "evidence_ids", []) or []),
            "rejection_reason": rejection_reason if rejection_reason != "None" else None,
            "claim_text": claim.claim_text or display_text,
            "evidence": evidence_items
        })

    report = {
        "session_id": session_id,
        "generated_at": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "claims": report_claims,
        "claim_count": len(report_claims)
    }

    return report


def _extract_claims_for_research_report(verifiable_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract claim dicts for reporting from verifiable metadata.
    
    Ensures evidence includes:
    - source_id: which input provided the evidence
    - source_type: pdf, audio, text, url, etc.
    - page_num: page number if from PDF
    - span_id: unique identifier for location
    - snippet: text content
    """
    claims_data = verifiable_metadata.get("claims") or []
    if claims_data:
        return claims_data

    claim_collection = verifiable_metadata.get("claim_collection")
    if not claim_collection:
        return []

    report_claims = []
    for claim in claim_collection.claims:
        status_value = getattr(claim.status, "value", str(claim.status))
        display_text = (
            claim.metadata.get("ui_display", "") or
            claim.claim_text or
            claim.metadata.get("draft_text", "")
        )
        
        # Extract claim_type from claim object
        claim_type = getattr(claim, "claim_type", None)
        if claim_type:
            claim_type = claim_type.value if hasattr(claim_type, "value") else str(claim_type)
        else:
            claim_type = "fact_claim"

        # Extract page_num and span_id from first evidence for report metadata
        page_num = None
        span_id = None
        
        evidence_items = []
        for ev in getattr(claim, "evidence_objects", []) or []:
            span_meta = getattr(ev, "span_metadata", {}) or {}
            
            # Extract page number if available
            page = span_meta.get("page") or span_meta.get("page_num")
            if page and page_num is None:
                page_num = page
            
            # Extract span_id from evidence
            ev_span_id = getattr(ev, "span_id", None)
            if ev_span_id and span_id is None:
                span_id = ev_span_id
            
            location = span_meta.get("location") or span_meta.get("line") or page
            
            evidence_items.append({
                "source_id": getattr(ev, "source_id", ""),
                "source_type": getattr(ev, "source_type", ""),
                "page_num": page,
                "snippet": (getattr(ev, "snippet", "") or getattr(ev, "text", ""))[:500],
                "similarity": getattr(ev, "similarity", None),
                "location": location,
                "span_id": ev_span_id,
            })

        report_claims.append({
            "claim_text": claim.claim_text or display_text,
            "status": status_value,
            "confidence": claim.confidence,
            "evidence": evidence_items,
            "page_num": page_num,
            "span_id": span_id,
            "claim_type": claim_type,
            "rejection_reason": getattr(claim, "rejection_reason", None),
        })

    return report_claims


def _compute_verification_stats(
    claims_data: List[Dict[str, Any]],
    metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute verification stats strictly from claim data.
    
    Returns:
        Dict with:
        - total_claims: Total number of claims
        - verified: Number verified
        - rejected: Number rejected
        - low_conf: Number low confidence
        - avg_conf: Average confidence across all claims
        - rejection_reasons: Dict[reason_str, count] for rejected claims
    """
    total_claims = len(claims_data)
    verified_count = sum(1 for c in claims_data if str(c.get("status", "")).upper() == "VERIFIED")
    low_conf_count = sum(1 for c in claims_data if str(c.get("status", "")).upper() == "LOW_CONFIDENCE")
    rejected_count = sum(1 for c in claims_data if str(c.get("status", "")).upper() == "REJECTED")

    avg_conf = None
    if total_claims > 0:
        avg_conf = metrics.get("avg_confidence")
        if not isinstance(avg_conf, (int, float)):
            confidences = [c.get("confidence", 0.0) for c in claims_data]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    # Compute rejection_reasons from rejected claims
    rejection_reasons = {}
    for c in claims_data:
        if str(c.get("status", "")).upper() == "REJECTED":
            reason = c.get("rejection_reason", "UNKNOWN_REASON") or "UNKNOWN_REASON"
            reason_str = reason if isinstance(reason, str) else str(reason)
            rejection_reasons[reason_str] = rejection_reasons.get(reason_str, 0) + 1

    return {
        "total_claims": total_claims,
        "verified": verified_count,
        "rejected": rejected_count,
        "low_conf": low_conf_count,
        "avg_conf": avg_conf,
        "rejection_reasons": rejection_reasons if rejection_reasons else None,
    }


def _summarize_ingestion_pages(ingestion_diagnostics: Dict[str, Any]) -> Dict[str, int]:
    """Summarize page counts from ingestion diagnostics."""
    pdf_files = ingestion_diagnostics.get("pdf_files", []) if ingestion_diagnostics else []
    pages = 0
    ocr_pages = 0
    for pdf in pdf_files:
        pages += int(pdf.get("pages_total") or 0)
        ocr_pages += int(pdf.get("ocr_pages") or 0)

    return {
        "pages": pages,
        "ocr_pages": ocr_pages,
    }


def _resolve_inputs_used(
    ingestion_payload: Optional[Dict[str, Any]],
    ingestion_diagnostics: Dict[str, Any]
) -> List[str]:
    """Resolve input types used for a run."""
    present = set()
    if ingestion_payload:
        if ingestion_payload.get("notes_text"):
            present.add("text")
        if ingestion_payload.get("urls_text"):
            present.add("urls")
        if ingestion_payload.get("audio_present"):
            present.add("audio")

    if ingestion_diagnostics and ingestion_diagnostics.get("pdf_files"):
        present.add("pdf")

    order = ["pdf", "audio", "text", "urls"]
    return [item for item in order if item in present]


def _load_artifact_text(path_value: Optional[str]) -> Optional[str]:
    """Load a text artifact from disk if available."""
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _build_verifiability_report_pdf(report: Dict[str, Any]) -> bytes:
    try:
        import fitz  # PyMuPDF
    except Exception as exc:
        raise RuntimeError("PyMuPDF is required to export PDF reports.") from exc

    def wrap_lines(text: str, width: int = 90) -> list[str]:
        return textwrap.wrap(text, width=width) if text else []

    def add_line(page_obj, line: str, y_pos: float) -> float:
        page_obj.insert_text((72, y_pos), line, fontsize=11)
        return y_pos + 14

    doc = fitz.open()
    page = doc.new_page()
    y = 72

    def ensure_space(lines_needed: int = 1):
        nonlocal page, y
        if y + (lines_needed * 14) > page.rect.height - 72:
            page = doc.new_page()
            y = 72

    title = f"Verifiability Report: {report.get('session_id', 'session')}"
    for line in wrap_lines(title, 80):
        ensure_space()
        y = add_line(page, line, y)
    ensure_space()
    y = add_line(page, f"Generated: {report.get('generated_at', '')}", y)
    ensure_space()
    y = add_line(page, "", y)

    metrics = report.get("metrics", {})
    summary_lines = [
        f"Total claims: {metrics.get('total_claims', report.get('claim_count', 0))}",
        f"Verified: {metrics.get('verified_claims', 0)}",
        f"Low confidence: {metrics.get('low_confidence_claims', 0)}",
        f"Rejected: {metrics.get('rejected_claims', 0)}",
        f"Rejection rate: {metrics.get('rejection_rate', 0):.1%}",
        f"Avg confidence: {metrics.get('avg_confidence', 0):.2f}",
    ]
    for line in summary_lines:
        ensure_space()
        y = add_line(page, line, y)

    y = add_line(page, "", y)

    def render_claim_table(title_text: str, claims: list[Dict[str, Any]]):
        nonlocal page, y
        ensure_space(2)
        y = add_line(page, title_text, y)
        if not claims:
            y = add_line(page, "(none)", y)
            return

        for claim in claims:
            claim_text = (claim.get("claim_text") or claim.get("text") or "").strip()
            claim_text = claim_text if len(claim_text) <= 140 else claim_text[:137] + "..."
            evidence = claim.get("evidence", [])
            top_evidence = evidence[0].get("snippet", "") if evidence else ""
            top_evidence = top_evidence if len(top_evidence) <= 160 else top_evidence[:157] + "..."
            line = (
                f"- {claim_text} (conf: {claim.get('confidence', 0):.2f}, "
                f"evidence: {len(evidence)})"
            )
            for wrapped in wrap_lines(line, 92):
                ensure_space()
                y = add_line(page, wrapped, y)
            if top_evidence:
                for wrapped in wrap_lines(f"  Top evidence: {top_evidence}", 92):
                    ensure_space()
                    y = add_line(page, wrapped, y)
        y = add_line(page, "", y)

    claims = report.get("claims", [])
    verified = [c for c in claims if str(c.get("status", "")).lower() == "verified"]
    low_conf = [c for c in claims if str(c.get("status", "")).lower() == "low_confidence"]
    rejected = [c for c in claims if str(c.get("status", "")).lower() == "rejected"]

    render_claim_table("Verified claims", verified)
    render_claim_table("Low-confidence claims", low_conf)
    render_claim_table("Rejected claims", rejected)

    return doc.tobytes()


def display_verifiability_report(
    verifiable_metadata: Dict[str, Any],
    output: Any,
    debug_mode: bool = False
) -> None:
    session_id = output.session_id if hasattr(output, "session_id") else "session"
    report = _build_verifiability_report(verifiable_metadata, session_id)
    metrics = report.get("metrics", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Verified", metrics.get("verified_claims", 0))
    with col2:
        st.metric("Low confidence", metrics.get("low_confidence_claims", 0))
    with col3:
        st.metric("Rejected", metrics.get("rejected_claims", 0))
    with col4:
        st.metric("Avg confidence", f"{metrics.get('avg_confidence', 0):.2f}")

    claims = report.get("claims", [])
    verified = [c for c in claims if str(c.get("status", "")).lower() == "verified"]
    low_conf = [c for c in claims if str(c.get("status", "")).lower() == "low_confidence"]
    rejected = [c for c in claims if str(c.get("status", "")).lower() == "rejected"]

    def _render_claim_table(title: str, rows: list[Dict[str, Any]]):
        st.subheader(title)
        if not rows:
            st.caption("No claims in this category.")
            return
        table_rows = []
        for claim in rows:
            evidence = claim.get("evidence", [])
            top_evidence = evidence[0].get("snippet", "") if evidence else ""
            table_rows.append({
                "Claim": (
                    claim.get("claim_text", "")[:140] + "..."
                    if len(claim.get("claim_text", "")) > 140
                    else claim.get("claim_text", "")
                ),
                "Confidence": f"{claim.get('confidence', 0):.2f}",
                "Evidence": len(evidence),
                "Top evidence": (top_evidence[:160] + "...") if len(top_evidence) > 160 else top_evidence
            })
        try:
            st.dataframe(table_rows, use_container_width=True)
        except Exception:
            st.table(table_rows)

    _render_claim_table("Verified claims", verified)
    _render_claim_table("Low-confidence claims", low_conf)
    _render_claim_table("Rejected claims", rejected)

    st.divider()
    report_json = export_report_json(report)
    pdf_bytes = None
    pdf_error = None
    try:
        pdf_bytes = export_report_pdf(report)
    except Exception as exc:
        pdf_error = str(exc)

    export_col1, export_col2 = st.columns(2)
    with export_col1:
        st.download_button(
            label="Export report JSON",
            data=report_json,
            file_name=f"verifiability_report_{session_id}.json",
            mime="application/json"
        )
    with export_col2:
        if pdf_bytes:
            st.download_button(
                label="Export report PDF",
                data=pdf_bytes,
                file_name=f"verifiability_report_{session_id}.pdf",
                mime="application/pdf"
            )
        else:
            st.caption(f"PDF export unavailable: {pdf_error}")

    if debug_mode:
        st.divider()
        st.caption("Debug: full report JSON")
        st.json(report)


def _display_research_reports(
    verifiable_metadata: Dict[str, Any],
    output: Any
) -> None:
    """
    Display research reports in multiple formats (MD, HTML, JSON).
    
    Args:
        verifiable_metadata: Verifiable mode metadata with verification results
        output: ClassSessionOutput object with session information
    """
    try:
        # Extract session metadata
        session_id = output.session_id if hasattr(output, "session_id") else "session"
        
        session_metadata = SessionMetadata(
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            version=config.__version__ if hasattr(config, '__version__') else "1.0.0",
            seed=getattr(config, 'GLOBAL_RANDOM_SEED', 42),
            language_model=getattr(config, 'LLM_MODEL_NAME', "gpt-4"),
            embedding_model=getattr(config, 'EMBEDDING_MODEL_NAME', "text-embedding-ada-002"),
            nli_model=getattr(config, 'NLI_MODEL_NAME', "cross-encoder/qnli"),
            inputs_used=st.session_state.get("input_sources", []),
        )
        
        # Extract ingestion report from actual ingestion diagnostics
        ingestion_diagnostics = {}
        if hasattr(st.session_state, "ingestion_diagnostics") and st.session_state.ingestion_diagnostics:
            ingestion_diagnostics = st.session_state.ingestion_diagnostics
        else:
            ingestion_diagnostics = verifiable_metadata.get("ingestion_diagnostics", {})
        ingestion_pages = _summarize_ingestion_pages(ingestion_diagnostics)
        ingestion_report = IngestionReport(
            total_pages=ingestion_diagnostics.get("total_pages", ingestion_pages["pages"]),
            pages_ocr=ingestion_diagnostics.get("pages_ocr", ingestion_pages["ocr_pages"]),
            headers_removed=ingestion_diagnostics.get("headers_removed", 0),
            footers_removed=ingestion_diagnostics.get("footers_removed", 0),
            watermarks_removed=ingestion_diagnostics.get("watermarks_removed", 0),
            total_chunks=ingestion_diagnostics.get("total_chunks", 0),
            avg_chunk_size=ingestion_diagnostics.get("avg_chunk_size", 512),
            extraction_methods=ingestion_diagnostics.get("extraction_methods", []),
        )
        
        # Extract verification summary and metrics from actual results
        metrics = verifiable_metadata.get("metrics", {})
        claims_data = _extract_claims_for_research_report(verifiable_metadata)

        stats = _compute_verification_stats(claims_data, metrics)
        total_claims = stats["total_claims"]
        verified_count = stats["verified"]
        low_conf_count = stats["low_conf"]
        rejected_count = stats["rejected"]
        
        # Invariant check: all claims must be accounted for
        count_sum = verified_count + low_conf_count + rejected_count
        if total_claims > 0 and count_sum != total_claims:
            logger.warning(
                f"Claim count mismatch: total={total_claims}, "
                f"verified={verified_count}, low_conf={low_conf_count}, rejected={rejected_count} "
                f"(sum={count_sum})"
            )
        
        # Only compute avg_confidence if there are claims
        if total_claims > 0:
            avg_conf = stats["avg_conf"]
        else:
            avg_conf = 0.0
        
        # Only include rejection reasons if there are actually rejected claims
        rejection_reasons = stats.get("rejection_reasons", {})
        if total_claims > 0 and rejected_count > 0:
            top_reasons = [(reason, count) for reason, count in sorted(
                rejection_reasons.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]]  # Top 5 reasons
        else:
            top_reasons = []  # No reasons if zero claims or zero rejected
        
        verification_summary = VerificationSummary(
            total_claims=total_claims,
            verified_count=verified_count,
            low_confidence_count=low_conf_count,
            rejected_count=rejected_count,
            avg_confidence=float(avg_conf),
            top_rejection_reasons=top_reasons,
            rejection_reasons_dict=rejection_reasons if rejection_reasons else None,
            calibration_metrics=metrics.get("calibration_metrics"),
        )
        
        # Build claims entries
        claims_entries = []
        for claim in claims_data:
            evidence = claim.get("evidence", [])
            top_evidence = evidence[0].get("snippet", "") if evidence else ""

            entry = ClaimEntry(
                claim_text=claim.get("claim_text", ""),
                status=claim.get("status", "UNKNOWN"),
                confidence=float(claim.get("confidence", 0)),
                evidence_count=len(evidence),
                top_evidence=top_evidence,
                page_num=claim.get("page_num"),
                span_id=claim.get("span_id"),
                claim_type=claim.get("claim_type", "fact_claim"),
            )
            claims_entries.append(entry)
        
        # Build the report
        md_content, html_content, audit_json = build_report(
            session_metadata,
            ingestion_report,
            verification_summary,
            claims_entries,
            performance_metrics=metrics.get("performance", {}),
        )

        # Persist report artifacts and run history
        evidence_stats = verifiable_metadata.get("evidence_stats", {})
        run_id = st.session_state.get("current_run_id")
        if not run_id:
            run_id = evidence_stats.get("artifact_run_id") or generate_run_id(session_id)
            st.session_state.current_run_id = run_id

        run_dir = config.ARTIFACTS_DIR / session_id / run_id
        report_md_path, report_html_path, audit_json_path = save_reports(
            run_dir,
            md_content,
            html_content,
            audit_json,
            prefix="research_report",
        )

        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        graphml_path = None
        claim_graph = verifiable_metadata.get("claim_graph")
        if claim_graph and hasattr(claim_graph, "export_graphml"):
            candidate_path = run_dir / "claim_graph.graphml"
            if claim_graph.export_graphml(str(candidate_path)):
                graphml_path = candidate_path

        ingestion_pages = _summarize_ingestion_pages(ingestion_diagnostics)
        inputs_used = _resolve_inputs_used(
            st.session_state.get("ingestion_payload"),
            ingestion_diagnostics,
        )

        run_summary = {
            "run_id": run_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "domain_profile": verifiable_metadata.get("domain_profile_display")
            or verifiable_metadata.get("domain_profile")
            or "unknown",
            "llm_model": getattr(config, "LLM_MODEL", "unknown"),
            "embedding_model": getattr(config, "EMBEDDING_MODEL_NAME", "unknown"),
            "nli_model": getattr(config, "NLI_MODEL_NAME", "unknown"),
            "inputs_used": inputs_used,
            "ingestion_stats": ingestion_pages,
            "verification_stats": {
                "total_claims": stats["total_claims"],
                "verified": stats["verified"],
                "rejected": stats["rejected"],
                "low_conf": stats["low_conf"],
                "avg_conf": stats["avg_conf"],
            },
            "artifact_paths": {
                "report_md": str(report_md_path),
                "report_html": str(report_html_path),
                "audit_json": str(audit_json_path),
                "metrics_json": str(metrics_path),
                "graphml": str(graphml_path) if graphml_path else None,
            },
        }

        global_index = config.ARTIFACTS_DIR / "run_history.json"
        session_index = config.ARTIFACTS_DIR / session_id / "run_history.json"

        for index_path in (global_index, session_index):
            history = load_run_history(index_path)
            history = append_run(history, run_summary, max_runs=3)
            save_run_history(index_path, history)

        st.session_state.latest_run_summary = run_summary
        
        # Display report options
        st.subheader("📄 Research Reports")
        st.caption("Download comprehensive session reports in multiple formats")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="⬇️ Markdown",
                data=md_content,
                file_name=f"research_report_{session_id}.md",
                mime="text/markdown",
                help="Human-readable markdown version"
            )
        
        with col2:
            st.download_button(
                label="⬇️ HTML",
                data=html_content,
                file_name=f"research_report_{session_id}.html",
                mime="text/html",
                help="Styled HTML version for browser viewing"
            )
        
        with col3:
            audit_json_str = json.dumps(audit_json, indent=2)
            st.download_button(
                label="⬇️ Audit JSON",
                data=audit_json_str,
                file_name=f"research_report_audit_{session_id}.json",
                mime="application/json",
                help="Structured audit trail for reproducibility"
            )
        
        # Display preview
        st.divider()
        st.subheader("📋 Report Preview")
        
        preview_format = st.radio(
            "View preview in:",
            ["Markdown", "HTML"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if preview_format == "Markdown":
            st.markdown(md_content)
        else:
            st.components.v1.html(html_content, height=800, scrolling=True)
    
    except Exception as e:
        st.error(f"Failed to generate research reports: {e}")
        logger.exception("Error in _display_research_reports")


def display_output(result: dict, verifiable_metadata: Optional[Dict[str, Any]] = None):
    """
    Display structured output using new streaming display system.
    
    Args:
        result: Result dictionary from process_session
        verifiable_metadata: Optional verifiable mode metadata
    """
    output = result["output"]
    evaluation = result["evaluation"]
    
    # Initialize display system
    display = StreamlitProgressDisplay()
    display.setup_display()
    
    # ========================================================================
    # SMART OUTPUT DISPLAY (Student-Friendly)
    # ========================================================================
    
    output_dict = {
        "summary": output.summary if hasattr(output, 'summary') else "",
        "topics": [
            {
                "name": t.name,
                "summary": t.summary,
                "subtopics": t.subtopics if hasattr(t, 'subtopics') else []
            } for t in (output.topics or [])
        ],
        "concepts": [
            {
                "name": c.name,
                "definition": c.definition,
                "difficulty": c.difficulty_level if hasattr(c, 'difficulty_level') else 3
            } for c in (output.key_concepts or [])
        ],
        "equations": [
            {
                "equation": e.equation,
                "explanation": e.explanation
            } for e in (output.equation_explanations or [])
        ],
        "misconceptions": [
            {
                "misconception": m.misconception,
                "correct_understanding": m.correct_understanding
            } for m in (output.common_mistakes or [])
        ],
        "faqs": [
            {
                "question": f.question,
                "answer": f.answer,
                "difficulty": f.difficulty
            } for f in (output.faqs or [])
        ],
        "worked_examples": [
            {
                "problem": e.problem,
                "solution": e.solution,
                "key_concepts": e.key_concepts if hasattr(e, 'key_concepts') else []
            } for e in (output.worked_examples or [])
        ],
    }
    
    # Display results
    display.display_streaming_output(output_dict)
    
    # ========================================================================
    # EVALUATION METRICS
    # ========================================================================
    
    st.divider()
    st.header("📊 Quality Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Correctness",
            f"{evaluation.reasoning_correctness:.0%}",
            help="Reasoning accuracy"
        )
    
    with col2:
        st.metric(
            "Completeness",
            f"{evaluation.structural_accuracy:.0%}",
            help="Output completeness"
        )
    
    with col3:
        st.metric(
            "Accuracy",
            f"{(1-evaluation.hallucination_rate):.0%}",
            help="Fact accuracy"
        )
    
    with col4:
        st.metric(
            "Usefulness",
            f"{evaluation.educational_usefulness:.1f}/5",
            help="Educational value"
        )
    
    if evaluation.passes_thresholds():
        st.success("✅ **PASSES** quality thresholds")
    else:
        st.warning("⚠️ **FAILS** quality thresholds")
    
    st.divider()
    
    # ========================================================================
    # CLASS SUMMARY
    # ========================================================================
    
    if st.session_state.output_filters.get('summary', True):
        st.header("Summary")
        st.write(output.class_summary)
        st.divider()
    
    # ========================================================================
    # TOPICS
    # ========================================================================
    
    if st.session_state.output_filters.get('topics', True):
        st.header(f"Topics ({len(output.topics)})")
        
        if output.topics:
            for i, topic in enumerate(output.topics, 1):
                with st.expander(f"**{i}. {topic.name}**", expanded=(i == 1)):
                    st.write("**Summary:**", topic.summary)
                    
                    if topic.subtopics:
                        st.write("**Subtopics:**")
                        for subtopic in topic.subtopics:
                            st.write(f"  • {subtopic}")
                    
                    if topic.learning_objectives:
                        st.write("**Learning Objectives:**")
                        for obj in topic.learning_objectives:
                            st.write(f"  • {obj}")
        else:
            st.info("No topics extracted")
        
        st.divider()
    
    # ========================================================================
    # KEY CONCEPTS
    # ========================================================================
    
    if st.session_state.output_filters.get('concepts', True):
        st.header(f"Key Concepts ({len(output.key_concepts)})")
        
        if output.key_concepts:
            for i, concept in enumerate(output.key_concepts, 1):
                with st.expander(
                    f"**{i}. {concept.name}** (Difficulty: {concept.difficulty_level}/5)",
                    expanded=(i == 1)
                ):
                    st.write("**Definition:**", concept.definition)
                    
                    if concept.prerequisites:
                        st.write("**Prerequisites:**", ", ".join(concept.prerequisites))
        else:
            st.info("No concepts extracted")
        
        st.divider()
    
    # ========================================================================
    # EQUATION EXPLANATIONS
    # ========================================================================
    
    if st.session_state.output_filters.get('equations', True):
        if output.equation_explanations:
            st.header(f"Equations ({len(output.equation_explanations)})")
        
        for i, eq_exp in enumerate(output.equation_explanations, 1):
            with st.expander(f"**{i}. {eq_exp.equation}**", expanded=(i == 1)):
                st.write("**Plain-Language Explanation:**")
                st.write(eq_exp.explanation)
                
                if eq_exp.variables:
                    st.write("**Variables:**")
                    for var, desc in eq_exp.variables.items():
                        st.write(f"  • **{var}**: {desc}")
                
                if eq_exp.applications:
                    st.write("**Applications:**")
                    for app in eq_exp.applications:
                        st.write(f"  • {app}")
        
        st.divider()
    
    # ========================================================================
    # WORKED EXAMPLES
    # ========================================================================
    
    if st.session_state.output_filters.get('worked_examples', True):
        st.header(f"Worked Examples ({len(output.worked_examples)})")
        
        if output.worked_examples:
            for i, example in enumerate(output.worked_examples, 1):
                with st.expander(f"**Example {i}**", expanded=(i == 1)):
                    st.write("**Problem:**")
                    st.write(example.problem)
                    
                    st.write("**Solution:**")
                    st.write(example.solution)
                    
                    if example.key_concepts:
                        st.write("**Concepts Used:**", ", ".join(example.key_concepts))
                    
                    if example.common_mistakes:
                        st.write("**Common Mistakes to Avoid:**")
                        for mistake in example.common_mistakes:
                            st.write(f"  ⚠️ {mistake}")
        else:
            st.info("No worked examples extracted")
        
        st.divider()
    
    # ========================================================================
    # COMMON MISTAKES
    # ========================================================================
    
    if st.session_state.output_filters.get('misconceptions', True):
        st.header(f"Common Mistakes ({len(output.common_mistakes)})")
        
        if output.common_mistakes:
            for i, misc in enumerate(output.common_mistakes, 1):
                with st.expander(f"**Misconception {i}**", expanded=(i == 1)):
                    st.error(f"**Incorrect:** {misc.misconception}")
                    st.write(f"**Why it's wrong:** {misc.explanation}")
                    st.success(f"**Correct understanding:** {misc.correct_understanding}")
                    
                    if misc.related_concepts:
                        st.write("**Related concepts:**", ", ".join(misc.related_concepts))
        else:
            st.info("No misconceptions identified")
        
        st.divider()
    
    # ========================================================================
    # FAQs
    # ========================================================================
    
    if st.session_state.output_filters.get('faqs', True):
        st.header(f"FAQs ({len(output.faqs)})")
        
        if output.faqs:
            for i, faq in enumerate(output.faqs, 1):
                difficulty_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(
                    faq.difficulty, "⚪"
                )
                
                with st.expander(f"{difficulty_emoji} **Q{i}: {faq.question}**", expanded=(i == 1)):
                    st.write("**Answer:**", faq.answer)
                    
                    if faq.related_concepts:
                        st.write("**Related concepts:**", ", ".join(faq.related_concepts))
        else:
            st.info("No FAQs generated")
        
        st.divider()
    
    # ========================================================================
    # REAL-WORLD CONNECTIONS
    # ========================================================================
    
    if st.session_state.output_filters.get('real_world', True):
        st.header(f"Connections ({len(output.real_world_connections)})")
        
        if output.real_world_connections:
            for i, conn in enumerate(output.real_world_connections, 1):
                with st.expander(
                    f"**{i}. {conn.concept} → {conn.application}**",
                    expanded=(i == 1)
                ):
                    st.write("**How it's applied:**", conn.description)
                    st.write("**Why it matters:**", conn.relevance)
        else:
            st.info("No real-world connections identified")
        
        st.divider()


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main Streamlit application."""
    
    # Initialize session state for persistent storage
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'verifiable_metadata' not in st.session_state:
        st.session_state.verifiable_metadata = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'ingestion_error' not in st.session_state:
        st.session_state.ingestion_error = None
    if 'ingestion_error_details' not in st.session_state:
        st.session_state.ingestion_error_details = None
    if 'ingestion_ready' not in st.session_state:
        st.session_state.ingestion_ready = False
    if 'ingestion_payload' not in st.session_state:
        st.session_state.ingestion_payload = None
    if 'current_run_id' not in st.session_state:
        st.session_state.current_run_id = None
    if 'latest_run_summary' not in st.session_state:
        st.session_state.latest_run_summary = None
    if 'loaded_run' not in st.session_state:
        st.session_state.loaded_run = None
    if 'loaded_run_artifacts' not in st.session_state:
        st.session_state.loaded_run_artifacts = None
    if 'show_loaded_run' not in st.session_state:
        st.session_state.show_loaded_run = False
    if 'notes_char_count' not in st.session_state:
        st.session_state.notes_char_count = 0
    if 'extraction_methods' not in st.session_state:
        st.session_state.extraction_methods = []
    if 'ingestion_diagnostics' not in st.session_state:
        st.session_state.ingestion_diagnostics = {}
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    if (
        st.session_state.processing_complete
        and not st.session_state.ingestion_error
        and st.session_state.ingestion_payload
    ):
        st.session_state.ingestion_ready = True
    if 'has_pyarrow' not in st.session_state:
        try:
            import pyarrow  # noqa: F401
            st.session_state.has_pyarrow = True
        except Exception:
            st.session_state.has_pyarrow = False
    if 'pyarrow_warned' not in st.session_state:
        st.session_state.pyarrow_warned = False
    
    # ====================================================================
    # SIDEBAR - LLM SELECTION
    # ====================================================================
    
    with st.sidebar:
        st.title("⚙️ Settings")
        st.divider()
        
        # LLM Selection
        st.subheader("🤖 AI Model")
        
        # Check available LLM providers
        available_providers = LLMProviderFactory.get_available_providers(
            openai_api_key=config.OPENAI_API_KEY,
            ollama_url="http://localhost:11434"
        )
        
        provider_options = []
        provider_map = {}
        
        if available_providers.get("OpenAI (GPT-4)", False):
            provider_options.append("🌐 OpenAI (GPT-4)")
            provider_map["🌐 OpenAI (GPT-4)"] = "openai"
        
        if available_providers.get("Local LLM - Ollama", False):
            provider_options.append("💻 Local LLM (Ollama)")
            provider_map["💻 Local LLM (Ollama)"] = "ollama"
        
        # Add demo mode as fallback
        provider_options.append("🎬 Demo Mode (Offline)")
        provider_map["🎬 Demo Mode (Offline)"] = "demo"
        
        if len(provider_options) == 1:  # Only demo mode available
            st.warning("⚠️ No LLM providers available. Using Demo Mode (read-only).\n\nTo enable processing:\n1. Set OPENAI_API_KEY in .env\n2. Or run Ollama locally at http://localhost:11434")
        
        default_index = 0
        if "🌐 OpenAI (GPT-4)" in provider_options:
            default_index = provider_options.index("🌐 OpenAI (GPT-4)")
        elif "💻 Local LLM (Ollama)" in provider_options:
            default_index = provider_options.index("💻 Local LLM (Ollama)")

        selected_llm = st.radio(
            "Choose AI model:",
            provider_options,
            index=default_index,
            help="OpenAI uses cloud API. Local LLM runs on your machine (faster, free, private)."
        )
        
        llm_type = provider_map[selected_llm]
        
        if llm_type == "ollama":
            st.success("💻 Using Local LLM - Faster & Private!")
            st.caption("Running on your machine · No API calls")
        elif llm_type == "demo":
            st.info("🎬 Demo Mode - Display and Export Features Only")
            st.caption("Processing disabled. View and download previous results.")
        else:
            st.info("🌐 Using OpenAI - Higher quality")
            st.caption(f"Model: GPT-4")
        
        # Processing options
        st.divider()
        st.subheader("⚡ Processing")
        
        # Verifiable Mode toggle
        enable_verifiable_mode = st.checkbox(
            "Enable Verifiable Mode (Research)",
            value=True,
            help=(
                "Enforces evidence-grounded, claim-based generation. "
                "Claims without sufficient evidence are rejected. "
                "Provides traceability and confidence estimates."
            )
        )

        if enable_verifiable_mode and not st.session_state.has_pyarrow and not st.session_state.pyarrow_warned:
            st.warning("pyarrow missing; install with pip install pyarrow. Dataframe rendering may be limited.")
            st.session_state.pyarrow_warned = True

        # Online Authority Verification toggle
        st.divider()
        st.subheader("🌐 Online Authority Verification")
        
        enable_online_verification = st.checkbox(
            "Augment with Trusted Online Sources",
            value=False,
            help=(
                "Enable retrieval from authoritative online sources "
                "(Python docs, RFC, academic sources, etc.) "
                "to supplement local evidence."
            )
        )
        
        if enable_online_verification:
            st.info(
                "🔒 **Privacy & Security**\n\n"
                "• Queries redact personally identifiable information (email, phone, SSN)\n"
                "• Only requests from allowlisted authoritative domains\n"
                "• Tier 1 (official docs), Tier 2 (academic), Tier 3 (community)\n"
                "• Cached content is versioned and timestamped\n"
                "• Local evidence always takes precedence"
            )
        
        # Store in session state for use in processing
        st.session_state.enable_online_verification = enable_online_verification
        st.session_state.enable_verifiable_mode = enable_verifiable_mode

        st.session_state.debug_mode = st.checkbox(
            "Debug mode",
            value=st.session_state.debug_mode,
            help="Show diagnostic details and raw report data."
        )
        
        # Domain Profile selector (only shown in verifiable mode)
        domain_profile = "algorithms"  # Default
        if enable_verifiable_mode:
            st.info(
                "🔬 **Verifiable Mode** enabled\n\n"
                "• Claims require evidence\n"
                "• Unsupported claims rejected\n"
                "• Confidence tracking\n"
                "• Traceability graph"
            )
            
            domain_profile = st.selectbox(
                "Select Domain Profile",
                options=["algorithms"],
                format_func=lambda x: {
                    "algorithms": "💻 Computer Science (algorithms + complexity)"
                }[x],
                help=(
                    "Domain profile controls validation rules:\n"
                    "• Computer Science: pseudocode checks, complexity analysis"
                )
            )
        
        enable_streaming = st.checkbox("Stream results", value=True, help="Show results as they're generated")
        
        processing_depth = st.select_slider(
            "Processing depth",
            options=["Fast", "Balanced", "Thorough"],
            value="Balanced",
            help="Affects both quality and speed"
        )
        
        st.divider()
        st.subheader("🎯 Output Sections")
        st.caption("Select which sections to generate (Verifiability Graph always included)")
        
        # Store filter selections in session state
        if 'output_filters' not in st.session_state:
            st.session_state.output_filters = {
                'summary': True,
                'topics': True,
                'concepts': True,
                'equations': True,
                'misconceptions': True,
                'faqs': True,
                'worked_examples': True,
                'real_world': True
            }
        
        # Quick filter presets
        st.write("**Quick Presets:**")
        
        # Store preset in session state to persist across reruns
        if 'filter_preset' not in st.session_state:
            st.session_state.filter_preset = "🎯 All Sections"
        
        preset = st.radio(
            "Choose a preset or customize below",
            ["📝 Summary Only", "🎯 All Sections", "🔧 Custom"],
            index=["📝 Summary Only", "🎯 All Sections", "🔧 Custom"].index(st.session_state.filter_preset),
            horizontal=True,
            label_visibility="collapsed"
        )
        st.session_state.filter_preset = preset
        
        # Apply preset - set filters directly without checkbox interference
        if preset == "📝 Summary Only":
            st.session_state.output_filters = {
                'summary': True, 'topics': False, 'concepts': False, 'equations': False,
                'misconceptions': False, 'faqs': False, 'worked_examples': False, 'real_world': False
            }
        elif preset == "🎯 All Sections":
            st.session_state.output_filters = {
                'summary': True, 'topics': True, 'concepts': True, 'equations': True,
                'misconceptions': True, 'faqs': True, 'worked_examples': True, 'real_world': True
            }
        
        st.divider()
        
        # Show filter status differently based on mode
        if preset == "🔧 Custom":
            # Custom mode: show editable checkboxes
            st.write("**Customize sections:**")
            col_filters_1, col_filters_2 = st.columns(2)
            
            with col_filters_1:
                st.session_state.output_filters['summary'] = st.checkbox(
                    "📝 Summary", value=st.session_state.output_filters.get('summary', True))
                st.session_state.output_filters['topics'] = st.checkbox(
                    "📚 Topics", value=st.session_state.output_filters.get('topics', True))
                st.session_state.output_filters['concepts'] = st.checkbox(
                    "💡 Concepts", value=st.session_state.output_filters.get('concepts', True))
                st.session_state.output_filters['equations'] = st.checkbox(
                    "📐 Equations", value=st.session_state.output_filters.get('equations', True))
            
            with col_filters_2:
                st.session_state.output_filters['misconceptions'] = st.checkbox(
                    "⚠️ Misconceptions", value=st.session_state.output_filters.get('misconceptions', True))
                st.session_state.output_filters['faqs'] = st.checkbox(
                    "❓ FAQs", value=st.session_state.output_filters.get('faqs', True))
                st.session_state.output_filters['worked_examples'] = st.checkbox(
                    "🎯 Examples", value=st.session_state.output_filters.get('worked_examples', True))
                st.session_state.output_filters['real_world'] = st.checkbox(
                    "🌍 Connections", value=st.session_state.output_filters.get('real_world', True))
        else:
            # Preset mode: show as read-only status indicators
            st.write("**Selected sections:**")
            col_filters_1, col_filters_2 = st.columns(2)
            
            sections = [
                ("📝 Summary", 'summary', col_filters_1),
                ("📚 Topics", 'topics', col_filters_1),
                ("💡 Concepts", 'concepts', col_filters_1),
                ("📐 Equations", 'equations', col_filters_1),
                ("⚠️ Misconceptions", 'misconceptions', col_filters_2),
                ("❓ FAQs", 'faqs', col_filters_2),
                ("🎯 Examples", 'worked_examples', col_filters_2),
                ("🌍 Connections", 'real_world', col_filters_2)
            ]
            
            with col_filters_1:
                for label, key, col in sections[:4]:
                    if st.session_state.output_filters.get(key, False):
                        st.markdown(f"✅ {label}")
                    else:
                        st.markdown(f"⬜ {label}")
            
            with col_filters_2:
                for label, key, col in sections[4:]:
                    if st.session_state.output_filters.get(key, False):
                        st.markdown(f"✅ {label}")
                    else:
                        st.markdown(f"⬜ {label}")
            
            st.caption("💡 Switch to '🔧 Custom' to manually adjust sections")
        
        st.caption("✓ Verifiability report includes claim summaries and evidence highlights")

        st.divider()
        st.subheader("Recent Runs (last 3)")

        run_history_path = config.ARTIFACTS_DIR / "run_history.json"
        run_history = load_run_history(run_history_path)

        if not run_history:
            st.caption("No recent runs yet.")
        else:
            for run in reversed(run_history):
                run_id = run.get("run_id", "unknown")
                timestamp = run.get("timestamp", "")
                header = f"{run_id}"
                if timestamp:
                    header = f"{run_id} • {timestamp[:19]}"

                with st.expander(header, expanded=False):
                    st.write(f"Session: {run.get('session_id', 'unknown')}")
                    st.write(f"Domain: {run.get('domain_profile', 'unknown')}")
                    inputs_used = run.get("inputs_used", [])
                    if inputs_used:
                        st.write(f"Inputs: {', '.join(inputs_used)}")

                    ver_stats = run.get("verification_stats", {})
                    avg_conf = ver_stats.get("avg_conf")
                    avg_conf_display = f"{avg_conf:.2f}" if isinstance(avg_conf, (int, float)) else "N/A"
                    st.write(
                        "Claims: "
                        f"total={ver_stats.get('total_claims', 0)}, "
                        f"verified={ver_stats.get('verified', 0)}, "
                        f"rejected={ver_stats.get('rejected', 0)}, "
                        f"low_conf={ver_stats.get('low_conf', 0)}, "
                        f"avg_conf={avg_conf_display}"
                    )

                    paths = run.get("artifact_paths", {})
                    report_md = _load_artifact_text(paths.get("report_md"))
                    report_html = _load_artifact_text(paths.get("report_html"))
                    audit_json = _load_artifact_text(paths.get("audit_json"))
                    metrics_json = _load_artifact_text(paths.get("metrics_json"))
                    graphml_text = _load_artifact_text(paths.get("graphml"))

                    col_a, col_b = st.columns(2)
                    with col_a:
                        if report_md:
                            st.download_button(
                                label="Download report.md",
                                data=report_md,
                                file_name=f"report_{run_id}.md",
                                mime="text/markdown",
                                key=f"dl_md_{run_id}"
                            )
                        if audit_json:
                            st.download_button(
                                label="Download audit.json",
                                data=audit_json,
                                file_name=f"audit_{run_id}.json",
                                mime="application/json",
                                key=f"dl_audit_{run_id}"
                            )
                        if graphml_text:
                            st.download_button(
                                label="Download graph.graphml",
                                data=graphml_text,
                                file_name=f"graph_{run_id}.graphml",
                                mime="application/xml",
                                key=f"dl_graph_{run_id}"
                            )
                    with col_b:
                        if report_html:
                            st.download_button(
                                label="Download report.html",
                                data=report_html,
                                file_name=f"report_{run_id}.html",
                                mime="text/html",
                                key=f"dl_html_{run_id}"
                            )
                        if metrics_json:
                            st.download_button(
                                label="Download metrics.json",
                                data=metrics_json,
                                file_name=f"metrics_{run_id}.json",
                                mime="application/json",
                                key=f"dl_metrics_{run_id}"
                            )

                    if st.button("Load this run", key=f"load_run_{run_id}"):
                        st.session_state.loaded_run = run
                        st.session_state.loaded_run_artifacts = {
                            "report_md": report_md,
                            "report_html": report_html,
                            "audit_json": audit_json,
                            "metrics_json": metrics_json,
                            "graphml": graphml_text,
                        }
                        st.session_state.show_loaded_run = True
        
        if st.session_state.debug_mode:
            # Debug: Show current filter state
            enabled_sections = [k for k, v in st.session_state.output_filters.items() if v]
            disabled_sections = [k for k, v in st.session_state.output_filters.items() if not v]
            
            with st.expander("Filter Debug Info", expanded=False):
                st.caption("Enabled sections:")
                st.write(", ".join(enabled_sections) if enabled_sections else "None")
                st.caption("Disabled sections:")
                st.write(", ".join(disabled_sections) if disabled_sections else "None")
            
            # Diagnostics expander for troubleshooting
            with st.expander("System Diagnostics", expanded=False):
                st.caption("Python Environment:")
                st.code(sys.executable, language="text")
                
                st.caption("Package Versions:")
                try:
                    import streamlit
                    import pandas
                    import networkx
                    st.text(f"streamlit: {streamlit.__version__}")
                    st.text(f"pandas: {pandas.__version__}")
                    st.text(f"networkx: {networkx.__version__}")
                    
                    try:
                        import pyarrow
                        st.text(f"pyarrow: {pyarrow.__version__} OK")
                    except ImportError:
                        st.text("pyarrow: NOT INSTALLED")
                        st.caption("Install: pip install pyarrow")
                    
                    try:
                        import matplotlib
                        st.text(f"matplotlib: {matplotlib.__version__} OK")
                    except ImportError:
                        st.text("matplotlib: NOT INSTALLED")
                        st.caption("Install: pip install matplotlib")
                    
                    try:
                        import faiss
                        st.text(f"faiss: {faiss.__version__} OK")
                    except ImportError:
                        st.text("faiss: NOT INSTALLED")
                        st.caption("Install: pip install faiss-cpu")
                except Exception as e:
                    st.error(f"Error loading diagnostic info: {e}")
                
                st.divider()
                st.caption("System Information")
                st.code(f"Python: {sys.version}", language="text")
                st.code(f"OS: {os.name}", language="text")
                
                if st.button("Copy Diagnostic Info"):
                    st.success("Diagnostic info displayed above - copy manually")
        
        st.divider()
    
    # Title and description
    st.title("📘 Smart Notes")
    st.markdown(
        """
        <div style="margin-bottom: 2rem; padding: 1.5rem; background: linear-gradient(135deg, rgba(10, 132, 255, 0.05) 0%, rgba(52, 199, 89, 0.05) 100%); border-radius: 16px; border-left: 4px solid var(--apple-accent);">
        <p style="margin: 0; font-size: 1rem; line-height: 1.6; color: var(--apple-text);">
        <strong>Transform classroom content into structured, verified study notes</strong><br>
        <span style="color: var(--apple-subtle); font-size: 0.9rem;">AI-powered extraction with optional evidence validation and authenticity assessment</span>
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.divider()
    
    # ====================================================================
    # INPUT
    # ====================================================================
    
    st.header("📥 Input")
    st.caption("Upload notes, images, or audio. We'll extract and structure the content.")

    # Initialize variables at the top level
    notes_text = ""
    notes_images = []
    audio_file = None

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📝 Notes")

        notes_input_method = st.radio(
            "Choose input method:",
            ["Type/Paste Text", "Upload Images & PDFs"],
            horizontal=True
        )

        if notes_input_method == "Type/Paste Text":
            notes_text = st.text_area(
                "Paste or type your notes",
                height=220,
                placeholder="Example:\nCombustion and Flame\n\n1. Combustion is a chemical reaction...",
                label_visibility="collapsed"
            )
        else:
            notes_images = st.file_uploader(
                "Upload note images or PDF files",
                type=["jpg", "jpeg", "png", "bmp", "pdf"],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )

            if notes_images:
                st.success(f"✓ {len(notes_images)} file(s) uploaded - OCR and PDF extraction will extract text")

                if len(notes_images) <= 3:
                    cols = st.columns(len(notes_images))
                    for idx, (col, img) in enumerate(zip(cols, notes_images)):
                        with col:
                            if img.type == "application/pdf":
                                st.info(f"📄 PDF: {img.name}")
                            else:
                                st.image(img, caption=f"Image {idx+1}")

        with st.expander("🎤 Audio (Optional)", expanded=False):
            st.caption("Upload a lecture recording for transcription")
            audio_file = st.file_uploader(
                "Upload audio",
                type=["wav", "mp3", "m4a"],
                label_visibility="collapsed"
            )

    with col2:
        st.subheader("⚙️ Advanced")

        with st.expander("Equations", expanded=False):
            equations = st.text_area(
                "Equations (one per line)",
                height=80,
                placeholder="E=mc²\nF=ma\n...",
                label_visibility="collapsed"
            )

        with st.expander("External Context", expanded=False):
            external_context = st.text_area(
                "Reference material",
                height=80,
                placeholder="Textbook excerpts, guidelines, etc.",
                label_visibility="collapsed"
            )

        with st.expander("🌐 URL Sources (Beta)", expanded=False):
            st.caption("Add YouTube videos or web articles as evidence sources")
            urls_text = st.text_area(
                "URLs (one per line)",
                height=100,
                placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ\nhttps://example.com/article\n...",
                label_visibility="collapsed",
                help="Enter YouTube video URLs or web article URLs (one per line). Content will be fetched and used as evidence for claim verification."
            )
            
            if urls_text and not config.ENABLE_URL_SOURCES:
                st.warning("⚠️ URL ingestion is disabled. Set ENABLE_URL_SOURCES=true in .env to enable.")

        with st.expander("Session ID", expanded=False):
            session_id = st.text_input(
                "Custom session ID",
                value=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                label_visibility="collapsed"
            )

    st.divider()

    has_input_now = has_any_input(
        notes_text=notes_text,
        notes_images=notes_images,
        audio_file=audio_file,
        urls_text=urls_text,
        min_text_chars=1
    )
    if has_input_now and not st.session_state.ingestion_error:
        st.session_state.ingestion_ready = True

    notes_text_len = len(notes_text.strip()) if notes_text else 0
    files_present = bool(notes_images and len(notes_images) > 0)
    
    # Check for valid URLs
    valid_urls, url_error = validate_urls_for_processing(urls_text)
    has_valid_urls = len(valid_urls) > 0
    
    notes_ready_for_verification = (
        (notes_text_len >= config.MIN_INPUT_CHARS_FOR_VERIFICATION)
        or files_present
        or bool(audio_file)
        or has_valid_urls  # URLs count as valid input for verification
    )

    # Primary + secondary actions
    col_fast, col_verifiable = st.columns(2)

    fast_button = False
    verifiable_button = False

    with col_fast:
        fast_button = st.button(
            "Generate Notes",
            type="primary",
            use_container_width=True,
            key="fast_btn",
            disabled=(llm_type == "demo"),
            help="Clean notes without verification"
        )

    with col_verifiable:
        verifiable_button = st.button(
            "Run Verification",
            type="secondary",
            use_container_width=True,
            key="verifiable_btn",
            disabled=(
                llm_type == "demo"
                or not enable_verifiable_mode
                or not notes_ready_for_verification
            ),
            help="Evidence-grounded verification with claim scoring"
        )
        if not enable_verifiable_mode:
            st.caption("Enable Verifiable Mode in Settings to run the report.")
        elif not has_input_now:
            status_msg = get_input_status_message(
                notes_text=notes_text,
                notes_images=notes_images,
                audio_file=audio_file,
                urls_text=urls_text,
                min_text_chars=1
            )
            st.caption(status_msg)
        elif notes_text_len and notes_text_len < config.MIN_INPUT_CHARS_FOR_VERIFICATION and not files_present and not has_valid_urls:
            st.caption(
                f"Need at least {config.MIN_INPUT_CHARS_FOR_VERIFICATION} characters before verification."
            )
        elif files_present and notes_text_len < config.MIN_INPUT_CHARS_FOR_VERIFICATION and not has_valid_urls:
            st.caption("Text will be extracted during verification.")

        if st.session_state.get("debug_mode"):
            logger.info(
                "Verification gate: notes_len=%s, files_present=%s, has_input=%s, ready=%s",
                notes_text_len,
                files_present,
                has_input_now,
                notes_ready_for_verification
            )
    
    # Show message if buttons are disabled
    if llm_type == "demo":
        st.info("💡 To enable processing, set OPENAI_API_KEY or run Ollama locally")
    
    # Determine which mode to run
    generate_button = fast_button or verifiable_button
    should_run_verifiable = verifiable_button
    
    # Process when either button clicked
    if generate_button:
        st.session_state.ingestion_ready = False
        st.session_state.current_run_id = None
        st.session_state.latest_run_summary = None
        # Initialize OCR for PDF processing
        ocr_instance = initialize_ocr()
        
        # Extract text from images and PDFs if uploaded
        ocr_extracted_text = ""
        pdf_extracted_text = ""
        combined_extracted_text = ""
        extraction_methods = []
        ingestion_diagnostics = {
            "pdf_files": [],
            "ocr_images": 0,
            "pdf_chars": 0,
            "ocr_chars": 0
        }
        
        if notes_images and len(notes_images) > 0:
            # Separate images and PDFs
            image_files = [f for f in notes_images if f.type != "application/pdf"]
            pdf_files = [f for f in notes_images if f.type == "application/pdf"]
            
            # Process PDFs
            if pdf_files:
                with st.spinner("📄 Extracting text from PDF files..."):
                    from src.exceptions import EvidenceIngestError, INGESTION_ERRORS
                    
                    try:
                        for pdf_file in pdf_files:
                            try:
                                pdf_text, pdf_metadata = _extract_text_from_pdf(pdf_file, ocr=ocr_instance)
                                if pdf_text:
                                    pdf_extracted_text += pdf_text
                                    # Handle both old and new metadata keys
                                    extraction_method = pdf_metadata.get("extraction_method") or pdf_metadata.get("extraction_method_used", "unknown")
                                    ocr_pages = pdf_metadata.get("ocr_pages", 0)
                                    chars_extracted = pdf_metadata.get("chars_extracted", len(pdf_text))
                                    extraction_methods.append(extraction_method)
                                    
                                    # Get ingestion report for detailed diagnostics
                                    ingestion_report = pdf_metadata.get("ingestion_report")
                                    
                                    pdf_diag = {
                                        "file": pdf_file.name,
                                        "method": extraction_method,
                                        "chars": chars_extracted,
                                        "ocr_pages": ocr_pages
                                    }
                                    
                                    # Add detailed ingestion report if available
                                    if ingestion_report:
                                        pdf_diag.update({
                                            "pages_total": ingestion_report.pages_total,
                                            "pages_low_quality": ingestion_report.pages_low_quality,
                                            "headers_removed": ingestion_report.headers_removed_count,
                                            "watermarks_removed": ingestion_report.watermark_removed_count,
                                            "removed_lines": ingestion_report.removed_lines_count,
                                            "removed_patterns": ingestion_report.removed_patterns_hit,
                                            "quality": ingestion_report.quality_assessment
                                        })
                                    
                                    ingestion_diagnostics["pdf_files"].append(pdf_diag)
                                    ingestion_diagnostics["pdf_chars"] += chars_extracted
                                    logger.info(f"PDF extraction success: {pdf_file.name} ({extraction_method})")
                                    
                                    # Display brief summary with option to expand
                                    st.caption(
                                        f"✓ {pdf_file.name}: {chars_extracted:,} chars, {ocr_pages} OCR pages"
                                    )
                                    
                                    # Show detailed ingestion report in expander
                                    if ingestion_report:
                                        with st.expander(f"📊 Ingestion Report: {pdf_file.name}", expanded=False):
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("Pages", ingestion_report.pages_total)
                                                st.metric("OCR Pages", ingestion_report.pages_ocr)
                                            with col2:
                                                st.metric("Low Quality", ingestion_report.pages_low_quality)
                                                st.metric("Headers Removed", ingestion_report.headers_removed_count)
                                            with col3:
                                                st.metric("Watermarks", ingestion_report.watermark_removed_count)
                                                st.metric("Lines Cleaned", ingestion_report.removed_lines_count)
                                            
                                            if ingestion_report.removed_patterns_hit:
                                                st.write("**Patterns Removed:**")
                                                for pattern, count in ingestion_report.removed_patterns_hit.items():
                                                    st.text(f"  • {pattern}: {count}")
                                            
                                            st.caption(f"Quality: {ingestion_report.quality_assessment}")
                            except EvidenceIngestError as e:
                                # Store ingestion error for display
                                st.session_state.ingestion_error = e.code
                                st.session_state.ingestion_error_details = {
                                    "file": pdf_file.name,
                                    "code": e.code,
                                    "message": e.get_user_message(),
                                    "next_steps": e.get_next_steps(),
                                    "details": e.details
                                }
                                
                                # Show error with actionable next steps
                                error_config = INGESTION_ERRORS.get(e.code, {})
                                st.error(f"❌ **Evidence ingestion failed: {pdf_file.name}**\n\n{e.get_user_message()}")
                                
                                if e.get_next_steps():
                                    st.info("💡 **Next steps:**\n" + "\n".join(f"- {step}" for step in e.get_next_steps()))
                                
                                logger.error(f"Evidence ingestion error for {pdf_file.name}: code={e.code}, {e.message}")
                                # Don't continue - mark as ingestion failure and stop processing
                                break
                        
                        if pdf_extracted_text and not st.session_state.ingestion_error:
                            words = len(pdf_extracted_text.split())
                            st.success(f"✓ PDF extraction complete: {len(pdf_extracted_text)} chars, ~{words} words")
                    except EvidenceIngestError as e:
                        # Ingestion error occurred
                        st.session_state.ingestion_error = e.code
                        st.session_state.ingestion_error_details = {
                            "code": e.code,
                            "message": e.get_user_message(),
                            "next_steps": e.get_next_steps()
                        }
                        st.error(f"❌ **Evidence ingestion failed**\n\n{e.get_user_message()}")
                        if e.get_next_steps():
                            st.info("💡 **Next steps:**\n" + "\n".join(f"- {step}" for step in e.get_next_steps()))
                        logger.error(f"PDF ingestion error: code={e.code}, {e.message}")
                    except Exception as e:
                        st.error(f"❌ PDF extraction failed: {str(e)}")
                        logger.error(f"PDF extraction error: {str(e)}")
            
            # Process Images with OCR
            if image_files:
                with st.spinner("📸 Extracting text from images using OCR..."):
                    try:
                        cache = _load_ocr_cache()
                        image_hashes = []
                        for img in image_files:
                            img_bytes = img.getvalue()
                            image_hashes.append(_image_hash(img_bytes))

                        cache_key = "|".join(image_hashes)
                        if cache_key in cache["items"]:
                            ocr_extracted_text = cache["items"][cache_key]
                            st.success(f"✓ Using cached OCR: {len(ocr_extracted_text)} characters")
                            ingestion_diagnostics["ocr_chars"] = len(ocr_extracted_text)
                            ingestion_diagnostics["ocr_images"] = len(image_files)
                            extraction_methods.append("image_ocr_cache")
                        else:
                            selected_model = config.OLLAMA_MODEL if llm_type == "ollama" else config.LLM_MODEL
                            ocr_extracted_text = process_images(
                                image_files, 
                                correct_with_llm=True,
                                provider_type=llm_type,
                                api_key=config.OPENAI_API_KEY,
                                ollama_url=config.OLLAMA_URL,
                                model=selected_model
                            )
                            ingestion_diagnostics["ocr_chars"] = len(ocr_extracted_text)
                            ingestion_diagnostics["ocr_images"] = len(image_files)
                            extraction_methods.append("image_ocr")
                            cache["items"][cache_key] = ocr_extracted_text
                            cache["order"] = [k for k in cache["order"] if k != cache_key]
                            cache["order"].append(cache_key)
                            while len(cache["order"]) > 3:
                                old_key = cache["order"].pop(0)
                                cache["items"].pop(old_key, None)
                            _save_ocr_cache(cache)
                            st.success(f"✓ OCR extraction + LLM correction complete: {len(ocr_extracted_text)} characters")
                        
                    except Exception as e:
                        st.error(f"❌ Image OCR extraction failed: {str(e)}")
                        logger.error(f"Image OCR error: {str(e)}")
            
            # Combine all extracted text
            combined_extracted_text = (pdf_extracted_text.strip() + "\n" + ocr_extracted_text.strip()).strip()
            
            if combined_extracted_text:
                with st.expander("✏️ Review & Edit Extracted Text", expanded=True):
                    st.info("💡 The text below has been extracted from your files. You can edit it before processing.")
                    combined_extracted_text = st.text_area(
                        "Extracted Text (editable)",
                        value=combined_extracted_text,
                        height=250,
                        key="extracted_text_area"
                    )
            else:
                st.warning("⚠️ No text could be extracted from the uploaded files.")
        
        # Combine typed notes and extracted text (from PDFs and OCR)
        combined_notes = notes_text
        if combined_extracted_text:
            if combined_notes:
                combined_notes += "\n\n---\n\n" + combined_extracted_text
            else:
                combined_notes = combined_extracted_text

        st.session_state.notes_char_count = len(combined_notes or "")
        st.session_state.extraction_methods = extraction_methods
        st.session_state.ingestion_diagnostics = ingestion_diagnostics
        
        # Parse and validate URLs
        valid_urls, url_validation_error = validate_urls_for_processing(urls_text)
        
        # Validation - check if we have any input at all
        # Include valid_urls in input check
        has_input = combined_notes or audio_file or (notes_images and len(notes_images) > 0) or len(valid_urls) > 0
        
        if not has_input:
            st.error("Please provide notes, images, audio, or URL sources.")
        # Check if there's a URL validation error
        elif url_validation_error:
            st.error(f"⚠️ {url_validation_error}")
        # Check if ingestion failed (don't proceed with verification)
        elif st.session_state.ingestion_error:
            st.error(
                f"🚫 **Evidence ingestion failed**\n\n"
                f"**Error Code**: {st.session_state.ingestion_error}\n\n"
                f"**Reason**: {st.session_state.ingestion_error_details.get('message', 'Unknown error')}"
            )
            
            if st.session_state.ingestion_error_details.get('next_steps'):
                st.info("💡 **How to fix this:**\n" + "\n".join(f"• {step}" for step in st.session_state.ingestion_error_details['next_steps']))
            
            st.error("❌ Cannot proceed to verification or analysis until evidence ingestion succeeds.")
            logger.warning(f"Verification skipped due to ingestion error: {st.session_state.ingestion_error}")
        else:
            # Override verifiable mode based on button clicked
            actual_verifiable_mode = should_run_verifiable and enable_verifiable_mode
            
            if should_run_verifiable and not enable_verifiable_mode:
                st.warning(
                    "⚠️ Verifiable Mode not enabled in settings. "
                    "Enable it in the sidebar Settings → Processing to use evidence verification."
                )
                actual_verifiable_mode = False
            
            # Apply online verification setting from session state to config
            if hasattr(st.session_state, 'enable_online_verification'):
                config.ENABLE_ONLINE_VERIFICATION = st.session_state.enable_online_verification
            
            # Show information about what will happen
            if actual_verifiable_mode and len(combined_notes or "") < config.MIN_INPUT_CHARS_FOR_VERIFICATION:
                st.warning(
                    f"Verification needs at least {config.MIN_INPUT_CHARS_FOR_VERIFICATION} characters. "
                    "Running notes only."
                )
                actual_verifiable_mode = False

            if actual_verifiable_mode:
                st.info(
                    "🔬 **Running in Verifiable Mode**\n\n"
                    "• Claims will be verified against evidence\n"
                    "• Only well-supported claims will be included\n"
                    "• Each claim shows confidence and evidence sources"
                )
            else:
                st.info(
                    "🚀 **Running in Fast Mode**\n\n"
                    "• Quick analysis without evidence verification\n"
                    "• All extracted content is processed\n"
                    "• Suitable for any content type"
                )
            
            progress_callback = None
            if actual_verifiable_mode:
                progress_callback = _create_verifiability_progress_ui()

            # Process the session and store in session state
            result, verifiable_metadata = process_session(
                audio_file=audio_file,
                notes_text=combined_notes,
                equations=equations,
                external_context=external_context,
                session_id=session_id,
                verifiable_mode=actual_verifiable_mode,
                domain_profile=domain_profile if actual_verifiable_mode else "algorithms",
                llm_provider_type=llm_type,
                progress_callback=progress_callback,
                debug_mode=st.session_state.debug_mode,
                urls=valid_urls if valid_urls else None
            )
            
            # Store results in session state
            st.session_state.result = result
            st.session_state.verifiable_metadata = verifiable_metadata
            st.session_state.processing_complete = True
            st.session_state.ingestion_ready = True
            st.session_state.ingestion_payload = {
                "notes_text": combined_notes,
                "equations": equations,
                "external_context": external_context,
                "session_id": session_id,
                "urls_text": urls_text if 'urls_text' in locals() else "",
                "audio_present": audio_file is not None
            }
    
    # ========================================================================
    # RESULTS DISPLAY
    # ========================================================================
    
    # Show ingestion error if present (takes precedence over results)
    if st.session_state.ingestion_error:
        st.divider()
        with st.container(border=True):
            st.error(
                f"🚫 **Evidence ingestion failed** - Cannot proceed with analysis\n\n"
                f"**Error**: {st.session_state.ingestion_error_details.get('message', 'Unknown error')}\n\n"
                f"**Status**: Verification pipeline skipped (no usable evidence extracted)"
            )
            
            if st.session_state.ingestion_error_details.get('next_steps'):
                st.markdown("**How to fix this:**")
                for step in st.session_state.ingestion_error_details['next_steps']:
                    st.markdown(f"• {step}")
        
        st.divider()
        st.info("🔄 Please fix the issue above and try again with different input.")
    
    # Display results only if processing succeeded and no ingestion error
    elif st.session_state.processing_complete and st.session_state.result:
        st.success("✅ Processing complete!")
        st.divider()

        notes_tab, report_tab, reports_tab = st.tabs(["Notes", "Verifiability Report", "Reports"])

        with notes_tab:
            if st.session_state.verifiable_metadata and st.session_state.verifiable_metadata.get("verification_unavailable"):
                diagnostics = st.session_state.verifiable_metadata.get("evidence_diagnostics", {})
                st.warning(
                    "Verification unavailable due to insufficient evidence. "
                    "Showing baseline study guide instead."
                )
                if diagnostics:
                    st.caption(
                        f"Extracted {diagnostics.get('extracted_text_length', 0)} chars "
                        f"(min required: {diagnostics.get('minimum_required', 0)})."
                    )
                    suggestions = diagnostics.get("suggestions", [])
                    if suggestions:
                        st.info("How to fix:\n" + "\n".join(f"- {s}" for s in suggestions))

            display_output(st.session_state.result, None)

            st.divider()
            output_obj = st.session_state.result.get('output')
            session_id_for_export = output_obj.session_id if output_obj and hasattr(output_obj, 'session_id') else ""
            export_data = {}
            if output_obj is not None:
                if hasattr(output_obj, "model_dump"):
                    export_data = output_obj.model_dump()
                elif hasattr(output_obj, "dict"):
                    export_data = output_obj.dict()
                else:
                    export_data = output_obj

                if isinstance(export_data, dict):
                    if "summary" not in export_data and "class_summary" in export_data:
                        export_data["summary"] = export_data["class_summary"]

            notes_md = QuickExportButtons._to_markdown(export_data) if export_data else ""
            st.download_button(
                label="Export Notes (Markdown)",
                data=notes_md,
                file_name=f"notes_{session_id_for_export}.md",
                mime="text/markdown",
                disabled=not bool(notes_md)
            )

            with st.expander("Diagnostics", expanded=False):
                methods = st.session_state.extraction_methods or []
                if methods:
                    st.write(f"Extraction methods: {', '.join(sorted(set(methods)))}")
                st.write(f"Notes characters: {st.session_state.notes_char_count:,}")

                evidence_stats = (st.session_state.verifiable_metadata or {}).get("evidence_stats")
                if evidence_stats:
                    st.write(f"Evidence chunks: {evidence_stats.get('num_chunks', 0)}")

                metrics = (st.session_state.verifiable_metadata or {}).get("metrics", {})
                evidence_metrics = metrics.get("evidence_metrics", {})
                if evidence_metrics:
                    st.write(
                        "Retrieval stats: avg evidence/claim "
                        f"{evidence_metrics.get('avg_evidence_per_claim', 0):.2f}, "
                        "avg evidence quality "
                        f"{evidence_metrics.get('avg_evidence_quality', 0):.2f}"
                    )

        with report_tab:
            if st.session_state.verifiable_metadata and st.session_state.verifiable_metadata.get("verifiable_mode"):
                display_verifiability_report(
                    st.session_state.verifiable_metadata,
                    st.session_state.result.get('output'),
                    debug_mode=st.session_state.get("debug_mode", False)
                )
            else:
                st.info("Run verification to generate the report.")

        with reports_tab:
            if st.session_state.show_loaded_run and st.session_state.loaded_run_artifacts:
                loaded_run = st.session_state.loaded_run or {}
                st.subheader("Loaded Run (cached)")
                st.caption(
                    f"Run ID: {loaded_run.get('run_id', 'unknown')} · "
                    f"Session: {loaded_run.get('session_id', 'unknown')}"
                )

                cached = st.session_state.loaded_run_artifacts
                col1, col2, col3 = st.columns(3)
                with col1:
                    if cached.get("report_md"):
                        st.download_button(
                            label="Download Markdown",
                            data=cached["report_md"],
                            file_name=f"research_report_{loaded_run.get('run_id', 'run')}.md",
                            mime="text/markdown",
                            help="Cached markdown report"
                        )
                with col2:
                    if cached.get("report_html"):
                        st.download_button(
                            label="Download HTML",
                            data=cached["report_html"],
                            file_name=f"research_report_{loaded_run.get('run_id', 'run')}.html",
                            mime="text/html",
                            help="Cached HTML report"
                        )
                with col3:
                    if cached.get("audit_json"):
                        st.download_button(
                            label="Download Audit JSON",
                            data=cached["audit_json"],
                            file_name=f"research_report_audit_{loaded_run.get('run_id', 'run')}.json",
                            mime="application/json",
                            help="Cached audit JSON"
                        )

                st.divider()
                preview_format = st.radio(
                    "View cached report in:",
                    ["Markdown", "HTML"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key="cached_report_preview"
                )
                if preview_format == "Markdown" and cached.get("report_md"):
                    st.markdown(cached["report_md"])
                elif preview_format == "HTML" and cached.get("report_html"):
                    st.components.v1.html(cached["report_html"], height=800, scrolling=True)
                else:
                    st.info("Cached report preview unavailable.")

            if st.session_state.verifiable_metadata and st.session_state.verifiable_metadata.get("verifiable_mode"):
                _display_research_reports(
                    st.session_state.verifiable_metadata,
                    st.session_state.result.get('output')
                )
            else:
                st.info("Run verification to generate research reports.")


if __name__ == "__main__":
    main()
