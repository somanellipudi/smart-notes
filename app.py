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
    # Execution Flow Dashboard imports
    from src.ui.progress_tracker import create_run_context, RunContext
    from src.ui.execution_flow_ui import render_execution_flow_dashboard
    from src.ui.pipeline_instrumentation import (
        track_stage,
        track_llm_call,
        track_ocr_usage,
        track_url_ingestion,
        track_retrieval_usage
    )
    from src.ui.redesigned_app import render_redesigned_ui
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
    # Check if OCR is enabled
    if not config.OCR_ENABLED:
        logger.info("OCR is disabled (OCR_ENABLED=False)")
        return None
    
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
    # Get run context from session state
    run_context = st.session_state.get("run_context")
    
    # STAGE: Inputs Received
    if run_context:
        run_context.start_stage("inputs_received")
        run_context.complete_stage("inputs_received", {
            "audio_provided": audio_file is not None,
            "urls_provided": len(urls) if urls else 0,
            "text_chars": len(notes_text),
            "equations_provided": len(equations.split('\n')) if equations else 0
        })
    
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
    url_extraction_summary = {"success": 0, "failed": 0}  # Track for logging only
    failed_urls_with_errors = []  # Track failed URLs with error messages
    youtube_count = 0
    article_count = 0
    if urls:
        with st.spinner("🌐 Fetching URL content..."):
            try:
                for url in urls:
                    try:
                        content, metadata = fetch_url_text(url)
                        if content:
                            source_type = metadata.get("source_type", "unknown")
                            url_extraction_summary["success"] += 1
                            # Count YouTube vs articles
                            if source_type == "youtube":
                                youtube_count += 1
                            else:
                                article_count += 1
                            logger.info(f"✓ URL ingestion: {source_type} from {url[:60]}... ({len(content)} chars)")
                            url_extracted_text += content + "\n\n"
                        else:
                            error_msg = metadata.get("error", "No content extracted")
                            url_extraction_summary["failed"] += 1
                            failed_urls_with_errors.append((url, error_msg))
                            logger.warning(f"URL extraction failed for {url}: {error_msg}")
                    except Exception as e:
                        url_extraction_summary["failed"] += 1
                        failed_urls_with_errors.append((url, str(e)))
                        logger.warning(f"Error fetching {url}: {str(e)}")
            except Exception as e:
                logger.error(f"URL ingestion error: {e}")
        
        # Show results with detailed error information
        if url_extraction_summary["success"] > 0:
            st.success(f"✓ Extracted content from {url_extraction_summary['success']} URL(s)")
        
        if failed_urls_with_errors:
            if url_extraction_summary["success"] == 0:
                st.info(f"ℹ️ URL extraction: Could not fetch content from {url_extraction_summary['failed']} URL(s). Proceeding with notes only.")
            with st.expander(f"📋 URL extraction details ({url_extraction_summary['failed']} failed)", expanded=False):
                for failed_url, error in failed_urls_with_errors:
                    st.write(f"**{failed_url}**")
                    st.write(f"⚠️ {error}")
                    st.divider()
        
        # Track URL ingestion in run context
        if run_context:
            track_url_ingestion(
                run_context,
                total=len(urls),
                success=url_extraction_summary["success"],
                youtube=youtube_count,
                articles=article_count
            )
    
    # Combine notes with URL-extracted content
    if url_extracted_text:
        if notes_text:
            notes_text = notes_text + "\n\n---\n\nContent from URLs:\n\n" + url_extracted_text
        else:
            notes_text = url_extracted_text
    
    # Step 2: Preprocessing
    if run_context:
        run_context.start_stage("ingestion_cleaning")
    
    with st.spinner("🔄 Preprocessing content..."):
        equations_list = [eq.strip() for eq in equations.split('\n') if eq.strip()]
        
        preprocessed = preprocess_classroom_content(
            handwritten_notes=notes_text,
            transcript=transcript,
            equations=equations_list
        )
        
        combined_text = preprocessed["combined_text"]
        st.success(f"✓ Preprocessing complete: {preprocessed['metadata']['num_segments']} segments")
        
        # Complete ingestion stage
        if run_context:
            run_context.complete_stage("ingestion_cleaning", {
                "segments": preprocessed['metadata']['num_segments'],
                "combined_chars": len(combined_text),
                "transcript_chars": len(transcript),
                "notes_chars": len(notes_text)
            })
            # Skip OCR stage for now (will be tracked separately if images are processed)
            run_context.skip_stage("ocr_extraction", "No OCR processing in main pipeline")
            run_context.skip_stage("chunking_provenance", "Fast chunking - no detailed tracking")
    
    # Step 3: Multi-Stage Reasoning
    if run_context:
        run_context.start_stage("llm_generation")
    
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
        
        # Track LLM generation completion
        if run_context:
            track_llm_call(
                run_context,
                provider=llm_provider_type if llm_provider_type != "fallback" else "openai",
                model=selected_model,
                tokens=None  # Could extract from verifiable_metadata if available
            )
            run_context.complete_stage("llm_generation", {
                "provider": llm_provider_type,
                "model": selected_model,
                "topics_generated": len(output.topics),
                "concepts_generated": len(output.key_concepts)
            })
            
            # Track verification stages
            if verifiable_mode and verifiable_metadata:
                # Claim extraction
                run_context.start_stage("claim_extraction")
                claims_count = verifiable_metadata.get("total_claims", 0)
                run_context.complete_stage("claim_extraction", {
                    "claims_total": claims_count
                })
                
                # Retrieval & Reranking
                run_context.start_stage("retrieval_reranking")
                track_retrieval_usage(run_context, enabled=True)
                run_context.complete_stage("retrieval_reranking", {
                    "retrieval_enabled": True
                })
                
                # Verification
                run_context.start_stage("verification")
                verified_count = verifiable_metadata.get("verified_claims", 0)
                run_context.complete_stage("verification", {
                    "verified_claims": verified_count,
                    "total_claims": claims_count
                })
            else:
                # Skip verification stages in fast mode
                run_context.skip_stage("claim_extraction", "Fast mode - no verification")
                run_context.skip_stage("retrieval_reranking", "Fast mode - no verification")
                run_context.skip_stage("verification", "Fast mode - no verification")
            
            # Skip embedding stage (not explicitly used in this flow)
            run_context.skip_stage("embedding_indexing", "Not used in this pipeline configuration")
    
    # Step 4: Evaluation
    with st.spinner("📊 Evaluating quality..."):
        evaluation = evaluate_session_output(
            output=output,
            source_content=combined_text,
            external_context=external_context
        )
        
        st.success(f"✓ Evaluation complete")
    
    # Step 5: Save Session
    if run_context:
        run_context.start_stage("reporting_exports")
    
    with st.spinner("💾 Saving session..."):
        session_manager = SessionManager()
        session_path = session_manager.save_session(output, overwrite=True)
        st.success(f"✓ Session saved")
        
        if run_context:
            run_context.complete_stage("reporting_exports", {
                "session_saved": True,
                "session_path": str(session_path)
            })
            # Save run context to artifacts
            artifacts_dir = Path(config.ARTIFACTS_DIR)
            run_context.save(artifacts_dir)
            logger.info(f"Run context saved for session {run_context.session_id}")
    
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
            # URL metrics
            url_count=ingestion_diagnostics.get("url_count", 0),
            url_fetch_success_count=ingestion_diagnostics.get("url_fetch_success_count", 0),
            url_chunks_total=ingestion_diagnostics.get("url_chunks_total", 0),
            # Text metrics
            text_chars_total=ingestion_diagnostics.get("text_chars_total", 0),
            text_chunks_total=ingestion_diagnostics.get("text_chunks_total", 0),
            # Audio metrics
            audio_seconds=ingestion_diagnostics.get("audio_seconds", 0.0),
            transcript_chars=ingestion_diagnostics.get("transcript_chars", 0),
            transcript_chunks_total=ingestion_diagnostics.get("transcript_chunks_total", 0),
            # Overall (renamed from total_chunks -> chunks_total_all_sources)
            chunks_total_all_sources=ingestion_diagnostics.get("chunks_total_all_sources", 
                                                                ingestion_diagnostics.get("total_chunks", 0)),
            avg_chunk_size_all_sources=ingestion_diagnostics.get("avg_chunk_size_all_sources",
                                                                  ingestion_diagnostics.get("avg_chunk_size")),
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
            label_visibility="collapsed",
            key="research_report_preview_format"
        )
        
        if preview_format == "Markdown":
            st.markdown(md_content)
        else:
            st.components.v1.html(html_content, height=800, scrolling=True)
    
    except Exception as e:
        logger.exception("Error in _display_research_reports")
        st.error(f"Failed to generate research reports: {e}")
        with st.expander("🐛 Error Details"):
            st.code(str(e))


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
    
    # Always use redesigned UI - Classic UI removed
    from src.ui.redesigned_app import render_redesigned_ui
    render_redesigned_ui()


if __name__ == "__main__":
    main()
