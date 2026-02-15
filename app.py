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
from typing import Tuple, Optional, Dict, Any
import requests
from io import StringIO

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '1'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import streamlit as st

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.audio.transcription import transcribe_audio
    from src.audio.image_ocr import ImageOCR, process_images
    from src.preprocessing.text_processing import preprocess_classroom_content
    from src.reasoning.verifiable_pipeline import VerifiablePipelineWrapper
    from src.evaluation.metrics import evaluate_session_output
    from src.study_book.session_manager import SessionManager
    from src.llm_provider import LLMProviderFactory
    from src.output_formatter import StreamingOutputDisplay
    from src.reasoning.fallback_handler import FallbackGenerator, PipelineEnhancer
    from src.streamlit_display import StreamlitProgressDisplay, QuickExportButtons
    from src.display.interactive_claims import InteractiveClaimDisplay
    from src.graph.graph_sanitize import export_graphml_bytes
    from src.graph.claim_graph import export_adjacency_json
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
    domain_profile: str = "physics",
    llm_provider_type: str = "openai"
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
        domain_profile: Domain profile (physics, discrete_math, algorithms)
        llm_provider_type: LLM provider to use
    
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

            output = ClassSessionOutput(
                session_id=session_id,
                class_summary=fallback_output["summary"],
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
            llm_provider_type = "ollama"

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
        
        # Debug: Log what filters are being passed
        st.info(f"🔍 Filters being sent to pipeline: {st.session_state.output_filters}")
        
        if llm_provider_type != "fallback":
            try:
                output, verifiable_metadata = pipeline.process(
                    combined_content=combined_text,
                    equations=equations_list,
                    external_context=external_context,
                    session_id=session_id,
                    verifiable_mode=verifiable_mode,
                    output_filters=st.session_state.output_filters
                )
            except Exception as e:
                error_text = str(e)
                if "insufficient_quota" in error_text or "RateLimitError" in error_text:
                    st.error(
                        "❌ OpenAI quota exceeded. Please check your plan/billing, "
                        "or switch to Local LLM (Ollama) in the sidebar."
                    )
                elif "WinError 10061" in error_text or "localhost" in error_text:
                    st.warning("Ollama is not running. Start it or switch to OpenAI. Using fallback output.")
                else:
                    st.error(f"❌ LLM call failed: {e}")

                output, verifiable_metadata = _build_fallback_output(error_text)
        
        if verifiable_mode and verifiable_metadata:
            metrics = verifiable_metadata["metrics"]
            st.success(
                f"✓ Verifiable reasoning complete: "
                f"{metrics['verified_claims']}/{metrics['total_claims']} claims verified "
                f"(rejection rate: {metrics['rejection_rate']:.1%})"
            )
            timings = verifiable_metadata.get("timings", {})
            total_time = verifiable_metadata.get("total_time_seconds")
            if timings or total_time is not None:
                with st.expander("⏱️ Processing Time Breakdown", expanded=False):
                    if total_time is not None:
                        st.metric("Total Time (s)", f"{total_time:.2f}")
                    if timings:
                        for step, seconds in timings.items():
                            st.write(f"• {step.replace('_', ' ')}: {seconds:.2f}s")
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


def display_output(result: dict, verifiable_metadata: Optional[Dict[str, Any]] = None):
    """
    Display structured output using new streaming display system.
    
    Args:
        result: Result dictionary from process_session
        verifiable_metadata: Optional verifiable mode metadata
    """
    output = result["output"]
    evaluation = result["evaluation"]
    
    # ========================================================================
    # VERIFIABLE MODE METRICS (if enabled)
    # ========================================================================
    
    if verifiable_metadata:
        st.header("🔬 Verifiable Mode Metrics")
        
        metrics = verifiable_metadata["metrics"]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Verified Claims",
                metrics["verified_claims"],
                help="Claims with sufficient evidence"
            )
        
        with col2:
            st.metric(
                "Rejected Claims",
                metrics["rejected_claims"],
                help="Claims lacking evidence"
            )
        
        with col3:
            st.metric(
                "Rejection Rate",
                f"{metrics['rejection_rate']:.1%}",
                help="Percentage of claims rejected"
            )
        
        with col4:
            st.metric(
                "Avg Confidence",
                f"{metrics['avg_confidence']:.2f}",
                help="Average claim confidence"
            )
        
        # Evidence metrics
        ev_metrics = metrics.get("evidence_metrics", {})
        if ev_metrics:
            st.subheader("Evidence Quality")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Avg Evidence/Claim",
                    f"{ev_metrics.get('avg_evidence_per_claim', 0):.1f}",
                    help="Average evidence items per claim"
                )
            
            with col2:
                st.metric(
                    "Unsupported Rate",
                    f"{ev_metrics.get('unsupported_rate', 0):.1%}",
                    help="Claims without evidence"
                )
            
            with col3:
                st.metric(
                    "Evidence Quality",
                    f"{ev_metrics.get('avg_evidence_quality', 0):.2f}",
                    help="Average evidence relevance score"
                )
        
        # Quality flags
        if metrics.get("quality_flags"):
            st.warning("**Quality Flags:**\n\n" + "\n".join(f"• {flag}" for flag in metrics["quality_flags"]))
        
        # Baseline comparison
        if "baseline_comparison" in metrics:
            with st.expander("📊 Comparison to Baseline (Standard Mode)"):
                comp = metrics["baseline_comparison"]
                st.write(f"**Baseline Items:** {comp['baseline_total_items']}")
                st.write(f"**Verified Claims:** {comp['verifiable_verified_claims']}")
                st.write(f"**Reduction Rate:** {comp['reduction_rate']:.1%}")
                st.write(f"**Est. Hallucination Reduction:** {comp['hallucination_reduction_estimate']:.1%}")
        
        # Claim review section for research
        st.subheader("📌 Claim Review")
        
        # Claim review section for research
        st.subheader("📌 Claim Review - AI Content Authenticity Assessment")
        
        st.info(
            "🔬 **Research Mode**: Hover over content to inspect AI-generated claims. "
            "Each claim shows confidence score, evidence sources, and verifiability assessment."
        )
        
        claim_filter = st.selectbox(
            "Show claims",
            ["All", "Verified", "Low confidence", "Rejected"],
            index=0
        )
        
        claim_collection = verifiable_metadata.get("claim_collection")
        all_claims = claim_collection.claims if claim_collection else []
        
        def _matches_filter(claim_status: str) -> bool:
            if claim_filter == "All":
                return True
            return claim_status.lower().replace("_", " ") == claim_filter.lower()
        
        filtered_claims = [
            c for c in all_claims
            if _matches_filter(getattr(c.status, "value", str(c.status)))
        ]
        
        claim_rows = []
        for c in filtered_claims:
            status_value = getattr(c.status, "value", str(c.status))
            # Use ui_display from metadata or fallback to claim_text or draft_text
            display_text = (
                c.metadata.get("ui_display", "") or 
                c.claim_text or 
                c.metadata.get("draft_text", "[No description]")
            )
            claim_rows.append({
                "Claim ID": c.claim_id[:8] + "...",  # Truncate ID for readability
                "Type": getattr(c.claim_type, "value", str(c.claim_type)),
                "Status": status_value,
                "Confidence": round(c.confidence, 3),
                "Evidence Count": len(c.evidence_ids),
                "Claim": (display_text[:100] + "…") if len(display_text) > 100 else display_text
            })
        
        # Use st.table() when pyarrow is available; otherwise fall back to JSON
        if claim_rows:
            try:
                import pyarrow  # noqa: F401
                st.table(claim_rows)
            except Exception:
                st.warning("PyArrow is unavailable. Showing claims as JSON.")
                st.json(claim_rows)
        else:
            st.info("No claims to display")
        
        # New interactive claims display with authenticity assessment
        st.subheader("🔬 Interactive Verifiability Assessment")
        st.write("Click on any claim below to inspect sources, confidence scores, and authenticity indicators.")
        
        # Use new interactive display
        InteractiveClaimDisplay.display_claims_summary_with_verifiability(
            filtered_claims,
            show_ai_badge=True
        )
        
        # Dependency warnings display
        if verifiable_metadata.get("dependency_warnings"):
            st.divider()
            dep_warnings = verifiable_metadata["dependency_warnings"]
            st.subheader("⚠️ Cross-Claim Dependency Warnings")
            st.write(f"Found {len(dep_warnings)} claims with undefined term references:")
            
            for i, warning in enumerate(dep_warnings[:10], 1):  # Show first 10
                with st.expander(f"Warning {i}: {warning.claim_text[:60]}..."):
                    st.write(f"**Claim ID:** {warning.claim_id[:8]}...")
                    st.write(f"**Severity:** {warning.severity}")
                    st.write(f"**Undefined Terms:** {', '.join(warning.undefined_terms)}")
                    st.write(f"**Suggestion:** {warning.suggestion}")
            
            if len(dep_warnings) > 10:
                st.info(f"+ {len(dep_warnings) - 10} more warnings (see export for full list)")
        
        # Domain profile and threat model info
        st.divider()
        st.subheader("🎯 Research Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Domain Profile:** {verifiable_metadata.get('domain_profile_display', 'N/A')}")
            if verifiable_metadata.get("domain_profile"):
                profile_name = verifiable_metadata["domain_profile"]
                try:
                    profile = config.get_domain_profile(profile_name)
                    st.caption(profile.description)
                except:
                    pass
        
        with col2:
            if verifiable_metadata.get("threat_model"):
                threat_summary = verifiable_metadata["threat_model"]
                st.write(f"**Threat Model:** {threat_summary['in_scope_count']} in-scope threats")
                st.caption(f"{', '.join(threat_summary['in_scope_threats'][:3])}...")
        
        # Authenticity report for research
        if filtered_claims:
            with st.expander("📊 Authenticity Report (for Research)", expanded=False):
                report = InteractiveClaimDisplay.create_claim_authenticity_report(
                    filtered_claims,
                    st.session_state.result['output'].session_id if 'output' in st.session_state.result else "unknown"
                )
                
                st.json(report)
                
                # Export button
                report_json = json.dumps(report, indent=2)
                st.download_button(
                    label="📥 Download Authenticity Report (JSON)",
                    data=report_json,
                    file_name=f"authenticity_report_{report['session_id']}.json",
                    mime="application/json"
                )
        
        # Graph visualization
        st.divider()
        st.subheader("📊 Claim-Evidence Network Graph")
        
        claim_graph = verifiable_metadata.get("claim_graph")
        if claim_graph:
            try:
                import networkx as nx
                HAS_MATPLOTLIB = True
                try:
                    import matplotlib.pyplot as plt
                    import matplotlib.patches as mpatches
                except ImportError:
                    HAS_MATPLOTLIB = False
                
                # Build networkx graph
                G = nx.DiGraph()
                claim_info = {}  # Store claim metadata for better labeling
                
                # Add nodes for claims
                if hasattr(claim_graph, 'graph') and claim_graph.graph:
                    for node_id, node_data in claim_graph.graph.nodes(data=True):
                        node_type = node_data.get("node_type", "claim")
                        G.add_node(node_id, node_type=node_type, data=node_data)
                        if node_type == "claim":
                            claim_info[node_id] = node_data
                    
                    # Add edges for evidence relationships
                    for source, target, edge_data in claim_graph.graph.edges(data=True):
                        G.add_edge(source, target, weight=edge_data.get("weight", 1.0))
                
                # If graph is empty, create fallback visualization with just claims
                if G.number_of_nodes() == 0:
                    # Fallback: Create graph from all claims (even if no evidence)
                    claim_collection = verifiable_metadata.get("claim_collection")
                    if claim_collection and claim_collection.claims:
                        for claim in claim_collection.claims:
                            claim_id = claim.claim_id
                            G.add_node(
                                claim_id,
                                node_type="claim",
                                claim_type=getattr(claim.claim_type, 'value', str(claim.claim_type)),
                                status=getattr(claim.status, 'value', str(claim.status)),
                                confidence=claim.confidence
                            )
                            claim_info[claim_id] = {
                                "claim_type": getattr(claim.claim_type, 'value', str(claim.claim_type)),
                                "status": getattr(claim.status, 'value', str(claim.status)),
                                "confidence": claim.confidence
                            }
                
                if G.number_of_nodes() > 0:
                    # Layout algorithm - use hierarchical layout if evidence exists
                    claim_nodes = [node for node, attr in G.nodes(data=True) if attr.get("node_type") == "claim"]
                    evidence_nodes = [node for node, attr in G.nodes(data=True) if attr.get("node_type") == "evidence"]

                    # Build DOT representation (always available)
                    status_colors = {
                        "verified": "#28a745",
                        "low_confidence": "#ffc107",
                        "rejected": "#dc3545"
                    }
                    dot_lines = ["digraph ClaimEvidence {", "  rankdir=TB;", "  node [shape=box, style=rounded];"]

                    for node in claim_nodes:
                        node_data = G.nodes[node].get("data", {})
                        status = str(node_data.get("status", "rejected")).lower()
                        color = status_colors.get(status, "#6c757d")
                        label_type = str(node_data.get("claim_type", "claim")).split(".")[-1].replace("_", " ").title()
                        confidence = node_data.get("confidence", 0)
                        label = f"{label_type}\\n{confidence:.0%}"
                        dot_lines.append(f"  \"{node}\" [label=\"{label}\", color=\"{color}\"];" )

                    for node in evidence_nodes:
                        dot_lines.append(f"  \"{node}\" [label=\"Evidence\\n{node[:6]}\", shape=ellipse, color=\"#17a2b8\"];" )

                    for u, v in G.edges():
                        weight = G[u][v].get("weight", 1.0)
                        dot_lines.append(f"  \"{u}\" -> \"{v}\" [label=\"{weight:.2f}\"];" )

                    dot_lines.append("}")
                    dot_graph = "\n".join(dot_lines)

                    # GraphML + adjacency JSON export (always available)
                    graphml_data = None
                    graphml_error = None
                    try:
                        graphml_data = export_graphml_bytes(G)
                    except Exception as e:
                        graphml_error = e
                        logger.error(f"GraphML export failed: {e}")

                    adjacency_json = export_adjacency_json(G)

                    export_col1, export_col2, export_col3 = st.columns(3)
                    with export_col1:
                        st.download_button(
                            label="📥 Download Graph (DOT)",
                            data=dot_graph,
                            file_name=f"claim_graph_{st.session_state.result.get('output').session_id if st.session_state.result else 'unknown'}.dot",
                            mime="text/vnd.graphviz"
                        )
                    with export_col2:
                        if graphml_data:
                            st.download_button(
                                label="📥 Download Graph (GraphML)",
                                data=graphml_data,
                                file_name=f"claim_graph_{st.session_state.result.get('output').session_id if st.session_state.result else 'unknown'}.graphml",
                                mime="application/graphml+xml"
                            )
                        elif graphml_error:
                            st.error(f"GraphML export failed: {graphml_error}")
                        else:
                            st.warning("GraphML export unavailable for this graph.")
                    with export_col3:
                        st.download_button(
                            label="📥 Download Graph (Adjacency JSON)",
                            data=adjacency_json,
                            file_name=f"claim_graph_{st.session_state.result.get('output').session_id if st.session_state.result else 'unknown'}.json",
                            mime="application/json"
                        )

                    if HAS_MATPLOTLIB:
                        # Create visualization with better styling
                        fig, ax = plt.subplots(figsize=(14, 10))
                        fig.patch.set_facecolor('#f8f9fa')
                        ax.set_facecolor('#ffffff')
                        
                        if evidence_nodes:
                            # Hierarchical layout: claims on top, evidence below
                            pos = {}
                            for i, node in enumerate(claim_nodes):
                                pos[node] = (i * 2, 1)
                            for i, node in enumerate(evidence_nodes):
                                pos[node] = (i * 1.5, 0)
                        else:
                            # Circular layout for claims only
                            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
                        
                        # Color claims by status
                        claim_colors = []
                        claim_labels = {}
                        for node in claim_nodes:
                            node_data = G.nodes[node].get("data", {})
                            status = node_data.get("status", "rejected")
                            if isinstance(status, str):
                                status_lower = status.lower()
                            else:
                                status_lower = str(status).lower()
                            
                            claim_colors.append(status_colors.get(status_lower, "#6c757d"))
                            
                            # Create readable label
                            claim_type = node_data.get("claim_type", "claim")
                            if isinstance(claim_type, str):
                                claim_type = claim_type.split(".")[-1].replace("_", " ").title()
                            confidence = node_data.get("confidence", 0)
                            claim_labels[node] = f"{claim_type}\n{confidence:.0%}"
                        
                        # Draw claim nodes with status colors
                        if claim_nodes:
                            nx.draw_networkx_nodes(
                                G, pos, nodelist=claim_nodes, 
                                node_color=claim_colors,
                                node_size=2000, 
                                node_shape='o',
                                edgecolors='#333333',
                                linewidths=2,
                                ax=ax
                            )
                        
                        # Draw evidence nodes
                        if evidence_nodes:
                            nx.draw_networkx_nodes(
                                G, pos, nodelist=evidence_nodes, 
                                node_color="#17a2b8",
                                node_size=1200, 
                                node_shape="s",
                                edgecolors='#333333',
                                linewidths=2,
                                ax=ax,
                                alpha=0.8
                            )
                            
                            # Evidence labels
                            evidence_labels = {node: f"Evidence\n{node[:6]}" for node in evidence_nodes}
                            nx.draw_networkx_labels(G, pos, evidence_labels, font_size=8, font_weight='bold', ax=ax)
                        
                        # Draw edges with weights
                        if G.number_of_edges() > 0:
                            edge_weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
                            nx.draw_networkx_edges(
                                G, pos, 
                                edge_color='#666666', 
                                arrows=True,
                                arrowsize=15, 
                                arrowstyle='-|>',
                                width=[w * 2 for w in edge_weights],
                                ax=ax,
                                alpha=0.6,
                                connectionstyle='arc3,rad=0.1'
                            )
                        
                        # Draw claim labels
                        nx.draw_networkx_labels(G, pos, claim_labels, font_size=9, font_weight='bold', ax=ax)
                        
                        # Create informative title
                        if evidence_nodes:
                            title = f"📊 Claim-Evidence Network\n{len(claim_nodes)} Claims • {len(evidence_nodes)} Evidence Sources"
                        else:
                            title = f"📊 Extracted Claims Overview\n{len(claim_nodes)} Claims Identified"
                        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
                        
                        # Create legend
                        legend_elements = [
                            mpatches.Patch(color='#28a745', label='✓ Verified (High Confidence)'),
                            mpatches.Patch(color='#ffc107', label='⚠ Low Confidence'),
                            mpatches.Patch(color='#dc3545', label='✗ Rejected (No Evidence)')
                        ]
                        if evidence_nodes:
                            legend_elements.append(mpatches.Patch(color='#17a2b8', label='📄 Evidence Source'))
                        
                        ax.legend(handles=legend_elements, loc='upper left', fontsize=10, frameon=True, 
                                 fancybox=True, shadow=True, bbox_to_anchor=(0, 1))
                        
                        # Add explanation text
                        explanation = (
                            "Each circle represents a claim extracted from the AI output.\n"
                            "Colors indicate verification status based on evidence found in source materials."
                        )
                        if evidence_nodes:
                            explanation += "\nSquares are evidence sources supporting claims (arrows show connections)."
                        
                        ax.text(0.5, -0.05, explanation, 
                               transform=ax.transAxes, 
                               fontsize=9, 
                               ha='center',
                               style='italic',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
                        
                        ax.axis("off")
                        plt.tight_layout()
                        
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                        
                        # Add summary statistics below graph
                        col1, col2, col3 = st.columns(3)
                        verified_count = sum(1 for c in claim_colors if c == status_colors["verified"])
                        low_conf_count = sum(1 for c in claim_colors if c == status_colors["low_confidence"])
                        rejected_count = sum(1 for c in claim_colors if c == status_colors["rejected"])
                        
                        with col1:
                            st.metric("✓ Verified", verified_count, help="Claims with strong evidence support")
                        with col2:
                            st.metric("⚠ Low Confidence", low_conf_count, help="Claims with weak evidence")
                        with col3:
                            st.metric("✗ Rejected", rejected_count, help="Claims lacking sufficient evidence")
                    else:
                        st.warning(
                            "Matplotlib not installed. Install with pip install matplotlib to enable PNG rendering. "
                            f"Python: {sys.executable}"
                        )
                        st.info("Showing a DOT graph plus a readable edge list.")
                        st.write(
                            f"Claims: {len(claim_nodes)} • Evidence Sources: {len(evidence_nodes)} • "
                            f"Edges: {G.number_of_edges()}"
                        )

                        st.code(dot_graph, language="dot")

                        # Render a readable edge list as markdown
                        if G.number_of_edges() > 0:
                            st.markdown("**Edges (source → target, weight):**")
                            edge_rows = [
                                f"- {u} → {v} (w={G[u][v].get('weight', 1.0):.2f})"
                                for u, v in G.edges()
                            ]
                            st.markdown("\n".join(edge_rows[:100]))
                else:
                    st.info("❌ No claims or evidence available for graph visualization")
            except Exception as e:
                st.warning(f"Could not visualize graph: {e}")
                import traceback
                st.error(traceback.format_exc())
        else:
            st.info("Graph data not available")
        
        st.divider()
    
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
        if "💻 Local LLM (Ollama)" in provider_options:
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
            value=False,
            help=(
                "Enforces evidence-grounded, claim-based generation. "
                "Claims without sufficient evidence are rejected. "
                "Provides traceability and confidence estimates."
            )
        )

        if enable_verifiable_mode and not st.session_state.has_pyarrow and not st.session_state.pyarrow_warned:
            st.warning("pyarrow missing; install with pip install pyarrow. Dataframe rendering may be limited.")
            st.session_state.pyarrow_warned = True
        
        # Domain Profile selector (only shown in verifiable mode)
        domain_profile = "physics"  # Default
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
                options=["physics", "discrete_math", "algorithms"],
                format_func=lambda x: {
                    "physics": "⚛️ Physics (equations + units)",
                    "discrete_math": "🔢 Discrete Math (definitions + proofs)",
                    "algorithms": "💻 Algorithms (pseudocode + complexity)"
                }[x],
                help=(
                    "Domain profile controls validation rules:\n"
                    "• Physics: unit checking, equation validation\n"
                    "• Discrete Math: strict dependencies, proof-step rigor\n"
                    "• Algorithms: pseudocode checks, complexity analysis"
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
        
        st.caption("✓ 🔬 **Verifiability Graph** always generates in Research Mode")
        
        # Debug: Show current filter state
        enabled_sections = [k for k, v in st.session_state.output_filters.items() if v]
        disabled_sections = [k for k, v in st.session_state.output_filters.items() if not v]
        
        with st.expander("🔍 Filter Debug Info", expanded=False):
            st.caption("**Enabled sections:**")
            st.write(", ".join(enabled_sections) if enabled_sections else "None")
            st.caption("**Disabled sections:**")
            st.write(", ".join(disabled_sections) if disabled_sections else "None")
        
        # Diagnostics expander for troubleshooting
        with st.expander("🔧 System Diagnostics", expanded=False):
            st.caption("**Python Environment:**")
            st.code(sys.executable, language="text")
            
            st.caption("**Package Versions:**")
            try:
                import streamlit
                import pandas
                import networkx
                st.text(f"streamlit: {streamlit.__version__}")
                st.text(f"pandas: {pandas.__version__}")
                st.text(f"networkx: {networkx.__version__}")
                
                try:
                    import pyarrow
                    st.text(f"pyarrow: {pyarrow.__version__} ✅")
                except ImportError:
                    st.text("pyarrow: NOT INSTALLED ❌")
                    st.caption("Install: pip install pyarrow")
                
                try:
                    import matplotlib
                    st.text(f"matplotlib: {matplotlib.__version__} ✅")
                except ImportError:
                    st.text("matplotlib: NOT INSTALLED ❌")
                    st.caption("Install: pip install matplotlib")
                
                try:
                    import sentence_transformers
                    st.text(f"sentence-transformers: {sentence_transformers.__version__} ✅")
                except ImportError:
                    st.text("sentence-transformers: NOT INSTALLED ⚠️")
                    st.caption("Semantic verification unavailable")
                
            except Exception as e:
                st.error(f"Error fetching package info: {e}")
            
            st.caption("**Working Directory:**")
            st.code(str(Path.cwd()), language="text")
            
            if st.button("📋 Copy Diagnostic Info"):
                diag_info = f"""Python: {sys.executable}
Working Directory: {Path.cwd()}
Streamlit: {streamlit.__version__}
Pandas: {pandas.__version__}
NetworkX: {networkx.__version__}
PyArrow: {'Available' if st.session_state.has_pyarrow else 'Missing'}
"""
                st.code(diag_info)
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
            ["Type/Paste Text", "Upload Images"],
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
                "Upload note images",
                type=["jpg", "jpeg", "png", "bmp"],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )

            if notes_images:
                st.success(f"✓ {len(notes_images)} image(s) uploaded - OCR will extract text")

                if len(notes_images) <= 3:
                    cols = st.columns(len(notes_images))
                    for idx, (col, img) in enumerate(zip(cols, notes_images)):
                        with col:
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

        with st.expander("Session ID", expanded=False):
            session_id = st.text_input(
                "Custom session ID",
                value=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                label_visibility="collapsed"
            )

    st.divider()

    generate_button = st.button(
        "Generate",
        type="primary",
        use_container_width=True,
        key="generate_btn",
        disabled=(llm_type == "demo")
    )
    
    # Show message if button is disabled
    if llm_type == "demo":
        st.info("💡 To enable processing, set OPENAI_API_KEY or run Ollama locally")
    
    # Process when button clicked
    if generate_button:
        # Extract text from images if uploaded
        ocr_extracted_text = ""
        if notes_images and len(notes_images) > 0:
            with st.spinner("📸 Extracting text from images using OCR..."):
                try:
                    cache = _load_ocr_cache()
                    image_hashes = []
                    for img in notes_images:
                        img_bytes = img.getvalue()
                        image_hashes.append(_image_hash(img_bytes))

                    cache_key = "|".join(image_hashes)
                    if cache_key in cache["items"]:
                        ocr_extracted_text = cache["items"][cache_key]
                        st.success(f"✓ Using cached OCR: {len(ocr_extracted_text)} characters")
                    else:
                        selected_model = config.OLLAMA_MODEL if llm_type == "ollama" else config.LLM_MODEL
                        ocr_extracted_text = process_images(
                            notes_images, 
                            correct_with_llm=True,
                            provider_type=llm_type,
                            api_key=config.OPENAI_API_KEY,
                            ollama_url=config.OLLAMA_URL,
                            model=selected_model
                        )
                        cache["items"][cache_key] = ocr_extracted_text
                        cache["order"] = [k for k in cache["order"] if k != cache_key]
                        cache["order"].append(cache_key)
                        while len(cache["order"]) > 3:
                            old_key = cache["order"].pop(0)
                            cache["items"].pop(old_key, None)
                        _save_ocr_cache(cache)
                        st.success(f"✓ OCR extraction + LLM correction complete: {len(ocr_extracted_text)} characters")
                    
                    if ocr_extracted_text:
                        with st.expander("✏️ Review & Edit OCR Text", expanded=True):
                            st.info("💡 The text below has been corrected using AI to fix OCR errors. You can edit it before processing.")
                            ocr_extracted_text = st.text_area(
                                "Corrected OCR Output (editable)",
                                value=ocr_extracted_text,
                                height=250,
                                key="ocr_text_area"
                            )
                    else:
                        st.warning("⚠️ OCR extraction returned no text. The image may be empty or text unreadable.")
                        
                except Exception as e:
                    st.error(f"❌ OCR extraction failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    ocr_extracted_text = ""
        
        # Combine typed notes and OCR-extracted text
        combined_notes = notes_text
        if ocr_extracted_text:
            if combined_notes:
                combined_notes += "\n\n---\n\n" + ocr_extracted_text
            else:
                combined_notes = ocr_extracted_text
        
        # Validation - check if we have any input at all
        has_input = combined_notes or audio_file or (notes_images and len(notes_images) > 0)
        
        if not has_input:
            st.error("Please provide notes, images, or audio.")
        else:
            # Process the session and store in session state
            result, verifiable_metadata = process_session(
                audio_file=audio_file,
                notes_text=combined_notes,
                equations=equations,
                external_context=external_context,
                session_id=session_id,
                verifiable_mode=enable_verifiable_mode,
                domain_profile=domain_profile if enable_verifiable_mode else "physics",
                llm_provider_type=llm_type
            )
            
            # Store results in session state
            st.session_state.result = result
            st.session_state.verifiable_metadata = verifiable_metadata
            st.session_state.processing_complete = True
    
    # Display results if available (persists across tab switches)
    if st.session_state.processing_complete and st.session_state.result:
        st.success("✅ Processing complete!")
        st.divider()
        display_output(st.session_state.result, st.session_state.verifiable_metadata)
        
        # Export buttons
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

        QuickExportButtons.show_export_buttons(export_data, session_id_for_export)


if __name__ == "__main__":
    main()
