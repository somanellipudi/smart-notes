"""
Streamlit UI for Structured Understanding of Classroom Content.

This is a RESEARCH DEMO interface for demonstrating the GenAI framework
to advisors and reviewers. Not intended for production use.

Usage:
    streamlit run app.py
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
import tempfile
import hashlib

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
    from src.reasoning.pipeline import ReasoningPipeline
    from src.evaluation.metrics import evaluate_session_output
    from src.study_book.session_manager import SessionManager
    from src.llm_provider import LLMProviderFactory
    from src.output_formatter import StreamingOutputDisplay
    from src.reasoning.fallback_handler import FallbackGenerator, PipelineEnhancer
    from src.streamlit_display import StreamlitProgressDisplay, QuickExportButtons
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
    .block-container {padding-top: 2rem; max-width: 1100px;}
    h1, h2, h3 {font-weight: 600; letter-spacing: -0.02em;}
    .stButton>button {border-radius: 10px; padding: 0.6rem 1rem;}
    .stTextArea textarea {border-radius: 10px;}
    .stFileUploader, .stTextInput input {border-radius: 10px;}
    .stTabs [data-baseweb="tab"] {font-size: 0.95rem;}
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
    session_id: str
) -> dict:
    """
    Process classroom session through the complete pipeline.
    
    Args:
        audio_file: Uploaded audio file object (or None)
        notes_text: Handwritten notes text
        equations: Equations string (one per line)
        external_context: Reference material text
        session_id: Unique session identifier
    Returns:
        Dictionary with output, evaluation, and metadata
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
        if not config.OPENAI_API_KEY:
            st.error("❌ OPENAI_API_KEY is not set. Please add it to your .env file.")
            st.stop()

        pipeline = ReasoningPipeline()
        
        output = pipeline.process(
            combined_content=combined_text,
            equations=equations_list,
            external_context=external_context,
            session_id=session_id
        )
        
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
    
    return {
        "output": output,
        "evaluation": evaluation,
        "session_path": session_path,
        "transcript_length": len(transcript)
    }


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


def display_output(result: dict):
    """
    Display structured output using new streaming display system.
    
    Args:
        result: Result dictionary from process_session
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
                "title": t.title,
                "description": t.description,
                "importance": t.importance
            } for t in (output.topics or [])
        ],
        "concepts": [
            {
                "name": c.name,
                "definition": c.definition,
                "importance": c.importance
            } for c in (output.key_concepts or [])
        ],
        "equations": [
            {
                "equation": e.equation,
                "explanation": e.explanation
            } for e in (output.equations or [])
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
                "explanation": e.explanation
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
        st.success("✅ **PASSES** quality thresholds")
    else:
        st.warning("⚠️ **FAILS** quality thresholds")
    
    st.divider()
    
    # ========================================================================
    # CLASS SUMMARY
    # ========================================================================
    
    st.header("Summary")
    st.write(output.class_summary)
    st.divider()
    
    # ========================================================================
    # TOPICS
    # ========================================================================
    
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


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main Streamlit application."""
    
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
        
        if not provider_options:
            st.error("❌ No LLM providers available!\n\nPlease:\n1. Set OPENAI_API_KEY in .env\n2. Or run Ollama locally at http://localhost:11434")
            st.stop()
        
        selected_llm = st.radio(
            "Choose AI model:",
            provider_options,
            index=0,
            help="OpenAI uses cloud API. Local LLM runs on your machine (faster, free, private)."
        )
        
        llm_type = provider_map[selected_llm]
        
        if llm_type == "ollama":
            st.success("💻 Using Local LLM - Faster & Private!")
            st.caption("Running on your machine · No API calls")
        else:
            st.info("🌐 Using OpenAI - Higher quality")
            st.caption(f"Model: GPT-4")
        
        # Processing options
        st.divider()
        st.subheader("⚡ Processing")
        
        enable_streaming = st.checkbox("Stream results", value=True, help="Show results as they're generated")
        
        processing_depth = st.select_slider(
            "Processing depth",
            options=["Fast", "Balanced", "Thorough"],
            value="Balanced",
            help="Affects both quality and speed"
        )
        
        st.divider()
    
    # Title and description
    st.title("Smart Notes")
    st.markdown(
        "Transform classroom content into clean, structured study notes."
    )
    
    st.divider()
    
    # ====================================================================
    # INPUT
    # ====================================================================
    
    st.header("Input")
    st.caption("Provide notes or images. Audio is optional.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Notes")

        notes_input_method = st.radio(
            "Notes input",
            ["Type/Paste Text", "Upload Images"],
            horizontal=True
        )

        notes_text = ""
        notes_images = []

        if notes_input_method == "Type/Paste Text":
            notes_text = st.text_area(
                "Paste or type notes",
                height=200,
                placeholder="Combustion and Flame\n\n1. Combustion is a chemical process...",
            )
        else:
            notes_images = st.file_uploader(
                "Upload images",
                type=["jpg", "jpeg", "png", "bmp"],
                accept_multiple_files=True
            )

            if notes_images:
                st.info(f"{len(notes_images)} image(s) uploaded. OCR will extract text.")

                if len(notes_images) <= 3:
                    cols = st.columns(len(notes_images))
                    for idx, (col, img) in enumerate(zip(cols, notes_images)):
                        with col:
                            st.image(img, caption=f"Image {idx+1}", width="stretch")

        with st.expander("Audio (optional)", expanded=False):
            audio_file = st.file_uploader(
                "Upload audio",
                type=["wav", "mp3", "m4a"],
                help="Lecture recording to transcribe"
            )

    with col2:
        with st.expander("Advanced (optional)", expanded=False):
            equations = st.text_area(
                "Equations (one per line)",
                height=100
            )

            external_context = st.text_area(
                "External context",
                height=100
            )

            session_id = st.text_input(
                "Session ID",
                value=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

    st.divider()

    generate_button = st.button(
        "Generate",
        type="primary",
        use_container_width=True
    )
    
    # Process when button clicked
    if generate_button:
        # Extract text from images if uploaded
        ocr_extracted_text = ""
        if notes_images:
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
                    else:
                        ocr_extracted_text = process_images(notes_images, correct_with_llm=True)
                        cache["items"][cache_key] = ocr_extracted_text
                        cache["order"] = [k for k in cache["order"] if k != cache_key]
                        cache["order"].append(cache_key)
                        while len(cache["order"]) > 3:
                            old_key = cache["order"].pop(0)
                            cache["items"].pop(old_key, None)
                        _save_ocr_cache(cache)

                    st.success(f"✓ OCR extraction + LLM correction complete: {len(ocr_extracted_text)} characters")
                    with st.expander("✏️ Review & Edit OCR Text", expanded=True):
                        st.info("💡 The text below has been corrected using AI to fix OCR errors. You can edit it before processing.")
                        ocr_extracted_text = st.text_area(
                            "Corrected OCR Output (editable)",
                            value=ocr_extracted_text,
                            height=250,
                            key="ocr_text_area"
                        )
                except Exception as e:
                    st.error(f"❌ OCR extraction failed: {e}")
                    ocr_extracted_text = ""
        
        # Combine typed notes and OCR-extracted text
        combined_notes = notes_text
        if ocr_extracted_text:
            if combined_notes:
                combined_notes += "\n\n---\n\n" + ocr_extracted_text
            else:
                combined_notes = ocr_extracted_text
        
        # Validation
        if not combined_notes and not audio_file:
            st.error("Please provide notes, images, or audio.")
        else:
            # Process the session
            result = process_session(
                audio_file=audio_file,
                notes_text=combined_notes,
                equations=equations,
                external_context=external_context,
                session_id=session_id
            )
            
            # Display results
            st.success("✅ Processing complete!")
            st.divider()
            display_output(result)
            
            # Export buttons
            st.divider()
            QuickExportButtons.show_export_buttons(output_dict if 'output_dict' in locals() else {}, session_id)


if __name__ == "__main__":
    main()
