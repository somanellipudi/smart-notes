"""
Redesigned UX for Smart Notes - Clean, tab-based interface with prominent flow visualization.

Key improvements:
- Tab-based navigation (Input → Configure → Process → Results)
- Clean, uncluttered interface
- Prominent pipeline flow visualization
- Better organization of settings
- Progressive disclosure (show what matters when it matters)
"""

import streamlit as st
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
import logging

from src.ui.progress_tracker import RunContext, create_run_context
from src.ui.execution_flow_ui import (
    render_execution_flow_dashboard,
    render_engines_used_panel,
    render_pipeline_stepper
)

logger = logging.getLogger(__name__)


def get_input_summary():
    """Get summary of current inputs for validation."""
    notes_text = st.session_state.get("ui_notes_text", "")
    notes_images = st.session_state.get("ui_notes_images", [])
    audio_file = st.session_state.get("ui_audio_file")
    urls_text = st.session_state.get("ui_urls_text", "")
    equations = st.session_state.get("ui_equations", "")
    
    has_text = len(notes_text.strip()) > 0
    has_files = notes_images and len(notes_images) > 0
    has_audio = audio_file is not None
    has_urls = len(urls_text.strip()) > 0
    has_equations = len(equations.strip()) > 0
    
    return {
        "has_input": has_text or has_files or has_audio or has_urls,
        "text_chars": len(notes_text.strip()),
        "files_count": len(notes_images) if notes_images else 0,
        "has_audio": has_audio,
        "urls_count": len([u for u in urls_text.split('\n') if u.strip()]),
        "has_equations": has_equations
    }


def initialize_session_state():
    """Initialize all required session state variables."""
    defaults = {
        "active_tab": 0,
        "ui_verifiable_mode": False,
        "ui_notes_text": "",
        "ui_notes_images": None,
        "ui_audio_file": None,
        "ui_urls_text": "",
        "ui_equations": "",
        "ui_external_context": "",
        "ui_llm_provider": "openai",
        "ui_model": "gpt-4o-mini",
        "ui_temperature": 0.3,
        "ui_debug": False,
        "ui_auto_save": True,
        "ui_num_topics": 5,
        "ui_num_concepts": 5,
        "ui_show_summary": True,
        "ui_show_concepts": True,
        "ui_show_mcqs": True,
        "ui_show_flashcards": True,
        "ui_export_md": True,
        "ui_export_html": True,
        "ui_export_json": False,
        "ui_min_evidence": 2,
        "ui_min_confidence": 0.7,
        "ui_top_k": 10,
        "ui_use_reranker": True,
        "ui_flag_contradictions": True,
        "ui_show_sources": True,
        "ui_verifiable_mode": False,  # Default to Fast mode
        "show_session_loader": False,
        "show_detailed_report": False,
        "trigger_save": False,
        "trigger_processing": False,
        "current_output": None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header():
    """Render the main header with mode switcher and session controls."""
    # Title row
    col_title, col_mode = st.columns([2, 2])
    
    with col_title:
        st.markdown("# 📘 Smart Notes")
        st.caption("AI-powered research notes with evidence validation")
    
    with col_mode:
        st.markdown("")
        # Mode switcher - aligned right
        mode_col1, mode_col2 = st.columns([1, 2])
        with mode_col2:
            # Set initial index based on session state
            initial_mode = "🔬 Verifiable" if st.session_state.get("ui_verifiable_mode", False) else "⚡ Fast"
            mode = st.radio(
                "Processing Mode",
                ["⚡ Fast", "🔬 Verifiable"],
                index=0 if initial_mode == "⚡ Fast" else 1,
                horizontal=True,
                help="Fast: Quick notes generation | Verifiable: Research-grade with citations",
                label_visibility="collapsed",
                key="header_processing_mode"
            )
            verifiable_mode = mode == "🔬 Verifiable"
            st.session_state.ui_verifiable_mode = verifiable_mode
    
    # Session controls row
    st.markdown("")
    col_btn1, col_btn2, col_spacer = st.columns([1, 1, 2])
    
    with col_btn1:
        if st.button("📂 Load Session", use_container_width=True, key="load_session_header"):
            st.session_state.show_session_loader = True
    
    with col_btn2:
        if st.button("💾 Quick Save", use_container_width=True, key="save_session_header"):
            st.session_state.trigger_save = True


def render_sidebar_settings():
    """Render sidebar with essential settings and status."""
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # LLM Provider
        llm_provider = st.selectbox(
            "LLM Provider",
            ["openai", "ollama", "mixed"],
            help="OpenAI (GPT-4), Ollama (local), or Mixed"
        )
        
        # Model selection
        if llm_provider == "openai":
            model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"])
        elif llm_provider == "ollama":
            model = st.text_input("Ollama Model", "llama3.2:latest")
        else:
            model = "mixed"
        
        st.session_state.ui_llm_provider = llm_provider
        st.session_state.ui_model = model
        
        # Temperature
        temperature = st.slider(
            "Temperature",
            0.0, 1.0, 0.3, 0.1,
            help="Lower = more focused, Higher = more creative"
        )
        st.session_state.ui_temperature = temperature
        
        st.divider()
        
        # Quick toggles
        st.markdown("### 🎛️ Toggles")
        debug = st.checkbox("Debug Mode", False)
        auto_save = st.checkbox("Auto-save Sessions", True)
        
        st.session_state.ui_debug = debug
        st.session_state.ui_auto_save = auto_save
        
        st.divider()
        
        # Session info
        st.markdown("### 📊 Session Info")
        if "session_id" in st.session_state:
            st.caption(f"**ID:** `{st.session_state.session_id}`")
        
        if "run_context" in st.session_state:
            run_context = st.session_state.run_context
            progress = run_context.get_progress_fraction()
            st.progress(progress, text=f"{int(progress * 100)}% Complete")
            
            # Show current stage
            for stage_key in run_context.STAGE_ORDER:
                stage = run_context.stage_events[stage_key]
                if stage.status.value == "running":
                    st.caption(f"{stage.status.icon()} {stage.stage_name}")
                    break


def render_input_tab():
    """Tab 1: Clean input collection interface."""
    st.markdown("### 📥 Provide Your Content")
    st.caption("Add notes, media, or external sources to process")
    st.markdown("")
    
    # Create a clean 2-column layout
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("#### Primary Content")
        
        # Text input
        with st.container():
            st.markdown("**📝 Text Input**")
            notes_text = st.text_area(
                "Type or paste your notes",
                height=250,
                placeholder="Example:\n\nCombustion and Flame\n\n1. Combustion is a chemical process...\n2. Types of flames...",
                help="Paste lecture notes, textbook content, or any study material",
                label_visibility="collapsed"
            )
            st.session_state.ui_notes_text = notes_text
        
        st.markdown("")  # Spacing
        
        # File upload
        with st.container():
            st.markdown("**📄 Files (Images, PDFs)**")
            notes_images = st.file_uploader(
                "Upload files",
                type=["jpg", "jpeg", "png", "bmp", "pdf"],
                accept_multiple_files=True,
                help="Images will be OCR'd, PDFs will be extracted",
                label_visibility="collapsed"
            )
            st.session_state.ui_notes_images = notes_images
            
            if notes_images:
                st.success(f"✓ {len(notes_images)} file(s) ready for processing")
    
    with col2:
        st.markdown("#### Additional Sources")
        
        # Audio
        with st.expander("🎤 Audio Recording", expanded=False):
            audio_file = st.file_uploader(
                "Upload lecture audio",
                type=["wav", "mp3", "m4a"],
                help="Will be transcribed automatically",
                label_visibility="collapsed"
            )
            st.session_state.ui_audio_file = audio_file
            if audio_file:
                st.audio(audio_file)
        
        # URLs
        with st.expander("🌐 External URLs", expanded=False):
            st.caption("YouTube videos or web articles")
            urls_text = st.text_area(
                "URLs (one per line)",
                height=120,
                placeholder="https://www.youtube.com/watch?v=...\nhttps://example.com/article",
                label_visibility="collapsed"
            )
            st.session_state.ui_urls_text = urls_text
        
        # Equations
        with st.expander("🧮 Equations", expanded=False):
            equations = st.text_area(
                "Mathematical equations (one per line)",
                height=100,
                placeholder="E=mc²\nF=ma\na² + b² = c²",
                label_visibility="collapsed"
            )
            st.session_state.ui_equations = equations
        
        # External context
        with st.expander("📚 Reference Material", expanded=False):
            external_context = st.text_area(
                "Additional context or guidelines",
                height=100,
                placeholder="Textbook excerpts, curriculum guidelines, etc.",
                label_visibility="collapsed"
            )
            st.session_state.ui_external_context = external_context
    
    st.divider()
    
    # Input status summary
    input_summary = get_input_summary()
    
    if input_summary["has_input"]:
        col_status1, col_status2, col_status3 = st.columns(3)
        
        with col_status1:
            st.metric("Text Characters", f"{input_summary['text_chars']:,}")
        with col_status2:
            st.metric("Files", input_summary['files_count'])
        with col_status3:
            status_items = []
            if input_summary['has_audio']:
                status_items.append("🎤 Audio")
            if input_summary['urls_count'] > 0:
                status_items.append(f"🌐 {input_summary['urls_count']} URLs")
            if input_summary['has_equations']:
                status_items.append("🧮 Equations")
            
            if status_items:
                st.caption("**Extras:**")
                for item in status_items:
                    st.caption(item)
        
        st.success("✓ Ready to process")
    else:
        st.info("💡 Add at least one content source to begin")
    
    st.divider()
    
    # Action buttons
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        if st.button("➡️ Next: Configure", type="primary", use_container_width=True):
            st.session_state.active_tab = 1
            st.rerun()
    
    with col_btn2:
        if st.button("🚀 Quick Process", use_container_width=True,
                     help="Process with default settings (uses mode selected above)",
                     type="secondary"):
            # Validate and trigger processing with defaults
            from src.ui.processing_integration import check_processing_requirements, trigger_processing
            
            valid, message = check_processing_requirements(
                notes_text=st.session_state.get("ui_notes_text", ""),
                notes_images=st.session_state.get("ui_notes_images"),
                audio_file=st.session_state.get("ui_audio_file"),
                urls_text=st.session_state.get("ui_urls_text", "")
            )
            
            if not valid:
                st.error(message)
            else:
                # Trigger with default settings (respects header mode selection)
                trigger_processing(
                    notes_text=st.session_state.get("ui_notes_text", ""),
                    notes_images=st.session_state.get("ui_notes_images"),
                    audio_file=st.session_state.get("ui_audio_file"),
                    urls_text=st.session_state.get("ui_urls_text", ""),
                    equations=st.session_state.get("ui_equations", ""),
                    external_context=st.session_state.get("ui_external_context", ""),
                    verifiable_mode=st.session_state.get("ui_verifiable_mode", False),  # Use header mode
                    num_topics=5,
                    num_concepts=5,
                    llm_provider=st.session_state.get("llm_provider", "openai"),
                    model=st.session_state.get("model_name", "gpt-4o-mini"),
                    temperature=0.3
                )
                st.session_state.active_tab = 2  # Switch to Process tab
                st.rerun()
    
    with col_btn3:
        mode_indicator = "🔬 Verifiable" if st.session_state.get("ui_verifiable_mode", False) else "⚡ Fast"
        st.caption(f"💡 Tip: Quick process uses {mode_indicator} mode")


def render_configure_tab():
    """Tab 2: Configuration and settings interface."""
    st.markdown("### ⚙️ Configure Processing")
    st.caption("Customize how your content will be processed")
    st.markdown("")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Output Configuration")
        
        with st.container():
            st.markdown("**Topics & Concepts**")
            num_topics = st.slider("Number of Topics", 3, 15, 5)
            num_concepts = st.slider("Concepts per Topic", 3, 10, 5)
            st.session_state.ui_num_topics = num_topics
            st.session_state.ui_num_concepts = num_concepts
        
        st.markdown("")
        
        with st.container():
            st.markdown("**Output Sections**")
            show_summary = st.checkbox("Study Guide Summary", True)
            show_concepts = st.checkbox("Concept Definitions", True)
            show_mcqs = st.checkbox("Practice MCQs", True)
            show_flashcards = st.checkbox("Flashcards", True)
            
            st.session_state.ui_show_summary = show_summary
            st.session_state.ui_show_concepts = show_concepts
            st.session_state.ui_show_mcqs = show_mcqs
            st.session_state.ui_show_flashcards = show_flashcards
        
        st.markdown("")
        
        with st.container():
            st.markdown("**Export Options**")
            export_md = st.checkbox("Markdown Report", True)
            export_html = st.checkbox("HTML Report", True)
            export_json = st.checkbox("JSON Metadata", False)
            
            st.session_state.ui_export_md = export_md
            st.session_state.ui_export_html = export_html
            st.session_state.ui_export_json = export_json
    
    with col2:
        st.markdown("#### 🔬 Verification Settings")
        
        verifiable_mode = st.session_state.get("ui_verifiable_mode", False)
        
        if not verifiable_mode:
            st.info("💡 Enable **Verifiable Mode** in the header to use research-grade verification")
        else:
            with st.container():
                st.markdown("**Evidence Requirements**")
                min_evidence = st.slider("Min Evidence per Claim", 1, 5, 2)
                min_confidence = st.slider("Min Confidence Score", 0.0, 1.0, 0.7, 0.05)
                
                st.session_state.ui_min_evidence = min_evidence
                st.session_state.ui_min_confidence = min_confidence
            
            st.markdown("")
            
            with st.container():
                st.markdown("**Retrieval Settings**")
                top_k = st.slider("Top-K Passages", 3, 20, 10)
                use_reranker = st.checkbox("Use Reranker", True)
                
                st.session_state.ui_top_k = top_k
                st.session_state.ui_use_reranker = use_reranker
            
            st.markdown("")
            
            with st.container():
                st.markdown("**Quality Controls**")
                flag_contradictions = st.checkbox("Flag Contradictions", True)
                show_sources = st.checkbox("Show Evidence Sources", True)
                
                st.session_state.ui_flag_contradictions = flag_contradictions
                st.session_state.ui_show_sources = show_sources
    
    st.divider()
    
    # Navigation
    col_back, col_process = st.columns([1, 1])
    
    with col_back:
        if st.button("⬅️ Back to Input", use_container_width=True, type="secondary"):
            st.session_state.active_tab = 0
            st.rerun()
    
    with col_process:
        if st.button("🚀 Start Processing", type="primary", use_container_width=True):
            # Validate inputs
            from src.ui.processing_integration import check_processing_requirements, trigger_processing
            
            valid, message = check_processing_requirements(
                notes_text=st.session_state.get("ui_notes_text", ""),
                notes_images=st.session_state.get("ui_notes_images"),
                audio_file=st.session_state.get("ui_audio_file"),
                urls_text=st.session_state.get("ui_urls_text", "")
            )
            
            if not valid:
                st.error(message)
            else:
                # Trigger processing
                if trigger_processing(
                    notes_text=st.session_state.get("ui_notes_text", ""),
                    notes_images=st.session_state.get("ui_notes_images"),
                    audio_file=st.session_state.get("ui_audio_file"),
                    urls_text=st.session_state.get("ui_urls_text", ""),
                    equations=st.session_state.get("ui_equations", ""),
                    external_context=st.session_state.get("ui_external_context", ""),
                    verifiable_mode=st.session_state.get("ui_verifiable_mode", False),
                    num_topics=st.session_state.get("ui_num_topics", 5),
                    num_concepts=st.session_state.get("ui_num_concepts_per_topic", 5),
                    llm_provider=st.session_state.get("llm_provider", "openai"),
                    model=st.session_state.get("model_name", "gpt-4o-mini"),
                    temperature=st.session_state.get("temperature", 0.3)
                ):
                    # Switch to Process tab
                    st.session_state.active_tab = 2
                    st.rerun()


def render_process_tab():
    """Tab 3: Live pipeline flow visualization - THE MOST IMPORTANT TAB."""
    st.markdown("### 🚀 Processing Pipeline")
    st.caption("Real-time view of the AI pipeline with full transparency")
    st.markdown("")
    
    # Check if we have a run context
    if "run_context" not in st.session_state:
        # Show a helpful message and pipeline preview
        st.info(
            "💡 **Ready to See the Pipeline in Action?**\n\n"
            "Start processing from the **Input** or **Configure** tabs to see real-time "
            "pipeline visualization here. The system will track all 10 stages with "
            "detailed metrics, timing, and engine transparency.",
            icon="📊"
        )
        
        st.markdown("#### Pipeline Stages Overview")
        
        stages_preview = [
            ("1️⃣", "Inputs Received", "Collect and validate all input sources"),
            ("2️⃣", "Ingestion & Cleaning", "Extract and clean text from all sources"),
            ("3️⃣", "OCR Extraction", "Process scanned images/PDFs (if enabled)"),
            ("4️⃣", "Chunking & Provenance", "Split content with source tracking"),
            ("5️⃣", "Embedding & Indexing", "Create vector embeddings for retrieval"),
            ("6️⃣", "LLM Generation", "Generate study notes with AI"),
            ("7️⃣", "Claim Extraction", "Extract verifiable claims (research mode)"),
            ("8️⃣", "Retrieval & Reranking", "Find supporting evidence (research mode)"),
            ("9️⃣", "Verification", "Score claims against evidence (research mode)"),
            ("🔟", "Reporting & Exports", "Generate final outputs and save session"),
        ]
        
        for emoji, name, desc in stages_preview:
            col1, col2 = st.columns([0.15, 2])
            with col1:
                st.markdown(f"# {emoji}")
            with col2:
                st.markdown(f"**{name}**")
                st.caption(desc)
            st.markdown("")
        
        st.divider()
        
        # Navigation to start processing
        col_nav1, col_nav2 = st.columns(2)
        with col_nav1:
            if st.button("📥 Go to Input", use_container_width=True, type="secondary"):
                st.session_state.active_tab = 0
                st.rerun()
        with col_nav2:
            if st.button("⚙️ Go to Configure", use_container_width=True, type="secondary"):
                st.session_state.active_tab = 1
                st.rerun()
        
        return
    
    run_context: RunContext = st.session_state.run_context
    
    # Hero metrics at the top
    progress = run_context.get_progress_fraction()
    completed = sum(1 for s in run_context.stage_events.values() if s.status.value == "completed")
    total = len(run_context.STAGE_ORDER)
    total_time = run_context.get_total_duration()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Pipeline Progress", f"{int(progress * 100)}%", 
                  delta=None, delta_color="normal")
    
    with col2:
        st.metric("Stages Complete", f"{completed}/{total}")
    
    with col3:
        st.metric("Total Time", f"{total_time:.1f}s" if total_time > 0 else "0.0s")
    
    with col4:
        if progress >= 1.0:
            st.success("✅ Complete", icon="✅")
        elif progress > 0:
            st.info("⏳ Processing...", icon="⏳")
        else:
            st.caption("⬜ Initializing...")
    
    st.divider()
    
    # Show engine transparency at the top
    st.markdown("#### 🔧 Engines & Models")
    st.caption("What's powering your processing")
    render_engines_used_panel(run_context)
    
    st.divider()
    
    # Main pipeline visualization
    st.markdown("#### 📊 Stage-by-Stage Progress")
    st.caption("Real-time status with expandable details")
    
    # Render the pipeline stepper with full details
    render_pipeline_stepper(run_context, show_details=True)
    
    st.divider()
    
    # Navigation
    col_nav1, col_nav2 = st.columns(2)
    
    with col_nav1:
        if st.button("⬅️ Back to Configure", use_container_width=True, type="secondary"):
            st.session_state.active_tab = 1
            st.rerun()
    
    with col_nav2:
        if progress >= 1.0:
            if st.button("📄 View Results →", type="primary", use_container_width=True):
                st.session_state.active_tab = 3
                st.rerun()
        else:
            # Show option to watch in classic UI if processing is ongoing
            if progress > 0 and not st.session_state.get("use_redesigned_ui"):
                if st.button("👁️ Watch in Classic UI", use_container_width=True, type="secondary"):
                    st.session_state.use_redesigned_ui = False
                    st.rerun()
            else:
                st.caption("⏳ Processing in progress...")


def render_results_tab():
    """Tab 4: Clean results display with filtering."""
    st.markdown("### 📄 Results")
    st.markdown("")
    
    # Check if we have results from the classic UI
    if "result" not in st.session_state or not st.session_state.result:
        st.info(
            "⏳ **No Results Yet**\n\n"
            "Process your content first to see results here. "
            "Go to the **Input** tab to add content, then start processing.",
            icon="📋"
        )
        
        col_nav1, col_nav2 = st.columns(2)
        with col_nav1:
            if st.button("📥 Go to Input", use_container_width=True, type="primary"):
                st.session_state.active_tab = 0
                st.rerun()
        with col_nav2:
            if "run_context" in st.session_state:
                if st.button("🚀 View Pipeline", use_container_width=True, type="secondary"):
                    st.session_state.active_tab = 2
                    st.rerun()
        
        return
    
    # Get the output dictionary (stored by processing_integration.py)
    output_data = st.session_state.get("current_output", {})
    output_obj = output_data.get("output") if output_data else None
    verifiable_metadata = st.session_state.get("verifiable_metadata")
    
    # Show success message (or warning if verification failed)
    if verifiable_metadata and verifiable_metadata.get("status") == "UNVERIFIABLE_INPUT":
        st.warning("⚠️ Processing completed with warnings. See details below.", icon="📋")
    else:
        st.success("✅ Processing complete! Your study notes are ready.", icon="🎉")
    st.markdown("")
    
    # Show verifiability report if available (including error cases)
    if verifiable_metadata and (verifiable_metadata.get("verifiable_mode") or verifiable_metadata.get("status")):
        st.markdown("### 🔬 Verifiability Report")
        
        # Check for special status (like UNVERIFIABLE_INPUT)
        status = verifiable_metadata.get("status")
        if status == "UNVERIFIABLE_INPUT":
            st.error(
                "❌ **Input Too Short for Verification**\n\n"
                f"The content provided is too short to extract and verify claims.\n\n"
                f"**Required:** {verifiable_metadata.get('minimum_required', 100)} characters\n"
                f"**Provided:** {verifiable_metadata.get('text_length', 0)} characters\n\n"
                f"**Suggestion:** {verifiable_metadata.get('suggestion', 'Provide more detailed content.')}",
                icon="📏"
            )
            
            # Show quality report details if available
            quality_report = verifiable_metadata.get("quality_report", {})
            if quality_report:
                with st.expander("📊 Quality Details", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Alphabetic %", f"{quality_report.get('alphabetic_ratio', 0):.1%}")
                    with col2:
                        st.metric("CID Glyphs %", f"{quality_report.get('cid_ratio', 0):.1%}")
                    with col3:
                        st.metric("Printable %", f"{quality_report.get('printable_ratio', 0):.1%}")
                    
                    failure_reasons = verifiable_metadata.get("failure_reasons", [])
                    if failure_reasons:
                        st.markdown("**Issues Detected:**")
                        for reason in failure_reasons:
                            st.markdown(f"- {reason}")
            
            st.divider()
            return
        
        # Normal metrics display
        metrics = verifiable_metadata.get("metrics", {})
        total_claims = metrics.get("total_claims", 0)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Claims", total_claims)
        with col2:
            st.metric("Verified", metrics.get("verified_claims", 0))
        with col3:
            st.metric("Rejected", metrics.get("rejected_claims", 0))
        with col4:
            avg_conf = metrics.get("avg_confidence", 0)
            st.metric("Avg Confidence", f"{avg_conf:.2f}" if avg_conf > 0 else "N/A")
        
        # Show warning if no claims (but not UNVERIFIABLE_INPUT)
        if total_claims == 0 and not status:
            st.warning(
                "⚠️ **No claims were extracted**\n\n"
                "Possible reasons:\n"
                "- The content doesn't contain factual claims or assertions\n"
                "- The text is primarily procedural/narrative without verifiable facts\n"
                "- The claim extraction model didn't identify any extractable claims\n\n"
                "**Try:**\n"
                "- Add more specific facts, numbers, or technical details\n"
                "- Include equations, definitions, or scientific concepts\n"
                "- Provide more detailed explanations with concrete examples",
                icon="🔍"
            )
        
        st.divider()
    
    # Output filtering controls
    st.markdown("### 📋 Study Notes Content")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        show_summary = st.checkbox("📋 Summary", True, key="filter_summary")
    with col2:
        show_topics = st.checkbox("📚 Topics", True, key="filter_topics")
    with col3:
        show_concepts = st.checkbox("💡 Concepts", True, key="filter_concepts")
    with col4:
        show_equations = st.checkbox("📐 Equations", True, key="filter_equations")
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        show_examples = st.checkbox("🎯 Examples", True, key="filter_examples")
    with col6:
        show_mistakes = st.checkbox("⚠️ Mistakes", True, key="filter_mistakes")
    with col7:
        show_faqs = st.checkbox("❓ FAQs", True, key="filter_faqs")
    with col8:
        show_connections = st.checkbox("🌐 Connections", True, key="filter_connections")
    
    st.divider()
    
    # Display the structured output
    if output_obj:
        # Summary
        if show_summary and hasattr(output_obj, 'class_summary') and output_obj.class_summary:
            st.markdown("## 📋 Summary")
            st.write(output_obj.class_summary)
            st.divider()
        
        # Topics
        if show_topics and hasattr(output_obj, 'topics') and output_obj.topics:
            st.markdown(f"## 📚 Topics ({len(output_obj.topics)})")
            for i, topic in enumerate(output_obj.topics, 1):
                with st.expander(f"**{i}. {topic.name}**", expanded=(i == 1)):
                    st.write("**Summary:**", topic.summary)
                    if hasattr(topic, 'subtopics') and topic.subtopics:
                        st.write("**Subtopics:**")
                        for subtopic in topic.subtopics:
                            st.write(f"  • {subtopic}")
            st.divider()
        
        # Key Concepts
        if show_concepts and hasattr(output_obj, 'key_concepts') and output_obj.key_concepts:
            st.markdown(f"## 💡 Key Concepts ({len(output_obj.key_concepts)})")
            for i, concept in enumerate(output_obj.key_concepts, 1):
                difficulty = getattr(concept, 'difficulty_level', 3)
                with st.expander(f"**{i}. {concept.name}** (Difficulty: {difficulty}/5)", expanded=(i == 1)):
                    st.write("**Definition:**", concept.definition)
                    if hasattr(concept, 'prerequisites') and concept.prerequisites:
                        st.write("**Prerequisites:**", ", ".join(concept.prerequisites))
            st.divider()
        
        # Equations
        if show_equations and hasattr(output_obj, 'equation_explanations') and output_obj.equation_explanations:
            st.markdown(f"## 📐 Equations ({len(output_obj.equation_explanations)})")
            for i, eq in enumerate(output_obj.equation_explanations, 1):
                with st.expander(f"**{i}. {eq.equation}**", expanded=(i == 1)):
                    st.write("**Explanation:**", eq.explanation)
                    if hasattr(eq, 'variables') and eq.variables:
                        st.write("**Variables:**")
                        for var, desc in eq.variables.items():
                            st.write(f"  • **{var}**: {desc}")
            st.divider()
        
        # Worked Examples
        if show_examples and hasattr(output_obj, 'worked_examples') and output_obj.worked_examples:
            st.markdown(f"## 🎯 Worked Examples ({len(output_obj.worked_examples)})")
            for i, example in enumerate(output_obj.worked_examples, 1):
                with st.expander(f"**Example {i}**", expanded=(i == 1)):
                    st.write("**Problem:**")
                    st.write(example.problem)
                    st.write("**Solution:**")
                    st.write(example.solution)
            st.divider()
        
        # Common Mistakes
        if show_mistakes and hasattr(output_obj, 'common_mistakes') and output_obj.common_mistakes:
            st.markdown(f"## ⚠️ Common Mistakes ({len(output_obj.common_mistakes)})")
            for i, mistake in enumerate(output_obj.common_mistakes, 1):
                with st.expander(f"**Misconception {i}**", expanded=(i == 1)):
                    st.error(f"**Incorrect:** {mistake.misconception}")
                    st.write(f"**Why it's wrong:** {mistake.explanation}")
                    st.success(f"**Correct understanding:** {mistake.correct_understanding}")
            st.divider()
        
        # FAQs
        if show_faqs and hasattr(output_obj, 'faqs') and output_obj.faqs:
            st.markdown(f"## ❓ FAQs ({len(output_obj.faqs)})")
            for i, faq in enumerate(output_obj.faqs, 1):
                difficulty_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(faq.difficulty, "⚪")
                with st.expander(f"{difficulty_emoji} **Q{i}: {faq.question}**", expanded=(i == 1)):
                    st.write("**Answer:**", faq.answer)
            st.divider()
        
        # Real-world Connections
        if show_connections and hasattr(output_obj, 'real_world_connections') and output_obj.real_world_connections:
            st.markdown(f"## 🌐 Real-World Connections ({len(output_obj.real_world_connections)})")
            for i, conn in enumerate(output_obj.real_world_connections, 1):
                with st.expander(f"**{i}. {conn.concept}**", expanded=(i == 1)):
                    st.write("**Application:**", conn.application)
                    st.write("**Description:**", conn.description)
                    st.write("**Relevance:**", conn.relevance)
            st.divider()
    
    elif "markdown" in output_data:
        # Fallback to markdown display
        st.markdown(output_data["markdown"])
        st.divider()
    
    else:
        st.info("⏳ Processing not yet complete. Start processing from the **Input** or **Configure** tabs.")
    
    st.divider()
    
    # Export buttons
    st.markdown("#### 💾 Export & Actions")
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        # Download Markdown
        if output_data and "markdown" in output_data:
            session_id = st.session_state.get("session_id", "session")
            st.download_button(
                label="📄 Download MD",
                data=output_data["markdown"],
                file_name=f"study_notes_{session_id}.md",
                mime="text/markdown",
                use_container_width=True
            )
        else:
            st.button("📄 Download MD", disabled=True, use_container_width=True)
    
    with col_exp2:
        # Download HTML
        if output_data and "markdown" in output_data:
            session_id = st.session_state.get("session_id", "session")
            # Simple HTML wrapper without markdown conversion
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Study Notes</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
               max-width: 800px; margin: 40px auto; padding: 20px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #1a1a1a; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background: #f4f4f4; padding: 12px; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; }}
    </style>
</head>
<body>
<pre style="white-space: pre-wrap; font-family: inherit;">{output_data["markdown"]}</pre>
</body>
</html>"""
            st.download_button(
                label="🌐 Download HTML",
                data=html_content,
                file_name=f"study_notes_{session_id}.html",
                mime="text/html",
                use_container_width=True
            )
        else:
            st.button("🌐 Download HTML", disabled=True, use_container_width=True)
    
    with col_exp3:
        # View detailed reports
        if verifiable_metadata and (verifiable_metadata.get("verifiable_mode") or verifiable_metadata.get("status")):
            if st.button("📊 View Detailed Report", use_container_width=True):
                st.session_state.show_detailed_report = True
                st.rerun()
        else:
            st.button("📊 View Reports", disabled=True, use_container_width=True, 
                     help="Enable Verifiable mode in Configure tab to generate reports")
    
    # Show detailed verifiability report if requested
    if st.session_state.get("show_detailed_report") and verifiable_metadata:
        st.divider()
        st.markdown("### 📊 Detailed Verifiability Report")
        
        # Check if using cited generation mode
        is_cited_mode = verifiable_metadata.get("mode") == "cited_generation"
        if is_cited_mode:
            st.info("🚀 **Fast Cited Generation Mode**: Content generated with inline source citations (5-10x faster)")
        
        # Show quality report if available
        quality_report = verifiable_metadata.get("quality_report")
        if quality_report:
            with st.expander("🔍 Content Quality Analysis", expanded=True):
                extraction = quality_report.get("extraction_count", {})
                evidence_cov = quality_report.get("evidence_coverage", {})
                richness = quality_report.get("content_richness", {})
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Topics Extracted", extraction.get("topics", 0))
                    st.caption(extraction.get("status", ""))
                with col2:
                    st.metric("Concepts Extracted", extraction.get("concepts", 0))
                    expected = extraction.get("expected_minimum", 10)
                    st.caption(f"Expected: {expected}+ for rich content")
                with col3:
                    coverage = evidence_cov.get("coverage_rate", 0)
                    st.metric("Evidence Coverage", f"{coverage:.0f}%")
                    st.caption(f"{evidence_cov.get('with_sources', 0)}/{evidence_cov.get('with_sources', 0) + evidence_cov.get('without_sources', 0)} verified")
                
                # Show recommendations if any
                recommendations = quality_report.get("recommendations", [])
                if recommendations:
                    st.warning("**💡 Quality Recommendations:**")
                    for i, rec in enumerate(recommendations, 1):
                        st.markdown(f"{i}. {rec}")
        
        with st.expander("📈 Verification Metrics", expanded=True):
            metrics = verifiable_metadata.get("metrics", {})
            
            col1, col2 = st.columns(2)
            with col1:
                verified_count = metrics.get("verified_claims", 0)
                st.metric(
                    "Verified Claims" if not is_cited_mode else "Verified Concepts",
                    verified_count,
                    help="Concepts extracted and verified with online sources" if is_cited_mode else None
                )
                st.metric("Low Confidence", metrics.get("low_confidence_claims", 0))
            with col2:
                st.metric("Rejected Claims", metrics.get("rejected_claims", 0))
                rejection_rate = metrics.get("rejection_rate", 0)
                if is_cited_mode:
                    st.metric("Verification Rate", f"{metrics.get('verification_rate', 1.0):.1%}",
                             help="Percentage of citations verified against online evidence")
                else:
                    st.metric("Rejection Rate", f"{rejection_rate:.1%}" if rejection_rate > 0 else "0%")
        
        # Show claim details
        claim_collection = verifiable_metadata.get("claim_collection")
        if claim_collection and hasattr(claim_collection, 'claims') and claim_collection.claims:
            with st.expander("🔍 Claim Details", expanded=False):
                from src.claims.schema import VerificationStatus
                
                # Separate skipped claims
                skipped = [c for c in claim_collection.claims if c.metadata.get("skipped", False)]
                verified = [c for c in claim_collection.claims if c.status == VerificationStatus.VERIFIED and not c.metadata.get("skipped")]
                rejected = [c for c in claim_collection.claims if c.status == VerificationStatus.REJECTED and not c.metadata.get("skipped")]
                low_conf = [c for c in claim_collection.claims if c.status == VerificationStatus.LOW_CONFIDENCE and not c.metadata.get("skipped")]
                
                tab1, tab2, tab3, tab4 = st.tabs([
                    f"✅ Verified ({len(verified)})",
                    f"❌ Skipped ({len(skipped)})",
                    f"⚠️ Low Confidence ({len(low_conf)})",
                    f"🚫 Rejected ({len(rejected)})"
                ])
                
                with tab1:
                    if verified:
                        for i, claim in enumerate(verified[:20], 1):  # Show first 20
                            with st.expander(f"**{i}. {claim.claim_text[:80]}...**" if len(claim.claim_text) > 80 else f"**{i}. {claim.claim_text}**", expanded=(i == 1)):
                                st.write(f"**Full Claim:** {claim.claim_text}")
                                st.caption(f"Confidence: {claim.confidence:.2f}")
                                
                                # Show sources from metadata (for cited generation)
                                meta = claim.metadata or {}
                                source_urls = meta.get("source_urls", [])
                                sources = meta.get("sources", [])
                                
                                if sources:
                                    st.markdown(f"**📚 Verified Sources ({len(sources)}):**")
                                    for src_idx, source in enumerate(sources[:3], 1):
                                        with st.container():
                                            url = source.get("url", "Unknown")
                                            domain = source.get("domain", "")
                                            title = source.get("title", "")
                                            snippet = source.get("snippet", "")
                                            authority_tier = source.get("authority_tier", "")
                                            
                                            # Display source with authority badge
                                            tier_label = ""
                                            if authority_tier and "TIER_1" in str(authority_tier):
                                                tier_label = "🏆"  # Official/Authoritative
                                            elif authority_tier and "TIER_2" in str(authority_tier):
                                                tier_label = "✅"  # Academic/Institutional
                                            else:
                                                tier_label = "📖"  # Community/General
                                            
                                            st.markdown(f"**{tier_label} [{title or domain}]({url})**")
                                            if snippet:
                                                st.caption(f"📝 \"{snippet.strip()}...\"")
                                            st.caption(f"🔗 {domain}")
                                            st.markdown("---")
                                elif source_urls:
                                    st.markdown(f"**📚 Verified Sources ({len(source_urls)}):**")
                                    for url in source_urls[:3]:
                                        st.markdown(f"🔗 [{url}]({url})")
                                
                                # Show evidence objects if available (for standard mode)
                                elif hasattr(claim, 'evidence_objects') and claim.evidence_objects:
                                    st.markdown(f"**📚 Evidence ({len(claim.evidence_objects)} sources):**")
                                    for ev_idx, evidence in enumerate(claim.evidence_objects[:3], 1):
                                        with st.container():
                                            st.markdown(f"**Source {ev_idx}:**")
                                            st.text(evidence.snippet[:300] + "..." if len(evidence.snippet) > 300 else evidence.snippet)
                                            if hasattr(evidence, 'similarity'):
                                                st.caption(f"Similarity: {evidence.similarity:.2f}")
                                            st.markdown("---")
                                elif claim.evidence_ids:
                                    st.caption(f"Evidence: {len(claim.evidence_ids)} source(s) (details not available)")
                    else:
                        st.info("No verified claims")
                
                with tab2:
                    if skipped:
                        st.warning("⏭️ **These topics/concepts had insufficient supporting evidence in reliable sources and were excluded from the study guide.**")
                        st.info("🔍 **Authority Tiers:**\n- 🏆 TIER 1: Official documentation, standards, academic papers\n- ✅ TIER 2: Institutional sources, textbooks\n- 📖 TIER 3: Community resources, blogs")
                        
                        for i, claim in enumerate(skipped, 1):
                            concept_name = claim.claim_text.split(':')[0]
                            meta = claim.metadata or {}
                            skip_reason = meta.get("skip_reason", "Unknown reason")
                            
                            with st.expander(f"**{i}. {concept_name}** ⏭️", expanded=i <= 2):
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"**Concept:** {concept_name}")
                                    st.error(f"**Reason:** {skip_reason}")
                                with col2:
                                    st.caption("Status")
                                    st.caption("⏭️ Skipped")
                                
                                st.markdown("---")
                                st.markdown("**💡 What you can do:**")
                                st.markdown(f"""
1. **Add more details** - Include more information about "{concept_name}" in your notes
2. **Cite sources** - If you have textbooks or references, mention them
3. **Use external resources** - Search online resources and share relevant content
4. **Define yourself** - Provide a clear definition if the concept is important to your studies
                                """)
                    else:
                        st.success("✅ **All extracted concepts have supporting evidence sources!**")
                
                with tab3:
                    if low_conf:
                        for i, claim in enumerate(low_conf[:20], 1):
                            with st.expander(f"**{i}. {claim.claim_text[:80]}...**" if len(claim.claim_text) > 80 else f"**{i}. {claim.claim_text}**", expanded=(i == 1)):
                                st.write(f"**Full Claim:** {claim.claim_text}")
                                st.caption(f"Confidence: {claim.confidence:.2f}")
                                
                                # Show sources from metadata
                                meta = claim.metadata or {}
                                sources = meta.get("sources", [])
                                
                                if sources:
                                    st.markdown(f"**📚 Partial Sources ({len(sources)}):**")
                                    for src_idx, source in enumerate(sources[:2], 1):
                                        with st.container():
                                            url = source.get("url", "Unknown")
                                            domain = source.get("domain", "")
                                            title = source.get("title", "")
                                            snippet = source.get("snippet", "")
                                            authority_tier = source.get("authority_tier", "")
                                            
                                            # Display source with authority badge
                                            tier_label = ""
                                            if authority_tier and "TIER_1" in str(authority_tier):
                                                tier_label = "🏆"
                                            elif authority_tier and "TIER_2" in str(authority_tier):
                                                tier_label = "✅"
                                            else:
                                                tier_label = "📖"
                                            
                                            st.markdown(f"**{tier_label} [{title or domain}]({url})**")
                                            if snippet:
                                                st.caption(f"📝 \"{snippet.strip()}...\"")
                                            st.caption(f"🔗 {domain}")
                                            st.markdown("---")
                                
                                # Show evidence objects if available
                                elif hasattr(claim, 'evidence_objects') and claim.evidence_objects:
                                    st.markdown(f"**📚 Evidence ({len(claim.evidence_objects)} sources):**")
                                    for ev_idx, evidence in enumerate(claim.evidence_objects[:3], 1):
                                        with st.container():
                                            st.markdown(f"**Source {ev_idx}:**")
                                            st.text(evidence.snippet[:300] + "..." if len(evidence.snippet) > 300 else evidence.snippet)
                                            if hasattr(evidence, 'similarity'):
                                                st.caption(f"Similarity: {evidence.similarity:.2f}")
                                            st.markdown("---")
                                elif claim.evidence_ids:
                                    st.caption(f"Evidence: {len(claim.evidence_ids)} source(s) (details not available)")
                    else:
                        st.info("No low confidence claims")
                
                with tab4:
                    if rejected:
                        for claim in rejected[:10]:
                            st.write(f"**Claim:** {claim.claim_text}")
                            st.caption(f"Reason: {claim.rejection_reason or 'Unknown'}")
                            if hasattr(claim, 'confidence'):
                                st.caption(f"Confidence: {claim.confidence:.2f}")
                            st.divider()
                    else:
                        st.info("No rejected claims")
        elif verifiable_metadata.get("status"):
            # Special status already shown above
            pass
        else:
            st.info(
                "💡 **No detailed claim information available**\n\n"
                "Claims were processed but detailed breakdown is not available. "
                "Check the metrics above for summary statistics."
            )
        
        if st.button("Close Report", type="secondary"):
            st.session_state.show_detailed_report = False
            st.rerun()
    
    st.divider()
    
    # New processing
    if st.button("🆕 Start New Session", type="primary", use_container_width=True):
        # Clear relevant session state
        for key in ["current_output", "result", "verifiable_metadata"]:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.active_tab = 0
        st.rerun()


def render_redesigned_ui():
    """
    Main entry point for the redesigned UI.
    
    This replaces the old single-page design with a clean tab-based interface.
    """
    # Initialize session state
    initialize_session_state()
    
    # Check if processing should be executed
    from src.ui.processing_integration import execute_processing_pipeline
    execute_processing_pipeline()
    
    # Initialize or get run context
    if "run_context" not in st.session_state or "session_id" not in st.session_state:
        import time
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.session_state.session_id = session_id
        
        # Try to import config for random seed
        try:
            import config
            seed = config.GLOBAL_RANDOM_SEED if hasattr(config, 'GLOBAL_RANDOM_SEED') else None
        except:
            seed = None
        
        st.session_state.run_context = create_run_context(session_id, seed=seed)
        logger.info(f"Created run context for session {session_id}")
    
    # Render header (always visible)
    render_header()
    
    # Show context-aware tips
    has_results = "run_context" in st.session_state
    
    if has_results and st.session_state.active_tab == 0:
        # User has processed something and is back at input
        st.success(
            "✅ **Pipeline Complete!** Your results are ready. "
            "Go to the **Process** tab to see detailed execution flow, "
            "or **Results** tab to view your generated notes.",
            icon="🎉"
        )
    elif not has_results and st.session_state.active_tab == 0:
        # First time or no results yet
        st.info(
            "📋 **Workflow:** Enter content → Configure → Start processing → View pipeline visualization",
            icon="💡"
        )
    
    # Render sidebar settings (always visible)
    render_sidebar_settings()
    
    st.divider()
    
    # Tab navigation with proper styling
    tab_labels = ["📥 Input", "⚙️ Configure", "🚀 Process", "📄 Results"]
    
    # Create tab buttons
    cols = st.columns(4)
    for idx, (col, label) in enumerate(zip(cols, tab_labels)):
        with col:
            is_active = idx == st.session_state.active_tab
            button_type = "primary" if is_active else "secondary"
            
            if st.button(
                label,
                key=f"tab_{idx}",
                use_container_width=True,
                type=button_type,
                disabled=is_active
            ):
                st.session_state.active_tab = idx
                st.rerun()
    
    st.markdown("---")
    
    # Handle processing trigger - inform user about Classic UI switch
    if st.session_state.get("trigger_processing"):
        st.session_state.trigger_processing = False
        st.info(
            "🔄 **Switching to Processing Interface**\n\n"
            "You'll be redirected to the Classic UI where processing will begin. "
            "The pipeline includes all stages with real-time tracking:\n\n"
            "✅ Your inputs are preserved\n"
            "✅ Pipeline flow is tracked automatically\n"
            "✅ After completion, return to this UI to view visualization\n\n"
            "Click below to continue...",
            icon="📊"
        )
        
        if st.button("▶️ Start Processing Now", type="primary", use_container_width=True):
            st.session_state.use_redesigned_ui = False
            st.rerun()
    
    # Render selected tab content
    if st.session_state.active_tab == 0:
        render_input_tab()
    elif st.session_state.active_tab == 1:
        render_configure_tab()
    elif st.session_state.active_tab == 2:
        render_process_tab()
    elif st.session_state.active_tab == 3:
        render_results_tab()
