"""
Example integration of Execution Flow Dashboard into app.py

This file shows how to integrate the progress tracking system
into the main Smart Notes application.
"""

import streamlit as st
from pathlib import Path

# Import progress tracking
from src.ui.progress_tracker import create_run_context, RunContext
from src.ui.execution_flow_ui import (
    render_execution_flow_dashboard,
    render_demo_mode_controls
)
from src.ui.pipeline_instrumentation import (
    track_stage,
    track_llm_call,
    track_ocr_usage,
    track_url_ingestion,
    track_retrieval_usage,
    update_stage_metrics
)
import config


def initialize_run_context(session_id: str, seed: int = None) -> RunContext:
    """
    Initialize or get existing run context from session state.
    
    Args:
        session_id: Session identifier
        seed: Random seed for reproducibility
    
    Returns:
        RunContext instance
    """
    if "run_context" not in st.session_state:
        st.session_state.run_context = create_run_context(session_id, seed=seed)
    
    return st.session_state.run_context


# ============================================================================
# EXAMPLE: Add to sidebar for persistent display
# ============================================================================

def render_sidebar_dashboard():
    """Render compact dashboard in sidebar."""
    with st.sidebar:
        st.divider()
        st.markdown("### 🚀 Pipeline Status")
        
        if "run_context" in st.session_state:
            run_context = st.session_state.run_context
            
            # Show progress
            progress = run_context.get_progress_fraction()
            st.progress(progress, text=f"{int(progress * 100)}% Complete")
            
            # Show total time
            total_time = run_context.get_total_duration()
            st.caption(f"⏱️ Total: {total_time:.1f}s")
            
            # Show currently running stage
            for stage_key in run_context.STAGE_ORDER:
                stage = run_context.stage_events[stage_key]
                if stage.status.value == "running":
                    st.caption(f"{stage.status.icon()} {stage.stage_name}")
                    break
            
            # Option to view full dashboard
            if st.button("View Full Dashboard", use_container_width=True):
                st.session_state.show_dashboard = not st.session_state.get("show_dashboard", False)


# ============================================================================
# EXAMPLE: Add dashboard in main area with expander
# ============================================================================

def render_main_area_dashboard():
    """Render full dashboard in main area."""
    if st.session_state.get("show_dashboard", False):
        with st.expander("🚀 Execution Flow Dashboard", expanded=True):
            if "run_context" in st.session_state:
                run_context = st.session_state.run_context
                render_execution_flow_dashboard(
                    run_context,
                    show_engines=True,
                    show_details=True
                )
                
                # Optional: Demo mode controls
                if st.session_state.get("demo_mode", False):
                    render_demo_mode_controls(run_context)


# ============================================================================
# EXAMPLE: Instrument the main processing pipeline
# ============================================================================

def instrumented_pipeline_example():
    """
    Example showing how to instrument the main processing pipeline.
    
    Replace existing pipeline code with instrumented version.
    """
    # Initialize run context
    session_id = st.session_state.get("session_id", "session_default")
    run_context = initialize_run_context(session_id)
    
    # Track OCR configuration early
    track_ocr_usage(
        run_context,
        enabled=config.OCR_ENABLED,
        engine="easyocr" if config.OCR_ENABLED else None,
        device="cpu"  # Will be detected during actual use
    )
    
    # STAGE 1: Inputs Received
    with track_stage(run_context, "inputs_received"):
        uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True)
        urls_input = st.text_area("URLs (one per line)")
        notes_text = st.text_area("Paste notes")
        
        urls = [u.strip() for u in urls_input.split('\n') if u.strip()]
        
        # Complete with metrics
        run_context.complete_stage("inputs_received", {
            "files_uploaded": len(uploaded_files) if uploaded_files else 0,
            "urls_provided": len(urls),
            "text_chars": len(notes_text)
        })
    
    # STAGE 2: Ingestion & Cleaning
    with track_stage(run_context, "ingestion_cleaning"):
        # Process PDFs
        pdf_text = ""
        pdf_pages = 0
        ocr_chars = 0
        
        if uploaded_files:
            for file in uploaded_files:
                # ... PDF extraction logic ...
                pdf_pages += 10  # Example
                pdf_text += "extracted text"
        
        # Clean/preprocess
        cleaned_text = pdf_text  # ... cleaning logic ...
        lines_removed = 50  # Example
        
        run_context.complete_stage("ingestion_cleaning", {
            "total_pages": pdf_pages,
            "pdf_chars": len(pdf_text),
            "lines_removed": lines_removed
        })
    
    # STAGE 3: OCR Extraction
    if config.OCR_ENABLED and has_images:
        with track_stage(run_context, "ocr_extraction"):
            # Import OCR safely
            from src.utils.ocr_safe import ocr_extract_text, get_ocr_status
            
            status = get_ocr_status()
            
            # Process images
            ocr_text = ""
            for image in images:
                text, metadata = ocr_extract_text(image)
                ocr_text += text
            
            # Track OCR usage
            track_ocr_usage(
                run_context,
                enabled=True,
                engine=status["engine"],
                device=status["device"],
                pages_processed=len(images),
                model_downloaded=status["model_downloaded"]
            )
            
            run_context.complete_stage("ocr_extraction", {
                "pages_ocr": len(images),
                "ocr_chars": len(ocr_text),
                "device": status["device"]
            })
    else:
        run_context.skip_stage("ocr_extraction", "No images or OCR disabled")
    
    # STAGE 4: Chunking & Provenance
    with track_stage(run_context, "chunking_provenance"):
        # ... chunking logic ...
        chunks = []  # create_chunks(combined_text)
        
        run_context.complete_stage("chunking_provenance", {
            "chunks_total": len(chunks),
            "avg_chunk_size": 500,  # Calculate average
            "chunk_sources": {
                "pdf": 30,
                "url": 8,
                "text": 4
            }
        })
    
    # STAGE 5: Embedding & Indexing (if verifiable mode)
    if verifiable_mode:
        with track_stage(run_context, "embedding_indexing"):
            # ... embedding logic ...
            
            track_retrieval_usage(
                run_context,
                enabled=True,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            run_context.complete_stage("embedding_indexing", {
                "embeddings_created": len(chunks),
                "index_size": len(chunks)
            })
    else:
        run_context.skip_stage("embedding_indexing", "Fast mode - retrieval disabled")
    
    # STAGE 6: LLM Generation
    with track_stage(run_context, "llm_generation"):
        # Determine provider
        provider = "openai" if use_openai else "ollama"
        model = "gpt-4o-mini" if provider == "openai" else "llama3"
        
        # Generate notes
        output = generate_notes_with_llm(...)
        
        # Track LLM call
        track_llm_call(
            run_context,
            provider=provider,
            model=model,
            tokens=output.get("usage", {}).get("total_tokens")
        )
        
        run_context.complete_stage("llm_generation", {
            "provider": provider,
            "model": model,
            "notes_generated": True
        })
    
    # STAGE 7: Claim Extraction (if verifiable mode)
    if verifiable_mode:
        with track_stage(run_context, "claim_extraction"):
            claims = extract_claims(output)
            
            run_context.complete_stage("claim_extraction", {
                "claims_total": len(claims)
            })
    else:
        run_context.skip_stage("claim_extraction", "Fast mode")
    
    # STAGE 8: Retrieval & Reranking (if verifiable mode)
    if verifiable_mode:
        with track_stage(run_context, "retrieval_reranking"):
            # ... retrieval logic ...
            
            track_retrieval_usage(
                run_context,
                enabled=True,
                reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            
            run_context.complete_stage("retrieval_reranking", {
                "topk": 5,
                "reranker_used": True
            })
    else:
        run_context.skip_stage("retrieval_reranking", "Fast mode")
    
    # STAGE 9: Verification (if verifiable mode)
    if verifiable_mode:
        with track_stage(run_context, "verification"):
            # ... verification logic ...
            
            track_retrieval_usage(
                run_context,
                enabled=True,
                nli_model="microsoft/deberta-v3-base"
            )
            
            verified_count = 20
            rejected_count = 3
            
            run_context.complete_stage("verification", {
                "verified": verified_count,
                "rejected": rejected_count,
                "low_confidence": 2
            })
            
            # Add warning if many rejections
            if rejected_count > 5:
                run_context.warn_stage("verification", 
                    f"High rejection rate: {rejected_count} claims rejected")
    else:
        run_context.skip_stage("verification", "Fast mode")
    
    # STAGE 10: Reporting + Exports
    with track_stage(run_context, "reporting_exports"):
        # Generate reports
        report_paths = []
        # ... report generation ...
        
        run_context.complete_stage("reporting_exports", {
            "reports_generated": len(report_paths),
            "formats": ["markdown", "html", "json"]
        })
    
    # Save run context to artifacts
    artifacts_dir = Path(config.ARTIFACTS_DIR)
    run_context.save(artifacts_dir)
    
    return output


# ============================================================================
# EXAMPLE: Track URL Ingestion
# ============================================================================

def instrumented_url_ingestion(urls):
    """Example of tracking URL ingestion."""
    if "run_context" not in st.session_state:
        return
    
    run_context = st.session_state.run_context
    
    success_count = 0
    youtube_count = 0
    article_count = 0
    
    for url in urls:
        content, metadata = fetch_url_text(url)
        
        if content:
            success_count += 1
            if metadata.get("source_type") == "youtube":
                youtube_count += 1
            else:
                article_count += 1
    
    # Track ingestion
    track_url_ingestion(
        run_context,
        total=len(urls),
        success=success_count,
        youtube=youtube_count,
        articles=article_count
    )


# ============================================================================
# EXAMPLE: Complete app.py Integration Pattern
# ============================================================================

def main():
    """Main application with execution flow tracking."""
    st.set_page_config(page_title="Smart Notes", layout="wide")
    
    # Initialize session
    if "session_id" not in st.session_state:
        import time
        st.session_state.session_id = f"session_{int(time.time())}"
    
    # Initialize run context
    run_context = initialize_run_context(st.session_state.session_id)
    
    # Render sidebar dashboard
    render_sidebar_dashboard()
    
    # Main UI
    st.title("Smart Notes")
    
    # Show full dashboard if requested
    render_main_area_dashboard()
    
    # ... rest of app UI ...
    
    # Demo mode toggle in sidebar
    with st.sidebar:
        st.divider()
        st.session_state.demo_mode = st.checkbox("🎬 Demo Mode", value=False)
        
        if st.session_state.demo_mode:
            st.caption("Demo mode: Use controls in dashboard to simulate pipeline")


if __name__ == "__main__":
    main()
