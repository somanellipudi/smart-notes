"""
Processing integration for the redesigned UI.

This module provides the bridge between the UI and the processing pipeline,
handling all the orchestration of Smart Notes generation.
"""

import streamlit as st
import logging
import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from src.utils.performance_logger import PerformanceTimer, set_session_id

logger = logging.getLogger(__name__)


def trigger_processing(
    notes_text: str,
    notes_images: Optional[List] = None,
    audio_file: Optional[Any] = None,
    urls_text: str = "",
    equations: str = "",
    external_context: str = "",
    verifiable_mode: bool = False,
    num_topics: int = 5,
    num_concepts: int = 5,
    llm_provider: str = "openai",
    model: str = "gpt-4o-mini",
    temperature: float = 0.3
) -> bool:
    """
    Trigger the processing pipeline with the given inputs.
    
    This function orchestrates the entire Smart Notes generation process,
    integrating with the execution flow tracker for real-time visualization.
    
    Args:
        notes_text: Input text content
        notes_images: Uploaded images/PDFs
        audio_file: Audio file for transcription
        urls_text: URLs (one per line)
        equations: Mathematical equations
        external_context: Additional reference material
        verifiable_mode: Enable research-grade verification
        num_topics: Number of topics to generate
        num_concepts: Concepts per topic
        llm_provider: LLM provider to use
        model: Model name
        temperature: Temperature for LLM generation
    
    Returns:
        bool: True if processing was triggered successfully
    """
    try:
        # Store inputs in session state for processing
        st.session_state.processing_inputs = {
            "notes_text": notes_text,
            "notes_images": notes_images,
            "audio_file": audio_file,
            "urls_text": urls_text,
            "equations": equations,
            "external_context": external_context,
            "verifiable_mode": verifiable_mode,
            "num_topics": num_topics,
            "num_concepts": num_concepts,
            "llm_provider": llm_provider,
            "model": model,
            "temperature": temperature
        }
        
        # Set processing flag
        st.session_state.start_processing = True
        
        logger.info("Processing triggered with verifiable_mode=%s", verifiable_mode)
        return True
        
    except Exception as e:
        logger.error(f"Failed to trigger processing: {e}")
        st.error(f"Failed to start processing: {str(e)}")
        return False


def execute_processing_pipeline():
    """
    Execute the actual processing pipeline.
    
    This is called by the main app when processing flag is set.
    It orchestrates all stages of the Smart Notes generation pipeline.
    """
    if not st.session_state.get("start_processing"):
        return
    
    # Clear the flag
    st.session_state.start_processing = False
    
    # Get inputs
    inputs = st.session_state.get("processing_inputs", {})
    
    if not inputs:
        st.error("No processing inputs found")
        return
    
    # Import necessary modules
    try:
        from src.reasoning.verifiable_pipeline import VerifiablePipelineWrapper
        import config
    except ImportError as e:
        st.error(f"Failed to import required modules: {e}")
        return
    
    # Get run context for tracking
    run_context = st.session_state.get("run_context")
    
    if not run_context:
        st.error("No run context available for tracking")
        return
    
    # Use a single session ID per request for consistent logs and reporting
    session_id = st.session_state.get("session_id")
    if not session_id:
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.session_state.session_id = session_id
    set_session_id(session_id)
    
    # Track inputs received
    run_context.start_stage("inputs_received")
    run_context.complete_stage("inputs_received", {
        "text_chars": len(inputs.get("notes_text", "")),
        "files_provided": len(inputs.get("notes_images", [])) if inputs.get("notes_images") else 0,
        "audio_provided": inputs.get("audio_file") is not None,
        "urls_provided": len([u for u in inputs.get("urls_text", "").split("\n") if u.strip()])
    })
    
    # Show processing UI in a container for better visibility
    st.markdown("### 🚀 Processing your content...")
    progress_container = st.container()
    status_placeholder = st.empty()
    
    with progress_container:
        st.write("📥 Inputs received")
        st.write(f"📝 {len(inputs.get('notes_text', ''))} characters")
        
        # Stage 1: Ingestion & Cleaning
        run_context.start_stage("ingestion_cleaning")
        st.write("🧹 Ingesting and cleaning content...")
        
        # Combine all text sources
        combined_text = inputs.get("notes_text", "")
        
        # Process uploaded files (images, PDFs)
        notes_images = inputs.get("notes_images", [])
        if notes_images:
            run_context.start_stage("ocr_extraction")
            st.write(f"📄 Processing {len(notes_images)} file(s)...")
            
            ocr_stats = {"files_processed": 0, "chars_extracted": 0, "files_failed": 0}
            
            try:
                from src.audio.image_ocr import ImageOCR
                
                # Initialize OCR system with performance tracking
                ocr = None
                try:
                    with PerformanceTimer("ocr_initialization", session_id=session_id):
                        ocr = ImageOCR()
                        st.write("  ✓ OCR system initialized")
                except Exception as ocr_init_err:
                    with st.expander("📖 How to Install OCR", expanded=True):
                        st.markdown("""
                        **Option 1: EasyOCR (Recommended)**
                        ```bash
                        pip install easyocr==1.7.1 python-bidi
                        ```
                        
                        **Option 2: Tesseract OCR**
                        ```bash
                        pip install pytesseract
                        # Also install Tesseract binary from: https://github.com/tesseract-ocr/tesseract
                        ```
                        
                        **For now:** Add text manually in the text input box instead of uploading images.
                        """)
                    
                    run_context.fail_stage("ocr_extraction", f"OCR init failed: {ocr_init_err}")
                    # Still process PDFs without OCR
                    ocr = None
                
                for file_obj in notes_images:
                    file_name = file_obj.name
                    file_bytes = file_obj.read()
                    file_start_length = len(combined_text)
                    
                    # Determine file type
                    if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        # OCR image with performance tracking
                        if not ocr:
                            st.warning(f"  ⚠️ Skipping {file_name} - OCR not available")
                            ocr_stats["files_failed"] += 1
                            continue
                            
                        st.write(f"  📷 OCR: {file_name}...")
                        with PerformanceTimer("ocr_image_extraction", session_id=session_id,
                                            metadata={"filename": file_name, "file_size": len(file_bytes)}):
                            result = ocr.extract_text_from_bytes(file_bytes, image_type="notes")
                        extracted_text = result.get("text", "")
                        confidence = result.get("confidence", 0)
                        
                        if extracted_text and extracted_text.strip():
                            combined_text += f"\n\n--- From {file_name} ---\n{extracted_text}"
                            chars_added = len(combined_text) - file_start_length
                            ocr_stats["chars_extracted"] += chars_added
                            ocr_stats["files_processed"] += 1
                            st.write(f"  ✅ Extracted {chars_added} chars (confidence: {confidence:.2f})")
                        else:
                            ocr_stats["files_failed"] += 1
                            error_msg = result.get("metadata", {}).get("error", "Unknown error")
                            st.error(f"  ❌ No text extracted from {file_name}")
                            if error_msg != "Unknown error":
                                st.caption(f"     Error: {error_msg}")
                            st.info(
                                "💡 **OCR Tips:**\n"
                                "- Ensure image is clear and well-lit\n"
                                "- Try cropping to focus on text area\n"
                                "- For handwriting, use dark pen on white paper\n"
                                "- Check if OCR system is properly installed"
                            )
                            
                    elif file_name.lower().endswith('.pdf'):
                        # Extract PDF text
                        st.write(f"  📑 PDF: {file_name}...")
                        try:
                            import io
                            import PyPDF2
                            pdf_file = io.BytesIO(file_bytes)
                            pdf_reader = PyPDF2.PdfReader(pdf_file)
                            pdf_text = ""
                            for page in pdf_reader.pages:
                                pdf_text += page.extract_text()
                            
                            if pdf_text.strip():
                                combined_text += f"\n\n--- From {file_name} ---\n{pdf_text}"
                                chars_added = len(combined_text) - file_start_length
                                ocr_stats["chars_extracted"] += chars_added
                                ocr_stats["files_processed"] += 1
                                st.write(f"  ✅ Extracted {chars_added} chars from PDF")
                            else:
                                if ocr:
                                    st.warning(f"  ⚠️ No text extracted from PDF, trying OCR...")
                                    # Fallback to OCR for scanned PDFs
                                    result = ocr.extract_text_from_bytes(file_bytes, image_type="notes")
                                    extracted_text = result.get("text", "")
                                    if extracted_text:
                                        combined_text += f"\n\n--- From {file_name} (OCR) ---\n{extracted_text}"
                                        chars_added = len(combined_text) - file_start_length
                                        ocr_stats["chars_extracted"] += chars_added
                                        ocr_stats["files_processed"] += 1
                                        st.write(f"  ✅ OCR extracted {chars_added} chars")
                                    else:
                                        ocr_stats["files_failed"] += 1
                                else:
                                    st.warning(f"  ⚠️ No text in PDF and OCR not available")
                                    ocr_stats["files_failed"] += 1
                        except Exception as pdf_err:
                            logger.warning(f"PDF extraction failed for {file_name}: {pdf_err}")
                            st.warning(f"  ⚠️ PDF extraction failed: {pdf_err}")
                            ocr_stats["files_failed"] += 1
                    
                    # Reset file object position for potential reuse
                    file_obj.seek(0)
                
                run_context.complete_stage("ocr_extraction", ocr_stats)
                st.write(f"✅ Processed {ocr_stats['files_processed']}/{len(notes_images)} files successfully")
                
                if ocr_stats["files_failed"] > 0:
                    st.warning(f"⚠️ {ocr_stats['files_failed']} file(s) failed to process")
                    
            except Exception as ocr_err:
                logger.error(f"File processing failed: {ocr_err}")
                st.error(f"⚠️ File processing error: {ocr_err}")
                run_context.fail_stage("ocr_extraction", str(ocr_err))
        else:
            run_context.skip_stage("ocr_extraction", "No files provided")
        
        run_context.complete_stage("ingestion_cleaning", {
            "combined_chars": len(combined_text)
        })
        st.write(f"✅ Content collected: {len(combined_text)} total characters")
        
        # Check if we have enough content
        if len(combined_text.strip()) < 100:
            st.error(
                "❌ **Insufficient Content**\n\n"
                f"Only {len(combined_text.strip())} characters extracted. Need at least 100 characters.\n\n"
                "**Options:**\n"
                "1. Add text manually in the text box\n"
                "2. Upload clearer images with better lighting\n"
                "3. Try different image format (PNG usually works best)\n"
                "4. Ensure handwriting is dark and clear"
            )
            run_context.fail_stage("llm_generation", "Insufficient content for processing")
            status_placeholder.error("❌ Processing failed: Insufficient content")
            return
        
        # Skip stages that aren't needed
        run_context.skip_stage("chunking_provenance", "Fast mode")
        run_context.skip_stage("embedding_indexing", "Not used")
        
        # Stage 6: LLM Generation
        run_context.start_stage("llm_generation")
        st.write("🤖 Generating study notes with AI...")
        
        try:
            # Create pipeline
            pipeline = VerifiablePipelineWrapper(
                provider_type=inputs.get("llm_provider", "openai"),
                api_key=config.OPENAI_API_KEY if hasattr(config, "OPENAI_API_KEY") else None,
                ollama_url=config.OLLAMA_URL if hasattr(config, "OLLAMA_URL") else "http://localhost:11434",
                model=inputs.get("model", "gpt-4o-mini"),
                domain_profile=None  # TODO: Add domain profile support
            )
            
            # Track LLM usage
            from src.ui.pipeline_instrumentation import track_llm_call
            track_llm_call(
                run_context,
                provider=inputs.get("llm_provider", "openai"),
                model=inputs.get("model", "gpt-4o-mini")
            )
            
            # Parse URLs
            urls = []
            if inputs.get("urls_text"):
                urls = [u.strip() for u in inputs.get("urls_text", "").split('\n') if u.strip()]
            
            # Parse equations
            equations_list = []
            if inputs.get("equations"):
                equations_list = [eq.strip() for eq in inputs.get("equations", "").split('\n') if eq.strip()]
            
            def progress_callback(message: str):
                """Progress callback for pipeline."""
                st.write(f"  ↳ {message}")
            
            # Process content with performance tracking
            with PerformanceTimer("total_pipeline", session_id=session_id,
                                metadata={"verifiable_mode": inputs.get("verifiable_mode", False)}):
                output, verifiable_metadata = pipeline.process(
                    combined_content=combined_text,
                    equations=equations_list,
                    external_context=inputs.get("external_context", ""),
                    session_id=session_id,
                    verifiable_mode=inputs.get("verifiable_mode", False),
                    output_filters=None,  # Use default filters
                    urls=urls,
                    progress_callback=progress_callback
                )
            
            run_context.complete_stage("llm_generation", {
                "topics_generated": len(output.topics) if hasattr(output, 'topics') else 0,
                "provider": inputs.get("llm_provider"),
                "model": inputs.get("model")
            })
            st.write("✅ Notes generated")
            
            # Handle verification stages
            if inputs.get("verifiable_mode") and verifiable_metadata:
                run_context.start_stage("claim_extraction")
                st.write("🔍 Extracting claims...")
                claims_count = len(verifiable_metadata.get("verified_claims", [])) if verifiable_metadata else 0
                run_context.complete_stage("claim_extraction", {"claims_count": claims_count})
                
                run_context.start_stage("retrieval_reranking")
                st.write("📚 Retrieving evidence...")
                run_context.complete_stage("retrieval_reranking", {
                    "sources_found": len(verifiable_metadata.get("evidence_sources", [])) if verifiable_metadata else 0
                })
                
                run_context.start_stage("verification")
                st.write("✓ Verifying claims...")
                run_context.complete_stage("verification", {
                    "verification_complete": True
                })
            else:
                run_context.skip_stage("claim_extraction", "Fast mode")
                run_context.skip_stage("retrieval_reranking", "Fast mode")
                run_context.skip_stage("verification", "Fast mode")
            
            # Stage 10: Reporting
            run_context.start_stage("reporting_exports")
            st.write("📄 Generating reports...")
            
            # Store result in session state
            st.session_state.result = output
            st.session_state.verifiable_metadata = verifiable_metadata
            
            # Debug log
            if verifiable_metadata:
                logger.info(f"Stored verifiable_metadata: verifiable_mode={verifiable_metadata.get('verifiable_mode')}, status={verifiable_metadata.get('status')}")
            else:
                logger.info("No verifiable_metadata returned from pipeline")
            
            # Format output for display
            try:
                from src.output_formatter import OutputFormatter
                formatter = OutputFormatter()
                markdown_output = formatter.format_to_markdown(output)
            except Exception as fmt_err:
                logger.warning(f"Failed to format output: {fmt_err}")
                markdown_output = "# Study Notes\n\nProcessing complete. View results in the Results tab."
            
            st.session_state.current_output = {
                "output": output,
                "markdown": markdown_output,
                "topics": output.topics if hasattr(output, 'topics') else []
            }
            
            # Save session
            try:
                artifacts_dir = Path(config.ARTIFACTS_DIR) if hasattr(config, "ARTIFACTS_DIR") else Path("artifacts")
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                run_context.save(artifacts_dir)
                
                # Save output
                session_dir = artifacts_dir / st.session_state.session_id
                session_dir.mkdir(parents=True, exist_ok=True)
                output_file = session_dir / "output.json"
                with open(output_file, 'w') as f:
                    json.dump({
                        "output": output.dict() if hasattr(output, 'dict') else str(output),
                        "verifiable_metadata": verifiable_metadata
                    }, f, indent=2)
                
                run_context.complete_stage("reporting_exports", {
                    "session_saved": True,
                    "session_dir": str(session_dir)
                })
            except Exception as save_err:
                logger.warning(f"Failed to save session: {save_err}")
                run_context.complete_stage("reporting_exports", {
                    "session_saved": False,
                    "error": str(save_err)
                })
            
            status_placeholder.success("✅ Processing complete!")
            st.success("🎉 Your study notes are ready!")
            
            # Auto-navigate to results
            st.session_state.active_tab = 3
            st.rerun()
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            run_context.fail_stage("llm_generation", str(e))
            st.error(f"❌ Processing failed: {str(e)}")
            status_placeholder.error("❌ Processing failed")


def check_processing_requirements(
    notes_text: str,
    notes_images: Optional[List],
    audio_file: Optional[Any],
    urls_text: str
) -> tuple[bool, str]:
    """
    Check if minimum requirements are met for processing.
    
    Returns:
        tuple: (is_valid, message)
    """
    has_text = len(notes_text.strip()) > 0
    has_files = notes_images and len(notes_images) > 0
    has_audio = audio_file is not None
    has_urls = len([u for u in urls_text.split('\n') if u.strip()]) > 0
    
    has_any_input = has_text or has_files or has_audio or has_urls
    
    if not has_any_input:
        return False, "Please provide at least one content source (text, files, audio, or URLs)"
    
    return True, "Ready to process"
