"""
Streamlit UI components for execution flow dashboard.

Renders real-time pipeline progress with:
- Engine transparency panel (LLM, OCR, URL ingestion, retrieval)
- Pipeline stepper with stage status
- Detailed metrics for each stage
"""

import streamlit as st
from typing import Optional, Any
from src.ui.progress_tracker import RunContext, StageStatus, StageEvent, EnginesUsed


def render_engines_used_panel(run_context: RunContext):
    """
    Render the engines transparency panel at the top.
    
    Shows which engines/models are being used with key metrics.
    """
    st.markdown("### 🔧 Engines Used")
    
    engines = run_context.engines_used
    
    # Create columns for different engine categories
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**🤖 LLM Provider**")
        if engines.llm_provider:
            provider_display = {
                "openai": "OpenAI",
                "ollama": "Ollama (Local)",
                "mixed": "Mixed (OpenAI + Ollama)"
            }.get(engines.llm_provider, engines.llm_provider)
            
            st.markdown(f"**{provider_display}**")
            if engines.llm_model:
                st.caption(f"Model: `{engines.llm_model}`")
            if engines.llm_calls > 0:
                st.caption(f"Calls: {engines.llm_calls}")
            if engines.llm_tokens_total:
                st.caption(f"Tokens: {engines.llm_tokens_total:,}")
        else:
            st.caption("_Not yet determined_")
    
    with col2:
        st.markdown("**👁️ OCR**")
        if engines.ocr_enabled:
            st.markdown(f"**Enabled** ✅")
            if engines.ocr_engine:
                st.caption(f"Engine: {engines.ocr_engine}")
            if engines.ocr_device:
                device_emoji = "🚀" if engines.ocr_device == "cuda" else "💻"
                st.caption(f"{device_emoji} Device: {engines.ocr_device.upper()}")
            if engines.ocr_pages_processed > 0:
                st.caption(f"Pages OCR'd: {engines.ocr_pages_processed}")
            if engines.ocr_model_downloaded:
                st.caption("⬇️ Model downloaded")
        else:
            st.markdown("**Disabled** ⊘")
            st.caption("_Scanned PDFs not supported_")
    
    with col3:
        st.markdown("**🌐 URL Ingestion**")
        if engines.url_count_total > 0:
            st.markdown(f"**{engines.url_count_success}/{engines.url_count_total}** URLs")
            if engines.youtube_transcripts_extracted > 0:
                st.caption(f"📺 YouTube: {engines.youtube_transcripts_extracted}")
            if engines.article_extractions > 0:
                st.caption(f"📰 Articles: {engines.article_extractions}")
        else:
            st.caption("_No URLs_")
    
    with col4:
        st.markdown("**🔍 Retrieval**")
        if engines.retrieval_enabled:
            st.markdown("**Enabled** ✅")
            if engines.embedding_model:
                st.caption(f"Embed: `{engines.embedding_model}`")
            if engines.reranker_model:
                st.caption(f"Rerank: `{engines.reranker_model}`")
            if engines.nli_model:
                st.caption(f"NLI: `{engines.nli_model}`")
        else:
            st.caption("_Fast mode_")
    
    st.divider()


def render_stage_details(stage_event: StageEvent):
    """
    Render detailed view of a single stage event.
    
    Shows metrics, warnings, and errors in an expandable panel.
    """
    # Status badge
    status_icon = stage_event.status.icon()
    status_text = stage_event.status.value.replace("_", " ").title()
    
    # Duration
    duration_text = ""
    if stage_event.duration_s is not None:
        duration_text = f" • {stage_event.duration_s:.2f}s"
    elif stage_event.status == StageStatus.RUNNING:
        duration_text = " • In progress..."
    
    # Skip reason
    if stage_event.status == StageStatus.SKIPPED and stage_event.skip_reason:
        st.caption(f"⊘ Skipped: {stage_event.skip_reason}")
        return
    
    # Error
    if stage_event.status == StageStatus.FAILED and stage_event.error:
        st.error(f"❌ **Error:** {stage_event.error}")
        return
    
    # Metrics
    if stage_event.metrics:
        st.markdown("**📊 Metrics:**")
        
        # Group metrics by type
        count_metrics = {k: v for k, v in stage_event.metrics.items() 
                        if any(word in k.lower() for word in ['count', 'total', 'num', 'pages'])}
        time_metrics = {k: v for k, v in stage_event.metrics.items() 
                       if any(word in k.lower() for word in ['time', 'duration', 'seconds'])}
        other_metrics = {k: v for k, v in stage_event.metrics.items() 
                        if k not in count_metrics and k not in time_metrics}
        
        # Display counts
        if count_metrics:
            cols = st.columns(min(len(count_metrics), 3))
            for idx, (key, value) in enumerate(count_metrics.items()):
                col_idx = idx % 3
                with cols[col_idx]:
                    # Format key nicely
                    display_key = key.replace("_", " ").title()
                    if isinstance(value, (int, float)):
                        st.metric(label=display_key, value=f"{value:,}" if isinstance(value, int) else f"{value:.2f}")
                    else:
                        st.caption(f"{display_key}: {value}")
        
        # Display other metrics
        if other_metrics:
            for key, value in other_metrics.items():
                display_key = key.replace("_", " ").title()
                if isinstance(value, dict):
                    st.caption(f"**{display_key}:**")
                    for sub_key, sub_value in value.items():
                        st.caption(f"  • {sub_key}: {sub_value}")
                elif isinstance(value, list):
                    st.caption(f"**{display_key}:** {', '.join(str(v) for v in value)}")
                else:
                    st.caption(f"**{display_key}:** {value}")
    
    # Warnings
    if stage_event.warnings:
        st.warning("**⚠️ Warnings:**\n" + "\n".join(f"- {w}" for w in stage_event.warnings))


def render_pipeline_stepper(run_context: RunContext, show_details: bool = True):
    """
    Render the pipeline stepper showing all stages.
    
    Args:
        run_context: Current run context
        show_details: If True, show expandable details for each stage
    """
    # Render each stage with clean Streamlit components
    for idx, stage_key in enumerate(run_context.STAGE_ORDER, 1):
        stage = run_context.stage_events[stage_key]
        
        # Status icon and color
        status_icon = stage.status.icon()
        status_color = stage.status.color()
        
        # Build stage title with status
        title_parts = [f"{idx}.", status_icon, stage.stage_name]
        
        # Add duration if available
        if stage.duration_s is not None and stage.duration_s > 0:
            title_parts.append(f"({stage.duration_s:.1f}s)")
        elif stage.status == StageStatus.RUNNING:
            title_parts.append("(running...)")
        
        stage_title = " ".join(title_parts)
        
        # Render based on status
        if stage.status == StageStatus.COMPLETED:
            with st.container():
                st.success(stage_title, icon="✅")
                if show_details and stage.metrics:
                    with st.expander("View Details", expanded=False):
                        render_stage_details(stage)
        
        elif stage.status == StageStatus.RUNNING:
            with st.container():
                st.info(stage_title, icon="⏳")
                if show_details:
                    with st.expander("View Details", expanded=True):
                        render_stage_details(stage)
        
        elif stage.status == StageStatus.SKIPPED:
            with st.container():
                skip_msg = stage_title
                if stage.skip_reason:
                    skip_msg += f" - {stage.skip_reason}"
                st.info(skip_msg, icon="⏭️")
        
        elif stage.status == StageStatus.FAILED:
            with st.container():
                error_msg = stage_title
                if stage.error:
                    error_msg += f" - {stage.error}"
                st.error(error_msg, icon="❌")
                if show_details:
                    with st.expander("View Error Details", expanded=True):
                        render_stage_details(stage)
        
        elif stage.status == StageStatus.WARNING:
            with st.container():
                st.warning(stage_title, icon="⚠️")
                if show_details:
                    with st.expander("View Details", expanded=False):
                        render_stage_details(stage)
        
        else:  # NOT_STARTED
            # Use a simple container with icon for not-started stages
            st.markdown(f"**{status_icon} {idx}. {stage.stage_name}**")
            st.caption("_Not started yet_")
        
        st.markdown("")  # Spacing


def render_execution_flow_dashboard(
    run_context: RunContext,
    show_engines: bool = True,
    show_details: bool = True,
    container: Optional[Any] = None
):
    """
    Render complete execution flow dashboard.
    
    Args:
        run_context: Current run context
        show_engines: Show engines used panel
        show_details: Show detailed metrics for each stage
        container: Optional Streamlit container to render into
    """
    # Note: container parameter is ignored as functions use st directly
    st.markdown("## 🚀 Execution Flow Dashboard")
    st.caption("Real-time pipeline progress and engine transparency")
    
    if show_engines:
        render_engines_used_panel(run_context)
    
    render_pipeline_stepper(run_context, show_details=show_details)


def render_demo_mode_controls(run_context: RunContext):
    """
    Render demo mode controls for presentations.
    
    Allows simulating stage transitions with delays.
    """
    with st.expander("🎬 Demo Mode Controls", expanded=False):
        st.caption("Simulate pipeline execution for demo purposes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Start Next Stage"):
                # Find next not-started stage
                for stage_key in run_context.STAGE_ORDER:
                    stage = run_context.stage_events[stage_key]
                    if stage.status == StageStatus.NOT_STARTED:
                        run_context.start_stage(stage_key)
                        st.rerun()
                        break
        
        with col2:
            if st.button("✅ Complete Current Stage"):
                # Find running stage
                for stage_key in run_context.STAGE_ORDER:
                    stage = run_context.stage_events[stage_key]
                    if stage.status == StageStatus.RUNNING:
                        run_context.complete_stage(stage_key, metrics={"demo": "simulated"})
                        st.rerun()
                        break
        
        if st.button("⏭️ Skip Current Stage"):
            for stage_key in run_context.STAGE_ORDER:
                stage = run_context.stage_events[stage_key]
                if stage.status == StageStatus.RUNNING:
                    run_context.skip_stage(stage_key, "Skipped in demo mode")
                    st.rerun()
                    break
        
        if st.button("🔄 Reset All Stages"):
            for stage in run_context.stage_events.values():
                stage.status = StageStatus.NOT_STARTED
                stage.started_at = None
                stage.ended_at = None
                stage.duration_s = None
                stage.metrics.clear()
                stage.warnings.clear()
                stage.error = None
                stage.skip_reason = None
            st.rerun()
