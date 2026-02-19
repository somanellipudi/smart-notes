"""
Citation display helpers for Streamlit UI.

Renders provenance-aware citations with clickable URLs, timestamps, and source icons.
"""

import streamlit as st
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class CitationInfo:
    """Citation information for display."""
    source_type: str  # pdf_page, url_article, youtube_transcript, etc.
    origin: Optional[str] = None  # URL or filename
    page_num: Optional[int] = None
    timestamp_range: Optional[tuple] = None  # (start_sec, end_sec)
    snippet: Optional[str] = None
    confidence: float = 1.0


def render_citation_inline(citation: CitationInfo, max_length: int = 60) -> None:
    """
    Render a single citation inline (compact format).
    
    Args:
        citation: Citation information
        max_length: Max characters for display
    """
    # Source type icon
    source_icons = {
        "pdf_page": "📄",
        "url_article": "🔗",
        "youtube_transcript": "▶️",
        "audio_transcript": "🎤",
        "notes_text": "📝",
        "external_context": "🌐"
    }
    icon = source_icons.get(citation.source_type, "📌")
    
    # Build citation text
    parts = [icon]
    
    # Origin (URL or filename)
    if citation.origin:
        if citation.origin.startswith("http"):
            # Clickable URL
            display_text = citation.origin[:max_length] + "..." if len(citation.origin) > max_length else citation.origin
            parts.append(f"[{display_text}]({citation.origin})")
        else:
            # Filename
            parts.append(f"`{citation.origin}`")
    
    # Page number for PDFs
    if citation.page_num is not None:
        parts.append(f"p.{citation.page_num}")
    
    # Timestamp for audio/video
    if citation.timestamp_range:
        start_sec, end_sec = citation.timestamp_range
        start_time = f"{int(start_sec // 60):02d}:{int(start_sec % 60):02d}"
        end_time = f"{int(end_sec // 60):02d}:{int(end_sec % 60):02d}"
        
        # Make YouTube URLs timestamp-aware
        if citation.origin and "youtube.com" in citation.origin:
            timestamp_url = f"{citation.origin}&t={int(start_sec)}"
            parts.append(f"[{start_time}-{end_time}]({timestamp_url})")
        else:
            parts.append(f"⏱️ {start_time}-{end_time}")
    
    citation_text = " ".join(parts)
    st.markdown(citation_text, unsafe_allow_html=False)


def render_citation_list(
    citations: List[CitationInfo],
    title: str = "Sources",
    max_visible: int = 2,
    show_snippets: bool = True
) -> None:
    """
    Render a list of citations with expandable details.
    
    Args:
        citations: List of citations to display
        title: Section title
        max_visible: Number of citations to show before "Show more" expander
        show_snippets: Whether to show evidence snippets
    """
    if not citations:
        st.caption("No citations available")
        return
    
    st.write(f"**{title}** ({len(citations)})")
    
    # Show first max_visible citations inline
    for i, citation in enumerate(citations[:max_visible], 1):
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(f"**Citation {i}:**")
                render_citation_inline(citation)
                if show_snippets and citation.snippet:
                    st.caption(f"_{citation.snippet[:100]}..._" if len(citation.snippet) > 100 else f"_{citation.snippet}_")
            with col2:
                if citation.confidence < 1.0:
                    st.metric("Conf", f"{citation.confidence:.0%}", label_visibility="collapsed")
    
    # Remaining citations in expander
    if len(citations) > max_visible:
        with st.expander(f"➕ Show {len(citations) - max_visible} more citations"):
            for i, citation in enumerate(citations[max_visible:], max_visible + 1):
                st.caption(f"**Citation {i}:**")
                render_citation_inline(citation)
                if show_snippets and citation.snippet:
                    st.caption(f"_{citation.snippet[:100]}..._")
                st.divider()


def render_citation_table(citations: List[Dict[str, Any]]) -> None:
    """
    Render citations as a compact table.
    
    Args:
        citations: List of citation dictionaries with keys:
            - source_type
            - origin (URL or filename)
            - page_num (optional)
            - timestamp_range (optional)
    """
    if not citations:
        return
    
    # Build table data
    table_data = []
    for cit in citations:
        source_type = cit.get("source_type", "unknown")
        origin = cit.get("origin", "N/A")
        
        # Icon
        icon = {
            "pdf_page": "📄",
            "url_article": "🔗",
            "youtube_transcript": "▶️",
            "audio_transcript": "🎤",
            "notes_text": "📝"
        }.get(source_type, "📌")
        
        # Location
        location_parts = []
        if cit.get("page_num"):
            location_parts.append(f"p.{cit['page_num']}")
        if cit.get("timestamp_range"):
            start, end = cit["timestamp_range"]
            location_parts.append(f"{int(start//60):02d}:{int(start%60):02d}")
        location = " ".join(location_parts) if location_parts else "-"
        
        table_data.append({
            "Type": f"{icon} {source_type}",
            "Origin": origin[:40] + "..." if len(origin) > 40 else origin,
            "Location": location
        })
    
    # Display as Streamlit dataframe
    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True
    )


def format_timestamp_link(url: str, seconds: float) -> str:
    """
    Create a timestamp-aware YouTube link.
    
    Args:
        url: YouTube URL
        seconds: Timestamp in seconds
    
    Returns:
        URL with timestamp parameter
    """
    if "youtube.com" in url or "youtu.be" in url:
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}t={int(seconds)}"
    return url


def get_source_icon(source_type: str) -> str:
    """
    Get emoji icon for source type.
    
    Args:
        source_type: Source type identifier
    
    Returns:
        Emoji icon
    """
    icons = {
        "pdf_page": "📄",
        "url_article": "🔗",
        "youtube_transcript": "▶️",
        "audio_transcript": "🎤",
        "notes_text": "📝",
        "external_context": "🌐",
        "equation": "📐"
    }
    return icons.get(source_type, "📌")
