"""
Citation Rendering

Renders citations from notes in Markdown and HTML formats with stable numbering,
proper formatting, and authority tier indicators.
"""

import logging
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from src.schema.output_schema import Citation

logger = logging.getLogger(__name__)


@dataclass
class CitationReference:
    """Represents a citation reference point in text."""
    index: int
    citation: Citation
    authority_indicator: str


class CitationRenderer:
    """Render citations in various formats with stable numbering."""
    
    # Authority tier display strings
    AUTHORITY_INDICATORS = {
        "TIER_1": "🔒",  # Official/locked
        "TIER_2": "🏫",  # Academic
        "TIER_3": "👥",  # Community
        None: "📄",  # Default
    }
    
    @staticmethod
    def _get_authority_indicator(tier: Optional[str]) -> str:
        """Get visual indicator for authority tier."""
        return CitationRenderer.AUTHORITY_INDICATORS.get(tier, "📄")
    
    @staticmethod
    def render_markdown(
        text: str,
        citations: List[Citation],
        include_snippets: bool = True
    ) -> Tuple[str, str]:
        """
        Render text with Markdown citations.
        
        Uses stable [1][2] numbering. Returns both rendered text and footnote section.
        
        Args:
            text: Main text to annotate
            citations: List of citations to embed
            include_snippets: Include snippet quotes in footnotes
        
        Returns:
            Tuple of (annotated_text, footnotes_section)
        
        Example:
            text = "The derivative is the rate of change."
            citations = [Citation(...)]
            annotated, notes = CitationRenderer.render_markdown(text, citations)
            # annotated: "The derivative is the rate of change.[1]"
            # notes: "[1] (Local, TIER_1) ...\n\n"
        """
        if not citations:
            return text, ""
        
        # Build citation index
        citation_map = {}  # citation.span_id -> index
        for i, citation in enumerate(citations, 1):
            citation_map[citation.span_id] = i
        
        # Append citation markers to text
        # Note: In a real implementation, this would mark specific portions
        # For now, append all citations at the end of the text
        annotated_text = text
        citation_numbers = list(range(1, len(citations) + 1))
        
        if citation_numbers:
            citation_markers = "".join(f"[{n}]" for n in citation_numbers)
            annotated_text = f"{text} {citation_markers}"
        
        # Build footnotes section
        footnotes = CitationRenderer._build_footnotes_md(citations)
        
        return annotated_text, footnotes
    
    @staticmethod
    def _build_footnotes_md(citations: List[Citation]) -> str:
        """Build Markdown footnotes section."""
        if not citations:
            return ""
        
        lines = ["\n---\n### Sources\n"]
        
        for i, citation in enumerate(citations, 1):
            authority_ind = CitationRenderer._get_authority_indicator(citation.authority_tier)
            source_type_label = "Online 🌐" if citation.source_type == "online" else "Local 📚"
            
            line = f"[{i}] ({source_type_label}{f', {authority_ind}' if citation.authority_tier else ''}) "
            line += f"**{citation.source_id}**"
            
            if citation.page_num:
                line += f" (p. {citation.page_num})"
            
            if citation.snippet:
                # Escape and truncate snippet if too long
                snippet = citation.snippet.replace('"', '\\"')
                if len(snippet) > 150:
                    snippet = snippet[:147] + "..."
                line += f"\n   > \"{snippet}\""
            
            lines.append(line + "\n")
        
        return "".join(lines)
    
    @staticmethod
    def render_html(
        text: str,
        citations: List[Citation],
        include_snippets: bool = True,
        collapsible: bool = True
    ) -> str:
        """
        Render text with HTML citations (optional collapsible panel).
        
        Args:
            text: Main text to annotate
            citations: List of citations
            include_snippets: Include snippet quotes
            collapsible: Make citations collapsible
        
        Returns:
            HTML string
        """
        if not citations:
            return f"<p>{text}</p>"
        
        # Add citation markers to text
        citation_numbers = list(range(1, len(citations) + 1))
        citation_markers = "".join(
            f'<sup><a href="#cite-{n}">[{n}]</a></sup>'
            for n in citation_numbers
        )
        annotated_text = f"<p>{text} {citation_markers}</p>"
        
        # Build citations panel
        if collapsible:
            citations_html = CitationRenderer._build_citations_html_collapsible(citations)
        else:
            citations_html = CitationRenderer._build_citations_html_expanded(citations)
        
        return f"{annotated_text}\n{citations_html}"
    
    @staticmethod
    def _build_citations_html_expanded(citations: List[Citation]) -> str:
        """Build expanded HTML citations section."""
        html = ['<section class="citations">', '<h4>Sources</h4>', '<ol>']
        
        for i, citation in enumerate(citations, 1):
            authority_ind = CitationRenderer._get_authority_indicator(citation.authority_tier)
            source_type = "Online" if citation.source_type == "online" else "Local"
            
            html.append(f'<li id="cite-{i}">')
            html.append(f'  <strong>{source_type}</strong> {authority_ind}')
            html.append(f'  <strong>{citation.source_id}</strong>')
            
            if citation.page_num:
                html.append(f' (p. {citation.page_num})')
            
            if citation.snippet:
                snippet = citation.snippet.replace('<', '&lt;').replace('>', '&gt;')
                if len(snippet) > 150:
                    snippet = snippet[:147] + "..."
                html.append(f'<blockquote>{snippet}</blockquote>')
            
            html.append('</li>')
        
        html.append('</ol>')
        html.append('</section>')
        
        return '\n'.join(html)
    
    @staticmethod
    def _build_citations_html_collapsible(citations: List[Citation]) -> str:
        """Build collapsible HTML citations section."""
        html = [
            '<details class="citations-panel">',
            '  <summary>📚 Show Sources (' + str(len(citations)) + ')</summary>',
            '  <div class="citations-content">',
        ]
        
        for i, citation in enumerate(citations, 1):
            authority_ind = CitationRenderer._get_authority_indicator(citation.authority_tier)
            source_type = "Online 🌐" if citation.source_type == "online" else "Local 📚"
            
            html.append(f'    <div class="citation" id="cite-{i}">')
            html.append(f'      <span class="citation-marker">[{i}]</span>')
            html.append(f'      <span class="source-type">{source_type}</span>')
            
            if citation.authority_tier:
                html.append(f'      <span class="authority">{authority_ind}</span>')
            
            html.append(f'      <strong>{citation.source_id}</strong>')
            
            if citation.page_num:
                html.append(f' (p. {citation.page_num})')
            
            if citation.snippet:
                snippet = citation.snippet.replace('<', '&lt;').replace('>', '&gt;')
                if len(snippet) > 150:
                    snippet = snippet[:147] + "..."
                html.append(f'      <blockquote><em>{snippet}</em></blockquote>')
            
            html.append('    </div>')
        
        html.extend([
            '  </div>',
            '</details>',
        ])
        
        return '\n'.join(html)
    
    @staticmethod
    def get_citation_statistics(citations: List[Citation]) -> Dict:
        """
        Compute statistics about citations.
        
        Args:
            citations: List of citations
        
        Returns:
            Dict with statistics
        """
        if not citations:
            return {
                "total_citations": 0,
                "by_source_type": {},
                "by_authority_tier": {},
                "unique_sources": 0,
            }
        
        stats = {
            "total_citations": len(citations),
            "by_source_type": {},
            "by_authority_tier": {},
        }
        
        # Count by source type
        for citation in citations:
            source_type = citation.source_type
            stats["by_source_type"][source_type] = stats["by_source_type"].get(source_type, 0) + 1
        
        # Count by authority tier
        for citation in citations:
            tier = citation.authority_tier or "unknown"
            stats["by_authority_tier"][tier] = stats["by_authority_tier"].get(tier, 0) + 1
        
        # Count unique sources
        unique_sources = set(c.source_id for c in citations)
        stats["unique_sources"] = len(unique_sources)
        
        return stats
    
    @staticmethod
    def render_plain_text(
        text: str,
        citations: List[Citation]
    ) -> str:
        """
        Render text with citations for plain text output.
        
        Args:
            text: Main text
            citations: List of citations
        
        Returns:
            Plain text with citations
        """
        if not citations:
            return text
        
        lines = [text]
        lines.append("\nSOURCES:")
        
        for i, citation in enumerate(citations, 1):
            source_type = "ONLINE" if citation.source_type == "online" else "LOCAL"
            line = f"  [{i}] {source_type}: {citation.source_id}"
            
            if citation.page_num:
                line += f" (p. {citation.page_num})"
            
            if citation.authority_tier:
                line += f" [{citation.authority_tier}]"
            
            lines.append(line)
            
            if citation.snippet:
                snippet = citation.snippet
                if len(snippet) > 100:
                    snippet = snippet[:97] + "..."
                lines.append(f"       \"{snippet}\"")
        
        return "\n".join(lines)


# Convenience functions
def render_markdown_citations(
    text: str,
    citations: List[Citation]
) -> Tuple[str, str]:
    """Render text with Markdown citations."""
    return CitationRenderer.render_markdown(text, citations)


def render_html_citations(
    text: str,
    citations: List[Citation],
    collapsible: bool = True
) -> str:
    """Render text with HTML citations."""
    return CitationRenderer.render_html(text, citations, collapsible=collapsible)


def render_plain_text_citations(
    text: str,
    citations: List[Citation]
) -> str:
    """Render text with plain text citations."""
    return CitationRenderer.render_plain_text(text, citations)
