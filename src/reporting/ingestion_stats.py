"""
Ingestion statistics aggregator.

Collects and aggregates statistics from all ingestion sources (PDF, URLs, text, audio)
and ensures proper reporting in RunContext and reports.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SourceStats:
    """Statistics for a single source."""
    source_type: str  # pdf, url, text, audio
    source_id: str
    chunks: int
    chars: int
    metadata: Dict[str, Any]


class IngestionStatsAggregator:
    """
    Aggregate ingestion statistics from multiple sources.
    
    Ensures proper tracking of:
    - PDF metrics (pages, OCR, cleaning)
    - URL metrics (count, fetch success, chunks)
    - Text metrics (chars, chunks)
    - Audio metrics (duration, transcript)
    - Overall metrics (total chunks, avg chunk size)
    """
    
    def __init__(self):
        """Initialize aggregator."""
        self.sources: List[SourceStats] = []
        
        # PDF metrics
        self.pdf_pages = 0
        self.pdf_pages_ocr = 0
        self.pdf_headers_removed = 0
        self.pdf_footers_removed = 0
        self.pdf_watermarks_removed = 0
        
        # URL metrics
        self.url_count = 0
        self.url_fetch_success = 0
        self.url_chunks = 0
        
        # Text metrics
        self.text_chars = 0
        self.text_chunks = 0
        
        # Audio metrics
        self.audio_seconds = 0.0
        self.transcript_chars = 0
        self.transcript_chunks = 0
        
        # Overall
        self.extraction_methods: List[str] = []
    
    def add_pdf_source(
        self,
        source_id: str,
        pages: int,
        pages_ocr: int = 0,
        headers_removed: int = 0,
        footers_removed: int = 0,
        watermarks_removed: int = 0,
        chunks: int = 0,
        chars: int = 0
    ):
        """Add PDF source statistics."""
        self.pdf_pages += pages
        self.pdf_pages_ocr += pages_ocr
        self.pdf_headers_removed += headers_removed
        self.pdf_footers_removed += footers_removed
        self.pdf_watermarks_removed += watermarks_removed
        
        self.sources.append(SourceStats(
            source_type="pdf",
            source_id=source_id,
            chunks=chunks,
            chars=chars,
            metadata={
                "pages": pages,
                "pages_ocr": pages_ocr,
                "headers_removed": headers_removed,
                "footers_removed": footers_removed,
                "watermarks_removed": watermarks_removed
            }
        ))
        
        if "pdf" not in self.extraction_methods:
            self.extraction_methods.append("pdf")
    
    def add_url_source(
        self,
        source_id: str,
        fetch_success: bool,
        chunks: int = 0,
        chars: int = 0
    ):
        """Add URL source statistics."""
        self.url_count += 1
        if fetch_success:
            self.url_fetch_success += 1
            self.url_chunks += chunks
            
            self.sources.append(SourceStats(
                source_type="url",
                source_id=source_id,
                chunks=chunks,
                chars=chars,
                metadata={"fetch_success": True}
            ))
        
        if "url" not in self.extraction_methods:
            self.extraction_methods.append("url")
    
    def add_text_source(
        self,
        source_id: str,
        chunks: int,
        chars: int
    ):
        """Add text source statistics."""
        self.text_chunks += chunks
        self.text_chars += chars
        
        self.sources.append(SourceStats(
            source_type="text",
            source_id=source_id,
            chunks=chunks,
            chars=chars,
            metadata={}
        ))
        
        if "text" not in self.extraction_methods:
            self.extraction_methods.append("text")
    
    def add_audio_source(
        self,
        source_id: str,
        duration_seconds: float,
        transcript_chars: int,
        chunks: int
    ):
        """Add audio/video source statistics."""
        self.audio_seconds += duration_seconds
        self.transcript_chars += transcript_chars
        self.transcript_chunks += chunks
        
        self.sources.append(SourceStats(
            source_type="audio",
            source_id=source_id,
            chunks=chunks,
            chars=transcript_chars,
            metadata={"duration_seconds": duration_seconds}
        ))
        
        if "audio" not in self.extraction_methods:
            self.extraction_methods.append("audio")
    
    def get_total_chunks(self) -> int:
        """Get total chunks across all sources."""
        return sum(s.chunks for s in self.sources)
    
    def get_total_chars(self) -> int:
        """Get total characters across all sources."""
        return sum(s.chars for s in self.sources)
    
    def get_avg_chunk_size(self) -> Optional[float]:
        """Calculate average chunk size across all sources."""
        total_chunks = self.get_total_chunks()
        if total_chunks == 0:
            return None
        
        total_chars = self.get_total_chars()
        return total_chars / total_chunks
    
    def to_ingestion_report_context(self):
        """
        Convert to IngestionReportContext for RunContext.
        
        Returns:
            IngestionReportContext instance
        """
        from src.reporting.run_context import IngestionReportContext
        
        return IngestionReportContext(
            # PDF metrics
            total_pages=self.pdf_pages,
            pages_ocr=self.pdf_pages_ocr,
            headers_removed=self.pdf_headers_removed,
            footers_removed=self.pdf_footers_removed,
            watermarks_removed=self.pdf_watermarks_removed,
            
            # URL metrics
            url_count=self.url_count,
            url_fetch_success_count=self.url_fetch_success,
            url_chunks_total=self.url_chunks,
            
            # Text metrics
            text_chars_total=self.text_chars,
            text_chunks_total=self.text_chunks,
            
            # Audio metrics
            audio_seconds=self.audio_seconds,
            transcript_chars=self.transcript_chars,
            transcript_chunks_total=self.transcript_chunks,
            
            # Overall
            chunks_total_all_sources=self.get_total_chunks(),
            avg_chunk_size_all_sources=self.get_avg_chunk_size(),
            extraction_methods=self.extraction_methods,
            sources_processed={
                "pdf": sum(1 for s in self.sources if s.source_type == "pdf"),
                "url": sum(1 for s in self.sources if s.source_type == "url"),
                "text": sum(1 for s in self.sources if s.source_type == "text"),
                "audio": sum(1 for s in self.sources if s.source_type == "audio")
            },
            total_text_length=self.get_total_chars()
        )
    
    def to_ingestion_report(self):
        """
        Convert to IngestionReport for reports.
        
        Returns:
            IngestionReport instance
        """
        from src.reporting.research_report import IngestionReport
        
        return IngestionReport(
            # PDF metrics
            total_pages=self.pdf_pages,
            pages_ocr=self.pdf_pages_ocr,
            headers_removed=self.pdf_headers_removed,
            footers_removed=self.pdf_footers_removed,
            watermarks_removed=self.pdf_watermarks_removed,
            
            # URL metrics
            url_count=self.url_count,
            url_fetch_success_count=self.url_fetch_success,
            url_chunks_total=self.url_chunks,
            
            # Text metrics
            text_chars_total=self.text_chars,
            text_chunks_total=self.text_chunks,
            
            # Audio metrics
            audio_seconds=self.audio_seconds,
            transcript_chars=self.transcript_chars,
            transcript_chunks_total=self.transcript_chunks,
            
            # Overall
            chunks_total_all_sources=self.get_total_chunks(),
            avg_chunk_size_all_sources=self.get_avg_chunk_size(),
            extraction_methods=self.extraction_methods
        )
    
    def validate(self) -> List[str]:
        """
        Validate statistics for consistency.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Invariant: chunks_total_all_sources must match sum of individual sources
        total_chunks = self.get_total_chunks()
        computed_chunks = self.url_chunks + self.text_chunks + self.transcript_chunks
        # Note: PDF chunks included in self.sources
        
        if total_chunks == 0 and computed_chunks > 0:
            errors.append(f"Total chunks is 0 but computed chunks is {computed_chunks}")
        
        # Invariant: if total_chunks > 0, avg_chunk_size must be computable
        if total_chunks > 0 and self.get_avg_chunk_size() is None:
            errors.append("Total chunks > 0 but cannot compute avg_chunk_size")
        
        # Invariant: url_fetch_success <= url_count
        if self.url_fetch_success > self.url_count:
            errors.append(
                f"URL fetch success ({self.url_fetch_success}) exceeds URL count ({self.url_count})"
            )
        
        # Invariant: pages_ocr <= total_pages
        if self.pdf_pages_ocr > self.pdf_pages:
            errors.append(
                f"PDF pages OCR ({self.pdf_pages_ocr}) exceeds total pages ({self.pdf_pages})"
            )
        
        return errors
    
    def log_summary(self):
        """Log summary of ingestion statistics."""
        logger.info("=" * 60)
        logger.info("INGESTION STATISTICS SUMMARY")
        logger.info("=" * 60)
        
        if self.pdf_pages > 0:
            logger.info(f"PDF: {self.pdf_pages} pages ({self.pdf_pages_ocr} OCR'd)")
        
        if self.url_count > 0:
            logger.info(
                f"URLs: {self.url_count} provided, "
                f"{self.url_fetch_success} fetched, "
                f"{self.url_chunks} chunks"
            )
        
        if self.text_chars > 0:
            logger.info(f"Text: {self.text_chars:,} chars, {self.text_chunks} chunks")
        
        if self.audio_seconds > 0:
            minutes = int(self.audio_seconds / 60)
            logger.info(
                f"Audio: {minutes}m {int(self.audio_seconds % 60)}s, "
                f"{self.transcript_chars:,} chars, "
                f"{self.transcript_chunks} chunks"
            )
        
        logger.info(f"TOTAL: {self.get_total_chunks()} chunks, {self.get_total_chars():,} chars")
        
        avg_size = self.get_avg_chunk_size()
        if avg_size:
            logger.info(f"Avg chunk size: {avg_size:.0f} chars")
        
        logger.info("=" * 60)
