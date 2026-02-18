"""
Text preprocessing module for cleaning and segmenting classroom content.

This module handles normalization, cleaning, and intelligent segmentation of
both handwritten notes and lecture transcripts, preparing them for downstream
GenAI reasoning tasks.
"""

import re
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass

from src.preprocessing.text_cleaner import clean_extracted_text
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

import config

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


@dataclass
class TextSegment:
    """
    Represents a segmented piece of classroom content.
    
    Attributes:
        text: The segment content
        start_time: Start timestamp (if from audio)
        end_time: End timestamp (if from audio)
        source: Origin of segment ('notes', 'transcript', 'combined')
        topic_hint: Optional topic boundary indicator
    """
    text: str
    start_time: float = None
    end_time: float = None
    source: str = "unknown"
    topic_hint: str = None


class TextPreprocessor:
    """
    Preprocesses and segments classroom text content.
    
    This class provides methods for cleaning, normalizing, and intelligently
    segmenting educational content based on topic boundaries, timestamps,
    and discourse markers.
    """
    
    def __init__(self):
        """Initialize the text preprocessor with necessary resources."""
        if NLTK_AVAILABLE:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                logger.info("Downloading NLTK punkt tokenizer")
                nltk.download('punkt', quiet=True)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Removes excessive whitespace, normalizes special characters,
        and standardizes formatting for educational content processing.
        
        Args:
            text: Raw text to clean
        
        Returns:
            Cleaned and normalized text
        
        Example:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.clean_text("Today:  Derivatives\\n\\n  Rate of change")
            "Today: Derivatives. Rate of change"
        """
        if not text:
            return ""
        
        text, _ = clean_extracted_text(text)

        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize common educational notation
        text = text.replace('–', '-')  # En dash to hyphen
        text = text.replace('—', '-')  # Em dash to hyphen
        text = text.replace('\u2018', "'")  # Left single quotation mark to straight quote
        text = text.replace('\u2019', "'")  # Right single quotation mark to straight quote
        text = text.replace('\u201c', '"')  # Left double quotation mark
        text = text.replace('\u201d', '"')  # Right double quotation mark
        
        # Remove repeated punctuation (e.g., "!!!" -> "!")
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        
        # Ensure spacing after punctuation
        text = re.sub(r'([.!?,;:])([A-Za-z])', r'\1 \2', text)
        
        # Strip and return
        return text.strip()
    
    def normalize_equations(self, equations: List[str]) -> List[str]:
        """
        Normalize equation notation for consistency.
        
        Args:
            equations: List of equation strings
        
        Returns:
            List of normalized equations
        """
        normalized = []
        for eq in equations:
            # Remove excessive spaces
            eq = re.sub(r'\s+', ' ', eq.strip())
            
            # Standardize common notation
            eq = eq.replace('×', '*')
            eq = eq.replace('÷', '/')
            
            normalized.append(eq)
        
        return normalized
    
    def segment_by_time(
        self,
        transcript_segments: List[Dict],
        max_segment_length: int = None
    ) -> List[TextSegment]:
        """
        Segment transcript by time boundaries.
        
        Groups transcript segments into meaningful chunks based on
        timestamp information and length constraints.
        
        Args:
            transcript_segments: List of segments from audio transcription
            max_segment_length: Maximum characters per segment
        
        Returns:
            List of TextSegment objects
        """
        max_len = max_segment_length or config.MAX_SEGMENT_LENGTH
        segments = []
        
        current_text = ""
        current_start = None
        current_end = None
        
        for seg in transcript_segments:
            seg_text = seg.get("text", "").strip()
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)
            
            if current_start is None:
                current_start = seg_start
            
            # Check if adding this would exceed max length
            if len(current_text) + len(seg_text) > max_len and current_text:
                # Save current segment
                segments.append(TextSegment(
                    text=self.clean_text(current_text),
                    start_time=current_start,
                    end_time=current_end,
                    source="transcript"
                ))
                
                # Start new segment
                current_text = seg_text
                current_start = seg_start
                current_end = seg_end
            else:
                # Accumulate
                current_text += " " + seg_text if current_text else seg_text
                current_end = seg_end
        
        # Add final segment
        if current_text:
            segments.append(TextSegment(
                text=self.clean_text(current_text),
                start_time=current_start,
                end_time=current_end,
                source="transcript"
            ))
        
        logger.info(f"Created {len(segments)} time-based segments")
        return segments
    
    def segment_by_topic(self, text: str) -> List[TextSegment]:
        """
        Segment text by topic boundaries.
        
        Identifies topic transitions using discourse markers, section headers,
        and semantic cues common in educational content.
        
        Args:
            text: Full text to segment
        
        Returns:
            List of topic-based TextSegment objects
        
        Example:
            Detects transitions like:
            - "Next, we'll discuss..."
            - "Moving on to..."
            - "Now let's talk about..."
        """
        # Topic boundary markers in educational discourse
        boundary_patterns = [
            r'(?:^|\n)\s*(?:Today|Now|Next|First|Second|Third|Finally)',
            r'(?:let\'?s|we\'?ll|we\'?re going to)\s+(?:talk about|discuss|look at|move on to|turn to)',
            r'(?:Moving on to|Turning to|Switching to)',
            r'\n\s*\d+\.',  # Numbered sections
            r'\n\s*[A-Z][a-z]+:',  # Section headers like "Topic:"
        ]
        
        combined_pattern = '|'.join(f'({p})' for p in boundary_patterns)
        
        # Find all boundary positions
        boundaries = [0]  # Start with beginning
        for match in re.finditer(combined_pattern, text, re.IGNORECASE | re.MULTILINE):
            pos = match.start()
            if pos > config.MIN_SEGMENT_LENGTH:  # Avoid very short segments
                boundaries.append(pos)
        boundaries.append(len(text))  # Add end
        
        # Create segments
        segments = []
        for i in range(len(boundaries) - 1):
            start_pos = boundaries[i]
            end_pos = boundaries[i + 1]
            segment_text = text[start_pos:end_pos]
            
            if len(segment_text.strip()) >= config.MIN_SEGMENT_LENGTH:
                segments.append(TextSegment(
                    text=self.clean_text(segment_text),
                    source="notes",
                    topic_hint=self._extract_topic_hint(segment_text)
                ))
        
        # If no boundaries found, create single segment
        if not segments:
            segments.append(TextSegment(
                text=self.clean_text(text),
                source="notes"
            ))
        
        logger.info(f"Created {len(segments)} topic-based segments")
        return segments
    
    def _extract_topic_hint(self, text: str) -> str:
        """
        Extract topic hint from segment text.
        
        Args:
            text: Segment text
        
        Returns:
            Topic hint string or None
        """
        # Look for common topic indicators
        patterns = [
            r'(?:topic|subject|today):\s*([^\n.]+)',
            r'(?:discuss|talk about|focus on|cover)\s+([^\n.]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()[:100]  # Limit length
        
        return None
    
    def combine_sources(
        self,
        notes: str,
        transcript: str,
        equations: List[str] = None
    ) -> Dict:
        """
        Combine and align multiple content sources.
        
        Merges handwritten notes, lecture transcript, and equations into
        a unified representation suitable for multi-stage reasoning.
        
        Args:
            notes: Cleaned handwritten notes text
            transcript: Lecture transcript text
            equations: List of equation strings
        
        Returns:
            Dictionary containing:
                - combined_text: Integrated text content
                - notes_segments: Segmented notes
                - transcript_segments: Segmented transcript
                - equations: Normalized equations
                - metadata: Processing information
        """
        # Clean inputs
        notes_clean = self.clean_text(notes) if notes else ""
        transcript_clean = self.clean_text(transcript) if transcript else ""
        equations_clean = self.normalize_equations(equations or [])
        
        # Segment each source
        notes_segments = []
        if notes_clean:
            notes_segments = self.segment_by_topic(notes_clean)
        
        transcript_segments = []
        if transcript_clean:
            # Create simple segments from full transcript
            # (In production, would use time-based segments from transcription)
            sentences = self._split_sentences(transcript_clean)
            transcript_segments = self._group_sentences(sentences)
        
        # Combine all text
        combined_parts = []
        if notes_clean:
            combined_parts.append(f"NOTES:\n{notes_clean}")
        if transcript_clean:
            combined_parts.append(f"TRANSCRIPT:\n{transcript_clean}")
        if equations_clean:
            combined_parts.append(f"EQUATIONS:\n" + "\n".join(equations_clean))
        
        combined_text = "\n\n".join(combined_parts)
        
        logger.info(
            f"Combined sources: {len(notes_clean)} chars notes, "
            f"{len(transcript_clean)} chars transcript, "
            f"{len(equations_clean)} equations"
        )
        
        return {
            "combined_text": combined_text,
            "notes_segments": notes_segments,
            "transcript_segments": transcript_segments,
            "equations": equations_clean,
            "metadata": {
                "notes_length": len(notes_clean),
                "transcript_length": len(transcript_clean),
                "num_equations": len(equations_clean),
                "num_segments": len(notes_segments) + len(transcript_segments)
            }
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK or regex fallback."""
        if NLTK_AVAILABLE:
            try:
                return sent_tokenize(text)
            except Exception as e:
                logger.warning(f"NLTK sentence tokenization failed: {e}")
        
        # Fallback: simple regex-based splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _group_sentences(
        self,
        sentences: List[str],
        target_length: int = None
    ) -> List[TextSegment]:
        """Group sentences into segments of appropriate length."""
        target = target_length or config.MAX_SEGMENT_LENGTH // 2
        segments = []
        current_group = []
        current_length = 0
        
        for sentence in sentences:
            sent_len = len(sentence)
            
            if current_length + sent_len > target and current_group:
                # Save current group
                segments.append(TextSegment(
                    text=" ".join(current_group),
                    source="transcript"
                ))
                current_group = [sentence]
                current_length = sent_len
            else:
                current_group.append(sentence)
                current_length += sent_len
        
        # Add final group
        if current_group:
            segments.append(TextSegment(
                text=" ".join(current_group),
                source="transcript"
            ))
        
        return segments


def preprocess_classroom_content(
    handwritten_notes: str,
    transcript: str,
    equations: List[str] = None
) -> Dict:
    """
    Convenience function to preprocess all classroom content.
    
    Args:
        handwritten_notes: Raw notes text
        transcript: Lecture transcript
        equations: List of equation strings
    
    Returns:
        Preprocessed content dictionary
    """
    preprocessor = TextPreprocessor()
    return preprocessor.combine_sources(handwritten_notes, transcript, equations)
