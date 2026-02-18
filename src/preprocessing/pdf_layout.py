"""
Layout-aware PDF text processing: multi-column detection and reordering.

This module provides heuristics for detecting multi-column layouts
and reordering text to preserve reading order.
"""

import logging
import re
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


def detect_multicolumn(lines: List[str], threshold: float = 0.30) -> bool:
    """
    Detect if text is likely multi-column layout.
    
    Heuristics:
    - Many short lines (< 60 chars)
    - Unnatural line breaks mid-sentence
    - Mixed indentation patterns
    
    Args:
        lines: List of text lines
        threshold: Fraction of short lines to trigger detection
    
    Returns:
        True if multi-column layout likely detected
    """
    if len(lines) < 10:
        return False
    
    # Count short lines (likely column breaks)
    SHORT_LINE_THRESHOLD = 60
    short_lines = sum(1 for line in lines if 10 < len(line.strip()) < SHORT_LINE_THRESHOLD)
    short_ratio = short_lines / len(lines)
    
    # Count mid-sentence breaks (line ends without punctuation)
    mid_sentence_breaks = 0
    for line in lines:
        stripped = line.strip()
        if len(stripped) > 20 and stripped[-1] not in '.!?,;:)]}\"\'':
            # Check if next char would be lowercase (continuation)
            if any(c.islower() for c in stripped[-5:]):
                mid_sentence_breaks += 1
    
    mid_sentence_ratio = mid_sentence_breaks / max(len(lines), 1)
    
    # Mixed indentation patterns (column alignment)
    indent_lengths = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    if len(indent_lengths) > 5:
        unique_indents = len(set(indent_lengths))
        indent_diversity = unique_indents / len(indent_lengths)
    else:
        indent_diversity = 0.0
    
    # Decision
    is_multicolumn = (
        short_ratio > threshold or
        (mid_sentence_ratio > 0.20 and short_ratio > 0.15) or
        (indent_diversity > 0.30 and short_ratio > 0.20)
    )
    
    if is_multicolumn:
        logger.info(
            f"Multi-column layout detected: short_ratio={short_ratio:.2f}, "
            f"mid_sentence={mid_sentence_ratio:.2f}, indent_diversity={indent_diversity:.2f}"
        )
    
    return is_multicolumn


def reorder_columns(text: str, safe_mode: bool = True) -> str:
    """
    Attempt to reorder multi-column text into reading order.
    
    Strategy:
    - Split into lines
    - Detect column boundaries by indentation/spacing
    - Reorder left-to-right, top-to-bottom
    
    Args:
        text: Raw text (potentially multi-column)
        safe_mode: If True, return original text if uncertain
    
    Returns:
        Reordered text (or original if uncertain)
    """
    lines = text.split('\n')
    
    if not detect_multicolumn(lines):
        # Not multi-column, return as-is
        return text
    
    # Simple heuristic: group by indentation levels
    # Assume 2 columns with different starting positions
    
    # Calculate indentation for each non-empty line
    line_data = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped:
            indent = len(line) - len(line.lstrip())
            line_data.append({
                'index': i,
                'indent': indent,
                'text': stripped,
                'original': line
            })
    
    if len(line_data) < 10:
        # Too few lines to confidently reorder
        return text
    
    # Find common indentation levels (likely column starts)
    indent_counts = {}
    for ld in line_data:
        indent = ld['indent']
        indent_counts[indent] = indent_counts.get(indent, 0) + 1
    
    # Get top 2 most common indents (likely 2 columns)
    sorted_indents = sorted(indent_counts.items(), key=lambda x: x[1], reverse=True)
    
    if len(sorted_indents) < 2:
        # Only one indent level, not multi-column
        return text
    
    left_indent, left_count = sorted_indents[0]
    right_indent, right_count = sorted_indents[1]
    
    # Ensure left < right (column order)
    if left_indent > right_indent:
        left_indent, right_indent = right_indent, left_indent
        left_count, right_count = right_count, left_count
    
    # Check if this looks like a valid column split
    total_lines = len(line_data)
    if (left_count + right_count) / total_lines < 0.40:
        # Not enough lines in these two columns
        if safe_mode:
            logger.warning("Column reordering uncertain, returning original")
            return text
    
    # Group lines by column
    left_column = []
    right_column = []
    other_lines = []
    
    INDENT_TOLERANCE = 5
    
    for ld in line_data:
        if abs(ld['indent'] - left_indent) <= INDENT_TOLERANCE:
            left_column.append(ld)
        elif abs(ld['indent'] - right_indent) <= INDENT_TOLERANCE:
            right_column.append(ld)
        else:
            other_lines.append(ld)
    
    # Interleave columns by vertical position
    # Assume columns are side-by-side, maintain vertical order
    reordered = []
    
    # Merge by original index (top-to-bottom)
    all_lines_sorted = sorted(line_data, key=lambda x: x['index'])
    
    # Simple strategy: output left column chunk, then right column chunk at same height
    # For now, just concatenate left then right (safe fallback)
    for ld in left_column:
        reordered.append(ld['text'])
    
    for ld in right_column:
        reordered.append(ld['text'])
    
    for ld in other_lines:
        reordered.append(ld['text'])
    
    reordered_text = '\n'.join(reordered)
    
    logger.info(
        f"Reordered {len(line_data)} lines: "
        f"left_col={len(left_column)}, right_col={len(right_column)}, other={len(other_lines)}"
    )
    
    return reordered_text


def split_into_columns(
    lines: List[str],
    num_columns: int = 2
) -> List[List[str]]:
    """
    Split lines into columns by indentation clustering.
    
    Args:
        lines: List of text lines
        num_columns: Expected number of columns
    
    Returns:
        List of column line groups
    """
    # This is a placeholder for more sophisticated column detection
    # For now, return single column
    return [lines]


def merge_columns_interleaved(columns: List[List[str]]) -> str:
    """
    Merge columns by interleaving (paragraph-aware).
    
    Args:
        columns: List of column line groups
    
    Returns:
        Merged text in reading order
    """
    if len(columns) == 1:
        return '\n'.join(columns[0])
    
    # Simple interleave by line count
    merged = []
    max_len = max(len(col) for col in columns)
    
    for i in range(max_len):
        for col in columns:
            if i < len(col):
                merged.append(col[i])
    
    return '\n'.join(merged)
