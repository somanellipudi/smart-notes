"""
Noise injection utilities for robustness evaluation.

Simulates common document ingestion issues:
- Headers/footers from scanned documents
- OCR typos and character confusion
- Column layout issues (two-column text interleaving)
"""

import random
import re
from typing import Optional


def inject_headers_footers(
    text: str,
    header: str = "UNIT 5",
    footer: str = "Scanned with CamScanner",
    freq: int = 10,
    seed: Optional[int] = None
) -> str:
    """
    Inject headers and footers at regular intervals to simulate scanned documents.
    
    Args:
        text: Input text
        header: Header text to inject
        footer: Footer text to inject
        freq: Inject every N lines (default: 10)
        seed: Random seed for reproducibility
        
    Returns:
        Text with headers/footers injected
    """
    if seed is not None:
        random.seed(seed)
    
    lines = text.split('\n')
    result = []
    
    for i, line in enumerate(lines):
        # Add header before the line at intervals
        if i > 0 and i % freq == 0:
            result.append(f"--- {header} ---")
        
        result.append(line)
        
        # Add footer after the line at intervals
        if (i + 1) % freq == 0 and i < len(lines) - 1:
            result.append(f"[{footer}]")
    
    return '\n'.join(result)


def inject_ocr_typos(
    text: str,
    rate: float = 0.01,
    seed: Optional[int] = None
) -> str:
    """
    Inject OCR-style character substitution errors.
    
    Common OCR confusions:
    - l ↔ 1 (lowercase L vs digit one)
    - O ↔ 0 (uppercase O vs zero)
    - rn ↔ m (two chars vs one)
    - I ↔ l (uppercase i vs lowercase L)
    
    Args:
        text: Input text
        rate: Probability of substitution per character (default: 0.01 = 1%)
        seed: Random seed for reproducibility
        
    Returns:
        Text with OCR typos injected
    """
    if seed is not None:
        random.seed(seed)
    
    # OCR confusion pairs
    substitutions = {
        'l': '1',
        '1': 'l',
        'O': '0',
        '0': 'O',
        'I': 'l',
        'i': '1',
    }
    
    # Special case: rn ↔ m
    # Handle this first before character-level substitutions
    result = list(text)
    chars_to_skip = set()
    
    # First pass: handle multi-character substitutions (rn ↔ m)
    i = 0
    while i < len(result) - 1:
        if random.random() < rate:
            if result[i:i+2] == ['r', 'n']:
                # Replace 'rn' with 'm'
                result[i] = 'm'
                result[i+1] = ''
                chars_to_skip.add(i+1)
                i += 2
                continue
            elif result[i] == 'm':
                # Replace 'm' with 'rn'
                result[i] = 'rn'
                i += 1
                continue
        i += 1
    
    # Remove empty strings from rn→m substitution
    result = [c for idx, c in enumerate(result) if idx not in chars_to_skip]
    
    # Second pass: single character substitutions
    for i in range(len(result)):
        char = result[i]
        if char in substitutions and random.random() < rate:
            result[i] = substitutions[char]
    
    return ''.join(result)


def inject_column_shuffle(
    text: str,
    seed: Optional[int] = None
) -> str:
    """
    Simulate two-column layout interleaving issues.
    
    Common in PDFs where text extraction incorrectly alternates between
    left and right columns instead of reading left column fully, then right.
    
    Strategy:
    - Split text into sentences
    - Divide sentences into two pseudo-columns (alternating)
    - Interleave shorter chunks from each column
    
    Args:
        text: Input text
        seed: Random seed for reproducibility
        
    Returns:
        Text with column shuffle applied
    """
    if seed is not None:
        random.seed(seed)
    
    # Split into sentences (rough approximation)
    sentences = re.split(r'([.!?]+\s+)', text)
    
    # Recombine sentence + punctuation
    reconstructed = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            reconstructed.append(sentences[i] + sentences[i + 1])
        else:
            reconstructed.append(sentences[i])
    
    if not reconstructed:
        return text
    
    # Divide into two pseudo-columns
    mid = len(reconstructed) // 2
    left_col = reconstructed[:mid]
    right_col = reconstructed[mid:]
    
    # Interleave: take 1-2 sentences from each column alternately
    shuffled = []
    left_idx = 0
    right_idx = 0
    
    while left_idx < len(left_col) or right_idx < len(right_col):
        # Take from left column
        chunk_size = random.randint(1, 2)
        for _ in range(chunk_size):
            if left_idx < len(left_col):
                shuffled.append(left_col[left_idx])
                left_idx += 1
        
        # Take from right column
        chunk_size = random.randint(1, 2)
        for _ in range(chunk_size):
            if right_idx < len(right_col):
                shuffled.append(right_col[right_idx])
                right_idx += 1
    
    return ' '.join(shuffled)


def inject_all_noise(
    text: str,
    header: str = "UNIT 5",
    footer: str = "Scanned with CamScanner",
    header_freq: int = 10,
    ocr_rate: float = 0.01,
    apply_headers: bool = True,
    apply_ocr: bool = True,
    apply_shuffle: bool = True,
    seed: Optional[int] = None
) -> str:
    """
    Apply all noise injection types in sequence.
    
    Args:
        text: Input text
        header: Header text for header/footer injection
        footer: Footer text for header/footer injection
        header_freq: Frequency for header/footer injection
        ocr_rate: OCR typo rate
        apply_headers: Whether to inject headers/footers
        apply_ocr: Whether to inject OCR typos
        apply_shuffle: Whether to apply column shuffle
        seed: Random seed for reproducibility
        
    Returns:
        Text with all specified noise types applied
    """
    result = text
    
    # Order: headers first (structural), then shuffle (reordering), then OCR (character-level)
    if apply_headers:
        result = inject_headers_footers(
            result,
            header=header,
            footer=footer,
            freq=header_freq,
            seed=seed
        )
    
    if apply_shuffle:
        result = inject_column_shuffle(result, seed=seed)
    
    if apply_ocr:
        result = inject_ocr_typos(result, rate=ocr_rate, seed=seed)
    
    return result
