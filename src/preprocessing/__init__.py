"""Text preprocessing module for classroom content normalization."""

from .text_processing import (
    TextPreprocessor,
    TextSegment,
    preprocess_classroom_content
)

__all__ = ["TextPreprocessor", "TextSegment", "preprocess_classroom_content"]
