"""Ingestion module for Smart Notes."""

from .document_ingestor import (
    ingest_document,
    detect_scanned_or_low_text,
    extract_text_from_pdf_pymupdf,
    extract_text_from_pdf_pdfplumber,
    extract_text_from_pdf_ocr,
    extract_text_from_image,
    IngestionDiagnostics,
)

__all__ = [
    "ingest_document",
    "detect_scanned_or_low_text",
    "extract_text_from_pdf_pymupdf",
    "extract_text_from_pdf_pdfplumber",
    "extract_text_from_pdf_ocr",
    "extract_text_from_image",
    "IngestionDiagnostics",
]
