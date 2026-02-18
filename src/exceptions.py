"""
Custom exceptions for Smart Notes application.

Error Codes:
    TEXT_TOO_SHORT: Extracted text below minimum threshold (100 chars)
    OCR_UNAVAILABLE: OCR required but not available
    PDF_PARSE_FAILED: PDF parsing/rendering failed
    OCR_FAILED: OCR processing encountered an error
"""

# Standard error codes for ingestion failures
INGESTION_ERRORS = {
    "TEXT_TOO_SHORT": {
        "description": "Extracted text is too short",
        "min_chars": 100,
        "user_message": "The document provided contains insufficient text for analysis.",
        "next_steps": [
            "Try providing a longer document",
            "Ensure the document has substantive content",
            "Check that the file isn't corrupted"
        ]
    },
    "OCR_UNAVAILABLE": {
        "description": "OCR not available but needed",
        "user_message": "The scanned document cannot be processed (OCR unavailable).",
        "next_steps": [
            "Install OCR dependencies: pip install easyocr python-bidi",
            "Provide a text-based PDF instead of a scanned document",
            "Try a different scanned document with clearer text"
        ]
    },
    "PDF_PARSE_FAILED": {
        "description": "PDF parsing or rendering failed",
        "user_message": "The PDF file could not be processed.",
        "next_steps": [
            "Verify the file is a valid PDF",
            "Try opening the PDF in Adobe Reader",
            "If corrupted, try using a different version of the file"
        ]
    },
    "OCR_FAILED": {
        "description": "OCR processing encountered an error",
        "user_message": "Text extraction via OCR failed.",
        "next_steps": [
            "Check that image quality is sufficient",
            "Try a scanned or photographed document with better clarity",
            "Ensure the document is in English or set the appropriate language"
        ]
    }
}


class EvidenceIngestError(Exception):
    """
    Raised when evidence ingestion fails in a way that prevents verification.
    
    This is distinct from:
    - General extraction errors (temporary issues)
    - Verification rejection (claims rejected due to insufficient evidence)
    
    Ingestion failure means the pipeline cannot even attempt verification
    because the evidence source is unusable or inaccessible.
    
    Error Codes:
        TEXT_TOO_SHORT: Extracted text below minimum threshold
        OCR_UNAVAILABLE: OCR required but dependencies missing
        PDF_PARSE_FAILED: PDF file cannot be parsed
        OCR_FAILED: OCR processing error
    """
    
    def __init__(self, code: str, message: str = None, details: dict = None):
        """
        Initialize EvidenceIngestError.
        
        Args:
            code: Error code (one of INGESTION_ERRORS keys)
            message: Optional detailed error message
            details: Optional dict with additional context (e.g., chars_extracted)
        """
        self.code = code
        self.message = message or INGESTION_ERRORS.get(code, {}).get("description", code)
        self.details = details or {}
        self.is_ingestion_failure = True  # Flag to distinguish from rejection
        super().__init__(f"{code}: {self.message}")
    
    def get_user_message(self) -> str:
        """Get user-friendly error message."""
        return INGESTION_ERRORS.get(self.code, {}).get("user_message", self.message)
    
    def get_next_steps(self) -> list:
        """Get suggested next steps for user."""
        return INGESTION_ERRORS.get(self.code, {}).get("next_steps", [])
    
    def is_user_recoverable(self) -> bool:
        """Check if user can recover from this error."""
        return self.code in ["TEXT_TOO_SHORT", "OCR_UNAVAILABLE", "PDF_PARSE_FAILED"]
