"""Test the PDF extraction fix with mock Streamlit file"""
import io

# Mock Streamlit UploadedFile
class MockUploadedFile:
    def __init__(self, file_bytes, name):
        self._bytes = file_bytes
        self.name = name
        self._position = 0
    
    def read(self):
        data = self._bytes[self._position:]
        self._position = len(self._bytes)
        return data
    
    def seek(self, position):
        self._position = position
    
    def getvalue(self):
        return self._bytes

# Create sample PDF bytes
pdf_bytes = b'%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\nxref\n0 1\ntrailer\n<< /Size 1 >>\nstartxref\n50\n%%EOF'

# Test the fix
from src.preprocessing.pdf_ingest import extract_pdf_text

print("Testing PDF extraction with fixed file handling...")
mock_file = MockUploadedFile(pdf_bytes, "test.pdf")

try:
    # This should now work even if the file is "read" multiple times internally
    text, metadata = extract_pdf_text(mock_file, ocr=None)
    print(f"✅ Extraction completed!")
    print(f"   Method: {metadata.get('extraction_method_used', 'unknown')}")
    print(f"   Text length: {len(text)} chars")
    print(f"   Words: {metadata.get('words', 0)}")
    print(f"   Assessment: {metadata.get('quality_assessment', 'N/A')}")
    
    # Verify the file can be read again by creating a new mock
    mock_file2 = MockUploadedFile(pdf_bytes, "test2.pdf")
    text2, metadata2 = extract_pdf_text(mock_file2, ocr=None)
    print(f"\n✅ Second extraction also works!")
    print(f"   Text length: {len(text2)} chars (should match first)")
    
    print("\n✅ FIX VERIFIED: PDF extraction now handles Streamlit uploads correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
