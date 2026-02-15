"""
Debug script to test PDF extraction with Streamlit file handling
"""
import io
from src.preprocessing.pdf_ingest import extract_pdf_text

# Simulate a Streamlit UploadedFile behavior
class MockUploadedFile:
    def __init__(self, file_bytes, name):
        self._bytes = file_bytes
        self.name = name
        self._position = 0
    
    def read(self):
        """Reads from current position to end"""
        data = self._bytes[self._position:]
        self._position = len(self._bytes)
        return data
    
    def seek(self, position):
        """Reset file pointer"""
        self._position = position
    
    def getvalue(self):
        """Get all bytes without changing position"""
        return self._bytes

# Create a minimal valid PDF
pdf_header = b'%PDF-1.4\n'
pdf_content = b'1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n'
pdf_content += b'2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n'
pdf_content += b'3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n'
pdf_content += b'4 0 obj\n<< /Length 44 >>\nstream\nBT\n/F1 12 Tf\n100 700 Td\n(Hello World) Tj\nET\nendstream\nendobj\n'
pdf_content += b'xref\n0 5\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\n0000000214 00000 n\n'
pdf_content += b'trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n316\n%%EOF'
pdf_bytes = pdf_header + pdf_content

print("Testing PDF extraction with Streamlit file handling...")
print(f"PDF size: {len(pdf_bytes)} bytes")

# Test 1: Using read() - simulates current bug
print("\n=== Test 1: Using read() (current implementation) ===")
mock_file = MockUploadedFile(pdf_bytes, "test.pdf")
print(f"Before read: position=0, hasattr(name)={hasattr(mock_file, 'name')}")
first_read = mock_file.read()
print(f"First read: {len(first_read)} bytes")
second_read = mock_file.read()
print(f"Second read: {len(second_read)} bytes (BUG: should be same as first!)")

# Test 2: Using getvalue() - potential fix
print("\n=== Test 2: Using getvalue() (proposed fix) ===")
mock_file2 = MockUploadedFile(pdf_bytes, "test.pdf")
first_get = mock_file2.getvalue()
print(f"First getvalue(): {len(first_get)} bytes")
second_get = mock_file2.getvalue()
print(f"Second getvalue(): {len(second_get)} bytes (GOOD: same size!)")

print("\n=== DIAGNOSIS ===")
print("✓ Problem identified: read() moves file pointer")
print("✓ Solution: Use getvalue() or seek(0) before each read()")
print("\n=== REQUIRED FIX ===")
print("In src/preprocessing/pdf_ingest.py line ~94:")
print("  Change: file_bytes = uploaded_file.read()")
print("  To: file_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, 'getvalue') else uploaded_file.read()")
