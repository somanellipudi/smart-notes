# OCR Configuration Guide

## Overview

Smart Notes automatically disables OCR (Optical Character Recognition) when deployed on Streamlit Cloud to avoid heavy dependency issues. OCR is enabled by default for local deployments.

## How It Works

### Automatic Detection

The system detects Streamlit Cloud environment by checking:
1. `STREAMLIT_SHARING` environment variable
2. `STREAMLIT_CLOUD` environment variable
3. Home directory path heuristics (`/home/appuser`)

### Default Behavior

| Environment | OCR_ENABLED | ENABLE_OCR_FALLBACK |
|------------|-------------|---------------------|
| Local | ✅ True | ✅ True |
| Streamlit Cloud | ❌ False | ❌ False |

### User Experience

#### When OCR is Disabled (Streamlit Cloud)

Users see:
- Warning banner at the top: "⚠️ OCR is disabled in hosted mode"
- Image/PDF upload option replaced with info message
- Clear guidance to paste text directly instead

#### When OCR is Enabled (Local)

Users can:
- Upload images (JPG, PNG, BMP)
- Upload scanned PDFs
- Full OCR extraction with LLM correction

## Manual Configuration

### Enable OCR on Cloud (Advanced)

If you have a custom Streamlit Cloud deployment with OCR dependencies installed:

**Option 1: Environment Variable**
```bash
OCR_ENABLED=true
```

**Option 2: Streamlit Secrets**
```toml
# .streamlit/secrets.toml
OCR_ENABLED = "true"
```

### Disable OCR Locally (Testing)

To test the Cloud experience locally:

```bash
OCR_ENABLED=false
```

## Testing

Run the test script to verify OCR configuration:

```bash
python test_ocr_config.py
```

Output shows:
- Environment detection (Local vs Cloud)
- OCR status (Enabled/Disabled)
- Available OCR packages

### Simulate Cloud Environment

```bash
# Windows PowerShell
$env:STREAMLIT_CLOUD="true"; python test_ocr_config.py; Remove-Item Env:\STREAMLIT_CLOUD

# Linux/Mac
STREAMLIT_CLOUD=true python test_ocr_config.py
```

## Configuration Variables

### Master Switch

**`OCR_ENABLED`**
- Type: Boolean
- Default: `true` (local), `false` (Cloud)
- Controls: All OCR functionality
- Override: Can be set explicitly via environment variable

### PDF OCR Fallback

**`ENABLE_OCR_FALLBACK`**
- Type: Boolean
- Default: `OCR_ENABLED AND true`
- Controls: OCR fallback for low-quality PDF pages
- Note: Automatically disabled when `OCR_ENABLED=false`

### OCR Parameters

**`OCR_MAX_PAGES`**
- Type: Integer
- Default: `10`
- Description: Maximum number of PDF pages to OCR

**`OCR_DPI`**
- Type: Integer
- Default: `200`
- Description: DPI for PDF rendering before OCR

## Code Implementation

### Detection Logic (config.py)

```python
def _is_streamlit_cloud() -> bool:
    """Detect if running on Streamlit Cloud."""
    if os.getenv("STREAMLIT_SHARING") == "true":
        return True
    if os.getenv("STREAMLIT_CLOUD") == "true":
        return True
    
    home = os.path.expanduser("~")
    if "/home/appuser" in home or "\\appuser" in home:
        return True
    
    return False

IS_STREAMLIT_CLOUD = _is_streamlit_cloud()

# OCR enabled by default locally, disabled on Cloud
_ocr_default = "false" if IS_STREAMLIT_CLOUD else "true"
OCR_ENABLED = os.getenv("OCR_ENABLED", _ocr_default).lower() == "true"
```

### UI Integration (app.py)

```python
# Warning banner
if not config.OCR_ENABLED:
    st.warning(
        "⚠️ **OCR is disabled in hosted mode** — Upload images and scanned PDFs are not supported. "
        "For scanned content, please paste text directly or run locally with OCR enabled."
    )

# Image upload disabled
if not config.OCR_ENABLED:
    st.info(
        "ℹ️ **Image/PDF upload is disabled** because OCR is not available. "
        "Please use the 'Type/Paste Text' option instead."
    )
```

### Processing Guard (app.py)

```python
# Process Images with OCR
if image_files:
    if not config.OCR_ENABLED:
        st.warning(
            "⚠️ **OCR is disabled** - Cannot process uploaded images. "
            "Please paste text directly or enable OCR in local deployment."
        )
        logger.warning("Skipping image OCR - OCR_ENABLED=False")
    else:
        # Proceed with OCR processing
        ...
```

## Dependencies

OCR requires these packages (automatically installed locally):
- `easyocr==1.7.1` (primary OCR engine)
- `pytesseract>=0.3.10` (fallback OCR engine)
- `Pillow>=10.0.0` (image processing)
- `opencv-python-headless>=4.8.0` (image preprocessing)
- `python-bidi>=0.4.2` (RTL text support)
- `arabic-reshaper>=0.0.7` (Arabic text support)

These are included in `requirements.txt` but won't be loaded if `OCR_ENABLED=false`.

## Troubleshooting

### Issue: OCR disabled when it should be enabled

**Check:**
```bash
python test_ocr_config.py
```

**Common causes:**
- `OCR_ENABLED=false` set in environment
- Missing OCR dependencies
- Streamlit Cloud detected incorrectly

### Issue: OCR enabled on Cloud but failing

**Solution:**
OCR dependencies are too heavy for Streamlit Cloud free tier. Either:
1. Deploy to custom infrastructure with OCR packages
2. Set `OCR_ENABLED=false` and guide users to paste text
3. Use external OCR API service instead

### Issue: Local deployment has OCR disabled

**Fix:**
1. Check environment variables: `echo $OCR_ENABLED`
2. Remove any `.env` entries setting `OCR_ENABLED=false`
3. Restart the application

## Best Practices

### For Hosted Deployments (Streamlit Cloud)

✅ **Do:**
- Keep `OCR_ENABLED=false` (default)
- Show clear instructions for text input
- Guide users to local deployment for OCR needs

❌ **Don't:**
- Force enable OCR without proper infrastructure
- Hide the OCR disabled message (transparency)

### For Local Deployments

✅ **Do:**
- Keep `OCR_ENABLED=true` (default)
- Install all OCR dependencies
- Test OCR functionality before deployment

❌ **Don't:**
- Disable OCR unless testing Cloud behavior
- Skip dependency installation

### For Custom Infrastructure

✅ **Do:**
- Ensure OCR dependencies are installed
- Set `OCR_ENABLED=true` explicitly
- Monitor memory usage (OCR is memory-intensive)
- Consider OCR_MAX_PAGES limit for cost control

## Performance Considerations

### Memory Usage

- EasyOCR initial load: ~500 MB
- Per-image processing: ~100-200 MB
- Cached model: Persistent in memory

### Processing Time

- Image OCR: 2-5 seconds per image
- PDF OCR fallback: 3-10 seconds per page
- LLM correction: +2-3 seconds

### Recommendations

- Limit `OCR_MAX_PAGES` in production
- Enable caching for repeated images
- Consider batch processing for multiple images

## Related Files

- `config.py` - OCR configuration and detection
- `app.py` - UI integration and guards
- `src/audio/image_ocr.py` - OCR implementation
- `src/preprocessing/pdf_ingest.py` - PDF OCR fallback
- `test_ocr_config.py` - Configuration testing tool
- `.streamlit/secrets.toml` - Deployment secrets
- `requirements.txt` - OCR dependencies
