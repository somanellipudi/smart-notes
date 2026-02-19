# URL Provenance & Multi-Source Stats - Quick Reference

## 🎯 What Changed

### Citations Now Show:
- ✅ **URLs**: `[https://example.com](https://example.com)` (clickable)
- ✅ **PDFs**: `📄 textbook.pdf p.42`
- ✅ **YouTube**: `▶️ [video](url) ⏱️ 01:23-02:45`
- ✅ **Audio**: `🎤 lecture.mp3 ⏱️ 12:30-13:15`
- ✅ **Notes**: `📝 user_notes`

### Stats Now Track:
- ✅ **PDF**: Pages, OCR, headers/footers/watermarks removed
- ✅ **URLs**: Count, fetch success, chunks
- ✅ **Text**: Characters, chunks
- ✅ **Audio**: Duration, transcript chars, chunks
- ✅ **Overall**: Total chunks across all sources, avg chunk size

## 📋 Key Files

| Component | File | Purpose |
|-----------|------|---------|
| **Schema** | `src/schema/verifiable_schema.py` | EvidenceCitation with provenance |
| **Evidence** | `src/retrieval/semantic_retriever.py` | EvidenceSpan with origin/timestamp |
| **Evidence Store** | `src/retrieval/evidence_store.py` | Evidence with provenance |
| **Stats Aggregator** | `src/reporting/ingestion_stats.py` | Multi-source stats collection |
| **Report Context** | `src/reporting/run_context.py` | IngestionReportContext extended |
| **Report Builder** | `src/reporting/research_report.py` | IngestionReport + citation rendering |
| **UI Citations** | `src/display/citation_display.py` | Streamlit citation helpers |
| **Interactive UI** | `src/display/interactive_claims.py` | Evidence tab with provenance |
| **Tests** | `tests/test_url_provenance_and_stats.py` | 19 comprehensive tests |

## 🔧 Usage Examples

### Tracking Sources in Pipeline

```python
from src.reporting.ingestion_stats import IngestionStatsAggregator

aggregator = IngestionStatsAggregator()

# PDF
aggregator.add_pdf_source("doc.pdf", pages=10, chunks=30, chars=15000)

# URL
aggregator.add_url_source("https://example.com", fetch_success=True, chunks=20, chars=8000)

# Text
aggregator.add_text_source("notes", chunks=10, chars=4000)

# Audio
aggregator.add_audio_source("lecture.mp3", duration_seconds=1800, transcript_chars=12000, chunks=40)

# Get stats
print(f"Total chunks: {aggregator.get_total_chunks()}")  # 100
print(f"Avg chunk size: {aggregator.get_avg_chunk_size()}")  # 400.0

# Convert to report format
report = aggregator.to_ingestion_report()
```

### Creating Citations

```python
from src.schema.verifiable_schema import EvidenceCitation

# URL Citation
url_citation = EvidenceCitation(
    source_type="url_article",
    quote="Python is a high-level language",
    origin="https://python.org/docs",
    confidence=0.92
)

# PDF Citation
pdf_citation = EvidenceCitation(
    source_type="pdf_page",
    quote="Machine learning is a subset of AI",
    origin="textbook.pdf",
    page_num=42,
    confidence=0.98
)

# Video Citation
video_citation = EvidenceCitation(
    source_type="youtube_transcript",
    quote="Bubble sort has O(n²) complexity",
    origin="https://youtube.com/watch?v=abc",
    timestamp_range=(65.0, 95.0),
    confidence=0.88
)
```

### Rendering Citations in Streamlit

```python
from src.display.citation_display import CitationInfo, render_citation_inline

citation = CitationInfo(
    source_type="youtube_transcript",
    origin="https://youtube.com/watch?v=abc",
    timestamp_range=(65.0, 95.0),
    snippet="The algorithm runs in linear time..."
)

# Inline display
render_citation_inline(citation)
# Output: ▶️ [youtube.com/...](url) ⏱️ 01:05-01:35

# List with expandable details
from src.display.citation_display import render_citation_list

render_citation_list([citation1, citation2, citation3], max_visible=2)
```

### Building Reports with Provenance

```python
from src.reporting.research_report import ResearchReportBuilder, ClaimEntry

claim = ClaimEntry(
    claim_text="Python supports duck typing",
    status="VERIFIED",
    confidence=0.92,
    evidence_count=2,
    top_evidence="Python uses dynamic typing...",
    citation_origin="https://docs.python.org/3/reference",
    citation_source_type="url_article"
)

report_builder = ResearchReportBuilder()
report_builder.add_claims([claim])
markdown = report_builder.build_markdown()
# Citations will show: 🔗 [docs.python.org/...](url)
```

## 📊 Report Output Examples

### Before:
```markdown
## Ingestion Statistics
- **Total Pages**: 0
- **Total Chunks**: 0
- **Avg Chunk Size**: N/A

| Claim | Citation |
|-------|----------|
| Python is dynamic | Yes |
```

### After:
```markdown
## Ingestion Statistics

### URL Sources
- **URLs Provided**: 5
- **Successfully Fetched**: 4 (80%)
- **Chunks Extracted**: 120

### Text Input
- **Characters**: 5,000
- **Chunks**: 15

### Overall Extraction
- **Total Chunks (All Sources)**: 135
- **Avg Chunk Size**: 385 chars

| Claim | Citation |
|-------|----------|
| Python is dynamic | 🔗 [docs.python.org](https://docs.python.org/3) |
| Sort has O(n²) cost | ▶️ [youtube.com/...](url) ⏱️ 02:15-02:45 |
| ML is AI subset | 📄 textbook.pdf p.42 |
```

## 🧪 Testing

```bash
# Run all URL provenance tests
python -m pytest tests/test_url_provenance_and_stats.py -v

# Run specific test class
python -m pytest tests/test_url_provenance_and_stats.py::TestCitationURLProvenance -v

# Check coverage
pytest tests/test_url_provenance_and_stats.py --cov=src/reporting --cov=src/schema
```

## 🔍 Validation

### Check Invariants
```python
from src.reporting.ingestion_stats import IngestionStatsAggregator

aggregator = IngestionStatsAggregator()
# ... add sources ...

errors = aggregator.validate()
if errors:
    print("Invariant violations:", errors)
else:
    print("All invariants satisfied ✓")
```

### Verify Stats
```python
aggregator.log_summary()
# Output:
# ============================================================
# INGESTION STATISTICS SUMMARY
# ============================================================
# PDF: 10 pages (2 OCR'd)
# URLs: 5 provided, 4 fetched, 120 chunks
# Text: 5,000 chars, 15 chunks
# Audio: 30m 0s, 12,000 chars, 40 chunks
# TOTAL: 175 chunks, 42,000 chars
# Avg chunk size: 240 chars
# ============================================================
```

## 🎨 Source Type Icons

| Source Type | Icon | Display Example |
|-------------|------|-----------------|
| `pdf_page` | 📄 | `📄 textbook.pdf p.42` |
| `url_article` | 🔗 | `🔗 [example.com](url)` |
| `youtube_transcript` | ▶️ | `▶️ [youtube](url) ⏱️ 01:23` |
| `audio_transcript` | 🎤 | `🎤 lecture.mp3 ⏱️ 12:30` |
| `notes_text` | 📝 | `📝 user_notes` |
| `external_context` | 🌐 | `🌐 external_source` |

## 📐 Supported Source Types

### EvidenceSpan / Evidence
- `pdf_page`
- `notes_text`
- `external_context`
- `url_article`
- `youtube_transcript`
- `audio_transcript`

### EvidenceCitation (extends above)
- All above types +
- `equation`
- `inferred`

## ⚠️ Important Notes

1. **Timestamps are in seconds**: `timestamp_range=(65.0, 95.0)` means 1:05-1:35
2. **YouTube links auto-timestamped**: `format_timestamp_link(url, seconds)` adds `&t=` parameter
3. **Avg chunk size is None when no chunks**: Maintained by `__post_init__` validation
4. **URL fetch failures tracked**: `url_count` vs `url_fetch_success_count`
5. **All stats are cumulative**: `add_*_source()` methods accumulate totals

## 🚀 Deployment Checklist

- [ ] Import `IngestionStatsAggregator` in pipeline
- [ ] Call `add_*_source()` for each ingested source
- [ ] Convert to report: `aggregator.to_ingestion_report()`
- [ ] Pass report to `ResearchReportBuilder`
- [ ] Populate `ClaimEntry` with `citation_origin`, `citation_source_type`, `citation_timestamp`
- [ ] Run tests: `pytest tests/test_url_provenance_and_stats.py`
- [ ] Verify reports show multi-source stats
- [ ] Verify citations are clickable in UI

## 📚 Related Documentation

- [Implementation Summary](URL_PROVENANCE_IMPLEMENTATION_SUMMARY.md)
- [Schema Documentation](src/schema/verifiable_schema.py)
- [Stats Aggregator](src/reporting/ingestion_stats.py)
- [Citation Display](src/display/citation_display.py)
- [Tests](tests/test_url_provenance_and_stats.py)

---

**Status**: ✅ All 19 tests passing  
**Coverage**: Citations + Stats fully implemented  
**Next Steps**: Integrate `IngestionStatsAggregator` in main pipeline
