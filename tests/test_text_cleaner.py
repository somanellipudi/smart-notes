import config
from src.preprocessing.text_cleaner import clean_extracted_text


def _build_paged_text(pages: list[list[str]]) -> str:
    parts = []
    for idx, lines in enumerate(pages, start=1):
        parts.append(f"--- Page {idx} ---")
        parts.extend(lines)
    return "\n".join(parts)


def test_repeated_header_removed_across_pages():
    pages = []
    for i in range(5):
        pages.append([
            "DATA STRUCTURE NOTES",
            f"Page {i + 1}",
            "Stacks follow LIFO ordering.",
            "Push adds to top and pop removes from top."
        ])

    raw_text = _build_paged_text(pages)
    cleaned, diag = clean_extracted_text(raw_text)

    assert "DATA STRUCTURE NOTES" not in cleaned
    assert diag.removed_repeated_lines_count > 0
    assert diag.repeat_threshold_used >= 2


def test_watermark_removed_by_regex():
    raw_text = """
--- Page 1 ---
Scanned with Adobe Scan
Queues follow FIFO ordering.
""".strip()

    cleaned, diag = clean_extracted_text(raw_text)

    assert "Adobe Scan" not in cleaned
    assert diag.removed_by_regex.get("scan_watermark", 0) >= 1


def test_code_line_protection_keeps_cs_tokens():
    raw_text = """
--- Page 1 ---
O(n log n) time complexity for merge sort.
stack.push(x)
stack.pop()
""".strip()

    cleaned, diag = clean_extracted_text(raw_text)

    assert "O(n log n)" in cleaned
    assert "push" in cleaned
    assert "pop" in cleaned
    assert diag.removed_lines_count == 0 or diag.removed_by_regex.get("low_info", 0) == 0
