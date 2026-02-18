"""
Generalized line-based text cleaning for boilerplate and repeated headers.
"""

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import config


@dataclass
class CleanDiagnostics:
    removed_lines_count: int
    removed_by_regex: Dict[str, int]
    removed_repeated_lines_count: int
    top_removed_lines: List[str]
    kept_lines_count: int
    repeat_threshold_used: int


def normalize_line(line: str) -> str:
    if not line:
        return ""

    normalized = line.strip()
    for src, dst in [
        ("\u2018", "'"),
        ("\u2019", "'"),
        ("\u201c", '"'),
        ("\u201d", '"'),
        ("\u2014", "-"),
        ("\u2013", "-")
    ]:
        normalized = normalized.replace(src, dst)

    normalized = re.sub(r"^[\s\-*\u2022]+", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip().lower()


def _is_mostly_punct(line: str) -> bool:
    if not line:
        return True
    total = len(line)
    if total == 0:
        return True
    punct = sum(1 for c in line if not c.isalnum() and not c.isspace())
    return punct / total > 0.6


def is_low_info(line: str) -> bool:
    if not line:
        return True
    stripped = line.strip()
    if len(stripped) < config.MIN_LINE_LEN:
        return True

    if _is_mostly_punct(stripped):
        return True

    page_patterns = [
        r"^\d+$",
        r"^page\s*\d+$",
        r"^\d+\s*/\s*\d+$",
        r"^[-–—]*\s*\d+\s*[-–—]*$"
    ]
    for pattern in page_patterns:
        if re.match(pattern, stripped, re.IGNORECASE):
            return True

    return False


def _compile_boilerplate_rules() -> List[Tuple[str, re.Pattern]]:
    rules = []
    for rule in config.BOILERPLATE_REGEX_RULES:
        try:
            rules.append((rule["name"], re.compile(rule["pattern"], re.IGNORECASE)))
        except Exception:
            continue
    return rules


def matches_boilerplate_regex(line: str, rules: List[Tuple[str, re.Pattern]]) -> Optional[str]:
    for name, pattern in rules:
        if pattern.search(line):
            return name
    return None


def _compile_code_protect_regexes() -> List[re.Pattern]:
    compiled = []
    for pattern in config.CODE_LINE_PROTECT_REGEXES:
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except Exception:
            continue
    return compiled


def _is_code_like(line: str, protect_regexes: List[re.Pattern]) -> bool:
    lowered = line.lower()
    for token in config.CODE_LINE_PROTECT_TOKENS:
        if token in lowered:
            return True
    for pattern in protect_regexes:
        if pattern.search(line):
            return True
    return False


def _split_pages(raw_text: str, page_separator_regex: str) -> List[List[str]]:
    lines = raw_text.splitlines()
    pages: List[List[str]] = [[]]
    sep = re.compile(page_separator_regex, re.IGNORECASE)
    alt_sep = re.compile(r"^\[Page\s+\d+\]\s*$", re.IGNORECASE)

    for line in lines:
        if (sep.match(line.strip()) or alt_sep.match(line.strip())) and pages[-1]:
            pages.append([])
            continue
        pages[-1].append(line)

    return pages


def repeated_line_detector(pages: List[List[str]], repeat_frac: float) -> Tuple[set[str], Dict[str, int], int]:
    page_count = len(pages)
    if page_count == 0:
        return set(), {}, 0

    threshold = int(math.ceil(repeat_frac * page_count))
    line_page_counts: Dict[str, int] = {}
    protect_regexes = _compile_code_protect_regexes()

    for page in pages:
        seen = set()
        header_footer = page[:5] + page[-5:]
        for line in header_footer:
            normalized = normalize_line(line)
            if not normalized:
                continue
            if _is_code_like(normalized, protect_regexes):
                continue
            seen.add(normalized)
        for normalized in seen:
            line_page_counts[normalized] = line_page_counts.get(normalized, 0) + 1

    repeated = {
        line
        for line, count in line_page_counts.items()
        if count >= threshold
    }
    return repeated, line_page_counts, threshold


def _uppercase_ratio(line: str) -> float:
    letters = [c for c in line if c.isalpha()]
    if not letters:
        return 0.0
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters)


def clean_extracted_text(
    raw_text: str,
    page_separator_regex: str = r"--- Page \d+ ---"
) -> Tuple[str, CleanDiagnostics]:
    if not raw_text:
        diagnostics = CleanDiagnostics(0, {}, 0, [], 0, 0)
        return "", diagnostics

    if not config.CLEANING_ENABLED:
        kept_lines = [line for line in raw_text.splitlines() if line.strip()]
        diagnostics = CleanDiagnostics(0, {}, 0, [], len(kept_lines), 0)
        return raw_text, diagnostics

    pages = _split_pages(raw_text, page_separator_regex)
    repeated_lines, line_page_counts, repeat_threshold = repeated_line_detector(
        pages,
        config.REPEAT_FRAC
    )
    boilerplate_rules = _compile_boilerplate_rules()
    protect_regexes = _compile_code_protect_regexes()

    removed_by_regex: Dict[str, int] = {}
    removed_repeated_lines = 0
    removed_lines = 0
    removed_samples: Dict[str, int] = {}
    kept_lines = 0

    cleaned_lines: List[str] = []

    for page in pages:
        for idx, line in enumerate(page):
            stripped = line.strip()
            normalized = normalize_line(line)

            if not stripped:
                cleaned_lines.append("")
                continue

            if _is_code_like(line, protect_regexes):
                cleaned_lines.append(line)
                kept_lines += 1
                continue

            rule_name = matches_boilerplate_regex(stripped, boilerplate_rules)
            if rule_name:
                removed_by_regex[rule_name] = removed_by_regex.get(rule_name, 0) + 1
                removed_lines += 1
                removed_samples[stripped] = removed_samples.get(stripped, 0) + 1
                continue

            if is_low_info(stripped):
                removed_by_regex["low_info"] = removed_by_regex.get("low_info", 0) + 1
                removed_lines += 1
                removed_samples[stripped] = removed_samples.get(stripped, 0) + 1
                continue

            if normalized in repeated_lines:
                removed_repeated_lines += 1
                removed_lines += 1
                removed_samples[stripped] = removed_samples.get(stripped, 0) + 1
                continue

            next_line = ""
            if idx + 1 < len(page):
                next_line = page[idx + 1].strip()

            if (
                len(stripped) <= config.MAX_TITLE_LEN
                and _uppercase_ratio(stripped) >= 0.7
                and line_page_counts.get(normalized, 0) >= repeat_threshold
                and next_line
                and len(next_line) > config.MAX_TITLE_LEN
            ):
                removed_by_regex["title_block"] = removed_by_regex.get("title_block", 0) + 1
                removed_lines += 1
                removed_samples[stripped] = removed_samples.get(stripped, 0) + 1
                continue

            cleaned_lines.append(line)
            kept_lines += 1

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()

    top_removed_lines = [
        line for line, _ in sorted(removed_samples.items(), key=lambda item: item[1], reverse=True)[:5]
    ]

    diagnostics = CleanDiagnostics(
        removed_lines_count=removed_lines,
        removed_by_regex=removed_by_regex,
        removed_repeated_lines_count=removed_repeated_lines,
        top_removed_lines=top_removed_lines,
        kept_lines_count=kept_lines,
        repeat_threshold_used=repeat_threshold
    )

    return cleaned_text, diagnostics
