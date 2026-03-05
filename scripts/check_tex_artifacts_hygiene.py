#!/usr/bin/env python3
"""
Fail-fast hygiene checker for generated LaTeX artifacts.

Validates that generated .tex (and related) files contain no hidden Unicode
artifacts and that inserted content is LaTeX-safe (percent/underscore/etc.
properly escaped). Designed for deterministic paper pipelines.

Exit codes:
  0 = OK
  1 = Hygiene violation
  2 = Execution error (I/O, missing files, etc.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

BANNED_CODEPOINTS = {
    0x00AD: "soft hyphen U+00AD",
    0x200B: "zero-width space U+200B",
    0x200C: "zero-width non-joiner U+200C",
    0x200D: "zero-width joiner U+200D",
    0xFEFF: "byte order mark U+FEFF",
}

BANNED_RANGES = [
    (0x2060, 0x2064),  # word joiner / invisible separators
]

ALLOWED_C0 = {9, 10, 13}  # tab, newline, carriage return

LATEX_SPECIALS = {
    "%": "unescaped %",
    "_": "unescaped _",
    "#": "unescaped #",
    "$": "unescaped $",
    "&": "unescaped &",
    "~": "unescaped ~",
    "^": "unescaped ^",
}

DEFAULT_REL_PATHS = [
    Path("paper") / "metrics_values.tex",
    Path("paper") / "significance_values.tex",
    Path("paper") / "references.bib",
]


def _safe(msg: str) -> str:
    """Return ASCII-safe representation for stdout."""
    return msg.encode("ascii", "backslashreplace").decode("ascii")


def _is_escaped(line: str, idx: int) -> bool:
    """Check if the character at idx is escaped by an odd number of backslashes."""
    backslashes = 0
    j = idx - 1
    while j >= 0 and line[j] == "\\":
        backslashes += 1
        j -= 1
    return backslashes % 2 == 1


def _is_comment(line: str) -> bool:
    return line.lstrip().startswith("%")


def _iter_banned_codepoints(line: str, lineno: int) -> Iterable[Tuple[int, str, str]]:
    for idx, ch in enumerate(line):
        codepoint = ord(ch)
        if codepoint in BANNED_CODEPOINTS:
            yield lineno, f"U+{codepoint:04X}", BANNED_CODEPOINTS[codepoint]
        elif any(lo <= codepoint <= hi for lo, hi in BANNED_RANGES):
            yield lineno, f"U+{codepoint:04X}", "invisible separator"
        elif codepoint < 0x20 and codepoint not in ALLOWED_C0:
            yield lineno, f"U+{codepoint:04X}", "disallowed control char"


def _iter_latex_specials(line: str, lineno: int) -> Iterable[Tuple[int, str, str]]:
    if _is_comment(line):
        return []

    violations: List[Tuple[int, str, str]] = []
    for idx, ch in enumerate(line):
        if ch in LATEX_SPECIALS and not _is_escaped(line, idx):
            violations.append((lineno, LATEX_SPECIALS[ch], line))
    return violations


def _snip(line: str, marker: int | None = None) -> str:
    if marker is None:
        return _safe(line.strip())
    start = max(0, marker - 20)
    end = min(len(line), marker + 20)
    snippet = line[start:end].strip()
    return _safe(snippet)


def scan_file(path: Path) -> list[dict]:
    """Scan a file for Unicode/LaTeX hygiene issues."""
    issues: list[dict] = []
    check_specials = path.suffix.lower() == ".tex"

    try:
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:  # pragma: no cover - reported via caller
        issues.append({
            "type": "read_error",
            "line": 0,
            "codepoint": "",
            "detail": _safe(str(exc)),
            "snippet": "",
        })
        return issues

    for lineno, line in enumerate(text, start=1):
        for ln, codepoint, detail in _iter_banned_codepoints(line, lineno):
            issues.append({
                "type": "banned_codepoint",
                "line": ln,
                "codepoint": codepoint,
                "detail": detail,
                "snippet": _snip(line, None),
            })

        if check_specials:
            for ln, detail, snippet in _iter_latex_specials(line, lineno):
                issues.append({
                    "type": "latex_special",
                    "line": ln,
                    "codepoint": detail,
                    "detail": detail,
                    "snippet": _snip(snippet, None),
                })

    return issues


def resolve_default_paths(repo_root: Path) -> list[Path]:
    resolved = []
    for rel in DEFAULT_REL_PATHS:
        candidate = repo_root / rel
        if candidate.exists():
            resolved.append(candidate)
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-fast Unicode/LaTeX hygiene checker for generated .tex artifacts",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Files to check (default: paper/metrics_values.tex, paper/significance_values.tex, paper/references.bib)",
    )

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    targets = [Path(p) for p in args.paths] if args.paths else resolve_default_paths(repo_root)
    if not targets:
        print(_safe("[ERROR] No target files provided or found; specify at least one .tex/.bib file"))
        return 2

    missing = [p for p in targets if not p.exists()]
    if missing:
        for path in missing:
            print(_safe(f"[ERROR] Missing file: {path}"))
        return 2

    any_fail = False
    for path in targets:
        issues = scan_file(path)
        if issues:
            any_fail = True
            for issue in issues:
                prefix = _safe(f"[ERROR] {path}:{issue['line']}")
                detail = _safe(issue["detail"])
                snippet = issue["snippet"]
                codepoint = issue.get("codepoint", "")
                cp_part = f" ({codepoint})" if codepoint else ""
                print(f"{prefix}: {detail}{cp_part} | snippet: {snippet}")
        else:
            print(_safe(f"[OK] {path}"))

    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
