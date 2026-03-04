#!/usr/bin/env python3
"""Final camera-ready manuscript consistency checks."""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
TARGET_TEX = sorted((ROOT / "submission_bundle").glob("**/*.tex"))

FORBIDDEN_REF_PATTERNS = [
    r"\bV-F\b",
    r"\bV-G\b",
    r"Section~IV-E",
    r"Sec\.\s*IV-E",
]

FORBIDDEN_STALE_VALUES = [
    "0.1247",
    "0.1304",
    "0.8803",
    "0.9364",
    "0.6962",
    "0.0092",
]


def collect_matches(path: Path, pattern: str, flags=0):
    regex = re.compile(pattern, flags)
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if regex.search(line):
            out.append((i, line.rstrip()))
    return out


def check_seed_ambiguity(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    issues = []

    for i, line in enumerate(lines, start=1):
        if re.search(r"\b5\s+seeds\b|\b5\s+deterministic\s+seeds\b|\b5\s+evaluation\s+seeds\b", line, re.IGNORECASE):
            window = " ".join(lines[max(0, i - 2): min(len(lines), i + 1)])
            if not re.search(r"evaluation|no retraining", window, re.IGNORECASE):
                issues.append((i, line.rstrip()))
    return issues


def main() -> int:
    errors = []

    print("Running final camera-ready checks")
    print("=" * 72)

    for tex in TARGET_TEX:
        rel = tex.relative_to(ROOT)

        for pat in FORBIDDEN_REF_PATTERNS:
            for line_no, line in collect_matches(tex, pat):
                errors.append(f"{rel}:{line_no} forbidden reference '{pat}' -> {line}")

        for stale in FORBIDDEN_STALE_VALUES:
            for line_no, line in collect_matches(tex, re.escape(stale)):
                errors.append(f"{rel}:{line_no} stale value '{stale}' -> {line}")

    # Ambiguous seed wording only for main manuscript sources
    manuscript_files = [
        ROOT / "submission_bundle" / "OVERLEAF_TEMPLATE.tex",
        ROOT / "submission_bundle" / "overleaf_upload_pack" / "OVERLEAF_TEMPLATE.tex",
    ]
    for tex in manuscript_files:
        rel = tex.relative_to(ROOT)
        for line_no, line in check_seed_ambiguity(tex):
            errors.append(f"{rel}:{line_no} ambiguous seed wording -> {line}")

    # Confirm key metric values appear in camera-ready source
    overleaf = ROOT / "submission_bundle" / "overleaf_upload_pack" / "OVERLEAF_TEMPLATE.tex"
    overleaf_content = overleaf.read_text(encoding="utf-8")
    if "0.1076" not in overleaf_content and "\\ECEValue" not in overleaf_content:
        errors.append("overleaf_upload_pack/OVERLEAF_TEMPLATE.tex missing ECE 0.1076/\\ECEValue")
    if "0.8711" not in overleaf_content and "\\AUCACValue" not in overleaf_content:
        errors.append("overleaf_upload_pack/OVERLEAF_TEMPLATE.tex missing AUC-AC 0.8711/\\AUCACValue")

    print("-" * 72)
    if errors:
        print("FAIL")
        for err in errors:
            print(f"ERROR: {err}")
        return 1

    print("PASS: final camera-ready checks succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
