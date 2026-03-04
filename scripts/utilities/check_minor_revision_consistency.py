#!/usr/bin/env python3
"""Minor-revision consistency checks for camera-ready wording/cross-references."""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
MANUSCRIPTS = [
    ROOT / "submission_bundle" / "OVERLEAF_TEMPLATE.tex",
    ROOT / "submission_bundle" / "overleaf_upload_pack" / "OVERLEAF_TEMPLATE.tex",
]


def find_lines(path: Path, pattern: str, flags=0):
    regex = re.compile(pattern, flags)
    matches = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if regex.search(line):
            matches.append((line_no, line.rstrip()))
    return matches


def section_letter_for_metric_impl(content: str):
    section_match = re.search(r"\\section\{Experimental Setup\}(.*?)\\section\{", content, re.DOTALL)
    if not section_match:
        return None
    exp_block = section_match.group(1)
    subs = re.findall(r"\\subsection\{([^}]+)\}", exp_block)
    for idx, title in enumerate(subs, start=1):
        if title.strip() == "Metric Implementation Details":
            return chr(ord("A") + idx - 1)
    return None


def main() -> int:
    errors = []
    print("Running minor-revision consistency checks")
    print("=" * 72)

    for manuscript in MANUSCRIPTS:
        if not manuscript.exists():
            errors.append(f"Missing manuscript: {manuscript}")
            continue

        content = manuscript.read_text(encoding="utf-8")
        rel = manuscript.relative_to(ROOT)

        # 1) "5 seeds" wording must include evaluation/no retraining context
        for line_no, line in find_lines(manuscript, r"5\s+seeds", flags=re.IGNORECASE):
            if not re.search(r"evaluation|no retraining", line, flags=re.IGNORECASE):
                errors.append(
                    f"{rel}:{line_no} has '5 seeds' without evaluation-only context: {line}"
                )

        # 2) Section reference correctness for seed policy location
        letter = section_letter_for_metric_impl(content)
        if letter is None:
            errors.append(f"{rel}: could not locate 'Metric Implementation Details' subsection letter")
        else:
            stale_ref = f"Section~IV-E"
            if letter != "E":
                for line_no, line in find_lines(manuscript, re.escape(stale_ref)):
                    errors.append(f"{rel}:{line_no} stale reference '{stale_ref}': {line}")

                expected = f"Section~IV-{letter}"
                if expected not in content:
                    errors.append(f"{rel}: expected reference '{expected}' not found")

        # 3) Optional stale V-F/V-G checks
        for stale in [r"\bV-F\b", r"\bV-G\b"]:
            for line_no, line in find_lines(manuscript, stale):
                errors.append(f"{rel}:{line_no} stale section reference '{stale}': {line}")

    print("-" * 72)
    if errors:
        print("FAIL")
        for err in errors:
            print(f"ERROR: {err}")
        return 1

    print("PASS: minor-revision consistency checks succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
