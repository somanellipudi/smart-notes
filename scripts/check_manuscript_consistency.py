#!/usr/bin/env python3
"""Fail-fast manuscript consistency checks for final IEEE cleanup."""

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TARGET_FILES = [
    ROOT / "submission_bundle" / "OVERLEAF_TEMPLATE.tex",
    ROOT / "submission_bundle" / "overleaf_upload_pack" / "OVERLEAF_TEMPLATE.tex",
    ROOT / "submission_bundle" / "tables" / "seed_policy.tex",
]

FORBIDDEN_PATTERNS = [
    r"seed\s*=\s*42",
    r"strongest possible evidence against cherry-picking",
    r"\bV-F\b",
    r"\bV-G\b",
]


def find_matches(path: Path, pattern: str):
    content = path.read_text(encoding="utf-8")
    regex = re.compile(pattern, re.IGNORECASE)
    matches = []
    for line_num, line in enumerate(content.splitlines(), start=1):
        if regex.search(line):
            matches.append((line_num, line.strip()))
    return matches


def extract_seed_policy_block(content: str):
    marker = r"\textbf{Seed and determinism policy}:"
    idx = content.find(marker)
    if idx == -1:
        return None
    tail = content[idx:]
    end = tail.find("\n\n")
    if end == -1:
        return tail
    return tail[:end]


def main() -> int:
    errors = []

    print("Running manuscript consistency checks")
    print("=" * 72)

    for target in TARGET_FILES:
        if not target.exists():
            errors.append(f"Missing file: {target}")
            continue
        print(f"Checked file exists: {target.relative_to(ROOT)}")

    # 1) Forbidden patterns in all manuscript source files
    for target in TARGET_FILES:
        if not target.exists():
            continue
        for pattern in FORBIDDEN_PATTERNS:
            bad = find_matches(target, pattern)
            for line_num, line in bad:
                errors.append(
                    f"Forbidden pattern /{pattern}/ in {target.relative_to(ROOT)}:{line_num}: {line}"
                )

    # 2) Seed statement appears exactly once and is evaluation-only in each main manuscript
    manuscript_files = TARGET_FILES[:2]
    seed_set_pattern = re.compile(r"\\?\{\s*0\s*,\s*1\s*,\s*2\s*,\s*3\s*,\s*4\s*\\?\}")
    for target in manuscript_files:
        if not target.exists():
            continue
        content = target.read_text(encoding="utf-8")
        occurrences = len(seed_set_pattern.findall(content))
        if occurrences != 1:
            errors.append(
                f"Seed-set statement must appear exactly once in {target.relative_to(ROOT)} (found {occurrences})"
            )

        policy_block = extract_seed_policy_block(content)
        if policy_block is None:
            errors.append(
                f"Missing 'Seed and determinism policy' paragraph in {target.relative_to(ROOT)}"
            )
        else:
            for required in ["evaluation only", "not retraining", "fixed seed $0$", "std $=0.0000$"]:
                if required not in policy_block:
                    errors.append(
                        f"Seed policy missing '{required}' in {target.relative_to(ROOT)}"
                    )

    # 3) FEVER transfer should reference Section V-I
    for target in manuscript_files:
        if not target.exists():
            continue
        content = target.read_text(encoding="utf-8")
        if "Transfer Learning: FEVER Evaluation" not in content:
            errors.append(f"Missing FEVER transfer subsection in {target.relative_to(ROOT)}")
        if "Section~V-I" not in content:
            errors.append(f"Missing Section~V-I reference in {target.relative_to(ROOT)}")

    print("-" * 72)
    if errors:
        print("FAIL")
        for err in errors:
            print(f"ERROR: {err}")
        return 1

    print("PASS: manuscript consistency checks succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
