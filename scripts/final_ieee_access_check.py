from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPTS = [
    ROOT / "submission_bundle" / "OVERLEAF_TEMPLATE.tex",
    ROOT / "submission_bundle" / "overleaf_upload_pack" / "OVERLEAF_TEMPLATE.tex",
]

STALE_VALUES = ["0.1247", "0.1304", "0.8803", "0.9364", "0.6962", "0.0092"]


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def check_stale_values(text: str, file_name: str, errors: list[str]) -> None:
    for value in STALE_VALUES:
        if value in text:
            fail(errors, f"[{file_name}] stale value found: {value}")


def check_abstract_seed_statement(text: str, file_name: str, errors: list[str]) -> None:
    if "5 evaluation seeds (no retraining)" not in text:
        fail(errors, f"[{file_name}] missing exact abstract seed phrase: '5 evaluation seeds (no retraining)'")


def check_tau_protocol(text: str, file_name: str, errors: list[str]) -> None:
    validation_only_pattern = re.compile(r"\\tau\$\s+is\s+optimized\s+\\textit\{only\}\s+on\s+the\s+validation\s+set", re.IGNORECASE)
    test_unchanged_pattern = re.compile(r"applied\s+to\s+the\s+test\s+set\s+.*unchanged", re.IGNORECASE)

    if not validation_only_pattern.search(text):
        fail(errors, f"[{file_name}] missing tau protocol clause: optimized only on validation")
    if not test_unchanged_pattern.search(text):
        fail(errors, f"[{file_name}] missing tau protocol clause: applied to test unchanged")
    if "sensitivity characterization only" not in text:
        fail(errors, f"[{file_name}] missing tau protocol clause: bootstrap re-selection is sensitivity characterization only")


def check_equation_references(text: str, file_name: str, errors: list[str]) -> None:
    if "Equation 6" in text:
        fail(errors, f"[{file_name}] manual 'Equation 6' reference still present")
    if "Eq.~\\eqref{" not in text:
        fail(errors, f"[{file_name}] no stable equation reference found (expected Eq.~\\eqref{{...}})")


def check_argmin_definition(text: str, file_name: str, errors: list[str]) -> None:
    uses_argmin = "\\argmin" in text
    has_definition = "\\DeclareMathOperator*{\\argmin}{arg\\,min}" in text
    uses_arg_min_fallback = "\\arg\\min" in text

    if uses_argmin and not has_definition and not uses_arg_min_fallback:
        fail(errors, f"[{file_name}] possible undefined macro: \\argmin used without declaration or fallback")


def check_labels(text: str, file_name: str, errors: list[str]) -> None:
    required = [
        "\\label{eq:auth_score}",
        "\\label{sec:selective_prediction}",
        "\\ref{sec:selective_prediction}",
        "\\eqref{eq:auth_score}",
    ]
    for token in required:
        if token not in text:
            fail(errors, f"[{file_name}] missing label/reference token: {token}")


def main() -> int:
    errors: list[str] = []

    for path in MANUSCRIPTS:
        if not path.exists():
            fail(errors, f"Missing manuscript file: {path}")
            continue

        text = path.read_text(encoding="utf-8")
        name = path.relative_to(ROOT).as_posix()

        check_stale_values(text, name, errors)
        check_abstract_seed_statement(text, name, errors)
        check_tau_protocol(text, name, errors)
        check_equation_references(text, name, errors)
        check_argmin_definition(text, name, errors)
        check_labels(text, name, errors)

    if errors:
        print("FINAL IEEE ACCESS CHECK: FAIL")
        for err in errors:
            print(f" - {err}")
        return 1

    print("FINAL IEEE ACCESS CHECK: PASS")
    print(" - No stale values found")
    print(" - Abstract seed statement includes '5 evaluation seeds (no retraining)'")
    print(" - Tau protocol language is unambiguous (validation-only, test unchanged, analysis-only bootstrap)")
    print(" - Manual 'Equation 6' references removed in favor of Eq.~\\eqref{...}")
    print(" - argmin usage is defined safely")
    print(" - Required labels/references are present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
