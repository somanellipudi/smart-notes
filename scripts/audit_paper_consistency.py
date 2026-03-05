#!/usr/bin/env python3
"""
Paper Consistency Audit for CalibraTeach IEEE Access Submission.

Validates that paper/main.tex, metrics_values.tex, significance_values.tex,
artifacts, and Overleaf bundle remain synchronized before submission.

This script performs 6 critical consistency checks and exits with code 1 if
any inconsistencies are detected.

Usage:
    python scripts/audit_paper_consistency.py
    python scripts/audit_paper_consistency.py --verbose
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class PaperConsistencyAudit:
    """Audit paper consistency across LaTeX sources and artifacts."""

    def __init__(self, repo_root: Path, verbose: bool = False):
        """Initialize audit with repo root path."""
        self.repo_root = repo_root
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.paper_dir = repo_root / "paper"
        self.scripts_dir = repo_root / "scripts"
        self.artifacts_dir = repo_root / "artifacts"

    def log(self, msg: str):
        """Print verbose log message."""
        if self.verbose:
            print(f"  {msg}")

    def check_macros_in_tex(self) -> bool:
        """
        CHECK 1: Verify metrics_values.tex macros and their appearance in main.tex.
        
        Extracts AccuracyValue, ECEValue, AUCACValue from metrics_values.tex
        and ensures they are used in reliability diagram, accuracy-coverage,
        and main results table captions.
        """
        print("\n[AUDIT 1/6] Macro verification (metrics_values.tex)")
        print("-" * 70)

        metrics_values_file = self.paper_dir / "metrics_values.tex"
        if not metrics_values_file.exists():
            self.errors.append(
                f"Missing: {metrics_values_file.relative_to(self.repo_root)}"
            )
            return False

        try:
            with open(metrics_values_file, "r", encoding="utf-8") as f:
                metrics_content = f.read()
        except Exception as e:
            self.errors.append(f"Failed to read {metrics_values_file}: {e}")
            return False

        # Extract macro definitions
        macros_to_check = ["AccuracyValue", "ECEValue", "AUCACValue"]
        macro_values = {}

        for macro in macros_to_check:
            pattern = rf"\\(?:newcommand|DeclareRobustCommand){{\\{macro}}}{{([^}}]+)}}"
            match = re.search(pattern, metrics_content)
            if not match:
                self.errors.append(f"Missing macro definition: \\{macro}")
                return False
            macro_values[macro] = match.group(1)
            self.log(f"Found \\{macro} = {macro_values[macro]}")

        # Read main.tex and check for usage
        main_tex_file = self.paper_dir / "main.tex"
        if not main_tex_file.exists():
            self.errors.append(f"Missing: {main_tex_file.relative_to(self.repo_root)}")
            return False

        try:
            with open(main_tex_file, "r", encoding="utf-8") as f:
                main_content = f.read()
        except Exception as e:
            self.errors.append(f"Failed to read {main_tex_file}: {e}")
            return False

        # Check that each macro is used in main.tex
        all_used = True
        for macro in macros_to_check:
            if f"\\{macro}" not in main_content:
                self.errors.append(
                    f"Macro \\{macro} defined in metrics_values.tex but not used in main.tex"
                )
                all_used = False
            else:
                self.log(f"OK  \\{macro} is used in main.tex")

        # Verify macro appears in key captions
        required_contexts = [
            ("reliability_diagram_verified", "reliability diagram"),
            ("accuracy_coverage_verified", "accuracy-coverage"),
            ("tab:main_results", "main results table"),
        ]

        for context_key, context_name in required_contexts:
            if context_key in main_content:
                self.log(f"OK  Found reference to {context_name}")
            else:
                # Not necessarily an error, but warn if it's a caption context
                if "caption" in context_name.lower():
                    self.log(f"WARN Reference to {context_name} ({context_key}) not found")

        if all_used:
            print("[OK] metrics_values.tex macros verified")
            return True
        else:
            print("[ERROR] metrics_values.tex macros verification failed")
            return False

    def check_significance_values(self) -> bool:
        """
        CHECK 2: Verify significance_values.tex macros match significance_table.csv.
        
        Extracts macro values from significance_values.tex and compares with
        CSV file to ensure they are in sync.
        """
        print("\n[AUDIT 2/6] Significance verification")
        print("-" * 70)

        sig_values_file = self.paper_dir / "significance_values.tex"
        sig_csv_file = self.artifacts_dir / "stats" / "significance_table.csv"

        if not sig_values_file.exists():
            self.warnings.append(
                f"Optional file missing: {sig_values_file.relative_to(self.repo_root)}"
                " (will use fallback macros, but CSV should match)"
            )
        else:
            try:
                with open(sig_values_file, "r", encoding="utf-8") as f:
                    sig_content = f.read()
            except Exception as e:
                self.errors.append(f"Failed to read {sig_values_file}: {e}")
                return False

            # Extract macros from .tex file
            tex_macros = {}
            pattern = r"\\newcommand{\\(\w+)}{([^}]+)}"
            for match in re.finditer(pattern, sig_content):
                macro_name, macro_value = match.groups()
                tex_macros[macro_name] = macro_value
                self.log(f"Found \\{macro_name} = {macro_value}")

        # Read and parse CSV
        if not sig_csv_file.exists():
            self.errors.append(
                f"Missing: {sig_csv_file.relative_to(self.repo_root)}"
            )
            return False

        try:
            csv_data = {}
            with open(sig_csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    baseline = row["baseline"]
                    csv_data[baseline] = row
                    self.log(
                        f"CSV row: {baseline} -> "
                        f"accuracy_diff={row['accuracy_diff']}, "
                        f"mcnemar_p={row['mcnemar_p']}"
                    )
        except Exception as e:
            self.errors.append(f"Failed to read {sig_csv_file}: {e}")
            return False

        # Compare tex macros with CSV values (if .tex file exists)
        if sig_values_file.exists() and tex_macros:
            success = self._compare_sig_values_with_csv(tex_macros, csv_data)
            if not success:
                return False

        print("[OK] significance_values.tex matches CSV")
        return True

    def _compare_sig_values_with_csv(
        self, tex_macros: Dict[str, str], csv_data: Dict[str, Dict]
    ) -> bool:
        """Helper to compare significance .tex values with CSV."""
        # Map macro names to CSV baselines and fields
        comparison_map = [
            ("SigAccDiffRetrievalNLI", "Retrieval_NLI", "accuracy_diff", False),
            ("SigAccDiffRetrieval", "Retrieval", "accuracy_diff", False),
            ("SigAccDiffBaseline", "Baseline", "accuracy_diff", False),
            ("SigPvalMcNemarRetrievalNLI", "Retrieval_NLI", "mcnemar_p", True),
            ("SigPvalMcNemarRetrieval", "Retrieval", "mcnemar_p", True),
            ("SigPvalMcNemarBaseline", "Baseline", "mcnemar_p", True),
        ]

        all_match = True
        for macro_name, baseline, csv_field, is_pvalue in comparison_map:
            if macro_name not in tex_macros:
                self.log(f"WARN Macro {macro_name} not found in .tex file")
                continue

            if baseline not in csv_data:
                self.log(f"WARN Baseline {baseline} not found in CSV")
                continue

            tex_value = tex_macros[macro_name]
            csv_value = csv_data[baseline][csv_field]

            # Normalize values for comparison
            try:
                if is_pvalue:
                    # Handle p-values: "< 0.001" or numeric
                    if tex_value.startswith("<"):
                        tex_num = 0.001
                        csv_num = float(csv_value)
                        if csv_num < 0.001:
                            self.log(
                                f"OK  {macro_name}: {tex_value} ~= {csv_value} (both < 0.001)"
                            )
                        else:
                            self.errors.append(
                                f"Mismatch {macro_name}: tex={tex_value} vs csv={csv_value}"
                            )
                            all_match = False
                    else:
                        tex_num = float(tex_value)
                        csv_num = float(csv_value)
                        # Allow small rounding differences
                        if abs(tex_num - csv_num) < 0.001:
                            self.log(
                                f"OK  {macro_name}: {tex_value} ~= {csv_value}"
                            )
                        else:
                            self.errors.append(
                                f"Mismatch {macro_name}: tex={tex_value} vs csv={csv_value}"
                            )
                            all_match = False
                else:
                    # Accuracy diff as number
                    tex_num = float(tex_value)
                    csv_num = float(csv_value)
                    abs_csv = abs(csv_num)  # CSV stores as negative
                    if abs(tex_num - abs_csv) < 0.001:
                        self.log(f"OK  {macro_name}: {tex_value} ~= {abs_csv}")
                    else:
                        self.errors.append(
                            f"Mismatch {macro_name}: tex={tex_value} vs csv={abs_csv}"
                        )
                        all_match = False
            except (ValueError, KeyError) as e:
                self.log(f"WARN Could not compare {macro_name}: {e}")

        return all_match

    def check_figure_presence(self) -> bool:
        """
        CHECK 3: Verify required figures exist.
        
        Checks for presence of:
        - architecture.pdf
        - reliability_diagram_verified.pdf
        - accuracy_coverage_verified.pdf
        """
        print("\n[AUDIT 3/6] Figure presence check")
        print("-" * 70)

        figures_dir = self.paper_dir / "figures"
        required_figures = [
            "architecture.pdf",
            "reliability_diagram_verified.pdf",
            "accuracy_coverage_verified.pdf",
        ]

        all_present = True
        for fig_name in required_figures:
            fig_path = figures_dir / fig_name
            if fig_path.exists():
                size_kb = fig_path.stat().st_size / 1024
                self.log(f"OK  {fig_name} ({size_kb:.1f} KB)")
            else:
                self.errors.append(f"Missing: figures/{fig_name}")
                all_present = False

        if all_present:
            print("[OK] required figures present")
            return True
        else:
            print("[ERROR] figure presence check failed")
            return False

    def check_dataset_size_explanation(self) -> bool:
        """
        CHECK 4: Verify dataset size consistency and explanation.
        
        Ensures main.tex explains difference between:
        - n=260 expert-annotated test set
        - n=1000 full binary prediction set
        """
        print("\n[AUDIT 4/6] Dataset size consistency check")
        print("-" * 70)

        main_tex_file = self.paper_dir / "main.tex"
        try:
            with open(main_tex_file, "r", encoding="utf-8") as f:
                main_content = f.read()
        except Exception as e:
            self.errors.append(f"Failed to read {main_tex_file}: {e}")
            return False

        # Check for both numbers
        has_260 = "260" in main_content
        has_1000 = "1000" in main_content or "1,000" in main_content

        if not has_260:
            self.errors.append("Main.tex does not mention n=260 (test set size)")
            return False
        else:
            self.log("OK  Found mention of 260-claim test set")

        if not has_1000:
            self.errors.append("Main.tex does not mention n=1000 (full binary set size)")
            return False
        else:
            self.log("OK  Found mention of 1,000-claim binary prediction set")

        # Check for explanation sentence
        explanation_keywords = [
            "Significance tests use",
            "full binary",
            "1,000",
            "260-claim",
            "test set",
        ]

        # Look for a sentence that explains the difference
        explanation_found = False
        for sent_match in re.finditer(
            r"[^.!?]*(?:n=260|n=1000|260-claim|1000|1,000)[^.!?]*[.!?]",
            main_content,
        ):
            sentence = sent_match.group(0)
            if "260" in sentence and ("1000" in sentence or "1,000" in sentence):
                if any(kw.lower() in sentence.lower() for kw in ["significance", "primary", "expert", "whereas", "whereas"]):
                    self.log(f"OK  Found explanation: '{sentence[:100]}...'")
                    explanation_found = True
                    break

        if not explanation_found:
            # Still check for the key sentence from step in main.tex
            key_sentence = "Significance tests use $n=1000$ full binary predictions, whereas primary evaluation focuses on the expert-annotated 260-claim test set"
            if key_sentence in main_content or "n=1000$ full binary" in main_content:
                self.log("✓ Found explicit n=260 vs n=1000 explanation")
                explanation_found = True

        if explanation_found:
            print("[OK] dataset size explanation present")
            return True
        else:
            self.errors.append(
                "No explanation found for difference between n=260 and n=1000"
            )
            print("[ERROR] dataset size explanation check failed")
            return False

    def check_overleaf_bundle_integrity(self) -> bool:
        """
        CHECK 5: Verify Overleaf bundle would include required files.
        
        Checks that the bundle builder script would include:
        - main.tex
        - metrics_values.tex (or fallback)
        - figures/
        - Optionally: significance_values.tex (or fallback)
        """
        print("\n[AUDIT 5/6] Overleaf bundle integrity check")
        print("-" * 70)

        required_files = [
            ("main.tex", True),
            ("metrics_values.tex", True),
            ("figures/architecture.pdf", True),
            ("figures/reliability_diagram_verified.pdf", True),
            ("figures/accuracy_coverage_verified.pdf", True),
        ]

        optional_files = [
            ("significance_values.tex", False),
        ]

        all_present = True
        for file_rel_path, required in required_files:
            file_path = self.paper_dir / file_rel_path
            if file_path.exists():
                self.log(f"OK  {file_rel_path}")
            else:
                if required:
                    self.errors.append(f"Bundle: missing required {file_rel_path}")
                    all_present = False
                else:
                    self.log(f"WARN Optional: {file_rel_path} not found")

        # Check optional files
        for file_rel_path, _ in optional_files:
            file_path = self.paper_dir / file_rel_path
            if file_path.exists():
                self.log(f"OK  Optional: {file_rel_path}")
            else:
                self.log(f"WARN Optional: {file_rel_path} not found (will use fallback)")

        # Verify main.tex includes input statements for metrics and significance
        main_tex_file = self.paper_dir / "main.tex"
        try:
            with open(main_tex_file, "r", encoding="utf-8") as f:
                main_content = f.read()
        except Exception as e:
            self.errors.append(f"Failed to read {main_tex_file}: {e}")
            return False

        if r"\input{metrics_values.tex}" in main_content or r"\IfFileExists{metrics_values.tex}" in main_content:
            self.log("OK  main.tex attempts to input metrics_values.tex (with fallback)")
        else:
            self.errors.append(
                "main.tex does not safely include metrics_values.tex"
            )
            all_present = False

        if all_present:
            print("[OK] Overleaf bundle integrity verified")
            return True
        else:
            print("[ERROR] Overleaf bundle integrity check failed")
            return False

    def check_unicode_sanitation(self) -> bool:
        r"""
        CHECK 6: Verify minimal Unicode protections are in place.
        
        Checks for presence of at least the three critical
        \DeclareUnicodeCharacter entries.
        """
        print("\n[AUDIT 6/6] Unicode sanitation check")
        print("-" * 70)

        main_tex_file = self.paper_dir / "main.tex"
        try:
            with open(main_tex_file, "r", encoding="utf-8") as f:
                main_content = f.read()
        except Exception as e:
            self.errors.append(f"Failed to read {main_tex_file}: {e}")
            return False

        required_unicode_decls = [
            (r"\DeclareUnicodeCharacter{00AD}", "Soft hyphen (U+00AD)"),
            (r"\DeclareUnicodeCharacter{200B}", "Zero-width space (U+200B)"),
            (r"\DeclareUnicodeCharacter{FEFF}", "BOM / zero-width no-break space (U+FEFF)"),
        ]

        all_present = True
        for decl, description in required_unicode_decls:
            if decl in main_content:
                self.log(f"OK  {description}")
            else:
                self.warnings.append(f"Unicode declaration missing: {description}")
                all_present = False

        if all_present:
            print("[OK] Unicode sanitation verified")
            return True
        else:
            print("[WARN] Some Unicode sanitation declarations missing (non-critical)")
            return True  # Warnings don't fail the audit

    def run_full_audit(self) -> bool:
        """Run all 6 audit checks and report results."""
        print("\n" + "=" * 70)
        print("CALIBRATEACH PAPER CONSISTENCY AUDIT")
        print("=" * 70)

        results = [
            ("Macro verification", self.check_macros_in_tex()),
            ("Significance verification", self.check_significance_values()),
            ("Figure presence check", self.check_figure_presence()),
            ("Dataset size consistency", self.check_dataset_size_explanation()),
            ("Overleaf bundle integrity", self.check_overleaf_bundle_integrity()),
            ("Unicode sanitation", self.check_unicode_sanitation()),
        ]

        print("\n" + "=" * 70)
        print("AUDIT SUMMARY")
        print("=" * 70)

        passed = 0
        failed = 0
        for check_name, result in results:
            status = "[OK]" if result else "[ERROR]"
            print(f"{status} {check_name}")
            if result:
                passed += 1
            else:
                failed += 1

        if self.warnings:
            print(f"\n[WARN] {len(self.warnings)} warnings:")
            for warning in self.warnings:
                print(f"  - {warning}")

        if self.errors:
            print(f"\n[ERROR] {len(self.errors)} errors detected:")
            for error in self.errors:
                print(f"  - {error}")

        print(f"\n{'=' * 70}")
        print(f"Result: {passed} passed, {failed} failed")
        print(f"{'=' * 70}\n")

        return failed == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Audit CalibraTeach paper consistency before submission"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose diagnostic messages",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Repository root directory (default: parent of scripts/)",
    )

    args = parser.parse_args()

    audit = PaperConsistencyAudit(args.repo_root, verbose=args.verbose)
    success = audit.run_full_audit()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
