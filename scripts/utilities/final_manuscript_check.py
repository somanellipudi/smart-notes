#!/usr/bin/env python3
"""
Final Manuscript Validation Script

Verifies:
1. No forbidden overstatement phrases
2. Table numeric consistency 
3. AI-disclosure sentence exists
4. No formatting anomalies (split decimals, etc.)
5. Statistical caution language present

Usage: python final_manuscript_check.py
"""

import re
import sys
from pathlib import Path

# Configuration
MANUSCRIPT_PATH = Path(__file__).parent.parent / "submission_bundle" / "OVERLEAF_TEMPLATE.tex"

# Forbidden phrases that indicate overstatement
FORBIDDEN_PHRASES = {
    "generalizable across domains": "Should be scoped (e.g., 'on CSClaimBench')",
    "robust across educational settings": "Should be scoped to evaluated settings",
    "state-of-the-art": "Avoid comparative claims without rigorous benchmarking",
    "demonstrates robustness": "Use 'indicates stability' or specify scope",
    "proven effective": "Use 'demonstrates feasibility' instead",
    "always works": "Avoid absolutist claims",
    "completely solves": "Acknowledge limitations explicitly",
}

# Required phrases for caution/scope qualification
REQUIRED_PHRASES = [
    r"Pedagogical benefits.*hypotheses.*RCT",
    r"demonstrated technical feasibility",
    r"non-overlapping 95% confidence intervals suggest statistically meaningful",
    r"formal hypothesis testing was not conducted",
    r"broader cross-domain validation is required",
    r"Portions of manuscript drafting were assisted by AI-based writing tools",
]

# Table formatting checks
TABLE_PATTERNS = {
    "decimal_split": r"0\.\d+\s+\d{2}\s+",  # e.g., "0.14 87" (BAD)
    "valid_decimal": r"0\.\d{4}",  # e.g., "0.1487" (GOOD)
    "valid_percent": r"\d{2}\.\d{2}%",  # e.g., "80.77%" (GOOD)
}

class ManuscriptValidator:
    def __init__(self, filepath):
        self.filepath = filepath
        self.content = None
        self.errors = []
        self.warnings = []
        
    def load(self):
        """Load manuscript content."""
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                self.content = f.read()
            return True
        except FileNotFoundError:
            self.errors.append(f"Manuscript not found: {self.filepath}")
            return False
        except Exception as e:
            self.errors.append(f"Error loading manuscript: {e}")
            return False
    
    def check_forbidden_phrases(self):
        """Check for overstatement phrases."""
        print("  • Checking for forbidden overstatement phrases...")
        for phrase, reason in FORBIDDEN_PHRASES.items():
            # Case-insensitive search
            if re.search(rf"\b{re.escape(phrase)}\b", self.content, re.IGNORECASE):
                self.errors.append(
                    f"Forbidden phrase found: '{phrase}' - {reason}"
                )
    
    def check_required_phrases(self):
        """Verify cautionary language is present."""
        print("  • Checking for required cautionary language...")
        for pattern in REQUIRED_PHRASES:
            if not re.search(pattern, self.content, re.IGNORECASE):
                self.warnings.append(
                    f"Expected cautionary phrase not found: {pattern[:50]}..."
                )
    
    def check_table_formatting(self):
        """Check for formatting anomalies in tables."""
        print("  • Checking table numeric consistency...")
        
        # Find all table sections
        table_matches = re.finditer(
            r"\\begin\{tabular\}.*?\\end\{tabular\}",
            self.content,
            re.DOTALL
        )
        
        table_num = 0
        for table_match in table_matches:
            table_num += 1
            table_content = table_match.group(0)
            
            # Check for split decimals (e.g., "0.14 87")
            if re.search(TABLE_PATTERNS["decimal_split"], table_content):
                self.errors.append(
                    f"Table {table_num}: Found split decimal - e.g., '0.14 87'. "
                    "Should be '0.1487'"
                )
            
            # Count numeric consistency
            decimals = re.findall(r"(\d+\.\d+(?:\d+)?)", table_content)
            if decimals:
                # Check if ECE values are 4-digit decimals
                ece_like = [d for d in decimals if 0.08 < float(d) < 0.25]
                if ece_like:
                    lengths = [len(d.split('.')[1]) for d in ece_like]
                    if not all(l >= 4 for l in lengths):
                        self.warnings.append(
                            f"Table {table_num}: ECE values may lack precision. "
                            "Recommend 4 decimal places for calibration metrics."
                        )
    
    def check_ai_disclosure(self):
        """Verify AI-assistance disclosure exists."""
        print("  • Checking for AI-disclosure statement...")
        if not re.search(
            r"AI-based writing tools.*reviewed.*edited.*verified by the authors",
            self.content,
            re.IGNORECASE
        ):
            self.errors.append(
                "AI-disclosure statement not found in Acknowledgments section. "
                "Add: 'Portions of manuscript drafting were assisted by AI-based "
                "writing tools and subsequently reviewed, edited, and verified by the authors.'"
            )
    
    def check_statistical_language(self):
        """Verify statistical caution language."""
        print("  • Checking statistical caution language...")
        
        # Should NOT have "demonstrates statistical significance" without qualifier
        if re.search(
            r"demonstrates statistical significance(?!.*formal hypothesis testing|suggest.*meaningful)",
            self.content
        ):
            self.warnings.append(
                "Found 'demonstrates statistical significance' - "
                "Consider adding 'suggest statistically meaningful' with caution qualifier"
            )
    
    def check_no_emojis(self):
        """Verify no emojis present."""
        print("  • Checking for emoji usage...")
        # Common emoji code points (rough check)
        emoji_ranges = r"[\U0001F300-\U0001F9FF]|[\u2600-\u27BF]"
        if re.search(emoji_ranges, self.content):
            self.errors.append("Found emoji in manuscript - remove for IEEE format")
    
    def check_percentage_format(self):
        """Verify consistent % format usage."""
        print("  • Checking percentage formatting...")
        
        # Should use % symbol, not "percent" in numeric contexts
        if re.search(r"\d+\s+percent\b", self.content, re.IGNORECASE):
            self.warnings.append(
                "Found 'N percent' - consider using '%' symbol for consistency"
            )
    
    def validate(self):
        """Run all validation checks."""
        if not self.load():
            return False
        
        print("\n" + "="*70)
        print("FINAL MANUSCRIPT VALIDATION")
        print("="*70)
        
        self.check_forbidden_phrases()
        self.check_required_phrases()
        self.check_table_formatting()
        self.check_ai_disclosure()
        self.check_statistical_language()
        self.check_no_emojis()
        self.check_percentage_format()
        
        # Print results
        print("\n" + "-"*70)
        if self.errors:
            print(f"\n❌ ERRORS FOUND ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"   {i}. {error}")
        else:
            print("\n✓ No critical errors found")
        
        if self.warnings:
            print(f"\n⚠ WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")
        else:
            print("✓ No warnings")
        
        print("\n" + "-"*70)
        
        # Final status
        if self.errors:
            print("\n STATUS: ❌ NEEDS FIXES")
            print("   Please address errors above before submission.")
            return False
        elif self.warnings:
            print("\n STATUS: ⚠ REVIEW RECOMMENDED")
            print("   Warnings are informational; manual review suggested.")
            return True
        else:
            print("\n STATUS: ✓ PASS")
            print("   Manuscript ready for final submission!")
            return True


def main():
    """Main entry point."""
    validator = ManuscriptValidator(MANUSCRIPT_PATH)
    success = validator.validate()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
