"""
Project Cleanup & Organization Guide

This document identifies obsolete/redundant files and cleanup steps for Smart Notes.

Generated: 2026-02-17
"""

# ============================================================================
# ROOT LEVEL TEST FILES - CLEANUP CANDIDATES
# ============================================================================

ROOT_TEST_FILES_TO_ARCHIVE = [
    "test_config.py",                   # Config testing - move to tests/
    "test_export_fix.py",               # Bug fix test - consider removing
    "test_fix_verification.py",         # Bug fix test - consider removing
    "test_graph_metrics_fix.py",        # Bug fix test - now in tests/
    "test_ingestion_vs_rejection.py",   # Feature test - move to tests/
    "test_pdf_end_to_end.py",          # PDF testing - move to tests/
    "test_pdf_extraction.py",          # PDF testing - move to tests/
    "test_pdf_fix.py",                 # Bug fix test - consider removing
    "test_pdf_pipeline_demo.py",       # Demo - can remove
    "test_pdf_upload_bug.py",          # Bug fix test - consider removing
    "test_summary_padding.py",         # Bug fix test - consider removing
]

CLEANUP_SCRIPTS = [
    "validate_fixes.py",               # Legacy validation - archive
    "run_cli.py",                      # CLI runner - check if used
    "validate_pdf_ingestion.py",       # Legacy validation - archive
    "verify_implementation.py",        # Legacy verification - archive
    "verify_pdf_ocr_implementation.py",# Legacy verification - archive
    "diagnose.py",                     # Diagnostic tool - check if used
]

# ============================================================================
# DIRECTORY CLEANUP
# ============================================================================

DIRECTORIES_TO_CONSOLIDATE = {
    "cache/": "Keep if needed for OCR caching, verify no sensitive data",
    "profiling/": "Remove if empty, set up proper profiling in tools/",
    "outputs/sessions/": "Archive old sessions periodically",
}

DIRECTORIES_TO_VERIFY = {
    "artifacts/": "Check for build artifacts to exclude from git",
    ".pytest_cache/": "Should be in .gitignore (already is)",
    "__pycache__/": "Should be in .gitignore (already is)",
}

# ============================================================================
# CONFIGURATION FILES - CLEANUP RECOMMENDATIONS
# ============================================================================

CONFIG_FILES = {
    ".env": "Keep secret - add to .gitignore if not already",
    ".env.example": "Keep as template",
    "secrets.toml.example": "Keep as template",
    ".gitignore": "Verify includes: __pycache__, .pytest_cache, .venv, cache/, artifacts/",
}

# ============================================================================
# DOCUMENTATION CLEANUP
# ============================================================================

DOCUMENTATION_STATUS = {
    "README.md": "✅ Main documentation",
    "README_RUN.md": "✅ Running guide",
    "RESEARCH_FOUNDATION.md": "Check if superseded by docs/",
    "docs/README.md": "✅ Documentation index",
    "docs/ARCH_FLOW.md": "✅ Architecture flow",
    "docs/PROJECT_STRUCTURE.md": "✅ NEW - Comprehensive structure guide",
    "docs/QUICK_REFERENCE_RESEARCH.md": "Review - update if outdated",
}

# ============================================================================
# RECOMMENDED CLEANUP STEPS
# ============================================================================

CLEANUP_CHECKLIST = """
PHASE 1: LOW RISK
===============
[ ] Run all tests to ensure nothing breaks
    pytest tests/ -v

[ ] Archive root-level bug fix tests (didn't break anything)
    - Move test_fix_*.py to tests/archived/old_fixes/
    - Move test_*_bug_report.py to tests/archived/

[ ] Create tests/archived/ directory with README explaining why archived

PHASE 2: MEDIUM RISK
====================
[ ] Review and consolidate legacy validation scripts
    - verify_*.py → consider archiving or integrating
    - validate_*.py → confirm tests cover same functionality
    - diagnose.py → check if still used by team

[ ] Clean up root test files (move to tests/)
    - test_config.py → tests/test_configuration.py
    - test_ingestion_*.py → tests/test_ingestion_*.py
    - test_pdf_*.py (non-archival) → tests/

[ ] Remove obsolete demo files if not part of examples/
    - test_pdf_pipeline_demo.py → examples/ if needed

PHASE 3: DOCUMENTATION
======================
[ ] Update README.md to reference docs/PROJECT_STRUCTURE.md

[ ] Consolidate docs/ - verify all documentation files are current
    - Remove or archive old docs (e.g. QUICK_REFERENCE_RESEARCH if covered elsewhere)
    - Make docs/ the canonical documentation location

[ ] Add CONTRIBUTING.md with:
    - How to add new tests (must go in tests/, organized by category)
    - How to add new datasets (cs_benchmark/ only growth point)
    - How to run benchmarks locally

PHASE 4: VERIFICATION
====================
[ ] Run full test suite
    pytest tests/ -v --tb=short

[ ] Run evaluation benchmarks
    python scripts/run_cs_benchmark.py --sample-size 5

[ ] Generate test coverage report
    pytest tests/ --cov=src --cov-report=term-missing

[ ] Commit changes with message:
    "refactor: consolidate tests and documentation, archive legacy files"
"""

# ============================================================================
# DIRECTORY STRUCTURE AFTER CLEANUP
# ============================================================================

PROPOSED_STRUCTURE = """
smart-notes/ (AFTER CLEANUP)
├── src/                       ✅ Application code
├── evaluation/                ✅ Research infrastructure
│   ├── cs_benchmark/
│   └── results/
├── tests/                     ✅ All test files organized
│   ├── test_*.py             (35 organized tests)
│   ├── archived/             (old bug fix tests)
│   └── README.md             (test organization guide)
├── scripts/                   ✅ Utilities
├── examples/                  ✅ Usage examples  
├── docs/                      ✅ Documentation
│   ├── PROJECT_STRUCTURE.md  (NEW)
│   ├── CONTRIBUTING.md       (NEW)
│   └── ...
├── data/                      ✅ Data files
├── logs/                      ✅ Logs
├── cache/                     ✅ Cache (verify empty or necessary)
└── outputs/                   ✅ Results
    └── sessions/

❌ REMOVED FILES:
- Root test_*.py files (moved to tests/)
- Obsolete validate_*.py scripts (archived or integrated)
- Obsolete verify_*.py scripts (archived)
"""

# ============================================================================
# BENEFITS OF CLEANUP
# ============================================================================

BENEFITS = """
1. ORGANIZATION
   - Clear separation: tests in tests/, not root
   - Documentation in docs/, not scattered
   - Benchmarks isolated in evaluation/

2. MAINTAINABILITY
   - Easy to find all tests (tests/ directory)
   - Easy to understand structure (docs/PROJECT_STRUCTURE.md)
   - Clear growth path (add datasets to cs_benchmark/)

3. RESEARCH PUBLISHING
   - Evaluation infrastructure is front-and-center
   - Results reproducibility guaranteed (seeds, versions)
   - Publication-ready outputs (CSV, markdown, JSON)

4. ONBOARDING
   - New contributers read docs/PROJECT_STRUCTURE.md
   - New developers read docs/CONTRIBUTING.md
   - Test organization makes adding tests intuitive

5. CI/CD
   - Easier to configure test discovery
   - Clear artifact paths for publishing
   - Reproducible benchmark builds
"""

if __name__ == "__main__":
    print(__doc__)
    print(CLEANUP_CHECKLIST)
    print("\n" + PROPOSED_STRUCTURE)
    print("\n" + BENEFITS)
