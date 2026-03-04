# CalibraTeach Reproducibility Makefile
# Provides quick entry points for verification and testing

.PHONY: help quickstart verify-paper test test-paper test-all validate overleaf-bundle leakage-scan clean

# Default target: show help
help:
	@echo "CalibraTeach Reproducibility Commands"
	@echo "======================================"
	@echo ""
	@echo "Targets:"
	@echo "  make test-paper      - Run paper-critical tests only (default test suite)"
	@echo "  make test-all        - Run all tests including external dependencies"
	@echo "  make validate        - Validate complete paper readiness"
	@echo "  make quickstart      - Run quickstart demo (smoke mode, 5 claims)"
	@echo "  make verify-paper    - Verify paper artifacts and generate report"
	@echo "  make leakage-scan    - Run automated leakage analysis"
	@echo "  make overleaf-bundle - Build Overleaf submission ZIP"
	@echo "  make clean           - Remove generated artifacts"
	@echo ""
	@echo "Windows Users:"
	@echo "  Use PowerShell scripts: .\\test-paper.ps1 or .\\test-all.ps1"
	@echo "  Or run commands directly: python -m pytest -q -m paper"
	@echo ""
	@echo "Examples:"
	@echo "  make test-paper"
	@echo "  make validate"
	@echo "  make quickstart"

# Run paper-critical tests (default for reviewers)
test-paper:
	@echo "Running paper-critical test suite..."
	python -m pytest -q -m paper
	@echo ""
	@echo "✓ Paper tests complete!"

# Run all tests including external dependencies
test-all:
	@echo "Running full test suite (including external tests)..."
	python -m pytest -q
	@echo ""
	@echo "✓ All tests complete!"

# Validate complete paper readiness
validate:
	@echo "Validating paper readiness..."
	python scripts/validate_paper_ready.py
	@echo ""
	@echo "✓ Validation complete! Report: artifacts/TEST_STATUS_STEP2_8.md"

# Run quickstart demo in smoke mode
quickstart:
	@echo "Running CalibraTeach quickstart demo..."
	python scripts/quickstart_demo.py --smoke
	@echo ""
	@echo "✓ Quickstart complete! Output: artifacts/quickstart/output.json"

# Verify paper artifacts
verify-paper:
	@echo "Verifying paper artifacts..."
	python scripts/verify_paper_artifacts.py
	@echo ""
	@echo "✓ Verification complete! Report: artifacts/verification/VerificationReport.md"

# Run tests (pytest if available, else unittest) - kept for backward compatibility
test: test-paper

# Build Overleaf submission bundle
overleaf-bundle:
	@echo "Building Overleaf submission bundle..."
	python scripts/build_overleaf_bundle.py
	@echo "✓ Bundle created: dist/overleaf_submission.zip"

# Run automated leakage analysis
leakage-scan:
	@echo "Running automated leakage analysis..."
	python scripts/leakage_scan.py --outdir artifacts/leakage --k 5 --k2 15
	@echo "✓ Leakage analysis complete! Report: artifacts/leakage/leakage_report.json"

# Clean generated artifacts
clean:
	@echo "Cleaning artifacts..."
	rm -rf artifacts/quickstart/output.json
	rm -rf artifacts/verification/VerificationReport.md
	rm -rf dist/
	@echo "✓ Clean complete"
