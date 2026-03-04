# CalibraTeach Reproducibility Makefile
# Provides quick entry points for verification and testing

.PHONY: help quickstart verify-paper test overleaf-bundle clean

# Default target: show help
help:
	@echo "CalibraTeach Reproducibility Commands"
	@echo "======================================"
	@echo ""
	@echo "Targets:"
	@echo "  make quickstart      - Run quickstart demo (smoke mode, 5 claims)"
	@echo "  make verify-paper    - Verify paper artifacts and generate report"
	@echo "  make test            - Run all tests"
	@echo "  make overleaf-bundle - Build Overleaf submission ZIP"
	@echo "  make clean           - Remove generated artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make quickstart"
	@echo "  make verify-paper"
	@echo "  make test"

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

# Run tests (pytest if available, else unittest)
test:
	@echo "Running tests..."
	@if command -v pytest >/dev/null 2>&1; then \
		pytest tests/ -v --tb=short; \
	else \
		python -m pytest tests/ -v --tb=short; \
	fi

# Build Overleaf submission bundle
overleaf-bundle:
	@echo "Building Overleaf submission bundle..."
	python scripts/build_overleaf_bundle.py
	@echo "✓ Bundle created: dist/overleaf_submission.zip"

# Clean generated artifacts
clean:
	@echo "Cleaning artifacts..."
	rm -rf artifacts/quickstart/output.json
	rm -rf artifacts/verification/VerificationReport.md
	rm -rf dist/
	@echo "✓ Clean complete"
