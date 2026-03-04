# IEEE Access Submission Verification Makefile
# ============================================
# Quick commands for verifying manuscript readiness
#
# Usage:
#   make verify      # Run all verification checks (recommended)
#   make verify-fix  # Auto-fix Unicode issues, then verify all
#   make clean       # Remove temporary LaTeX files
#   make pdf         # Build PDF only (no verification)
#
# Requirements:
#   - Python 3.8+
#   - pdflatex (TeX Live, MiKTeX, or compatible)
#   - pdftotext (poppler-utils)

PYTHON := python
TEX_FILE := OVERLEAF_TEMPLATE.tex
PDF_FILE := OVERLEAF_TEMPLATE.pdf

.PHONY: all verify verify-fix verify-source verify-pdf clean pdf help

# Default target: run all verification checks
all: verify

# Run all verification checks (Unicode + Integrity + PDF build + Text extraction)
verify:
	@echo "Running unified verification pipeline..."
	$(PYTHON) verify.py

# Auto-fix Unicode issues, then run all checks
verify-fix:
	@echo "Auto-fixing Unicode issues and running verification..."
	$(PYTHON) verify.py --fix

# Verify source files only (skip PDF build)
verify-source:
	@echo "Running source verification only (skip PDF build)..."
	$(PYTHON) verify.py --skip-build

# Build PDF and verify text extraction
verify-pdf:
	@echo "Building PDF and verifying text extraction..."
	$(PYTHON) scripts/verify_pdf_text.py

# Build PDF only (no verification)
pdf:
	@echo "Building PDF (3 passes)..."
	pdflatex -interaction=nonstopmode $(TEX_FILE)
	pdflatex -interaction=nonstopmode $(TEX_FILE)
	pdflatex -interaction=nonstopmode $(TEX_FILE)
	@echo "PDF built: $(PDF_FILE)"

# Clean temporary LaTeX files
clean:
	@echo "Cleaning temporary files..."
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz
	rm -f *.bak  # Backups from sanitizer
	@echo "Cleaned temporary files"

# Clean everything including PDF
clean-all: clean
	@echo "Removing PDF..."
	rm -f $(PDF_FILE)

# Help message
help:
	@echo "IEEE Access Submission Verification Makefile"
	@echo "============================================="
	@echo ""
	@echo "Targets:"
	@echo "  make verify       - Run all verification checks (default)"
	@echo "  make verify-fix   - Auto-fix Unicode, then verify all"
	@echo "  make verify-source - Verify source only (skip PDF build)"
	@echo "  make verify-pdf   - Build PDF and verify text extraction"
	@echo "  make pdf          - Build PDF only (no verification)"
	@echo "  make clean        - Remove temporary LaTeX files"
	@echo "  make clean-all    - Remove PDF + temporary files"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Recommended workflow:"
	@echo "  1. make verify-fix    # Fix Unicode + verify all"
	@echo "  2. Fix any reported issues"
	@echo "  3. make verify        # Confirm all checks pass"
	@echo "  4. Upload to Overleaf"
