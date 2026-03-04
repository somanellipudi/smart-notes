# Smart-Notes Test Commands
# Windows-friendly Makefile alternative (PowerShell commands)
#
# Usage:
#   .\test-paper.ps1     # Run paper-critical tests
#   .\test-all.ps1       # Run all tests including external
#
# Or manually:
#   python -m pytest -q -m paper
#   python -m pytest -q

# Run paper-critical test suite (default)
Write-Host "Running paper-critical test suite..." -ForegroundColor Cyan
python -m pytest -q -m paper
