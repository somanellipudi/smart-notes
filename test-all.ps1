# Run full test suite including external tests
# Windows-friendly Makefile alternative
#
# This runs ALL tests, including those that require network access
# and external dependencies. These may fail if resources are unavailable.

Write-Host "Running full test suite (including external tests)..." -ForegroundColor Cyan
python -m pytest -q -m ""
