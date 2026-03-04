@echo off
REM =====================================================================
REM IEEE ACCESS MANUSCRIPT - FINAL FIX COMPILATION & VERIFICATION
REM Purpose: Compile LaTeX with reference resolution and verify fixes
REM =====================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0CalibraTeach_IEEE_Access_Upload" || exit /b 1

echo.
echo =====================================================================
echo FINAL FIX COMPILATION SCRIPT
echo =====================================================================
echo This script will:
echo  1. Clean auxiliary files
echo  2. Compile TeX source (Pass 1 - collect labels)
echo  3. Compile TeX source (Pass 2 - resolve references)
echo  4. Verify forbidden strings are absent
echo =====================================================================
echo.

REM Step 1: Clean old files
echo [STEP 1] Cleaning auxiliary files...
del /q *.aux *.log *.out *.fls *.pdf 2>nul
echo  Done.
echo.

REM Step 2: Attempt compilation with available engine
set FOUND_ENGINE=0
for %%E in (pdflatex latexmk lualatex xelatex) do (
    if !FOUND_ENGINE! equ 0 (
        where /q %%E
        if !errorlevel! equ 0 (
            set FOUND_ENGINE=1
            set LATEX_ENGINE=%%E
            echo [STEP 2] Found LaTeX engine: %%E
        )
    )
)

if !FOUND_ENGINE! equ 0 (
    echo [ERROR] No LaTeX engines found in PATH
    echo.
    echo  Please install one of:
    echo   - MikTeX: choco install miktex
    echo   - TeX Live: choco install texlive
    echo   - Tectonic: cargo install tectonic
    echo.
    echo  Or use Overleaf: https://www.overleaf.com/project?
    exit /b 1
)

REM Step 3: Compile Pass 1
echo [STEP 3] Compiling Pass 1 (collect labels)...
%LATEX_ENGINE% -interaction=nonstopmode -halt-on-error OVERLEAF_TEMPLATE.tex >nul 2>&1
if !errorlevel! neq 0 (
    echo  ERROR: Compilation Pass 1 failed
    echo  Run manually: %LATEX_ENGINE% OVERLEAF_TEMPLATE.tex
    exit /b 1
)
echo  Done.
echo.

REM Step 4: Compile Pass 2
echo [STEP 4] Compiling Pass 2 (resolve references)...
%LATEX_ENGINE% -interaction=nonstopmode -halt-on-error OVERLEAF_TEMPLATE.tex >nul 2>&1
if !errorlevel! neq 0 (
    echo  ERROR: Compilation Pass 2 failed
    exit /b 1
)
echo  Done.
echo.

REM Step 5: Verify PDF created
if not exist OVERLEAF_TEMPLATE.pdf (
    echo [ERROR] PDF not created
    exit /b 1
)

echo [STEP 5] PDF created successfully: OVERLEAF_TEMPLATE.pdf
echo.

REM Step 6: Verify forbidden strings not in source
echo [STEP 6] Verifying source-level checks...
findstr /M "Sec. III-B" OVERLEAF_TEMPLATE.tex >nul 2>&1
if !errorlevel! equ 0 (
    echo  FAIL: Found "Sec. III-B" in source
    exit /b 1
) else (
    echo  PASS: No "Sec. III-B" in source
)

findstr /M "Eq. (6), Section III-B" OVERLEAF_TEMPLATE.tex >nul 2>&1
if !errorlevel! equ 0 (
    echo  FAIL: Found "Eq. (6), Section III-B" in source
    exit /b 1
) else (
    echo  PASS: No "Eq. (6), Section III-B" in source
)
echo.

echo =====================================================================
echo SUCCESS: All verifications PASSED
echo =====================================================================
echo.
echo  Next steps:
echo   1. Open OVERLEAF_TEMPLATE.pdf
echo   2. Search for "Equation (5)" or "S margin" - verify two-line display
echo   3. Search for "Authority Weight" table - verify auto-numbered section
echo   4. Search for "Authority Weights" appendix - verify auto-numbered refs
echo.
echo  Ready for submission to IEEE Access!
echo.
