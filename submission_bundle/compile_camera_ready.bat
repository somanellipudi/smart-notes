@echo off
REM Compilation script for camera-ready IEEE Access manuscript
REM Compiles LaTeX twice to resolve references and generates PDF
REM 
REM Place this file in the CalibraTeach_IEEE_Access_Upload folder and run it from there.
REM Or run from parent directory if folder structure is: parent/CalibraTeach_IEEE_Access_Upload/

REM If running from parent, change to the folder
if not exist OVERLEAF_TEMPLATE.tex (
    cd /d "%~dp0CalibraTeach_IEEE_Access_Upload"
)

if not exist OVERLEAF_TEMPLATE.tex (
    echo Error: OVERLEAF_TEMPLATE.tex not found
    echo Please run this script from the CalibraTeach_IEEE_Access_Upload folder
    exit /b 1
)

echo.
echo ================================================
echo IEEE Access Manuscript Final Compilation
echo ================================================
echo.

REM Check if pdflatex is available
where pdflatex >nul 2>&1
if errorlevel 1 (
    echo ERROR: pdflatex not found in PATH
    echo Please install MikTeX or TeX Live first:
    echo   Windows: choco install miktex
    echo   Or download from: https://miktex.org/download
    echo.
    echo Alternative: Use Overleaf (no installation needed)
    echo   https://www.overleaf.com
    exit /b 1
)

REM Remove old files
echo Cleaning old files...
del /Q OVERLEAF_TEMPLATE.pdf 2>nul
del /Q OVERLEAF_TEMPLATE.aux 2>nul
del /Q OVERLEAF_TEMPLATE.log 2>nul
del /Q OVERLEAF_TEMPLATE.out 2>nul
del /Q OVERLEAF_TEMPLATE.toc 2>nul
del /Q OVERLEAF_TEMPLATE.fls 2>nul
del /Q OVERLEAF_TEMPLATE.fdb_latexmk 2>nul

REM Compilation Pass 1
echo.
echo ================================================
echo Pass 1: Collecting labels and references...
echo ================================================
pdflatex -interaction=nonstopmode -halt-on-error OVERLEAF_TEMPLATE.tex >pass1.log 2>&1
if errorlevel 1 (
    echo ERROR: Compilation pass 1 failed
    echo See pass1.log for details
    exit /b 1
)
echo Pass 1 completed successfully

REM Compilation Pass 2
echo.
echo ================================================
echo Pass 2: Resolving references...
echo ================================================
pdflatex -interaction=nonstopmode -halt-on-error OVERLEAF_TEMPLATE.tex >pass2.log 2>&1
if errorlevel 1 (
    echo ERROR: Compilation pass 2 failed
    echo See pass2.log for details
    exit /b 1
)
echo Pass 2 completed successfully

REM Verify output
if not exist OVERLEAF_TEMPLATE.pdf (
    echo ERROR: PDF not generated
    exit /b 1
)

for /F "usebackq" %%A in ('OVERLEAF_TEMPLATE.pdf') do set size=%%~zA
set /A sizekb=%size% / 1024
echo.
echo ================================================
echo SUCCESS: PDF generated
echo ================================================
echo Output: OVERLEAF_TEMPLATE.pdf (%sizekb% KB)
echo.

REM Verification checks
echo Performing verification checks...
echo.

REM Check 1: No forbidden strings in source
echo [CHECK 1] Forbidden strings in source
findstr /L "Sec. III-B" OVERLEAF_TEMPLATE.tex >nul 2>&1
if errorlevel 1 (
    echo   - "Sec. III-B": NOT FOUND [PASS]
) else (
    echo   - "Sec. III-B": FOUND [FAIL]
)

findstr /L "Eq. (6), Section III-B" OVERLEAF_TEMPLATE.tex >nul 2>&1
if errorlevel 1 (
    echo   - "Eq. (6), Section III-B": NOT FOUND [PASS]
) else (
    echo   - "Eq. (6), Section III-B": FOUND [FAIL]
)

REM Check 2: Labels present
echo.
echo [CHECK 2] Required labels
findstr "\\label{sec:multi_component_ensemble}" OVERLEAF_TEMPLATE.tex >nul 2>&1
if not errorlevel 1 (
    echo   - sec:multi_component_ensemble: FOUND [PASS]
)

findstr "\\label{eq:auth_score}" OVERLEAF_TEMPLATE.tex >nul 2>&1
if not errorlevel 1 (
    echo   - eq:auth_score: FOUND [PASS]
)

REM Check 3: Metrics
echo.
echo [CHECK 3] Metrics preserved
findstr "80.77" OVERLEAF_TEMPLATE.tex >nul 2>&1
if not errorlevel 1 (
    echo   - Accuracy 80.77%%: FOUND [PASS]
)

findstr "0.1076" OVERLEAF_TEMPLATE.tex >nul 2>&1
if not errorlevel 1 (
    echo   - ECE 0.1076: FOUND [PASS]
)

findstr "0.8711" OVERLEAF_TEMPLATE.tex >nul 2>&1
if not errorlevel 1 (
    echo   - AUC-AC 0.8711: FOUND [PASS]
)

echo.
echo ================================================
echo Next steps:
echo ================================================
echo 1. Open OVERLEAF_TEMPLATE.pdf
echo 2. Verify:
echo    - Equation (5) on two lines
echo    - Table VII with auto-numbered section
echo    - Appendix with auto-numbered references
echo 3. Search PDF for "Sec. III-B" (should find 0)
echo 4. Search PDF for "Eq. (6). Section III-B" (should find 0)
echo 5. Upload to IEEE Access
echo.

pause
