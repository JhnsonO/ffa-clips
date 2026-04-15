@echo off
title FFA Clip Exporter
echo.
echo ============================================
echo  FFA Clip Exporter
echo ============================================
echo.
echo Drag and drop approved_manifest.json here, then press Enter:
set /p MANIFEST="Manifest: "
set MANIFEST=%MANIFEST:"=%

echo.
echo Exporting approved clips...
echo.
python export.py --manifest "%MANIFEST%" --output output

if errorlevel 1 (
    echo.
    echo [ERROR] Export failed. Check the output above.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Export complete

echo  Clips are in output\clips\

echo ============================================
echo.
pause
