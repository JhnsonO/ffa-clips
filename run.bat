@echo off
title FFA Clip Generator
echo.
echo FFA Clip Generator
echo.
echo Choose detection mode:
echo   1. Vision highlight detection
echo   2. Goal detection (local YOLO)
echo.
set /p MODE="Enter mode (1 or 2): "

if "%MODE%"=="1" goto vision
if "%MODE%"=="2" goto goals
echo Invalid mode selected.
goto end

:vision
echo.
set /p INPUT="Drag your MP4 file or folder here: "
set INPUT=%INPUT:"=%
set /p MULTI="Multi-cam? (y/n): "
if /i "%MULTI%"=="y" (
    python process.py --input "%INPUT%" --multi --output output --watch
) else (
    python process.py --input "%INPUT%" --output output --watch
)
goto after_run

:goals
echo.
set /p INPUT="Drag your GoPro MP4 file here: "
set INPUT=%INPUT:"=%
python detect_goals.py --input "%INPUT%" --output output
goto after_run

:after_run
echo.
if errorlevel 1 (
    echo [ERROR] Something went wrong. Check the output above.
    goto end
)
echo ============================================
echo  Done! Opening review page...
echo ============================================
echo.
start "" review.html

:end
pause
