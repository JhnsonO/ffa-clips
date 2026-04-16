@echo off
title FFA Clip Generator - Install
echo.
echo ============================================
echo  FFA Clip Generator - First Time Setup
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo.
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to tick "Add Python to PATH" during install.
    echo Then run this script again.
    pause
    exit /b 1
)
echo [OK] Python found.

:: Check ffmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [INFO] FFmpeg not found. Attempting to install via winget...
    winget install --id Gyan.FFmpeg -e --silent
    if errorlevel 1 (
        echo.
        echo [ERROR] Could not auto-install FFmpeg.
        echo.
        echo Please install it manually:
        echo   1. Go to https://www.gyan.dev/ffmpeg/builds/
        echo   2. Download ffmpeg-release-essentials.zip
        echo   3. Extract it and copy ffmpeg.exe to C:\Windows\System32\
        echo   4. Run this install script again.
        pause
        exit /b 1
    )
    echo [OK] FFmpeg installed.
) else (
    echo [OK] FFmpeg found.
)

echo.
echo Installing Python dependencies...
pip install ultralytics opencv-python
if errorlevel 1 (
    echo [ERROR] Failed to install Python dependencies.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Setup complete! You can now run run.bat
echo ============================================
echo.
pause
