@echo off
title FFA Clip Generator
echo.
echo ============================================
echo  FFA Clip Generator

echo  Detect only + review in browser

echo ============================================
echo.
echo  Modes:
echo    1. Single camera (one MP4 file)
echo    2. Multi camera  (folder with 2-3 MP4s)
echo.
set /p MODE="Choose mode (1 or 2): "

if "%MODE%"=="1" goto single
if "%MODE%"=="2" goto multi
echo Invalid choice. Please enter 1 or 2.
pause
exit /b 1

:single
echo.
echo Drag and drop your MP4 file here, then press Enter:
set /p INPUT="Video file: "
set INPUT=%INPUT:"=%
echo.
set OUTPUT=output
echo Running single-cam detection...
echo.
python process.py --input "%INPUT%" --output "%OUTPUT%" --watch
goto done

:multi
echo.
echo Drag and drop the FOLDER containing your MP4s here, then press Enter:
set /p INPUT="Folder: "
set INPUT=%INPUT:"=%
echo.
set OUTPUT=output
echo Running multi-cam detection with audio sync...
echo.
python process.py --input "%INPUT%" --output "%OUTPUT%" --multi --watch
goto done

:done
echo.
if errorlevel 1 (
    echo [ERROR] Something went wrong. Check the output above.
    pause
    exit /b 1
)
echo ============================================
echo  Done! Opening review page...
echo ============================================
echo.
echo In the review page:
echo   1. Load output\manifest.json
echo   2. Select the original source video or source folder
    
echo   3. Approve clips
    
echo   4. Click Export Approved and save approved_manifest.json
    
echo   5. Run export.bat

echo.
start "" review.html
pause
