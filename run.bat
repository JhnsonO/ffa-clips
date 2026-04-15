@echo off
title FFA Clip Generator
echo.
echo ============================================
echo  FFA Clip Generator
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
:: Strip quotes if dragged in
set INPUT=%INPUT:"=%
echo.
set OUTPUT=output
echo Running single-cam analysis...
echo This will take a few minutes depending on video length.
echo.
python process.py --input "%INPUT%" --output "%OUTPUT%"
goto done

:multi
echo.
echo Drag and drop the FOLDER containing your MP4s here, then press Enter:
set /p INPUT="Folder: "
set INPUT=%INPUT:"=%
echo.
set OUTPUT=output
echo Running multi-cam analysis with audio sync...
echo This will take several minutes.
echo.
python process.py --input "%INPUT%" --output "%OUTPUT%" --multi
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
echo When the review page opens:
echo   1. Click "Load manifest.json"
echo   2. Navigate to the "output" folder and select manifest.json
echo   3. Then select the "output\clips" folder when prompted
echo.
start "" review.html
pause
