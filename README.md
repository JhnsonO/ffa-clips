# FFA Clip Generator

Automatically finds highlight moments from your kickabout footage, lets you review them fast, then exports only the clips you approve.

## First Time Setup

1. Install Python: https://www.python.org/downloads/
   - **Important:** tick "Add Python to PATH" during install
2. Double-click `install.bat` — this checks/installs FFmpeg automatically

## Every Session

1. Double-click `run.bat`
2. Choose mode 1 (single camera) or 2 (multi-camera)
3. Drag your MP4 file or folder in when prompted
4. Wait for detection to finish
5. The review page opens in your browser automatically
6. Load `output/manifest.json`
7. Select the original source video or source folder
8. Approve the clips you want to keep
9. Click **Export Approved** in the browser to download `approved_manifest.json`
10. Double-click `export.bat` and provide `approved_manifest.json`

## Output

Approved clips are exported to `output/clips/`.

- Single-cam export tries `-c copy` first for near-instant cutting
- If the requested cut is not near a keyframe, it falls back to `libx264`
- Multi-cam export still uses stitched export on approval
