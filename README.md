# FFA Clip Generator

Automatically finds and cuts highlights from your kickabout footage.

## First Time Setup

1. Install Python: https://www.python.org/downloads/
   - **Important:** tick "Add Python to PATH" during install
2. Double-click `install.bat` — this checks/installs FFmpeg automatically

## Every Session

1. Double-click `run.bat`
2. Choose mode 1 (single camera) or 2 (multi-camera)
3. Drag your MP4 file or folder in when prompted
4. Wait for processing (5–15 min depending on video length)
5. The review page opens in your browser automatically
6. Load `output/manifest.json` when prompted, then select the `output/clips/` folder
7. Approve the clips you want to keep, skip the rest
8. Export list shows exactly which files to post

## Output

All clips are saved to `output/clips/` as standard MP4 files ready to drag into Instagram, TikTok, or YouTube Shorts.

## Camera Setup

| Mode | Use when |
|------|----------|
| Single | One camera, any source (Chameleon recommended) |
| Multi  | 2–3 cameras in a folder, auto-synced by audio |

## Tuning (optional)

Edit `process.py` top section `CFG = { ... }` to adjust:
- `clip_pre` / `clip_post` — how many seconds before/after each event
- `audio_threshold` — sensitivity for crowd/impact noise (lower = more clips)
- `motion_threshold` — sensitivity for scene changes (lower = more clips)
- `min_gap` — minimum seconds between separate events
