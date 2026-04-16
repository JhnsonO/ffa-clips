"""
FFA Goal Detector (YOLO + Net Motion)
  python detect_goals.py --input video.mp4 --output output/
"""

import os, sys, json, argparse, subprocess, base64, urllib.request, urllib.error
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO

CFG = {
    "clip_pre": 10,
    "clip_post": 10,
    "min_gap": 15,
    "sample_fps": 5,
    "yolo_model": "yolov8n.pt",
    "yolo_conf": 0.3,
    "yolo_imgsz": 640,
    "ball_class": 32,
    "net_motion_threshold": 8.0,
    "net_motion_cooldown": 15.0,
    "vision_timeout_sec": 60,
}

GOAL_REGION_PROMPT = (
    "You are analyzing a frame from a GoPro camera mounted on the back of a football goal, looking outward at the pitch. The camera sees the crossbar and net in the lower portion of the frame, with the pitch and players above/beyond.\n\n"
    "I need you to identify where the crossbar is in this image. The crossbar is the horizontal white bar that runs across the frame — it is the top edge of the goal net.\n\n"
    "Return a JSON object with this field:\n"
    "- \"net_top_y\": the y-coordinate as a fraction from 0.0 (top of image) to 1.0 (bottom of image) where the crossbar sits. Everything below this line is goal net. For a typical GoPro mounted behind a 7-a-side goal, this is usually between 0.40 and 0.55.\n\n"
    "Reply with ONLY the JSON object, no other text, no markdown backticks."
)


def ff(args):
    r = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return r.returncode, r.stdout + r.stderr


def duration(path):
    code, out = ff([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])
    if code != 0:
        return None
    try:
        return float(out.strip().splitlines()[-1])
    except Exception:
        return None


def log_path(output_dir):
    return output_dir / "detection.log"


def reset_log(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path(output_dir).write_text("", encoding="utf-8")


def log_line(output_dir, message):
    output_dir.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}\n"
    with log_path(output_dir).open("a", encoding="utf-8") as f:
        f.write(line)


def load_api_key_from_file():
    candidates = [
        Path(__file__).resolve().parent / "openai_key.txt",
        Path(__file__).resolve().parent / "openai_key.bat",
        Path.cwd() / "openai_key.txt",
        Path.cwd() / "openai_key.bat",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("::") or line.lower().startswith("rem "):
                    continue
                if line.lower().startswith("set "):
                    line = line[4:].strip()
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"')
        except Exception:
            pass
    return None


def require_api_key():
    api_key = os.environ.get("OPENAI_API_KEY") or load_api_key_from_file()
    if not api_key:
        sys.exit(
            "ERROR: OPENAI_API_KEY is not set in the environment and no local openai_key.txt/openai_key.bat file was found."
        )
    return api_key


def openai_json_request(method, path, api_key, body=None):
    req = urllib.request.Request(
        f"https://api.openai.com{path}",
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    with urllib.request.urlopen(req, timeout=CFG["vision_timeout_sec"]) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_json_object(text):
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            return None
    return None


def detect_goal_region(video_path, output_dir):
    cache_path = output_dir / "goal_region.json"
    frame_path = output_dir / "_goal_region_frame.jpg"

    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if "net_top_y" in data:
                print("  (delete output/goal_region.json to re-detect)")
                return {"net_top_y": float(data["net_top_y"])}
        except Exception:
            pass

    code, out = ff([
        "ffmpeg", "-y",
        "-ss", "10",
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(frame_path),
    ])
    if code != 0 or not frame_path.exists():
        fallback = {"net_top_y": 0.48}
        cache_path.write_text(json.dumps(fallback, indent=2), encoding="utf-8")
        return fallback

    api_key = require_api_key()
    b64 = base64.b64encode(frame_path.read_bytes()).decode("ascii")
    body = {
        "model": "gpt-4o-mini",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": GOAL_REGION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
            ],
        }],
        "max_tokens": 200,
        "temperature": 0,
    }

    fallback = {"net_top_y": 0.48}
    try:
        resp = openai_json_request("POST", "/v1/chat/completions", api_key, body)
        text = resp["choices"][0]["message"]["content"]
        parsed = parse_json_object(text)
        if isinstance(parsed, dict) and "net_top_y" in parsed:
            net_top_y = float(parsed["net_top_y"])
            if not (0.0 < net_top_y < 1.0):
                net_top_y = 0.48
            result = {"net_top_y": net_top_y}
        else:
            result = fallback
    except Exception:
        result = fallback

    cache_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def sample_video_frames(video_path, output_dir, fps):
    frames_dir = output_dir / "_goal_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(frames_dir.glob("frame_*.jpg"))
    if not existing:
        code, out = ff([
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-q:v", "5",
            str(frames_dir / "frame_%06d.jpg"),
        ])
        if code != 0:
            raise RuntimeError(out)
        existing = sorted(frames_dir.glob("frame_*.jpg"))

    frames = []
    for i, frame_path in enumerate(existing):
        frames.append((i / float(fps), frame_path))
    return frames


def detect_ball_in_goal(frames, goal_region, frame_height, frame_width):
    model = YOLO(CFG["yolo_model"])
    goal_top_px = int(goal_region["net_top_y"] * frame_height)
    hits = []

    for idx, (timestamp, frame_path) in enumerate(frames, start=1):
        results = model(
            str(frame_path),
            conf=CFG["yolo_conf"],
            imgsz=CFG["yolo_imgsz"],
            classes=[CFG["ball_class"]],
            verbose=False,
        )
        if results:
            result = results[0]
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                for box in xyxy:
                    x1, y1, x2, y2 = box.tolist()
                    center_y = (y1 + y2) / 2.0
                    if center_y > goal_top_px:
                        hits.append(round(float(timestamp), 3))
                        break
        if idx % 500 == 0:
            print(f"  YOLO progress: {idx}/{len(frames)} frames processed...")

    return hits


def detect_net_motion(frames, goal_region, frame_height):
    goal_top_px = int(goal_region["net_top_y"] * frame_height)
    hits = []
    triggered = []
    rejected = []
    last_trigger_ts = -9999.0
    prev_gray_blurred = None

    for idx, (timestamp, frame_path) in enumerate(frames, start=1):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_gray_blurred is not None:
            crop_prev = prev_gray_blurred[goal_top_px:, :]
            crop_curr = gray_blurred[goal_top_px:, :]
            if crop_prev.size != 0 and crop_curr.size != 0:
                net_diff_img = cv2.absdiff(crop_prev, crop_curr)
                net_diff = float(np.mean(net_diff_img))
                full_diff_img = cv2.absdiff(prev_gray_blurred, gray_blurred)
                full_diff = float(np.mean(full_diff_img))

                if net_diff > CFG["net_motion_threshold"]:
                    ts = round(float(timestamp), 3)
                    if full_diff < net_diff * 0.6:
                        if (timestamp - last_trigger_ts) >= CFG["net_motion_cooldown"]:
                            hits.append(ts)
                            triggered.append({
                                "timestamp": ts,
                                "net_diff": round(net_diff, 3),
                                "full_diff": round(full_diff, 3),
                            })
                            last_trigger_ts = timestamp
                    else:
                        rejected.append({
                            "timestamp": ts,
                            "net_diff": round(net_diff, 3),
                            "full_diff": round(full_diff, 3),
                        })
        prev_gray_blurred = gray_blurred

        if idx % 500 == 0:
            print(f"  Net motion progress: {idx}/{len(frames)} frames processed...")

    return hits, triggered, rejected


def merge(timestamps, gap):
    if not timestamps:
        return []
    timestamps = sorted(set(timestamps))
    out = [timestamps[0]]
    for t in timestamps[1:]:
        if t - out[-1] > gap:
            out.append(t)
    return out


def merge_detections(ball_timestamps, motion_timestamps, min_gap):
    return merge(list(ball_timestamps) + list(motion_timestamps), min_gap)


def build_manifest(events, video_path, output_dir):
    clips = []
    stem = video_path.stem
    for i, t in enumerate(events, start=1):
        mins = int(t // 60)
        secs = int(t % 60)
        clips.append({
            "file": f"{stem}_goal{i:03d}_{int(t)}s.mp4",
            "event_time": round(float(t), 1),
            "start_sec": round(max(0.0, float(t) - CFG["clip_pre"]), 3),
            "duration_sec": round(CFG["clip_pre"] + CFG["clip_post"], 3),
            "cam": stem,
            "label": f"Goal {i} @ {mins}:{secs:02d}",
            "approved": None,
        })

    manifest = {
        "generated": datetime.now().isoformat(),
        "clips_dir": str((output_dir / "clips").resolve()),
        "source": {
            "mode": "single",
            "source_video": str(video_path.resolve()),
        },
        "clips": clips,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    ap = argparse.ArgumentParser(description="FFA Goal Detector (YOLO + Net Motion)")
    ap.add_argument("--input", required=True, help="GoPro MP4 file")
    ap.add_argument("--output", default="output", help="Output folder (default: output/)")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    if not inp.is_file():
        sys.exit("ERROR: --input must be a .mp4 file")

    out.mkdir(parents=True, exist_ok=True)
    reset_log(out)
    log_line(out, f"Start goal detection run for {inp.resolve()}")
    print(f"\nFFA Goal Detector (YOLO + Net Motion)\n\n→ Goal detection: {inp.name}")

    goal_region = detect_goal_region(inp, out)
    print(f"  Goal region: net_top_y = {goal_region['net_top_y']:.2f}{' (cached)' if (out / 'goal_region.json').exists() else ''}")
    log_line(out, f"Goal region net_top_y={goal_region['net_top_y']:.4f}")

    print(f"  Sampling frames at {CFG['sample_fps']} fps...")
    frames = sample_video_frames(inp, out, CFG["sample_fps"])
    print(f"  Sampled {len(frames)} frames")
    log_line(out, f"Sampled {len(frames)} frames at {CFG['sample_fps']} fps")

    if not frames:
        sys.exit("ERROR: No frames were sampled from the video")

    first_frame = cv2.imread(str(frames[0][1]))
    if first_frame is None:
        sys.exit("ERROR: Could not read sampled frames")
    frame_height, frame_width = first_frame.shape[:2]

    print("  Running YOLO ball detection...")
    ball_timestamps = detect_ball_in_goal(frames, goal_region, frame_height, frame_width)
    print(f"  YOLO: {len(ball_timestamps)} ball-in-goal detections")
    log_line(out, f"YOLO detections: count={len(ball_timestamps)} timestamps={ball_timestamps}")

    print("  Running net motion detection...")
    motion_timestamps, motion_triggers, rejected_motion = detect_net_motion(frames, goal_region, frame_height)
    print(f"  Net motion: {len(motion_timestamps)} triggers")
    log_line(out, f"Net motion detections: count={len(motion_timestamps)} timestamps={motion_timestamps}")
    log_line(out, f"Net motion trigger details: {motion_triggers}")
    log_line(out, f"Net motion rejected (camera movement): count={len(rejected_motion)} timestamps={rejected_motion}")

    combined = sorted(ball_timestamps + motion_timestamps)
    events = merge_detections(ball_timestamps, motion_timestamps, CFG["min_gap"])
    log_line(out, f"Combined raw detections: {combined}")
    log_line(out, f"Final merged events: {events}")

    build_manifest(events, inp, out)
    log_line(out, f"Manifest clip count: {len(events)}")

    print(f"  Combined: {len(combined)} raw detections → {len(events)} merged events")
    print(f"\n✓ Detection complete: {len(events)} goal events")
    print(f"  YOLO detections: {len(ball_timestamps)}")
    print(f"  Net motion detections: {len(motion_timestamps)}")
    print(f"✓ Manifest written: {out / 'manifest.json'}")
    print("✓ Open review.html in your browser to review clips from source video.")


if __name__ == "__main__":
    main()
