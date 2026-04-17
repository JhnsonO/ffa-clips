"""
FFA Goal Detector (YOLO + Net Motion)
  python detect_goals.py --input video.mp4 --output output/
"""

import os, sys, json, argparse, subprocess, base64, urllib.request, urllib.error, time
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
    "net_motion_threshold": 5.0,
    "net_motion_cooldown": 15.0,
    "vision_timeout_sec": 60,
    "goal_density_window_sec": 5.0,
    "goal_min_yolo_hits": 2,
    "goal_line_zone_frac": 0.58,
    "goal_center_margin_frac": 0.18,
    "goal_net_support_window_sec": 3.0,
    "ai_review_enabled": True,
    "ai_review_model": "gpt-4o-mini",
    "ai_review_detail": "high",
    "ai_review_window_sec": 5.0,
    "ai_review_max_frames": 15,
}


GOAL_REGION_PROMPT = (
    "You are analyzing a frame from a GoPro camera mounted on the back of a football goal, looking outward at the pitch. The camera sees the crossbar and net in the lower portion of the frame, with the pitch and players above/beyond.\n\n"
    "I need you to identify where the crossbar is in this image. The crossbar is the horizontal white bar that runs across the frame — it is the top edge of the goal net.\n\n"
    "Return a JSON object with this field:\n"
    '- "net_top_y": the y-coordinate as a fraction from 0.0 (top of image) to 1.0 (bottom of image) where the crossbar sits. Everything below this line is goal net. For a typical GoPro mounted behind a 7-a-side goal, this is usually between 0.40 and 0.55.\n\n'
    "Reply with ONLY the JSON object, no other text, no markdown backticks."
)

AI_REVIEW_PROMPT = (
    "You are reviewing footage from a GoPro camera mounted on the back of a football goal, looking outward at the pitch. The crossbar and net are visible in the lower portion of the frame. The pitch and players are visible above/beyond the crossbar.\n\n"
    "These frames span approximately 10 seconds around a possible goal event, shown in chronological order.\n\n"
    "Your task: determine whether a GOAL was scored in this sequence. A goal means the ball crossed the goal line and entered the net.\n\n"
    "Look carefully for:\n"
    "- A ball traveling from the pitch side (above the crossbar) into the net area (below the crossbar)\n"
    "- The ball visibly inside the net, especially sitting on the ground behind the goal line\n"
    "- Sudden net deformation or movement caused by a ball striking it\n"
    "- Player celebrations immediately after — arms raised, running away, teammates converging\n\n"
    "Do NOT count as a goal:\n"
    "- A shot that the goalkeeper saved or caught\n"
    "- A ball that hit the post or crossbar and bounced away\n"
    "- General play near the goal without the ball entering the net\n"
    "- A ball sitting near the goal but not clearly inside the net\n\n"
    "Reply with ONLY a JSON object:\n"
    '{"goal": true, "confidence": "high", "reason": "ball visible in net at frame 7, celebrations follow"}\n'
    "or\n"
    '{"goal": false, "confidence": "high", "reason": "goalkeeper caught the ball, no net movement"}\n\n'
    "No markdown backticks, no other text."
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
            return json.loads(text[start:end + 1])
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
                confs = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else None
                best_hit = None
                for box_idx, box in enumerate(xyxy):
                    x1, y1, x2, y2 = box.tolist()
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0
                    if center_y <= goal_top_px:
                        continue
                    box_w = x2 - x1
                    box_h = y2 - y1
                    if box_w < 25 or box_h < 25:
                        continue
                    hit = {
                        "timestamp": round(float(timestamp), 3),
                        "x1": round(float(x1), 2),
                        "y1": round(float(y1), 2),
                        "x2": round(float(x2), 2),
                        "y2": round(float(y2), 2),
                        "center_x": round(float(center_x), 2),
                        "center_y": round(float(center_y), 2),
                        "conf": round(float(confs[box_idx]), 4) if confs is not None else None,
                    }
                    if best_hit is None or hit["center_y"] > best_hit["center_y"]:
                        best_hit = hit
                if best_hit is not None:
                    hits.append(best_hit)
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


def longest_streak(timestamps, max_gap=1.0):
    if not timestamps:
        return 0
    ordered = sorted(timestamps)
    best = 1
    current = 1
    for prev, curr in zip(ordered, ordered[1:]):
        if curr - prev <= max_gap:
            current += 1
            if current > best:
                best = current
        else:
            current = 1
    return best


def summarise_goal_event(event_time, ball_hits, motion_timestamps, goal_region, frame_height, frame_width):
    window = CFG["goal_density_window_sec"]
    near_hits = [
        hit for hit in ball_hits
        if abs(hit["timestamp"] - event_time) <= window
    ]
    near_motion = [
        ts for ts in motion_timestamps
        if abs(ts - event_time) <= CFG["goal_net_support_window_sec"]
    ]

    goal_top_px = goal_region["net_top_y"] * frame_height
    net_height = max(1.0, frame_height - goal_top_px)
    deep_line_y = goal_top_px + (net_height * CFG["goal_line_zone_frac"])
    center_left = frame_width * CFG["goal_center_margin_frac"]
    center_right = frame_width * (1.0 - CFG["goal_center_margin_frac"])

    central_hits = [
        hit for hit in near_hits
        if center_left <= hit["center_x"] <= center_right
    ]
    deep_hits = [
        hit for hit in near_hits
        if hit["center_y"] >= deep_line_y
    ]
    deep_central_hits = [
        hit for hit in deep_hits
        if center_left <= hit["center_x"] <= center_right
    ]

    density_count = len(near_hits)
    streak_count = longest_streak([hit["timestamp"] for hit in near_hits], max_gap=1.0)
    net_support = len(near_motion) > 0

    score = 0
    score += min(density_count, 6)
    if streak_count >= 2:
        score += 1
    if streak_count >= 3:
        score += 1
    if len(central_hits) >= 2:
        score += 1
    if len(deep_hits) >= 1:
        score += 1
    if len(deep_central_hits) >= 1:
        score += 2
    if net_support:
        score += 1

    keep = (
        density_count >= CFG["goal_min_yolo_hits"] and (
            len(deep_central_hits) >= 1
            or (density_count >= 4 and len(central_hits) >= 2 and len(deep_hits) >= 1)
            or (density_count >= 3 and len(deep_hits) >= 1 and net_support)
        )
    )

    if score >= 8:
        confidence = "high"
    elif score >= 5:
        confidence = "medium"
    else:
        confidence = "low"

    reasons = []
    reasons.append(f"yolo_hits={density_count}")
    reasons.append(f"streak={streak_count}")
    reasons.append(f"central_hits={len(central_hits)}")
    reasons.append(f"deep_hits={len(deep_hits)}")
    reasons.append(f"deep_central_hits={len(deep_central_hits)}")
    reasons.append(f"net_support={'yes' if net_support else 'no'}")

    return {
        "event_time": round(float(event_time), 3),
        "keep": keep,
        "confidence": confidence,
        "score": score,
        "yolo_hits": density_count,
        "yolo_streak": streak_count,
        "central_hits": len(central_hits),
        "deep_hits": len(deep_hits),
        "deep_central_hits": len(deep_central_hits),
        "net_support": net_support,
        "reasons": reasons,
    }


def classify_goal_events(events, ball_hits, motion_timestamps, goal_region, frame_height, frame_width):
    kept = []
    summaries = []
    rejected = []

    for event_time in events:
        summary = summarise_goal_event(
            event_time,
            ball_hits,
            motion_timestamps,
            goal_region,
            frame_height,
            frame_width,
        )
        summaries.append(summary)
        if summary["keep"]:
            kept.append(round(float(event_time), 3))
        else:
            rejected.append(summary)

    return kept, summaries, rejected


def ai_review_goals(kept_events, event_summaries, frames, fps, output_dir):
    if not kept_events:
        log_line(output_dir, "AI review: reviewing 0 candidates")
        log_line(output_dir, "AI review complete: 0 confirmed goals, 0 rejected")
        return [], event_summaries, []

    api_key = os.environ.get("OPENAI_API_KEY") or load_api_key_from_file()
    if not api_key:
        warning = "AI review: skipped (missing API key)"
        print(f"[WARN] {warning}")
        log_line(output_dir, warning)
        return kept_events, event_summaries, []

    log_line(output_dir, f"AI review: reviewing {len(kept_events)} candidates")

    summary_by_time = {
        round(float(item["event_time"]), 3): item
        for item in event_summaries
    }
    confirmed_events = []
    confirmed_summaries = []
    ai_rejected = []

    for idx, event_time in enumerate(kept_events, start=1):
        event_key = round(float(event_time), 3)
        summary = dict(summary_by_time.get(event_key, {"event_time": event_key}))
        mins = int(event_time // 60)
        secs = int(event_time % 60)
        label = f"Goal {idx} @ {mins}:{secs:02d}"

        try:
            window = CFG["ai_review_window_sec"]
            window_frames = [
                item for item in frames
                if abs(float(item[0]) - float(event_time)) <= window
            ]
            window_frames = sorted(window_frames, key=lambda item: item[0])

            if len(window_frames) < 3:
                reason = "too few frames in review window, kept by fail-open"
                summary["ai_review"] = "confirmed"
                summary["ai_review_confidence"] = "unknown"
                summary["ai_review_reason"] = reason
                confirmed_events.append(event_key)
                confirmed_summaries.append(summary)
                print(f"  AI review: {idx}/{len(kept_events)} — {label} — goal (fail-open)")
                log_line(output_dir, f"AI review [{event_key}]: {len(window_frames)} frames, goal=true, confidence=unknown, reason={reason}")
                time.sleep(0.5)
                continue

            if len(window_frames) > CFG["ai_review_max_frames"]:
                max_frames = CFG["ai_review_max_frames"]
                selected_indices = np.linspace(0, len(window_frames) - 1, num=max_frames)
                selected_indices = [int(round(i)) for i in selected_indices]
                selected_indices[0] = 0
                selected_indices[-1] = len(window_frames) - 1
                deduped = []
                seen = set()
                for i in selected_indices:
                    if i not in seen:
                        deduped.append(i)
                        seen.add(i)
                if len(deduped) < max_frames:
                    for i in range(len(window_frames)):
                        if i not in seen:
                            deduped.append(i)
                            seen.add(i)
                        if len(deduped) == max_frames:
                            break
                selected_indices = sorted(deduped[:max_frames])
                selected_frames = [window_frames[i] for i in selected_indices]
            else:
                selected_frames = window_frames

            content = [{"type": "text", "text": AI_REVIEW_PROMPT}]
            for _, frame_path in selected_frames:
                b64 = base64.b64encode(Path(frame_path).read_bytes()).decode("ascii")
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": CFG["ai_review_detail"],
                    },
                })

            body = {
                "model": CFG["ai_review_model"],
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 300,
                "temperature": 0,
            }
            resp = openai_json_request("POST", "/v1/chat/completions", api_key, body)
            text = resp["choices"][0]["message"]["content"]
            parsed = parse_json_object(text)
            frame_count = len(selected_frames)

            if not isinstance(parsed, dict) or "goal" not in parsed:
                reason = "invalid AI response, kept by fail-open"
                summary["ai_review"] = "confirmed"
                summary["ai_review_confidence"] = "unknown"
                summary["ai_review_reason"] = reason
                confirmed_events.append(event_key)
                confirmed_summaries.append(summary)
                print(f"  AI review: {idx}/{len(kept_events)} — {label} — goal (fail-open)")
                log_line(output_dir, f"AI review [{event_key}]: {frame_count} frames, goal=true, confidence=unknown, reason={reason}")
            else:
                goal_value = bool(parsed.get("goal"))
                confidence = str(parsed.get("confidence", "unknown"))
                reason = str(parsed.get("reason", ""))
                log_line(output_dir, f"AI review [{event_key}]: {frame_count} frames, goal={'true' if goal_value else 'false'}, confidence={confidence}, reason={reason}")
                if goal_value:
                    summary["ai_review"] = "confirmed"
                    summary["ai_review_confidence"] = confidence
                    summary["ai_review_reason"] = reason
                    confirmed_events.append(event_key)
                    confirmed_summaries.append(summary)
                    print(f"  AI review: {idx}/{len(kept_events)} — {label} — goal ({confidence} confidence)")
                else:
                    rejected_summary = dict(summary)
                    rejected_summary["ai_review"] = "rejected"
                    rejected_summary["ai_review_confidence"] = confidence
                    rejected_summary["ai_review_reason"] = reason
                    ai_rejected.append(rejected_summary)
                    print(f"  AI review: {idx}/{len(kept_events)} — {label} — not goal ({reason})")
        except Exception as e:
            reason = f"api error: {e}"
            summary["ai_review"] = "confirmed"
            summary["ai_review_confidence"] = "unknown"
            summary["ai_review_reason"] = reason
            confirmed_events.append(event_key)
            confirmed_summaries.append(summary)
            print(f"[WARN] AI review failed for {label}; keeping event ({e})")
            log_line(output_dir, f"AI review [{event_key}]: goal=true, confidence=unknown, reason={reason}")

        time.sleep(0.5)

    log_line(output_dir, f"AI review complete: {len(confirmed_events)} confirmed goals, {len(ai_rejected)} rejected")
    return confirmed_events, confirmed_summaries, ai_rejected


def build_manifest(events, event_summaries, video_path, output_dir, ai_reviewed_events=None):
    clips = []
    stem = video_path.stem
    summary_by_time = {
        round(float(item["event_time"]), 3): item
        for item in event_summaries
    }
    ai_reviewed_events = {
        round(float(t), 3) for t in (ai_reviewed_events or [])
    }

    for i, t in enumerate(events, start=1):
        mins = int(t // 60)
        secs = int(t % 60)
        event_key = round(float(t), 3)
        summary = summary_by_time.get(event_key, {})
        confidence = summary.get("confidence", "unknown")
        clip = {
            "file": f"{stem}_goal{i:03d}_{int(t)}s.mp4",
            "event_time": round(float(t), 1),
            "start_sec": round(max(0.0, float(t) - CFG["clip_pre"]), 3),
            "duration_sec": round(CFG["clip_pre"] + CFG["clip_post"], 3),
            "cam": stem,
            "label": f"Goal {i} @ {mins}:{secs:02d}",
            "approved": None,
            "detector": "true_goal_v1",
            "confidence": confidence,
            "score": summary.get("score"),
            "why_flagged": summary.get("reasons", []),
        }
        if event_key in ai_reviewed_events:
            clip["ai_review"] = "confirmed"
        clips.append(clip)

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
    print(f"\nFFA Goal Detector (True Goal Candidate Mode)\n\n→ Goal detection: {inp.name}")

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
    ball_hits = detect_ball_in_goal(frames, goal_region, frame_height, frame_width)
    ball_timestamps = [hit["timestamp"] for hit in ball_hits]
    print(f"  YOLO: {len(ball_hits)} ball-in-goal detections")
    log_line(out, f"YOLO detections: count={len(ball_hits)} timestamps={ball_timestamps}")
    log_line(out, f"YOLO detection details: {ball_hits}")

    print("  Running net motion detection...")
    motion_timestamps, motion_triggers, rejected_motion = detect_net_motion(frames, goal_region, frame_height)
    print(f"  Net motion: {len(motion_timestamps)} triggers")
    log_line(out, f"Net motion detections: count={len(motion_timestamps)} timestamps={motion_timestamps}")
    log_line(out, f"Net motion trigger details: {motion_triggers}")
    log_line(out, f"Net motion rejected (camera movement): count={len(rejected_motion)} timestamps={rejected_motion}")

    combined = sorted(ball_timestamps + motion_timestamps)
    merged_events = merge_detections(ball_timestamps, motion_timestamps, CFG["min_gap"])
    log_line(out, f"Combined raw detections: {combined}")
    log_line(out, f"Final merged events before goal filter: {merged_events}")

    kept_events, event_summaries, rejected_events = classify_goal_events(
        merged_events,
        ball_hits,
        motion_timestamps,
        goal_region,
        frame_height,
        frame_width,
    )
    log_line(out, f"Goal event summaries: {event_summaries}")
    log_line(out, f"Rejected merged events after goal filter: {rejected_events}")
    log_line(out, f"Final kept goal candidates: {kept_events}")

    heuristic_kept_events = list(kept_events)
    confirmed_events = list(kept_events)
    confirmed_summaries = list(event_summaries)
    ai_rejected = []

    if CFG["ai_review_enabled"]:
        confirmed_events, confirmed_summaries, ai_rejected = ai_review_goals(
            kept_events,
            event_summaries,
            frames,
            CFG["sample_fps"],
            out,
        )
        log_line(out, f"AI rejected events: {ai_rejected}")
    else:
        log_line(out, "AI review: skipped (disabled in config)")

    build_manifest(confirmed_events, confirmed_summaries, inp, out, ai_reviewed_events=confirmed_events)
    log_line(out, f"Manifest clip count: {len(confirmed_events)}")

    print(f"  Combined: {len(combined)} raw detections → {len(merged_events)} merged events")
    print(f"  Heuristic candidates: {len(heuristic_kept_events)}")
    print(f"  AI confirmed: {len(confirmed_events)}")
    print(f"  AI rejected: {len(ai_rejected)}")
    print(f"\n✓ Detection complete: {len(confirmed_events)} true-goal candidates")
    print(f"  YOLO detections: {len(ball_hits)}")
    print(f"  Net motion detections: {len(motion_timestamps)}")
    print(f"✓ Manifest written: {out / 'manifest.json'}")
    print("✓ Open review.html in your browser to review clips from source video.")


if __name__ == "__main__":
    main()
