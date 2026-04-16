"""
FFA Clip Generator
  Detect only:
    Single cam:  python process.py --input video.mp4 --output output/
    Multi cam:   python process.py --input folder/ --multi --output output/

Detection uses OpenAI Vision via the Batch API.
Set OPENAI_API_KEY in your environment before running, or place it in a local
openai_key.txt / openai_key.bat file as: set OPENAI_API_KEY=...
"""

import os, sys, json, argparse, subprocess, struct, wave, tempfile, base64, urllib.request, urllib.error, uuid, time
from pathlib import Path
from datetime import datetime

CFG = {
    "clip_pre": 6,
    "clip_post": 10,
    "min_gap": 30,
    "audio_threshold": 2.5,
    "motion_threshold": 0.4,
    "output_res": "1920x1080",
    "output_crf": 23,
    "switch_interval": 3.0,
    "vision_sample_every": 5,
    "vision_timeout_sec": 60,
    "coarse_interval": 5,
    "refine_interval": 2,
}

COARSE_VISION_PROMPT = (
    "This is a frame from a 7-a-side football match. "
    "Does this frame show or strongly suggest any football highlight or dangerous moment, such as a goal, shot, shot build-up, clear chance, 1v1, goalmouth scramble, celebration, strong tackle, interception, dribble or skill? "
    "Reply with only YES, MAYBE, or NO. Use MAYBE if it could be an attacking or defensive highlight but the frame alone is not fully conclusive."
)

REFINE_VISION_PROMPT = (
    "This is a frame from a 7-a-side football match near a possible highlight moment. "
    "Does this frame show a clear football highlight or dangerous moment worth clipping, such as a goal, shot, clear chance, goalmouth action, celebration, strong tackle, interception, or obvious skill move? "
    "Reply with only YES or NO."
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
    except:
        return None


def extract_wav(video, wav, sr=16000):
    code, out = ff([
        "ffmpeg", "-y",
        "-i", str(video),
        "-ac", "1",
        "-ar", str(sr),
        "-vn",
        str(wav),
    ])
    if code != 0:
        raise RuntimeError(out)


def read_wav(path):
    with wave.open(str(path), 'rb') as w:
        sr, n, sw = w.getframerate(), w.getnframes(), w.getsampwidth()
        raw = w.readframes(n)
    fmt = {1: f'{n}b', 2: f'{n}h', 4: f'{n}i'}.get(sw, f'{n}h')
    samples = list(struct.unpack(fmt, raw[:sw*n]))
    mx = max(abs(s) for s in samples) or 1
    return [s / mx for s in samples], sr


def rms_windows(samples, sr, win=0.5):
    w = max(1, int(sr * win))
    out = []
    for i in range(0, len(samples) - w, w):
        chunk = samples[i:i+w]
        rms = (sum(x*x for x in chunk) / len(chunk)) ** 0.5
        out.append((i / sr, rms))
    return out


def spikes(windows, stddevs):
    vals = [r for _, r in windows]
    mean = sum(vals) / len(vals)
    std = (sum((v-mean)**2 for v in vals) / len(vals)) ** 0.5
    cutoff = mean + stddevs * std
    return [t for t, r in windows if r > cutoff]


def merge(timestamps, gap):
    if not timestamps:
        return []
    timestamps = sorted(timestamps)
    out = [timestamps[0]]
    for t in timestamps[1:]:
        if t - out[-1] > gap:
            out.append(t)
    return out


def progress_path(output_dir):
    return output_dir / "progress.json"


def load_progress(output_dir):
    path = progress_path(output_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except:
        return None


def save_progress(output_dir, data):
    progress_path(output_dir).write_text(json.dumps(data, indent=2), encoding="utf-8")


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
        except:
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


def openai_upload_file(file_path, api_key):
    boundary = f"----FFAClipBoundary{uuid.uuid4().hex}"
    file_bytes = file_path.read_bytes()
    parts = []
    parts.append(f"--{boundary}\r\n".encode("utf-8"))
    parts.append(b'Content-Disposition: form-data; name="purpose"\r\n\r\n')
    parts.append(b"batch\r\n")
    parts.append(f"--{boundary}\r\n".encode("utf-8"))
    parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'.encode("utf-8")
    )
    parts.append(b"Content-Type: application/jsonl\r\n\r\n")
    parts.append(file_bytes)
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    data = b"".join(parts)

    req = urllib.request.Request(
        "https://api.openai.com/v1/files",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=CFG["vision_timeout_sec"]) as resp:
        return json.loads(resp.read().decode("utf-8"))


def openai_create_batch(input_file_id, api_key):
    return openai_json_request(
        "POST",
        "/v1/batches",
        api_key,
        {
            "input_file_id": input_file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )


def openai_get_batch(batch_id, api_key):
    return openai_json_request("GET", f"/v1/batches/{batch_id}", api_key)


def openai_download_file(file_id, api_key):
    req = urllib.request.Request(
        f"https://api.openai.com/v1/files/{file_id}/content",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=CFG["vision_timeout_sec"]) as resp:
        return resp.read().decode("utf-8")


def batch_status_text(batch):
    status = batch.get("status", "unknown")
    if status == "failed":
        return f"failed: {batch.get('errors')}"
    return status


def wait_for_batch_completion(batch_id, api_key, label, poll_seconds):
    last_status = None
    while True:
        batch = openai_get_batch(batch_id, api_key)
        status = batch.get("status")
        if status != last_status:
            print(f"  {label} batch status: {batch_status_text(batch)}")
            last_status = status
        if status == "completed":
            return batch
        if status in {"failed", "cancelled", "expired"}:
            raise RuntimeError(f"{label} batch ended with status: {batch_status_text(batch)}")
        print(f"  Waiting {poll_seconds}s before checking again...")
        time.sleep(poll_seconds)


def sample_frames(video_path, output_dir, every_sec, prefix, width=512, quality=10):
    frames_dir = output_dir / f"_vision_frames_{prefix}"
    if frames_dir.exists():
        for old in frames_dir.glob("*.jpg"):
            old.unlink(missing_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    pattern = frames_dir / f"{prefix}_%06d.jpg"
    code, out = ff([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps=1/{every_sec},scale={width}:-2",
        "-q:v", str(quality),
        str(pattern),
    ])
    if code != 0:
        raise RuntimeError(out)

    frames = []
    for i, frame_path in enumerate(sorted(frames_dir.glob(f"{prefix}_*.jpg"))):
        frames.append((round(i * every_sec, 3), frame_path))
    return frames


def sample_frame_at(video_path, timestamp, frame_path, width=512, quality=10):
    code, out = ff([
        "ffmpeg", "-y",
        "-ss", f"{timestamp:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-vf", f"scale={width}:-2",
        "-q:v", str(quality),
        str(frame_path),
    ])
    if code != 0:
        raise RuntimeError(out)


def refine_timestamps_from_yes(yes_times, video_path):
    vid_dur = duration(video_path) or 0
    times = []
    seen = set()
    for t in yes_times:
        start = max(0, int(t - 30))
        end = int(t + 30)
        cur = start
        while cur <= end:
            ts = round(float(cur), 3)
            if vid_dur and ts > vid_dur:
                break
            if ts not in seen:
                seen.add(ts)
                times.append(ts)
            cur += CFG["refine_interval"]
    return sorted(times)


def build_refine_frames(video_path, output_dir, timestamps):
    frames_dir = output_dir / "_vision_frames_refine"
    if frames_dir.exists():
        for old in frames_dir.glob("*.jpg"):
            old.unlink(missing_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for i, ts in enumerate(timestamps):
        frame_path = frames_dir / f"refine_{i+1:06d}.jpg"
        sample_frame_at(video_path, ts, frame_path)
        frames.append((ts, frame_path))
    return frames


def build_batch_jsonl(frames, output_dir, name, prompt):
    jsonl_path = output_dir / f"{name}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for i, (ts, frame_path) in enumerate(frames):
            b64 = base64.b64encode(frame_path.read_bytes()).decode("ascii")
            req = {
                "custom_id": f"{name}_{i:06d}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{b64}",
                                        "detail": "low",
                                    },
                                },
                            ],
                        }
                    ],
                    "max_tokens": 5,
                    "temperature": 0,
                },
            }
            f.write(json.dumps(req) + "\n")
    return jsonl_path


def submit_batch_for_frames(frames, output_dir, name, api_key, prompt):
    jsonl_path = build_batch_jsonl(frames, output_dir, name, prompt)
    upload = openai_upload_file(jsonl_path, api_key)
    batch = openai_create_batch(upload["id"], api_key)
    return batch["id"]


def parse_label(text):
    text = (text or "").strip().upper()
    if text.startswith("YES"):
        return "YES"
    if text.startswith("MAYBE"):
        return "MAYBE"
    if text.startswith("NO"):
        return "NO"
    if "YES" in text:
        return "YES"
    if "MAYBE" in text:
        return "MAYBE"
    return "NO"


def collect_hits_from_batch(batch_id, timestamps, api_key, allow_maybe=False):
    batch = openai_get_batch(batch_id, api_key)
    status = batch.get("status")
    if status != "completed":
        return status, None
    output_file_id = batch.get("output_file_id")
    if not output_file_id:
        raise RuntimeError("Batch completed but no output_file_id was returned.")

    content = openai_download_file(output_file_id, api_key)
    hits = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        custom_id = row.get("custom_id", "")
        try:
            idx = int(custom_id.rsplit("_", 1)[-1])
        except:
            continue
        if idx < 0 or idx >= len(timestamps):
            continue

        body = ((row.get("response") or {}).get("body") or {})
        text = ""
        try:
            text = body["choices"][0]["message"]["content"]
        except:
            text = ""
        label = parse_label(text)
        if label == "YES" or (allow_maybe and label == "MAYBE"):
            hits.append(float(timestamps[idx]))
    return status, sorted(hits)


def vision_events(video_path, output_dir, watch=False, poll_seconds=30):
    api_key = require_api_key()
    progress = load_progress(output_dir)

    if progress and progress.get("stage") == "complete":
        return merge(progress.get("yes_timestamps", []), CFG["min_gap"])

    if not progress:
        coarse_frames = sample_frames(video_path, output_dir, CFG["coarse_interval"], "coarse")
        coarse_timestamps = [t for t, _ in coarse_frames]
        print(f"  Coarse scan: {len(coarse_frames)} frames ({CFG['coarse_interval']}s interval)")
        batch_id = submit_batch_for_frames(coarse_frames, output_dir, "coarse_batch", api_key, COARSE_VISION_PROMPT)
        save_progress(output_dir, {
            "stage": "coarse_submitted",
            "batch_id": batch_id,
            "coarse_timestamps": coarse_timestamps,
            "refine_timestamps": [],
            "yes_timestamps": [],
        })
        print(f"  Coarse batch submitted: {batch_id}")
        if not watch:
            sys.exit("Batch submitted. Run the script again later to collect results.")
        wait_for_batch_completion(batch_id, api_key, "Coarse", poll_seconds)
        progress = load_progress(output_dir)

    stage = progress.get("stage")
    batch_id = progress.get("batch_id")

    if stage == "coarse_submitted":
        if watch:
            wait_for_batch_completion(batch_id, api_key, "Coarse", poll_seconds)
        else:
            batch = openai_get_batch(batch_id, api_key)
            status = batch.get("status")
            if status != "completed":
                sys.exit(f"Coarse batch status: {batch_status_text(batch)}. Run the script again later.")

        _, coarse_hits = collect_hits_from_batch(batch_id, progress.get("coarse_timestamps", []), api_key, allow_maybe=True)
        coarse_hits = sorted(coarse_hits or [])
        if not coarse_hits:
            save_progress(output_dir, {
                "stage": "complete",
                "batch_id": batch_id,
                "coarse_timestamps": progress.get("coarse_timestamps", []),
                "refine_timestamps": [],
                "yes_timestamps": [],
            })
            return []

        refine_timestamps = refine_timestamps_from_yes(coarse_hits, video_path)
        refine_frames = build_refine_frames(video_path, output_dir, refine_timestamps)
        print(f"  Refine scan: {len(refine_frames)} frames ({CFG['refine_interval']}s interval around positives)")
        refine_batch_id = submit_batch_for_frames(refine_frames, output_dir, "refine_batch", api_key, REFINE_VISION_PROMPT)
        save_progress(output_dir, {
            "stage": "refine_submitted",
            "batch_id": refine_batch_id,
            "coarse_timestamps": progress.get("coarse_timestamps", []),
            "refine_timestamps": refine_timestamps,
            "yes_timestamps": coarse_hits,
        })
        print(f"  Refine batch submitted: {refine_batch_id}")
        if not watch:
            sys.exit("Refine batch submitted. Run the script again later to collect final results.")
        wait_for_batch_completion(refine_batch_id, api_key, "Refine", poll_seconds)
        progress = load_progress(output_dir)
        stage = progress.get("stage")
        batch_id = progress.get("batch_id")

    if stage == "refine_submitted":
        if watch:
            wait_for_batch_completion(batch_id, api_key, "Refine", poll_seconds)
        else:
            batch = openai_get_batch(batch_id, api_key)
            status = batch.get("status")
            if status != "completed":
                sys.exit(f"Refine batch status: {batch_status_text(batch)}. Run the script again later.")

        _, refine_yes = collect_hits_from_batch(batch_id, progress.get("refine_timestamps", []), api_key, allow_maybe=False)
        final_yes = sorted(refine_yes or [])
        merged = merge(final_yes, CFG["min_gap"])
        save_progress(output_dir, {
            "stage": "complete",
            "batch_id": batch_id,
            "coarse_timestamps": progress.get("coarse_timestamps", []),
            "refine_timestamps": progress.get("refine_timestamps", []),
            "yes_timestamps": final_yes,
        })
        return merged

    sys.exit(f"ERROR: Unknown progress stage: {stage}")


def all_events(video_path, output_dir, watch=False, poll_seconds=30):
    return vision_events(video_path, output_dir, watch=watch, poll_seconds=poll_seconds)


def make_meta(name, t, cam, idx):
    pre, post = CFG["clip_pre"], CFG["clip_post"]
    return {
        "file": name,
        "event_time": round(t, 1),
        "start_sec": round(max(0, t - pre), 3),
        "duration_sec": round(pre + post, 3),
        "cam": cam,
        "label": f"Event {idx} @ {int(t//60)}:{int(t%60):02d}",
        "approved": None,
    }


def cross_correlate(s_a, s_b, sr, max_sec=300):
    step = max(1, sr // 10)
    a = s_a[:sr*60:step]
    b = s_b[:sr*60:step]
    max_shift = int(max_sec * 10)
    best, best_score = 0, float('-inf')
    for shift in range(-max_shift, max_shift):
        if shift >= 0:
            av, bv = a[shift:shift+len(b)], b[:len(a)-shift]
        else:
            av, bv = a[:len(a)+shift], b[-shift:-shift+len(a)]
        n = min(len(av), len(bv))
        if n < 100:
            continue
        score = sum(av[k]*bv[k] for k in range(n))
        if score > best_score:
            best_score, best = score, shift
    return best / 10.0


def single_mode(input_path, output_dir, watch=False, poll_seconds=30):
    print(f"\n→ Single cam detect-only: {input_path.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    events = all_events(input_path, output_dir, watch=watch, poll_seconds=poll_seconds)
    print(f"  Events detected: {len(events)}")

    vid_dur = duration(input_path)
    clips = []
    for i, t in enumerate(events):
        if vid_dur and t > vid_dur:
            continue
        name = f"{input_path.stem}_event{i+1:03d}_{int(t)}s.mp4"
        clips.append(make_meta(name, t, input_path.stem, i+1))
        print(f"  • {name}")
    return clips, {
        "mode": "single",
        "source_video": str(input_path.resolve()),
    }


def multi_mode(input_dir, output_dir, watch=False, poll_seconds=30):
    mp4s = sorted(list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.MP4")))
    if len(mp4s) < 2:
        sys.exit("ERROR: Need at least 2 MP4 files in folder for --multi mode.")

    print(f"\n→ Multi cam detect-only: {len(mp4s)} cameras")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("  Syncing cameras via audio correlation...")
    all_samples = []
    ref_sr = None
    for mp4 in mp4s:
        wav = output_dir / f"_tmp_{mp4.stem}.wav"
        extract_wav(mp4, wav)
        s, sr = read_wav(wav)
        wav.unlink(missing_ok=True)
        all_samples.append(s)
        ref_sr = sr

    offsets = [0.0]
    for i in range(1, len(mp4s)):
        off = cross_correlate(all_samples[0], all_samples[i], ref_sr)
        print(f"  {mp4s[i].name}: offset {off:+.1f}s vs reference")
        offsets.append(off)

    print("  Detecting events on reference camera with OpenAI Vision Batch API...")
    events = all_events(mp4s[0], output_dir, watch=watch, poll_seconds=poll_seconds)
    print(f"  Events detected: {len(events)}")

    clips = []
    for i, t in enumerate(events):
        name = f"multicam_event{i+1:03d}_{int(t)}s.mp4"
        clips.append(make_meta(name, t, "multi", i+1))
        print(f"  • {name}")

    return clips, {
        "mode": "multi",
        "source_dir": str(input_dir.resolve()),
        "source_videos": [str(p.resolve()) for p in mp4s],
        "offsets": offsets,
    }


def main():
    ap = argparse.ArgumentParser(description="FFA Kickabout Clip Generator")
    ap.add_argument("--input", required=True, help="MP4 file (single) or folder (multi)")
    ap.add_argument("--output", default="output", help="Output folder (default: output/)")
    ap.add_argument("--multi", action="store_true", help="Multi-cam mode")
    ap.add_argument("--watch", action="store_true", help="Stay open and poll batch progress until complete")
    ap.add_argument("--poll-seconds", type=int, default=30, help="Seconds between status checks in --watch mode")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    if args.multi:
        if not inp.is_dir():
            sys.exit("ERROR: --multi requires a folder as --input")
        clips, source = multi_mode(inp, out, watch=args.watch, poll_seconds=args.poll_seconds)
    else:
        if not inp.is_file():
            sys.exit("ERROR: --input must be a .mp4 file")
        clips, source = single_mode(inp, out, watch=args.watch, poll_seconds=args.poll_seconds)

    manifest = {
        "generated": datetime.now().isoformat(),
        "clips_dir": str((out / "clips").resolve()),
        "source": source,
        "clips": clips,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n✓ Detection complete: {len(clips)} candidate clips")
    print(f"✓ Manifest written: {out / 'manifest.json'}")
    print("✓ Open review.html in your browser to review clips from source video.")


if __name__ == "__main__":
    main()
