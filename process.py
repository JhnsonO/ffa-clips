"""
FFA Clip Generator
  Detect only:
    Single cam:  python process.py --input video.mp4 --output output/
    Multi cam:   python process.py --input folder/ --multi --output output/

Detection uses OpenAI Vision on sampled frames.
Set OPENAI_API_KEY in your environment before running.
"""

import os, sys, json, argparse, subprocess, struct, wave, tempfile, base64, urllib.request, urllib.error
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
}


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
    out = [timestamps[0]]
    for t in timestamps[1:]:
        if t - out[-1] > gap:
            out.append(t)
    return out


def sample_frames(video_path, output_dir, every_sec):
    frames_dir = output_dir / "_vision_frames"
    if frames_dir.exists():
        for old in frames_dir.glob("*.jpg"):
            old.unlink(missing_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    pattern = frames_dir / "frame_%06d.jpg"
    code, out = ff([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps=1/{every_sec}",
        "-q:v", "2",
        str(pattern),
    ])
    if code != 0:
        raise RuntimeError(out)

    frames = []
    for i, frame_path in enumerate(sorted(frames_dir.glob("frame_*.jpg"))):
        frames.append((i * every_sec, frame_path))
    return frames


def ask_vision_yes_no(frame_path, api_key):
    prompt = (
        "This is a frame from a 7-a-side football match. "
        "Does this frame show an exciting moment — a goal, shot, skill, tackle, or celebration? "
        "Reply with just YES or NO."
    )
    b64 = base64.b64encode(frame_path.read_bytes()).decode("ascii")
    body = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ],
        "max_tokens": 3,
        "temperature": 0,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=CFG["vision_timeout_sec"]) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = data["choices"][0]["message"]["content"].strip().upper()
    return text.startswith("YES")


def vision_events(video_path, output_dir):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY is not set in the environment.")

    frames = sample_frames(video_path, output_dir, CFG["vision_sample_every"])
    yes_times = []

    print(f"  Vision sampling: {len(frames)} frames ({CFG['vision_sample_every']}s interval)")
    for t, frame_path in frames:
        try:
            is_exciting = ask_vision_yes_no(frame_path, api_key)
        except urllib.error.HTTPError as e:
            details = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Vision API HTTP {e.code}: {details}")
        except Exception as e:
            raise RuntimeError(f"Vision API request failed: {e}")

        status = "YES" if is_exciting else "NO "
        print(f"  [{status}] {int(t//60)}:{int(t%60):02d}  {frame_path.name}")
        if is_exciting:
            yes_times.append(float(t))

    return merge(yes_times, CFG["min_gap"])


def all_events(video_path, output_dir):
    return vision_events(video_path, output_dir)


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


def single_mode(input_path, output_dir):
    print(f"\n→ Single cam detect-only: {input_path.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    events = all_events(input_path, output_dir)
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


def multi_mode(input_dir, output_dir):
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

    print("  Detecting events on reference camera with OpenAI Vision...")
    events = all_events(mp4s[0], output_dir)
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
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    if args.multi:
        if not inp.is_dir():
            sys.exit("ERROR: --multi requires a folder as --input")
        clips, source = multi_mode(inp, out)
    else:
        if not inp.is_file():
            sys.exit("ERROR: --input must be a .mp4 file")
        clips, source = single_mode(inp, out)

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
