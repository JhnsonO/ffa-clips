"""
FFA Clip Generator
  Single cam:  python process.py --input video.mp4 --output clips/
  Multi cam:   python process.py --input folder/ --multi --output clips/
"""

import os, sys, json, argparse, subprocess, struct, wave, shutil, tempfile
from pathlib import Path
from datetime import datetime

CFG = {
    "clip_pre":          6,
    "clip_post":         10,
    "min_gap":           8,
    "audio_threshold":   2.5,
    "motion_threshold":  0.15,   # scene change sensitivity (0-1)
    "output_res":        "1920x1080",
    "output_crf":        23,
    "switch_interval":   3.0,    # seconds per cam in multi-cam clips
}

# ── FFmpeg helpers ────────────────────────────────────────────────────────────

def ff(cmd):
    """Run ffmpeg/ffprobe command, return stdout."""
    r = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return r.stdout + r.stderr


def duration(path):
    out = ff(f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{path}"')
    try:
        return float(out.strip().splitlines()[-1])
    except:
        return None


def extract_wav(video, wav, sr=16000):
    ff(f'ffmpeg -y -i "{video}" -ac 1 -ar {sr} -vn "{wav}"')


# ── Audio analysis (pure Python, no numpy) ───────────────────────────────────

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


def audio_events(video_path, output_dir):
    wav = output_dir / "_tmp_audio.wav"
    extract_wav(video_path, wav)
    samples, sr = read_wav(wav)
    wav.unlink(missing_ok=True)
    wins = rms_windows(samples, sr)
    return merge(spikes(wins, CFG["audio_threshold"]), CFG["min_gap"])


def motion_events(video_path):
    out = ff(
        f'ffmpeg -i "{video_path}" '
        f'-vf "select=gt(scene\\,{CFG["motion_threshold"]}),metadata=print:file=-" '
        f'-an -f null -'
    )
    times = []
    for line in out.splitlines():
        if "pts_time:" in line:
            try:
                times.append(float(line.split("pts_time:")[-1].split()[0]))
            except:
                pass
    return merge(times, CFG["min_gap"])


def all_events(video_path, output_dir):
    ae = audio_events(video_path, output_dir)
    me = motion_events(video_path)
    return merge(sorted(set(ae + me)), CFG["min_gap"])


# ── Clip extraction ───────────────────────────────────────────────────────────

def extract_clip(video, start, dur, out):
    res = CFG["output_res"]
    crf = CFG["output_crf"]
    ff(
        f'ffmpeg -y -ss {max(0,start):.2f} -i "{video}" -t {dur:.2f} '
        f'-vf "scale={res}:force_original_aspect_ratio=decrease,pad={res}:(ow-iw)/2:(oh-ih)/2" '
        f'-c:v libx264 -crf {crf} -preset fast -c:a aac -b:a 128k "{out}"'
    )


def clips_from_events(events, video, clips_dir, label):
    pre, post = CFG["clip_pre"], CFG["clip_post"]
    vid_dur = duration(video)
    results = []
    for i, t in enumerate(events):
        if vid_dur and t > vid_dur:
            continue
        name = f"{label}_event{i+1:03d}_{int(t)}s.mp4"
        extract_clip(video, t - pre, pre + post, clips_dir / name)
        results.append(make_meta(name, t, label, i+1))
        print(f"  ✓ {name}")
    return results


def make_meta(name, t, cam, idx):
    return {
        "file": name,
        "event_time": round(t, 1),
        "cam": cam,
        "label": f"Event {idx} @ {int(t//60)}:{int(t%60):02d}",
        "approved": None
    }


# ── Multi-cam: dynamic switching ─────────────────────────────────────────────

def cross_correlate(s_a, s_b, sr, max_sec=300):
    """Find offset of s_b relative to s_a. Returns seconds to add to s_b timestamps."""
    step = max(1, sr // 10)   # downsample to ~10Hz
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


def multicam_clip(vids_offsets, event_time, out_path):
    """vids_offsets: list of (path, offset). offset = seconds to subtract from reference time."""
    pre, post = CFG["clip_pre"], CFG["clip_post"]
    clip_dur = pre + post
    sw = CFG["switch_interval"]
    tmpdir = Path(tempfile.mkdtemp())
    seg_files = []
    cam_idx = 0
    t = 0.0

    while t < clip_dur:
        seg_dur = min(sw, clip_dur - t)
        vid, offset = vids_offsets[cam_idx % len(vids_offsets)]
        local_start = max(0, (event_time - pre + t) - offset)
        seg_out = tmpdir / f"seg{len(seg_files):03d}.mp4"
        extract_clip(vid, local_start, seg_dur, seg_out)
        seg_files.append(seg_out)
        t += seg_dur
        cam_idx += 1

    concat = tmpdir / "list.txt"
    concat.write_text("\n".join(f"file '{p}'" for p in seg_files))
    ff(f'ffmpeg -y -f concat -safe 0 -i "{concat}" -c copy "{out_path}"')
    shutil.rmtree(tmpdir, ignore_errors=True)


# ── Modes ─────────────────────────────────────────────────────────────────────

def single_mode(input_path, output_dir):
    print(f"\n→ Single cam: {input_path.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    events = all_events(input_path, output_dir)
    print(f"  Events detected: {len(events)}")
    return clips_from_events(events, input_path, clips_dir, input_path.stem)


def multi_mode(input_dir, output_dir):
    mp4s = sorted(list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.MP4")))
    if len(mp4s) < 2:
        sys.exit("ERROR: Need at least 2 MP4 files in folder for --multi mode.")

    print(f"\n→ Multi cam: {len(mp4s)} cameras")
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    # Extract audio and compute offsets relative to first camera
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

    # Detect events on reference camera
    print("  Detecting events on reference camera...")
    events = all_events(mp4s[0], output_dir)
    print(f"  Events detected: {len(events)}")

    vids_offsets = list(zip(mp4s, offsets))
    clips = []
    for i, t in enumerate(events):
        name = f"multicam_event{i+1:03d}_{int(t)}s.mp4"
        multicam_clip(vids_offsets, t, clips_dir / name)
        clips.append(make_meta(name, t, "multi", i+1))
        print(f"  ✓ {name}")

    return clips


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="FFA Kickabout Clip Generator")
    ap.add_argument("--input",  required=True, help="MP4 file (single) or folder (multi)")
    ap.add_argument("--output", default="output", help="Output folder (default: output/)")
    ap.add_argument("--multi",  action="store_true", help="Multi-cam mode")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    if args.multi:
        if not inp.is_dir():
            sys.exit("ERROR: --multi requires a folder as --input")
        clips = multi_mode(inp, out)
    else:
        if not inp.is_file():
            sys.exit("ERROR: --input must be a .mp4 file")
        clips = single_mode(inp, out)

    manifest = {
        "generated": datetime.now().isoformat(),
        "clips_dir": str((out / "clips").resolve()),
        "clips": clips
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n✓ {len(clips)} clips → {out / 'clips'}")
    print("✓ Open review.html in your browser to review clips.")

if __name__ == "__main__":
    main()
