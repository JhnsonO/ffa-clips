"""
Export approved clips from approved_manifest.json.

Single cam:
- tries stream copy first for near-instant export
- falls back to libx264 if the requested cut is not near a keyframe

Multi cam:
- exports approved clips using the existing stitched multi-cam logic
"""

import sys, json, argparse, subprocess, tempfile, shutil
from pathlib import Path

CFG = {
    "output_res": "1920x1080",
    "output_crf": 23,
    "switch_interval": 3.0,
}


def ff(args):
    r = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return r.returncode, r.stdout + r.stderr


def nearest_keyframe_delta(video, start_sec):
    code, out = ff([
        "ffprobe", "-v", "error",
        "-skip_frame", "nokey",
        "-select_streams", "v:0",
        "-show_frames",
        "-show_entries", "frame=pts_time",
        "-of", "csv=p=0",
        str(video),
    ])
    if code != 0:
        return None
    best = None
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pts = float(line.split(',')[0])
        except:
            continue
        delta = abs(pts - start_sec)
        if best is None or delta < best:
            best = delta
    return best


def export_copy(video, start, dur, out_path):
    return ff([
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(video),
        "-t", f"{dur:.3f}",
        "-c", "copy",
        str(out_path),
    ])


def export_reencode(video, start, dur, out_path):
    res = CFG["output_res"]
    crf = CFG["output_crf"]
    return ff([
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(video),
        "-t", f"{dur:.3f}",
        "-vf", f"scale={res}:force_original_aspect_ratio=decrease,pad={res}:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "128k",
        str(out_path),
    ])


def extract_clip(video, start, dur, out_path):
    delta = nearest_keyframe_delta(video, start)
    if delta is not None and delta <= 0.25:
        code, out = export_copy(video, start, dur, out_path)
        if code == 0 and Path(out_path).exists() and Path(out_path).stat().st_size > 0:
            return "copy"
    code, out = export_reencode(video, start, dur, out_path)
    if code != 0:
        raise RuntimeError(out)
    return "reencode"


def multicam_clip(vids_offsets, event_time, clip_pre, clip_post, out_path):
    clip_dur = clip_pre + clip_post
    sw = CFG["switch_interval"]
    tmpdir = Path(tempfile.mkdtemp())
    seg_files = []
    cam_idx = 0
    t = 0.0
    try:
        while t < clip_dur:
            seg_dur = min(sw, clip_dur - t)
            vid, offset = vids_offsets[cam_idx % len(vids_offsets)]
            local_start = max(0, (event_time - clip_pre + t) - offset)
            seg_out = tmpdir / f"seg{len(seg_files):03d}.mp4"
            code, out = export_reencode(vid, local_start, seg_dur, seg_out)
            if code != 0:
                raise RuntimeError(out)
            seg_files.append(seg_out)
            t += seg_dur
            cam_idx += 1

        concat = tmpdir / "list.txt"
        concat.write_text("\n".join(f"file '{p.as_posix()}'" for p in seg_files), encoding="utf-8")
        code, out = ff([
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat),
            "-c", "copy",
            str(out_path),
        ])
        if code != 0:
            raise RuntimeError(out)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="Export approved FFA clips")
    ap.add_argument("--manifest", required=True, help="Path to approved_manifest.json")
    ap.add_argument("--output", default="output", help="Output folder")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.output)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("source", {})
    clips = [c for c in manifest.get("clips", []) if c.get("approved") is True]

    if not clips:
        print("No approved clips found.")
        return

    mode = source.get("mode")
    print(f"Exporting {len(clips)} approved clips...")

    if mode == "single":
        video = source.get("source_video")
        if not video:
            sys.exit("ERROR: source_video missing from manifest")
        for clip in clips:
            out_path = clips_dir / clip["file"]
            method = extract_clip(video, float(clip["start_sec"]), float(clip["duration_sec"]), out_path)
            print(f"  ✓ {clip['file']} [{method}]")
    elif mode == "multi":
        videos = source.get("source_videos") or []
        offsets = source.get("offsets") or []
        if not videos or not offsets or len(videos) != len(offsets):
            sys.exit("ERROR: multi-cam source data missing from manifest")
        vids_offsets = list(zip(videos, offsets))
        for clip in clips:
            out_path = clips_dir / clip["file"]
            event_time = float(clip["event_time"])
            clip_pre = event_time - float(clip["start_sec"])
            clip_post = float(clip["duration_sec"]) - clip_pre
            multicam_clip(vids_offsets, event_time, clip_pre, clip_post, out_path)
            print(f"  ✓ {clip['file']} [multi]")
    else:
        sys.exit("ERROR: Unknown manifest mode")

    print(f"\nDone. Clips exported to {clips_dir}")


if __name__ == "__main__":
    main()
