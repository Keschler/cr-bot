#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
CHANNEL_URL = "https://www.youtube.com/@ryleycr1/videos"
OUT_ROOT = ROOT / "data" / "ryleycr1_pre_202512"
VIDEOS_DIR = OUT_ROOT / "videos"
MANIFEST_PATH = OUT_ROOT / "manifest.json"
PLAYLIST_CACHE = OUT_ROOT / "playlist_flat.json"
CUTOFF = "20251130"
DEFAULT_MAX_VIDEOS = 100
DEFAULT_FPS = 3


def detect_browser() -> str:
    for browser in ("firefox", "chromium"):
        if shutil.which(browser):
            return browser
    raise RuntimeError("Need Firefox or Chromium installed for --cookies-from-browser downloads.")


BROWSER = detect_browser()
YTDLP = shutil.which("yt-dlp") or "yt-dlp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-videos", type=int, default=DEFAULT_MAX_VIDEOS)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--keep-videos", action="store_true")
    parser.add_argument("--force", action="store_true", help="Redownload/reextract even if output already exists.")
    parser.add_argument(
        "--video-id",
        help="Download and extract exactly this YouTube video ID, skipping channel scan and date filtering.",
    )
    return parser.parse_args()


def run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=capture,
    )


def ensure_playlist() -> list[dict]:
    frames_dir = OUT_ROOT / f"frames_{DEFAULT_FPS}fps"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    if not PLAYLIST_CACHE.exists():
        result = run(
            [YTDLP, "--remote-components", "ejs:github", "--flat-playlist", "--dump-single-json", CHANNEL_URL],
            capture=True,
        )
        PLAYLIST_CACHE.write_text(result.stdout, encoding="utf-8")
    return json.loads(PLAYLIST_CACHE.read_text(encoding="utf-8"))["entries"]


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"selected": []}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def item_complete(item: dict) -> bool:
    frames_rel = item.get("frames")
    if not frames_rel:
        return False
    frame_dir = ROOT / frames_rel
    if not frame_dir.exists() or not (frame_dir / ".done").exists():
        return False
    if item.get("video") is not None and not (ROOT / item["video"]).exists():
        return False
    return True


def resolve_upload_date(video_id: str) -> str | None:
    result = subprocess.run(
        [
            YTDLP,
            "--remote-components",
            "ejs:github",
            "--extractor-args",
            "youtube:player_client=android",
            "--print",
            "%(upload_date)s",
            f"https://www.youtube.com/watch?v={video_id}",
        ],
        text=True,
        capture_output=True,
    )
    for line in result.stdout.splitlines()[::-1]:
        line = line.strip()
        if line.isdigit() and len(line) == 8:
            return line
    return None


def sanitize_id(video_id: str) -> str:
    return "".join(ch for ch in video_id if ch.isalnum() or ch in ("-", "_"))


def download_video(video_id: str) -> Path:
    out_tmpl = str(VIDEOS_DIR / f"{sanitize_id(video_id)}.%(ext)s")
    run(
        [
            YTDLP,
            "--remote-components",
            "ejs:github",
            "--cookies-from-browser",
            BROWSER,
            "--merge-output-format",
            "mp4",
            "-f",
            "298+140/136+140/300+140/135+140/22/18/best[height<=720]",
            "--remux-video",
            "mp4",
            "-o",
            out_tmpl,
            f"https://www.youtube.com/watch?v={video_id}",
        ]
    )
    matches = sorted(VIDEOS_DIR.glob(f"{sanitize_id(video_id)}.*"))
    if not matches:
        raise FileNotFoundError(f"Failed to locate downloaded file for {video_id}")
    return matches[0]


def extract_frames(video_path: Path, fps: int, keep_videos: bool, force: bool) -> Path:
    frames_dir = OUT_ROOT / f"frames_{fps}fps"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = frames_dir / video_path.stem
    frame_dir.mkdir(parents=True, exist_ok=True)
    sentinel = frame_dir / ".done"
    if sentinel.exists() and not force:
        return frame_dir
    if force:
        for jpg in frame_dir.glob("*.jpg"):
            jpg.unlink()
        sentinel.unlink(missing_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps}",
            str(frame_dir / "%06d.jpg"),
        ]
    )
    sentinel.write_text("ok\n", encoding="utf-8")
    if not keep_videos:
        video_path.unlink(missing_ok=True)
    return frame_dir


def main() -> None:
    args = parse_args()
    if args.video_id:
        frame_dir_existing = OUT_ROOT / f"frames_{args.fps}fps" / args.video_id
        if args.force and frame_dir_existing.exists():
            for jpg in frame_dir_existing.glob("*.jpg"):
                jpg.unlink()
            (frame_dir_existing / ".done").unlink(missing_ok=True)
        for existing in VIDEOS_DIR.glob(f"{sanitize_id(args.video_id)}.*"):
            if args.force:
                existing.unlink(missing_ok=True)
        video_path = download_video(args.video_id)
        frame_dir = extract_frames(video_path, fps=args.fps, keep_videos=args.keep_videos, force=args.force)
        print(f"saved direct video: {args.video_id} -> {frame_dir}")
        return

    entries = ensure_playlist()
    manifest = load_manifest()
    selected = manifest["selected"]
    selected[:] = [item for item in selected if item_complete(item)]
    save_manifest(manifest)
    processed_ids = {item["id"] for item in selected}

    for entry in entries:
        if len(selected) >= args.max_videos:
            break
        video_id = entry["id"]
        if video_id in processed_ids:
            continue
        upload_date = resolve_upload_date(video_id)
        if not upload_date or upload_date > CUTOFF:
            continue
        video_path = download_video(video_id)
        frame_dir = extract_frames(video_path, fps=args.fps, keep_videos=args.keep_videos, force=args.force)
        item = {
            "id": video_id,
            "title": entry.get("title"),
            "upload_date": upload_date,
            "video": str(video_path.relative_to(ROOT)) if args.keep_videos else None,
            "frames": str(frame_dir.relative_to(ROOT)),
        }
        selected.append(item)
        processed_ids.add(video_id)
        save_manifest(manifest)
        print(f"saved {len(selected):03d}/{args.max_videos}: {video_id} {upload_date}")

    print(f"done: {len(selected)} videos in {OUT_ROOT}")


if __name__ == "__main__":
    main()
