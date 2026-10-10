"""Download the demo clips (free Pexels licence) into data/videos/ and normalise them.

Portrait clips are cropped to the band containing the track and horizon, and high frame rates
are reduced to ~30 fps, since the pipeline does not benefit from more.

    python scripts/fetch_demo_videos.py
"""

import argparse
import urllib.request
from pathlib import Path

import cv2
import imageio.v2 as imageio

# name: (Pexels page, direct file, crop as fractions (y0, y1) of the height or None, note)
CLIPS = {
    "level_crossing": (
        "https://www.pexels.com/video/37330572/",
        "https://videos.pexels.com/video-files/37330572/15811613_1920_1080_24fps.mp4",
        None, "low front camera, open country, level crossing"),
    "rural_track": (
        "https://www.pexels.com/video/9687456/",
        "https://videos.pexels.com/video-files/9687456/9687456-hd_1920_1080_30fps.mp4",
        None, "straight single track, clear conditions"),
    "fog_forest": (
        "https://www.pexels.com/video/8118594/",
        "https://videos.pexels.com/video-files/8118594/8118594-hd_1920_1080_30fps.mp4",
        None, "fog: reduced visibility"),
    "india_people_crossing": (
        "https://www.pexels.com/video/35333107/",
        "https://videos.pexels.com/video-files/35333107/14970845_1920_1080_60fps.mp4",
        None, "Indian Railways, people crossing; filmed from the door, so the visible track is a "
              "neighbouring line"),
    "cab_ride_station_tunnel": (
        "https://www.pexels.com/video/14281701/",
        "https://videos.pexels.com/video-files/14281701/14281701-uhd_1440_2560_60fps.mp4",
        (0.33, 1.0), "forward cab view through a station into a tunnel (portrait, cropped)"),
    "cab_interior_front": (
        "https://www.pexels.com/video/36777744/",
        "https://videos.pexels.com/video-files/36777744/15585499_1920_1080_30fps.mp4",
        None, "camera inside the cab: windscreen frame and dashboard in view"),
    "cab_interior_elevated": (
        "https://www.pexels.com/video/14309960/",
        "https://videos.pexels.com/video-files/14309960/14309960-hd_1920_1080_30fps.mp4",
        None, "camera inside the cab on an elevated line"),
}


def normalise(src: Path, dst: Path, crop, max_fps: float = 30.0, max_width: int = 1280):
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / max_fps))
    writer = imageio.get_writer(dst, fps=fps / step, codec="libx264", quality=6,
                                macro_block_size=8, ffmpeg_log_level="error")
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            if crop:
                h = frame.shape[0]
                y0, y1 = int(crop[0] * h), int(crop[1] * h)
                frame = frame[y0 : y0 + (y1 - y0) // 8 * 8]  # codec-friendly height
            if frame.shape[1] > max_width:
                s = max_width / frame.shape[1]
                frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/videos")
    ap.add_argument("--cache", default="data/raw/pexels")
    args = ap.parse_args()
    out, cache = Path(args.out), Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    for name, (page, url, crop, note) in CLIPS.items():
        raw = cache / Path(url).name
        if not raw.exists():
            print(f"downloading {name} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as r, open(raw, "wb") as f:
                f.write(r.read())
        dst = out / f"{name}.mp4"
        if not dst.exists():
            normalise(raw, dst, crop)
        print(f"{dst}  ({note})  source: {page}")


if __name__ == "__main__":
    main()
