"""Build docs/demo.gif from the rendered demo videos (outputs/final, see scripts/run_video.py).

Picks one representative moment per clip from its per-frame log, letterboxes every frame to 16:9
and encodes with ffmpeg's two-pass palette (bundled with imageio-ffmpeg).

    python scripts/make_demo_gif.py
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import pandas as pd

W, H, BAR = 480, 270, 22


def first_run(d: pd.DataFrame, cond, n: int) -> int:
    """First frame of a run of n consecutive frames where cond holds."""
    c = cond(d).astype(int).values
    return next((i for i in range(len(c) - n) if c[i : i + n].all()), 0)


SEGMENTS = [  # clip, start-frame rule, seconds, caption
    ("india_people_crossing",
     lambda d: max(0, first_run(d, lambda d: (d.state == "CRITICAL") & (d.n_detections >= 3), 60) - 15), 4,
     "People on the track ~14 m ahead at 60 km/h -> CRITICAL (stop needs 172 m)"),
    ("level_crossing", lambda d: 40, 3, "Level crossing, track clear -> SAFE"),
    ("cab_ride_station_tunnel", lambda d: first_run(d, lambda d: (d.state == "SAFE") & d.corridor, 90), 3,
     "Station approach: ego-track corridor with range ticks"),
    ("fog_forest", lambda d: 60, 3, "Fog: track still found, nothing in path"),
]


def fit(im):
    s = min(W / im.shape[1], H / im.shape[0])
    im = cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
    py, px = H - im.shape[0], W - im.shape[1]
    return cv2.copyMakeBorder(im, py // 2, py - py // 2, px // 2, px - px // 2, cv2.BORDER_CONSTANT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs/final")
    ap.add_argument("--out", default="docs/demo.gif")
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()

    frames = []
    for name, pick, seconds, caption in SEGMENTS:
        d = pd.read_csv(Path(args.src) / f"{name}_log.csv")
        cap = cv2.VideoCapture(str(Path(args.src) / f"{name}_railsight.mp4"))
        vfps = cap.get(cv2.CAP_PROP_FPS)
        start = pick(d)
        for k in np.arange(start, min(len(d), start + int(seconds * vfps)), vfps / args.fps).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, im = cap.read()
            if not ok:
                break
            bar = np.full((BAR, W, 3), 25, np.uint8)
            cv2.putText(bar, caption, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1,
                        cv2.LINE_AA)
            frames.append(cv2.cvtColor(np.vstack([fit(im), bar]), cv2.COLOR_BGR2RGB))

    tmp = Path(tempfile.gettempdir()) / "railsight_demo_tmp.mp4"
    imageio.mimsave(tmp, frames, fps=args.fps, codec="libx264", quality=9, macro_block_size=2)
    vf = ("split[a][b];[a]palettegen=max_colors=160:stats_mode=diff[p];"
          "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(tmp),
                    "-vf", vf, "-loop", "0", args.out], check=True)
    os.remove(tmp)
    print(f"{len(frames)} frames, {os.path.getsize(args.out) / 2**20:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
