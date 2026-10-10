"""Run RailSight on video files from the command line.

    python scripts/run_video.py data/videos/clip.mp4 --speed 80
    python scripts/run_video.py data/videos/*.mp4 --show-seg --max-seconds 60
"""

import argparse
import glob
import json

from railsight.config import load_config
from railsight.pipeline import RailSightPipeline
from railsight.video import process_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--config", default=None)
    ap.add_argument("--speed", type=float, default=None, help="train speed in km/h")
    ap.add_argument("--hfov", type=float, default=None,
                    help="camera horizontal field of view in degrees (sets the ranging scale)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--show-seg", action="store_true", help="blend the segmentation map in")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    if args.hfov is not None:
        cfg["camera"]["hfov_deg"] = args.hfov
    pipe = RailSightPipeline(cfg, show_seg=args.show_seg)
    if args.speed is not None:
        pipe.speed_kmh = args.speed
    for pattern in args.videos:
        for v in glob.glob(pattern) or [pattern]:
            res = process_video(pipe, v, args.out, args.max_seconds)
            print(json.dumps({k: str(x) if not isinstance(x, (int, float, dict)) else x
                              for k, x in res.items()}, indent=2))


if __name__ == "__main__":
    main()
