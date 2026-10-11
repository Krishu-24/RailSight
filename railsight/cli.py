"""Command-line entry point.

    railsight video data/videos/clip.mp4 --speed 80          # annotated video + log + timeline
    railsight video "data/videos/*.mp4" --show-seg --max-seconds 60
    railsight demo                                            # interactive Gradio app

Both read configs/default.yaml; flags override the per-camera and per-run settings.
"""

import argparse
import glob
import json
import sys

from .config import ROOT, load_config


def _video(args):
    from .pipeline import RailSightPipeline
    from .video import process_video

    cfg = load_config(args.config) if args.config else load_config()
    if args.hfov is not None:
        cfg["camera"]["hfov_deg"] = args.hfov
    if args.ignore_below is not None:
        cfg["camera"]["ignore_below"] = args.ignore_below
    pipe = RailSightPipeline(cfg, show_seg=args.show_seg)
    if args.speed is not None:
        pipe.speed_kmh = args.speed
    for pattern in args.videos:
        for v in glob.glob(pattern) or [pattern]:
            res = process_video(pipe, v, args.out, args.max_seconds)
            print(json.dumps({k: str(x) if not isinstance(x, (int, float, dict)) else x
                              for k, x in res.items()}, indent=2))


def _demo(args):
    import runpy

    sys.argv = [str(ROOT / "app" / "gradio_app.py")]
    runpy.run_path(sys.argv[0], run_name="__main__")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="railsight", description="Railway hazard detection and collision risk.")
    sub = ap.add_subparsers(dest="command", required=True)

    v = sub.add_parser("video", help="run the pipeline on video files")
    v.add_argument("videos", nargs="+", help="files or glob patterns")
    v.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    v.add_argument("--speed", type=float, default=None, help="train speed in km/h")
    v.add_argument("--hfov", type=float, default=None,
                   help="camera horizontal field of view in degrees (sets the ranging scale)")
    v.add_argument("--ignore-below", type=float, default=None,
                   help="fraction of image height showing the train itself (dashboard, hood)")
    v.add_argument("--out", default="outputs", help="output folder")
    v.add_argument("--max-seconds", type=float, default=None, help="process only the first N seconds")
    v.add_argument("--show-seg", action="store_true", help="blend the segmentation map in")
    v.set_defaults(func=_video)

    d = sub.add_parser("demo", help="launch the interactive Gradio app")
    d.set_defaults(func=_demo)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
