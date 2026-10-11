"""Inference speed on this GPU: every trained segmentation model and detector, and the full
pipeline. Run only when no training is using the GPU.

    python scripts/eval/benchmark.py
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import torch

from railsight.config import load_config
from railsight.pipeline import RailSightPipeline
from railsight.seg import Segmenter


def timed(fn, frames, warmup=10):
    for f in frames[:warmup]:
        fn(f)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for f in frames[warmup:]:
        fn(f)
    torch.cuda.synchronize()
    return (len(frames) - warmup) / (time.perf_counter() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/videos/rural_track.mp4")
    ap.add_argument("--frames", type=int, default=110)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--out", default="reports/benchmark")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "benchmark needs the GPU"

    cap = cv2.VideoCapture(args.video)
    frames = []
    while len(frames) < args.frames:
        ok, f = cap.read()
        if not ok:
            break
        h = round(f.shape[0] * args.width / f.shape[1])
        frames.append(cv2.resize(f, (args.width, h)))
    print(f"{len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}")

    rows = []
    for w in sorted(Path("runs/seg").glob("*/best")):
        seg = Segmenter(w, "cuda", "fp32" if w.parent.name == "segformer-b0" else "bf16")
        fps = timed(seg, frames)
        rows.append({"stage": "segmentation", "model": w.parent.name, "fps": fps,
                     "peak_mem_GB": torch.cuda.max_memory_allocated() / 2**30})
        del seg
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    from ultralytics import RTDETR, YOLO

    for w in sorted(Path("runs/det").glob("*/weights/best.pt")):
        name = w.parent.parent.name
        det = (RTDETR if "rtdetr" in name else YOLO)(str(w))
        imgsz = 960 if name.endswith("_960") else 640  # each detector at the size it was trained for
        fps = timed(lambda f, d=det, s=imgsz: d.predict(f, imgsz=s, quantize="fp16", verbose=False, device=0),
                    frames)
        rows.append({"stage": "detection", "model": f"{name} @{imgsz}px", "fps": fps})
        del det
        torch.cuda.empty_cache()

    cfg = load_config()
    pipe = RailSightPipeline(cfg)
    for n in (1, cfg["track"]["seg_every_n"]):
        pipe.cfg["track"]["seg_every_n"] = n
        pipe.reset()
        t0 = [0.0]

        def step(f, t0=t0):
            pipe.process(f, t0[0])
            t0[0] += 1 / 30

        rows.append({"stage": "full pipeline", "model": f"default config, segmentation every {n} frame(s)",
                     "fps": timed(step, frames)})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = ["| stage | model | FPS |", "|---|---|---|"]
    lines += [f"| {r['stage']} | {r['model']} | {r['fps']:.1f} |" for r in rows]
    lines.append(f"\nGPU: {torch.cuda.get_device_name(0)}; frames {frames[0].shape[1]}x{frames[0].shape[0]}; "
                 f"mean of {len(frames) - 10} frames after 10 warm-up frames.")
    (out / "benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
