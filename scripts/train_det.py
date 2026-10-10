"""Fine-tune YOLO11 on the merged railway-obstacle dataset.

    python scripts/train_det.py --model yolo11s.pt --name yolo11s_rail
    python scripts/train_det.py --model yolo11n.pt --name yolo11n_rail --epochs 60
"""

import argparse
from pathlib import Path

from ultralytics import RTDETR, YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--name", default="yolo11s_rail")
    ap.add_argument("--data", default="data/processed/det/data.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=float, default=16,
                    help="batch size, or a fraction <1 of GPU memory for Ultralytics AutoBatch")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    model = (RTDETR if "rtdetr" in args.model else YOLO)(args.model)  # COCO-pretrained start
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=int(args.batch) if args.batch >= 1 else args.batch,
        workers=args.workers,
        project=str(Path("runs/det").resolve()),  # relative paths get nested under runs/detect
        name=args.name,
        exist_ok=True,
        patience=30,
        cos_lr=True,
        close_mosaic=10,
        seed=0,
        plots=True,
    )


if __name__ == "__main__":
    main()
