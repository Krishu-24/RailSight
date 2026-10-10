"""Compare detectors on the held-out test split with one shared evaluator.

Rows:
  * COCO-pretrained YOLO11s, zero-shot: its COCO classes are mapped onto ours
    (person; animals; road vehicles). It has no notion of rock / vegetation / obstacle.
  * Every fine-tuned run found under runs/det/*/weights/best.pt.

    python scripts/eval_det.py
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import RTDETR, YOLO

from railsight.metrics import evaluate_detections

CLASSES = ["person", "animal", "vehicle", "rock", "vegetation", "obstacle"]
SHARED = [0, 1, 2]  # classes a COCO model can know about

COCO_TO_OURS = {0: 0, 1: 2, 2: 2, 3: 2, 5: 2, 7: 2}  # person, bicycle, car, motorcycle, bus, truck
COCO_TO_OURS.update({k: 1 for k in range(14, 24)})  # bird ... giraffe

# COCO size convention on sqrt(box area) in pixels at the 640 px evaluation size
SIZE_BINS = {"small(<32px)": (0, 32), "medium(32-96px)": (32, 96), "large(>96px)": (96, 1e9)}


def load_gt(img_dir: Path):
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"})
    gts = []
    for p in paths:
        h, w = cv2.imread(str(p)).shape[:2]
        lbl = p.parent.parent / "labels" / f"{p.stem}.txt"
        rows = [list(map(float, ln.split())) for ln in lbl.read_text().splitlines() if ln.strip()]
        a = np.array(rows).reshape(-1, 5)
        cx, cy, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        gts.append((boxes, a[:, 0].astype(int)))
    return paths, gts


def predict(model, paths, class_map=None, imgsz=640, device=None):
    preds = []
    results = (r for i in range(0, len(paths), 8)  # chunks: a whole list is sent as one batch
               for r in model.predict([str(p) for p in paths[i : i + 8]], conf=0.001, iou=0.6,
                                      imgsz=imgsz, verbose=False, device=device))
    for r in results:
        b = r.boxes
        boxes, cls, conf = b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy()
        if class_map is not None:
            keep = np.array([c in class_map for c in cls], bool)
            boxes, conf = boxes[keep], conf[keep]
            cls = np.array([class_map[c] for c in cls[keep]], int)
        preds.append((boxes, cls, conf))
    return preds


def summarize(name, res, ultra=None):
    pc = res["per_class"]
    row = {"model": name,
           "mAP50_all": np.mean([pc[c]["AP"] if c in pc else 0.0 for c in range(len(CLASSES))]),
           "mAP50_shared": np.mean([pc[c]["AP"] for c in SHARED if c in pc])}
    row.update({f"AP50_{CLASSES[c]}": pc[c]["AP"] for c in pc})
    row.update({f"recall_{k}": v for k, v in res["recall_by_size"].items()})
    if ultra:
        row.update(ultra)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/det")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="reports/detection")
    ap.add_argument("--device", default=None, help="e.g. cpu while the GPU is training")
    args = ap.parse_args()

    paths, gts = load_gt(Path(args.data) / args.split / "images")
    print(f"{len(paths)} {args.split} images")
    rows = []

    coco = YOLO("yolo11s.pt")
    res = evaluate_detections(gts, predict(coco, paths, COCO_TO_OURS, device=args.device), len(CLASSES),
                              size_bins=SIZE_BINS)
    rows.append(summarize("yolo11s COCO zero-shot", res))

    for w in sorted(Path("runs/det").glob("*/weights/best.pt")):
        name = w.parent.parent.name
        model = RTDETR(str(w)) if "rtdetr" in name else YOLO(str(w))
        preds = predict(model, paths, device=args.device)
        res = evaluate_detections(gts, preds, len(CLASSES), size_bins=SIZE_BINS)
        # Ultralytics' own validator for the standard mAP50-95 figure
        m = model.val(data=str(Path(args.data) / "data.yaml"), split=args.split, verbose=False,
                      plots=True, device=args.device, project=str(Path("runs/det_eval").resolve()), name=name,
                      exist_ok=True)
        speed = m.speed.get("inference", float("nan"))
        rows.append(summarize(name, res, {"mAP50-95 (ultralytics)": float(m.box.map),
                                          "mAP50 (ultralytics)": float(m.box.map50),
                                          "ms/img": speed}))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"results_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=float)

    cols = sorted({k for r in rows for k in r if k != "model"}, key=lambda k: (k[0] != "m", k))
    lines = ["| model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for r in rows:
        cells = [f"{r[c]:.3f}" if isinstance(r.get(c), float) else str(r.get(c, "-")) for c in cols]
        lines.append(f"| {r['model']} | " + " | ".join(cells) + " |")
    (out / f"results_{args.split}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
