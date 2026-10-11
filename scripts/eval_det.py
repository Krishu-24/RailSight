"""Compare detectors on a held-out test split with one shared evaluator.

Rows:
  * COCO-pretrained YOLO11s, zero-shot: its COCO classes are mapped onto ours by name
    (person; animals; road vehicles; "train" -> rail_vehicle). It has no notion of
    rock / vegetation / obstacle.
  * Every fine-tuned run found under runs/det/*/weights/best.pt.

Model classes are matched to the dataset's classes by name, so models trained on the 6-class
dataset (v1) can be scored on the 7-class dataset (v2) for the classes they know. mAP50-95 is
computed by the same evaluator for every model; Ultralytics' own validator is added as a
cross-check when a model's classes match the dataset exactly.

    python scripts/eval_det.py                                                  # v1 test set
    python scripts/eval_det.py --data data/processed/det_v2/data_test_rs19.yaml  # RailSem19 frames
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import RTDETR, YOLO

from railsight.metrics import evaluate_detections

SHARED = ("person", "animal", "vehicle")  # classes a COCO model can know about
COCO_TO_OURS = {"person": "person", "bicycle": "vehicle", "car": "vehicle", "motorcycle": "vehicle",
                "bus": "vehicle", "truck": "vehicle", "train": "rail_vehicle",
                **{a: "animal" for a in ("bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
                                         "bear", "zebra", "giraffe")}}

# COCO size convention on sqrt(box area), in original image pixels (Roboflow 640x640, RailSem19 1280x720)
SIZE_BINS = {"small(<32px)": (0, 32), "medium(32-96px)": (32, 96), "large(>96px)": (96, 1e9)}
IOU_THRS = np.arange(0.5, 0.96, 0.05)


def load_dataset(data: str, split: str):
    p = Path(data)
    yml = p if p.suffix in (".yaml", ".yml") else p / "data.yaml"
    cfg = yaml.safe_load(open(yml, encoding="utf-8"))
    names = cfg["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    root = Path(cfg.get("path", yml.parent))
    dirs = cfg[split] if isinstance(cfg[split], list) else [cfg[split]]
    paths = sorted(q for d in dirs for q in (root / d).iterdir()
                   if q.suffix.lower() in {".jpg", ".png", ".jpeg"})
    return yml, names, paths


def load_gt(paths):
    gts = []
    for p in paths:
        h, w = cv2.imread(str(p)).shape[:2]
        lbl = p.parent.parent / "labels" / f"{p.stem}.txt"
        rows = [list(map(float, ln.split())) for ln in lbl.read_text().splitlines() if ln.strip()]
        a = np.array(rows).reshape(-1, 5)
        cx, cy, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        gts.append((boxes, a[:, 0].astype(int)))
    return gts


def class_map(model_names: dict, dataset: list[str], rename: dict | None = None) -> dict[int, int]:
    """Model class id -> dataset class id, matched by (optionally renamed) class name."""
    out = {}
    for i, n in model_names.items():
        n = rename.get(n) if rename else n
        if n in dataset:
            out[int(i)] = dataset.index(n)
    return out


def predict(model, paths, cmap, imgsz=640, device=None):
    preds = []
    results = (r for i in range(0, len(paths), 8)  # chunks: a whole list is sent as one batch
               for r in model.predict([str(p) for p in paths[i : i + 8]], conf=0.001, iou=0.6,
                                      imgsz=imgsz, verbose=False, device=device))
    for r in results:
        b = r.boxes
        boxes, cls, conf = b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy()
        keep = np.array([c in cmap for c in cls], bool)
        preds.append((boxes[keep], np.array([cmap[c] for c in cls[keep]], int), conf[keep]))
    return preds


def summarize(name, names, gts, preds, cmap):
    res = evaluate_detections(gts, preds, len(names), size_bins=SIZE_BINS)
    pc = res["per_class"]  # only classes that have ground truth
    known = set(cmap.values())
    row = {"model": name,
           "mAP50_all": float(np.mean([pc[c]["AP"] if c in known else 0.0 for c in pc])),
           "mAP50_known_classes": float(np.mean([pc[c]["AP"] for c in pc if c in known])),
           "mAP50_shared": float(np.mean([pc[c]["AP"] for c in pc if names[c] in SHARED and c in known]
                                         or [np.nan]))}
    per_thr = [evaluate_detections(gts, preds, len(names), iou_thr=t)["per_class"] for t in IOU_THRS]
    row["mAP50-95"] = float(np.mean([np.mean([p[c]["AP"] if c in known else 0.0 for c in p])
                                     for p in per_thr]))
    row.update({f"AP50_{names[c]}": pc[c]["AP"] for c in pc if c in known})
    row.update({f"recall_{k}": v for k, v in res["recall_by_size"].items()})
    row["gt_boxes"] = {names[c]: pc[c]["n_gt"] for c in pc}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/det", help="dataset folder or data*.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="reports/detection")
    ap.add_argument("--tag", default=None, help="output file suffix (default: split, + yaml name)")
    ap.add_argument("--device", default=None, help="e.g. cpu while the GPU is training")
    ap.add_argument("--models", nargs="*", default=None, help="run names to evaluate (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="first N images only (smoke tests)")
    ap.add_argument("--imgsz", type=int, default=640, help="inference size for every model")
    args = ap.parse_args()

    yml, names, paths = load_dataset(args.data, args.split)
    paths = paths[: args.limit]
    tag = args.tag or (args.split if yml.name == "data.yaml" and "det_v2" not in str(yml)
                       else f"{args.split}_{yml.parent.name}_{yml.stem}")
    if args.imgsz != 640:
        tag += f"_{args.imgsz}px"
    gts = load_gt(paths)
    print(f"{len(paths)} {args.split} images, classes {names}")
    rows = []

    coco = YOLO("yolo11s.pt")
    cmap = class_map(coco.names, names, COCO_TO_OURS)
    rows.append(summarize("yolo11s COCO zero-shot", names, gts,
                          predict(coco, paths, cmap, imgsz=args.imgsz, device=args.device), cmap))

    for w in sorted(Path("runs/det").glob("*/weights/best.pt")):
        name = w.parent.parent.name
        if args.models is not None and name not in args.models:
            continue
        model = RTDETR(str(w)) if "rtdetr" in name else YOLO(str(w))
        cmap = class_map(model.names, names)
        row = summarize(name, names, gts, predict(model, paths, cmap, imgsz=args.imgsz, device=args.device),
                        cmap)
        if [model.names[i] for i in sorted(model.names)] == names:  # cross-check with Ultralytics
            m = model.val(data=str(yml), split=args.split, imgsz=args.imgsz, verbose=False, plots=True,
                          device=args.device,
                          project=str(Path("runs/det_eval").resolve()), name=f"{name}_{tag}", exist_ok=True)
            row.update({"mAP50-95 (ultralytics)": float(m.box.map), "mAP50 (ultralytics)": float(m.box.map50),
                        "ms/img": m.speed.get("inference", float("nan"))})
        rows.append(row)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"results_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=float)

    cols = ["mAP50_all", "mAP50_known_classes", "mAP50_shared", "mAP50-95",
            *[f"recall_{k}" for k in SIZE_BINS]]
    lines = ["| model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for r in rows:
        lines.append(f"| {r['model']} | " + " | ".join(f"{r[c]:.3f}" if r.get(c) is not None else "–"
                                                      for c in cols) + " |")
    (out / f"results_{tag}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
