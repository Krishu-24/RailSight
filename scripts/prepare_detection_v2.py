"""Detection dataset v2: the cleaned Roboflow set plus boxes derived from RailSem19's dense masks.

RailSem19 labels every human, car/truck and rail vehicle ("on-rails") pixel in 8,500 real
cab-view frames. Connected components of those classes become boxes, which:
  * adds in-domain (forward-facing train camera) examples of people and vehicles,
  * adds a new class, `rail_vehicle`: another train or tram on or beside the line,
  * gives a test set of real cab-view frames, split exactly as for segmentation (RailSem19
    test images stay unseen by every model).

The dense masks are partly weakly supervised and noisier than hand-drawn boxes. A visual check
showed vehicle boxes on billboards / watermarks and rail-vehicle boxes on catenary gantries, so
components are filtered: minimum size, minimum fill ratio (thin lattices fail it), no small
boxes on the image border, and rail vehicles must stand on track. Remaining caveats: fragments
of one object can become several boxes, and groups of people can merge into one.

RailSem19 frames must not leave this machine (licence), so this dataset is trained locally.
"""

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from railsight.seg import CLS

CLASSES = ["person", "animal", "vehicle", "rock", "vegetation", "obstacle", "rail_vehicle"]
RS19_GROUPS = {
    "person": [CLS["human"]],
    "vehicle": [CLS["car"], CLS["truck"]],
    "rail_vehicle": [CLS["on-rails"]],
}
MIN_SIDE_PX = 8      # at 1280 px image width
MIN_AREA_PX = 100
MIN_FILL = 0.35      # component pixels / box area: thin fragments (gantries, poles) fall below
BORDER_MAX_PX = 40   # small boxes touching the image border are watermarks / frame fragments
TRACK_BELOW = [CLS[c] for c in ("rail-track", "rail-raised", "rail-embedded", "tram-track", "trackbed")]


def link_or_copy(src: Path, dst: Path):
    try:
        os.link(src, dst)  # hard link: no extra disk space
    except OSError:
        shutil.copy2(src, dst)


def rs19_boxes(mask: np.ndarray, dropped: Counter | None = None) -> list[str]:
    """Boxes from connected components, with filters against the dense masks' typical noise."""
    h, w = mask.shape
    track = np.isin(mask, TRACK_BELOW)
    lines = []
    for name, ids in RS19_GROUPS.items():
        binary = cv2.dilate(np.isin(mask, ids).astype(np.uint8), np.ones((3, 3), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        for x, y, bw, bh, area in stats[1:]:
            reason = None
            if min(bw, bh) < MIN_SIDE_PX or area < MIN_AREA_PX:
                reason = "too small"
            elif area / (bw * bh) < MIN_FILL:
                reason = "thin fragment"
            elif max(bw, bh) < BORDER_MAX_PX and (x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1):
                reason = "border fragment"
            elif name == "rail_vehicle":
                below = track[y + bh : min(h, y + bh + max(4, bh // 2)), x : x + bw]
                if below.size and below.mean() < 0.05 and y + bh < h - 2:
                    reason = "not on track"
            if reason:
                if dropped is not None:
                    dropped[(name, reason)] += 1
                continue
            cls = CLASSES.index(name)
            lines.append(f"{cls} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roboflow", default="data/processed/det")
    ap.add_argument("--rs19", default="data/processed/rs19")
    ap.add_argument("--dst", default="data/processed/det_v2")
    args = ap.parse_args()

    rf, rs, dst = Path(args.roboflow), Path(args.rs19), Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)
    stats = defaultdict(Counter)

    # Roboflow part: same images and labels (class ids 0-5 are unchanged)
    for split in ("train", "val", "test"):
        out = "test_roboflow" if split == "test" else split
        (dst / out / "images").mkdir(parents=True, exist_ok=True)
        (dst / out / "labels").mkdir(parents=True, exist_ok=True)
        for img in (rf / split / "images").iterdir():
            link_or_copy(img, dst / out / "images" / img.name)
            lbl = rf / split / "labels" / f"{img.stem}.txt"
            text = lbl.read_text() if lbl.exists() else ""
            (dst / out / "labels" / f"{img.stem}.txt").write_text(text)
            stats[out]["_images"] += 1
            for ln in text.splitlines():
                if ln.strip():
                    stats[out][CLASSES[int(ln.split()[0])]] += 1

    # RailSem19 part, using the segmentation split
    splits = json.load(open(rs / "splits.json", encoding="utf-8"))
    dropped = Counter()
    for split, ids in splits.items():
        out = "test_rs19" if split == "test" else split
        (dst / out / "images").mkdir(parents=True, exist_ok=True)
        (dst / out / "labels").mkdir(parents=True, exist_ok=True)
        for name in ids:
            mask = cv2.imread(str(rs / "masks" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
            lines = rs19_boxes(mask, dropped)
            link_or_copy(rs / "images" / f"{name}.jpg", dst / out / "images" / f"rs19_{name}.jpg")
            (dst / out / "labels" / f"rs19_{name}.txt").write_text("\n".join(lines))
            stats[out]["_images"] += 1
            for ln in lines:
                stats[out][CLASSES[int(ln.split()[0])]] += 1

    common = {"path": str(dst.resolve()), "train": "train/images", "val": "val/images",
              "names": dict(enumerate(CLASSES))}
    for name, test in (("data.yaml", ["test_roboflow/images", "test_rs19/images"]),
                       ("data_test_roboflow.yaml", "test_roboflow/images"),
                       ("data_test_rs19.yaml", "test_rs19/images")):
        with open(dst / name, "w", encoding="utf-8") as f:
            yaml.safe_dump({**common, "test": test}, f, sort_keys=False)
    with open(dst / "stats.json", "w", encoding="utf-8") as f:
        json.dump({"splits": {k: dict(v) for k, v in stats.items()},
                   "dropped_rs19_boxes": {f"{a}: {b}": n for (a, b), n in dropped.items()}}, f, indent=2)
    print("dropped RailSem19 boxes:", {f"{a}: {b}": n for (a, b), n in sorted(dropped.items())})

    cols = ["_images"] + CLASSES
    print("split          " + " ".join(f"{c[:10]:>10}" for c in cols))
    for s in ("train", "val", "test_roboflow", "test_rs19"):
        print(f"{s:<14} " + " ".join(f"{stats[s][c]:>10}" for c in cols))


if __name__ == "__main__":
    main()
