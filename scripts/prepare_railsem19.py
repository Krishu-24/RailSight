"""Resize RailSem19 to the working resolution once, and create a fixed train/val/test split.

Decoding 1920x1080 JPEGs every iteration is the main training bottleneck on a laptop, so
images are stored at 1280x720 (masks resized with nearest-neighbour to keep label ids).
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from railsight.seg import RS19_CLASSES, load_rs19_class_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/railsem19")
    ap.add_argument("--dst", default="data/processed/rs19")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    cfgs = list(src.rglob("rs19-config.json"))
    if cfgs:
        names = load_rs19_class_names(cfgs[0])
        assert names == RS19_CLASSES, f"class list differs from railsight.seg: {names}"
        print("class list verified against", cfgs[0])

    masks = {p.stem: p for p in src.rglob("*.png") if "uint8" in p.parts}
    images = {p.stem: p for p in src.rglob("*.jpg") if p.stem in masks}
    ids = sorted(images)
    print(f"found {len(ids)} image/mask pairs")
    assert ids, "no RailSem19 data found - expected jpgs/ and uint8/ folders under --src"

    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "masks").mkdir(parents=True, exist_ok=True)

    def convert(name):
        img = cv2.imread(str(images[name]))
        mask = cv2.imread(str(masks[name]), cv2.IMREAD_GRAYSCALE)
        h, w = mask.shape
        size = (args.width, round(h * args.width / w))
        cv2.imwrite(str(dst / "images" / f"{name}.jpg"),
                    cv2.resize(img, size, interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(dst / "masks" / f"{name}.png"),
                    cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST))

    with ThreadPoolExecutor(8) as ex:
        list(tqdm(ex.map(convert, ids), total=len(ids), desc="resizing"))

    perm = np.random.default_rng(args.seed).permutation(len(ids))
    n_val = n_test = len(ids) // 10
    splits = {
        "val": [ids[i] for i in perm[:n_val]],
        "test": [ids[i] for i in perm[n_val : n_val + n_test]],
        "train": [ids[i] for i in perm[n_val + n_test :]],
    }
    with open(dst / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f)
    print({k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()
