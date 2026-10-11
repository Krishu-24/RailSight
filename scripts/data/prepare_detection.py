"""Merge the Roboflow railway-obstacle exports (YOLO format) into one clean, leak-free dataset.

Cleaning steps, each motivated by an audit of the raw exports:

1. Class mapping. Every source class is mapped onto six hazard categories. Classes that are
   not obstacles in the train's path are dropped:
   - "Plant" (goal set): weeds between sleepers seen by a top-down inspection camera, plus
     large boxes around trackside greenery. 74% of all boxes in that set.
   - "Tree" (research set): standing trees beside the line, not fallen ones.
   - "flood": a surface condition, not an object.
   Images whose labels were *all* dropped are removed so the top-down weed photos do not
   dominate the background distribution. Originally unlabelled images stay as negatives.

2. Duplicate grouping. Roboflow stores three augmented copies of each training image, the
   sources contain consecutive video frames, and the two datasets share some images.
   Images are grouped by source filename and by perceptual hash (dHash), and every group is
   assigned to exactly one split, so no near-duplicate can sit in both train and test.

3. Re-splitting. The original splits leak (see above), so groups are re-split 80/10/10 with a
   greedy stratification on the rarest class in each group. Val/test keep one image per
   source (no augmented copies); train keeps all copies.
"""

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

CLASSES = ["person", "animal", "vehicle", "rock", "vegetation", "obstacle"]

NAME_MAP = {
    "person": "person", "people": "person", "human": "person",
    "animal": "animal", "animals": "animal", "bird": "animal",
    "vehicle": "vehicle", "car": "vehicle", "truck": "vehicle",
    "rock": "rock", "stone": "rock", "rolling_stones": "rock",
    "branches": "vegetation", "fallen trees": "vegetation",
    "trash": "obstacle", "other_objects": "obstacle",
    "plant": None, "tree": None, "flood": None,
}  # fmt: skip

SPLIT_DIRS = {"train", "valid", "val", "test"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HASH_DIST = 3  # max Hamming distance (of 64 bits) for two images to count as duplicates
TARGET = {"train": 0.8, "val": 0.1, "test": 0.1}


def source_stem(p: Path) -> str:
    """'abc_jpg.rf.<hash>.jpg' -> 'abc_jpg' (shared by all augmented copies)."""
    return re.sub(r"\.rf\.[0-9a-f]+$", "", p.stem)


def dhash(path: Path) -> int:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def load_source(src: Path):
    """Yield (image_path, mapped_label_lines, had_labels) for one Roboflow export."""
    yml = sorted(src.rglob("data.yaml"))[0]
    with open(yml, encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]
    names = list(names.values()) if isinstance(names, dict) else names
    remap = {}
    for i, n in enumerate(names):
        key = str(n).strip().lower()
        if key not in NAME_MAP:
            raise KeyError(f"{src.name}: class '{n}' has no mapping - add it to NAME_MAP")
        if NAME_MAP[key] is not None:
            remap[i] = CLASSES.index(NAME_MAP[key])
    print(f"{src.name}: {names}\n  -> keeps {sorted({CLASSES[v] for v in remap.values()})}")

    for split_dir in sorted(yml.parent.iterdir()):
        if split_dir.name not in SPLIT_DIRS or not (split_dir / "images").exists():
            continue
        for img in sorted((split_dir / "images").iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            lbl = split_dir / "labels" / f"{img.stem}.txt"
            raw = [ln.split() for ln in lbl.read_text().splitlines() if ln.strip()] if lbl.exists() else []
            lines = [" ".join([str(remap[int(p[0])])] + p[1:5])
                     for p in raw if len(p) == 5 and int(p[0]) in remap]
            yield img, lines, bool(raw)


def group_duplicates(items) -> list[list[int]]:
    uf = UnionFind(len(items))
    by_stem = {}
    for i, it in enumerate(items):
        key = (it["src"], source_stem(it["img"]))
        if key in by_stem:
            uf.union(i, by_stem[key])
        by_stem[key] = i
    hashes = np.array([it["hash"] for it in items], np.uint64)
    for i in range(len(items)):
        x = np.bitwise_xor(hashes[i + 1 :], hashes[i])
        d = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
        for j in np.flatnonzero(d <= HASH_DIST):
            uf.union(i, i + 1 + int(j))
    groups = defaultdict(list)
    for i in range(len(items)):
        groups[uf.find(i)].append(i)
    return list(groups.values())


def assign_splits(items, groups, seed):
    """Greedy stratification: rarest-class groups first, each to the split furthest below
    its target share for that class."""
    rng = np.random.default_rng(seed)
    g_counts = []
    for g in groups:
        c = Counter()
        for i in _one_per_source(items, g):
            c.update(int(ln.split()[0]) for ln in items[i]["lines"])
        g_counts.append(c)
    totals = sum(g_counts, Counter())
    n_img = sum(len(_one_per_source(items, g)) for g in groups)

    def rarity(k):
        c = g_counts[k]
        return min((totals[x] for x in c), default=10**9)

    order = sorted(rng.permutation(len(groups)), key=rarity)
    cur = {s: Counter() for s in TARGET}
    cur_img = Counter()
    out = {}
    for k in order:
        c = g_counts[k]
        if c:
            rare = min(c, key=lambda x: totals[x])
            need = {s: (TARGET[s] * totals[rare] - cur[s][rare]) / (TARGET[s] * totals[rare])
                    for s in TARGET}
        else:
            need = {s: (TARGET[s] * n_img - cur_img[s]) / (TARGET[s] * n_img) for s in TARGET}
        s = max(need, key=need.get)
        out[k] = s
        cur[s].update(c)
        cur_img[s] += len(_one_per_source(items, groups[k]))
    return out


def _one_per_source(items, g):
    seen, keep = set(), []
    for i in g:
        key = (items[i]["src"], source_stem(items[i]["img"]))
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+",
                    default=["data/raw/roboflow_goal", "data/raw/roboflow_research"])
    ap.add_argument("--dst", default="data/processed/det")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    items, dropped = [], Counter()
    for src in map(Path, args.sources):
        for img, lines, had in load_source(src):
            if had and not lines:
                dropped["only dropped classes"] += 1
                continue
            items.append({"src": src.name, "img": img, "lines": lines})

    print(f"hashing {len(items)} images ...")
    for it in items:
        it["hash"] = dhash(it["img"])
    groups = group_duplicates(items)
    sizes = sorted((len(_one_per_source(items, g)) for g in groups), reverse=True)
    print(f"{len(groups)} duplicate groups; largest (unique sources): {sizes[:5]}")

    split_of = assign_splits(items, groups, args.seed)

    dst = Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)
    stats, manifest, n = defaultdict(Counter), [], 0
    for k, g in enumerate(groups):
        s = split_of[k]
        keep = g if s == "train" else _one_per_source(items, g)
        dropped[f"augmented copies not used in {s}"] += len(g) - len(keep) if s != "train" else 0
        for i in keep:
            it = items[i]
            (dst / s / "images").mkdir(parents=True, exist_ok=True)
            (dst / s / "labels").mkdir(parents=True, exist_ok=True)
            # short names: Roboflow filenames can exceed the Windows 260-char path limit
            stem = f"{it['src'].removeprefix('roboflow_')}_{n:06d}"
            n += 1
            shutil.copy2(it["img"], dst / s / "images" / f"{stem}{it['img'].suffix}")
            (dst / s / "labels" / f"{stem}.txt").write_text("\n".join(it["lines"]))
            manifest.append(f"{s},{stem},{k},{it['img'].as_posix()}")
            stats[s]["_images"] += 1
            for ln in it["lines"]:
                stats[s][CLASSES[int(ln.split()[0])]] += 1

    with open(dst / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"path": str(dst.resolve()), "train": "train/images",
                        "val": "val/images", "test": "test/images",
                        "names": dict(enumerate(CLASSES))}, f, sort_keys=False)
    (dst / "manifest.csv").write_text("split,name,group,source\n" + "\n".join(manifest),
                                      encoding="utf-8")
    dropped = {k: v for k, v in dropped.items() if v}
    with open(dst / "stats.json", "w", encoding="utf-8") as f:
        json.dump({"splits": {k: dict(v) for k, v in stats.items()}, "dropped": dropped,
                   "groups": len(groups)}, f, indent=2)

    print("\ndropped:", dropped)
    cols = ["_images"] + CLASSES
    print("\nsplit   " + " ".join(f"{c:>10}" for c in cols))
    for s in TARGET:
        print(f"{s:<7} " + " ".join(f"{stats[s][c]:>10}" for c in cols))


if __name__ == "__main__":
    main()
