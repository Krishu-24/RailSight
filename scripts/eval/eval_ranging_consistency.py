"""Check rail-gauge ranging against an independent cue: human height.

For a person standing near the track at image row y,
    gauge distance   Z_g = f * G / track_width_px(y)
    height distance  Z_h = f * H / person_height_px      (H ~ 1.7 m)
The focal length f cancels in the ratio Z_g / Z_h = G * h_px / (H * track_width_px), so the
check needs no camera calibration. A median ratio near 1 means gauge ranging has no systematic
scale bias; the spread bounds its random error (person height varies by roughly +-5-8 %,
so part of the spread comes from people, not from ranging).

Uses RailSem19 ground-truth masks only (no model), so it isolates the geometry.

    python scripts/eval/eval_ranging_consistency.py
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from railsight.config import load_config
from railsight.corridor import Corridor, focal_from_hfov, select_ego_track
from railsight.seg import CLS, TRACK_IDS

PERSON_H = 1.7
# surfaces level with the rails; people on platforms ("construction") stand 0.5-0.8 m higher
TRACK_LEVEL = TRACK_IDS + [CLS[c] for c in ("trackbed", "terrain", "road", "sidewalk", "vegetation")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/rs19")
    ap.add_argument("--max-lateral", type=float, default=6.0, help="metres from the track centre")
    ap.add_argument("--out", default="reports/ranging")
    args = ap.parse_args()

    cfg = load_config()
    gauge = cfg["track"]["gauge_m"] + cfg["track"]["rail_head_m"]
    root = Path(args.data)
    ids = sorted(p.stem for p in (root / "masks").glob("*.png"))
    rows = []
    for name in tqdm(ids, desc="frames"):
        m = cv2.imread(str(root / "masks" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        h, w = m.shape
        human = (m == CLS["human"]).astype(np.uint8)
        if human.sum() < 200:
            continue
        cor = Corridor.fit(select_ego_track(np.isin(m, TRACK_IDS)), gauge, focal_from_hfov(w, 60))
        if cor is None:
            continue
        n, _, stats, _ = cv2.connectedComponentsWithStats(human, 8)
        for x, y, bw, bh, area in stats[1:]:
            y2 = y + bh
            standing = bh >= 24 and bh / bw >= 1.8 and area / (bw * bh) >= 0.3  # single upright person
            inside = x > 1 and y > 1 and x + bw < w - 1 and y2 < h - 1             # fully visible
            if not (standing and inside) or not (cor.y_top <= y2 <= cor.bottom_y):
                continue
            feet = m[y2 : min(h, y2 + 6), x : x + bw]
            on_track_level = np.isin(feet, TRACK_LEVEL).mean() >= 0.6
            lateral = float(cor.lateral_m(x + bw / 2, y2))
            if abs(lateral) > args.max_lateral:
                continue  # far from the track: the flat-ground assumption is weaker
            ratio = gauge * bh / (PERSON_H * float(cor.width_px(y2)))
            rows.append({"frame": name, "ratio": ratio, "lateral_m": lateral,
                         "track_level": bool(on_track_level),
                         "z_gauge_m": float(cor.distance_m(y2)), "height_px": int(bh)})

    def stats(sel):
        r = np.array([x["ratio"] for x in sel])
        log_r = np.log(r)
        return {"people": len(sel), "median_ratio_gauge_over_height": float(np.median(r)),
                "iqr_ratio": [float(np.percentile(r, 25)), float(np.percentile(r, 75))],
                "robust_spread_pct": float(100 * 1.4826 * np.median(np.abs(log_r - np.median(log_r))))}

    level = [x for x in rows if x["track_level"]]
    other = [x for x in rows if not x["track_level"]]
    summary = {"all_people": stats(rows), "feet_at_track_level": stats(level),
               "elsewhere (platforms etc.)": stats(other),
               "assumed_person_height_m": PERSON_H}
    r = np.array([x["ratio"] for x in level])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "consistency.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "people": rows}, f, indent=1)
    print(json.dumps(summary, indent=2))

    z = np.array([x["z_gauge_m"] for x in level])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.hist(r, bins=40, range=(0.3, 2.0), color="#4c72b0")
    a1.axvline(1, color="k", lw=1)
    a1.axvline(np.median(r), color="#e6194b", lw=1.5, label=f"median {np.median(r):.2f}")
    a1.set(xlabel="gauge distance / height distance", ylabel="people",
           title="Ranging consistency (feet at track level)")
    a1.legend()
    a2.scatter(z, r, s=6, alpha=0.5)
    a2.axhline(1, color="k", lw=1)
    a2.set(xscale="log", ylim=(0.3, 2.0), xlabel="gauge distance (m, 60° FOV assumed)",
           ylabel="ratio", title="Ratio vs distance")
    for ax in (a1, a2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "consistency.png", dpi=140)


if __name__ == "__main__":
    main()
