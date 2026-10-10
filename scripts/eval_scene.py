"""Scene-level evaluation on RailSem19: danger-corridor accuracy and the foreign-object check.

Corridor: the corridor fitted to the *predicted* segmentation is compared with the one fitted
to the *ground-truth* segmentation: how often it is found, mask IoU, horizon error, and how far
the predicted gauge ranging deviates from ground-truth geometry at fixed distances. This isolates
the error that segmentation adds to ranging (absolute depth needs lidar, e.g. OSDaR23).

Foreign objects:

Ground truth per frame: the corridor is fitted to the *ground-truth* ego track, and any
ground-truth pixels inside the inner track zone that are not normal track surface
(e.g. human, car, on-rails, pole, construction) form obstacle blobs.

Prediction: the same procedure on the *predicted* segmentation, optionally also treating
low-confidence pixels as suspicious. Each variant is scored with
  * frame level: does the frame contain an obstacle? (precision / recall / false alarms)
  * object level: fraction of ground-truth obstacle blobs overlapped by a predicted blob.

Segmentation outputs are cached, so variants can be re-scored without re-running the model.

    python scripts/eval_scene.py --weights runs/seg/segformer-b2/best --split test
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from railsight.anomaly import find_foreign_blobs
from railsight.config import load_config
from railsight.corridor import Corridor, focal_from_hfov, select_ego_track
from railsight.seg import GROUND_IDS, IGNORE_INDEX, TRACK_IDS, Segmenter

RANGE_CHECK_M = (25, 50, 100)

VARIANTS = {  # name -> confidence threshold (pixels below it count as suspicious)
    "class only": 0.0,
    "class + conf<0.3": 0.3,
    "class + conf<0.5": 0.5,
    "class + conf<0.7": 0.7,
}


def fit_corridor(labels, cfg):
    t = cfg["track"]
    ego = select_ego_track(np.isin(labels, TRACK_IDS))
    f = focal_from_hfov(labels.shape[1], cfg["camera"]["hfov_deg"])
    return Corridor.fit(ego, t["gauge_m"] + t["rail_head_m"], f)


def gt_blobs(gt, cor, a, max_range):
    """Ground-truth obstacle blobs: non-ground, non-ignore pixels in the inner zone."""
    lab = gt.copy()
    lab[lab == IGNORE_INDEX] = GROUND_IDS[0]  # unlabeled pixels are not evidence of an obstacle
    conf = np.ones(gt.shape, np.float32)
    return find_foreign_blobs(lab, conf, cor, a["half_width_m"], 0.0, a["min_size_m"],
                              a["min_area_px"], max_range)


def overlaps(b1, b2):
    return not (b1[2] < b2[0] or b2[2] < b1[0] or b1[3] < b2[1] or b2[3] < b1[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/seg/segformer-b2/best")
    ap.add_argument("--data", default="data/processed/rs19")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--width", type=int, default=960, help="pipeline working width")
    ap.add_argument("--out", default="reports/scene")
    args = ap.parse_args()

    cfg = load_config()
    a, max_range = cfg["anomaly"], cfg["risk"]["max_range_m"]
    root = Path(args.data)
    ids = json.load(open(root / "splits.json", encoding="utf-8"))[args.split][: args.limit]
    cache = Path(args.out) / "cache" / Path(args.weights).parent.name / args.split
    cache.mkdir(parents=True, exist_ok=True)
    seg = None

    cor_stats = {"frames": 0, "found": 0, "iou": [], "horizon_px": [],
                 **{f"range_err_{d}m": [] for d in RANGE_CHECK_M}}
    stats = {v: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "gt_blobs": 0, "gt_hit": 0} for v in VARIANTS}
    n_eval = 0
    for name in tqdm(ids, desc="frames"):
        img = cv2.imread(str(root / "images" / f"{name}.jpg"))
        h = round(img.shape[0] * args.width / img.shape[1])
        img = cv2.resize(img, (args.width, h))
        gt = cv2.resize(cv2.imread(str(root / "masks" / f"{name}.png"), 0), (args.width, h),
                        interpolation=cv2.INTER_NEAREST)
        gt_cor = fit_corridor(gt, cfg)
        if gt_cor is None:
            continue  # no ego track in this frame (e.g. depot views): nothing to judge

        cf = cache / f"{name}.npz"
        if cf.exists():
            z = np.load(cf)
            labels, conf = z["labels"], z["conf"].astype(np.float32)
        else:
            seg = seg or Segmenter(args.weights, args.device)
            labels, conf = seg(img)
            np.savez_compressed(cf, labels=labels, conf=conf.astype(np.float16))
        pred_fit = fit_corridor(labels, cfg)
        cor_stats["frames"] += 1
        if pred_fit is not None:
            cor_stats["found"] += 1
            hw = cfg["track"]["train_width_m"] / 2 + cfg["track"]["side_margin_m"]
            gm, pm = gt_cor.mask(hw), pred_fit.mask(hw)
            cor_stats["iou"].append((gm & pm).sum() / max((gm | pm).sum(), 1))
            cor_stats["horizon_px"].append(abs(gt_cor.horizon_y - pred_fit.horizon_y))
            a_, b_ = gt_cor.width_coef
            for d in RANGE_CHECK_M:
                y = (gt_cor.focal_px * gt_cor.gauge_m / d - b_) / a_  # row that GT geometry puts at d
                if gt_cor.y_top <= y < gt.shape[0]:
                    cor_stats[f"range_err_{d}m"].append(abs(float(pred_fit.distance_m(y)) - d) / d)
        pred_cor = pred_fit or gt_cor  # fall back so object detection is still judged

        g = gt_blobs(gt, gt_cor, a, max_range)
        has_gt = len(g) > 0
        n_eval += 1
        for v, thr in VARIANTS.items():
            p = find_foreign_blobs(labels, conf, pred_cor, a["half_width_m"], thr, a["min_size_m"],
                                   a["min_area_px"], max_range)
            s = stats[v]
            if p and has_gt:
                s["tp"] += 1
            elif p:
                s["fp"] += 1
            elif has_gt:
                s["fn"] += 1
            else:
                s["tn"] += 1
            s["gt_blobs"] += len(g)
            s["gt_hit"] += sum(any(overlaps(gb.box, pb.box) for pb in p) for gb in g)

    rows = []
    for v, s in stats.items():
        prec = s["tp"] / max(s["tp"] + s["fp"], 1)
        rec = s["tp"] / max(s["tp"] + s["fn"], 1)
        rows.append({
            "variant": v, "frames": n_eval, "frames_with_obstacle": s["tp"] + s["fn"],
            "precision": prec, "recall": rec, "F1": 2 * prec * rec / max(prec + rec, 1e-9),
            "false_alarm_rate": s["fp"] / max(s["fp"] + s["tn"], 1),
            "object_recall": s["gt_hit"] / max(s["gt_blobs"], 1),
        })
    out = Path(args.out)
    cs = cor_stats
    corridor = {"frames_with_gt_track": cs["frames"], "found_rate": cs["found"] / max(cs["frames"], 1),
                "mean_IoU": float(np.mean(cs["iou"])) if cs["iou"] else None,
                "median_horizon_err_px": float(np.median(cs["horizon_px"])) if cs["horizon_px"] else None}
    corridor.update({f"median_rel_range_err_{d}m": float(np.median(cs[f"range_err_{d}m"]))
                     for d in RANGE_CHECK_M if cs[f"range_err_{d}m"]})
    print("corridor:", json.dumps(corridor, indent=2))
    with open(out / f"corridor_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(corridor, f, indent=2)
    with open(out / f"results_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|---" * len(cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])
                                       for c in cols) + " |")
    (out / f"results_{args.split}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
