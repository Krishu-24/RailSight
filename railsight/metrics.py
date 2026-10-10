"""Detection metrics computed independently of any framework, so a COCO-pretrained baseline and
the fine-tuned model are scored with exactly the same code."""

import numpy as np


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (N,4) and (M,4) xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """All-point interpolated AP (area under the monotone precision envelope)."""
    r = np.concatenate([[0.0], recall, [1.0]])
    p = np.concatenate([[1.0], precision, [0.0]])
    p = np.maximum.accumulate(p[::-1])[::-1]
    idx = np.flatnonzero(r[1:] != r[:-1])
    return float(np.sum((r[idx + 1] - r[idx]) * p[idx + 1]))


def _greedy_match(p, g, iou_thr):
    """Match score-sorted predictions p to ground truth g; returns matched-gt mask."""
    matched = np.zeros(len(g), bool)
    ious = box_iou(p, g)
    for j in range(len(p)):
        if not len(g):
            break
        cand = np.where(~matched, ious[j], 0.0)
        if cand.max() >= iou_thr:
            matched[int(cand.argmax())] = True
    return matched


def evaluate_detections(gts, preds, num_classes, iou_thr=0.5, size_bins=None, recall_conf=0.25):
    """gts: list per image of (boxes (N,4), classes (N,)).
    preds: list per image of (boxes (M,4), classes (M,), scores (M,)).
    size_bins: optional {name: (lo, hi)} on sqrt(box area) in pixels.
    Returns per-class AP (all scores) and, per size bin, the recall at the operating
    confidence `recall_conf` (the fraction of objects the deployed detector would report)."""
    per_class = {}
    size_hits = {k: [0, 0] for k in (size_bins or {})}
    for c in range(num_classes):
        records, n_gt = [], 0
        for (gb, gc), (pb, pc, ps) in zip(gts, preds):
            g = gb[gc == c]
            p, s = pb[pc == c], ps[pc == c]
            n_gt += len(g)
            order = np.argsort(-s)
            p, s = p[order], s[order]
            matched = np.zeros(len(g), bool)
            ious = box_iou(p, g)
            for j in range(len(p)):
                k = -1
                if len(g):
                    cand = np.where(~matched, ious[j], 0.0)
                    k = int(cand.argmax()) if cand.max() >= iou_thr else -1
                if k >= 0:
                    matched[k] = True
                records.append((s[j], k >= 0))
            if size_bins and len(g):
                hit = _greedy_match(p[s >= recall_conf], g, iou_thr)
                sizes = np.sqrt(np.prod(g[:, 2:] - g[:, :2], axis=1))
                for name, (lo, hi) in size_bins.items():
                    sel = (sizes >= lo) & (sizes < hi)
                    size_hits[name][0] += int(hit[sel].sum())
                    size_hits[name][1] += int(sel.sum())
        if n_gt == 0:
            continue
        records.sort(key=lambda r: -r[0])
        tp = np.cumsum([r[1] for r in records]) if records else np.array([0])
        fp = np.cumsum([not r[1] for r in records]) if records else np.array([0])
        recall = tp / n_gt
        precision = tp / np.maximum(tp + fp, 1)
        per_class[c] = {
            "AP": average_precision(recall, precision) if records else 0.0,
            "max_recall": float(recall[-1]) if records else 0.0,
            "n_gt": n_gt,
        }
    out = {"per_class": per_class,
           "mAP": float(np.mean([v["AP"] for v in per_class.values()])) if per_class else 0.0}
    if size_bins:
        out["recall_by_size"] = {k: (h / t if t else None) for k, (h, t) in size_hits.items()}
    return out
