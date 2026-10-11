"""Class-agnostic foreign-object check inside the track.

The segmentation model knows what a normal railway bed looks like (rails, sleepers, ballast,
grass). Inside the inner track zone, any pixel predicted as something else, or predicted with
low confidence, is treated as a potential foreign object. This catches obstacles the detector
was never trained on, at the cost of the detector's class label.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .classes import GROUND_IDS
from .corridor import Corridor


@dataclass
class Blob:
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    area: int
    contour: np.ndarray


def find_foreign_blobs(
    labels: np.ndarray,
    conf: np.ndarray,
    corridor: Corridor,
    half_width_m: float,
    conf_threshold: float,
    min_size_m: float,
    min_area_px: int,
    max_range_m: float,
    exclude_boxes: list[tuple[float, float, float, float]] = (),
    ignore_mask: np.ndarray | None = None,
) -> list[Blob]:
    """Blobs must be at least `min_size_m` across at their own distance (via the gauge scale),
    with `min_area_px` as an absolute floor against single-pixel noise."""
    zone = corridor.mask(half_width_m)
    rows = np.arange(labels.shape[0])
    zone[corridor.distance_m(rows) > max_range_m] = False
    zone[rows <= corridor.horizon_y + 2] = False
    if ignore_mask is not None:  # e.g. the train's own cab / hood
        zone &= ~ignore_mask
    ground = np.isin(labels, GROUND_IDS)
    suspicious = zone & (~ground | (conf < conf_threshold))
    for x1, y1, x2, y2 in exclude_boxes:  # already explained by the detector
        suspicious[int(y1) : int(y2) + 1, int(x1) : int(x2) + 1] = False
    m = cv2.morphologyEx(suspicious.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    blobs = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        need = max(min_area_px, (min_size_m * float(corridor.px_per_m(y + h))) ** 2)
        if area < need:
            continue
        cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        blobs.append(Blob((x, y, x + w, y + h), int(area), cnts[0]))
    return blobs


def _near(a, b) -> bool:
    """Box centres closer than the larger box side (tolerates motion between updates)."""
    ax, ay, bx, by = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    size = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1], 8)
    return abs(ax - bx) < size and abs(ay - by) < size


class BlobPersistence:
    """Keep only unknown objects seen at roughly the same place on `need` consecutive
    segmentation updates; one-update flicker (glare, far-range noise) is dropped."""

    def __init__(self, need: int = 3):
        self.need = need
        self.tracks: list[tuple[tuple, int]] = []  # (box, consecutive updates seen)

    def update(self, blobs: list[Blob], seg_updated: bool) -> list[Blob]:
        if self.need <= 1:
            return blobs
        if seg_updated:
            self.tracks = [(b.box, max((n for box, n in self.tracks if _near(b.box, box)), default=0) + 1)
                           for b in blobs]
        confirmed = [box for box, n in self.tracks if n >= self.need]
        return [b for b in blobs if any(_near(b.box, c) for c in confirmed)]
