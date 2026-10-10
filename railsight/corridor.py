"""Ego-track extraction and the metric danger corridor.

The ego track is followed row by row from the bottom of the frame upwards. Two models are then
fitted to it:

* centre(y): quadratic in the image row, which follows gentle curves;
* width(y): linear in the image row. On flat ground the projected gauge shrinks linearly
  towards the horizon, so the fit also gives the horizon row where width reaches zero.

Because the rail gauge is a known physical length, width(y) converts pixels to metres at
every row, which yields both object distance and the corridor's real-world width.
"""

from dataclasses import dataclass

import cv2
import numpy as np


def focal_from_hfov(image_width: int, hfov_deg: float) -> float:
    return image_width / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))


def select_ego_track(track_mask: np.ndarray, bottom_frac: float = 0.15) -> np.ndarray:
    """Keep the connected track region that starts at the bottom centre of the frame."""
    h, w = track_mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(track_mask.astype(np.uint8), 8)
    if n <= 1:
        return np.zeros_like(track_mask, dtype=bool)

    y0 = int(h * (1 - bottom_frac))
    cols = np.arange(w)
    centre_weight = 1.0 - np.abs(cols - w / 2) / (w / 2)  # 1 at centre, 0 at the edges
    best, best_score = 0, 0.0
    for i in range(1, n):
        band = labels[y0:] == i
        if not band.any():
            continue
        score = float((band * centre_weight[None, :]).sum())
        if score > best_score:
            best, best_score = i, score
    return labels == best if best else np.zeros_like(track_mask, dtype=bool)


def trace_rows(ego_mask: np.ndarray, min_width: int = 2) -> np.ndarray:
    """Follow the ego path upwards; at switches keep the branch nearest the previous centre.

    Returns an array of (y, centre_x, width_px) rows.
    """
    h, _ = ego_mask.shape
    out = []
    prev_c = None
    for y in range(h - 1, -1, -1):
        row = ego_mask[y]
        if not row.any():
            if out:
                break  # the track has ended (or is fully occluded from here up)
            continue
        padded = np.concatenate([[0], row.astype(np.int8), [0]])
        edges = np.flatnonzero(np.diff(padded))
        segs = edges.reshape(-1, 2)  # [start, end) per run of track pixels
        segs = segs[(segs[:, 1] - segs[:, 0]) >= min_width]
        if len(segs) == 0:
            continue
        centres = segs.mean(axis=1)
        k = int(np.argmax(segs[:, 1] - segs[:, 0])) if prev_c is None else int(
            np.argmin(np.abs(centres - prev_c))
        )
        c, wd = centres[k], segs[k, 1] - segs[k, 0]
        out.append((y, c, wd))
        prev_c = c
    return np.array(out, dtype=np.float64).reshape(-1, 3)


def _runs(row: np.ndarray, max_gap: int = 2) -> np.ndarray:
    """[start, end) of True runs in a 1-D mask, bridging gaps of up to `max_gap` pixels."""
    padded = np.concatenate([[0], row.astype(np.int8), [0]])
    r = np.flatnonzero(np.diff(padded)).reshape(-1, 2)
    if len(r) > 1:
        keep = np.concatenate([[True], (r[1:, 0] - r[:-1, 1]) > max_gap])
        starts = r[keep, 0]
        ends = np.concatenate([r[np.flatnonzero(keep)[1:] - 1, 1], [r[-1, 1]]])
        r = np.stack([starts, ends], 1)
    return r


def trace_rails(ego_mask: np.ndarray, rail_mask: np.ndarray, tol: float = 0.35) -> np.ndarray:
    """Like `trace_rows`, but measures the gauge between the two rails of the ego track.

    Where several tracks merge (switches, yards) the track region becomes one wide blob and
    its width overstates the gauge. Inside the blob, adjacent rail pairs are considered and the
    pair whose spacing matches the last measured gauge (within `tol`) and whose midpoint is
    nearest the previous centre is kept; rows without such a pair are skipped. Tracing starts
    with the rail pair whose midpoint is nearest the image centre (the camera sits over the
    ego track).
    """
    h, w = ego_mask.shape
    out = []
    prev_c = prev_w = None
    for y in range(h - 1, -1, -1):
        row = ego_mask[y]
        if not row.any():
            if out:
                break
            continue
        segs = _runs(row, 0)
        centres = segs.mean(axis=1)
        k = int(np.argmax(segs[:, 1] - segs[:, 0])) if prev_c is None else int(
            np.argmin(np.abs(centres - prev_c))
        )
        s0, s1 = segs[k]
        rails = _runs(rail_mask[y, s0:s1]) + s0
        if len(rails) < 2:
            continue
        if prev_w is None:
            mids = (rails[:-1, 0] + rails[1:, 1]) / 2
            i = int(np.argmin(np.abs(mids - w / 2)))
            if abs(mids[i] - w / 2) > w / 4:
                continue
            c, wd = mids[i], rails[i + 1, 1] - rails[i, 0]
        else:
            best = None
            for i in range(len(rails) - 1):
                wd_i = rails[i + 1, 1] - rails[i, 0]
                if abs(wd_i / prev_w - 1) > tol:
                    continue
                c_i = (rails[i, 0] + rails[i + 1, 1]) / 2
                if best is None or abs(c_i - prev_c) < abs(best[0] - prev_c):
                    best = (c_i, wd_i)
            if best is None:
                continue
            c, wd = best
        out.append((y, c, wd))
        prev_c, prev_w = c, wd
    return np.array(out, dtype=np.float64).reshape(-1, 3)


def _robust_polyfit(x: np.ndarray, y: np.ndarray, deg: int, iters: int = 3) -> np.ndarray:
    keep = np.ones_like(x, dtype=bool)
    coef = np.polyfit(x, y, deg)
    for _ in range(iters):
        res = y - np.polyval(coef, x)
        mad = np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-6
        new_keep = np.abs(res) <= 3.0 * 1.4826 * mad
        if new_keep.sum() < deg + 3 or (new_keep == keep).all():
            break
        keep = new_keep
        coef = np.polyfit(x[keep], y[keep], deg)
    return coef


def _same_track(rail_rows: np.ndarray, region_rows: np.ndarray) -> bool:
    """Rails may refine the gauge, but must not move the corridor onto another track: near the
    bottom of the image their centre has to lie within half a track-region width of it."""
    if len(region_rows) < 5:
        return True
    y = rail_rows[:, 0].max()
    near = region_rows[np.argsort(np.abs(region_rows[:, 0] - y))[:5]]
    c_region, w_region = near[:, 1].mean(), near[:, 2].mean()
    c_rail = rail_rows[rail_rows[:, 0] >= y - 5, 1].mean()
    return abs(c_rail - c_region) <= 0.5 * w_region


@dataclass
class Corridor:
    centre_coef: np.ndarray   # quadratic: centre_x(y)
    width_coef: np.ndarray    # linear: track width in px at row y
    y_top: float              # furthest (smallest) row where the track was observed
    height: int
    width: int
    gauge_m: float
    focal_px: float

    # ---------- construction ----------
    @classmethod
    def fit(
        cls,
        ego_mask: np.ndarray,
        gauge_m: float,
        focal_px: float,
        min_rows: int = 20,
        rail_mask: np.ndarray | None = None,
    ) -> "Corridor | None":
        """Fit to the ego track. With a rail mask, the gauge is measured between the rails
        (robust at switches); otherwise from the width of the track region."""
        region_rows = trace_rows(ego_mask)
        rows = region_rows
        if rail_mask is not None:
            rail_rows = trace_rails(ego_mask, rail_mask)
            if len(rail_rows) >= min_rows and _same_track(rail_rows, region_rows):
                rows = rail_rows
        if len(rows) < min_rows:
            return None
        ys, cs, ws = rows[:, 0], rows[:, 1], rows[:, 2]
        width_coef = _robust_polyfit(ys, ws, 1)
        if width_coef[0] <= 0:  # width must grow towards the bottom of the frame
            return None
        centre_coef = _robust_polyfit(ys, cs, 2 if len(rows) > 40 else 1)
        if len(centre_coef) == 2:
            centre_coef = np.concatenate([[0.0], centre_coef])
        h, w = ego_mask.shape
        return cls(centre_coef, width_coef, float(ys.min()), h, w, gauge_m, focal_px)

    def blend(self, prev: "Corridor | None", alpha: float) -> "Corridor":
        """Exponential moving average with the previous frame's fit."""
        if prev is None or alpha <= 0:
            return self

        def mix(a, b):
            return alpha * b + (1 - alpha) * a

        return Corridor(
            mix(self.centre_coef, prev.centre_coef),
            mix(self.width_coef, prev.width_coef),
            mix(self.y_top, prev.y_top),
            self.height,
            self.width,
            self.gauge_m,
            self.focal_px,
        )

    # ---------- geometry ----------
    @property
    def horizon_y(self) -> float:
        a, b = self.width_coef
        return -b / a

    def centre_x(self, y):
        return np.polyval(self.centre_coef, y)

    def width_px(self, y):
        return np.maximum(np.polyval(self.width_coef, y), 1e-3)

    def px_per_m(self, y):
        return self.width_px(y) / self.gauge_m

    def distance_m(self, y):
        """Pinhole ranging: Z = f * gauge / projected_gauge."""
        return self.focal_px * self.gauge_m / self.width_px(y)

    def lateral_m(self, x, y):
        return (x - self.centre_x(y)) / self.px_per_m(y)

    def polygon(self, half_width_m: float, y_far: float | None = None, step: int = 4) -> np.ndarray:
        """Corridor outline (left edge bottom→top, then right edge top→bottom) as int32 points."""
        y_far = self.y_top if y_far is None else y_far
        y_far = max(y_far, self.horizon_y + 2)
        ys = np.arange(self.height - 1, y_far, -step, dtype=np.float64)
        if len(ys) < 2:
            return np.empty((0, 2), np.int32)
        half_px = half_width_m * self.px_per_m(ys)
        cx = self.centre_x(ys)
        left = np.stack([cx - half_px, ys], axis=1)
        right = np.stack([cx + half_px, ys], axis=1)[::-1]
        return np.round(np.concatenate([left, right])).astype(np.int32)

    def mask(self, half_width_m: float) -> np.ndarray:
        m = np.zeros((self.height, self.width), np.uint8)
        poly = self.polygon(half_width_m)
        if len(poly):
            cv2.fillPoly(m, [poly], 1)
        return m.astype(bool)
