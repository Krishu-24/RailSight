"""End-to-end per-frame processing: segmentation -> corridor -> detection + tracking ->
unknown-object check -> distance -> risk -> annotated frame."""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .anomaly import BlobPersistence, find_foreign_blobs
from .config import ROOT
from .corridor import Corridor, focal_from_hfov, select_ego_track
from .risk import (
    Hazard,
    HazardAssessment,
    RiskLevel,
    RiskStateMachine,
    assess_hazard,
    stopping_distance,
)
from .seg import RAIL_IDS, TRACK_IDS, Segmenter, colorize

LEVEL_COLOR = {RiskLevel.SAFE: (80, 200, 80), RiskLevel.WARNING: (0, 190, 255),
               RiskLevel.CRITICAL: (40, 40, 235)}  # BGR
UNKNOWN_COLOR = (230, 60, 200)


@dataclass
class FrameResult:
    image: np.ndarray
    state: RiskLevel
    observed: RiskLevel
    assessments: list[HazardAssessment] = field(default_factory=list)
    info: dict = field(default_factory=dict)


def _resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


class RailSightPipeline:
    def __init__(self, cfg: dict, work_width: int = 960, show_seg: bool = False):
        self.cfg = cfg
        self.work_width = work_width
        self.show_seg = show_seg
        m = cfg["models"]
        seg_dir = _resolve(m["seg_weights"])
        precision = m.get("seg_precision", "bf16")
        self.seg = Segmenter(seg_dir, m["device"], precision) if seg_dir.exists() else None
        if self.seg is None:
            print(f"[railsight] no segmentation weights at {seg_dir}: corridor disabled")

        from ultralytics import RTDETR, YOLO

        det_path = _resolve(m["det_weights"])
        if not det_path.exists():
            raise FileNotFoundError(
                f"detector weights not found at {det_path}. Download the RailSight weights package and "
                "unzip it in the repository root (see README, 'Quick start'), or train your own "
                "(scripts/train/train_det.py).")
        self.det = (RTDETR if "rtdetr" in det_path.as_posix() else YOLO)(str(det_path))
        self.seg_name = seg_dir.parent.name if self.seg is not None else "none"  # e.g. segformer-b2
        self.det_name = det_path.parent.parent.name  # e.g. yolo11m_rail_v2_960
        self.speed_kmh = cfg["train"]["speed_kmh"]
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.corridor: Corridor | None = None
        self.labels = self.conf = None
        self.track_conf = 1.0
        self.persistence = BlobPersistence(self.cfg["anomaly"].get("persist_updates", 1))
        self.frame_idx = 0
        self.history = defaultdict(lambda: deque(maxlen=30))  # track id -> (t, lateral_m)
        r = self.cfg["risk"]
        self.sm = RiskStateMachine(r["escalate_frames"], r["deescalate_frames"])
        self._t_last = None
        self._fps = 0.0
        if hasattr(self, "det") and getattr(self.det, "predictor", None) is not None:
            self.det.predictor = None  # restart ByteTrack ids for a new video

    @property
    def half_width_m(self) -> float:
        t = self.cfg["track"]
        return t["train_width_m"] / 2 + t["side_margin_m"]

    # ------------------------------------------------------------------ main step
    def process(self, frame_bgr: np.ndarray, t_s: float) -> FrameResult:
        tic = time.perf_counter()
        h0, w0 = frame_bgr.shape[:2]
        frame = cv2.resize(frame_bgr, (self.work_width, round(h0 * self.work_width / w0)))
        own = self._vehicle_mask(frame.shape[:2])
        tc, rc, ac = self.cfg["track"], self.cfg["risk"], self.cfg["anomaly"]

        # 1. segmentation + corridor (every N frames; the track changes slowly)
        seg_updated = self.seg is not None and self.frame_idx % tc["seg_every_n"] == 0
        if seg_updated:
            self.labels, self.conf = self.seg(frame)
            track = np.isin(self.labels, TRACK_IDS)
            if own is not None:
                track &= ~own
            ego = select_ego_track(track)
            # how sure the model is about the track itself; drops sharply in darkness or glare
            self.track_conf = float(self.conf[ego].mean()) if ego.sum() > 200 else 0.0
            f = focal_from_hfov(frame.shape[1], self.cfg["camera"]["hfov_deg"])
            rails = np.isin(self.labels, RAIL_IDS) & ego if tc.get("rail_pairs") else None
            fit = Corridor.fit(ego, tc["gauge_m"] + tc.get("rail_head_m", 0.0), f, rail_mask=rails)
            if fit is not None:
                self.corridor = fit.blend(self.corridor, tc["smoothing"])
            elif self.corridor is not None and self.frame_idx % (tc["seg_every_n"] * 10) == 0:
                self.corridor = None  # lost the track for a while: stop trusting the old fit

        # 2. detection + tracking
        res = self.det.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False,
                             device=self.seg.device if self.seg else self.cfg["models"]["device"],
                             quantize="fp16" if self.cfg["models"].get("det_half", False)
                             and self.seg is not None and self.seg.device == "cuda" else None,
                             conf=self.cfg["models"]["det_conf"], imgsz=self.cfg["models"]["det_imgsz"])[0]
        names = res.names
        dets = []
        if res.boxes is not None and len(res.boxes):
            ids = res.boxes.id.int().tolist() if res.boxes.id is not None else [-1] * len(res.boxes)
            for box, c, conf, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.int().tolist(),
                                         res.boxes.conf.tolist(), ids):
                if own is not None:
                    x1, y1, x2, y2 = (int(v) for v in box)
                    if own[y1:y2 + 1, x1:x2 + 1].mean() > 0.5:
                        continue  # an object inside the cab or on the vehicle itself
                dets.append((box, names[c], conf, tid))

        # 3. physics
        v = self.speed_kmh / 3.6
        stop_d = stopping_distance(v, self.cfg["train"]["reaction_s"], self.cfg["train"]["decel_mps2"])
        blobs = []
        det_a: list[HazardAssessment | None] = [None] * len(dets)
        blob_a: list[HazardAssessment | None] = []
        cor = self.corridor
        if cor is not None:
            for i, (box, label, _conf, tid) in enumerate(dets):
                hz = self._to_hazard(box, label, tid, t_s, cor)
                det_a[i] = self._assess(hz, v, stop_d) if hz is not None else None
            low_visibility = self.track_conf < ac.get("min_track_conf", 0.0)
            if ac["enabled"] and self.labels is not None and not low_visibility:
                blobs = find_foreign_blobs(self.labels, self.conf, cor, ac["half_width_m"],
                                           ac["conf_threshold"], ac["min_size_m"],
                                           ac["min_area_px"], rc["max_range_m"],
                                           [d[0] for d in dets], ignore_mask=own)
                blobs = self.persistence.update(blobs, seg_updated)
                for b in blobs:
                    hz = self._to_hazard(np.array(b.box, float), "unknown", -1, t_s, cor)
                    blob_a.append(self._assess(hz, v, stop_d) if hz is not None else None)
        assessments = [a for a in det_a + blob_a if a is not None]

        observed = max((a.level for a in assessments), default=RiskLevel.SAFE)
        state = self.sm.update(observed)

        # 4. render
        dt = time.perf_counter() - tic
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(dt, 1e-3)) if self._fps else 1.0 / max(dt, 1e-3)
        in_path = [a for a in assessments if a.in_path]
        nearest = min((a.hazard.distance_m for a in in_path), default=None)
        info = {"frame": self.frame_idx, "t": round(t_s, 3), "state": state.name,
                "observed": observed.name, "speed_kmh": self.speed_kmh,
                "stop_dist_m": round(stop_d, 1), "n_detections": len(dets),
                "n_unknown": len(blobs), "nearest_in_path_m": None if nearest is None else round(nearest, 1),
                "corridor": cor is not None, "track_conf": round(self.track_conf, 3),
                "low_visibility": cor is not None and self.track_conf < ac.get("min_track_conf", 0.0),
                "fps": round(self._fps, 1)}
        img = self._render(frame, dets, det_a, blobs, blob_a, state, stop_d, info)
        self.frame_idx += 1
        return FrameResult(img, state, observed, assessments, info)

    # ------------------------------------------------------------------ helpers
    def _vehicle_mask(self, shape) -> np.ndarray | None:
        """Image region showing the train itself (dashboard, hood), set once per camera mounting."""
        frac = self.cfg["camera"].get("ignore_below")
        if not frac:
            return None
        m = np.zeros(shape, bool)
        m[int(frac * shape[0]) :] = True
        return m

    def _to_hazard(self, box, label, tid, t_s, cor: Corridor) -> Hazard | None:
        x1, y1, x2, y2 = box
        y = min(float(y2), cor.height - 1)
        if y > cor.bottom_y + 8:
            return None  # ground contact below the visible track: inside the cab / on the vehicle
        if y <= cor.horizon_y + 2:
            return None  # at or above the horizon: no ground contact point to range from
        d = float(cor.distance_m(y))
        if d > self.cfg["risk"]["max_range_m"]:
            return None
        l1, l2 = float(cor.lateral_m(x1, y)), float(cor.lateral_m(x2, y))
        lateral = 0.0 if l1 <= 0 <= l2 else (l1 if abs(l1) < abs(l2) else l2)
        lat_speed = 0.0
        if tid >= 0:
            hist = self.history[tid]
            hist.append((t_s, lateral))
            pts = [(t, lat) for t, lat in hist if t_s - t <= 1.5]
            if len(pts) >= 4 and pts[-1][0] - pts[0][0] > 0.3:
                ts, ls = np.array(pts).T
                lat_speed = float(np.polyfit(ts, ls, 1)[0])
                lat_speed = float(np.clip(lat_speed, -6.0, 6.0))  # faster than a sprint = noise
        return Hazard(tid, label, d, lateral, lat_speed)

    def _assess(self, hz, v, stop_d):
        r = self.cfg["risk"]
        return assess_hazard(hz, v, self.half_width_m, stop_d, r["stop_margin_m"],
                             r["near_buffer_m"], r["max_range_m"])

    def _render(self, frame, dets, det_a, blobs, blob_a, state, stop_d, info):
        img = frame.copy()
        if self.show_seg and self.labels is not None:
            img = cv2.addWeighted(img, 0.6, colorize(self.labels), 0.4, 0)
        cor = self.corridor
        if cor is not None:
            overlay = img.copy()
            poly = cor.polygon(self.half_width_m)
            if len(poly):
                cv2.fillPoly(overlay, [poly], LEVEL_COLOR[state])
                img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
                cv2.polylines(img, [poly], True, LEVEL_COLOR[state], 2, cv2.LINE_AA)
            self._draw_range_ticks(img, cor, stop_d)

        for (box, label, _conf, tid), a in zip(dets, det_a):
            col = LEVEL_COLOR[a.level] if a else (200, 200, 200)
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            txt = f"{label}" + (f" #{tid}" if tid >= 0 else "")
            if a:
                txt += f" {a.hazard.distance_m:.0f}m"
            _label(img, txt, (x1, y1), col)
        for b, a in zip(blobs, blob_a):
            cv2.drawContours(img, [b.contour], -1, UNKNOWN_COLOR, 2)
            txt = "unknown" + (f" {a.hazard.distance_m:.0f}m" if a else "")
            _label(img, txt, (b.box[0], b.box[1]), UNKNOWN_COLOR)
        self._draw_hud(img, state, info)
        return img

    def _draw_range_ticks(self, img, cor: Corridor, stop_d):
        a, b = cor.width_coef
        for d in (25, 50, 100, 200, 400):
            y = (cor.focal_px * cor.gauge_m / d - b) / a
            if not (cor.y_top - 5 <= y < img.shape[0]):
                continue
            x = cor.centre_x(y) - self.half_width_m * cor.px_per_m(y)
            cv2.line(img, (int(x) - 12, int(y)), (int(x), int(y)), (255, 255, 255), 2)
            cv2.putText(img, f"{d}m", (int(x) - 52, int(y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
        y = (cor.focal_px * cor.gauge_m / max(stop_d, 1) - b) / a
        if cor.y_top <= y < img.shape[0]:  # where the train could stop at the current speed
            half = self.half_width_m * cor.px_per_m(y)
            c = cor.centre_x(y)
            cv2.line(img, (int(c - half), int(y)), (int(c + half), int(y)), (0, 0, 255), 2)

    def _draw_hud(self, img, state, info):
        col = LEVEL_COLOR[state]
        cv2.rectangle(img, (0, 0), (300, 112), (20, 20, 20), -1)
        cv2.rectangle(img, (0, 0), (300, 34), col, -1)
        cv2.putText(img, state.name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)
        near = info["nearest_in_path_m"]
        lines = [
            f"speed {info['speed_kmh']:.0f} km/h   stop {info['stop_dist_m']:.0f} m",
            f"nearest in path: {'-' if near is None else f'{near:.0f} m'}",
            ("track: NOT FOUND" if not info["corridor"] else "track: LOW VISIBILITY"
             if info["low_visibility"] else "track: locked") + f"   {info['fps']:.0f} fps",
        ]
        for i, s in enumerate(lines):
            cv2.putText(img, s, (10, 56 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1,
                        cv2.LINE_AA)


def _label(img, text, org, col):
    x, y = int(org[0]), int(org[1])
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(y, th + 6)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), col, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
