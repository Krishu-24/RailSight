"""Figures for the project report, drawn in one visual theme from the files under reports/ and the
rendered demo videos in outputs/.

    uv run python docs/report/make_figures.py
"""

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
REP = ROOT / "reports"

INK, MUTED, GRID = "#14213D", "#6B7280", "#E5E7EB"
NAVY, BLUE, AMBER, RED, GREEN, SLATE = "#0B1F3A", "#2563EB", "#F2A900", "#D7263D", "#16A34A", "#94A3B8"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def label_bars(ax, bars, fmt="{:.2f}", color=INK):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8,
                    color=color)


def frame(video, idx, out, width=1280):
    cap = cv2.VideoCapture(str(ROOT / video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, im = cap.read()
    assert ok, video
    h = round(width * 9 / 16)
    s = min(width / im.shape[1], h / im.shape[0])  # fit into 16:9, letterbox tall frames
    im = cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
    px, py = width - im.shape[1], h - im.shape[0]
    im = cv2.copyMakeBorder(im, py // 2, py - py // 2, px // 2, px - px // 2, cv2.BORDER_CONSTANT,
                            value=(58, 31, 11))  # navy bars
    cv2.imwrite(str(OUT / out), im, [cv2.IMWRITE_JPEG_QUALITY, 90])


def first_run(log, cond, n):
    c = cond(log).astype(int).values
    return next((i for i in range(len(c) - n) if c[i : i + n].all()), 0)


def demo_frames():
    log = pd.read_csv(ROOT / "outputs/final/india_people_crossing_log.csv").fillna({"labels": ""})
    people = log.labels.str.count("person")
    # most people in view while CRITICAL, in a frame where the camera's own coach is not boxed
    ok = (log.state == "CRITICAL") & ~log.labels.str.contains("rail_vehicle")
    i = int(people.where(ok, 0).idxmax())
    frame("outputs/final/india_people_crossing_railsight.mp4", i, "demo_india.jpg")
    log = pd.read_csv(ROOT / "outputs/final/level_crossing_log.csv")
    locked = (log.state == "SAFE") & log.corridor & ~log.low_confidence
    frame("outputs/final/level_crossing_railsight.mp4", first_run(log, lambda d: locked, 40) + 20,
          "demo_crossing.jpg")
    log = pd.read_csv(ROOT / "outputs/final/cab_ride_station_tunnel_log.csv")
    frame("outputs/final/cab_ride_station_tunnel_railsight.mp4",
          first_run(log, lambda d: (d.state == "SAFE") & d.corridor, 90) + 40, "demo_station.jpg")
    frame("outputs/final/fog_forest_railsight.mp4", 90, "demo_fog.jpg")
    # before / after the false-alarm fixes on the in-cab clip: a frame CRITICAL before, SAFE after
    before = pd.read_csv(ROOT / "outputs/final_v1/cab_interior_front_log.csv")
    after = pd.read_csv(ROOT / "outputs/final/cab_interior_front_log.csv")
    n = min(len(before), len(after))
    ok = (before.state[:n] == "CRITICAL") & (after.state[:n] == "SAFE") & (before.n_unknown[:n] > 0)
    k = int(ok.idxmax())
    for tag, folder in (("before", "final_v1"), ("after", "final")):
        frame(f"outputs/{folder}/cab_interior_front_railsight.mp4", k, f"cab_{tag}.jpg")


def stopping():
    kmh = np.linspace(10, 160, 151)
    v = kmh / 3.6
    f = 1920 / (2 * np.tan(np.radians(30)))
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 2.25))
    for dec, name, c in ((0.5, "freight, 0.5 m/s²", SLATE), (1.0, "passenger, 1.0 m/s²", BLUE),
                         (1.4, "emergency EMU, 1.4 m/s²", NAVY)):
        d = v * 2.0 + v**2 / (2 * dec)
        a.plot(kmh, d, color=c, lw=2, label=name)
        b.plot(kmh, f * 1.7 / d, color=c, lw=2, label=name)
    a.set(xlabel="speed (km/h)", ylabel="stopping distance (m)", title="Distance needed to stop")
    b.set(xlabel="speed (km/h)", ylabel="pixels", yscale="log",
          title="Height of a person at that distance")
    b.axhline(16, color=RED, lw=1, ls="--")
    b.text(95, 18.5, "~16 px: hard to detect", color=RED, fontsize=8)
    a.legend(fontsize=8)
    fig.savefig(OUT / "stopping.png")
    plt.close(fig)


def segmentation():
    models = [("segformer-b2", "SegFormer-B2"), ("deeplab-r50", "DeepLabV3-R50"),
              ("segformer-b0", "SegFormer-B0")]
    seg = {k: load(f"runs/seg/{k}/metrics_test.json")
           if (ROOT / f"runs/seg/{k}/metrics_test.json").exists() else None for k, _ in models}
    cor = {k: load(f"reports/scene_{k}/corridor_test.json") for k, _ in models}
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 2.9), gridspec_kw={"width_ratios": [1.1, 1]})
    x = np.arange(len(models))
    if all(seg.values()):
        b1 = a.bar(x - 0.18, [seg[k]["mIoU"] for k, _ in models], 0.36, color=SLATE, label="mIoU, 19 classes")
        b2 = a.bar(x + 0.18, [seg[k]["track_IoU"] for k, _ in models], 0.36, color=NAVY, label="track IoU")
        label_bars(a, b1)
        label_bars(a, b2)
    a.set_xticks(x, [n for _, n in models])
    a.set(ylim=(0.5, 1.0), title="Segmentation, RailSem19 test")
    a.legend(fontsize=8, loc="upper right")
    for (k, n), c in zip(models, (NAVY, AMBER, SLATE)):
        b.plot([25, 50, 100], [100 * cor[k][f"median_rel_range_err_{d}m"] for d in (25, 50, 100)], "o-",
               color=c, lw=2, label=n)
    b.set(xlabel="distance along the track (m)", ylabel="median deviation (%)",
          title="Ranging deviation caused by segmentation", xticks=[25, 50, 100])
    b.legend(fontsize=8)
    fig.savefig(OUT / "segmentation.png")
    plt.close(fig)


def detection():
    rf = {r["model"]: r for r in load("reports/detection/roboflow_test.json")}
    cv = {r["model"]: r for r in load("reports/detection/cabview_test.json")}
    hi = {r["model"]: r for r in load("reports/detection/cabview_test_960px.json")}
    rf_hi = {r["model"]: r for r in load("reports/detection/roboflow_test_960px.json")}
    rows = [("COCO\nzero-shot", rf["yolo11s COCO zero-shot"], cv["yolo11s COCO zero-shot"]),
            ("YOLO11s\nv1", rf["yolo11s_rail"], cv["yolo11s_rail"]),
            ("YOLO11m\nv1", rf["yolo11m_rail"], cv["yolo11m_rail"]),
            ("RT-DETR-l\nv1", rf["rtdetr_l_rail"], cv["rtdetr_l_rail"]),
            ("YOLO11m\nv2", rf["yolo11m_rail_v2"], cv["yolo11m_rail_v2"]),
            ("YOLO11m v2\n960 px", rf_hi["yolo11m_rail_v2_960"], hi["yolo11m_rail_v2_960"])]
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    x = np.arange(len(rows))
    b1 = ax.bar(x - 0.19, [r[1]["mAP50_all"] for r in rows], 0.38, color=SLATE,
                label="Roboflow test (web images)")
    b2 = ax.bar(x + 0.19, [r[2]["mAP50_all"] for r in rows], 0.38,
                color=[NAVY if i >= 4 else RED for i in range(len(rows))], label="RailSem19 cab-view test")
    label_bars(ax, b1, color=MUTED)
    label_bars(ax, b2)
    ax.set_xticks(x, [r[0] for r in rows], fontsize=8.5)
    ax.set(ylabel="mAP50", ylim=(0, 0.85), title="Detection: web-image test vs real cab-view frames")
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(OUT / "detection.png")
    plt.close(fig)

    ref = cv["yolo11m_rail_v2"]
    steps = [("v2 @ 640 px", ref), ("v2 run @ 960 px", hi["yolo11m_rail_v2"]),
             ("v2 fine-tuned @ 960 px", hi["yolo11m_rail_v2_960"])]
    fig, ax = plt.subplots(figsize=(8.2, 2.5))
    keys = [("mAP50_all", "mAP50"), ("recall_small(<32px)", "recall: small"),
            ("recall_medium(32-96px)", "recall: medium"), ("recall_large(>96px)", "recall: large")]
    x = np.arange(len(keys))
    for i, ((name, r), c) in enumerate(zip(steps, (SLATE, BLUE, NAVY))):
        bars = ax.bar(x + (i - 1) * 0.26, [r[k] for k, _ in keys], 0.26, color=c, label=name)
        label_bars(ax, bars)
    ax.set_xticks(x, [n for _, n in keys])
    ax.set(ylim=(0, 0.9), title="Input resolution on RailSem19 cab-view frames")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    fig.savefig(OUT / "resolution.png")
    plt.close(fig)


def ranging():
    c = load("reports/ranging/consistency.json")
    level = np.array([p["ratio"] for p in c["people"] if p["track_level"]])
    other = np.array([p["ratio"] for p in c["people"] if not p["track_level"]])
    fig, ax = plt.subplots(figsize=(8.2, 2.25))
    bins = np.linspace(0.3, 2.0, 35)
    ax.hist(level, bins=bins, color=NAVY, alpha=0.9, label=f"feet at track level (n={len(level)})")
    ax.hist(other, bins=bins, color=AMBER, alpha=0.85, label=f"on platforms / elsewhere (n={len(other)})")
    for vals, col in ((level, NAVY), (other, AMBER)):
        ax.axvline(np.median(vals), color=col, lw=1.5, ls="--")
    ax.axvline(1.0, color=INK, lw=1)
    top = ax.get_ylim()[1] * 0.85
    ax.text(np.median(level) + 0.02, top, f"median {np.median(level):.3f}", color=NAVY, fontsize=8)
    ax.text(np.median(other) - 0.32, top, f"median {np.median(other):.2f}", color="#9A6B00", fontsize=8)
    ax.set(xlabel="gauge distance ÷ human-height distance", ylabel="people",
           title="Calibration-free ranging check on RailSem19")
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(OUT / "ranging.png")
    plt.close(fig)


def unknown_objects():
    rows = load("reports/scene_segformer-b2/unknown_objects_test.json")
    names = ["class only", "+ conf < 0.3", "+ conf < 0.5", "+ conf < 0.7"]
    fig, ax = plt.subplots(figsize=(8.2, 2.2))
    x = np.arange(len(rows))
    for i, (k, n, c) in enumerate((("precision", "precision", NAVY), ("recall", "recall", BLUE),
                                   ("false_alarm_rate", "false-alarm rate", RED))):
        bars = ax.bar(x + (i - 1) * 0.26, [r[k] for r in rows], 0.26, color=c, label=n)
        label_bars(ax, bars)
    ax.set_xticks(x, names)
    ax.set(ylim=(0, 1.05), title="Unknown-object rule variants (SegFormer-B2, RailSem19 test)")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    fig.savefig(OUT / "unknown.png")
    plt.close(fig)


def false_alarms():
    clips = [("cab_interior_front", "in-cab\ncamera"), ("cab_ride_station_tunnel", "station +\ndark tunnel"),
             ("india_people_crossing", "people crossing\n(real hazard)"), ("rural_track", "rural\ntrack"),
             ("level_crossing", "level\ncrossing"), ("fog_forest", "fog")]

    def critical(folder, clip):
        return (pd.read_csv(ROOT / f"outputs/{folder}/{clip}_log.csv").state == "CRITICAL").mean()

    share = {tag: [critical(folder, c) for c, _ in clips]
             for tag, folder in (("first render", "final_v1"), ("final", "final"))}
    fig, ax = plt.subplots(figsize=(8.2, 2.6))
    x = np.arange(len(clips))
    b1 = ax.bar(x - 0.19, share["first render"], 0.38, color=SLATE, label="first render")
    b2 = ax.bar(x + 0.19, share["final"], 0.38, color=NAVY, label="after the fixes")
    label_bars(ax, b1, "{:.0%}", MUTED)
    label_bars(ax, b2, "{:.0%}")
    ax.set_xticks(x, [n for _, n in clips], fontsize=8)
    ax.set(ylabel="time in CRITICAL", ylim=(0, 1.05), title="False alarms on real video, before and after")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(fontsize=8)
    fig.savefig(OUT / "false_alarms.png")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    demo_frames()
    stopping()
    segmentation()
    detection()
    ranging()
    unknown_objects()
    false_alarms()
    print("figures ->", OUT, sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
