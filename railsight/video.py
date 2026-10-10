"""Run the pipeline over a whole video and save the annotated clip, a per-frame log and a
risk timeline plot."""

from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .pipeline import RailSightPipeline
from .risk import RiskLevel


def process_video(pipe: RailSightPipeline, src: str | Path, out_dir: str | Path,
                  max_seconds: float | None = None, progress=None) -> dict:
    src, out_dir = Path(src), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_seconds:
        total = min(total, int(max_seconds * fps)) if total else int(max_seconds * fps)

    pipe.reset()
    out_mp4 = out_dir / f"{src.stem}_railsight.mp4"
    writer = imageio.get_writer(out_mp4, fps=fps, codec="libx264", quality=7,
                                macro_block_size=8, ffmpeg_log_level="error")
    rows, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok or (total and i >= total):
            break
        r = pipe.process(frame, i / fps)
        writer.append_data(cv2.cvtColor(r.image, cv2.COLOR_BGR2RGB))
        rows.append(r.info)
        i += 1
        if progress and total:
            progress(i / total)
    cap.release()
    writer.close()

    log = pd.DataFrame(rows)
    log.to_csv(out_dir / f"{src.stem}_log.csv", index=False)
    plot = out_dir / f"{src.stem}_timeline.png"
    plot_timeline(log, plot)
    counts = log["state"].value_counts(normalize=True).to_dict() if len(log) else {}
    return {"video": out_mp4, "log": out_dir / f"{src.stem}_log.csv", "timeline": plot,
            "frames": len(log), "share_of_time": counts,
            "mean_fps": float(log["fps"].mean()) if len(log) else 0.0}


def plot_timeline(log: pd.DataFrame, path: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 2]})
    if len(log):
        lvl = log["state"].map({lv.name: int(lv) for lv in RiskLevel})
        colors = {0: "#3cb44b", 1: "#ffb000", 2: "#e6194b"}
        ax1.scatter(log["t"], lvl, c=lvl.map(colors), s=6, marker="s")
        ax1.set_yticks([0, 1, 2], ["SAFE", "WARN", "CRIT"])
        ax2.plot(log["t"], log["nearest_in_path_m"], ".", ms=3, label="nearest hazard in path")
        ax2.plot(log["t"], log["stop_dist_m"], "-", lw=1.5, color="#e6194b", label="stopping distance")
        ax2.set_ylabel("distance (m)")
        ax2.legend(loc="upper right", fontsize=8)
    ax2.set_xlabel("time (s)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
