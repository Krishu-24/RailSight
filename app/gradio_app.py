"""RailSight interactive demo.

    uv run python app/gradio_app.py
"""

import tempfile
from pathlib import Path

import cv2
import gradio as gr

from railsight.config import load_config
from railsight.pipeline import RailSightPipeline
from railsight.video import process_video

CFG = load_config()
PIPE = RailSightPipeline(CFG)


def _configure(speed, gauge, show_seg):
    PIPE.speed_kmh = speed
    PIPE.cfg["track"]["gauge_m"] = gauge
    PIPE.show_seg = show_seg


def run_image(image, speed, gauge, show_seg):
    if image is None:
        return None, "Upload an image first."
    _configure(speed, gauge, show_seg)
    PIPE.reset()
    PIPE.sm.escalate_frames = 1  # a single frame has no history to smooth over
    r = PIPE.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.0)
    PIPE.reset()
    PIPE.sm.escalate_frames = CFG["risk"]["escalate_frames"]
    rows = [f"| {a.hazard.label} | {a.hazard.distance_m:.0f} m | {a.hazard.lateral_m:+.1f} m | "
            f"{'yes' if a.in_path else 'no'} | **{a.level.name}** | {a.reason} |"
            for a in sorted(r.assessments, key=lambda a: a.hazard.distance_m)]
    md = (f"### {r.state.name}\nStopping distance at {speed:.0f} km/h: "
          f"**{r.info['stop_dist_m']:.0f} m**  ·  track {'found' if r.info['corridor'] else 'NOT found'}\n\n")
    if rows:
        md += "| object | distance | lateral | in path | risk | why |\n|---|---|---|---|---|---|\n"
        md += "\n".join(rows)
    else:
        md += "No hazards assessed."
    return cv2.cvtColor(r.image, cv2.COLOR_BGR2RGB), md


def run_video(video, speed, gauge, show_seg, max_seconds, progress=gr.Progress()):
    if video is None:
        return None, None, "Upload a video first."
    _configure(speed, gauge, show_seg)
    out = Path(tempfile.mkdtemp(prefix="railsight_"))
    res = process_video(PIPE, video, out, max_seconds or None, progress=progress)
    share = res["share_of_time"]
    md = (f"**{res['frames']} frames** processed at **{res['mean_fps']:.1f} fps** on average.\n\n"
          + "Time in each state: " + ", ".join(f"{k} {v:.0%}" for k, v in share.items()))
    return str(res["video"]), str(res["timeline"]), md


with gr.Blocks(title="RailSight") as demo:
    gr.Markdown("# RailSight\nRailway hazard detection and collision-risk estimation from a "
                "forward-facing train camera. Research prototype, not a safety system.")
    with gr.Row():
        speed = gr.Slider(10, 160, value=CFG["train"]["speed_kmh"], step=5, label="Train speed (km/h)")
        gauge = gr.Radio([1.435, 1.676, 1.0], value=CFG["track"]["gauge_m"],
                         label="Track gauge (m): standard / Indian broad / metre")
        show_seg = gr.Checkbox(False, label="Show segmentation")
    with gr.Tab("Video"):
        with gr.Row():
            vin = gr.Video(label="Cab-view video")
            vout = gr.Video(label="RailSight output")
        max_s = gr.Number(30, label="Process at most N seconds (0 = all)")
        vbtn = gr.Button("Run", variant="primary")
        timeline = gr.Image(label="Risk timeline")
        vmd = gr.Markdown()
        vbtn.click(run_video, [vin, speed, gauge, show_seg, max_s], [vout, timeline, vmd])
    with gr.Tab("Image"):
        with gr.Row():
            iin = gr.Image(label="Cab-view image")
            iout = gr.Image(label="RailSight output")
        ibtn = gr.Button("Run", variant="primary")
        imd = gr.Markdown()
        ibtn.click(run_image, [iin, speed, gauge, show_seg], [iout, imd])

if __name__ == "__main__":
    demo.queue().launch()
