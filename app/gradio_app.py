"""RailSight interactive demo.

    uv run railsight demo          (or: uv run python app/gradio_app.py)
"""

import tempfile
from pathlib import Path

import cv2
import gradio as gr

from railsight.config import ROOT, load_config
from railsight.pipeline import RailSightPipeline
from railsight.video import process_video

CFG = load_config()
PIPE = RailSightPipeline(CFG)
EXAMPLES = sorted(str(p) for p in (ROOT / "data" / "videos").glob("*.mp4"))

INTRO = f"""
# RailSight
Railway hazard detection and collision-risk estimation from a forward-facing train camera.
For every hazard: **is it in the train's path**, and **can the train still stop?**

Models: segmentation `{PIPE.seg_name}`, detection `{PIPE.det_name}` @ {CFG['models']['det_imgsz']} px.
_Research prototype for decision support, not a certified safety system._
"""

HELP = """
**Reading the output.** The shaded corridor is the space the train will sweep through; its colour
is the current risk state. White ticks mark distance along the track; the red line is where the train
could stop at the current speed. Boxes are detected objects with their distance; magenta outlines are
unknown objects inside the track. **Camera settings:** the field of view sets the distance scale
(phones and action cameras are often 65 to 120 degrees); *ignore below* masks a dashboard or hood that
the camera sees.
"""


def _configure(speed, gauge, show_seg, hfov, ignore_below):
    PIPE.speed_kmh = speed
    PIPE.cfg["camera"]["hfov_deg"] = hfov
    PIPE.cfg["camera"]["ignore_below"] = ignore_below if ignore_below < 1.0 else None
    PIPE.cfg["track"]["gauge_m"] = gauge
    PIPE.show_seg = show_seg


def run_image(image, speed, gauge, show_seg, hfov, ignore_below):
    if image is None:
        return None, "Upload an image first."
    _configure(speed, gauge, show_seg, hfov, ignore_below)
    PIPE.reset()
    PIPE.sm.escalate_frames = 1  # a single frame has no history to smooth over
    PIPE.persistence.need = 1    # nor can unknown objects persist over time
    r = PIPE.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.0)
    PIPE.reset()
    PIPE.sm.escalate_frames = CFG["risk"]["escalate_frames"]
    rows = [f"| {a.hazard.label} | {a.hazard.distance_m:.0f} m | {a.hazard.lateral_m:+.1f} m | "
            f"{'yes' if a.in_path else 'no'} | **{a.level.name}** | {a.reason} |"
            for a in sorted(r.assessments, key=lambda a: a.hazard.distance_m)]
    track = ("NOT found" if not r.info["corridor"] else
             "found, low confidence" if r.info["low_confidence"] else "found")
    md = (f"### {r.state.name}\nStopping distance at {speed:.0f} km/h: "
          f"**{r.info['stop_dist_m']:.0f} m**  ·  track {track}\n\n")
    if rows:
        md += "| object | distance | lateral | in path | risk | why |\n|---|---|---|---|---|---|\n"
        md += "\n".join(rows)
    else:
        md += "No hazards assessed."
    return cv2.cvtColor(r.image, cv2.COLOR_BGR2RGB), md


def run_video(video, speed, gauge, show_seg, hfov, ignore_below, max_seconds, progress=gr.Progress()):
    if video is None:
        return None, None, "Upload a video first."
    _configure(speed, gauge, show_seg, hfov, ignore_below)
    out = Path(tempfile.mkdtemp(prefix="railsight_"))
    res = process_video(PIPE, video, out, max_seconds or None, progress=progress)
    share = res["share_of_time"]
    md = (f"**{res['frames']} frames** processed at **{res['mean_fps']:.1f} fps** on average.\n\n"
          + "Time in each state: " + ", ".join(f"{k} {v:.0%}" for k, v in share.items()))
    return str(res["video"]), str(res["timeline"]), md


with gr.Blocks(title="RailSight") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        speed = gr.Slider(10, 160, value=CFG["train"]["speed_kmh"], step=5, label="Train speed (km/h)")
        gauge = gr.Radio([1.435, 1.676, 1.0], value=CFG["track"]["gauge_m"],
                         label="Track gauge (m): standard / Indian broad / metre")
    with gr.Row():
        hfov = gr.Slider(30, 130, value=CFG["camera"]["hfov_deg"], step=5,
                         label="Camera field of view (deg): sets the distance scale")
        ignore_below = gr.Slider(0.5, 1.0, value=CFG["camera"].get("ignore_below") or 1.0, step=0.01,
                                 label="Ignore below (fraction of height; 1.0 = off): dashboard / hood")
        show_seg = gr.Checkbox(False, label="Show segmentation")
    controls = [speed, gauge, show_seg, hfov, ignore_below]
    with gr.Tab("Video"):
        with gr.Row():
            vin = gr.Video(label="Cab-view video")
            vout = gr.Video(label="RailSight output")
        max_s = gr.Number(30, label="Process at most N seconds (0 = all)")
        vbtn = gr.Button("Run", variant="primary")
        if EXAMPLES:
            gr.Examples(EXAMPLES, inputs=vin, label="Demo clips (data/videos)")
        timeline = gr.Image(label="Risk timeline")
        vmd = gr.Markdown()
        vbtn.click(run_video, [vin, *controls, max_s], [vout, timeline, vmd])
    with gr.Tab("Image"):
        with gr.Row():
            iin = gr.Image(label="Cab-view image")
            iout = gr.Image(label="RailSight output")
        ibtn = gr.Button("Run", variant="primary")
        imd = gr.Markdown()
        ibtn.click(run_image, [iin, *controls], [iout, imd])
    gr.Markdown(HELP)

if __name__ == "__main__":
    demo.queue().launch()
