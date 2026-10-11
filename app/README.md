# Interactive demo

```bash
uv run railsight demo          # or: uv run python app/gradio_app.py
```

Opens a local Gradio app (http://127.0.0.1:7860) with two tabs:

- **Video** — upload a cab-view clip (or pick one of the demo clips in `data/videos/`), get back the
  annotated video, a risk timeline and the share of time spent in each state.
- **Image** — a single frame with a table of every hazard: distance, lateral offset, whether it is in
  the path, risk level and the reason.

## Controls

| Control | Effect |
|---|---|
| Train speed | Sets the stopping distance the risk is judged against |
| Track gauge | 1.435 m standard, 1.676 m Indian broad gauge, 1.0 m metre gauge; sets the distance scale |
| Camera field of view | The focal length used for ranging; phones and action cameras are often 65 to 120 degrees |
| Ignore below | Masks the part of the image that shows the train itself (dashboard, hood); 1.0 = off |
| Show segmentation | Blends the 19-class segmentation into the output |

## Reading the output

The shaded corridor is the space the train will sweep through; its colour is the current state
(green SAFE, amber WARNING, red CRITICAL). White ticks mark distance along the track and the red line
marks where the train could stop at the current speed. Boxes are detected objects with their distance;
magenta outlines are unknown objects found inside the track. The panel shows speed, stopping distance,
the nearest hazard in the path, whether the track is locked (or low confidence) and the frame rate.

Needs the trained weights (see the main README, *Quick start*).
