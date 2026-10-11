# Configuration

[`default.yaml`](default.yaml) holds every run-time setting of the pipeline and the demo. Pass another
file with `railsight video … --config my.yaml`; the CLI flags `--speed`, `--hfov` and `--ignore-below`
override single values.

## `models`

| Key | Default | Meaning |
|---|---|---|
| `seg_weights` | `runs/seg/segformer-b2/best` | Segmentation model folder |
| `det_weights` | `runs/det/yolo11m_rail_v2_960/weights/best.pt` | Detector weights |
| `device` | `cuda` | Falls back to CPU when no GPU is present |
| `seg_precision` | `bf16` | `bf16`, `fp16` or `fp32`; fp32 is fastest for the tiny SegFormer-B0 |
| `det_conf`, `det_imgsz`, `det_half` | 0.30, 960, true | Detector confidence threshold, input size, fp16 |

## `camera` — set once per camera mounting

| Key | Default | Meaning |
|---|---|---|
| `hfov_deg` | 60 | Horizontal field of view; sets the focal length and therefore the distance scale |
| `ignore_below` | off | Fraction of image height below which the image shows the train itself |

## `track`

| Key | Default | Meaning |
|---|---|---|
| `gauge_m` | 1.435 | Rail gauge: 1.435 standard, 1.676 Indian broad gauge |
| `rail_head_m` | 0.072 | Added to the gauge because the track mask spans the outer rail edges |
| `train_width_m`, `side_margin_m` | 3.2, 0.5 | Width of the danger corridor (envelope + clearance each side) |
| `seg_every_n` | 3 | Segment every N-th frame; the corridor changes slowly |
| `smoothing` | 0.6 | Weight of the previous corridor fit (exponential moving average) |
| `rail_pairs` | false | Experimental gauge measurement between the two rails (mixed results) |

## `train` and `risk`

| Key | Default | Meaning |
|---|---|---|
| `speed_kmh` | 60 | Train speed (the demo has a slider) |
| `reaction_s`, `decel_mps2` | 2.0, 1.0 | Reaction time and braking deceleration: stopping distance = v·t + v²/2a |
| `stop_margin_m` | 50 | Safety buffer added to the stopping distance |
| `max_range_m` | 1000 | Objects estimated further away are ignored |
| `near_buffer_m` | 2.0 | Lateral zone beside the corridor that counts as "near" (WARNING) |
| `escalate_frames`, `deescalate_frames` | 3, 15 | Hysteresis: frames needed to raise / lower the alert |

## `anomaly` — unknown objects inside the track

| Key | Default | Meaning |
|---|---|---|
| `enabled` | true | Turn the check on or off |
| `min_track_conf` | 0.70 | Below this segmentation confidence on the track (darkness, glare) the check is skipped and the HUD shows LOW CONFIDENCE |
| `half_width_m` | 1.0 | Search only the track bed |
| `persist_updates` | 3 | Consecutive segmentation updates an object must survive (~0.3 s) |
| `min_size_m`, `min_area_px` | 0.3, 30 | Minimum physical width at the object's distance, and a pixel floor |
| `conf_threshold` | 0.0 | 0 = only non-track classes count; adding low-confidence pixels quadrupled false alarms |
