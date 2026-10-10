# RailSight

**Vision-based railway hazard detection and collision-risk estimation from a forward-facing train camera.**

RailSight does more than detect objects. For every hazard it answers the two questions that matter to a
train, which cannot steer around an obstacle:

1. **Is it in the train's path?** A segmentation model finds the ego track and builds a metric
   *danger corridor* (the space the train will sweep through).
2. **Can the train still stop?** The rail gauge acts as a known-size ruler, which turns pixels into
   metres; the distance is compared with the braking distance at the current speed.

The result is an interpretable **SAFE / WARNING / CRITICAL** state per frame.

> Academic prototype (MAHE, School of Computer Engineering) for perception and decision support.
> It is **not** a certified signalling or braking system.

---

## How it works

```
frame ─► SegFormer (RailSem19, 19 classes) ─► ego track ─► corridor fit ─► metric danger corridor
     │                                                          │  (gauge → px-per-metre per row)
     ├─► YOLO11 / RT-DETR (fine-tuned) ─► ByteTrack ──────────┐ │
     └─► foreign-pixel check inside the track (class-agnostic) ┤ │
                                                               ▼ ▼
            distance (pinhole + gauge) · lateral offset · lateral speed (from track history)
                                                               │
       speed + reaction time + braking ─► rule-based risk ─► hysteresis ─► SAFE / WARNING / CRITICAL
```

| Stage | Method | Trained by us? |
|---|---|---|
| Rail-scene segmentation | SegFormer-B2 (vs B0, DeepLabV3-ResNet50) | ✅ fine-tuned on RailSem19 |
| Obstacle detection | YOLO11s / YOLO11m / RT-DETR-l | ✅ fine-tuned on merged railway-obstacle data |
| Danger corridor | Row-wise ego-track tracing + robust width/centre fits | geometry |
| Distance | `Z = f · gauge / projected_gauge(row)` | geometry |
| Unknown objects | Non-ground or low-confidence pixels inside the track | uses the segmentation model |
| Tracking | ByteTrack | off-the-shelf |
| Risk | Stopping distance `v·t_r + v²/2a`, in-path test over an arrival time window, hysteresis | rules |

### Why rail-gauge ranging
On flat track the projected rail gauge shrinks linearly towards the horizon. Fitting
`width(y) = a·y + b` to the segmented track gives the horizon row and a pixels-per-metre scale at
every image row. With gauge 1.435 m (standard) or 1.676 m (Indian broad gauge), the distance of
any object touching the ground at row `y` is `Z = f · G / width(y)`. The same scale widens the
corridor to the real train width (3.2 m + margin) regardless of how far away a row is.

### Why the risk is rule-based
No dataset labels "this frame was dangerous", and a safety decision should be explainable, so the
learned parts handle perception and the decision is made by transparent physics.

---

## Data

| Dataset | Used for | Notes |
|---|---|---|
| [RailSem19](https://wilddash.cc/railsem19) | Segmentation | 8,500 frames, 19 classes; fixed 80/10/10 split (seed 0) |
| [railway (goal)](https://universe.roboflow.com/goal-lmqiu/railway-tfabb) | Detection | MIT; 11 classes |
| [Railway track obstacle detection](https://universe.roboflow.com/railway-research/railway-track-obstacle-detection-gca3w) | Detection | CC BY 4.0; 6 classes |

**Detection data cleaning** (`scripts/prepare_detection.py`), based on an audit of the raw exports:
- Classes are mapped onto six hazards: `person, animal, vehicle, rock, vegetation, obstacle`.
- `Plant` (74% of boxes in one source: weeds between sleepers seen by a top-down inspection
  camera), standing `Tree`s and `flood` are not obstacles in the train's path, so they were dropped.
- The original splits **leaked**: Roboflow's augmented copies and consecutive video frames appeared
  in both train and test. Images are grouped by source filename and perceptual hash (dHash), and
  whole groups are re-split 80/10/10 with stratification on the rarest class. Val/test contain no
  augmented copies.

---

## Results

_Filled in from `reports/` after the training runs finish._

| Segmentation (RailSem19 test) | mIoU | track IoU | FPS |
|---|---|---|---|
| SegFormer-B2 | | | |
| SegFormer-B0 | | | |
| DeepLabV3-ResNet50 | | | |

| Detection (held-out test) | mAP50 | mAP50-95 | recall small / medium / large |
|---|---|---|---|
| YOLO11s COCO zero-shot | | | |
| YOLO11s fine-tuned | | | |
| YOLO11m fine-tuned | | | |
| RT-DETR-l fine-tuned | | | |

---

## Setup

Requires an NVIDIA GPU. The lockfile pins PyTorch built for CUDA 12.8, which RTX 50-series cards need.

```bash
uv sync
uv run pytest
```

## Reproduce

```bash
# data (place raw downloads under data/raw/, see the table above)
uv run python scripts/prepare_detection.py
uv run python scripts/prepare_railsem19.py

# every training run, one GPU job at a time (pass a step number to resume)
bash scripts/train_all.sh

# evaluation tables -> reports/
uv run python scripts/eval_det.py
uv run python scripts/train_seg.py --arch segformer-b2 --eval-only --split test
```

## Run

```bash
uv run python scripts/run_video.py data/videos/clip.mp4 --speed 80   # annotated video + log + timeline
uv run python app/gradio_app.py                                      # interactive demo
```

Settings such as gauge, camera field of view, braking deceleration and margins are in
[`configs/default.yaml`](configs/default.yaml).

## Repository layout

```
railsight/   corridor.py  risk.py  seg.py  anomaly.py  pipeline.py  video.py  metrics.py  datasets.py
scripts/     prepare_*.py  train_*.py  eval_det.py  run_video.py  train_all.sh
app/         gradio_app.py
configs/     default.yaml
tests/       geometry, risk logic and metric unit tests
```

## Limitations

- Gauge ranging assumes locally flat track and a known (or assumed) focal length; error grows with
  distance as the projected gauge shrinks to a few pixels.
- At 100 km/h the stopping distance is about 440 m, where a person is only a few pixels tall for a
  standard camera. Monocular perception mainly helps at lower speeds and in yards and at crossings.
- Switches: the ego path follows the branch closest to the previous row, which is not necessarily
  the route the train will take.

## Team

Arnav Keshav Sinha · Faizaan Naiman Naveed Nasir. Guide: Dr. Jayita Saha.
