# Scripts

Run everything from the repository root with `uv run python scripts/<group>/<script>.py`. Each script
documents its options in its docstring and `--help`.

## `data/` — prepare datasets

| Script | What it does |
|---|---|
| [`prepare_railsem19.py`](data/prepare_railsem19.py) | Verifies RailSem19's class list, resizes to 1280×720 and writes a fixed 80/10/10 split |
| [`prepare_detection.py`](data/prepare_detection.py) | Detection v1: merges and cleans the two Roboflow sets, groups near-duplicates by perceptual hash and re-splits without leakage |
| [`prepare_detection_v2.py`](data/prepare_detection_v2.py) | Detection v2: adds boxes mined from RailSem19 masks (person, vehicle, `rail_vehicle`) with noise filters |
| [`fetch_demo_videos.py`](data/fetch_demo_videos.py) | Downloads the free Pexels demo clips to `data/videos/` and normalises them |

## `train/` — train models

| Script | What it does |
|---|---|
| [`train_seg.py`](train/train_seg.py) | Segmentation training (SegFormer B0/B1/B2, DeepLabV3); resumable; `--eval-only` for test metrics |
| [`train_det.py`](train/train_det.py) | Detector training or fine-tuning (YOLO11, RT-DETR); resumable; skips runs marked as trained on Kaggle |
| [`train_all.sh`](train/train_all.sh) | Every run behind the report, one GPU job at a time, with retries |

## `eval/` — measure

| Script | Writes | What it measures |
|---|---|---|
| [`eval_det.py`](eval/eval_det.py) | `reports/detection/` | Every detector on a test set with one evaluator: mAP, per-class AP, recall by object size |
| [`eval_scene.py`](eval/eval_scene.py) | `reports/scene_<model>/` | Corridor accuracy (found rate, IoU, horizon, ranging deviation) and unknown-object variants |
| [`eval_ranging_consistency.py`](eval/eval_ranging_consistency.py) | `reports/ranging/` | Gauge distance vs human-height distance, a calibration-free ranging check |
| [`benchmark.py`](eval/benchmark.py) | `reports/benchmark/` | FPS of every model and of the full pipeline |

## `tools/` — outputs and packaging

| Script | What it does |
|---|---|
| [`make_readme_tables.py`](tools/make_readme_tables.py) | Regenerates the README's Results section from `reports/` |
| [`make_demo_gif.py`](tools/make_demo_gif.py) | Builds `docs/assets/demo.gif` from rendered demo videos |
| [`plot_stopping.py`](tools/plot_stopping.py) | Stopping-distance vs speed figure |
| [`package_weights.py`](tools/package_weights.py) | Zips the trained weights with checksums for private sharing |

## Typical order

```bash
# 1. data (raw downloads in data/raw/)
uv run python scripts/data/prepare_railsem19.py
uv run python scripts/data/prepare_detection.py
uv run python scripts/data/prepare_detection_v2.py

# 2. training
bash scripts/train/train_all.sh

# 3. evaluation
uv run python scripts/train/train_seg.py --arch segformer-b2 --eval-only --split test
uv run python scripts/eval/eval_scene.py --weights runs/seg/segformer-b2/best --split test
uv run python scripts/eval/eval_det.py
uv run python scripts/eval/eval_det.py --data data/processed/det_v2/data_test_rs19.yaml
uv run python scripts/eval/eval_ranging_consistency.py
uv run python scripts/eval/benchmark.py

# 4. outputs
uv run railsight video "data/videos/*.mp4" --out outputs/final
uv run python scripts/tools/make_demo_gif.py
uv run python scripts/tools/make_readme_tables.py
```

Note: never run an evaluation on the GPU while a training job is using it on an 8 GB card; pass
`--device cpu` (and hide the GPU with `CUDA_VISIBLE_DEVICES=-1`) instead.
