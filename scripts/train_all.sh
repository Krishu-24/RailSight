#!/usr/bin/env bash
# Every training run behind the report, in priority order (one GPU job at a time).
# Usage: bash scripts/train_all.sh [first_step]
# Batch sizes are chosen to fit an 8 GB GPU (SegFormer-B2 at batch 4 x 640 px peaks at ~6.9 GB).
set -e
cd "$(dirname "$0")/.."
steps=(
  "uv run python scripts/train_det.py --model yolo11s.pt  --name yolo11s_rail  --batch 16"
  "uv run python scripts/train_seg.py --arch segformer-b2 --iters 60000 --batch 4 --crop 640 --resume"
  "uv run python scripts/train_seg.py --arch deeplab-r50  --iters 60000 --batch 4 --crop 640 --resume"
  "uv run python scripts/train_seg.py --arch segformer-b0 --iters 60000 --batch 4 --crop 640 --resume"
  "uv run python scripts/train_det.py --model yolo11m.pt  --name yolo11m_rail  --batch 0.7"
  "uv run python scripts/train_det.py --model rtdetr-l.pt --name rtdetr_l_rail --batch 0.7"
)
for ((i=${1:-0}; i<${#steps[@]}; i++)); do
  echo "=== step $i: ${steps[$i]}"
  eval "${steps[$i]}"
done
