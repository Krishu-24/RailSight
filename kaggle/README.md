# Training the larger detectors on Kaggle

Kaggle's free GPUs (2× T4, 30 GPU-hours/week) train **YOLO11m** and **RT-DETR-l** while the laptop
trains the segmentation models. Only the detection dataset goes to Kaggle (Roboflow sources,
CC BY 4.0 / MIT). **RailSem19 must not be uploaded**: its licence forbids sharing the frames.

## One-time setup

1. Create an account at <https://www.kaggle.com> and **verify your phone number**
   (Settings → Phone verification). Without it, GPUs and internet access stay disabled.
2. **Create the dataset:** *Create → New Dataset*
   - Upload `data/kaggle/railsight-det.zip` (391 MB). Kaggle unzips it automatically.
   - Title: `railsight-det`
   - Visibility: **Private**
3. **Create the notebook:** *Create → New Notebook → File → Import Notebook*, then pick
   `kaggle/train_detector.ipynb` from this repository.
4. In the notebook's right-hand panel:
   - *Input → Add Input → Your Datasets →* `railsight-det`
   - *Session options → Accelerator:* **GPU T4 x2**
   - *Session options → Internet:* **On**

## Run 1: YOLO11m

The first code cell already says `MODEL = "yolo11m.pt"`, `NAME = "yolo11m_rail"`, `BATCH = 32`.

Click **Save Version → Save & Run All (Commit) → Save**. It runs in the background, so you can
close the tab. Expect about 2–4 hours. Progress: *View Active Events* (bell icon) or the version's log.

## Run 2: RT-DETR-l

Edit the first code cell to:

```python
MODEL = "rtdetr-l.pt"
NAME = "rtdetr_l_rail"
BATCH = 16
```

and **Save Version → Save & Run All** again. Free accounts may only run one GPU session at a
time, in which case start this after run 1 finishes.

RT-DETR is slower. Kaggle stops sessions at 12 hours, so check the first epoch's duration in the
log: if one epoch takes more than about 6 minutes, stop the run, set `EPOCHS` so that
`EPOCHS × epoch time` stays under ~10 hours, and start again. Record the epoch count used.

**Do not use Ultralytics' `time=` argument as a safety budget.** It re-plans the number of epochs
to fill the budget, which stretches the cosine learning-rate schedule. In our first YOLO11m run
the learning rate stayed near 1e-3 for all 102 epochs instead of decaying (val mAP50 0.715). That
run is kept in `runs/archive/` and was retrained without `time=`: the learning rate then decayed
normally (to 3.4e-5), yet val mAP50 was 0.695, still below YOLO11s (0.739). On the held-out
test split the order flips (YOLO11m 0.695 vs YOLO11s 0.667 mAP50): with ~270-image evaluation sets,
differences of a few points are within noise, so the larger model's gain is small and uncertain.

## Bringing results back

1. Open the finished version → **Output** tab → download `yolo11m_rail.zip` (or `rtdetr_l_rail.zip`).
2. Put the zip in the laptop's `RailSight/data/incoming/`.
3. Tell Claude. The zip is unpacked into `runs/det/<name>/`, where `scripts/eval_det.py` picks it up.

## How the laptop avoids duplicate work

`runs/det/yolo11m_rail/REMOTE` and `runs/det/rtdetr_l_rail/REMOTE` tell `scripts/train_det.py`
to skip these runs. If a Kaggle run fails, delete the marker and run on the laptop:

```bash
uv run python scripts/train_det.py --model rtdetr-l.pt --name rtdetr_l_rail --batch 0.7
```

## Differences from laptop training (for the report)

| | Laptop | Kaggle |
|---|---|---|
| GPU | 1× RTX 5060 Laptop (8 GB) | 2× Tesla T4 (16 GB each), DDP |
| Batch | AutoBatch (70% of VRAM) | 32 (YOLO11m) / 16 (RT-DETR-l) |
| Precision | AMP | AMP (fp16) |
| Everything else | same data, splits, epochs, patience, schedule, seed | same |
