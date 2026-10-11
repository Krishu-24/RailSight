# Reports

Every number quoted in the README and the project report comes from a file here. Each evaluation
writes a machine-readable `.json` and a human-readable `.md`; the README's Results section is
regenerated from these files by [`make_readme_tables.py`](../scripts/tools/make_readme_tables.py).

| Folder | Produced by | Contents |
|---|---|---|
| [`detection/`](detection) | [`eval_det.py`](../scripts/eval/eval_det.py) | `roboflow_test` — Roboflow test split (268 images); `cabview_test` — RailSem19 cab-view test frames (850); `*_960px` — the same at 960 px inference |
| `scene_<model>/` | [`eval_scene.py`](../scripts/eval/eval_scene.py) | `corridor_test` — corridor found rate, IoU, horizon and ranging deviation; `unknown_objects_test` — precision / recall / false alarms of the unknown-object variants |
| [`ranging/`](ranging) | [`eval_ranging_consistency.py`](../scripts/eval/eval_ranging_consistency.py) | Gauge vs human-height distance per person, summary and figure |
| [`benchmark/`](benchmark) | [`benchmark.py`](../scripts/eval/benchmark.py) | FPS of every model and the full pipeline; `benchmark_before_optimisation.md` is the pre-optimisation run |

Notes for reading them:

- **Recall** is measured at the operating confidence (0.25). **Object sizes** (small < 32 px,
  medium 32–96 px, large > 96 px, as sqrt of box area) are in original image pixels: 640×640 for
  Roboflow, 1280×720 for RailSem19.
- **mAP50-95** in `detection/` is computed by the project's own evaluator so every model is scored
  identically; Ultralytics' value is added as a cross-check when a model's classes match the dataset.
- RailSem19-derived detection labels come from weakly supervised masks; the Roboflow test set is
  small, so differences of a few points are within noise.
- Model weights, datasets and cached predictions are not in the repository (size and the RailSem19
  licence).
