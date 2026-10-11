# Tests

```bash
uv run pytest
```

| File | Covers |
|---|---|
| [`test_corridor.py`](test_corridor.py) | Corridor fitting on synthetic tracks: horizon, pinhole distance, lateral offset, occlusion gaps, ego-track choice, merged tracks at switches, a dashboard hiding the bottom of the view |
| [`test_risk.py`](test_risk.py) | Stopping distance, CRITICAL / WARNING / SAFE rules, objects crossing before the train arrives, hysteresis |
| [`test_metrics.py`](test_metrics.py) | IoU, average precision, recall at the operating confidence by object size |
| [`test_persistence.py`](test_persistence.py) | Temporal confirmation of unknown objects: confirmation, flicker rejection, behaviour between updates |

The tests need only numpy, OpenCV and PyYAML, so CI runs them without PyTorch or a GPU
([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)).
