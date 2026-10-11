"""Regenerate the README's Results section from the JSON files under reports/ and runs/.

Everything between `<!-- results:start -->` and `<!-- results:end -->` is replaced, so numbers
in the README always match the latest evaluation outputs.

    python scripts/make_readme_tables.py
"""

import json
from pathlib import Path

SEG = [("segformer-b2", "SegFormer-B2"), ("deeplab-r50", "DeepLabV3-ResNet50"),
       ("segformer-b0", "SegFormer-B0")]
DET = [("yolo11s COCO zero-shot", "YOLO11s, COCO zero-shot"), ("yolo11s_rail", "YOLO11s v1"),
       ("yolo11m_rail", "YOLO11m v1"), ("rtdetr_l_rail", "RT-DETR-l v1"),
       ("yolo11m_rail_v2", "YOLO11m v2")]
START, END = "<!-- results:start -->", "<!-- results:end -->"


def load(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(x):
    return f"{x:.1%}"


def segmentation(bench):
    lines = ["| Segmentation (RailSem19 test, 850 frames) | mIoU (19 classes) | track IoU | FPS |",
             "|---|---|---|---|"]
    for key, name in SEG:
        m = load(f"runs/seg/{key}/metrics_test.json")
        if m:
            fps = f"{bench[key]:.0f}" if key in bench else "–"
            lines.append(f"| {name} | {m['mIoU']:.3f} | {m['track_IoU']:.3f} | {fps} |")
    return "\n".join(lines)


def _recall(r):
    return " / ".join("–" if r.get(f"recall_{k}") is None else f"{r[f'recall_{k}']:.2f}"
                      for k in ("small(<32px)", "medium(32-96px)", "large(>96px)"))


def detection():
    rows = {r["model"]: r for r in load("reports/detection/results_test.json") or []}
    lines = ["| Detection, Roboflow test (268 images) | mAP50 (6 classes) "
             "| mAP50 (person / animal / vehicle) | mAP50-95 | recall small / medium / large |",
             "|---|---|---|---|---|"]
    for key, name in DET:
        r = rows.get(key)
        if r:
            lines.append(f"| {name} | {r['mAP50_all']:.3f} | {r['mAP50_shared']:.3f} | {r['mAP50-95']:.3f} "
                         f"| {_recall(r)} |")
    return "\n".join(lines)


def detection_cabview():
    rows = {r["model"]: r for r in load("reports/detection/results_test_det_v2_data_test_rs19.json") or []}
    if not rows:
        return ""
    lines = ["| Detection, real cab-view frames (RailSem19 test, 850 images) "
             "| mAP50 (person / vehicle / rail vehicle) | AP50 person | AP50 vehicle | AP50 rail vehicle "
             "| mAP50-95 | recall small / medium / large |", "|---|---|---|---|---|---|---|"]
    for key, name in DET:
        r = rows.get(key)
        if r:
            ap = [f"{r[f'AP50_{c}']:.2f}" if f"AP50_{c}" in r else "–"
                  for c in ("person", "vehicle", "rail_vehicle")]
            lines.append(f"| {name} | {r['mAP50_all']:.3f} | {' | '.join(ap)} | {r['mAP50-95']:.3f} "
                         f"| {_recall(r)} |")
    return "\n".join(lines)


def resolution():
    ref = {r["model"]: r for r in load("reports/detection/results_test_det_v2_data_test_rs19.json") or []}
    hi = {r["model"]: r for r in load("reports/detection/results_test_rs19_960_cmp_960px.json") or []}
    if "yolo11m_rail_v2" not in ref or not hi:
        return ""
    rows = [("YOLO11m v2, trained and run at 640 px", ref["yolo11m_rail_v2"]),
            ("YOLO11m v2, trained at 640, run at 960 px", hi.get("yolo11m_rail_v2")),
            ("**YOLO11m v2 fine-tuned at 960 px (default)**", hi.get("yolo11m_rail_v2_960"))]
    lines = ["| Input resolution (RailSem19 cab-view test) | mAP50 | mAP50-95 "
             "| AP50 person / vehicle / rail vehicle "
             "| recall small / medium / large |", "|---|---|---|---|---|"]
    for name, r in rows:
        if r:
            ap = " / ".join(f"{r[f'AP50_{c}']:.2f}" for c in ("person", "vehicle", "rail_vehicle"))
            lines.append(f"| {name} | {r['mAP50_all']:.3f} | {r['mAP50-95']:.3f} | {ap} | {_recall(r)} |")
    return "\n".join(lines)


def scene():
    have = [(k, n) for k, n in SEG if load(f"reports/scene_{k}/corridor_test.json")]
    cor = {k: load(f"reports/scene_{k}/corridor_test.json") for k, _ in have}
    an = {k: {r["variant"]: r for r in load(f"reports/scene_{k}/results_test.json")} for k, _ in have}

    def row(label, f):
        return f"| {label} | " + " | ".join(f(k) for k, _ in have) + " |"

    def unk(k, v):
        r = an[k][v]
        return f"{r['precision']:.2f} / {r['recall']:.2f} / {pct(r['false_alarm_rate'])}"

    return "\n".join([
        "| Corridor and unknown objects (RailSem19 test) | " + " | ".join(n for _, n in have) + " |",
        "|---" * (len(have) + 1) + "|",
        row("Corridor found (frames with a real track)", lambda k: pct(cor[k]["found_rate"])),
        row("Corridor IoU vs ground-truth corridor", lambda k: f"{cor[k]['mean_IoU']:.2f}"),
        row("Horizon error (median)", lambda k: f"{cor[k]['median_horizon_err_px']:.1f} px"),
        row("Ranging deviation from segmentation, 25 / 50 / 100 m", lambda k: " / ".join(
            pct(cor[k][f"median_rel_range_err_{d}m"]) for d in (25, 50, 100))),
        row("Unknown objects (class-only, default): precision / recall / false alarms",
            lambda k: unk(k, "class only")),
        row("Unknown objects + low-confidence pixels (<0.5)", lambda k: unk(k, "class + conf<0.5")),
    ])


def ranging():
    c = load("reports/ranging/consistency.json")
    if not c:
        return ""
    s = c["summary"]
    lvl, oth = s["feet_at_track_level"], s["elsewhere (platforms etc.)"]
    return "\n".join([
        "| Ranging check: gauge distance ÷ human-height distance (RailSem19 people) | people | median ratio "
        "| robust spread |", "|---|---|---|---|",
        f"| Feet at track level | {lvl['people']} | {lvl['median_ratio_gauge_over_height']:.3f} | "
        f"±{lvl['robust_spread_pct']:.0f}% |",
        f"| On platforms / elsewhere | {oth['people']} | {oth['median_ratio_gauge_over_height']:.3f} | "
        f"±{oth['robust_spread_pct']:.0f}% |",
    ])


def speed(bench_rows):
    b = {r["model"]: r["fps"] for r in bench_rows}
    one, three = (b.get(f"default config, segmentation every {n} frame(s)") for n in (1, 3))
    lines = ["| Speed (RTX 5060 Laptop GPU, 960x540) | FPS |", "|---|---|"]
    if one:
        lines.append(f"| Full pipeline, segmentation every frame | {one:.1f} |")
    if three:
        lines.append(f"| Full pipeline, segmentation every 3rd frame (default) | {three:.1f} |")
    dets = [f"{k}: {v:.0f}" for k, v in b.items() if k.endswith("px")]
    if dets:
        lines.append(f"| Detectors alone (fp16) | {', '.join(dets)} |")
    return "\n".join(lines)


def main():
    bench_rows = load("reports/benchmark/benchmark.json") or []
    bench = {r["model"]: r["fps"] for r in bench_rows if r["stage"] == "segmentation"}
    full = {r["model"]: r["fps"] for r in bench_rows if r["stage"] == "full pipeline"}
    full3 = full.get("default config, segmentation every 3 frame(s)", float("nan"))
    body = f"""{START}
_Generated by `scripts/make_readme_tables.py` from `reports/` and `runs/`._

{segmentation(bench)}

{detection()}

{detection_cabview()}

Both tables above are at 640 px. v1 detectors were fine-tuned on the merged Roboflow sets only.
They score well on the Roboflow test split but **transfer poorly to real cab-view footage**: on
RailSem19 frames they fall below even the un-tuned COCO model, because the Roboflow images are
mostly close-up, staged scenes. **v2** adds ~16k training boxes derived from RailSem19's dense masks
(person, vehicle and a new `rail_vehicle` class, filtered against mask noise) and is about 7x
better than v1 on cab-view frames while also slightly better on the Roboflow split, so v2 is used
(fine-tuned at 960 px, below). RT-DETR-l v1 (trained on Kaggle, 2x T4) has the highest Roboflow
recall but is the slowest. Recall is measured at the operating confidence (0.25). Object size is
sqrt(box area) in original image pixels (Roboflow 640x640, RailSem19 1280x720), so a "small"
RailSem19 object is under 16 px at a 640 px detector input. RailSem19-derived labels come from
weakly supervised masks, and the Roboflow test has ~270 images, so small differences are within
noise.

{resolution()}

Most of the higher-resolution gain comes from running the detector at 960 px; fine-tuning at 960 px
adds ~1.5 points of mAP50 but no small-object recall. Distant objects (small recall ~0.25) remain the
main limitation of single-camera detection. The 960 px model is the default: on this GPU it costs
less than 1 FPS in the full pipeline.

{scene()}

SegFormer-B2 and DeepLabV3 score the same track IoU, yet B2's corridor geometry is clearly better
(about 20% lower ranging deviation): pixel overlap does not capture how cleanly the track edges
follow the rails at long range, which is what ranging depends on. B2 is the default model.

{ranging()}

The focal length cancels in this ratio, so the check needs no camera calibration. For people
standing at rail level, gauge ranging shows no scale bias; the spread is mostly the reference
itself (body height, posture, mask outlines). People on platforms stand above the rail plane, so
the flat-ground assumption underestimates their distance, as expected.

{speed(bench_rows)}

In back-to-back measurements, profiling-driven optimisation (fused SDPA attention, bf16, GPU-side
normalisation, fp16 detector) raised SegFormer-B2 from 14.8 to 21.0 FPS (+41%) and SegFormer-B0
from 22.3 to 47.0 FPS with unchanged predictions (>99.9% pixel agreement, identical detector test
mAP). Absolute FPS on a laptop depends on its power and thermal state: the same pipeline measured
between 17.7 and {full3:.1f} FPS across sessions; the table shows the latest run on an idle machine.
{END}"""

    readme = Path("README.md")
    text = readme.read_text(encoding="utf-8").replace("\r\n", "\n")
    if START in text:
        head, rest = text.split(START, 1)
        tail = rest.split(END, 1)[1]
    else:  # first run: replace everything from "## Results" up to the next "---" separator
        head, rest = text.split("## Results\n", 1)
        head += "## Results\n\n"
        tail = "\n" + rest[rest.index("\n---\n"):]
    readme.write_text(head + body + tail, encoding="utf-8")
    print("README results section regenerated")

if __name__ == "__main__":
    main()
