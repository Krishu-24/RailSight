"""Regenerate the README's Results section from the JSON files under reports/ and runs/.

Everything between `<!-- results:start -->` and `<!-- results:end -->` is replaced, so numbers
in the README always match the latest evaluation outputs.

    python scripts/make_readme_tables.py
"""

import json
from pathlib import Path

SEG = [("segformer-b2", "SegFormer-B2"), ("deeplab-r50", "DeepLabV3-ResNet50"),
       ("segformer-b0", "SegFormer-B0")]
DET = [("yolo11s COCO zero-shot", "YOLO11s, COCO zero-shot"), ("yolo11s_rail", "YOLO11s fine-tuned"),
       ("yolo11m_rail", "YOLO11m fine-tuned"), ("rtdetr_l_rail", "RT-DETR-l fine-tuned")]
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


def detection():
    rows = {r["model"]: r for r in load("reports/detection/results_test.json") or []}
    lines = ["| Detection (Roboflow test, 268 images) | mAP50 (6 classes) "
             "| mAP50 (person / animal / vehicle) "
             "| mAP50-95 | recall small / medium / large |", "|---|---|---|---|---|"]
    for key, name in DET:
        r = rows.get(key)
        if not r:
            continue
        m5095 = r.get("mAP50-95 (ultralytics)")
        lines.append(f"| {name} | {r['mAP50_all']:.3f} | {r['mAP50_shared']:.3f} | "
                     f"{'–' if m5095 is None else f'{m5095:.3f}'} | "
                     f"{r['recall_small(<32px)']:.2f} / {r['recall_medium(32-96px)']:.2f} / "
                     f"{r['recall_large(>96px)']:.2f} |")
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
    dets = [f"{k}: {v:.0f}" for k, v in b.items() if k.endswith("_rail")]
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

Recall is measured at the operating confidence (0.25); object size is sqrt(box area) at 640 px.
The COCO model has no rock / vegetation / obstacle classes, the main gap fine-tuning closes.
RT-DETR-l (trained on Kaggle, 2x T4) finds the most objects but is the slowest; YOLO11m matches
its box precision (mAP50-95). With ~270 test images, differences of a few points are within noise.

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

Profiling-driven optimisation (fused SDPA attention, bf16, GPU-side normalisation, fp16 detector)
raised SegFormer-B2 from 14.8 to {bench.get('segformer-b2', float('nan')):.1f} FPS and the full pipeline from
15.0 to {full3:.1f} FPS, with unchanged predictions (>99.9% pixel agreement, identical detector test mAP).
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
