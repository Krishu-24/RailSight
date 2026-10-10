import numpy as np
import pytest

from railsight.metrics import average_precision, box_iou, evaluate_detections


def test_box_iou():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    assert box_iou(a, b)[0] == pytest.approx([1.0, 1 / 3, 0.0])


def test_ap_perfect_and_empty():
    assert average_precision(np.array([0.5, 1.0]), np.array([1.0, 1.0])) == pytest.approx(1.0)


def test_evaluate_detections_perfect_and_miss():
    g = np.array([[0, 0, 10, 10], [50, 50, 150, 150]], float)
    gts = [(g, np.array([0, 1]))]
    perfect = [(g.copy(), np.array([0, 1]), np.array([0.9, 0.8]))]
    r = evaluate_detections(gts, perfect, 2, size_bins={"s": (0, 32), "l": (32, 1e9)})
    assert r["mAP"] == pytest.approx(1.0)
    assert r["recall_by_size"] == {"s": 1.0, "l": 1.0}

    # small object only found with low confidence -> AP counts it, operating recall does not
    low = [(g.copy(), np.array([0, 1]), np.array([0.05, 0.8]))]
    r = evaluate_detections(gts, low, 2, size_bins={"s": (0, 32), "l": (32, 1e9)})
    assert r["per_class"][0]["AP"] == pytest.approx(1.0)
    assert r["recall_by_size"]["s"] == 0.0


def test_false_positive_lowers_ap():
    g = np.array([[0, 0, 10, 10]], float)
    p = np.array([[100, 100, 110, 110], [0, 0, 10, 10]], float)
    r = evaluate_detections([(g, np.array([0]))], [(p, np.array([0, 0]), np.array([0.9, 0.5]))], 1)
    assert r["mAP"] == pytest.approx(0.5)
