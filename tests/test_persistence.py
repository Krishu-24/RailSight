import numpy as np

from railsight.anomaly import Blob
from railsight.pipeline import RailSightPipeline


def blob(x, y, s=20):
    return Blob((x, y, x + s, y + s), s * s, np.zeros((1, 1, 2), np.int32))


def pipe():
    p = RailSightPipeline.__new__(RailSightPipeline)  # logic only: no models loaded
    p._blob_tracks = []
    return p


def test_unknown_object_confirmed_after_three_updates():
    p = pipe()
    seen = [p._persistent([blob(100 + 2 * k, 200)], True, 3) for k in range(3)]
    assert [len(s) for s in seen] == [0, 0, 1]  # slight motion between updates is tolerated


def test_flicker_is_dropped():
    p = pipe()
    p._persistent([blob(100, 200)], True, 3)
    p._persistent([], True, 3)  # gone for one update: the count restarts
    assert p._persistent([blob(100, 200)], True, 3) == []


def test_between_updates_the_confirmed_set_is_kept():
    p = pipe()
    for _ in range(3):
        p._persistent([blob(50, 50)], True, 3)
    assert len(p._persistent([blob(52, 50)], False, 3)) == 1
