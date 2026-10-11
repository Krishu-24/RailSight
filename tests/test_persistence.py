import numpy as np

from railsight.anomaly import Blob, BlobPersistence


def blob(x, y, s=20):
    return Blob((x, y, x + s, y + s), s * s, np.zeros((1, 1, 2), np.int32))


def test_unknown_object_confirmed_after_three_updates():
    p = BlobPersistence(3)
    seen = [p.update([blob(100 + 2 * k, 200)], True) for k in range(3)]
    assert [len(s) for s in seen] == [0, 0, 1]  # slight motion between updates is tolerated


def test_flicker_is_dropped():
    p = BlobPersistence(3)
    p.update([blob(100, 200)], True)
    p.update([], True)  # gone for one update: the count restarts
    assert p.update([blob(100, 200)], True) == []


def test_between_updates_the_confirmed_set_is_kept():
    p = BlobPersistence(3)
    for _ in range(3):
        p.update([blob(50, 50)], True)
    assert len(p.update([blob(52, 50)], False)) == 1


def test_disabled_passes_everything_through():
    assert len(BlobPersistence(1).update([blob(0, 0)], True)) == 1
