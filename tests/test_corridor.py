import numpy as np
import pytest

from railsight.corridor import Corridor, focal_from_hfov, select_ego_track, trace_rows

H, W = 540, 960
HORIZON = 200.0
GAUGE = 1.435


def synthetic_track(slope: float = 1.2, offset: float = 0.0) -> np.ndarray:
    """Straight track whose width shrinks linearly to zero at the horizon row."""
    m = np.zeros((H, W), bool)
    for y in range(int(HORIZON) + 5, H):
        half = slope * (y - HORIZON) / 2
        c = W / 2 + offset
        m[y, int(c - half) : int(c + half)] = True
    return m


def test_width_fit_recovers_horizon():
    cor = Corridor.fit(synthetic_track(), GAUGE, focal_from_hfov(W, 60))
    assert cor is not None
    assert cor.horizon_y == pytest.approx(HORIZON, abs=3)
    assert cor.centre_x(400) == pytest.approx(W / 2, abs=2)


def test_distance_matches_pinhole_model():
    f = focal_from_hfov(W, 60)
    cor = Corridor.fit(synthetic_track(slope=1.2), GAUGE, f)
    y = 450
    expected = f * GAUGE / (1.2 * (y - HORIZON))
    assert cor.distance_m(y) == pytest.approx(expected, rel=0.03)
    # rows closer to the horizon are further away
    assert cor.distance_m(250) > cor.distance_m(500)


def test_lateral_offset_in_metres():
    cor = Corridor.fit(synthetic_track(), GAUGE, 800)
    y = 500
    one_gauge_right = W / 2 + cor.width_px(y)
    assert cor.lateral_m(one_gauge_right, y) == pytest.approx(GAUGE, rel=0.02)


def test_occlusion_hole_does_not_break_fit():
    m = synthetic_track()
    m[380:420, :] = False  # an object hides part of the track
    rows = trace_rows(m)
    assert rows[:, 0].min() > 380  # tracing stops at the gap from the bottom
    cor = Corridor.fit(m, GAUGE, 800)
    assert cor is not None and cor.horizon_y == pytest.approx(HORIZON, abs=6)


def test_ego_track_prefers_bottom_centre():
    ego = synthetic_track()
    side = np.zeros_like(ego)
    side[300:H, 20:120] = True  # a neighbouring track at the image edge
    picked = select_ego_track(ego | side)
    assert picked[H - 1, W // 2]
    assert not picked[H - 1, 50]


def test_corridor_polygon_and_mask():
    cor = Corridor.fit(synthetic_track(), GAUGE, 800)
    mask = cor.mask(half_width_m=2.1)
    assert mask[H - 1, W // 2]
    # at row 300 the track is ~120 px wide, so a 2.1 m half-width is ~176 px
    assert mask[300, W // 2 + 150]
    assert not mask[300, W // 2 + 220]
    assert not mask[int(HORIZON) - 20, W // 2]  # nothing above the horizon


def test_too_little_track_returns_none():
    m = np.zeros((H, W), bool)
    m[H - 5 :, 400:500] = True
    assert Corridor.fit(m, GAUGE, 800) is None


def synthetic_switch(slope: float = 1.2, rail_frac: float = 0.06):
    """Ego track plus a neighbouring track that merges into it in the lower half of the frame.
    Returns (track_mask, rail_mask)."""
    track = np.zeros((H, W), bool)
    rails = np.zeros((H, W), bool)
    for y in range(int(HORIZON) + 5, H):
        wd = slope * (y - HORIZON)
        rw = max(1, int(rail_frac * wd))
        for c in (W / 2, W / 2 - 1.6 * wd if y > 380 else None):  # neighbour appears near the bottom
            if c is None:
                continue
            x0, x1 = int(c - wd / 2), int(c + wd / 2)
            track[y, max(x0, 0) : max(x1, 0)] = True
            for rx in (x0, x1 - rw):
                if rx >= 0:
                    rails[y, rx : rx + rw] = True
        if y > 380:  # ballast between the two tracks is labelled as track too: one merged blob
            track[y, max(int(W / 2 - 1.6 * wd), 0) : int(W / 2)] = True
    return track, rails


def test_rail_pairs_resist_merged_tracks():
    track, rails = synthetic_switch()
    f = 800
    naive = Corridor.fit(track, GAUGE, f)
    robust = Corridor.fit(track, GAUGE, f, rail_mask=rails)
    true_w = 1.2 * (500 - HORIZON)
    assert robust.width_px(500) == pytest.approx(true_w, rel=0.05)
    assert robust.centre_x(500) == pytest.approx(W / 2, abs=4)
    assert abs(naive.width_px(500) - true_w) > abs(robust.width_px(500) - true_w)


def test_rail_mask_without_rails_falls_back():
    track = synthetic_track()
    cor = Corridor.fit(track, GAUGE, 800, rail_mask=np.zeros_like(track))
    assert cor is not None and cor.horizon_y == pytest.approx(HORIZON, abs=3)
