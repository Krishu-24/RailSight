import pytest

from railsight.risk import Hazard, RiskLevel, RiskStateMachine, assess_hazard, stopping_distance

KMH = 1 / 3.6
COMMON = dict(half_width_m=2.1, stop_margin_m=50, near_buffer_m=2.0, max_range_m=1000)


def test_stopping_distance_100kmh():
    v = 100 * KMH
    # 2 s reaction + v^2 / 2a with a = 1 m/s^2
    assert stopping_distance(v, 2.0, 1.0) == pytest.approx(55.6 + 385.8, abs=0.5)


def run(h, kmh=60):
    v = kmh * KMH
    return assess_hazard(h, v, stop_dist_m=stopping_distance(v, 2.0, 1.0), **COMMON)


def test_object_in_corridor_close_is_critical():
    assert run(Hazard(1, "person", 100, 0.3)).level == RiskLevel.CRITICAL


def test_object_in_corridor_far_is_warning():
    assert run(Hazard(1, "person", 600, 0.3)).level == RiskLevel.WARNING


def test_object_beside_corridor_is_safe_or_near():
    assert run(Hazard(1, "person", 100, 3.0)).level == RiskLevel.WARNING  # within near buffer
    assert run(Hazard(1, "person", 100, 8.0)).level == RiskLevel.SAFE


def test_crossing_object_predicted_into_path():
    # 6 m to the side, walking towards the track at 1 m/s, train arrives in ~6 s
    a = run(Hazard(1, "person", 100, -6.0, lateral_speed_mps=1.0))
    assert a.in_path and a.level == RiskLevel.CRITICAL


def test_object_that_crosses_before_arrival_is_still_flagged():
    # crosses the corridor between ~2.6 s and ~5.4 s; train arrives at 6 s (within window)
    a = run(Hazard(1, "person", 100, -6.0, lateral_speed_mps=1.5))
    assert a.in_path


def test_object_long_gone_is_safe():
    # crosses quickly and is 20 m clear long before the train gets there
    a = run(Hazard(1, "animal", 300, -3.0, lateral_speed_mps=3.0))
    assert not a.in_path


def test_object_moving_away_is_not_in_path():
    a = run(Hazard(1, "animal", 100, 2.5, lateral_speed_mps=1.0))
    assert not a.in_path


def test_state_machine_hysteresis():
    sm = RiskStateMachine(escalate_frames=3, deescalate_frames=5)
    seq = [RiskLevel.CRITICAL] * 2 + [RiskLevel.SAFE]  # a 2-frame blip is ignored
    assert [sm.update(s) for s in seq][-1] == RiskLevel.SAFE
    for _ in range(3):
        sm.update(RiskLevel.CRITICAL)
    assert sm.state == RiskLevel.CRITICAL
    for _ in range(4):
        sm.update(RiskLevel.SAFE)
    assert sm.state == RiskLevel.CRITICAL  # still holding
    sm.update(RiskLevel.SAFE)
    assert sm.state == RiskLevel.SAFE
