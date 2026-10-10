"""Collision-risk assessment: braking physics + in-path reasoning + temporal hysteresis."""

from dataclasses import dataclass, field
from enum import IntEnum


class RiskLevel(IntEnum):
    SAFE = 0
    WARNING = 1
    CRITICAL = 2


def stopping_distance(speed_mps: float, reaction_s: float, decel_mps2: float) -> float:
    """Distance travelled during the reaction time plus the braking distance v^2 / 2a."""
    return speed_mps * reaction_s + speed_mps**2 / (2.0 * decel_mps2)


@dataclass
class Hazard:
    """One observed object, already expressed in track coordinates."""

    track_id: int
    label: str
    distance_m: float           # longitudinal distance along the track
    lateral_m: float            # signed offset from the corridor centreline
    lateral_speed_mps: float = 0.0


@dataclass
class HazardAssessment:
    hazard: Hazard
    level: RiskLevel
    in_path: bool
    time_to_reach_s: float
    reason: str


def assess_hazard(
    h: Hazard,
    speed_mps: float,
    half_width_m: float,
    stop_dist_m: float,
    stop_margin_m: float,
    near_buffer_m: float,
    max_range_m: float,
    time_window_s: float = 2.0,
) -> HazardAssessment:
    """Rule-based risk for a single hazard.

    The object counts as in the path if it is inside the corridor now, or if its lateral motion
    puts it inside the corridor at any moment within +-time_window_s of the train's arrival.
    The window covers estimation error and the time the train itself takes to pass.
    """
    ttr = h.distance_m / speed_mps if speed_mps > 0.1 else float("inf")
    t1 = max(0.0, min(ttr, 30.0) - time_window_s)
    t2 = min(ttr, 30.0) + time_window_s
    l1 = h.lateral_m + h.lateral_speed_mps * t1
    l2 = h.lateral_m + h.lateral_speed_mps * t2

    inside_now = abs(h.lateral_m) <= half_width_m
    # the object sweeps the lateral interval [l1, l2] during the window
    inside_later = min(l1, l2) <= half_width_m and max(l1, l2) >= -half_width_m
    near = abs(h.lateral_m) <= half_width_m + near_buffer_m
    in_path = inside_now or inside_later

    if h.distance_m > max_range_m or speed_mps <= 0.1:
        return HazardAssessment(h, RiskLevel.SAFE, in_path, ttr, "out of range or stationary train")

    if in_path and h.distance_m <= stop_dist_m + stop_margin_m:
        why = "in corridor" if inside_now else "predicted to enter corridor"
        return HazardAssessment(h, RiskLevel.CRITICAL, True, ttr, f"{why} within stopping distance")
    if in_path:
        return HazardAssessment(h, RiskLevel.WARNING, True, ttr, "in path, beyond stopping distance")
    if near:
        return HazardAssessment(h, RiskLevel.WARNING, False, ttr, "close to corridor")
    return HazardAssessment(h, RiskLevel.SAFE, False, ttr, "clear of corridor")


@dataclass
class RiskStateMachine:
    """Smooths the frame-level risk so the alert does not flicker.

    Raising the level needs `escalate_frames` consecutive frames at that level or above;
    lowering it needs `deescalate_frames` consecutive frames below the current level.
    """

    escalate_frames: int = 3
    deescalate_frames: int = 15
    state: RiskLevel = RiskLevel.SAFE
    _up: int = field(default=0, repr=False)
    _down: int = field(default=0, repr=False)

    def update(self, observed: RiskLevel) -> RiskLevel:
        if observed > self.state:
            self._up += 1
            self._down = 0
            if self._up >= self.escalate_frames:
                self.state, self._up = observed, 0
        elif observed < self.state:
            self._down += 1
            self._up = 0
            if self._down >= self.deescalate_frames:
                self.state, self._down = observed, 0
        else:
            self._up = self._down = 0
        return self.state
