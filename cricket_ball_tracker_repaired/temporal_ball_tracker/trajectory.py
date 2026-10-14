"""Constant-velocity trajectory gate for per-frame ball detections.

The detector answers every frame; this filter decides when it should abstain.
A tiny constant-velocity Kalman filter tracks the ball through time. A new
detection is accepted when it is consistent with the predicted position (or
when the track is lost and the detection is strong); inconsistent or weak
detections become abstentions, and the track coasts briefly through occlusion
before it is dropped. This mirrors the online tracker WASB pairs with its
detector, and converts gross localisation outliers into "no detection".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Detection:
    x: float
    y: float
    confidence: float


@dataclass
class TrackedPoint:
    x: float
    y: float
    visible: bool
    confidence: float


class ConstantVelocityKalman:
    """Kalman filter with state [x, y, vx, vy] and position observations."""

    def __init__(self, process_noise: float = 4.0, measurement_noise: float = 4.0):
        self.transition = np.eye(4)
        self.transition[0, 2] = self.transition[1, 3] = 1.0
        self.observation = np.zeros((2, 4))
        self.observation[0, 0] = self.observation[1, 1] = 1.0
        self.process_noise = np.diag([1.0, 1.0, 4.0, 4.0]) * process_noise
        self.measurement_noise = np.eye(2) * measurement_noise
        self.state: Optional[np.ndarray] = None
        self.covariance = np.eye(4) * 1e3

    def start(self, x: float, y: float) -> None:
        self.state = np.array([x, y, 0.0, 0.0])
        self.covariance = np.diag([4.0, 4.0, 100.0, 100.0])

    def predict(self) -> np.ndarray:
        assert self.state is not None
        self.state = self.transition @ self.state
        self.covariance = self.transition @ self.covariance @ self.transition.T + self.process_noise
        return self.state[:2].copy()

    def update(self, x: float, y: float) -> None:
        assert self.state is not None
        innovation = np.array([x, y]) - self.observation @ self.state
        innovation_cov = self.observation @ self.covariance @ self.observation.T + self.measurement_noise
        gain = self.covariance @ self.observation.T @ np.linalg.inv(innovation_cov)
        self.state = self.state + gain @ innovation
        self.covariance = (np.eye(4) - gain @ self.observation) @ self.covariance


class TrajectoryFilter:
    """Streaming acceptance gate over raw per-frame detections.

    ``gate_radius`` is the maximum distance (in the detection's pixel space)
    between the Kalman prediction and a detection for it to be accepted while
    tracking. ``max_coast`` frames may pass without an accepted detection
    before the track resets; while lost, a track restarts only from
    ``min_start_confidence`` detections that agree across two frames within
    ``start_radius``.
    """

    def __init__(
        self,
        gate_radius: float = 40.0,
        max_coast: int = 4,
        min_start_confidence: float = 0.6,
        start_radius: float = 150.0,
        speed_gate_factor: float = 1.5,
    ):
        self.gate_radius = gate_radius
        self.max_coast = max_coast
        self.min_start_confidence = min_start_confidence
        self.start_radius = start_radius
        self.speed_gate_factor = speed_gate_factor
        self.kalman = ConstantVelocityKalman()
        self.coasting = 0
        self.tracking = False
        self.pending: Optional[Detection] = None

    def update(self, detection: Detection) -> TrackedPoint:
        if self.tracking:
            speed = float(np.hypot(self.kalman.state[2], self.kalman.state[3]))
            predicted = self.kalman.predict()
            distance = float(np.hypot(detection.x - predicted[0], detection.y - predicted[1]))
            # A serve can cover >100 px/frame and reverses direction at the hit
            # and bounce, so the gate must scale with the current track speed.
            gate = max(self.gate_radius, self.speed_gate_factor * speed)
            if distance <= gate:
                self.kalman.update(detection.x, detection.y)
                self.coasting = 0
                return TrackedPoint(detection.x, detection.y, True, detection.confidence)
            self.coasting += 1
            if self.coasting > self.max_coast:
                self.tracking = False
                self.pending = None
            return TrackedPoint(float(predicted[0]), float(predicted[1]), False, detection.confidence)
        # Not tracking: require two consecutive strong, mutually-consistent
        # detections before starting a track, so single-frame false positives
        # on empty play never produce output.
        if detection.confidence >= self.min_start_confidence:
            if self.pending is not None:
                distance = float(np.hypot(detection.x - self.pending.x, detection.y - self.pending.y))
                if distance <= self.start_radius:
                    self.kalman.start(detection.x, detection.y)
                    self.kalman.state[2] = detection.x - self.pending.x
                    self.kalman.state[3] = detection.y - self.pending.y
                    self.tracking = True
                    self.coasting = 0
                    self.pending = None
                    return TrackedPoint(detection.x, detection.y, True, detection.confidence)
            self.pending = detection
        else:
            self.pending = None
        return TrackedPoint(detection.x, detection.y, False, detection.confidence)


def filter_sequence(detections: List[Detection], **kwargs) -> List[TrackedPoint]:
    """Run the streaming filter over one clip's raw detections."""

    tracker = TrajectoryFilter(**kwargs)
    return [tracker.update(detection) for detection in detections]
