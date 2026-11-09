"""Bounded multi-rate target tracking. Educational controller, not Helix System 0."""

from dataclasses import dataclass

import numpy as np


class TemporalEnsemble:
    def __init__(self, coefficient: float = 0.01):
        self.coefficient = coefficient
        self.chunks: list[tuple[int, np.ndarray]] = []

    def update(self, tick: int, chunk: np.ndarray) -> np.ndarray:
        self.chunks.append((tick, chunk.copy()))
        self.chunks = [(t, a) for t, a in self.chunks if 0 <= tick - t < len(a)]
        ages = np.array([tick - t for t, _ in self.chunks])
        weights = np.exp(-self.coefficient * ages)
        values = np.stack([a[tick - t] for t, a in self.chunks])
        return np.average(values, axis=0, weights=weights)


@dataclass
class BoundedController:
    lower: np.ndarray
    upper: np.ndarray
    timeout: float = 0.25
    max_speed: float = 2.0

    def __post_init__(self):
        self.lower, self.upper = np.asarray(self.lower), np.asarray(self.upper)
        if (
            self.lower.ndim != 1
            or self.lower.shape != self.upper.shape
            or not np.isfinite(self.lower).all()
            or not np.isfinite(self.upper).all()
            or np.any(self.lower > self.upper)
            or not np.isfinite([self.timeout, self.max_speed]).all()
            or min(self.timeout, self.max_speed) <= 0
        ):
            raise ValueError("Invalid controller limits or timing")

    def step(self, position: np.ndarray, target: np.ndarray, age: float, dt: float) -> np.ndarray:
        position, target = np.asarray(position), np.asarray(target)
        if position.shape != self.lower.shape or target.shape != self.lower.shape:
            raise ValueError("Controller coordinate shape mismatch")
        if not np.isfinite(position).all():
            raise ValueError("Nonfinite measured position: no reliable hold target")
        if not np.isfinite([dt, age]).all() or dt <= 0 or age < 0:
            raise ValueError("Invalid control time")
        if age > self.timeout or not np.isfinite(target).all():
            return np.clip(position, self.lower, self.upper)
        delta = np.clip(target - position, -self.max_speed * dt, self.max_speed * dt)
        return np.clip(position + delta, self.lower, self.upper)


@dataclass(frozen=True)
class TaskGoal:
    """Explicit single-task dispatch interface; no high-level learned reasoning claim."""

    name: str = "pick_up_cube"
    lift_height_m: float = 0.08
    hold_ticks: int = 10

    def __post_init__(self) -> None:
        if self.name != "pick_up_cube":
            raise ValueError("Only the OpenArm pick_up_cube task has a policy/data contract")
        if self.lift_height_m <= 0 or self.hold_ticks < 1:
            raise ValueError("Positive task success thresholds required")
