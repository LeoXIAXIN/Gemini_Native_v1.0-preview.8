"""Reference watchdog and blend for the G1 safety gateway.

Import-cycle note: ``g1_safety.core`` helpers are resolved lazily at call time
so importing this module remains lightweight.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys
from pathlib import Path as _Path
from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Sequence

import numpy as np


def _core():
    # M-decoupling: the core state machine is the single source of truth in
    # src.safety.g1_core; no legacy package is involved.
    return _importlib.import_module("src.safety.g1_core")


def _validation_error(message: str) -> Exception:
    return _core().ValidationError(message)


def _vector(values, size, label):
    return _core()._vector(values, size, label)


def _minimum_jerk(progress):
    return _core().minimum_jerk(progress)


class ReferenceHealth(str, Enum):
    LIVE = "LIVE"
    SOFT_STALE = "SOFT_STALE"
    HARD_STALE = "HARD_STALE"
    WAIT_FRESH = "WAIT_FRESH"
    STANDBY = "STANDBY"
    INVALID = "INVALID"


@dataclass
class ReferenceSample:
    seq: int
    timestamp: float
    position: Sequence[float]
    velocity: Sequence[float]
    valid: bool = True


@dataclass
class ReferenceDecision:
    health: ReferenceHealth
    age_s: float
    use_live: bool
    resume_required: bool
    reason: str


@dataclass
class ReferenceOutput:
    position: np.ndarray
    velocity: np.ndarray
    teleop_weight: float
    decision: ReferenceDecision


class ReferenceWatchdog:
    """Source watchdog.  Manual resume after any loss is the safe default."""

    def __init__(
        self,
        size: int,
        *,
        soft_stale_s: float = 0.080,
        hard_stale_s: float = 0.250,
        fresh_dwell_s: float = 0.500,
        manual_resume: bool = True,
        future_tolerance_s: float = 0.020,
    ) -> None:
        if size <= 0 or not (0 < soft_stale_s < hard_stale_s):
            raise ValueError("invalid watchdog size or stale thresholds")
        self.size = size
        self.soft_stale_s = soft_stale_s
        self.hard_stale_s = hard_stale_s
        self.fresh_dwell_s = fresh_dwell_s
        self.manual_resume = manual_resume
        self.future_tolerance_s = future_tolerance_s
        self._last: Optional[ReferenceSample] = None
        self._last_seq: Optional[int] = None
        self._last_timestamp: Optional[float] = None
        self._loss_seen = False
        self._hard_seen = False
        self._fresh_since: Optional[float] = None
        self._resume_requested = False

    def request_resume(self) -> None:
        self._resume_requested = True

    def reset(self) -> None:
        """Forget sequence/session history after an authenticated source restart."""

        self._last = None
        self._last_seq = None
        self._last_timestamp = None
        self._loss_seen = False
        self._hard_seen = False
        self._fresh_since = None
        self._resume_requested = False

    def force_standby(self) -> None:
        self._loss_seen = True
        self._hard_seen = True
        self._fresh_since = None
        self._resume_requested = False

    def _accept(self, sample: ReferenceSample, now: float) -> None:
        if not sample.valid:
            raise _validation_error("reference marked invalid")
        if not isinstance(sample.seq, (int, np.integer)) or sample.seq < 0:
            raise _validation_error("reference sequence invalid")
        if not math.isfinite(sample.timestamp) or sample.timestamp > now + self.future_tolerance_s:
            raise _validation_error("reference timestamp invalid")
        position = _vector(sample.position, self.size, "reference.position")
        velocity = _vector(sample.velocity, self.size, "reference.velocity")
        if self._last_seq is not None:
            if sample.seq < self._last_seq:
                raise _validation_error("reference sequence moved backwards")
            if sample.seq == self._last_seq and sample.timestamp != self._last_timestamp:
                raise _validation_error("same reference sequence changed timestamp")
            if sample.seq == self._last_seq and self._last is not None:
                if not np.array_equal(position, np.asarray(self._last.position)) or not np.array_equal(
                    velocity, np.asarray(self._last.velocity)
                ):
                    raise _validation_error("same reference sequence changed payload")
            if sample.seq > self._last_seq and sample.timestamp <= float(self._last_timestamp):
                raise _validation_error("reference timestamp is not increasing")
        if self._last_seq is None or sample.seq > self._last_seq:
            self._last = sample
            self._last_seq = int(sample.seq)
            self._last_timestamp = float(sample.timestamp)

    def update(self, sample: Optional[ReferenceSample], now: float) -> ReferenceDecision:
        if sample is not None:
            try:
                self._accept(sample, now)
            except _core().ValidationError as exc:
                self._loss_seen = True
                self._hard_seen = True
                self._fresh_since = None
                return ReferenceDecision(ReferenceHealth.INVALID, math.inf, False, True, str(exc))
        age = math.inf if self._last is None else max(0.0, now - self._last.timestamp)
        if age > self.hard_stale_s:
            self._loss_seen = True
            self._hard_seen = True
            self._fresh_since = None
            return ReferenceDecision(ReferenceHealth.HARD_STALE, age, False, True, "reference hard stale")
        if age > self.soft_stale_s:
            self._loss_seen = True
            self._fresh_since = None
            return ReferenceDecision(
                ReferenceHealth.SOFT_STALE, age, False,
                self.manual_resume or self._hard_seen,
                "reference soft stale",
            )
        if self._loss_seen:
            if self._fresh_since is None:
                self._fresh_since = now
            dwell_met = now - self._fresh_since >= self.fresh_dwell_s
            resume_needed = self.manual_resume or self._hard_seen
            if not dwell_met:
                return ReferenceDecision(ReferenceHealth.WAIT_FRESH, age, False, resume_needed, "waiting for continuous fresh dwell")
            if resume_needed and not self._resume_requested:
                return ReferenceDecision(ReferenceHealth.STANDBY, age, False, True, "fresh source ready; manual resume required")
            self._loss_seen = False
            self._hard_seen = False
            self._fresh_since = None
            self._resume_requested = False
        return ReferenceDecision(ReferenceHealth.LIVE, age, True, False, "reference live")


class ReferenceBlend:
    """Minimum-jerk live/standby blender with zero-velocity standby."""

    def __init__(
        self,
        standby_position: Sequence[float],
        *,
        blend_in_s: float = 0.750,
        blend_out_s: float = 0.500,
        quaternion_slice: Optional[slice] = None,
    ) -> None:
        self.standby = np.asarray(standby_position, dtype=np.float64).reshape(-1).copy()
        if self.standby.size == 0 or not np.all(np.isfinite(self.standby)):
            raise ValueError("standby_position must be a finite non-empty vector")
        if blend_in_s <= 0 or blend_out_s <= 0:
            raise ValueError("blend durations must be positive")
        self.blend_in_s = blend_in_s
        self.blend_out_s = blend_out_s
        self.quaternion_slice = quaternion_slice
        if quaternion_slice is not None:
            indices = np.arange(self.standby.size)[quaternion_slice]
            if indices.shape != (4,):
                raise ValueError("quaternion_slice must select exactly four values")
            self._quaternion_indices = indices
            self.standby[indices] = self._normalise_quaternion(
                self.standby[indices], "standby quaternion"
            )
        else:
            self._quaternion_indices = None
        self._weight = 0.0
        self._start_weight = 0.0
        self._target_weight = 0.0
        self._transition_start: Optional[float] = None
        self._last_live_position = self.standby.copy()
        self._last_live_velocity = np.zeros_like(self.standby)

    @staticmethod
    def _normalise_quaternion(value: Sequence[float], label: str) -> np.ndarray:
        quat = np.asarray(value, dtype=np.float64).reshape(-1)
        if quat.shape != (4,) or not np.all(np.isfinite(quat)):
            raise _validation_error(f"{label} must contain four finite values")
        norm = float(np.linalg.norm(quat))
        if norm < 1e-8:
            raise _validation_error(f"{label} has near-zero norm")
        return quat / norm

    @classmethod
    def _slerp_wxyz(
        cls, left: Sequence[float], right: Sequence[float], weight: float
    ) -> np.ndarray:
        q0 = cls._normalise_quaternion(left, "standby quaternion")
        q1 = cls._normalise_quaternion(right, "live quaternion")
        dot = float(np.dot(q0, q1))
        # q and -q encode the same rotation.  Select the short arc so an
        # antipodal packet can never interpolate through the zero quaternion.
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))
        if dot > 0.9995:
            return cls._normalise_quaternion(
                (1.0 - weight) * q0 + weight * q1,
                "interpolated quaternion",
            )
        angle = math.acos(dot)
        sin_angle = math.sin(angle)
        return (
            math.sin((1.0 - weight) * angle) / sin_angle * q0
            + math.sin(weight * angle) / sin_angle * q1
        )

    def _set_target(self, target: float, now: float) -> None:
        if target != self._target_weight:
            self._start_weight = self._weight
            self._target_weight = target
            self._transition_start = now

    def step(
        self,
        decision: ReferenceDecision,
        sample: Optional[ReferenceSample],
        now: float,
    ) -> ReferenceOutput:
        if sample is not None and sample.valid:
            self._last_live_position = _vector(sample.position, self.standby.size, "reference.position")
            self._last_live_velocity = _vector(sample.velocity, self.standby.size, "reference.velocity")
            if self._quaternion_indices is not None:
                indices = self._quaternion_indices
                self._last_live_position[indices] = self._normalise_quaternion(
                    self._last_live_position[indices], "live quaternion"
                )
        self._set_target(1.0 if decision.use_live else 0.0, now)
        if self._transition_start is not None:
            duration = self.blend_in_s if self._target_weight > self._start_weight else self.blend_out_s
            s = _minimum_jerk((now - self._transition_start) / duration)
            self._weight = self._start_weight + (self._target_weight - self._start_weight) * s
            if s >= 1.0:
                self._transition_start = None
                self._weight = self._target_weight
        position = (1.0 - self._weight) * self.standby + self._weight * self._last_live_position
        if self._quaternion_indices is not None:
            indices = self._quaternion_indices
            position[indices] = self._slerp_wxyz(
                self.standby[indices], self._last_live_position[indices], self._weight
            )
        velocity = self._weight * self._last_live_velocity
        return ReferenceOutput(position, velocity, float(self._weight), decision)
