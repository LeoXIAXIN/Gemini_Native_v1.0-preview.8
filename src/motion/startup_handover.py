"""Deterministic handover primitives for G1 tracking.

This module intentionally depends only on NumPy.  The same trajectory can be
used by the real Unitree controller and by MuJoCo without importing DDS,
Unitree SDK2, Redis, or a policy runtime.

``StartupHandover`` is retained for the older measured-pose startup path.  The
remote-only alpha.5 flow uses :class:`RemoteOnlyHandover` instead::

    already standing -> START -> stabilize stand policy -> wait for A
                     -> blend to live -> B -> blend back to stand

The remote-only planner never invents a crouch pose and never calls a Unitree
motion service.  It only produces a live-reference weight; the caller keeps
the released tracking policy in full closed-loop control throughout.

The legacy handover has four phases::

    measured pose -> default stand -> settle -> blend to live -> tracking

``StartupHandover`` only plans targets and weights.  It never publishes a
motor command and therefore cannot bypass the caller's real-output gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np


class HandoverPhase(str, Enum):
    """Observable phases shared by real and simulated startup paths."""

    MOVE_TO_STAND = "move_to_stand"
    SETTLE = "settle"
    BLEND_TO_LIVE = "blend_to_live"
    TRACKING = "tracking"


class RemoteHandoverPhase(str, Enum):
    """Observable phases of the remote-only START/A/B reference handover."""

    WAIT_FOR_START = "wait_for_start"
    STABILIZING_STAND = "stabilizing_stand"
    WAIT_FOR_LIVE = "wait_for_live"
    BLEND_TO_LIVE = "blend_to_live"
    LIVE_TRACKING = "live_tracking"
    BLEND_TO_STAND = "blend_to_stand"
    STAND_HOLD = "stand_hold"


class RemoteHandoverCommand(str, Enum):
    """Read-only operator events understood by :class:`RemoteOnlyHandover`."""

    START = "start"
    LIVE = "a"
    STAND = "b"


@dataclass(frozen=True)
class RemoteHandoverSample:
    """One deterministic remote-only reference-selection sample."""

    phase: RemoteHandoverPhase
    live_weight: float
    phase_progress: float
    start_accepted: bool


@dataclass(frozen=True)
class HandoverSample:
    """One deterministic sample of a startup handover trajectory."""

    phase: HandoverPhase
    joint_target: np.ndarray
    stiffness_weight: float
    live_weight: float
    phase_progress: float


def smootherstep01(value: float) -> float:
    """Quintic smooth step on ``[0, 1]`` with zero endpoint velocity/accel."""

    x = float(value)
    if not math.isfinite(x):
        raise ValueError("smootherstep input must be finite")
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def _finite_duration(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative duration")
    return result


def _finite_vector(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinity")
    return result.copy()


class RemoteOnlyHandover:
    """Pure START/A/B state machine for the already-standing workflow.

    The caller advances the machine with its control-loop ``dt`` and applies
    ``live_weight`` to a blend between a stationary stand reference and the
    aligned live reference.  START is one-shot.  A can enter (or re-enter) live
    tracking only after stand stabilization.  B always returns toward the stand
    reference.  Reversing A/B during a blend is continuous because the next
    blend starts from the current weight.

    This class has no clock, Redis, MuJoCo, DDS, or SDK dependency, which keeps
    the transition semantics identical in unit tests and deployment callers.
    """

    def __init__(
        self,
        *,
        stabilize_seconds: float = 0.5,
        blend_seconds: float = 1.5,
    ) -> None:
        self.stabilize_seconds = _finite_duration(
            stabilize_seconds, "stabilize_seconds"
        )
        self.blend_seconds = _finite_duration(blend_seconds, "blend_seconds")
        self.phase = RemoteHandoverPhase.WAIT_FOR_START
        self._phase_elapsed = 0.0
        self._live_weight = 0.0
        self._blend_start_weight = 0.0
        self._start_accepted = False

    @property
    def start_accepted(self) -> bool:
        return self._start_accepted

    @property
    def live_weight(self) -> float:
        return float(self._live_weight)

    def command(self, command: RemoteHandoverCommand | str) -> bool:
        """Apply one operator event and return whether it changed the state."""

        try:
            event = (
                command
                if isinstance(command, RemoteHandoverCommand)
                else RemoteHandoverCommand(str(command).strip().lower())
            )
        except ValueError:
            return False

        if event == RemoteHandoverCommand.START:
            if self._start_accepted:
                return False
            self._start_accepted = True
            self.phase = RemoteHandoverPhase.STABILIZING_STAND
            self._phase_elapsed = 0.0
            self._live_weight = 0.0
            if self.stabilize_seconds == 0.0:
                self.phase = RemoteHandoverPhase.WAIT_FOR_LIVE
            return True

        if not self._start_accepted:
            return False

        if event == RemoteHandoverCommand.LIVE:
            if self.phase in {
                RemoteHandoverPhase.WAIT_FOR_LIVE,
                RemoteHandoverPhase.STAND_HOLD,
                RemoteHandoverPhase.BLEND_TO_STAND,
            }:
                self._blend_start_weight = self._live_weight
                self._phase_elapsed = 0.0
                self.phase = RemoteHandoverPhase.BLEND_TO_LIVE
                if self.blend_seconds == 0.0:
                    self._live_weight = 1.0
                    self.phase = RemoteHandoverPhase.LIVE_TRACKING
                return True
            return False

        if event == RemoteHandoverCommand.STAND:
            if self.phase in {
                RemoteHandoverPhase.BLEND_TO_LIVE,
                RemoteHandoverPhase.LIVE_TRACKING,
            }:
                self._blend_start_weight = self._live_weight
                self._phase_elapsed = 0.0
                self.phase = RemoteHandoverPhase.BLEND_TO_STAND
                if self.blend_seconds == 0.0:
                    self._live_weight = 0.0
                    self.phase = RemoteHandoverPhase.STAND_HOLD
                return True
            return False

        return False

    def advance(self, dt_seconds: float) -> RemoteHandoverSample:
        """Advance by a finite non-negative control-loop interval."""

        remaining = _finite_duration(dt_seconds, "dt_seconds")
        while remaining > 0.0:
            if self.phase == RemoteHandoverPhase.STABILIZING_STAND:
                needed = max(0.0, self.stabilize_seconds - self._phase_elapsed)
                consumed = min(remaining, needed)
                self._phase_elapsed += consumed
                remaining -= consumed
                if self._phase_elapsed >= self.stabilize_seconds:
                    self.phase = RemoteHandoverPhase.WAIT_FOR_LIVE
                    self._phase_elapsed = 0.0
                    self._live_weight = 0.0
                    continue
                break

            if self.phase in {
                RemoteHandoverPhase.BLEND_TO_LIVE,
                RemoteHandoverPhase.BLEND_TO_STAND,
            }:
                needed = max(0.0, self.blend_seconds - self._phase_elapsed)
                consumed = min(remaining, needed)
                self._phase_elapsed += consumed
                remaining -= consumed
                raw = (
                    self._phase_elapsed / self.blend_seconds
                    if self.blend_seconds > 0.0
                    else 1.0
                )
                curve = smootherstep01(raw)
                if self.phase == RemoteHandoverPhase.BLEND_TO_LIVE:
                    self._live_weight = self._blend_start_weight + (
                        1.0 - self._blend_start_weight
                    ) * curve
                    if self._phase_elapsed >= self.blend_seconds:
                        self._live_weight = 1.0
                        self.phase = RemoteHandoverPhase.LIVE_TRACKING
                        self._phase_elapsed = 0.0
                        continue
                else:
                    self._live_weight = self._blend_start_weight * (1.0 - curve)
                    if self._phase_elapsed >= self.blend_seconds:
                        self._live_weight = 0.0
                        self.phase = RemoteHandoverPhase.STAND_HOLD
                        self._phase_elapsed = 0.0
                        continue
                break

            # Waiting/steady phases do not consume wall time into a later phase.
            break

        return self.sample()

    def sample(self) -> RemoteHandoverSample:
        """Return the current phase without advancing time."""

        if self.phase == RemoteHandoverPhase.STABILIZING_STAND:
            progress = (
                self._phase_elapsed / self.stabilize_seconds
                if self.stabilize_seconds > 0.0
                else 1.0
            )
        elif self.phase in {
            RemoteHandoverPhase.BLEND_TO_LIVE,
            RemoteHandoverPhase.BLEND_TO_STAND,
        }:
            progress = (
                self._phase_elapsed / self.blend_seconds
                if self.blend_seconds > 0.0
                else 1.0
            )
        elif self.phase in {
            RemoteHandoverPhase.WAIT_FOR_LIVE,
            RemoteHandoverPhase.LIVE_TRACKING,
            RemoteHandoverPhase.STAND_HOLD,
        }:
            progress = 1.0
        else:
            progress = 0.0
        return RemoteHandoverSample(
            phase=self.phase,
            live_weight=float(np.clip(self._live_weight, 0.0, 1.0)),
            phase_progress=float(np.clip(progress, 0.0, 1.0)),
            start_accepted=self._start_accepted,
        )


class StartupHandover:
    """Plan a smooth, time-indexed START-only takeover.

    ``elapsed_seconds`` passed to :meth:`sample` is measured from the caller's
    START event.  Keeping time outside the planner makes the exact same class
    deterministic in unit tests, MuJoCo, and the real 50 Hz control loop.  The
    position trajectory uses a full-duration quintic curve, while stiffness
    ramps up sooner from a nonzero floor so a crouched robot is not left with
    near-zero support for the first part of a long stand transition.
    """

    def __init__(
        self,
        default_joint_qpos: np.ndarray,
        *,
        stand_seconds: float = 3.0,
        settle_seconds: float = 0.5,
        blend_seconds: float = 1.5,
        minimum_stiffness_weight: float = 0.2,
        stiffness_ramp_seconds: float = 0.75,
    ) -> None:
        self.default_joint_qpos = _finite_vector(
            default_joint_qpos, "default_joint_qpos"
        )
        self.stand_seconds = _finite_duration(stand_seconds, "stand_seconds")
        self.settle_seconds = _finite_duration(settle_seconds, "settle_seconds")
        self.blend_seconds = _finite_duration(blend_seconds, "blend_seconds")
        self.minimum_stiffness_weight = float(minimum_stiffness_weight)
        if (
            not math.isfinite(self.minimum_stiffness_weight)
            or not 0.0 <= self.minimum_stiffness_weight <= 1.0
        ):
            raise ValueError("minimum_stiffness_weight must be within [0, 1]")
        self.stiffness_ramp_seconds = _finite_duration(
            stiffness_ramp_seconds, "stiffness_ramp_seconds"
        )
        self._measured_joint_qpos: np.ndarray | None = None

    @property
    def total_seconds(self) -> float:
        return self.stand_seconds + self.settle_seconds + self.blend_seconds

    def start(self, measured_joint_qpos: np.ndarray) -> None:
        measured = _finite_vector(measured_joint_qpos, "measured_joint_qpos")
        if measured.shape != self.default_joint_qpos.shape:
            raise ValueError(
                "measured_joint_qpos shape does not match default_joint_qpos"
            )
        self._measured_joint_qpos = measured

    def sample(self, elapsed_seconds: float) -> HandoverSample:
        """Return the phase, direct stand target, and live-policy blend weight."""

        if self._measured_joint_qpos is None:
            raise RuntimeError("StartupHandover.start() must be called first")
        elapsed = float(elapsed_seconds)
        if not math.isfinite(elapsed):
            raise ValueError("elapsed_seconds must be finite")
        elapsed = max(0.0, elapsed)

        if elapsed < self.stand_seconds:
            raw_progress = (
                elapsed / self.stand_seconds if self.stand_seconds > 0.0 else 1.0
            )
            weight = smootherstep01(raw_progress)
            gain_duration = min(self.stand_seconds, self.stiffness_ramp_seconds)
            gain_progress = elapsed / gain_duration if gain_duration > 0.0 else 1.0
            stiffness_weight = self.minimum_stiffness_weight + (
                1.0 - self.minimum_stiffness_weight
            ) * smootherstep01(gain_progress)
            target = (
                (1.0 - weight) * self._measured_joint_qpos
                + weight * self.default_joint_qpos
            ).astype(np.float32)
            return HandoverSample(
                phase=HandoverPhase.MOVE_TO_STAND,
                joint_target=target,
                stiffness_weight=float(stiffness_weight),
                live_weight=0.0,
                phase_progress=float(raw_progress),
            )

        settle_elapsed = elapsed - self.stand_seconds
        if settle_elapsed < self.settle_seconds:
            progress = (
                settle_elapsed / self.settle_seconds
                if self.settle_seconds > 0.0
                else 1.0
            )
            return HandoverSample(
                phase=HandoverPhase.SETTLE,
                joint_target=self.default_joint_qpos.copy(),
                stiffness_weight=1.0,
                live_weight=0.0,
                phase_progress=float(progress),
            )

        blend_elapsed = settle_elapsed - self.settle_seconds
        if blend_elapsed < self.blend_seconds:
            raw_progress = (
                blend_elapsed / self.blend_seconds
                if self.blend_seconds > 0.0
                else 1.0
            )
            return HandoverSample(
                phase=HandoverPhase.BLEND_TO_LIVE,
                joint_target=self.default_joint_qpos.copy(),
                stiffness_weight=1.0,
                live_weight=smootherstep01(raw_progress),
                phase_progress=float(raw_progress),
            )

        return HandoverSample(
            phase=HandoverPhase.TRACKING,
            joint_target=self.default_joint_qpos.copy(),
            stiffness_weight=1.0,
            live_weight=1.0,
            phase_progress=1.0,
        )


def blend_floating_base_qpos(
    anchor_qpos: np.ndarray,
    live_qpos: np.ndarray,
    weight: float,
) -> np.ndarray:
    """Blend two floating-base qpos vectors, including a normalized quaternion.

    Translation and joints use linear interpolation.  The root quaternion uses
    shortest-hemisphere normalized lerp, which is stable at the small angular
    differences expected during the short startup blend and never emits a
    non-unit quaternion to MuJoCo FK.
    """

    anchor = _finite_vector(anchor_qpos, "anchor_qpos")
    live = _finite_vector(live_qpos, "live_qpos")
    if anchor.shape != live.shape or anchor.size < 7:
        raise ValueError("floating-base qpos vectors must share a shape of at least 7")
    blend_weight = float(weight)
    if not math.isfinite(blend_weight):
        raise ValueError("weight must be finite")
    blend_weight = float(np.clip(blend_weight, 0.0, 1.0))

    result = ((1.0 - blend_weight) * anchor + blend_weight * live).astype(np.float32)
    q_anchor = anchor[3:7].astype(np.float64)
    q_live = live[3:7].astype(np.float64)
    anchor_norm = float(np.linalg.norm(q_anchor))
    live_norm = float(np.linalg.norm(q_live))
    if anchor_norm < 1e-8 or live_norm < 1e-8:
        raise ValueError("floating-base quaternion norm is too small")
    q_anchor /= anchor_norm
    q_live /= live_norm
    if float(np.dot(q_anchor, q_live)) < 0.0:
        q_live = -q_live
    q_blend = (1.0 - blend_weight) * q_anchor + blend_weight * q_live
    q_blend_norm = float(np.linalg.norm(q_blend))
    if q_blend_norm < 1e-8:
        raise ValueError("blended floating-base quaternion norm is too small")
    result[3:7] = np.asarray(q_blend / q_blend_norm, dtype=np.float32)
    return result
