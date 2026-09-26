"""Pure checks and SDK ownership helpers for the remote-only G1 handover.

The operator places G1 in official damping/zero-torque mode and then Develop
mode *before* launching real control.  Fresh, version-matched LowState is the
field firmware's primary Develop lease; ``MotionSwitcher.CheckMode`` is an
optional secondary observation because its RPC service may be unavailable in
Develop.  The application remains read-only until it has validated a stable
supported pose and observed a released-then-pressed physical START button.

This module never changes the MotionSwitcher owner and never publishes LowCmd.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Iterable

import numpy as np


OFFICIAL_MOTION_OWNER = "ai"
LEG_WAIST_IDS = np.asarray(tuple(range(15)), dtype=np.intp)
LEFT_RIGHT_PAIRS = (
    (0, 6),   # hip pitch
    (1, 7),   # hip roll
    (2, 8),   # hip yaw
    (3, 9),   # knee
    (4, 10),  # ankle pitch
    (5, 11),  # ankle roll
)


class RemoteHandoverError(RuntimeError):
    """Fail-closed rejection of a remote-only control ownership transition."""


class ReleasedChordGate:
    """Debounce one exclusive chord after the complete remote was released.

    The caller is responsible for feeding one atomic button snapshot from each
    *new* LowState frame.  This helper deliberately has no knowledge of
    ``mode_machine``: that field identifies the G1 hardware variant and is not a
    reliable debug-mode signal.
    """

    def __init__(
        self,
        button_ids: Iterable[int],
        *,
        all_button_ids: Iterable[int] = range(16),
        minimum_pressed_frames: int = 2,
    ) -> None:
        ids = tuple(int(value) for value in button_ids)
        all_ids = tuple(int(value) for value in all_button_ids)
        if (
            not ids
            or len(set(ids)) != len(ids)
            or any(value < 0 for value in ids)
            or not all_ids
            or len(set(all_ids)) != len(all_ids)
            or any(value < 0 for value in all_ids)
            or not set(ids).issubset(all_ids)
            or int(minimum_pressed_frames) < 2
        ):
            raise ValueError("button_ids must be unique non-negative integers")
        self._ids = ids
        self._all_ids = all_ids
        self._other_ids = tuple(value for value in all_ids if value not in ids)
        self._minimum_pressed_frames = int(minimum_pressed_frames)
        self._released = False
        self._pressed_frames = 0

    @property
    def released(self) -> bool:
        """Whether one frame has shown every configured remote key released."""

        return self._released

    def update(self, buttons: Iterable[int]) -> bool:
        """Return true only after an exclusive chord remains held long enough."""

        snapshot = tuple(buttons)
        try:
            values = {
                index: int(snapshot[index]) for index in self._all_ids
            }
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("remote button snapshot does not contain the chord") from exc
        if any(value not in (0, 1) for value in values.values()):
            raise ValueError("remote button states must be exactly 0 or 1")

        chord = tuple(values[index] for index in self._ids)
        other_pressed = any(values[index] for index in self._other_ids)
        if not self._released:
            if not other_pressed and all(value == 0 for value in chord):
                self._released = True
            self._pressed_frames = 0
            return False
        if other_pressed or not all(value == 1 for value in chord):
            self._pressed_frames = 0
            return False
        self._pressed_frames += 1
        return self._pressed_frames >= self._minimum_pressed_frames


@dataclass(frozen=True)
class RemoteButtonEvents:
    """Modifier-safe actions decoded from one atomic LowState button frame."""

    actions: frozenset[int]
    failsafe_stop: bool


class SafeRemoteButtonEdges:
    """Decode ordinary actions without mistaking firmware chords for A/B.

    Ordinary action edges are accepted only while every shoulder modifier is
    released.  A dedicated released-then-pressed chord can request a controlled
    LowCmd shutdown without being emitted as its overlapping ordinary action.
    """

    def __init__(
        self,
        action_ids: Iterable[int],
        modifier_ids: Iterable[int],
        *,
        failsafe_chord: Iterable[int],
    ) -> None:
        actions = tuple(int(value) for value in action_ids)
        modifiers = tuple(int(value) for value in modifier_ids)
        chord = tuple(int(value) for value in failsafe_chord)
        all_ids = actions + modifiers + chord
        if (
            not actions
            or not modifiers
            or len(chord) < 2
            or any(value < 0 for value in all_ids)
            or len(set(actions)) != len(actions)
            or len(set(modifiers)) != len(modifiers)
            or len(set(chord)) != len(chord)
        ):
            raise ValueError("remote button IDs must be valid unique groups")
        self._actions = actions
        self._modifiers = modifiers
        self._failsafe_chord = chord
        self._armed = {value: False for value in actions}
        self._previous = {value: 1 for value in actions}
        self._failsafe_armed = False
        self._failsafe_previous = True

    def update(self, buttons: Iterable[int]) -> RemoteButtonEvents:
        """Decode exactly one fresh, atomic 0/1 remote-button snapshot."""

        snapshot = tuple(buttons)
        required = self._actions + self._modifiers + self._failsafe_chord
        try:
            values = {index: int(snapshot[index]) for index in set(required)}
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("remote button snapshot is incomplete") from exc
        if any(value not in (0, 1) for value in values.values()):
            raise ValueError("remote button states must be exactly 0 or 1")

        chord_pressed = all(values[index] == 1 for index in self._failsafe_chord)
        chord_released = all(values[index] == 0 for index in self._failsafe_chord)
        if chord_released:
            self._failsafe_armed = True
        failsafe_stop = (
            chord_pressed
            and not self._failsafe_previous
            and self._failsafe_armed
        )
        self._failsafe_previous = chord_pressed

        modifiers_clear = all(values[index] == 0 for index in self._modifiers)
        emitted: set[int] = set()
        for button_id in self._actions:
            pressed = values[button_id]
            if not pressed:
                self._armed[button_id] = True
            if (
                pressed
                and not self._previous[button_id]
                and self._armed[button_id]
                and modifiers_clear
                and not failsafe_stop
            ):
                emitted.add(button_id)
            self._previous[button_id] = pressed
        return RemoteButtonEvents(frozenset(emitted), failsafe_stop)


@dataclass(frozen=True)
class RemoteStandThresholds:
    """Conservative gates for handing over an already-standing supported G1."""

    stable_seconds: float = 0.50
    maximum_roll_pitch: float = 0.15
    maximum_gyro_norm: float = 0.35
    maximum_leg_waist_velocity: float = 0.20
    maximum_knee_angle: float = 1.20
    maximum_leg_pair_difference: float = 0.35
    maximum_policy_leg_target_delta: float = 0.15
    maximum_policy_full_target_delta: float = 0.25

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")


def _finite_vector(values: Iterable[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinity")
    return result


def roll_pitch_from_wxyz(quaternion: Iterable[float]) -> tuple[float, float]:
    """Return roll and pitch from one finite wxyz quaternion."""

    q = _finite_vector(quaternion, 4, "root_quat")
    norm = float(np.linalg.norm(q))
    if not 0.5 <= norm <= 1.5:
        raise ValueError(f"root_quat norm is invalid: {norm:.4f}")
    w, x, y, z = (float(value) / norm for value in q)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    return roll, pitch


def stable_stand_reason(
    root_quat: Iterable[float],
    root_gyro: Iterable[float],
    joint_qpos: Iterable[float],
    joint_qvel: Iterable[float],
    *,
    limits: RemoteStandThresholds | None = None,
) -> str | None:
    """Return ``None`` only for a stationary, upright, non-crouched pose.

    These are takeover gates, not a claim of foot contact or free-standing
    stability.  The released real policy has no measured foot-force input, so
    the physical support rig remains mandatory during commissioning.
    """

    limits = limits or RemoteStandThresholds()
    gyro = _finite_vector(root_gyro, 3, "root_gyro")
    qpos = _finite_vector(joint_qpos, 29, "joint_qpos")
    qvel = _finite_vector(joint_qvel, 29, "joint_qvel")
    roll, pitch = roll_pitch_from_wxyz(root_quat)
    if max(abs(roll), abs(pitch)) > limits.maximum_roll_pitch:
        return f"pelvis tilt is too large (roll={roll:.3f}, pitch={pitch:.3f})"
    gyro_norm = float(np.linalg.norm(gyro))
    if gyro_norm > limits.maximum_gyro_norm:
        return f"pelvis angular speed is too large ({gyro_norm:.3f} rad/s)"
    max_speed = float(np.max(np.abs(qvel[LEG_WAIST_IDS])))
    if max_speed > limits.maximum_leg_waist_velocity:
        return f"leg/waist joint speed is too large ({max_speed:.3f} rad/s)"
    knee = max(abs(float(qpos[3])), abs(float(qpos[9])))
    if knee > limits.maximum_knee_angle:
        return f"knees are still crouched ({knee:.3f} rad)"
    for left, right in LEFT_RIGHT_PAIRS:
        difference = abs(float(qpos[left] - qpos[right]))
        if difference > limits.maximum_leg_pair_difference:
            return (
                f"left/right leg pose is asymmetric at joints {left}/{right} "
                f"({difference:.3f} rad)"
            )
    return None


class StableStandDwell:
    """Require one uninterrupted stable window before permitting takeover."""

    def __init__(self, limits: RemoteStandThresholds | None = None) -> None:
        self.limits = limits or RemoteStandThresholds()
        self._stable_since: float | None = None
        self.last_reason: str | None = "no LowState sample observed"

    def update(
        self,
        monotonic_time: float,
        root_quat: Iterable[float],
        root_gyro: Iterable[float],
        joint_qpos: Iterable[float],
        joint_qvel: Iterable[float],
    ) -> bool:
        now = float(monotonic_time)
        if not math.isfinite(now):
            raise ValueError("monotonic_time must be finite")
        reason = stable_stand_reason(
            root_quat,
            root_gyro,
            joint_qpos,
            joint_qvel,
            limits=self.limits,
        )
        self.last_reason = reason
        if reason is not None:
            self._stable_since = None
            return False
        if self._stable_since is None:
            self._stable_since = now
        return now - self._stable_since >= self.limits.stable_seconds

    def reset(self) -> None:
        self._stable_since = None
        self.last_reason = "stable dwell was reset"


def validate_policy_stand_target(
    measured_joint_qpos: Iterable[float],
    policy_joint_target: Iterable[float],
    *,
    limits: RemoteStandThresholds | None = None,
) -> np.ndarray:
    """Reject a discontinuous first feedback-policy target before LowCmd opens."""

    limits = limits or RemoteStandThresholds()
    measured = _finite_vector(measured_joint_qpos, 29, "measured_joint_qpos")
    target = _finite_vector(policy_joint_target, 29, "policy_joint_target")
    delta = np.abs(target - measured)
    leg_delta = float(np.max(delta[:15]))
    full_delta = float(np.max(delta))
    if leg_delta > limits.maximum_policy_leg_target_delta:
        raise RemoteHandoverError(
            "policy standing target is discontinuous at the legs/waist "
            f"({leg_delta:.3f} rad)"
        )
    if full_delta > limits.maximum_policy_full_target_delta:
        raise RemoteHandoverError(
            f"policy standing target is discontinuous ({full_delta:.3f} rad)"
        )
    return target.copy()


def validate_shadow_target_match(
    shadow_target: Iterable[float],
    first_real_target: Iterable[float],
    *,
    maximum_delta: float = 0.08,
) -> np.ndarray:
    """Require the first commanded policy target to match its unsent dry run.

    The measured-pose continuity check above limits the absolute first command.
    This second check catches policy-history or reference-reset mistakes between
    the final shadow inference and the first inference that may reach LowCmd.
    """

    if not math.isfinite(float(maximum_delta)) or float(maximum_delta) <= 0.0:
        raise ValueError("maximum_delta must be positive and finite")
    expected = _finite_vector(shadow_target, 29, "shadow_target")
    actual = _finite_vector(first_real_target, 29, "first_real_target")
    delta = float(np.max(np.abs(actual - expected)))
    if delta > float(maximum_delta):
        raise RemoteHandoverError(
            "first policy target does not match the validated shadow target "
            f"({delta:.3f} rad)"
        )
    return actual.copy()


class MotionOwnerMonitor:
    """Observe ``CheckMode`` off the real-time policy thread.

    ``CheckMode`` is an RPC and may block up to the SDK timeout.  The 50 Hz
    policy thread therefore reads only this monitor's event/timestamp.  A
    non-empty owner, an RPC error, or a stale successful observation is a
    fail-closed ownership fault.
    """

    def __init__(
        self,
        motion_switcher: Any,
        *,
        expected_owner: str = "",
        poll_interval: float = 0.05,
        maximum_age: float = 0.25,
    ) -> None:
        if poll_interval <= 0.0 or maximum_age <= poll_interval:
            raise ValueError("owner monitor timing must satisfy 0 < poll < age")
        self._switcher = motion_switcher
        self._expected_owner = str(expected_owner)
        self._poll_interval = float(poll_interval)
        self._maximum_age = float(maximum_age)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._fault = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_success = 0.0
        self._fault_detail = "owner monitor has not started"

    @property
    def faulted(self) -> bool:
        return self._fault.is_set()

    @property
    def fault_detail(self) -> str:
        with self._lock:
            return self._fault_detail

    def _set_fault(self, detail: str) -> None:
        with self._lock:
            self._fault_detail = str(detail)
        self._fault.set()

    def start_after_validated_check(self) -> None:
        """Start after the caller synchronously validated the empty owner."""

        if self._thread is not None:
            raise RuntimeError("owner monitor already started")
        with self._lock:
            self._last_success = time.monotonic()
            self._fault_detail = ""
        self._thread = threading.Thread(
            target=self._run,
            name="g1-motion-owner-monitor",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set() and not self._fault.is_set():
            try:
                owner = checked_motion_owner(
                    self._switcher,
                    context="in the independent LowCmd ownership monitor",
                )
                if owner != self._expected_owner:
                    rendered = owner or "<none>"
                    self._set_fault(
                        "MotionSwitcher owner changed while LowCmd was active "
                        f"(owner={rendered!r})"
                    )
                    return
                with self._lock:
                    self._last_success = time.monotonic()
            except BaseException as exc:
                self._set_fault(
                    "MotionSwitcher ownership became unknown: "
                    f"{type(exc).__name__}: {exc}"
                )
                return
            self._stop.wait(self._poll_interval)

    def require_fresh_empty_owner(self) -> None:
        """Non-blocking real-time guard used immediately before LowCmd writes."""

        if self._fault.is_set():
            raise RemoteHandoverError(self.fault_detail)
        with self._lock:
            age = time.monotonic() - self._last_success
        if age > self._maximum_age:
            self._set_fault(
                "MotionSwitcher ownership observation is stale "
                f"({age:.3f}s > {self._maximum_age:.3f}s)"
            )
            raise RemoteHandoverError(self.fault_detail)

    def safe_for_damping(self) -> bool:
        """Return true only while an empty-owner observation remains fresh."""

        try:
            self.require_fresh_empty_owner()
        except RemoteHandoverError:
            return False
        return True

    def stop_and_join(self, timeout: float = 2.5) -> None:
        self._stop.set()
        thread = self._thread
        if thread is None:
            return
        thread.join(float(timeout))
        if thread.is_alive():
            self._set_fault("MotionSwitcher owner monitor did not stop in time")
            raise RemoteHandoverError(self.fault_detail)


def create_motion_switcher_client() -> Any:
    """Create the official read-only ownership client used by this workflow."""

    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
        MotionSwitcherClient,
    )

    client = MotionSwitcherClient()
    client.SetTimeout(2.0)
    client.Init()
    return client


def checked_motion_owner(
    motion_switcher: Any,
    *,
    context: str,
) -> str:
    """Return a validated owner name without selecting or releasing a mode."""

    status, result = motion_switcher.CheckMode()
    if int(status) != 0:
        raise RemoteHandoverError(
            f"Unitree CheckMode failed {context}: status={status!r}"
        )
    if not isinstance(result, dict) or not isinstance(result.get("name"), str):
        raise RemoteHandoverError("Unitree CheckMode returned an invalid response")
    return result["name"].strip()
