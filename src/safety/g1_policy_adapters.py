"""Joint-order adapters at the policy/safety boundary.

The safety gateway always reasons about the canonical 29-motor G1 order.  The
released TWIST policy controls a 25-DoF abstraction: it omits wrist pitch/yaw
on both sides.  Missing motors are filled from measured state, never guessed
from array length and never silently commanded to zero.

Import-cycle note: ``g1_safety.core`` classes are resolved lazily at call time
so importing this module remains lightweight.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys
from pathlib import Path as _Path
from typing import Sequence, Tuple

import numpy as np

# Canonical 29-motor joint order shared with the safety core.
_G1_29DOF_JOINT_NAMES: Tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


def _core():
    # M-decoupling: the core state machine is the single source of truth in
    # src.safety.g1_core; no legacy package is involved.
    return _importlib.import_module("src.safety.g1_core")


def _validation_error(message: str) -> Exception:
    return _core().ValidationError(message)


TWIST_25_CANONICAL_IDS: Tuple[int, ...] = tuple(range(20)) + tuple(range(22, 27))
TWIST_25_JOINT_NAMES: Tuple[str, ...] = tuple(
    _G1_29DOF_JOINT_NAMES[index] for index in TWIST_25_CANONICAL_IDS
)
TWIST_MISSING_WRIST_IDS: Tuple[int, ...] = (20, 21, 27, 28)


def _finite_vector(values: Sequence[float], size: int, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape != (size,):
        raise _validation_error(f"{label} shape {array.shape}, expected ({size},)")
    if not np.all(np.isfinite(array)):
        raise _validation_error(f"{label} contains NaN or infinity")
    return array.copy()


def twist25_to_g1_29(
    twist_values: Sequence[float], measured_g1_values: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Expand TWIST's explicit 25 joints into canonical G1 order.

    Returns ``(values29, active_mask29)``.  The four unsupported wrist motors
    hold their current measured values and are marked inactive so a transport
    cannot mistake them for policy-owned joints.
    """

    values25 = _finite_vector(twist_values, 25, "TWIST target")
    measured29 = _finite_vector(measured_g1_values, 29, "measured G1 state")
    result = measured29.copy()
    result[np.asarray(TWIST_25_CANONICAL_IDS, dtype=np.intp)] = values25
    active = np.zeros(29, dtype=bool)
    active[np.asarray(TWIST_25_CANONICAL_IDS, dtype=np.intp)] = True
    return result, active


def g1_29_to_twist25(values: Sequence[float]) -> np.ndarray:
    """Project canonical G1 values into the released TWIST actuator order."""

    values29 = _finite_vector(values, 29, "canonical G1 values")
    return values29[np.asarray(TWIST_25_CANONICAL_IDS, dtype=np.intp)].copy()
