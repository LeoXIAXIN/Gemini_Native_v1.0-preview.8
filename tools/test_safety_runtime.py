#!/usr/bin/env python3
"""Focused tests for the transport-independent G1 safety state machine."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.safety.g1_core import (  # noqa: E402
    G1SafetyGateway,
    RobotState,
    SafetyProfile,
    SafetyState,
)


def main() -> int:
    profile = SafetyProfile.simulation_default()
    gateway = G1SafetyGateway(profile)

    assert gateway.state is SafetyState.DISARMED
    assert not gateway.arm(0.0), "arming without a held deadman must fail"
    gateway.set_deadman(True, 0.001)
    assert gateway.arm(0.002)
    assert gateway.state is SafetyState.ARMING

    gateway.emergency_stop(0.003)
    assert gateway.state is SafetyState.E_STOP
    assert gateway.estop_latched

    robot = RobotState(
        timestamp=0.004,
        tick=1,
        q=profile.standby_q.copy(),
        dq=np.zeros(profile.dof),
        imu_rpy=np.zeros(3),
        imu_gyro=np.zeros(3),
        model=profile.model,
        dof_hash=profile.dof_hash,
    )
    assert not gateway.reset(
        0.004,
        robot,
        operator_confirmed=False,
        physical_estop_released=True,
    )
    assert gateway.reset(
        0.004,
        robot,
        operator_confirmed=True,
        physical_estop_released=True,
    )
    assert gateway.state is SafetyState.DISARMED
    assert not gateway.estop_latched

    try:
        profile.assert_transport_allowed("unitree")
    except PermissionError:
        pass
    else:
        raise AssertionError("simulation profile allowed a hardware transport")

    print("G1 SAFETY RUNTIME TEST PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
