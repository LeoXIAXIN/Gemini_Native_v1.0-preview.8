#!/usr/bin/env python3
"""Offline contract check: legacy imports are exact official SDK aliases.

Imports IDL and CRC only. This script never constructs a DDS channel.
"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapters.unitree_hg_compat import (  # noqa: E402
    FirmwareIMUState_,
    FirmwareLowCmd_,
    FirmwareLowState_,
    FirmwareMotorCmd_,
    FirmwareMotorState_,
    create_firmware_lowcmd,
    firmware_lowcmd_type_id,
    firmware_lowstate_type_id,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (  # noqa: E402
    IMUState_, LowCmd_, LowState_, MotorCmd_, MotorState_,
)
from unitree_sdk2py.utils.crc import CRC  # noqa: E402


def main() -> int:
    assert FirmwareIMUState_ is IMUState_
    assert FirmwareMotorCmd_ is MotorCmd_
    assert FirmwareMotorState_ is MotorState_
    assert FirmwareLowState_ is LowState_
    assert FirmwareLowCmd_ is LowCmd_
    assert create_firmware_lowcmd is unitree_hg_msg_dds__LowCmd_
    assert firmware_lowstate_type_id() == LowState_.__idl__.get_type_id()
    assert firmware_lowcmd_type_id() == LowCmd_.__idl__.get_type_id()

    command = create_firmware_lowcmd()
    assert type(command) is LowCmd_
    assert len(command.motor_cmd) == 35
    assert CRC().Crc(command) == CRC().Crc(unitree_hg_msg_dds__LowCmd_())
    print("UNITREE HG OFFICIAL SDK ALIAS CONTRACT TEST PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
