"""Deprecated import aliases for the official Unitree SDK2 HG message types.

Existing integrations may still import these historical names. They resolve
to the SDK classes and factory themselves; this module defines no alternate
IDL, byte-field schema, firmware type identifier, or serialization path.
"""

from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_ as create_firmware_lowcmd,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
    IMUState_ as FirmwareIMUState_,
    LowCmd_ as FirmwareLowCmd_,
    LowState_ as FirmwareLowState_,
    MotorCmd_ as FirmwareMotorCmd_,
    MotorState_ as FirmwareMotorState_,
)


def firmware_lowstate_type_id() -> object:
    """Return the official SDK LowState XTypes identifier."""
    return FirmwareLowState_.__idl__.get_type_id()


def firmware_lowcmd_type_id() -> object:
    """Return the official SDK LowCmd XTypes identifier."""
    return FirmwareLowCmd_.__idl__.get_type_id()
