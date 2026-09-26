"""Safety layer for simulation and real-output authorization.

The three-tier structure:
  * ``SafetyPolicy``      -> state machine / watchdog / blends
  * ``HardwareGate``      -> LowState acceptance, ARM G1, physical remote gates
  * ``ActuatorGateway``   -> LowCmd writer access (never constructed here)

Facades never bypass the state machine's latching rules.
"""

from src.safety.actuator_gateway import ActuatorGateway
from src.safety.hardware_gate import HardwareGate
from src.safety.policy import SafetyPolicy

__all__ = ["ActuatorGateway", "HardwareGate", "SafetyPolicy"]
