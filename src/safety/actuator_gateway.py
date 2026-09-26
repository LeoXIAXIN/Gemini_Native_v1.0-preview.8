"""M7 ActuatorGateway: lazy access to the legacy LowCmd/LowState writer.

RUNTIME: hgpt (3.12) for the real_robot module.

HARD RULE (frozen, guide §7): debug mode may only subscribe to LowState and
must never create a LowCmd publisher; real output additionally requires the
one-shot authorization chain.  This gateway exposes no publisher constructor
of its own.
"""

from __future__ import annotations

from typing import Any

from src.adapters.unitree import UnitreeAdapter


class ActuatorGateway:
    """Facade over the process-isolated DDS control boundary.

    This gateway exposes NO publisher constructor and never returns the
    legacy ``deploy/real_robot`` module: ``real_robot_module`` always raises
    DomainError (M7b boundary).  Real LowCmd control runs inside the
    ``deploy.play_track`` subprocess launched by the orchestration layer.
    """

    def __init__(self, adapter: UnitreeAdapter | None = None) -> None:
        self._adapter = adapter or UnitreeAdapter()

    @property
    def real_robot_module(self) -> Any:
        """M7b boundary marker: access always raises DomainError.

        The legacy DDS control module is process-isolated; src never imports
        it in-process (see src.adapters.unitree.UnitreeAdapter).
        """
        return self._adapter.real_robot_module

    def configure_windows_interface(self, interface_address: str) -> str:
        return self._adapter.configure_windows_interface(interface_address)
