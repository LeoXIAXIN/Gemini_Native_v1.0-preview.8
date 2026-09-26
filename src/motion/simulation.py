"""M6 SimulationService: MotionExecutionBackend protocol + legacy backends.

RUNTIME: hgpt (3.12) for real execution; this module's import is neutral.

The protocol is the guide's ``MotionExecutionBackend``
(``get_robot_state()`` / ``send_command()``).  ``MujocoBackend`` wraps the
legacy ``MJSim`` (utils/sim_mj.py); ``UnitreeBackend`` is a facade placeholder
that never creates a LowCmd publisher by itself (see M7 safety rules).
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from src.adapters.mujoco import MujocoAdapter
from src.adapters.unitree import UnitreeAdapter
from src.domain.enums import RobotStateSource
from src.domain.models import RobotState


class MotionExecutionBackend(Protocol):
    """Uniform state/command interface (guide §6.3)."""

    def get_robot_state(self) -> RobotState: ...

    def send_command(self, command: np.ndarray) -> None: ...


class MujocoBackend:
    """Facade over the legacy MJSim simulation loop."""

    def __init__(self, xml_path: str, *, headless: bool = True,
                 adapter: MujocoAdapter | None = None) -> None:
        self._adapter = adapter or MujocoAdapter()
        self._sim = self._adapter.make_sim(xml_path, headless=headless)
        self._state = self._sim.init_state()

    @property
    def mj_data(self) -> Any:
        return self._state.mj_data

    @property
    def mj_model(self) -> Any:
        return self._sim.mj_model

    def get_robot_state(self) -> RobotState:
        data = self.mj_data
        return RobotState(
            source=RobotStateSource.MUJOCO,
            qpos=np.asarray(data.qpos, dtype=float).copy(),
            qvel=np.asarray(data.qvel, dtype=float).copy(),
        )

    def send_command(self, command: np.ndarray) -> None:
        self._state = self._sim.step(self._state, np.asarray(command, dtype=float))

    def close(self) -> None:
        close = getattr(self._sim, "close", None)
        if close is not None:
            close()


class UnitreeBackend:
    """Facade placeholder; real control stays behind the hardware gates.

    This class intentionally exposes no publisher constructor.
    """

    def __init__(self, adapter: UnitreeAdapter | None = None) -> None:
        self._adapter = adapter or UnitreeAdapter()

    @property
    def real_robot_module(self) -> Any:
        return self._adapter.real_robot_module

    def get_robot_state(self) -> RobotState:
        raise NotImplementedError(
            "UnitreeBackend state reading must go through src.motion.real_robot"
        )

    def send_command(self, command: np.ndarray) -> None:
        raise NotImplementedError(
            "UnitreeBackend output requires the one-shot authorization chain "
            "and hardware gates; never call this directly"
        )


class SimulationService:
    """Backend registry for the supported execution modes."""

    def __init__(self) -> None:
        self._backends: dict[str, MotionExecutionBackend] = {}

    def register(self, name: str, backend: MotionExecutionBackend) -> None:
        self._backends[name] = backend

    def get(self, name: str) -> MotionExecutionBackend:
        if name not in self._backends:
            raise KeyError(f"unknown execution backend: {name}")
        return self._backends[name]


BACKENDS = {"mujoco": MujocoBackend, "unitree": UnitreeBackend}
