"""MuJoCo adapter: facade over the migrated src/motion simulation modules.

M-decoupling: no legacy imports.  Requires the hgpt runtime (mujoco); all
stepping/control math stays in the migrated MJSim implementation.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from src.domain.errors import DomainError


def _sim_mj() -> Any:
    try:
        return importlib.import_module("src.motion.sim_mj")
    except (ImportError, ModuleNotFoundError) as exc:
        raise DomainError(
            f"MuJoCo adapter requires the Humanoid-GPT runtime: {exc}"
        ) from exc


def _sim_base() -> Any:
    try:
        return importlib.import_module("src.motion.sim_base")
    except (ImportError, ModuleNotFoundError) as exc:
        raise DomainError(
            f"MuJoCo adapter requires the Humanoid-GPT runtime: {exc}"
        ) from exc


class MujocoAdapter:
    """Delegating wrapper around the migrated MuJoCo simulation utilities."""

    def get_qpos_ids(self, model: Any, names: list[str]) -> np.ndarray:
        return _sim_mj().get_qpos_ids(model, names)

    def get_dof_ids(self, model: Any, names: list[str]) -> np.ndarray:
        return _sim_mj().get_dof_ids(model, names)

    def get_sensor_data(
        self, model: Any, data: Any, sensor_name: str
    ) -> np.ndarray:
        return _sim_mj().get_sensor_data(model, data, sensor_name)

    def make_sim(
        self, xml_path: str, ctrl_dt: float = 0.02, sim_dt: float = 0.001,
        headless: bool = False,
    ) -> Any:
        return _sim_mj().MJSim(
            xml_path, ctrl_dt=ctrl_dt, sim_dt=sim_dt, headless=headless
        )

    def make_state(self, mj_data: Any | None = None, info: dict | None = None) -> Any:
        return _sim_base().State(mj_data=mj_data, mjx_data=None, info=info)
