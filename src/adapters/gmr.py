"""GMR adapter that lazily loads the canonical ``src.motion`` modules."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np


def _exact() -> Any:
    return importlib.import_module("src.motion.gmr_exact")


def _twist() -> Any:
    return importlib.import_module("src.motion.gmr_twist")


class GMRAdapter:
    """Retargeting/conversion facade; no algorithm is re-implemented here."""

    @property
    def body_names(self) -> list[str]:
        return list(_exact().BODY_NAMES)

    @property
    def gmr_29_dof_names(self) -> list[str]:
        return list(_twist().GMR_29_DOF_NAMES)

    @property
    def twist_25_dof_names(self) -> list[str]:
        return list(_twist().TWIST_25_DOF_NAMES)

    @property
    def default_mimic_obs(self) -> np.ndarray:
        return np.array(_twist().DEFAULT_MIMIC_OBS)

    def make_exact_gmr_frame(
        self, positions_mm: np.ndarray, local_xyzw: np.ndarray
    ) -> dict[str, list[np.ndarray]]:
        """CHINGMU live (mm, Z-up) -> GMR human frame (exact legacy math)."""
        return _exact().make_exact_gmr_frame(positions_mm, local_xyzw)

    def make_root_velocity_estimator(self, window: int = 5) -> Any:
        return _twist().RootVelocityEstimator(window=window)

    def gmr_qpos_to_twist_mimic(
        self, qpos: np.ndarray, timestamp: float, velocity_estimator: Any
    ) -> np.ndarray:
        """GMR 36-value qpos -> TWIST 33-value mimic target."""
        return _twist().gmr_qpos_to_twist_mimic(
            qpos, timestamp, velocity_estimator
        )
