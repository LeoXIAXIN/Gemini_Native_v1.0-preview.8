"""Motion/algorithm layer: semantic facades over the migrated motion code.

Every service here delegates to the migrated implementations — no algorithm is
re-implemented.  Runtime annotations: GMR side is runtime-gmr (3.10), policy
and simulation run under runtime-hgpt (3.12).

Import-light by design (PEP 562): ``import src.motion`` never eagerly loads
simulation/MuJoCo/Unitree/ONNX chains, so GMR entries do not pay for policy
dependencies and vice versa.  Submodules resolve lazily too
(``from src.motion import gmr_bridge``).
"""

from __future__ import annotations

import importlib
from importlib.util import find_spec

_SERVICE_ATTRS: dict[str, tuple[str, str]] = {
    "BACKENDS": ("src.motion.simulation", "BACKENDS"),
    "GMRService": ("src.motion.gmr_service", "GMRService"),
    "MotionExecutionBackend": ("src.motion.simulation", "MotionExecutionBackend"),
    "MujocoBackend": ("src.motion.simulation", "MujocoBackend"),
    "PolicyService": ("src.motion.policy", "PolicyService"),
    "ReplayService": ("src.motion.replay", "ReplayService"),
    "SimulationService": ("src.motion.simulation", "SimulationService"),
    "UnitreeBackend": ("src.motion.simulation", "UnitreeBackend"),
}

__all__ = [
    "BACKENDS",
    "GMRService",
    "MotionExecutionBackend",
    "MujocoBackend",
    "PolicyService",
    "ReplayService",
    "SimulationService",
    "UnitreeBackend",
]


def __getattr__(name: str):
    if name in _SERVICE_ATTRS:
        module_name, attribute = _SERVICE_ATTRS[name]
        return getattr(importlib.import_module(module_name), attribute)
    if find_spec(f"src.motion.{name}"):
        return importlib.import_module(f"src.motion.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
