"""Adapter layer: transport, SDK and runtime boundaries.

Import-light by design: this package imports NO submodules at package import
time.  Public names are resolved lazily through ``__getattr__`` (PEP 562), so
``import src.adapters`` has zero transitive side effects and cannot create
cross-layer import cycles.  Prefer ``from src.adapters.<module> import X`` for
hot paths.

No adapter contains user-flow decisions, and no adapter writes config.
"""

from __future__ import annotations

import importlib
from importlib.util import find_spec

_LAZY: dict[str, tuple[str, str]] = {
    "ChingmuAdapter": ("src.adapters.chingmu", "ChingmuAdapter"),
    "GMRAdapter": ("src.adapters.gmr", "GMRAdapter"),
    "LicenseAdapter": ("src.adapters.license", "LicenseAdapter"),
    "MujocoAdapter": ("src.adapters.mujoco", "MujocoAdapter"),
    "OnnxRuntimeAdapter": ("src.adapters.onnx", "OnnxRuntimeAdapter"),
    "RuntimeAdapter": ("src.adapters.runtime", "RuntimeAdapter"),
    "StateStoreClient": ("src.adapters.state_store", "StateStoreClient"),
    "StateStoreKeys": ("src.adapters.state_store", "StateStoreKeys"),
    "UnitreeAdapter": ("src.adapters.unitree", "UnitreeAdapter"),
    "parse_bridge_status": ("src.adapters.state_store", "parse_bridge_status"),
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY[name]
    except KeyError:
        if find_spec(f"src.adapters.{name}"):
            return importlib.import_module(f"src.adapters.{name}")
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    return getattr(importlib.import_module(module_name), attribute)
