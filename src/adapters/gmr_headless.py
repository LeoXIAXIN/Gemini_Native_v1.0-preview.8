"""Load GMR's retargeting core without importing its optional viewer.

The bundled third-party package eagerly imports ``RobotMotionViewer`` from its
``__init__`` module.  That imports ``mujoco.viewer`` and the unsigned
``_simulate.pyd`` extension even in services that never create a window.  Some
Windows Application Control policies reject that extension, which used to make
the headless retargeting preflight fail for an unrelated visualization module.

Load the upstream ``motion_retarget`` module under a private package alias.  Its
relative imports still resolve against the original package directory, while
the upstream ``__init__`` (and therefore ``mujoco.viewer``) is never executed.
Viewer processes continue to import the public third-party package normally.
"""

from __future__ import annotations

from importlib import import_module
from importlib.machinery import ModuleSpec, PathFinder
from pathlib import Path
import sys
import threading
from types import ModuleType
from typing import Any


_UPSTREAM_PACKAGE = "general_motion_retargeting"
_HEADLESS_PACKAGE = "_chingmu_gmr_headless"
_HEADLESS_MODULE = f"{_HEADLESS_PACKAGE}.motion_retarget"
_LOAD_LOCK = threading.RLock()


def _upstream_package_directory() -> Path:
    spec = PathFinder.find_spec(_UPSTREAM_PACKAGE, sys.path)
    locations = None if spec is None else spec.submodule_search_locations
    if not locations:
        raise ModuleNotFoundError(
            "general_motion_retargeting package is not installed in this runtime"
        )
    package_directory = Path(next(iter(locations))).resolve()
    if not (package_directory / "motion_retarget.py").is_file():
        raise ImportError(
            "general_motion_retargeting.motion_retarget is missing from "
            f"{package_directory}"
        )
    return package_directory


def _install_private_package(package_directory: Path) -> ModuleType:
    package = ModuleType(_HEADLESS_PACKAGE)
    package.__file__ = str(package_directory / "__init__.py")
    package.__package__ = _HEADLESS_PACKAGE
    package.__path__ = [str(package_directory)]
    package.__spec__ = ModuleSpec(
        _HEADLESS_PACKAGE,
        loader=None,
        is_package=True,
    )
    package.__spec__.submodule_search_locations = package.__path__
    sys.modules[_HEADLESS_PACKAGE] = package
    return package


def load_general_motion_retargeting() -> type[Any]:
    """Return the upstream retargeter class without importing its viewer."""

    with _LOAD_LOCK:
        loaded = sys.modules.get(_HEADLESS_MODULE)
        if loaded is None:
            package_directory = _upstream_package_directory()
            _install_private_package(package_directory)
            try:
                loaded = import_module(_HEADLESS_MODULE)
            except BaseException:
                for name in tuple(sys.modules):
                    if name == _HEADLESS_PACKAGE or name.startswith(
                        f"{_HEADLESS_PACKAGE}."
                    ):
                        sys.modules.pop(name, None)
                raise
        retargeter = getattr(loaded, "GeneralMotionRetargeting", None)
        if not isinstance(retargeter, type):
            raise ImportError(
                "GMR motion_retarget module has no GeneralMotionRetargeting class"
            )
        return retargeter
