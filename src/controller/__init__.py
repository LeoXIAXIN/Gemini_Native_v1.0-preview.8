"""Controller layer: frozen HTTP route contract plus the migrated entries.

Import-light by design (PEP 562): ``import src.controller`` never eagerly
loads http_controller/application/domain, so it cannot pull numpy or other
scientific dependencies.  The migrated entries are:

  * ``src.controller.motion_control_app`` — canonical gemini controller (M10b)
  * ``src.controller.native_app``         — Windows native production entry
  * ``src.controller.http_controller``    — frozen route table + dispatch

``python src/controller/motion_control_app.py --help`` and
``python -m src.controller.motion_control_app --help`` behave identically;
``native_app`` additionally imports cleanly (``--help``) even with a
stdlib-only interpreter.
"""

from __future__ import annotations

import importlib

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "HttpControllerFacade": (
        "src.controller.http_controller",
        "HttpControllerFacade",
    ),
    "ROUTES": ("src.controller.http_controller", "ROUTES"),
    "Route": ("src.controller.http_controller", "Route"),
    "origin_is_local": ("src.controller.http_controller", "origin_is_local"),
}

_LAZY_MODULES: dict[str, str] = {
    "motion_control_app": "src.controller.motion_control_app",
    "native_app": "src.controller.native_app",
}

__all__ = [
    "ROUTES",
    "HttpControllerFacade",
    "Route",
    "origin_is_local",
    "motion_control_app",
    "native_app",
]


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_name, attribute = _LAZY_ATTRS[name]
        return getattr(importlib.import_module(module_name), attribute)
    if name in _LAZY_MODULES:
        return importlib.import_module(_LAZY_MODULES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
