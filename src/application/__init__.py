"""Application layer: task/session services and controller-facing facades.

Import-light by design: no submodules are imported at package import time.
Public names are resolved lazily through ``__getattr__`` (PEP 562), so
``import src.application`` has zero transitive side effects and cannot create
cross-layer import cycles. Services receive their process manager by injection
and never instantiate it themselves.
"""

from __future__ import annotations

import importlib

_LAZY: dict[str, tuple[str, str]] = {
    "AuthorizationService": ("src.application.authorization",
                              "AuthorizationService"),
    "HealthService": ("src.application.health", "HealthService"),
    "LogService": ("src.application.logs", "LogService"),
    "PipelineService": ("src.application.pipeline", "PipelineService"),
    "PreflightService": ("src.application.preflight", "PreflightService"),
    "SessionRecord": ("src.application.session", "SessionRecord"),
    "SessionService": ("src.application.session", "SessionService"),
    "preflight_result_from_payload": ("src.application.preflight",
                                      "preflight_result_from_payload"),
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    return getattr(importlib.import_module(module_name), attribute)
