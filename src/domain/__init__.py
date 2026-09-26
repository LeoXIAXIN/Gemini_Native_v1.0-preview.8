"""Domain layer: pure data models, enums, errors and wire codecs.

Design rules:
  * every model documents type, dimensions, units, coordinate frame, time base,
    source/target, sequence number, schema version and freshness requirement;
  * this package never imports external SDKs, HTTP or subprocess;
  * conversion from/to established dict/pickle/JSON formats lives in
    ``legacy_codec.LegacyCodec`` only.

Import-light by design (PEP 562): importing the package loads no submodule, so
layers that only need ``src.domain.errors`` (for example the stdlib-only
controller) never pay for the numpy-backed models/codecs.  Public names and
submodules resolve lazily.
"""

from __future__ import annotations

import importlib
from importlib.util import find_spec

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "Backend": ("src.domain.enums", "Backend"),
    "BridgeStatus": ("src.domain.enums", "BridgeStatus"),
    "ExecutionMode": ("src.domain.enums", "ExecutionMode"),
    "ReferenceHealth": ("src.domain.enums", "ReferenceHealth"),
    "RobotStateSource": ("src.domain.enums", "RobotStateSource"),
    "SafetyState": ("src.domain.enums", "SafetyState"),
    "TaskPhase": ("src.domain.enums", "TaskPhase"),
    "AuthorizationError": ("src.domain.errors", "AuthorizationError"),
    "CodecError": ("src.domain.errors", "CodecError"),
    "ConfigurationError": ("src.domain.errors", "ConfigurationError"),
    "DomainError": ("src.domain.errors", "DomainError"),
    "FreshnessError": ("src.domain.errors", "FreshnessError"),
    "SafetyViolationError": ("src.domain.errors", "SafetyViolationError"),
    "ValidationError": ("src.domain.errors", "ValidationError"),
    "AuthorizationContext": ("src.domain.models", "AuthorizationContext"),
    "CheckItem": ("src.domain.models", "CheckItem"),
    "G1Reference": ("src.domain.models", "G1Reference"),
    "MocapFrame": ("src.domain.models", "MocapFrame"),
    "MotorCommand": ("src.domain.models", "MotorCommand"),
    "PipelineStatus": ("src.domain.models", "PipelineStatus"),
    "PolicyObservation": ("src.domain.models", "PolicyObservation"),
    "PreflightResult": ("src.domain.models", "PreflightResult"),
    "RobotState": ("src.domain.models", "RobotState"),
    "SafeCommand": ("src.domain.models", "SafeCommand"),
    "SafetyStatus": ("src.domain.models", "SafetyStatus"),
    "LegacyCodec": ("src.domain.legacy_codec", "LegacyCodec"),
}

__all__ = [
    "AuthorizationContext",
    "AuthorizationError",
    "Backend",
    "BridgeStatus",
    "CheckItem",
    "CodecError",
    "ConfigurationError",
    "DomainError",
    "ExecutionMode",
    "FreshnessError",
    "G1Reference",
    "LegacyCodec",
    "MocapFrame",
    "MotorCommand",
    "PipelineStatus",
    "PolicyObservation",
    "PreflightResult",
    "ReferenceHealth",
    "RobotState",
    "RobotStateSource",
    "SafeCommand",
    "SafetyState",
    "SafetyStatus",
    "SafetyViolationError",
    "TaskPhase",
    "ValidationError",
]


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_name, attribute = _LAZY_ATTRS[name]
        return getattr(importlib.import_module(module_name), attribute)
    if find_spec(f"src.domain.{name}"):
        return importlib.import_module(f"src.domain.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
