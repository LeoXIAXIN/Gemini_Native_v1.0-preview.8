"""M8 SafetyPolicy: facade over the migrated safety modules (single source).

RUNTIME: dual-runtime neutral for constants; state-machine use is gmr-side.

All four modules (core/reference/protocol/adapters) are the migrated src
implementations; no legacy g1_safety package is imported anywhere.
"""

from __future__ import annotations

import importlib
from typing import Any


class SafetyPolicy:
    """Delegating wrapper around the migrated safety modules."""

    @property
    def core(self) -> Any:
        # M-decoupling: single source of truth (no legacy package).
        return importlib.import_module("src.safety.g1_core")

    @property
    def reference(self) -> Any:
        # M-decoupling: the migrated reference module (no legacy import).
        return importlib.import_module("src.safety.g1_reference")

    @property
    def protocol(self) -> Any:
        # M-decoupling: the migrated protocol module (no legacy import).
        return importlib.import_module("src.safety.g1_protocol")

    @property
    def policy_adapters(self) -> Any:
        # M-decoupling: the migrated adapters module (no legacy import).
        return importlib.import_module("src.safety.g1_policy_adapters")

    # --- pure delegations -------------------------------------------------

    def compute_dof_hash(self, names: Any) -> str:
        return self.core.compute_dof_hash(names)

    def minimum_jerk(self, progress: float) -> float:
        return self.core.minimum_jerk(progress)

    def make_gateway(self, *args: Any, **kwargs: Any) -> Any:
        """Construct the legacy G1SafetyGateway (all logic stays legacy)."""
        return self.core.G1SafetyGateway(*args, **kwargs)

    def make_profile(self, *args: Any, **kwargs: Any) -> Any:
        return self.core.SafetyProfile(*args, **kwargs)

    def make_reference_watchdog(self, *args: Any, **kwargs: Any) -> Any:
        return self.reference.ReferenceWatchdog(*args, **kwargs)

    def twist25_to_g1_29(self, twist_values: Any, measured_g1_values: Any) -> Any:
        """Expand TWIST 25 joints into canonical G1 29 (returns (values, mask))."""
        return self.policy_adapters.twist25_to_g1_29(
            twist_values, measured_g1_values
        )

    def g1_29_to_twist25(self, values: Any) -> Any:
        return self.policy_adapters.g1_29_to_twist25(values)

    def encode_policy_target(self, *args: Any, **kwargs: Any) -> Any:
        return self.protocol.encode_policy_target(*args, **kwargs)

    def decode_policy_target(self, *args: Any, **kwargs: Any) -> Any:
        return self.protocol.decode_policy_target(*args, **kwargs)
