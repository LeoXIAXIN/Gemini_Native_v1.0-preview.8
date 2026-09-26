"""Domain error hierarchy.

New code raises these; adapters translate legacy exceptions into them so the
controller layer never depends on the old module's exception classes directly.
"""

from __future__ import annotations


class DomainError(Exception):
    """Base class for every error raised by the new src/ layers."""


class ValidationError(DomainError):
    """A domain value violates its own shape/range contract."""


class CodecError(ValidationError):
    """A legacy payload cannot be decoded or re-encoded."""


class FreshnessError(DomainError):
    """Consumed data exceeded its freshness deadline (fail-closed)."""


class ConfigurationError(DomainError):
    """Configuration is absent, invalid or was rejected by validation."""


class SafetyViolationError(DomainError):
    """A safety gate refused an action; the request must not be bypassed."""


class AuthorizationError(DomainError):
    """Real-output authorization is absent, stale or mismatched."""
