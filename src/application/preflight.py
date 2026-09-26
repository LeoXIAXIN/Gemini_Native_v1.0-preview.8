"""M10 PreflightService: delegate preflight and map to the domain model."""

from __future__ import annotations

from typing import Any, Mapping

from src.domain.models import CheckItem, PreflightResult


def preflight_result_from_payload(payload: Mapping[str, Any]) -> PreflightResult:
    """Map the legacy preflight dict to ``PreflightResult`` (no re-checking)."""
    checks = tuple(
        CheckItem(
            name=str(item.get("name", "")),
            ok=bool(item.get("ok", False)),
            message=str(item.get("message", "")),
            required=bool(item.get("required", True)),
        )
        for item in payload.get("checks", [])
        if isinstance(item, Mapping)
    )
    errors = tuple(str(error) for error in payload.get("errors", []))
    ok = bool(payload.get("ok", not errors))
    return PreflightResult(ok=ok, checks=checks, errors=errors)


class PreflightService:
    """Runs the legacy preflight (inspection only; never starts anything)."""

    def __init__(self, manager: Any) -> None:
        self._manager = manager

    def run(self, config: dict[str, Any] | None = None) -> PreflightResult:
        payload = self._manager.preflight(config)
        return preflight_result_from_payload(payload)
