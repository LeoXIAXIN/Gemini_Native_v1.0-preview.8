"""M10 LogService: session log access + legacy log classification."""

from __future__ import annotations

from typing import Any

from src.infrastructure.logging import LogClassification


class LogService:
    """Read-only log facade over the injected legacy manager."""

    def __init__(self, manager: Any) -> None:
        self._manager = manager
        self._classification = LogClassification()

    def recent(self, limit: int = 200) -> list[dict[str, Any]]:
        return list(self._manager.status(log_limit=limit).get("logs", []))

    def downloadable_log(self) -> Any:
        return self._manager.downloadable_log()

    def classify_component(self, line: str, backend: str | None = None) -> str:
        return self._classification.component(line, backend)

    def classify_level(self, line: str) -> str:
        return self._classification.level(line)
