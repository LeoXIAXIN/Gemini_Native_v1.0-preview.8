"""In-memory task/session registry keyed by session ID."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as _datetime
import threading
import uuid

from src.domain.enums import TaskPhase


def _iso_now() -> str:
    return _datetime.datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    created_at: str
    user_id: str
    backend: str | None = None
    execution_mode: str | None = None
    robot_id: str | None = None
    status: TaskPhase = TaskPhase.STOPPED


class SessionService:
    """Thread-safe in-memory session registry."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, SessionRecord] = {}
        self._current: str | None = None

    def create(
        self,
        *,
        backend: str | None = None,
        execution_mode: str | None = None,
        robot_id: str | None = None,
        user_id: str = "operator",
    ) -> SessionRecord:
        with self._lock:
            record = SessionRecord(
                session_id=uuid.uuid4().hex,
                created_at=_iso_now(),
                user_id=user_id,
                backend=backend,
                execution_mode=execution_mode,
                robot_id=robot_id,
            )
            self._sessions[record.session_id] = record
            self._current = record.session_id
            return record

    def get(self, session_id: str) -> SessionRecord:
        with self._lock:
            return self._sessions[session_id]

    def current(self) -> SessionRecord | None:
        with self._lock:
            if self._current is None:
                return None
            return self._sessions.get(self._current)

    def update_status(self, session_id: str, status: TaskPhase) -> SessionRecord:
        with self._lock:
            record = self._sessions[session_id]
            updated = SessionRecord(
                session_id=record.session_id,
                created_at=record.created_at,
                user_id=record.user_id,
                backend=record.backend,
                execution_mode=record.execution_mode,
                robot_id=record.robot_id,
                status=status,
            )
            self._sessions[session_id] = updated
            return updated

    def list(self) -> tuple[SessionRecord, ...]:
        with self._lock:
            return tuple(self._sessions.values())

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()
            self._current = None
