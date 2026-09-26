"""HTTP controller facade with a stable route and dispatch contract.

It performs no HTTP serving and makes no policy decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib.parse import urlsplit

from src.application.health import HealthService
from src.application.pipeline import PipelineService
from src.application.preflight import PreflightService


@dataclass(frozen=True)
class Route:
    method: str
    path: str
    action: str


ROUTES: tuple[Route, ...] = (
    Route("GET", "/api/health", "health"),
    Route("GET", "/api/license", "license"),
    Route("GET", "/api/config", "get_config"),
    Route("PUT", "/api/config", "put_config"),
    Route("POST", "/api/config", "post_config"),
    Route("GET", "/api/status", "status"),
    Route("POST", "/api/preflight", "preflight"),
    Route("POST", "/api/start", "start"),
    Route("POST", "/api/stop", "stop"),
    Route("POST", "/api/force-stop", "force_stop"),
    Route("POST", "/api/cleanup", "cleanup"),
    Route("POST", "/api/shutdown", "shutdown"),
    Route("POST", "/api/safety/estop", "safety_estop"),
    Route("POST", "/api/safety/reset", "safety_reset"),
    Route("GET", "/api/logs/download", "logs_download"),
)


def origin_is_local(origin: str | None, server_port: int) -> bool:
    """Pure mirror of the legacy ``_origin_is_local`` check."""
    if not origin:
        return True
    try:
        parsed = urlsplit(origin)
        origin_port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return (
            parsed.scheme == "http"
            and parsed.hostname in {"127.0.0.1", "localhost"}
            and origin_port == server_port
        )
    except ValueError:
        return False


class HttpControllerFacade:
    """Dispatch-only facade over the injected legacy manager services."""

    def __init__(
        self,
        manager: Any,
        *,
        pipeline: PipelineService | None = None,
        preflight: PreflightService | None = None,
        health: HealthService | None = None,
        license_adapter: Any = None,
    ) -> None:
        self._manager = manager
        self._pipeline = pipeline or PipelineService(manager)
        self._preflight = preflight or PreflightService(manager)
        self._health = health or HealthService(manager)

        from src.adapters.license import LicenseAdapter
        self._license = license_adapter or LicenseAdapter()

    @property
    def routes(self) -> tuple[Route, ...]:
        return ROUTES

    def dispatch(
        self,
        method: str,
        path: str,
        payload: Any = None,
        *,
        authorization: str | None = None,
        license_refresh: bool = False,
    ) -> tuple[int, Any]:
        """Map one (method, path) to the same semantics as the legacy handler."""
        method = method.upper()
        route = next(
            (entry for entry in ROUTES
             if entry.method == method and entry.path == path),
            None,
        )
        if route is None:
            return (
                HTTPStatus.NOT_FOUND,
                {"ok": False, "error": "Unknown API endpoint"},
            )
        action = route.action
        if action == "health":
            return HTTPStatus.OK, self._health.snapshot()
        if action == "license":
            if license_refresh:
                return (
                    HTTPStatus.OK,
                    self._license.check_product_license().to_public_dict(),
                )
            return HTTPStatus.OK, self._license.product_license_payload()
        if action == "get_config":
            return HTTPStatus.OK, self._pipeline.config_payload()
        if action in {"put_config", "post_config"}:
            proposed = (
                payload.get("config", payload)
                if isinstance(payload, dict)
                else payload
            )
            return HTTPStatus.OK, self._pipeline.update_config(proposed)
        if action == "status":
            return HTTPStatus.OK, self._pipeline.status()
        if action == "preflight":
            config = None
            if isinstance(payload, dict) and payload:
                config = payload.get("config", payload)
            return HTTPStatus.OK, self._preflight.run(config)
        if action == "start":
            auth = authorization
            if auth is None and isinstance(payload, dict):
                auth = payload.get("authorization")
            return self._pipeline.start(authorization=auth)
        if action == "stop":
            return self._pipeline.request_stop(
                force=False, source="operator API /api/stop"
            )
        if action == "force_stop":
            return self._pipeline.request_stop(
                force=True, source="operator API /api/force-stop"
            )
        if action == "cleanup":
            return self._pipeline.cleanup()
        if action == "shutdown":
            return (
                HTTPStatus.ACCEPTED,
                {"ok": True, "message": "Service shutdown requested"},
            )
        if action == "safety_estop":
            return self._pipeline.emergency_stop()
        if action == "safety_reset":
            return self._pipeline.safety_reset(payload)
        if action == "logs_download":
            path_obj = self._pipeline.downloadable_log()
            if path_obj is None:
                return (
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": "No session log is available"},
                )
            return HTTPStatus.OK, {"log_path": path_obj}
        raise AssertionError(f"unhandled route action: {action}")
