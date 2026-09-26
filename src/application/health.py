"""M10 HealthService: build the frozen ``/api/health`` payload."""

from __future__ import annotations

from typing import Any

from src.adapters.license import LicenseAdapter


class HealthService:
    """Mirrors the legacy health response fields (injected manager)."""

    def __init__(
        self,
        manager: Any,
        license_adapter: LicenseAdapter | None = None,
        *,
        service: str = "chingmu-gemini-native",
        product: str = "Chingmu Gemini Native v1.0 Preview",
        runtime: str = "windows_native",
        wsl_required: bool = False,
    ) -> None:
        self._manager = manager
        self._license = license_adapter or LicenseAdapter()
        self._service = service
        self._product = product
        self._runtime = runtime
        self._wsl_required = wsl_required

    def snapshot(self) -> dict[str, Any]:
        return {
            "service": self._service,
            "product": self._product,
            "ok": True,
            "runtime": self._runtime,
            "wsl_required": self._wsl_required,
            "phase": self._manager.status(log_limit=0)["phase"],
            "product_license": self._license.product_license_payload(),
        }
