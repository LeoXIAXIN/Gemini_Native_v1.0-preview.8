"""Lazy, fail-closed facade over ``src.application.gemini_license``."""

from __future__ import annotations

import importlib
import threading
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LICENSE_DIR = _PROJECT_ROOT / "resources" / "license"

_status_lock = threading.Lock()
_manager: Any = None
_last_license_status: Any = None


def _module() -> Any:
    return importlib.import_module("src.application.gemini_license")


def _ensure() -> None:
    global _manager, _last_license_status
    if _manager is not None:
        return
    module = _module()
    _manager = module.LicenseManager(
        public_key_path=_LICENSE_DIR / "public_key.json",
        license_path=_LICENSE_DIR / "ChingmuGemini.license",
    )
    _last_license_status = module.LicenseStatus(
        allowed=False,
        code="not_checked",
        message="尚未检查产品许可证。",
    )


def check_product_license() -> Any:
    """Authorize one new session without affecting a live task."""
    global _last_license_status
    _ensure()
    status = _manager.authorize_new_session()
    with _status_lock:
        _last_license_status = status
    return status


def product_license_payload() -> dict[str, Any]:
    """Operator-safe cached license status payload."""
    _ensure()
    with _status_lock:
        return _last_license_status.to_public_dict()


class LicenseAdapter:
    """Delegating wrapper around the migrated license functions."""

    def check_product_license(self) -> Any:
        return check_product_license()

    def product_license_payload(self) -> dict[str, Any]:
        return product_license_payload()
