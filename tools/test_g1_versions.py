#!/usr/bin/env python3
"""Offline regression checks for internal model assets and config migration."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.domain.g1_versions import (  # noqa: E402
    DEFAULT_MODEL_PROFILE,
    SUPPORTED_MODEL_PROFILES,
    model_profile_from_config,
    require_model_profile,
)
from src.infrastructure.config import (  # noqa: E402
    ConfigError,
    ConfigurationService,
    FROZEN_DEFAULTS,
    build_launch_environment,
    load_config,
    validate_config,
)


def main() -> int:
    assert DEFAULT_MODEL_PROFILE == "unitree_g1_5010"
    assert model_profile_from_config({}) == DEFAULT_MODEL_PROFILE
    service = ConfigurationService()
    for profile in SUPPORTED_MODEL_PROFILES:
        candidate = dict(FROZEN_DEFAULTS, model_profile=profile)
        assert validate_config(candidate)["model_profile"] == profile
        assert service.normalize_native(candidate)["model_profile"] == profile
        assert service.migration_preview(candidate)["model_profile"] == profile
        legacy = dict(candidate, g1_version=profile.removeprefix("unitree_g1_"))
        del legacy["model_profile"]
        assert model_profile_from_config(legacy) == profile
        for normalized in (
            validate_config(legacy),
            validate_config({"skeleton_id": 9}, legacy),
            service.migration_preview(legacy),
        ):
            assert normalized["model_profile"] == profile
            assert "g1_version" not in normalized
        environment = build_launch_environment(
            candidate, {"G1_VERSION": "4010", "UNCHANGED": "value"}
        )
        assert environment["G1_MODEL_PROFILE"] == profile
        assert "G1_VERSION" not in environment
        assert environment["UNCHANGED"] == "value"
        assets = PROJECT_ROOT / "storage" / "assets" / profile
        assert (assets / "g1_mjx_track.xml").is_file(), assets
        assert (assets / "scene_mjx_track.xml").is_file(), assets

    # An explicit deployment profile takes precedence during mixed-file migration.
    assert model_profile_from_config({
        "model_profile": "unitree_g1_4010", "g1_version": "5010"
    }) == "unitree_g1_4010"
    for invalid in ("unitree_g1_3010", "../unitree_g1_5010", "5", "15"):
        try:
            require_model_profile(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsupported profile was accepted: {invalid}")
    try:
        validate_config({"g1_version": "3010"})
    except ConfigError:
        pass
    else:
        raise AssertionError("unsupported legacy model was accepted")

    for filename in ("index.html", "app.js"):
        ui = (PROJECT_ROOT / "src" / "controller" / "static" / filename).read_text(
            encoding="utf-8"
        )
        for retired in ("g1Version", "g1_version", "4010", "5010", "model_profile"):
            assert retired not in ui, (filename, retired)

    # Only temporary settings are loaded or persisted. No robot, DDS channel,
    # launch process, or operator configuration is touched by these checks.
    from src.controller import native_app  # noqa: PLC0415
    from src.safety.hardware_gate import verify_unitree_lowstate_probe_report  # noqa: PLC0415

    assert native_app.base.verify_unitree_lowstate_probe_report is verify_unitree_lowstate_probe_report
    original_config_path = native_app.base.CONFIG_PATH
    original_log_dir = native_app.base.LOG_DIR
    original_defaults = dict(native_app.base.DEFAULT_CONFIG)
    try:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            native_app.base.CONFIG_PATH = temporary_path / "config.json"
            native_app.base.LOG_DIR = temporary_path / "logs"
            for version in ("4010", "5010"):
                legacy = dict(
                    FROZEN_DEFAULTS,
                    g1_version=version,
                    skeleton_id=17,
                    server_ip="192.168.50.99",
                    unitree_network_interface="192.168.123.200",
                    unitree_robot_ip="192.168.123.201",
                )
                del legacy["model_profile"]
                native_app.base.CONFIG_PATH.write_text(json.dumps(legacy), encoding="utf-8")
                with patch("src.infrastructure.config.persist_config", side_effect=PermissionError("read-only file")):
                    retained, warning = load_config(native_app.base.CONFIG_PATH, FROZEN_DEFAULTS)
                assert warning and "could not be persisted" in warning
                assert retained["model_profile"] == f"unitree_g1_{version}"
                for key in ("skeleton_id", "server_ip", "unitree_network_interface", "unitree_robot_ip"):
                    assert retained[key] == legacy[key], key
                loaded, warning = load_config(native_app.base.CONFIG_PATH, FROZEN_DEFAULTS)
                assert warning is None, warning
                assert loaded["model_profile"] == f"unitree_g1_{version}"
                assert "g1_version" not in json.loads(native_app.base.CONFIG_PATH.read_text(encoding="utf-8"))
                manager = native_app.GeminiNativePipelineManager()
                payload = manager.config_payload()
                assert "g1_versions" not in payload["options"]
                assert "model_profile" not in payload["config"]
                assert "g1_version" not in payload["config"]
                for key in ("g1_version", "model_profile"):
                    try:
                        manager.update_config({key: "4010"})
                    except ConfigError:
                        pass
                    else:
                        raise AssertionError(f"operator API accepted internal selector: {key}")
                manager.update_config({"human_height": 1.8})
                persisted = json.loads(native_app.base.CONFIG_PATH.read_text(encoding="utf-8"))
                assert persisted["model_profile"] == f"unitree_g1_{version}"
                assert persisted["human_height"] == 1.8
                assert "g1_version" not in persisted
                for key in ("skeleton_id", "server_ip", "unitree_network_interface", "unitree_robot_ip"):
                    assert persisted[key] == legacy[key], key
    finally:
        native_app.base.CONFIG_PATH = original_config_path
        native_app.base.LOG_DIR = original_log_dir
        native_app.base.DEFAULT_CONFIG.clear()
        native_app.base.DEFAULT_CONFIG.update(original_defaults)

    print("G1 INTERNAL MODEL / OFFICIAL SDK CONFIG CONTRACT PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
