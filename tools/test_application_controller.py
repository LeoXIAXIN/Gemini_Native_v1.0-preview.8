#!/usr/bin/env python3
"""Application and controller facade tests (read-only; gmr runtime).

Uses a FakeManager so the real controller never reads or persists operator
configuration during this unit test.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.application import (  # noqa: E402
    HealthService,
    LogService,
    PipelineService,
    PreflightService,
    SessionRecord,
    SessionService,
    preflight_result_from_payload,
)
from src.controller.http_controller import (  # noqa: E402
    ROUTES,
    HttpControllerFacade,
    origin_is_local,
)
from src.domain.enums import TaskPhase  # noqa: E402
from src.domain.models import PreflightResult  # noqa: E402
from src.orchestration.process_manager import ProcessManagerFacade  # noqa: E402
from src.orchestration.process_spec import (  # noqa: E402
    PREVIEW_PIPELINE_STOP_ORDER,
    REAL_PIPELINE_STOP_ORDER,
    SIM_PIPELINE_STOP_ORDER,
    default_pipeline_paths,
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  [ok]   {name}")
    else:
        FAILURES.append(f"{name}: {detail or 'assertion failed'}")
        print(f"  [FAIL] {name}: {detail or 'assertion failed'}")


class FakeLicense:
    def product_license_payload(self):
        return {"allowed": True, "code": "ok", "message": "fake"}

    def check_product_license(self):
        class _Status:
            def to_public_dict(self):
                return {"allowed": True, "code": "ok", "message": "fake"}
        return _Status()


class FakeManager:
    """Duck-typed manager that records every delegation."""

    def __init__(self):
        self.calls: list[tuple] = []
        self._phase = "stopped"
        self._config = {"backend": "humanoid_gpt",
                        "execution_mode": "simulation_only",
                        "debug_mode": True, "device": "cpu",
                        "model_profile": "unitree_g1_5010",
                        "server_ip": "127.0.0.1", "skeleton_id": 0,
                        "human_height": 1.75, "publish_fps": 50.0,
                        "transition_seconds": 1.5,
                        "safety_strategy_enabled": False,
                        "unitree_network_interface": "192.168.123.100",
                        "unitree_robot_ip": "192.168.123.164"}
        self._log_path = Path("session_fake.log")

    def status(self, log_limit=200):
        self.calls.append(("status", log_limit))
        return {"phase": self._phase, "logs": [{"id": 1, "message": "x"}]}

    def preflight(self, config=None):
        self.calls.append(("preflight", config))
        return {"ok": True,
                "checks": [{"name": "a", "ok": True, "message": "m",
                            "required": True}],
                "errors": []}

    def start(self, authorization=None):
        self.calls.append(("start", authorization))
        return 202, {"ok": True}

    def request_stop(self, force=False, source="internal request"):
        self.calls.append(("stop", force, source))
        return 202, {"ok": True}

    def cleanup(self):
        self.calls.append(("cleanup",))
        return 200, {"ok": True}

    def shutdown(self):
        self.calls.append(("shutdown",))

    def request_emergency_stop(self):
        self.calls.append(("estop",))
        return 202, {"ok": True, "latched": True}

    def request_safety_reset(self, payload):
        self.calls.append(("reset", payload))
        return 202, {"ok": True}

    def config_payload(self):
        self.calls.append(("config_payload",))
        return {"ok": True, "config": dict(self._config)}

    def update_config(self, update):
        self.calls.append(("update_config", update))
        return {"ok": True}

    def downloadable_log(self):
        self.calls.append(("downloadable_log",))
        return self._log_path


def main() -> int:
    print("M9/M10 application + controller facade tests")

    # ================= SessionService =================
    sessions = SessionService()
    first = sessions.create(backend="humanoid_gpt",
                            execution_mode="simulation_only")
    second = sessions.create(backend="humanoid_gpt",
                             execution_mode="real_only")
    check("session: unique ids", first.session_id != second.session_id)
    check("session: current is latest",
          sessions.current() is not None
          and sessions.current().session_id == second.session_id)
    updated = sessions.update_status(second.session_id, TaskPhase.RUNNING)
    check("session: status transition",
          updated.status == TaskPhase.RUNNING
          and sessions.get(second.session_id).status == TaskPhase.RUNNING)
    check("session: list length 2", len(sessions.list()) == 2)
    sessions.clear()
    check("session: clear", sessions.current() is None)

    # ================= PipelineService (delegation + dispatch mirror) =================
    manager = FakeManager()
    pipeline = PipelineService(manager)
    check("pipeline: status delegates",
          pipeline.status(log_limit=5)["phase"] == "stopped"
          and manager.calls[-1] == ("status", 5))
    pipeline.start(authorization="ARM G1")
    check("pipeline: start passes authorization",
          manager.calls[-1] == ("start", "ARM G1"))
    pipeline.request_stop(force=True, source="operator API /api/force-stop")
    check("pipeline: stop delegation",
          manager.calls[-1] == ("stop", True, "operator API /api/force-stop"))

    sim_spec = pipeline.build_supervisor_command(manager._config)
    real_spec = pipeline.build_supervisor_command(
        dict(manager._config, execution_mode="real_only"))
    preview_spec = pipeline.build_supervisor_command(
        dict(manager._config, backend="gmr_preview"))
    check("pipeline: mode dispatch mirror",
          sim_spec.name == "Humanoid-GPT simulation supervisor"
          and real_spec.name == "Unitree real supervisor"
          and preview_spec.name == "GMR preview supervisor")

    # ================= PreflightService =================
    preflight = PreflightService(manager)
    result = preflight.run()
    check("preflight: maps to domain model",
          isinstance(result, PreflightResult) and result.ok is True
          and len(result.checks) == 1 and result.failed_required == ())
    partial = preflight_result_from_payload({
        "ok": False,
        "checks": [
            {"name": "must", "ok": False, "message": "broken", "required": True},
            {"name": "optional", "ok": False, "message": "warn",
             "required": False},
        ],
        "errors": ["broken"],
    })
    check("preflight: failed_required filtering",
          partial.ok is False and len(partial.failed_required) == 1
          and partial.failed_required[0].name == "must")

    # ================= HealthService =================
    health = HealthService(manager, license_adapter=FakeLicense())
    snapshot = health.snapshot()
    check("health: frozen payload fields",
          set(snapshot) == {"service", "product", "ok", "runtime",
                            "wsl_required", "phase", "product_license"}
          and snapshot["service"] == "chingmu-gemini-native"
          and snapshot["runtime"] == "windows_native"
          and snapshot["phase"] == "stopped")

    # ================= LogService =================
    logs = LogService(manager)
    check("logs: recent delegation", logs.recent(1) == [{"id": 1, "message": "x"}])
    check("logs: downloadable delegation",
          logs.downloadable_log() == manager._log_path)
    check("logs: classification",
          logs.classify_component("Sender FPS: 50.0, frame: 1") == "chingmu"
          and logs.classify_level("some error") == "error")
    check("logs: healthy bridge stats stay informational",
          logs.classify_level(
              "Bridge 42.0 Hz | local RX 64.0 Hz | IK 64.0 Hz | rejected 0"
          ) == "info")
    check("logs: startup waits stay informational",
          logs.classify_level("waiting_for_local_frame") == "info")
    check("logs: actual reference rejection is warning",
          logs.classify_level(
              "Rejected live reference packet: packet is older than 0.75 s"
          ) == "warning"
          and logs.classify_level("Bridge 1.0 Hz | rejected 3") == "warning")

    # ================= ProcessManagerFacade =================
    facade = ProcessManagerFacade(manager)
    check("pm: stop order for sim spec",
          facade.stop_order_for(sim_spec) == SIM_PIPELINE_STOP_ORDER)
    check("pm: stop order for real spec",
          facade.stop_order_for(real_spec) == REAL_PIPELINE_STOP_ORDER)
    check("pm: stop order for preview spec",
          facade.stop_order_for(preview_spec) == PREVIEW_PIPELINE_STOP_ORDER)
    check("pm: launch spec equals pipeline builder",
          facade.launch_spec_for(manager._config) == sim_spec)

    # ================= HttpControllerFacade =================
    controller = HttpControllerFacade(manager, license_adapter=FakeLicense())
    unique_paths = {route.path for route in ROUTES}
    frozen_paths = {
        "/api/health", "/api/license", "/api/config", "/api/status",
        "/api/preflight", "/api/start", "/api/stop", "/api/force-stop",
        "/api/cleanup", "/api/shutdown", "/api/safety/estop",
        "/api/safety/reset", "/api/logs/download",
    }
    check("controller: frozen route paths",
          unique_paths == frozen_paths and len(ROUTES) == 15)
    check("controller: config has GET/PUT/POST",
          {(r.method, r.path) for r in ROUTES if r.path == "/api/config"}
          == {("GET", "/api/config"), ("PUT", "/api/config"),
              ("POST", "/api/config")})

    status, payload = controller.dispatch("GET", "/api/health")
    check("controller: health dispatch", status == 200
          and payload["service"] == "chingmu-gemini-native")
    status, payload = controller.dispatch("POST", "/api/start",
                                          {"authorization": "ARM G1"})
    check("controller: start dispatch", status == 202
          and manager.calls[-1] == ("start", "ARM G1"))
    status, payload = controller.dispatch("POST", "/api/safety/reset",
                                          {"operator_confirmed": True})
    check("controller: safety reset dispatch", status == 202
          and manager.calls[-1] == ("reset", {"operator_confirmed": True}))
    status, payload = controller.dispatch("POST", "/api/nope")
    check("controller: unknown endpoint 404", status == 404
          and payload == {"ok": False, "error": "Unknown API endpoint"})
    status, payload = controller.dispatch("POST", "/api/shutdown")
    check("controller: shutdown dispatch", status == 202
          and payload["ok"] is True)

    check("controller: origin checks",
          origin_is_local(None, 8765) is True
          and origin_is_local("http://127.0.0.1:8765", 8765) is True
          and origin_is_local("http://localhost:8765", 8765) is True
          and origin_is_local("http://evil.example:8765", 8765) is False
          and origin_is_local("https://127.0.0.1:8765", 8765) is False
          and origin_is_local("http://127.0.0.1:9999", 8765) is False)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("ALL M9/M10 TESTS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
