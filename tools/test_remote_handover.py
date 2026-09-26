#!/usr/bin/env python3
"""Regression checks for the Develop-first G1 handover gates."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.motion.remote_handover import (  # noqa: E402
    ReleasedChordGate,
    RemoteHandoverError,
    checked_motion_owner,
)
from src.adapters.unitree_lowstate_probe_windows import (  # noqa: E402
    persist_probe_result,
)


class FakeMotionSwitcher:
    def __init__(self, status: int = 0, name: str = "") -> None:
        self.status = status
        self.name = name

    def CheckMode(self):
        return self.status, {"form": "0", "name": self.name}


def buttons(*pressed: int) -> tuple[int, ...]:
    state = [0] * 16
    for button_id in pressed:
        state[button_id] = 1
    return tuple(state)


def test_optional_owner_decoder() -> None:
    develop = FakeMotionSwitcher(name="")
    assert checked_motion_owner(develop, context="test") == ""
    assert checked_motion_owner(FakeMotionSwitcher(name=" ai "), context="test") == "ai"

    try:
        checked_motion_owner(FakeMotionSwitcher(status=1, name=""), context="test")
    except RemoteHandoverError as exc:
        assert "CheckMode failed" in str(exc)
    else:
        raise AssertionError("failed optional CheckMode response passed")


def test_released_then_pressed_start_gate() -> None:
    start = 2
    gate = ReleasedChordGate((start,))

    # A key held before the application is ready cannot authorize takeover.
    assert not gate.update(buttons(start))
    assert not gate.update(buttons(start))

    # Another pressed key does not count as the mandatory all-buttons release.
    assert not gate.update(buttons(7))
    assert not gate.released

    assert not gate.update(buttons())
    assert gate.released

    # START must be exclusive and present in two consecutive fresh frames.
    assert not gate.update(buttons(start))
    assert not gate.update(buttons(start, 7))
    assert not gate.update(buttons(start))
    assert gate.update(buttons(start))


def test_failed_probe_preserves_last_acceptance() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        acceptance = Path(temporary) / "unitree_lowstate_windows_report.json"
        acceptance.write_text(
            json.dumps({"status": "pass", "marker": "previous"}),
            encoding="utf-8",
        )

        attempt, accepted = persist_probe_result(
            acceptance, {"status": "fail", "marker": "failed-attempt"}
        )
        assert not accepted
        assert json.loads(acceptance.read_text(encoding="utf-8"))["marker"] == "previous"
        assert json.loads(attempt.read_text(encoding="utf-8"))["marker"] == "failed-attempt"

        attempt, accepted = persist_probe_result(
            acceptance, {"status": "pass", "marker": "new-pass"}
        )
        assert accepted
        assert json.loads(acceptance.read_text(encoding="utf-8"))["marker"] == "new-pass"
        assert json.loads(attempt.read_text(encoding="utf-8"))["marker"] == "new-pass"


def test_lowstate_precedes_optional_owner_check() -> None:
    source = (ROOT / "src" / "motion" / "play_track.py").read_text(
        encoding="utf-8"
    )
    channel_init = source.index("ChannelFactoryInitialize(0, args.net)")
    policy_load = source.index("track_policy = get_policy_onnx(", channel_init)
    lowstate_wait = source.index("low_ctrl = LowLevelControlG1(", policy_load)
    optional_owner = source.index(
        "candidate_switcher = create_motion_switcher_client()", lowstate_wait
    )
    assert channel_init < policy_load < lowstate_wait < optional_owner
    assert "lowstate_timeout_seconds: float | None = 10.0" in source
    assert "require_develop_owner" not in source
    assert "require_official_owner" not in source
    assert "remote_damping_chord_timeout_seconds" not in source
    assert "remote_debug_chord_timeout_seconds" not in source

    web_prompt = (ROOT / "src" / "controller" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    assert "L2+R2 进入 Develop" in web_prompt
    assert "输入 ARM G1" in web_prompt

    supervisor = (
        ROOT / "src" / "orchestration" / "real_supervisor.py"
    ).read_text(encoding="utf-8")
    assert "pre-entered Develop and physical START" in supervisor

    remote_readme = (ROOT / "README_G1_REMOTE.md").read_text(encoding="utf-8")
    assert "先不要启动网页真机任务" in remote_readme
    assert "机器人 Develop 模式" in remote_readme
    assert "网页“真机调试模式”" in remote_readme

    quick_readme = (ROOT / "README_REAL_ROBOT.md").read_text(encoding="utf-8")
    enter_develop = quick_readme.index("先不要启动脚本或网页真机任务")
    start_probe = quick_readme.index("才双击", enter_develop)
    assert enter_develop < start_probe

    native_app = (ROOT / "src" / "controller" / "native_app.py").read_text(
        encoding="utf-8"
    )
    live_preflight = native_app.index('add("unitree_develop_mode"')
    authorization = native_app.index(
        "authorization != base.REAL_OUTPUT_AUTHORIZATION", live_preflight
    )
    assert live_preflight < authorization

    probe_source = (
        ROOT / "src" / "adapters" / "unitree_develop_probe_windows.py"
    ).read_text(encoding="utf-8")
    assert "rt/lowstate" in probe_source
    assert "write_report=False" in probe_source
    assert "MotionSwitcherClient" not in probe_source
    assert "unitree_sdk2py.idl" not in probe_source
    assert "ChannelPublisher(" not in probe_source
    assert ".SelectMode(" not in probe_source
    assert ".ReleaseMode(" not in probe_source

    real_robot = (ROOT / "src" / "motion" / "real_robot.py").read_text(
        encoding="utf-8"
    )
    publish = real_robot.index("def fast_publish(self) -> None:")
    fresh = real_robot.index("self._validated_sensor_snapshot()", publish)
    write = real_robot.index("self._write_lowcmd_confirmed()", fresh)
    assert publish < fresh < write

    real_a_gate = source.index("if KeyMap.A in events:")
    fresh_reference_read = source.index("mocap_buffer.read()", real_a_gate)
    live_handover = source.index(
        "remote_handover.command(RemoteHandoverCommand.LIVE)", real_a_gate
    )
    assert real_a_gate < fresh_reference_read < live_handover
    assert "A ignored: no fresh CHINGMU reference" in source


def main() -> int:
    test_optional_owner_decoder()
    test_released_then_pressed_start_gate()
    test_failed_probe_preserves_last_acceptance()
    test_lowstate_precedes_optional_owner_check()
    print("G1 DEVELOP-FIRST REMOTE HANDOVER TEST PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
