#!/usr/bin/env python3
"""Offline acceptance regressions. Fake SDK bindings never open DDS."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.adapters import unitree_lowstate_probe as probe
from src.adapters import unitree_lowstate_probe_windows as windows
from src.adapters import unitree_develop_probe_windows as develop
from src.safety.hardware_gate import verify_unitree_lowstate_probe_report

NOW = datetime(2026, 9, 6, tzinfo=timezone.utc)
ADDRESS = "192.168.123.120"


def message(tick: int, mode: int = 5):
    return SimpleNamespace(
        tick=tick, mode_machine=mode, crc=123,
        imu_state=SimpleNamespace(quaternion=[1.0, 0.0, 0.0, 0.0]),
        motor_state=[SimpleNamespace(q=0.0, dq=0.0, tau_est=0.0) for _ in range(35)],
    )


class FakeRuntime:
    def __init__(self, mutate=None, *, publish=True, interrupt=False):
        self.time = 0.0
        self.tick = 0
        self.handler = None
        self.closed = False
        self.initialize_calls = []
        self.subscribe_calls = []
        self.mutate = mutate
        self.publish = publish
        self.interrupt = interrupt

    def initialize(self, domain, interface):
        self.initialize_calls.append((domain, interface))

    def subscribe(self, topic, state_type):
        self.subscribe_calls.append((topic, state_type))
        return self

    def Init(self, handler, queue_depth):
        self.handler = handler

    def Close(self):
        self.closed = True

    def sleep(self, duration):
        self.time += duration
        if self.interrupt:
            raise KeyboardInterrupt()
        if self.publish:
            for _ in range(5):
                self.tick += 1
                state = message(self.tick)
                if self.mutate:
                    self.mutate(state, self.tick)
                self.handler(state)

    def bindings(self):
        return probe.SdkBindings(self.initialize, self.subscribe, object, lambda state: 123)


class OfficialProbeTests(unittest.TestCase):
    def run_fake(self, mutate=None, *, g1_version=None, publish=True, interrupt=False):
        runtime = FakeRuntime(mutate, publish=publish, interrupt=interrupt)
        config = probe.ProbeConfig(
            ADDRESS, Path("unused-test-report.json"), g1_version=g1_version,
            duration_seconds=60.0, verify_interface=False,
        )
        report = probe.run_probe(
            config, bindings_loader=runtime.bindings, clock=lambda: runtime.time,
            sleeper=runtime.sleep, utc_now=lambda: NOW, write_report=False,
        )
        self.assertTrue(runtime.closed)
        self.assertEqual(runtime.initialize_calls, [(0, ADDRESS)])
        self.assertEqual(runtime.subscribe_calls, [("rt/lowstate", object)])
        return report

    def verify(self, report, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            return verify_unitree_lowstate_probe_report(
                path, interface=ADDRESS, now_utc=NOW, **kwargs
            )

    def test_received_mode_is_independent_of_legacy_motor_label(self):
        for mode, legacy in ((5, "5010"), (15, "4010"), (42, None), (0, None), (255, None)):
            with self.subTest(mode=mode, legacy=legacy):
                report = self.run_fake(lambda state, tick: setattr(state, "mode_machine", mode), g1_version=legacy)
                self.assertEqual(report["status"], "pass")
                self.assertEqual(report["summary"]["last_mode_machine"], mode)
                self.assertNotIn("g1_version", report["configuration"])
                self.assertNotIn("expected_mode_machine", report["configuration"])
                self.assertTrue(self.verify(report, g1_version=legacy)[0])

    def test_mode_change_fails(self):
        report = self.run_fake(lambda state, tick: setattr(state, "mode_machine", 15 if tick > 10 else 5))
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["summary"]["mode_machine_changes"], 1)
        self.assertFalse(self.verify(report)[0])

    def test_bad_crc_fails(self):
        report = self.run_fake(lambda state, tick: setattr(state, "crc", 0) if tick == 2 else None)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["summary"]["crc_invalid_samples"], 1)

    def test_structural_and_mode_errors_fail(self):
        mutations = (
            lambda state, tick: setattr(state, "mode_machine", 256),
            lambda state, tick: setattr(state, "motor_state", state.motor_state[:28]),
            lambda state, tick: setattr(state.imu_state, "quaternion", [float("nan"), 0, 0, 0]),
            lambda state, tick: setattr(state.motor_state[0], "q", float("inf")),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.assertEqual(self.run_fake(mutate)["status"], "fail")

    def test_empty_stream_and_interruption_fail(self):
        self.assertEqual(self.run_fake(publish=False)["status"], "fail")
        interrupted = self.run_fake(interrupt=True)
        self.assertEqual(interrupted["status"], "fail")
        self.assertTrue(interrupted["interrupted"])

    def test_tick_freeze_regression_and_wrap(self):
        self.assertEqual(self.run_fake(lambda state, tick: setattr(state, "tick", 100))["status"], "fail")
        self.assertEqual(self.run_fake(lambda state, tick: setattr(state, "tick", 100000 - tick))["status"], "fail")
        self.assertEqual(self.run_fake(lambda state, tick: setattr(state, "tick", ((1 << 32) - 10 + tick) % (1 << 32)))["status"], "pass")

    def test_rejects_empty_non_object_missing_and_duplicate_checks(self):
        report = self.run_fake()
        for checks in ([], [None], [True], ["pass"], [{"passed": True}], report["checks"][:-1], report["checks"] + report["checks"][:1]):
            with self.subTest(checks=checks):
                invalid = copy.deepcopy(report)
                invalid["checks"] = checks
                self.assertFalse(self.verify(invalid)[0])

    def test_rejects_old_custom_sdk_forged_summary_and_short_reports(self):
        report = self.run_fake()
        mutations = (
            lambda item: item.update(schema_version=1),
            lambda item: item.update(sdk={}),
            lambda item: item["sdk"].update(state_type="FirmwareLowState_"),
            lambda item: item["summary"].update(mode_machine_changes=1),
            lambda item: item["summary"].update(crc_valid_samples=0),
            lambda item: item["summary"].update(receive_rate_hz=float("nan")),
            lambda item: item["summary"].update(elapsed_seconds=0.1),
            lambda item: item["summary"].update(observed_mode_machine_counts={}),
            lambda item: item["summary"].update(valid_samples=True),
            lambda item: item["configuration"].update(minimum_receive_rate_hz=1),
            lambda item: item["configuration"].update(duration_seconds=2),
            lambda item: item["configuration"].update(duration_seconds=59),
            lambda item: item["configuration"].update(network_interface="192.168.123.99"),
            lambda item: item["configuration"].update(domain_id=1),
            lambda item: item.update(finished_at_utc=(NOW - timedelta(days=2)).isoformat()),
            lambda item: item.update(finished_at_utc=(NOW + timedelta(hours=1)).isoformat()),
        )
        for mutate in mutations:
            invalid = copy.deepcopy(report)
            mutate(invalid)
            with self.subTest(mutate=mutate):
                self.assertFalse(self.verify(invalid)[0])

    def test_official_loader_uses_channel_hg_and_crc_only(self):
        channel = SimpleNamespace(ChannelFactoryInitialize=object(), ChannelSubscriber=object())
        hg = SimpleNamespace(LowState_=object())
        crc = SimpleNamespace(Crc=lambda state: 123)
        modules = {
            "unitree_sdk2py.core.channel": channel,
            "unitree_sdk2py.idl.unitree_hg.msg.dds_": hg,
            "unitree_sdk2py.utils.crc": SimpleNamespace(CRC=lambda: crc),
        }
        with patch.object(probe.importlib, "import_module", side_effect=modules.__getitem__) as importer:
            bindings = probe.load_sdk_bindings()
        self.assertEqual({call.args[0] for call in importer.call_args_list}, set(modules))
        self.assertIs(bindings.subscriber_type, channel.ChannelSubscriber)
        self.assertIs(bindings.state_type, hg.LowState_)
        self.assertIs(bindings.calculate_crc, crc.Crc)

    def test_windows_uses_same_official_subscriber_after_interface_configuration(self):
        runtime = FakeRuntime()
        native = runtime.bindings()
        events = []
        trace = Path("test-trace.log")
        with patch.object(probe, "load_sdk_bindings", return_value=native), patch(
            "src.adapters.unitree.configure_windows_unitree_interface",
            side_effect=lambda address, **kwargs: events.append((address, kwargs)) or address,
        ):
            bindings = windows.load_windows_bindings(trace)
            self.assertEqual(runtime.initialize_calls, [])
            bindings.initialize(0, ADDRESS)
        self.assertEqual(events, [(ADDRESS, {"trace_path": trace})])
        self.assertIs(bindings.subscriber_type, native.subscriber_type)
        self.assertIs(bindings.state_type, native.state_type)
        self.assertIs(bindings.calculate_crc, native.calculate_crc)
        self.assertEqual(runtime.initialize_calls, [(0, ADDRESS)])

    def test_cli_requires_no_motor_selection(self):
        self.assertIsNone(probe.build_argument_parser().parse_args(["--interface", "eth0"]).g1_version)
        self.assertIsNone(windows.build_argument_parser().parse_args(["--interface-address", ADDRESS]).g1_version)
        self.assertIsNone(develop.build_argument_parser().parse_args(["--interface-address", ADDRESS]).g1_version)
        self.assertNotIn("--g1-version", windows.build_argument_parser().format_help())


if __name__ == "__main__":
    unittest.main()
