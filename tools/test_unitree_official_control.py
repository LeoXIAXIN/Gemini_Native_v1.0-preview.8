"""Real controller regressions using official IDL/CRC and fake DDS endpoints.

No ChannelFactoryInitialize, DomainParticipant, or real publisher is created.
"""

from contextlib import ExitStack
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unitree_sdk2py.core import channel
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from src.motion import real_robot as robot


def frame(mode=5, tick=1):
    state = unitree_hg_msg_dds__LowState_()
    state.mode_machine = mode
    state.tick = tick
    state.imu_state.quaternion = [1., 0., 0., 0.]
    state.crc = CRC().Crc(state)
    return state


class OfficialControlTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.frames = [frame()]
        self.writers = []
        self.handler = None
        self.on_writer_init = None
        self.write_ok = True
        owner = self

        class Subscriber:
            def __init__(self, topic, message_type):
                owner.assertEqual(topic, "rt/lowstate")
                owner.assertIs(message_type, LowState_)

            def Init(self, callback, depth):
                owner.handler = callback
                for item in owner.frames:
                    callback(item)

        class Publisher:
            def __init__(self, topic, message_type):
                owner.assertEqual(topic, "rt/lowcmd")
                owner.assertIs(message_type, LowCmd_)
                self.sent = []
                self.closed = False
                owner.writers.append(self)

            def Init(self):
                if owner.on_writer_init:
                    owner.on_writer_init()

            def Write(self, command):
                owner.assertIs(type(command), LowCmd_)
                owner.assertEqual(command.crc, CRC().Crc(command))
                self.sent.append(copy.deepcopy(command))
                return owner.write_ok

            def Close(self):
                self.closed = True

        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(channel, "ChannelSubscriber", Subscriber))
        self.stack.enter_context(patch.object(channel, "ChannelPublisher", Publisher))
        self.stack.enter_context(patch.object(channel, "ChannelFactoryInitialize", side_effect=AssertionError("DDS forbidden")))
        self.stack.enter_context(patch.object(robot.time, "monotonic", side_effect=lambda: self.now))

    def control(self, **kwargs):
        return robot.LowLevelControlG1(**kwargs)

    def test_mode_five_with_legacy_5010_label_uses_official_command(self):
        ctrl = self.control(g1_version="5010", defer_publisher=True)
        self.assertEqual(ctrl.expected_mode_machine, 5)
        self.assertEqual(self.writers, [])
        ctrl.enable_lowcmd_publisher()
        ctrl.step(np.zeros(29))
        self.assertEqual(self.writers[0].sent[-1].mode_machine, 5)
        self.assertEqual(ctrl.lowcmd_write_count, 1)
        ctrl.low_cmd.mode_machine = 15  # Cannot override feedback with App value.
        ctrl.fast_publish()
        self.assertEqual(self.writers[0].sent[-1].mode_machine, 5)
        ctrl.disable_lowcmd_publisher()
        self.assertTrue(self.writers[0].closed)
        with self.assertRaises(RuntimeError):
            ctrl.step(np.zeros(29))

    def test_other_valid_modes_do_not_select_motor_protocols(self):
        for mode in (0, 13, 14, 15, 16, 18, 255):
            with self.subTest(mode=mode):
                self.frames = [frame(mode)]
                ctrl = self.control(debug=True)
                self.assertEqual(ctrl.expected_mode_machine, mode)
                self.assertEqual(ctrl.low_cmd.mode_machine, mode)
        self.assertEqual(self.writers, [])

    def test_debug_never_creates_writer(self):
        ctrl = self.control(debug=True)
        with self.assertRaisesRegex(RuntimeError, "read-only debug"):
            ctrl.enable_lowcmd_publisher()
        ctrl.step(np.zeros(29))
        ctrl.set_motor_damping()
        self.assertFalse(ctrl.has_publisher)
        self.assertEqual(self.writers, [])

    def test_corrupt_first_frame_cannot_select_session_mode(self):
        damaged = frame(15)
        damaged.crc ^= 1
        self.frames = [damaged, frame(5, 2)]
        ctrl = self.control(debug=True)
        self.assertEqual(ctrl.expected_mode_machine, 5)

    def test_corruption_after_start_blocks_commands(self):
        ctrl = self.control()
        damaged = frame(5, 2)
        damaged.crc ^= 1
        self.handler(damaged)
        with self.assertRaisesRegex(RuntimeError, "CRC"):
            ctrl.step(np.zeros(29))
        self.assertEqual(self.writers[0].sent, [])

    def test_mode_change_is_latched_even_after_switching_back(self):
        ctrl = self.control()
        self.handler(frame(15, 2))
        self.handler(frame(5, 3))
        with self.assertRaisesRegex(RuntimeError, "mode_machine changed"):
            ctrl.step(np.zeros(29))
        self.assertEqual(self.writers[0].sent, [])

    def test_mode_change_during_init_closes_unwritten_writer(self):
        self.on_writer_init = lambda: self.handler(frame(15, 2))
        with self.assertRaisesRegex(RuntimeError, "mode_machine changed"):
            self.control()
        self.assertTrue(self.writers[0].closed)
        self.assertEqual(self.writers[0].sent, [])

    def test_stale_or_regressed_feedback_cannot_publish(self):
        ctrl = self.control()
        self.now += .101
        with self.assertRaises(TimeoutError):
            ctrl.fast_publish()
        self.now = 100.
        self.handler(frame(5, 0))
        with self.assertRaisesRegex(RuntimeError, "tick regressed"):
            ctrl.fast_publish()
        self.assertEqual(self.writers[0].sent, [])

    def test_nan_targets_and_failed_write_are_not_success(self):
        ctrl = self.control()
        with self.assertRaises(ValueError):
            ctrl.step(np.full(29, np.nan))
        self.assertEqual(self.writers[0].sent, [])
        self.write_ok = False
        with self.assertRaisesRegex(RuntimeError, "Write"):
            ctrl.step(np.zeros(29))
        self.assertEqual(ctrl.lowcmd_write_count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
